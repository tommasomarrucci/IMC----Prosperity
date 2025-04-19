from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math, jsonpickle

class Trader:
    # --------‑‑ static parameters ‑‑---------
    PRODUCT          = "MAGNIFICENT_MACARONS"
    POS_LIMIT        = 75
    CONV_LIMIT       = 10                 # abs(conversions) per ts
    EDGE_TAKE        = 1                  # cross when edge ≥ 1
    EDGE_JOIN        = 2                  # distance of passive quotes
    SOFT_POS_LIM     = 30                 # lean quotes when |pos|>30
    EW_DECAY         = 0.05               # offset learning rate
    BASE_OFFSET      = 6.5                # initial storage/liquidity premium

    def __init__(self) -> None:
        self.offset = self.BASE_OFFSET    # adaptive premium

    # -------------- main entry --------------
    def run(
        self, state: TradingState
    ) -> tuple[Dict[str, List[Order]], int, str]:

        orders_by_prod: Dict[str, List[Order]] = {}
        conversions_net                       = 0  # signed
        traderData                            = ""

        depth: OrderDepth | None = state.order_depths.get(self.PRODUCT)
        obs_dict                  = getattr(state, "observations", None)
        obs                       = obs_dict.get(self.PRODUCT) if isinstance(obs_dict, dict) else None

        # graceful degradation on the very first tick
        if depth is None or obs is None:
            return {}, 0, traderData

        # 1) Conversion venue costs (integers already)
        cost_buy  = obs.askPrice + obs.transportFees + obs.importTariff
        cost_sell = obs.bidPrice - obs.transportFees - obs.exportTariff

        # 2) Adaptive fair value
        if depth.buy_orders and depth.sell_orders:
            book_mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
            anchor   = (cost_buy + cost_sell) / 2
            self.offset = (1 - self.EW_DECAY) * self.offset + self.EW_DECAY * (book_mid - anchor)
        fair = (cost_buy + cost_sell) / 2 + self.offset

        pos       = state.position.get(self.PRODUCT, 0)
        open_buy  = 0   # qty we buy in this tick (market OR quote)
        open_sell = 0   # qty we sell               "

        book_orders: List[Order] = []

        # ------------------------------------------------------------
        # 3) TAKE the book when mis‑priced
        # ------------------------------------------------------------
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            ask_qty  = -depth.sell_orders[best_ask]
            if best_ask <= fair - self.EDGE_TAKE:
                qty = min(ask_qty, self.POS_LIMIT - pos)
                if qty > 0:
                    book_orders.append(Order(self.PRODUCT, int(best_ask),  qty))
                    open_buy += qty

        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            bid_qty  =  depth.buy_orders[best_bid]
            if best_bid >= fair + self.EDGE_TAKE:
                qty = min(bid_qty, self.POS_LIMIT + pos)
                if qty > 0:
                    book_orders.append(Order(self.PRODUCT, int(best_bid), -qty))
                    open_sell += qty

        pos_after_take = pos + open_buy - open_sell

        # ------------------------------------------------------------
        # 4) CONVERSION arbitrage  (respect abs ≤ 10)
        # ------------------------------------------------------------
        remaining_conv_cap = self.CONV_LIMIT

        # a) buy from chefs, re‑sell on book
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            if best_ask >= cost_buy + self.EDGE_JOIN:
                qty = min(remaining_conv_cap, self.POS_LIMIT - pos_after_take)
                if qty > 0:
                    conversions_net   += qty           # + = buy from chefs
                    remaining_conv_cap -= qty
                    # hedge immediately
                    hedge_px = best_ask - 1
                    book_orders.append(Order(self.PRODUCT, int(hedge_px), -qty))
                    open_sell      += qty
                    pos_after_take -= qty

        # b) sell to chefs, re‑buy on book
        if depth.buy_orders and remaining_conv_cap > 0:
            best_bid = max(depth.buy_orders)
            if best_bid <= cost_sell - self.EDGE_JOIN:
                qty = min(remaining_conv_cap, self.POS_LIMIT + pos_after_take)
                if qty > 0:
                    conversions_net   -= qty           # – = sell to chefs
                    remaining_conv_cap -= qty
                    hedge_px = best_bid + 1
                    book_orders.append(Order(self.PRODUCT, int(hedge_px), qty))
                    open_buy       += qty
                    pos_after_take += qty

        # ------------------------------------------------------------
        # 5) CLEAR – dump inv that is deep ITM
        # ------------------------------------------------------------
        if pos_after_take > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders)
            if best_bid > fair + self.EDGE_TAKE:
                qty = min(pos_after_take, depth.buy_orders[best_bid])
                if qty > 0:
                    book_orders.append(Order(self.PRODUCT, int(best_bid), -qty))
                    open_sell      += qty
                    pos_after_take -= qty

        if pos_after_take < 0 and depth.sell_orders:
            best_ask = min(depth.sell_orders)
            if best_ask < fair - self.EDGE_TAKE:
                qty = min(-pos_after_take, -depth.sell_orders[best_ask])
                if qty > 0:
                    book_orders.append(Order(self.PRODUCT, int(best_ask), qty))
                    open_buy       += qty
                    pos_after_take += qty

        # ------------------------------------------------------------
        # 6) MAKE – passive quotes around fair
        # ------------------------------------------------------------
        bias = max(-self.SOFT_POS_LIM,
                   min(self.SOFT_POS_LIM, pos_after_take)) / self.SOFT_POS_LIM
        bid_px = int(round(fair - self.EDGE_JOIN - bias))
        ask_px = int(round(fair + self.EDGE_JOIN - bias))

        if bid_px < ask_px:     # safety
            buy_cap  = self.POS_LIMIT - pos_after_take
            sell_cap = self.POS_LIMIT + pos_after_take
            if buy_cap  > 0:
                book_orders.append(Order(self.PRODUCT, bid_px,  buy_cap))
            if sell_cap > 0:
                book_orders.append(Order(self.PRODUCT, ask_px, -sell_cap))

        orders_by_prod[self.PRODUCT] = book_orders
        return orders_by_prod, conversions_net, traderData