from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import jsonpickle, math

class Trader:
    PRODUCT      = "MAGNIFICENT_MACARONS"
    POS_LIMIT    = 75
    CONV_LIMIT   = 10
    EDGE_TAKE    = 1        # *** positive edge for crossing  ***
    EDGE_JOIN    = 1        # tighter passive quotes as you set
    SOFT_POS_LIM = 20
    EW_DECAY     = 0.05
    BASE_OFFSET  = 3

    def __init__(self) -> None:
        self.offset = self.BASE_OFFSET

    def run(self, state: TradingState):
        orders_by_prod: Dict[str, List[Order]] = {}
        conversions_net = 1
        trader_data     = ""

        depth: OrderDepth | None = state.order_depths.get(self.PRODUCT)

        # ---------- robust observation fetch ----------
        obs_src = getattr(state, "observations", None)
        if obs_src is None:
            return {}, 0, trader_data
        obs = obs_src[self.PRODUCT] if isinstance(obs_src, dict) else obs_src

        if depth is None or obs is None:
            return {}, 0, trader_data

        # ---------- conversion costs ----------
        cost_buy  = obs.askPrice + obs.transportFees + obs.importTariff
        cost_sell = obs.bidPrice - obs.transportFees - obs.exportTariff

        # ---------- adaptive fair ----------
        if depth.buy_orders and depth.sell_orders:
            book_mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
            anchor   = (cost_buy + cost_sell) / 2
            self.offset = (1 - self.EW_DECAY) * self.offset + self.EW_DECAY * (book_mid - anchor)
        fair = (cost_buy + cost_sell) / 2 + self.offset

        pos, buy_open, sell_open = state.position.get(self.PRODUCT, 0), 0, 0
        orders: List[Order] = []

        # ---------- 3) TAKE ----------
        if depth.sell_orders:
            best_ask, ask_qty = min(depth.sell_orders), -depth.sell_orders[min(depth.sell_orders)]
            if best_ask <= fair - self.EDGE_TAKE:
                qty = min(ask_qty, self.POS_LIMIT - pos)
                if qty:
                    orders.append(Order(self.PRODUCT, int(best_ask), qty))
                    buy_open += qty
        if depth.buy_orders:
            best_bid, bid_qty = max(depth.buy_orders), depth.buy_orders[max(depth.buy_orders)]
            if best_bid >= fair + self.EDGE_TAKE:
                qty = min(bid_qty, self.POS_LIMIT + pos)
                if qty:
                    orders.append(Order(self.PRODUCT, int(best_bid), -qty))
                    sell_open += qty

        pos_after = pos + buy_open - sell_open

        # ---------- 4) CONVERSION ----------
        conv_cap = self.CONV_LIMIT
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            if best_ask >= cost_buy + self.EDGE_JOIN:
                qty = min(conv_cap, self.POS_LIMIT - pos_after)
                if qty:
                    conversions_net += qty
                    conv_cap        -= qty
                    orders.append(Order(self.PRODUCT, int(best_ask - 1), -qty))
                    sell_open += qty
                    pos_after -= qty
        if depth.buy_orders and conv_cap:
            best_bid = max(depth.buy_orders)
            if best_bid <= cost_sell - self.EDGE_JOIN:
                qty = min(conv_cap, self.POS_LIMIT + pos_after)
                if qty:
                    conversions_net -= qty
                    orders.append(Order(self.PRODUCT, int(best_bid + 1), qty))
                    buy_open += qty
                    pos_after += qty

        # ---------- 5) CLEAR ----------
        if pos_after > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders)
            if best_bid > fair + self.EDGE_TAKE:
                qty = min(pos_after, depth.buy_orders[best_bid])
                if qty:
                    orders.append(Order(self.PRODUCT, int(best_bid), -qty))
                    sell_open += qty
                    pos_after -= qty
        if pos_after < 0 and depth.sell_orders:
            best_ask = min(depth.sell_orders)
            if best_ask < fair - self.EDGE_TAKE:
                qty = min(-pos_after, -depth.sell_orders[best_ask])
                if qty:
                    orders.append(Order(self.PRODUCT, int(best_ask), qty))
                    buy_open += qty
                    pos_after += qty

        # ---------- 6) MAKE ----------
        bias   = max(-self.SOFT_POS_LIM, min(self.SOFT_POS_LIM, pos_after)) / self.SOFT_POS_LIM
        bid_px = int(round(fair - self.EDGE_JOIN - bias))
        ask_px = int(round(fair + self.EDGE_JOIN - bias))
        if bid_px < ask_px:
            if self.POS_LIMIT - pos_after > 0:
                orders.append(Order(self.PRODUCT, bid_px,  self.POS_LIMIT - pos_after))
            if self.POS_LIMIT + pos_after > 0:
                orders.append(Order(self.PRODUCT, ask_px, -self.POS_LIMIT - pos_after))

        orders_by_prod[self.PRODUCT] = orders
        return orders_by_prod, conversions_net, trader_data