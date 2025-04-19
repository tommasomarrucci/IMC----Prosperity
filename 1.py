from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math, jsonpickle

class Trader:
    # ---------------- static parameters -----------------
    PRODUCT            = "MAGNIFICENT_MACARONS"
    POS_LIMIT          = 75
    CONV_LIMIT         = 10                  # per timestamp
    STORAGE_COST       = 0.1                 # Seashells / unit / ts
    EDGE_TAKE          = 1                   # cross when edge ≥ 1
    EDGE_JOIN          = 2                   # stand inside spread by 2
    SOFT_POS_LIM       = 30                  # start shading quotes
    EW_DECAY           = 0.05                # for offset update
    BASE_OFFSET        = 6.5                 # initial fair‑value premium

    def __init__(self):
        self.offset = self.BASE_OFFSET

    # ---------------- main entry point -----------------
    def run(
        self, state: TradingState
    ) -> tuple[Dict[str, List[Order]], int, str]:

        result: Dict[str, List[Order]] = {}
        conversions_used              = 0
        trader_data                   = {}

        # -------------------------------------------------
        # 0. fetch order‑book & chef quote
        # -------------------------------------------------
        depth: OrderDepth | None = state.order_depths.get(self.PRODUCT)
        obs                      = getattr(state, "observations", None)  # single object

        if depth is None or obs is None:
            # simulator sometimes sends empty book the very first tick
            return {}, 0, ""

        # 1. cost of converting against Pristine Cuisine
        cost_buy  = obs.askPrice + obs.transportFees + obs.importTariff
        cost_sell = obs.bidPrice - obs.transportFees - obs.exportTariff

        # 2. adaptive fair value
        if depth.buy_orders and depth.sell_orders:
            mkt_mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
            anchor  = (cost_buy + cost_sell) / 2
            raw_off = mkt_mid - anchor
            self.offset = (1 - self.EW_DECAY) * self.offset + self.EW_DECAY * raw_off
        fair = (cost_buy + cost_sell) / 2 + self.offset

        position = state.position.get(self.PRODUCT, 0)

        # bookkeeping for later steps
        buy_vol, sell_vol = 0, 0
        orders: List[Order] = []

        # -------------------------------------------------
        # 3. TAKE: lift / hit when book is mispriced
        # -------------------------------------------------
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            ask_qty  = -depth.sell_orders[best_ask]
            if best_ask <= fair - self.EDGE_TAKE:
                qty = min(ask_qty, self.POS_LIMIT - position)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, int(best_ask),  qty))
                    buy_vol += qty

        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            bid_qty  =  depth.buy_orders[best_bid]
            if best_bid >= fair + self.EDGE_TAKE:
                qty = min(bid_qty, self.POS_LIMIT + position)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, int(best_bid), -qty))
                    sell_vol += qty

        # -------------------------------------------------
        # 4. CONVERSION arbitrage   (<=10 units / ts)
        # -------------------------------------------------
        pos_after_take = position + buy_vol - sell_vol

        # pull from chefs if we can instantly re‑sell ≥ EDGE_JOIN higher
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            if best_ask >= cost_buy + self.EDGE_JOIN:
                conv_qty = min(self.CONV_LIMIT,
                               self.POS_LIMIT - pos_after_take)
                if conv_qty > 0:
                    conversions_used += conv_qty          # BUY from chefs
                    orders.append(
                        Order(self.PRODUCT, int(best_ask - 1), -conv_qty)  # re‑sell
                    )
                    sell_vol += conv_qty
                    pos_after_take -= conv_qty

        # push to chefs if book over‑bids
        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            if best_bid <= cost_sell - self.EDGE_JOIN:
                conv_qty = min(self.CONV_LIMIT - conversions_used,
                               self.POS_LIMIT + pos_after_take)
                if conv_qty > 0:
                    conversions_used += conv_qty          # SELL to chefs
                    orders.append(
                        Order(self.PRODUCT, int(best_bid + 1), conv_qty)   # re‑buy
                    )
                    buy_vol += conv_qty
                    pos_after_take += conv_qty

        # -------------------------------------------------
        # 5. CLEAR: dump inventory deep ITM
        # -------------------------------------------------
        if pos_after_take > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders)
            if best_bid > fair + self.EDGE_TAKE:
                qty = min(pos_after_take, depth.buy_orders[best_bid])
                orders.append(Order(self.PRODUCT, int(best_bid), -qty))
                sell_vol += qty
                pos_after_take -= qty

        if pos_after_take < 0 and depth.sell_orders:
            best_ask = min(depth.sell_orders)
            if best_ask < fair - self.EDGE_TAKE:
                qty = min(-pos_after_take, -depth.sell_orders[best_ask])
                orders.append(Order(self.PRODUCT, int(best_ask), qty))
                buy_vol += qty
                pos_after_take += qty

        # -------------------------------------------------
        # 6. MAKE: passive quotes around fair
        # -------------------------------------------------
        inv_bias = max(-self.SOFT_POS_LIM,
                       min(self.SOFT_POS_LIM, pos_after_take)) / self.SOFT_POS_LIM
        bid = int(round(fair - self.EDGE_JOIN - inv_bias))
        ask = int(round(fair + self.EDGE_JOIN - inv_bias))
        if bid < ask:        # safety
            # remaining capacity
            buy_cap  = self.POS_LIMIT - pos_after_take
            sell_cap = self.POS_LIMIT + pos_after_take
            if buy_cap  > 0: orders.append(Order(self.PRODUCT, bid,   buy_cap))
            if sell_cap > 0: orders.append(Order(self.PRODUCT, ask, -sell_cap))

        # -------------------------------------------------
        result[self.PRODUCT] = orders
        return result, conversions_used, jsonpickle.encode(trader_data)