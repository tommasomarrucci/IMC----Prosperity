from datamodel import Order, OrderDepth, TradingState, ConversionObservation
from typing import Dict, List
import jsonpickle
import numpy as np
import math

# ──────────────────────────────────────────────────────────────
#                                PRODUCT NAMES
# ──────────────────────────────────────────────────────────────
class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    RAINFOREST_RESIN     = "RAINFOREST_RESIN"
    SQUID_INK            = "SQUID_INK"
    KELP                 = "KELP"
    # baskets are handled as raw strings below:
    PICNIC_BASKET1       = "PICNIC_BASKET1"
    PICNIC_BASKET2       = "PICNIC_BASKET2"

# ──────────────────────────────────────────────────────────────
#                               PARAMETER STORE
# ──────────────────────────────────────────────────────────────
PARAMS = {
    # ---------- MAGNIFICENT MACARONS (conversion product) ----------
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 1.0,
        "make_min_edge": 0.3,
        "make_probability": 0.5,
        "init_make_edge": 0.9,
        "min_edge": 0.2,
        "volume_avg_timestamp": 5,
        "volume_bar": 50,
        "dec_edge_discount": 0.8,
        "step_size": 0.2,
    },
    # --------------------- RAINFOREST RESIN ------------------------
    Product.RAINFOREST_RESIN: {
        "fair_value": 10_000,
        "take_width": 0.8,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 1.5,
        "default_edge": 3,
        "soft_position_limit": 10,
    },
    # ------------------------ SQUID INK ----------------------------
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.25,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    # --------------------------- KELP ------------------------------
    Product.KELP: {
        "take_width": 1.5,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 20,
        "reversion_beta": -0.10,
        "disregard_edge": 1.5,
        "join_edge": 0.5,
        "default_edge": 1.5,
    },
}

# ──────────────────────────────────────────────────────────────
#                     BASKET ARBITRAGE CONFIG
# ──────────────────────────────────────────────────────────────
BASKET_COMPOSITION = {
    Product.PICNIC_BASKET1: {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
    Product.PICNIC_BASKET2: {"CROISSANTS": 4, "JAMS": 2},
}
BASKET_THRESHOLDS = {
    Product.PICNIC_BASKET1: (+100, -40, 5),
    Product.PICNIC_BASKET2: (+80,  -25, 8),
}
# component & basket limits
LIMITS = {
    Product.MAGNIFICENT_MACARONS: 75,
    Product.RAINFOREST_RESIN: 50,
    Product.SQUID_INK: 50,
    Product.KELP: 50,
    # from second algo:
    "CROISSANTS": 250,
    "JAMS":       350,
    "DJEMBES":     60,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
}

# ──────────────────────────────────────────────────────────────
#                                   TRADER
# ──────────────────────────────────────────────────────────────
class Trader:
    def __init__(self, params: Dict = None):
        self.params = params or PARAMS
        self.LIMIT = LIMITS

    # ──────────────────────────────────────────────
    # 1) MAGNIFICENT MACARONS HELPERS
    # ──────────────────────────────────────────────
    @staticmethod
    def implied_bid_ask(obs: ConversionObservation):
        bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        ask = obs.askPrice + obs.importTariff + obs.transportFees
        return bid, ask

    def adaptive_edge(self, product: str, curr_edge: float, position: int, t_obj: Dict):
        p = self.params[product]
        history = t_obj.setdefault("volume_history", [])
        history.append(abs(position))
        if len(history) > p["volume_avg_timestamp"]:
            history.pop(0)
        if len(history) < p["volume_avg_timestamp"]:
            return curr_edge
        avg_vol = np.mean(history)
        if avg_vol >= p["volume_bar"]:
            t_obj["volume_history"] = []
            return curr_edge + p["step_size"]
        threshold = p["dec_edge_discount"] * p["volume_bar"] * (curr_edge - p["step_size"])
        if threshold > avg_vol * curr_edge:
            t_obj["volume_history"] = []
            return max(curr_edge - p["step_size"], p["min_edge"])
        return curr_edge

    # ──────────────────────────────────────────────
    # 2) GENERIC MARKET‑TAKE & MAKING LOGIC
    # ──────────────────────────────────────────────
    def take_best_orders(
        self, product, fair_value, width, orders, od, position, buy_vol, sell_vol,
        prevent_adverse=False, adverse_volume=0
    ):
        plim = self.LIMIT[product]
        # buy aggressive
        if od.sell_orders:
            best_ask = min(od.sell_orders)
            best_ask_qty = -od.sell_orders[best_ask]
            if (not prevent_adverse or best_ask_qty <= adverse_volume) and best_ask <= fair_value - width:
                qty = min(best_ask_qty, plim - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_vol += qty
                    od.sell_orders[best_ask] += qty
                    if od.sell_orders[best_ask] == 0:
                        del od.sell_orders[best_ask]
        # sell aggressive
        if od.buy_orders:
            best_bid = max(od.buy_orders)
            best_bid_qty = od.buy_orders[best_bid]
            if (not prevent_adverse or best_bid_qty <= adverse_volume) and best_bid >= fair_value + width:
                qty = min(best_bid_qty, plim + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_vol += qty
                    od.buy_orders[best_bid] -= qty
                    if od.buy_orders[best_bid] == 0:
                        del od.buy_orders[best_bid]
        return buy_vol, sell_vol

    def market_make(self, product, orders, bid, ask, position, buy_vol, sell_vol):
        buy_qty = self.LIMIT[product] - (position + buy_vol)
        if buy_qty > 0:
            orders.append(Order(product, round(bid), buy_qty))
        sell_qty = self.LIMIT[product] + (position - sell_vol)
        if sell_qty > 0:
            orders.append(Order(product, round(ask), -sell_qty))
        return buy_vol, sell_vol

    # ──────────────────────────────────────────────
    # 3) FAIR‑VALUE FOR SQUID & KELP
    # ──────────────────────────────────────────────
    def _filtered_mid(self, od: OrderDepth, adverse_volume: int):
        if not (od.buy_orders and od.sell_orders):
            return None
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        big_bids = [p for p, v in od.buy_orders.items() if v >= adverse_volume]
        big_asks = [p for p, v in od.sell_orders.items() if -v >= adverse_volume]
        if not big_bids or not big_asks:
            return (best_bid + best_ask) / 2
        return (max(big_bids) + min(big_asks)) / 2

    def _mean_reversion_fair(self, product, od, t_obj):
        p = self.params[product]
        mid = self._filtered_mid(od, p["adverse_volume"])
        if mid is None:
            return None
        key = f"{product}_last_mid"
        last = t_obj.get(key)
        if last is None:
            fair = mid
        else:
            ret = (mid - last) / last
            fair = mid + mid * (ret * p["reversion_beta"])
        t_obj[key] = mid
        return fair

    # ──────────────────────────────────────────────
    # 4) BASKET ARB HELPERS
    # ──────────────────────────────────────────────
    @staticmethod
    def _mid(depth: OrderDepth):
        if depth and depth.buy_orders and depth.sell_orders:
            return (max(depth.buy_orders) + min(depth.sell_orders)) / 2
        return None

    @staticmethod
    def _cross(product: str, depth: OrderDepth, side: int, qty: int, orders: Dict[str, List[Order]]):
        if not depth or qty <= 0:
            return
        if side > 0:
            for ask, vol in sorted(depth.sell_orders.items()):
                hit = min(qty, abs(vol))
                if hit <= 0:
                    continue
                orders.setdefault(product, []).append(Order(product, ask,  hit))
                qty -= hit
                if qty == 0:
                    break
        else:
            for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
                hit = min(qty, vol)
                if hit <= 0:
                    continue
                orders.setdefault(product, []).append(Order(product, bid, -hit))
                qty -= hit
                if qty == 0:
                    break

    # ──────────────────────────────────────────────
    #            MAIN RUN LOOP
    # ──────────────────────────────────────────────
    def run(self, state: TradingState):
        # persistent state
        t_obj: Dict = jsonpickle.decode(state.traderData) if state.traderData else {}
        results: Dict[str, List[Order]] = {}
        conversions_total = 0

        # ── A) MAGNIFICENT MACARONS (conversion + adaptive spread) ──
        if Product.MAGNIFICENT_MACARONS in state.order_depths:
            prod = Product.MAGNIFICENT_MACARONS
            p    = self.params[prod]
            pos  = state.position.get(prod, 0)
            od   = state.order_depths[prod]
            obs  = state.observations.conversionObservations[prod]

            edge = t_obj.get("curr_edge", p["init_make_edge"])
            edge = self.adaptive_edge(prod, edge, pos, t_obj)
            t_obj["curr_edge"] = edge

            imp_bid, imp_ask = self.implied_bid_ask(obs)
            orders: List[Order] = []

            # TAKE
            for px in sorted(od.sell_orders):
                if px <= imp_bid - edge:
                    q = min(-od.sell_orders[px], self.LIMIT[prod] - pos)
                    if q>0:
                        orders.append(Order(prod, px, q))
                        pos += q
                else:
                    break
            for px in sorted(od.buy_orders, reverse=True):
                if px >= imp_ask + edge:
                    q = min( od.buy_orders[px], self.LIMIT[prod] + pos )
                    if q>0:
                        orders.append(Order(prod, px, -q))
                        pos -= q
                else:
                    break

            # MAKE
            mbid = round(imp_bid - edge)
            mask = round(imp_ask + edge)
            if pos <  self.LIMIT[prod]:
                orders.append(Order(prod, mbid,  self.LIMIT[prod] - pos))
            if pos > -self.LIMIT[prod]:
                orders.append(Order(prod, mask, -(self.LIMIT[prod] + pos)))

            results[prod] = orders
            conversions_total += -pos  # flatten

        # ── B) RAINFOREST RESIN
        if Product.RAINFOREST_RESIN in state.order_depths:
            prod = Product.RAINFOREST_RESIN
            p    = self.params[prod]
            pos  = state.position.get(prod, 0)
            od   = state.order_depths[prod]
            fv   = p["fair_value"]
            orders: List[Order] = []
            bv = sv = 0

            bv, sv = self.take_best_orders(prod, fv, p["take_width"], orders, od, pos, bv, sv)
            # clear large position
            pos_after = pos + bv - sv
            bid_fair = round(fv - p["clear_width"])
            ask_fair = round(fv + p["clear_width"])
            if pos_after>0:
                c = sum(q for px,q in od.buy_orders.items() if px>=ask_fair)
                q = min(pos_after, c)
                if q>0:
                    orders.append(Order(prod, ask_fair, -q))
                    sv += q
            elif pos_after<0:
                c = sum(-q for px,q in od.sell_orders.items() if px<=bid_fair)
                q = min(-pos_after, c)
                if q>0:
                    orders.append(Order(prod, bid_fair, q))
                    bv += q

            # make
            bd = round(fv - p["default_edge"])
            ad = round(fv + p["default_edge"])
            # join/penny logic
            asks = [px for px in od.sell_orders if px>fv+p["disregard_edge"]]
            if asks:
                ref = min(asks)
                ad = ref if (ref-fv)<=p["join_edge"] else ref-1
            bids = [px for px in od.buy_orders if px<fv-p["disregard_edge"]]
            if bids:
                ref = max(bids)
                bd = ref if (fv-ref)<=p["join_edge"] else ref+1

            self.market_make(prod, orders, bd, ad, pos, bv, sv)
            results[prod] = orders

        # ── C) SQUID INK
        if Product.SQUID_INK in state.order_depths:
            prod = Product.SQUID_INK; p = self.params[prod]
            pos = state.position.get(prod, 0)
            od  = state.order_depths[prod]
            fv  = self._mean_reversion_fair(prod, od, t_obj)
            if fv is not None:
                orders: List[Order] = []
                bv=sv=0
                bv, sv = self.take_best_orders(prod, fv, p["take_width"], orders, od, pos, bv, sv, p["prevent_adverse"], p["adverse_volume"])
                # make
                bd, ad = fv - p["default_edge"], fv + p["default_edge"]
                self.market_make(prod, orders, bd, ad, pos, bv, sv)
                results[prod] = orders

        # ── D) KELP
        if Product.KELP in state.order_depths:
            prod = Product.KELP; p = self.params[prod]
            pos = state.position.get(prod, 0)
            od  = state.order_depths[prod]
            fv  = self._mean_reversion_fair(prod, od, t_obj)
            if fv is not None:
                orders: List[Order] = []
                bv=sv=0
                bv, sv = self.take_best_orders(prod, fv, p["take_width"], orders, od, pos, bv, sv, p["prevent_adverse"], p["adverse_volume"])
                # make
                bd, ad = fv - p["default_edge"], fv + p["default_edge"]
                self.market_make(prod, orders, bd, ad, pos, bv, sv)
                results[prod] = orders

        # ── E) PICNIC BASKET ARBITRAGE ───────────────────────────────
        # ensure basket-state
        basket_lp = t_obj.setdefault("basket_last_prices", {})

        # 1) compute mid-prices
        mid: Dict[str, float] = {}
        for prod, od in state.order_depths.items():
            m = self._mid(od)
            if m is not None:
                basket_lp[prod] = m
            mid[prod] = m if m is not None else basket_lp.get(prod, 0.0)

        # 2) premiums
        premiums: Dict[str, float] = {}
        for basket, legs in BASKET_COMPOSITION.items():
            fair = sum(q*mid.get(p,0.0) for p,q in legs.items())
            premiums[basket] = (mid.get(basket,0.0) - fair) if fair else None

        # 3) arb per basket
        for basket, legs in BASKET_COMPOSITION.items():
            prem = premiums[basket]
            if prem is None:
                continue
            sell_thr, buy_thr, ticket = BASKET_THRESHOLDS[basket]
            direction = -1 if prem > sell_thr else (+1 if prem < buy_thr else 0)
            if direction == 0:
                continue
            pos_b  = state.position.get(basket, 0)
            limit_b = self.LIMIT[basket]
            room   = limit_b - pos_b if direction>0 else limit_b + pos_b
            size   = min(ticket, max(0, room))
            if size <= 0:
                continue

            # lift / hit basket
            self._cross(basket, state.order_depths[basket], direction, size, results)

            # hedge legs
            for comp, qty_per in legs.items():
                comp_dir = -direction
                want = qty_per * size
                pos_c = state.position.get(comp, 0)
                limit_c = self.LIMIT.get(comp, 0)
                room_c = limit_c - pos_c if comp_dir>0 else limit_c + pos_c
                want = min(want, max(0, room_c))
                if want>0:
                    self._cross(comp, state.order_depths[comp], comp_dir, want, results)

        # ── pack & return ────────────────────────────────────────────
        traderData = jsonpickle.encode(t_obj)
        return results, conversions_total, traderData