from datamodel import Order, OrderDepth, TradingState, ConversionObservation
from typing import Dict, List
import jsonpickle
import numpy as np
import math

# ===========================================================================
#  Core algorithm ------------------------------------------------------------
#  (unchanged logic from the user’s first bot – renamed to `CoreTrader`)
# ===========================================================================

class Product:
    """Product identifiers traded by the core strategy."""

    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"


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


class CoreTrader:
    """Implements the original logic for macarons / resin / ink / kelp."""

    # ------------------------------------------------------------------
    # Initialisation ----------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(self, params: Dict = None):
        self.params = params if params is not None else PARAMS

        # Position limits for every product we trade
        self.LIMIT = {
            Product.MAGNIFICENT_MACARONS: 75,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
            Product.KELP: 50,
        }

    # ------------------------------------------------------------------
    # 1)  MAGNIFICENT MACARONS helper methods --------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def implied_bid_ask(obs: ConversionObservation):
        """Infer a fair bid/ask from the conversion observation."""
        bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        ask = obs.askPrice + obs.importTariff + obs.transportFees
        return bid, ask

    def adaptive_edge(self, product: str, curr_edge: float, position: int, t_obj: Dict):
        """Volume‑aware spread adjustment (unchanged)."""
        p = self.params[product]
        history = t_obj.setdefault("volume_history", [])

        history.append(abs(position))
        if len(history) > p["volume_avg_timestamp"]:
            history.pop(0)

        # not enough history yet – keep current edge
        if len(history) < p["volume_avg_timestamp"]:
            return curr_edge

        avg_vol = np.mean(history)

        # spread widens on heavy trading
        if avg_vol >= p["volume_bar"]:
            t_obj["volume_history"] = []
            return curr_edge + p["step_size"]

        # spread tightens on light trading
        threshold = p["dec_edge_discount"] * p["volume_bar"] * (curr_edge - p["step_size"])
        if threshold > avg_vol * curr_edge:
            new_edge = max(curr_edge - p["step_size"], p["min_edge"])
            t_obj["volume_history"] = []
            return new_edge

        return curr_edge

    # ------------------------------------------------------------------
    # 2)  Generic market‑taking logic ----------------------------------
    # ------------------------------------------------------------------
    def take_best_orders(
        self,
        product: str,
        fair_value: float,
        width: float,
        orders: List[Order],
        od: OrderDepth,
        position: int,
        buy_vol: int,
        sell_vol: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ):
        plim = self.LIMIT[product]

        # --- consume best ask (→ buy) ---
        if od.sell_orders:
            best_ask = min(od.sell_orders.keys())
            best_ask_qty = -od.sell_orders[best_ask]
            if (not prevent_adverse) or (abs(best_ask_qty) <= adverse_volume):
                if best_ask <= fair_value - width:
                    qty = min(best_ask_qty, plim - position)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        buy_vol += qty
                        od.sell_orders[best_ask] += qty
                        if od.sell_orders[best_ask] == 0:
                            del od.sell_orders[best_ask]

        # --- consume best bid (→ sell) ---
        if od.buy_orders:
            best_bid = max(od.buy_orders.keys())
            best_bid_qty = od.buy_orders[best_bid]
            if (not prevent_adverse) or (abs(best_bid_qty) <= adverse_volume):
                if best_bid >= fair_value + width:
                    qty = min(best_bid_qty, plim + position)
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        sell_vol += qty
                        od.buy_orders[best_bid] -= qty
                        if od.buy_orders[best_bid] == 0:
                            del od.buy_orders[best_bid]

        return buy_vol, sell_vol

    # ------------------------------------------------------------------
    # 3)  Generic market‑making logic ----------------------------------
    # ------------------------------------------------------------------
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: float,
        ask: float,
        position: int,
        buy_vol: int,
        sell_vol: int,
    ):
        buy_qty = self.LIMIT[product] - (position + buy_vol)
        if buy_qty > 0:
            orders.append(Order(product, round(bid), buy_qty))

        sell_qty = self.LIMIT[product] + (position - sell_vol)
        if sell_qty > 0:
            orders.append(Order(product, round(ask), -sell_qty))

        return buy_vol, sell_vol

    # ------------------------------------------------------------------
    # 4)  Fair‑value estimators for SQUID/KELP --------------------------
    # ------------------------------------------------------------------
    def _filtered_mid(self, od: OrderDepth, adverse_volume: int):
        if not (od.buy_orders and od.sell_orders):
            return None
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        big_bids = [p for p, v in od.buy_orders.items() if v >= adverse_volume]
        big_asks = [p for p, v in od.sell_orders.items() if -v >= adverse_volume]
        mm_bid = max(big_bids) if big_bids else None
        mm_ask = min(big_asks) if big_asks else None
        if (mm_bid is None) or (mm_ask is None):
            return (best_bid + best_ask) / 2
        return (mm_bid + mm_ask) / 2

    def _mean_reversion_fair(self, product: str, od: OrderDepth, trader_obj: Dict):
        p = self.params[product]
        mid = self._filtered_mid(od, p["adverse_volume"])
        if mid is None:
            return None
        last_key = f"{product}_last_price"
        last_mid = trader_obj.get(last_key)
        if last_mid is None:
            fair = mid
        else:
            ret = (mid - last_mid) / last_mid
            fair = mid + mid * (ret * p["reversion_beta"])
        trader_obj[last_key] = mid
        return fair

    # ------------------------------------------------------------------
    # Main loop --------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, state: TradingState):
        t_obj: Dict = jsonpickle.decode(state.traderData) if state.traderData else {}
        results: Dict[str, List[Order]] = {}
        conversions_total = 0

        # ================= A) MAGNIFICENT MACARONS =====================
        if Product.MAGNIFICENT_MACARONS in state.order_depths:
            prod = Product.MAGNIFICENT_MACARONS
            p = self.params[prod]
            position = state.position.get(prod, 0)
            od = state.order_depths[prod]
            obs = state.observations.conversionObservations[prod]
            # dynamic spread
            curr_edge = t_obj.get("curr_edge", p["init_make_edge"])
            curr_edge = self.adaptive_edge(prod, curr_edge, position, t_obj)
            t_obj["curr_edge"] = curr_edge
            # implied fair bid/ask
            imp_bid, imp_ask = self.implied_bid_ask(obs)
            orders: List[Order] = []
            # TAKE sells
            for ask_px in sorted(od.sell_orders):
                if ask_px <= imp_bid - curr_edge:
                    vol = min(-od.sell_orders[ask_px], self.LIMIT[prod] - position)
                    if vol > 0:
                        orders.append(Order(prod, ask_px, vol))
                        position += vol
                else:
                    break
            # TAKE bids
            for bid_px in sorted(od.buy_orders, reverse=True):
                if bid_px >= imp_ask + curr_edge:
                    vol = min(od.buy_orders[bid_px], self.LIMIT[prod] + position)
                    if vol > 0:
                        orders.append(Order(prod, bid_px, -vol))
                        position -= vol
                else:
                    break
            # MAKE passive quotes
            make_bid = round(imp_bid - curr_edge)
            make_ask = round(imp_ask + curr_edge)
            if position < self.LIMIT[prod]:
                orders.append(Order(prod, make_bid, self.LIMIT[prod] - position))
            if position > -self.LIMIT[prod]:
                orders.append(Order(prod, make_ask, -(self.LIMIT[prod] + position)))
            results[prod] = orders
            conversions_total += -position  # flatten inventory

        # ================= B) RAINFOREST RESIN =========================
        if Product.RAINFOREST_RESIN in state.order_depths:
            prod = Product.RAINFOREST_RESIN
            p = self.params[prod]
            position = state.position.get(prod, 0)
            od = state.order_depths[prod]
            fv = p["fair_value"]
            orders: List[Order] = []
            buy_v = sell_v = 0
            buy_v, sell_v = self.take_best_orders(prod, fv, p["take_width"], orders, od, position, buy_v, sell_v)
            position_after_take = position + buy_v - sell_v
            fair_bid = round(fv - p["clear_width"])
            fair_ask = round(fv + p["clear_width"])
            if position_after_take > 0:
                clear_qty = min(position_after_take, sum(q for px, q in od.buy_orders.items() if px >= fair_ask))
                if clear_qty > 0:
                    orders.append(Order(prod, fair_ask, -clear_qty))
                    sell_v += clear_qty
            elif position_after_take < 0:
                clear_qty = min(-position_after_take, sum(-q for px, q in od.sell_orders.items() if px <= fair_bid))
                if clear_qty > 0:
                    orders.append(Order(prod, fair_bid, clear_qty))
                    buy_v += clear_qty
            bid_default = round(fv - p["default_edge"])
            ask_default = round(fv + p["default_edge"])
            ask_ref = min([px for px in od.sell_orders if px > fv + p["disregard_edge"]], default=None)
            if ask_ref is not None:
                ask_default = ask_ref if (ask_ref - fv) <= p["join_edge"] else ask_ref - 1
            bid_ref = max([px for px in od.buy_orders if px < fv - p["disregard_edge"]], default=None)
            if bid_ref is not None:
                bid_default = bid_ref if (fv - bid_ref) <= p["join_edge"] else bid_ref + 1
            buy_v, sell_v = self.market_make(prod, orders, bid_default, ask_default, position, buy_v, sell_v)
            results[prod] = orders

        # ================= C) SQUID INK ===============================
        if Product.SQUID_INK in state.order_depths:
            prod = Product.SQUID_INK
            p = self.params[prod]
            position = state.position.get(prod, 0)
            od = state.order_depths[prod]
            fv = self._mean_reversion_fair(prod, od, t_obj)
            if fv is not None:
                orders: List[Order] = []
                buy_v = sell_v = 0
                buy_v, sell_v = self.take_best_orders(
                    prod, fv, p["take_width"], orders, od, position, buy_v, sell_v,
                    p["prevent_adverse"], p["adverse_volume"],
                )
                ask, bid = fv + p["default_edge"], fv - p["default_edge"]
                buy_v, sell_v = self.market_make(prod, orders, bid, ask, position, buy_v, sell_v)
                results[prod] = orders

        # ================= D) KELP ====================================
        if Product.KELP in state.order_depths:
            prod = Product.KELP
            p = self.params[prod]
            position = state.position.get(prod, 0)
            od = state.order_depths[prod]
            fv = self._mean_reversion_fair(prod, od, t_obj)
            if fv is not None:
                orders: List[Order] = []
                buy_v = sell_v = 0
                buy_v, sell_v = self.take_best_orders(
                    prod, fv, p["take_width"], orders, od, position, buy_v, sell_v,
                    p["prevent_adverse"], p["adverse_volume"],
                )
                ask, bid = fv + p["default_edge"], fv - p["default_edge"]
                buy_v, sell_v = self.market_make(prod, orders, bid, ask, position, buy_v, sell_v)
                results[prod] = orders

        # ------------------------------------------------------------------
        traderData = jsonpickle.encode(t_obj)
        return results, conversions_total, traderData


# ===========================================================================
#  Basket arbitrage algorithm ------------------------------------------------
#  (trades only picnic baskets + component hedges – unchanged)
# ===========================================================================

class BasketArbTrader:
    """Trades PICNIC_BASKET1 and PICNIC_BASKET2 against their components."""

    BASKET_COMPOSITION = {
        "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
        "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2},
    }

    LIMIT = {
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
    }

    THRESHOLDS = {
        "PICNIC_BASKET1": (+100, -40, 5),
        "PICNIC_BASKET2": (+80, -25, 8),
    }

    def __init__(self):
        self.last_prices: Dict[str, float] = {}

    # ------------------------------- helpers -----------------------------
    @staticmethod
    def _mid(depth: OrderDepth):
        if depth and depth.buy_orders and depth.sell_orders:
            return (max(depth.buy_orders) + min(depth.sell_orders)) / 2
        return None

    def _cross(self, product: str, depth: OrderDepth, side: int, qty: int, orders: Dict[str, List[Order]]):
        if qty <= 0 or depth is None:
            return
        if side > 0:  # BUY
            for ask, vol in sorted(depth.sell_orders.items()):
                hit = min(qty, abs(vol))
                if hit <= 0:
                    continue
                orders.setdefault(product, []).append(Order(product, ask, hit))
                qty -= hit
                if qty == 0:
                    break
        else:  # SELL
            for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
                hit = min(qty, vol)
                if hit <= 0:
                    continue
                orders.setdefault(product, []).append(Order(product, bid, -hit))
                qty -= hit
                if qty == 0:
                    break

    # ------------------------------- main ---------------------------------
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0  # basket strategy performs no conversions
        tdata = jsonpickle.decode(state.traderData) if state.traderData else {}

        # 1) mid prices ---------------------------------------------------
        mid: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            m = self._mid(depth)
            if m is not None:
                self.last_prices[product] = m
            mid[product] = m if m is not None else self.last_prices.get(product)

        # 2) premiums -----------------------------------------------------
        premiums: Dict[str, float] = {}
        for basket, legs in self.BASKET_COMPOSITION.items():
            fair = sum(qty * mid.get(prod, 0) for prod, qty in legs.items())
            premiums[basket] = (mid.get(basket, 0) - fair) if fair else None

        # 3) trading loop -------------------------------------------------
        for basket, legs in self.BASKET_COMPOSITION.items():
            prem = premiums[basket]
            if prem is None:
                continue
            prem_sell, prem_buy, ticket = self.THRESHOLDS[basket]
            direction = -1 if prem > prem_sell else (+1 if prem < prem_buy else 0)
            if not direction:
                continue
            pos_basket = state.position.get(basket, 0)
            limit_basket = self.LIMIT[basket]
            room = limit_basket - pos_basket if direction > 0 else limit_basket + pos_basket
            size_basket = min(ticket, max(0, room))
            if size_basket == 0:
                continue
            # 3a) execute basket leg --------------------------------------
            self._cross(basket, state.order_depths[basket], direction, size_basket, orders)
            # 3b) hedge components ---------------------------------------
            for prod, qty_per in legs.items():
                comp_dir = -direction
                want = qty_per * size_basket
                pos_comp = state.position.get(prod, 0)
                limit_comp = self.LIMIT[prod]
                room_comp = limit_comp - pos_comp if comp_dir > 0 else limit_comp + pos_comp
                want = min(want, max(0, room_comp))
                if want:
                    self._cross(prod, state.order_depths[prod], comp_dir, want, orders)
        return orders, conversions, jsonpickle.encode(tdata)


# ===========================================================================
#  Combined trader -----------------------------------------------------------
# ===========================================================================

class Trader:
    """Aggregates `CoreTrader` and `BasketArbTrader` into a single strategy."""

    def __init__(self):
        self.core = CoreTrader()
        self.basket = BasketArbTrader()

    def run(self, state: TradingState):
        # --------------------------------------------------------------
        # Split traderData into sub‑objects ----------------------------
        # --------------------------------------------------------------
        outer_td = jsonpickle.decode(state.traderData) if state.traderData else {}
        core_td = outer_td.get("core")
        basket_td = outer_td.get("basket")

        # --------------------------------------------------------------
        # Run core strategy -------------------------------------------
        # --------------------------------------------------------------
        original_td = state.traderData  # backup outer blob
        state.traderData = core_td
        orders_core, conv_core, new_core_td = self.core.run(state)

        # --------------------------------------------------------------
        # Run basket strategy -----------------------------------------
        # --------------------------------------------------------------
        state.traderData = basket_td
        orders_basket, conv_basket, new_basket_td = self.basket.run(state)

        # restore original traderData (defensive – not strictly needed)
        state.traderData = original_td

        # --------------------------------------------------------------
        # Merge orders & conversions ----------------------------------
        # --------------------------------------------------------------
        all_orders: Dict[str, List[Order]] = {}
        for book in (orders_core, orders_basket):
            for prod, lst in book.items():
                all_orders.setdefault(prod, []).extend(lst)

        conversions = conv_core + conv_basket  # basket currently 0

        packed_td = jsonpickle.encode({"core": new_core_td, "basket": new_basket_td})

        return all_orders, conversions, packed_td
