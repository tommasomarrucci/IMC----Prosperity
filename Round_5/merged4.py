# ================================================================
#  Prosperity – Round 5 combined trading algorithm
#  --------------------------------------------------------------
#  * Options valuation upgraded for near‑expiry (2 days left)
#  * Strike‑specific implied‑vol surface fitted each tick
#  * Tighter delta hedge, updated Macaron CSI & momentum params
#  * Basket & base‑commodity logic kept (minor param tweaks)
# ================================================================

from datamodel import Order, TradingState, OrderDepth, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np
import math

# ================================================================
#  Configuration helpers
# ================================================================

SECONDS_IN_TRADING_DAY: int = 10_000   # ≈ number of ticks in one “day”
DAYS_TO_EXPIRY_INITIAL: int = 2        # Round‑5 spec (at the start of Round 5)
RISK_FREE_RATE: float = 0.0            # still zero

# ================================================================
#  Product enums
# ================================================================

class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

# ================================================================
#  Parameters – tuned after Round‑4 data inspection
# ================================================================

PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        # market‑making edge controls
        "make_edge": 1.0,
        "make_min_edge": 0.3,
        "init_make_edge": 0.9,
        "min_edge": 0.2,
        # volume adaptive edge controls
        "volume_avg_timestamp": 12,   # ↑ from 5
        "volume_bar": 80,             # ↑ from 50
        "dec_edge_discount": 0.8,
        "step_size": 0.2,
        # macro factors
        "CSI": 50.0,                  # Critical Sunlight Index (↑ from 40)
        "sun_bias_edge": 0.2,
        # conversion
        "conv_limit": 10,
        # sugar correlation
        "sugar_momentum_window": 5,   # ↑ from 3
        "sugar_bias_edge": 0.1,
    }
}

# ================================================================
#  Macaron + Options Trader
# ================================================================

class MacaronOptionsTrader:
    """Handles (i) MACARON market‑making and (ii) VOLCANIC_ROCK options."""

    def __init__(self):
        self.params = PARAMS
        self.LIMIT: Dict[str, int] = {Product.MAGNIFICENT_MACARONS: 75}

        # --- option symbols & caches ----------------------------
        self.underlying_symbol: str = "VOLCANIC_ROCK"
        self.option_symbols: List[Tuple[str, int]] = [
            ("VOLCANIC_ROCK_VOUCHER_9500", 9500),
            ("VOLCANIC_ROCK_VOUCHER_9750", 9750),
            ("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            ("VOLCANIC_ROCK_VOUCHER_10250", 10250),
            ("VOLCANIC_ROCK_VOUCHER_10500", 10500),
        ]
        self.risk_free_rate: float = RISK_FREE_RATE
        # last prices & IVs cache
        self.last_prices: Dict[str, float] = {}
        self.last_vols: Dict[str, float] = {}

    # ------------------------------------------------------------
    #  Helpers (general)
    # ------------------------------------------------------------

    @staticmethod
    def norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_all(self, S: float, K: float, T: float, sigma: float):
        """Return (delta, gamma, price) for a European call."""
        if T <= 0:
            price = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
            gamma = 0.0
            return delta, gamma, price
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        delta = self.norm_cdf(d1)
        gamma = self.norm_pdf(d1) / (S * sigma * math.sqrt(T))
        price = S * self.norm_cdf(d1) - K * math.exp(-self.risk_free_rate * T) * self.norm_cdf(d2)
        return delta, gamma, price

    # ---------- internal BS components for IV root finding -------
    def _bs_price(self, S: float, K: float, T: float, sigma: float) -> float:
        return self.black_scholes_all(S, K, T, sigma)[2]

    def _bs_vega(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        return S * math.sqrt(T) * self.norm_pdf(d1)

    # ---------- order‑book helper utilities ---------------------
    @staticmethod
    def get_mid_price(depth: OrderDepth):
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

    def position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    # ============================================================
    #  MACARON leg (same logic – updated parameters only)
    # ============================================================

    def implied_bid_ask(self, obs: ConversionObservation):
        bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        ask = obs.askPrice + obs.importTariff + obs.transportFees
        return bid, ask

    def adaptive_edge(self, curr_edge: float, position: int, traderObject: dict) -> float:
        p = self.params[Product.MAGNIFICENT_MACARONS]
        hist = traderObject.setdefault("volume_history", [])
        hist.append(abs(position))
        if len(hist) > p["volume_avg_timestamp"]:
            hist.pop(0)
        if len(hist) < p["volume_avg_timestamp"]:
            return curr_edge
        avg_vol = np.mean(hist)
        if avg_vol >= p["volume_bar"]:
            traderObject["volume_history"] = []
            return curr_edge + p["step_size"]
        if p["dec_edge_discount"] * p["volume_bar"] * (curr_edge - p["step_size"]) > avg_vol * curr_edge:
            traderObject["volume_history"] = []
            return max(curr_edge - p["step_size"], p["min_edge"])
        return curr_edge

    # ============================================================
    #  run() – main entry point for this sub‑trader
    # ============================================================

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0  # not used (platform‑specific)

        # --------------------------------------------------------
        #  Load private state from traderData blob ---------------
        # --------------------------------------------------------
        traderObject: dict = jsonpickle.decode(state.traderData) if state.traderData else {}

        # --------------------------------------------------------
        #  1) Magnificent Macarons strategy
        # --------------------------------------------------------
        mac = Product.MAGNIFICENT_MACARONS
        if mac in state.order_depths:
            pos_mac = self.position(state, mac)
            obs: ConversionObservation = state.observations.conversionObservations[mac]
            depth: OrderDepth = state.order_depths[mac]

            # dynamic edge control
            edge = traderObject.get("curr_edge", self.params[mac]["init_make_edge"])
            edge = self.adaptive_edge(edge, pos_mac, traderObject)
            traderObject["curr_edge"] = edge

            # implied fair bid / ask from chefs + biases
            implied_bid, implied_ask = self.implied_bid_ask(obs)

            # sugar momentum
            s_hist = traderObject.setdefault("sugar_history", [])
            s_hist.append(obs.sugarPrice)
            if len(s_hist) > self.params[mac]["sugar_momentum_window"]:
                s_hist.pop(0)
            sugar_mom = (s_hist[-1] - s_hist[0]) if len(s_hist) == self.params[mac]["sugar_momentum_window"] else 0.0

            bid_edge = ask_edge = edge
            if sugar_mom > 0:
                bid_edge = max(edge - self.params[mac]["sugar_bias_edge"], self.params[mac]["min_edge"])
            elif sugar_mom < 0:
                ask_edge = edge + self.params[mac]["sugar_bias_edge"]

            # sunlight bias
            if obs.sunlightIndex < self.params[mac]["CSI"]:
                bid_edge = max(bid_edge - self.params[mac]["sun_bias_edge"], self.params[mac]["min_edge"])
                ask_edge = ask_edge + self.params[mac]["sun_bias_edge"]

            orders_mac: List[Order] = []
            limit_mac = self.LIMIT[mac]

            # aggressive (take) fills
            for ask, vol in sorted(depth.sell_orders.items()):
                if ask <= implied_bid - bid_edge:
                    qty = min(vol, limit_mac - pos_mac)
                    if qty > 0:
                        orders_mac.append(Order(mac, ask, qty))
                        pos_mac += qty
            for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
                if bid >= implied_ask + ask_edge:
                    qty = min(vol, limit_mac + pos_mac)
                    if qty > 0:
                        orders_mac.append(Order(mac, bid, -qty))
                        pos_mac -= qty

            # passive make orders
            mbid = round(implied_bid - bid_edge)
            mask = round(implied_ask + ask_edge)
            if pos_mac < limit_mac:
                orders_mac.append(Order(mac, mbid, limit_mac - pos_mac))
            if pos_mac > -limit_mac:
                orders_mac.append(Order(mac, mask, -(limit_mac + pos_mac)))

            if orders_mac:
                result[mac] = orders_mac

        # --------------------------------------------------------
        #  2) VOLCANIC_ROCK base price (mid) ----------------------
        # --------------------------------------------------------
        ud_depth = state.order_depths.get(self.underlying_symbol)
        mid_u = self.get_mid_price(ud_depth)
        if mid_u is not None:
            self.last_prices[self.underlying_symbol] = mid_u
        else:
            mid_u = self.last_prices.get(self.underlying_symbol, 10000.0)

        # --------------------------------------------------------
        #  3) Options valuation near expiry ----------------------
        # --------------------------------------------------------

        # --- time‑to‑expiry (in years) -------------------------
        #  <<<<< new –‑ keep it frozen at 2 days >>>>>
        T = 2 / 365        # constant for the entire Round‑5 evaluation day

        # --- gather mid prices & infer IV per strike -----------
        moneynesses: List[float] = []
        vols:         List[float] = []
        option_data:  List[Tuple[str, int, float]] = []  # (sym, K, mid)

        for sym, strike in self.option_symbols:
            od = state.order_depths.get(sym)
            mp = self.get_mid_price(od)
            if mp is None:
                mp = self.last_prices.get(sym)
            if mp is None:
                continue
            self.last_prices[sym] = mp
            option_data.append((sym, strike, mp))

            # --- Newton root find for implied vol -------------
            sigma = self.last_vols.get(sym, 0.2)
            for _ in range(30):
                pr = self._bs_price(mid_u, strike, T, sigma)
                diff = pr - mp
                if abs(diff) < 0.1:
                    break
                vega = self._bs_vega(mid_u, strike, T, sigma) or 1e-6
                sigma = max(0.01, min(2.0, sigma - diff / vega))
            self.last_vols[sym] = sigma
            m = math.log(strike / mid_u) / math.sqrt(T) if T > 0 else 0.0
            moneynesses.append(m)
            vols.append(sigma)

        # --- fit quadratic smile if enough strikes ------------
        if len(moneynesses) >= 3:
            a, b, c = np.polyfit(moneynesses, vols, 2)
            smile = lambda m: max(0.05, a * m * m + b * m + c)
        else:
            mean_vol = np.mean(vols) if vols else 0.2
            smile = lambda m: max(0.05, mean_vol)

        # --- generate option orders & compute total delta ------
        total_delta = 0.0
        CROSS_EDGE = 1.5     # price units
        PASSIVE_SIZE = 10
        HEDGE_BAND  = 5

        for sym, strike, mp in option_data:
            # theoretical price with smoothed vol
            m = math.log(strike / mid_u) / math.sqrt(T) if T > 0 else 0.0
            sigma_sm = smile(m)
            delta, gamma, theo = self.black_scholes_all(mid_u, strike, T, sigma_sm)
            total_delta += delta * self.position(state, sym)

            od = state.order_depths.get(sym)
            if od is None:
                continue
            orders_opt: List[Order] = []

            # cross obvious mis‑pricings
            for ask, vol in sorted(od.sell_orders.items()):
                if theo - ask > CROSS_EDGE:
                    hit = min(PASSIVE_SIZE, vol)
                    orders_opt.append(Order(sym, ask, hit))
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if bid - theo > CROSS_EDGE:
                    hit = min(PASSIVE_SIZE, vol)
                    orders_opt.append(Order(sym, bid, -hit))

            # passive quotes
            pos_opt = self.position(state, sym)
            if pos_opt < 200:
                orders_opt.append(Order(sym, round(theo - 0.5), PASSIVE_SIZE))
            if pos_opt > -200:
                orders_opt.append(Order(sym, round(theo + 0.5), -PASSIVE_SIZE))

            if orders_opt:
                result.setdefault(sym, []).extend(orders_opt)

        # --------------------------------------------------------
        #  4) Hedge underlying delta -----------------------------
        # --------------------------------------------------------
        ud_orders: List[Order] = []
        if ud_depth is not None:
            for ask, vol in sorted(ud_depth.sell_orders.items()):
                if total_delta > HEDGE_BAND:
                    take = min(vol, int(total_delta))
                    if take > 0:
                        ud_orders.append(Order(self.underlying_symbol, ask, take))
                        total_delta -= take
            for bid, vol in sorted(ud_depth.buy_orders.items(), reverse=True):
                if total_delta < -HEDGE_BAND:
                    take = min(vol, int(-total_delta))
                    if take > 0:
                        ud_orders.append(Order(self.underlying_symbol, bid, -take))
                        total_delta += take
        if ud_orders:
            result[self.underlying_symbol] = ud_orders

        # --------------------------------------------------------
        # 5) Pack & return --------------------------------------
        # --------------------------------------------------------
        return result, conversions, jsonpickle.encode(traderObject)

# ================================================================
#  Basket Arbitrage Trader (unchanged except for imports)
# ================================================================

class BasketArbTrader:
    """Trades Picnic Baskets vs components."""

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

    THRESHOLDS = {  # (sellPrem, buyPrem, ticketSize)
        "PICNIC_BASKET1": (+100, -40, 5),
        "PICNIC_BASKET2": (+80, -25, 8),
    }

    def __init__(self):
        self.last_prices: Dict[str, float] = {}

    # ------------------------------------------------------------
    def _mid(self, depth: OrderDepth):
        if depth and depth.buy_orders and depth.sell_orders:
            return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
        return None

    def _cross(self, product: str, depth: OrderDepth, side: int, qty: int, orders: Dict[str, List[Order]]):
        """Hit/lift opposite best quotes until qty fulfilled."""
        if qty <= 0 or depth is None:
            return
        if side > 0:  # BUY
            for ask, vol in sorted(depth.sell_orders.items()):
                take = min(qty, abs(vol))
                if take:
                    orders.setdefault(product, []).append(Order(product, ask, take))
                    qty -= take
                if qty == 0:
                    break
        else:         # SELL
            for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
                take = min(qty, vol)
                if take:
                    orders.setdefault(product, []).append(Order(product, bid, -take))
                    qty -= take
                if qty == 0:
                    break

    # ------------------------------------------------------------
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        conversions = 0
        tdata = jsonpickle.decode(state.traderData) if state.traderData else {}

        # 1) mid prices cache
        mid: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            m = self._mid(depth)
            if m is not None:
                self.last_prices[product] = m
            mid[product] = m if m is not None else self.last_prices.get(product)

        # 2) basket premiums
        premiums: Dict[str, float] = {}
        for basket, legs in self.BASKET_COMPOSITION.items():
            fair = sum(qty * mid.get(prod, 0.0) for prod, qty in legs.items())
            basket_mid = mid.get(basket)
            premiums[basket] = basket_mid - fair if fair and basket_mid else None

        # 3) arbitrage logic
        for basket, legs in self.BASKET_COMPOSITION.items():
            prem = premiums[basket]
            if prem is None:
                continue
            prem_sell, prem_buy, ticket = self.THRESHOLDS[basket]
            direction = -1 if prem > prem_sell else (1 if prem < prem_buy else 0)
            if not direction:
                continue

            # room in basket position
            pos_b = state.position.get(basket, 0)
            lim_b = self.LIMIT[basket]
            room_b = lim_b - pos_b if direction > 0 else lim_b + pos_b
            size_b = min(ticket, max(0, room_b))
            if size_b == 0:
                continue

            # 3a) cross basket
            self._cross(basket, state.order_depths[basket], direction, size_b, orders)

            # 3b) hedge components
            for prod, qty_per in legs.items():
                comp_dir = -direction
                want = qty_per * size_b
                pos_p = state.position.get(prod, 0)
                lim_p = self.LIMIT[prod]
                room_p = lim_p - pos_p if comp_dir > 0 else lim_p + pos_p
                want = min(want, max(0, room_p))
                if want:
                    self._cross(prod, state.order_depths[prod], comp_dir, want, orders)

        return orders, conversions, jsonpickle.encode(tdata)

# ================================================================
#  Wrapper Trader combining sub‑algorithms
# ================================================================

class Trader:
    """Top‑level trader that delegates to Macaron+Options and Basket bots."""

    def __init__(self):
        self.mac = MacaronOptionsTrader()
        self.basket = BasketArbTrader()

    def run(self, state: TradingState):
        # decode outer traderData into sub‑blobs
        outer_td = jsonpickle.decode(state.traderData) if state.traderData else {}
        mac_td = outer_td.get("mac")
        basket_td = outer_td.get("basket")

        # --- run Macaron & Options
        orig_td = state.traderData
        state.traderData = mac_td
        orders_mac, conv_mac, new_mac_td = self.mac.run(state)

        # --- run Basket arb
        state.traderData = basket_td
        orders_basket, conv_bsk, new_basket_td = self.basket.run(state)
        state.traderData = orig_td  # restore

        # merge orders
        all_orders: Dict[str, List[Order]] = {}
        for book in (orders_mac, orders_basket):
            for prod, lst in book.items():
                all_orders.setdefault(prod, []).extend(lst)

        conversions = 0  # conversions handled inside sub‑algos if supported
        packed_td = jsonpickle.encode({"mac": new_mac_td, "basket": new_basket_td})

        return all_orders, conversions, packed_td
