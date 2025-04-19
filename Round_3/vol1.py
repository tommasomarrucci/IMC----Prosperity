from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np
import math
from typing import Dict, List, Tuple

def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
    # … your logic …
    return result, conversions, traderData

# 1. Implied–Realized Volatility Arbitrage
class Trader:
    def __init__(self):
        self.underlying_symbol = "VOLCANIC_ROCK"
        self.option_symbols = [
            ("VOLCANIC_ROCK_VOUCHER_9500", 9500),
            ("VOLCANIC_ROCK_VOUCHER_9750", 9750),
            ("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            ("VOLCANIC_ROCK_VOUCHER_10250", 10250),
            ("VOLCANIC_ROCK_VOUCHER_10500", 10500)
        ]
        self.risk_free_rate = 0.0
        self.last_prices = {}
        # for realized vol
        self.price_history: List[float] = []
        self.history_window = 20
        self.basis_threshold = 0.02

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        # 1) get mid underlying
        underlying_depth = state.order_depths.get(self.underlying_symbol)
        underlying_price = self.get_mid_price(underlying_depth)
        if underlying_price is not None:
            self.last_prices[self.underlying_symbol] = underlying_price
            self.price_history.append(underlying_price)
            if len(self.price_history) > self.history_window:
                self.price_history.pop(0)
        else:
            underlying_price = self.last_prices.get(self.underlying_symbol, 10000)

        # 2) compute realized vol
        rv = self.compute_realized_vol()

        # 3) loop options
        total_delta = 0.0
        for symbol, K in self.option_symbols:
            od = state.order_depths.get(symbol)
            if od is None: continue
            mid = self.get_mid_price(od)
            if mid is None: continue
            self.last_prices[symbol] = mid

            # implied vol via bisection
            iv = self.implied_vol(mid, underlying_price, K, self.time_to_expiry(state))
            basis = iv - rv

            # trade if basis large
            orders: List[Order] = []
            if basis > self.basis_threshold:
                # IV >> realized → net short vega
                orders.append(Order(symbol, mid, -5))
            elif basis < -self.basis_threshold:
                # IV << realized → net long vega
                orders.append(Order(symbol, mid, +5))
            result[symbol] = orders

            # track delta for hedge
            delta, _, _ = self.black_scholes_all(underlying_price, K, self.time_to_expiry(state), iv)
            total_delta += delta * self.position(state, symbol)

        # 4) delta‐hedge underlying
        hedge = []
        if underlying_depth:
            for price, vol in sorted(underlying_depth.buy_orders.items(), reverse=True):
                if total_delta < 0:
                    q = min(abs(vol), int(-total_delta), 10)
                    hedge.append(Order(self.underlying_symbol, price, -q))
                    total_delta += q
            for price, vol in sorted(underlying_depth.sell_orders.items()):
                if total_delta > 0:
                    q = min(abs(vol), int(total_delta), 10)
                    hedge.append(Order(self.underlying_symbol, price, +q))
                    total_delta -= q
        result[self.underlying_symbol] = hedge

        return result, conversions, traderData

    def compute_realized_vol(self) -> float:
        if len(self.price_history) < 2:
            return 0.2
        lr = np.diff(np.log(self.price_history))
        return float(np.std(lr) * math.sqrt(252))

    def implied_vol(self, market_price, S, K, T):
        # Bisection solver on sigma in [1e-4,5]
        low, high = 1e-4, 5.0
        for _ in range(50):
            mid = (low + high) / 2
            p = self.bs_price(S, K, T, mid)
            if p > market_price:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def bs_price(self, S, K, T, sigma):
        if T <= 0: return max(0.0, S - K)
        d1 = (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return S*self.norm_cdf(d1) - K*math.exp(-self.risk_free_rate*T)*self.norm_cdf(d2)

    def time_to_expiry(self, state): 
        days = max(1, 7 - state.timestamp)
        return days/365

    def get_mid_price(self, order_depth):
        if order_depth is None:
            return None
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        if not bids or not asks:
            return None
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        return (best_bid + best_ask) / 2

    def black_scholes_all(self, S,K,T,sigma):
        if T<=0: return 0,0,max(0,S-K)
        d1 = (math.log(S/K)+(0.5*sigma**2)*T)/(sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        delta = self.norm_cdf(d1)
        gamma = self.norm_pdf(d1)/(S*sigma*math.sqrt(T))
        price = S*self.norm_cdf(d1)-K*math.exp(-self.risk_free_rate*T)*self.norm_cdf(d2)
        return delta, gamma, price

    def norm_pdf(self, x): 
        return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
    def norm_cdf(self, x): 
        return 0.5*(1+math.erf(x/math.sqrt(2)))
    def position(self, state, prod): 
        return state.position.get(prod, 0)