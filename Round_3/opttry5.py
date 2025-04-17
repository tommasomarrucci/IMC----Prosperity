from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np
import math
import statistics

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
        self.implied_volatility = 0.2  # Calibrated value, adjust if needed
        self.last_prices = {}

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        conversions = 0
        traderData = ""

        # Estimate underlying price from order book
        underlying_price = self.get_mid_price(state.order_depths.get(self.underlying_symbol))
        if underlying_price is not None:
            self.last_prices[self.underlying_symbol] = underlying_price
        else:
            underlying_price = self.last_prices.get(self.underlying_symbol, 10000)

        # Delta hedge total
        total_delta = 0
        option_gammas = {}
        option_deltas = {}
        option_values = {}

        days_left = max(1, 7 - state.timestamp)
        T = days_left / 365

        for symbol, strike in self.option_symbols:
            order_depth = state.order_depths.get(symbol)
            if order_depth is None:
                continue

            option_price = self.get_mid_price(order_depth)
            if option_price is None:
                continue

            self.last_prices[symbol] = option_price

            delta, gamma, theo_price = self.black_scholes_all(underlying_price, strike, T, self.implied_volatility)

            option_gammas[symbol] = gamma
            option_deltas[symbol] = delta
            option_values[symbol] = theo_price

            total_delta += delta * self.position(state, symbol)

            # Trade logic
            orders = []
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < theo_price:
                    trade_vol = min(5, abs(ask_vol))
                    orders.append(Order(symbol, ask_price, trade_vol))

            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > theo_price:
                    trade_vol = min(5, abs(bid_vol))
                    orders.append(Order(symbol, bid_price, -trade_vol))

            result[symbol] = orders

        # Delta hedge the position
        hedge_orders = []
        underlying_depth = state.order_depths.get(self.underlying_symbol)
        if underlying_depth:
            for bid_price, bid_vol in sorted(underlying_depth.buy_orders.items(), reverse=True):
                if total_delta < -5:
                    volume = min(10, abs(bid_vol), int(-total_delta))
                    hedge_orders.append(Order(self.underlying_symbol, bid_price, -volume))
                    total_delta += volume

            for ask_price, ask_vol in sorted(underlying_depth.sell_orders.items()):
                if total_delta > 5:
                    volume = min(10, abs(ask_vol), int(total_delta))
                    hedge_orders.append(Order(self.underlying_symbol, ask_price, volume))
                    total_delta -= volume

            result[self.underlying_symbol] = hedge_orders

        return result, conversions, traderData

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

    def black_scholes_all(self, S, K, T, sigma):
        if T <= 0:
            return 0, 0, max(0, S - K)
        d1 = (math.log(S / K) + (0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        delta = self.norm_cdf(d1)
        gamma = self.norm_pdf(d1) / (S * sigma * math.sqrt(T))
        price = S * self.norm_cdf(d1) - K * math.exp(-self.risk_free_rate * T) * self.norm_cdf(d2)
        return delta, gamma, price

    def norm_pdf(self, x):
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    def norm_cdf(self, x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)
