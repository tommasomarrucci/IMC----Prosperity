from Round_3.datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import pandas as pd
import math

class Product:
    SQUID_INK = "SQUID_INK"

class Trader:
    def __init__(self):
        self.SQUID_INK_prices = []
        self.LIMIT = {
            Product.SQUID_INK: 50
        }
        self.prev_position_signal = 0

        # EMA Parameters
        self.short_window = 75
        self.long_window = 603
        self.min_periods = 50

    def ema(self, series: np.ndarray, span: int) -> np.ndarray:
        return pd.Series(series).ewm(span=span, adjust=False).mean().values

    def calculate_ema_cross(self, prices: List[float]):
        price_array = np.array(prices)
        ema_short = self.ema(price_array, self.short_window)
        ema_long = self.ema(price_array, self.long_window)
        return ema_short[-1], ema_long[-1]

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        return mid_price

    def take_best_orders(self, product: str, fair_value: float, position_signal: int, orders: List[Order], order_depth: OrderDepth, position: int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -order_depth.sell_orders[best_ask]
            if position_signal == 1:  # Buy signal
                quantity = min(best_ask_volume, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            if position_signal == -1:  # Sell signal
                quantity = min(best_bid_volume, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))

    def clear_position_order(self, product: str, fair_value: float, orders: List[Order], order_depth: OrderDepth, position: int):
        # If holding a wrong position after a signal flip, clear at fair value
        fair_bid = math.floor(fair_value)
        fair_ask = math.ceil(fair_value)

        if position > 0:
            if fair_bid in order_depth.buy_orders:
                sell_qty = min(position, order_depth.buy_orders[fair_bid])
                if sell_qty > 0:
                    orders.append(Order(product, fair_bid, -sell_qty))

        elif position < 0:
            if fair_ask in order_depth.sell_orders:
                buy_qty = min(-position, -order_depth.sell_orders[fair_ask])
                if buy_qty > 0:
                    orders.append(Order(product, fair_ask, buy_qty))

    def SQUID_INK_orders(self, order_depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            mid_price = self.get_mid_price(order_depth)
            self.SQUID_INK_prices.append(mid_price)

            if len(self.SQUID_INK_prices) > 700:
                self.SQUID_INK_prices.pop(0)

            if len(self.SQUID_INK_prices) >= self.min_periods:
                ema_short, ema_long = self.calculate_ema_cross(self.SQUID_INK_prices)

                # Trading signal
                current_position_signal = 1 if ema_short > ema_long else -1

                # Only act when crossover happens
                if current_position_signal != self.prev_position_signal:
                    self.take_best_orders(Product.SQUID_INK, mid_price, current_position_signal, orders, order_depth, position)

                # Try to clear wrong position
                if position * current_position_signal < 0:
                    self.clear_position_order(Product.SQUID_INK, mid_price, orders, order_depth, position)

                self.prev_position_signal = current_position_signal

        return orders

    def run(self, state: TradingState):
        result = {}
        conversions = 1

        if Product.SQUID_INK in state.order_depths:
            position = state.position.get(Product.SQUID_INK, 0)

            # Restore memory
            if state.traderData != "":
                trader_data = jsonpickle.decode(state.traderData)
                self.SQUID_INK_prices = trader_data.get("SQUID_INK_prices", [])
                self.prev_position_signal = trader_data.get("prev_position_signal", 0)

            orders = self.SQUID_INK_orders(state.order_depths[Product.SQUID_INK], position)
            result[Product.SQUID_INK] = orders

        # Save memory
        traderData = jsonpickle.encode({
            "SQUID_INK_prices": self.SQUID_INK_prices,
            "prev_position_signal": self.prev_position_signal
        })

        return result, conversions, traderData