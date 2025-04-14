import math
import jsonpickle
import numpy as np
from Round_3.datamodel import OrderDepth, Order, TradingState

# Best parameters from grid search:
SQUID_INK_make_width = 0.3
SQUID_INK_take_width = 0.35
SQUID_INK_position_limit = 17
SQUID_INK_market_make_offset = 0.1  # (Note: used in market making if needed)
moving_average_window = 25
bias_shift = 0.17

class Product:
    SQUID_INK = "SQUID_INK"

class Trader:
    def __init__(self):
        # Initialize price tracking and limits
        self.SQUID_INK_prices = []
        self.SQUID_INK_vwap = []
        self.LIMIT = {Product.SQUID_INK: SQUID_INK_position_limit}
        self.mid_prices = []
        self.moving_average_window = moving_average_window
        self.bias_shift = bias_shift

    def take_best_orders(self, product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    def market_make(self, product, orders, bid, ask, position, buy_order_volume, sell_order_volume):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume):
        fair_bid = math.floor(fair_value)
        fair_ask = math.ceil(fair_value)

        if position + buy_order_volume - sell_order_volume > 0:
            if fair_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_ask], position)
                if clear_quantity > 0:
                    orders.append(Order(product, fair_ask, -clear_quantity))
                    sell_order_volume += clear_quantity

        if position + buy_order_volume - sell_order_volume < 0:
            if fair_bid in order_depth.sell_orders:
                clear_quantity = min(-order_depth.sell_orders[fair_bid], -position)
                if clear_quantity > 0:
                    orders.append(Order(product, fair_bid, clear_quantity))
                    buy_order_volume += clear_quantity

        return buy_order_volume, sell_order_volume

    def SQUID_INK_orders(self, order_depth, timespan, width, take_width, position):
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
            filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid

            mmmid_price = (mm_ask + mm_bid) / 2
            
            # Moving average part:
            self.mid_prices.append(mmmid_price)
            if len(self.mid_prices) > self.moving_average_window:
                self.mid_prices.pop(0)
            moving_avg = np.mean(self.mid_prices)
            
            if mmmid_price > moving_avg:
                fair_value = mmmid_price + self.bias_shift
            else:
                fair_value = mmmid_price - self.bias_shift

            buy_order_volume, sell_order_volume = self.take_best_orders(
                Product.SQUID_INK, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume
            )

            buy_order_volume, sell_order_volume = self.clear_position_order(
                Product.SQUID_INK, fair_value, orders, order_depth, position, buy_order_volume, sell_order_volume
            )

            aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
            bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2

            buy_order_volume, sell_order_volume = self.market_make(
                Product.SQUID_INK, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume
            )

        return orders

    def run(self, state: TradingState):
        result = {}
        if Product.SQUID_INK in state.order_depths:
            SQUID_INK_position = state.position.get(Product.SQUID_INK, 0)
            SQUID_INK_orders = self.SQUID_INK_orders(
                state.order_depths[Product.SQUID_INK],
                timespan=10,
                width=SQUID_INK_make_width,
                take_width=SQUID_INK_take_width,
                position=SQUID_INK_position
            )
            result[Product.SQUID_INK] = SQUID_INK_orders

        traderData = jsonpickle.encode({"SQUID_INK_prices": self.SQUID_INK_prices})
        conversions = 1
        return result, conversions, traderData