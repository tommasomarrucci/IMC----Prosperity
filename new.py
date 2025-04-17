from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import jsonpickle
import math


class Trader:
    def __init__(self):
        self.last_prices = {}
        self.underlying_symbol = "VOLCANIC_ROCK"
        self.option_symbols = [
            ("VOLCANIC_ROCK_VOUCHER_9500", 9500),
            ("VOLCANIC_ROCK_VOUCHER_9750", 9750),
            ("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            ("VOLCANIC_ROCK_VOUCHER_10250", 10250),
            ("VOLCANIC_ROCK_VOUCHER_10500", 10500)
        ]
        self.risk_free_rate = 0.0
        self.implied_volatility = 0.18

    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        result = {}
        conversions = 0
        traderData = {}

        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)

        # Base product trading
        for product in ["RAINFOREST_RESIN", "SQUID_INK", "KELP"]:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            position = self.position(state, product)
            fair_value = self.get_fair_value(product, order_depth, traderData)

            buy_volume, sell_volume = 0, 0
            orders: List[Order] = []

            orders += self.take_orders(product, order_depth, fair_value, position, traderData, buy_volume, sell_volume)
            orders += self.clear_orders(product, order_depth, fair_value, position, buy_volume, sell_volume)
            orders += self.make_orders(product, order_depth, fair_value, position, buy_volume, sell_volume)

            result[product] = orders

        # Options strategy on VOLCANIC_ROCK
        underlying_price = self.get_mid_price(state.order_depths.get(self.underlying_symbol))
        if underlying_price is not None:
            self.last_prices[self.underlying_symbol] = underlying_price
        else:
            underlying_price = self.last_prices.get(self.underlying_symbol, 10000)

        total_delta = 0
        T = max(1, 7 - state.timestamp) / 365

        for symbol, strike in self.option_symbols:
            order_depth = state.order_depths.get(symbol)
            if order_depth is None:
                continue

            option_price = self.get_mid_price(order_depth)
            if option_price is None:
                continue

            self.last_prices[symbol] = option_price
            delta, gamma, theo_price = self.black_scholes_all(underlying_price, strike, T, self.implied_volatility)
            total_delta += delta * self.position(state, symbol)

            orders = []
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < theo_price:
                    orders.append(Order(symbol, ask_price, min(10, abs(ask_vol))))
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > theo_price:
                    orders.append(Order(symbol, bid_price, -min(10, abs(bid_vol))))
            result[symbol] = orders

        hedge_orders = []
        underlying_depth = state.order_depths.get(self.underlying_symbol)
        if underlying_depth:
            for bid_price, bid_vol in sorted(underlying_depth.buy_orders.items(), reverse=True):
                if total_delta < -9:
                    volume = min(10, abs(bid_vol), int(-total_delta))
                    hedge_orders.append(Order(self.underlying_symbol, bid_price, -volume))
                    total_delta += volume
            for ask_price, ask_vol in sorted(underlying_depth.sell_orders.items()):
                if total_delta > 9:
                    volume = min(10, abs(ask_vol), int(total_delta))
                    hedge_orders.append(Order(self.underlying_symbol, ask_price, volume))
                    total_delta -= volume
            result[self.underlying_symbol] = hedge_orders

        return result, conversions, jsonpickle.encode(traderData)

    def position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def get_fair_value(self, product: str, order_depth: OrderDepth, traderData) -> float:
        if product == "RAINFOREST_RESIN":
            return 10000
        return self.adaptive_fair_value(product, order_depth, traderData)

    def adaptive_fair_value(self, product: str, order_depth: OrderDepth, traderData) -> float:
        config = {
            "SQUID_INK": {"adverse_volume": 15, "reversion_beta": -0.25},
            "KELP": {"adverse_volume": 20, "reversion_beta": -0.1}
        }

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return traderData.get(f"{product}_last_price", 10000)

        best_ask = min(order_depth.sell_orders)
        best_bid = max(order_depth.buy_orders)

        filtered_ask = [p for p, v in order_depth.sell_orders.items() if abs(v) >= config[product]["adverse_volume"]]
        filtered_bid = [p for p, v in order_depth.buy_orders.items() if abs(v) >= config[product]["adverse_volume"]]

        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None

        if mm_ask is None or mm_bid is None:
            mmmid = traderData.get(f"{product}_last_price", (best_ask + best_bid) / 2)
        else:
            mmmid = (mm_ask + mm_bid) / 2

        last = traderData.get(f"{product}_last_price")
        if last:
            returns = (mmmid - last) / last
            pred_ret = returns * config[product]["reversion_beta"]
            fair = mmmid + (mmmid * pred_ret)
        else:
            fair = mmmid

        traderData[f"{product}_last_price"] = mmmid
        return fair

    def get_mid_price(self, order_depth):
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2

    def black_scholes_all(self, S, K, T, sigma):
        if T <= 0:
            return 0, 0, max(0, S - K)
        d1 = (math.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        delta = self.norm_cdf(d1)
        gamma = self.norm_pdf(d1) / (S * sigma * math.sqrt(T))
        price = S * self.norm_cdf(d1) - K * math.exp(-self.risk_free_rate * T) * self.norm_cdf(d2)
        return delta, gamma, price

    def norm_pdf(self, x):
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

    def norm_cdf(self, x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def take_orders(self, product, order_depth, fair_value, position, traderData, buy_volume, sell_volume):
        orders = []
        limit = 50
        config = {
            "RAINFOREST_RESIN": {"take_width": 0.8},
            "SQUID_INK": {"take_width": 1, "prevent_adverse": True, "adverse_volume": 15},
            "KELP": {"take_width": 1.5, "prevent_adverse": True, "adverse_volume": 20}
        }

        prevent_adverse = config[product].get("prevent_adverse", False)
        adverse_volume = config[product].get("adverse_volume", 0)

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            ask_vol = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or ask_vol <= adverse_volume:
                if best_ask <= fair_value - config[product]["take_width"]:
                    qty = min(ask_vol, limit - position)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            bid_vol = order_depth.buy_orders[best_bid]
            if not prevent_adverse or bid_vol <= adverse_volume:
                if best_bid >= fair_value + config[product]["take_width"]:
                    qty = min(bid_vol, limit + position)
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))

        return orders

    def clear_orders(self, product, order_depth, fair_value, position, buy_volume, sell_volume):
        orders = []
        limit = 50
        clear_width = 0
        pos_after_take = position + buy_volume - sell_volume
        bid_price = round(fair_value - clear_width)
        ask_price = round(fair_value + clear_width)
        buy_qty = limit - (position + buy_volume)
        sell_qty = limit + (position - sell_volume)

        if pos_after_take > 0:
            clear_qty = sum(v for p, v in order_depth.buy_orders.items() if p >= ask_price)
            sent = min(sell_qty, min(clear_qty, pos_after_take))
            if sent > 0:
                orders.append(Order(product, ask_price, -sent))

        if pos_after_take < 0:
            clear_qty = sum(-v for p, v in order_depth.sell_orders.items() if p <= bid_price)
            sent = min(buy_qty, min(clear_qty, -pos_after_take))
            if sent > 0:
                orders.append(Order(product, bid_price, sent))

        return orders

    def make_orders(self, product, order_depth, fair_value, position, buy_volume, sell_volume):
        orders = []
        limit = 50
        config = {
            "RAINFOREST_RESIN": {"disregard_edge": 1, "join_edge": 1.5, "default_edge": 3, "soft_position_limit": 10},
            "SQUID_INK": {"disregard_edge": 1, "join_edge": 0, "default_edge": 1},
            "KELP": {"disregard_edge": 1.5, "join_edge": 0.5, "default_edge": 1.5}
        }

        disregard_edge = config[product]["disregard_edge"]
        join_edge = config[product]["join_edge"]
        default_edge = config[product]["default_edge"]
        soft_limit = config[product].get("soft_position_limit", 0)

        asks_above = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge]
        bids_below = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge]

        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None

        ask = best_ask_above - 1 if best_ask_above and abs(best_ask_above - fair_value) > join_edge else best_ask_above or round(fair_value + default_edge)
        bid = best_bid_below + 1 if best_bid_below and abs(fair_value - best_bid_below) > join_edge else best_bid_below or round(fair_value - default_edge)

        if position > soft_limit:
            ask -= 1
        elif position < -soft_limit:
            bid += 1

        buy_qty = limit - (position + buy_volume)
        if buy_qty > 0:
            orders.append(Order(product, bid, buy_qty))

        sell_qty = limit + (position - sell_volume)
        if sell_qty > 0:
            orders.append(Order(product, ask, -sell_qty))

        return orders
