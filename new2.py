from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, ConversionObservation
import jsonpickle
import math
import numpy as np

# -------------------------------------------------------------------
# Product definitions and constants
# -------------------------------------------------------------------
class Product:
    # PICNIC basket and its components
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"  # marker for synthetic orders
    SPREAD = "SPREAD"
    # Base products for normal trading
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"
    # Underlying and options
    VOLCANIC_ROCK = "VOLCANIC_ROCK"


# Spread parameters for PICNIC_BASKET1 conversion
PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 379.50439988484239,
        "default_spread_std": 76.07966,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 58,
    },
}
# Basket weights for PICNIC_BASKET1 conversion components
BASKET_WEIGHTS = {
    Product.JAMS: 3,
    Product.CROISSANTS: 6,
    Product.DJEMBES: 1,
}

# -------------------------------------------------------------------
# Trader class with merged functionality for base product trading,
# PICNIC_BASKET1 conversion, and VOLCANIC_ROCK option strategy.
# -------------------------------------------------------------------
class Trader:
    def __init__(self):
        # For the options strategy on VOLCANIC_ROCK.
        self.last_prices = {}
        self.underlying_symbol = Product.VOLCANIC_ROCK
        self.option_symbols = [
            ("VOLCANIC_ROCK_VOUCHER_9500", 9500),
            ("VOLCANIC_ROCK_VOUCHER_9750", 9750),
            ("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            ("VOLCANIC_ROCK_VOUCHER_10250", 10250),
            ("VOLCANIC_ROCK_VOUCHER_10500", 10500)
        ]
        self.risk_free_rate = 0.0
        self.implied_volatility = 0.15

        # For PICNIC_BASKET1 conversion
        self.params = PARAMS
        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.JAMS: 350,
            Product.CROISSANTS: 250,
            Product.DJEMBES: 60,
        }

    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = {}

        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)

        # -------------------------------------------------------------------
        # Base Product Trading for RAINFOREST_RESIN, SQUID_INK, KELP
        # -------------------------------------------------------------------
        for product in [Product.RAINFOREST_RESIN, Product.SQUID_INK, Product.KELP]:
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

        # -------------------------------------------------------------------
        # PICNIC_BASKET1 Conversion Trading
        # -------------------------------------------------------------------
        # Check if we have PICNIC_BASKET1 order depth data.
        if Product.PICNIC_BASKET1 in state.order_depths:
            order_depth = state.order_depths[Product.PICNIC_BASKET1]
            basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
            spread_orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET1, basket_position, traderData.get(Product.SPREAD, {
                "spread_history": [], "prev_zscore": 0, "clear_flag": False, "curr_avg": 0
            }))
            if spread_orders is not None:
                result[Product.PICNIC_BASKET1] = spread_orders.get(Product.PICNIC_BASKET1, [])
                result[Product.JAMS] = spread_orders.get(Product.JAMS, [])
                result[Product.CROISSANTS] = spread_orders.get(Product.CROISSANTS, [])
                result[Product.DJEMBES] = spread_orders.get(Product.DJEMBES, [])

        # -------------------------------------------------------------------
        # Options Strategy on VOLCANIC_ROCK
        # -------------------------------------------------------------------
        underlying_depth = state.order_depths.get(self.underlying_symbol)
        underlying_price = self.get_mid_price(underlying_depth)
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
            # Sell orders: buy if option price is underpriced (ask below theo_price)
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < theo_price:
                    orders.append(Order(symbol, ask_price, min(10, abs(ask_vol))))
            # Buy orders: sell if option price is overpriced (bid above theo_price)
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > theo_price:
                    orders.append(Order(symbol, bid_price, -min(10, abs(bid_vol))))
            result[symbol] = orders

        hedge_orders = []
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

    # -------------------------------------------------------------------
    # Utility methods for base product trading (RAINFOREST_RESIN, SQUID_INK, KELP)
    # -------------------------------------------------------------------
    def position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def get_fair_value(self, product: str, order_depth: OrderDepth, traderData) -> float:
        if product == Product.RAINFOREST_RESIN:
            return 10000
        return self.adaptive_fair_value(product, order_depth, traderData)

    def adaptive_fair_value(self, product: str, order_depth: OrderDepth, traderData) -> float:
        config = {
            Product.SQUID_INK: {"adverse_volume": 15, "reversion_beta": -0.25},
            Product.KELP: {"adverse_volume": 20, "reversion_beta": -0.1}
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

    def get_mid_price(self, order_depth: OrderDepth):
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

    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, position: int, traderData, buy_volume: int, sell_volume: int) -> List[Order]:
        orders = []
        limit = 50
        config = {
            Product.RAINFOREST_RESIN: {"take_width": 0.8},
            Product.SQUID_INK: {"take_width": 1, "prevent_adverse": True, "adverse_volume": 15},
            Product.KELP: {"take_width": 1.5, "prevent_adverse": True, "adverse_volume": 20}
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

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int) -> List[Order]:
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

    def make_orders(self, product: str, order_depth: OrderDepth, fair_value: float, position: int, buy_volume: int, sell_volume: int) -> List[Order]:
        orders = []
        limit = 50
        config = {
            Product.RAINFOREST_RESIN: {"disregard_edge": 1, "join_edge": 1.5, "default_edge": 3, "soft_position_limit": 10},
            Product.SQUID_INK: {"disregard_edge": 1, "join_edge": 0, "default_edge": 1},
            Product.KELP: {"disregard_edge": 1.5, "join_edge": 0.5, "default_edge": 1.5}
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

    # -------------------------------------------------------------------
    # PICNIC_BASKET1 conversion helper methods
    # -------------------------------------------------------------------
    def get_swmid(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        synthetic_order = OrderDepth()
        JAMS_best_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
        JAMS_best_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float("inf")
        CROISSANTS_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else 0
        CROISSANTS_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else float("inf")
        DJEMBES_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys()) if order_depths[Product.DJEMBES].buy_orders else 0
        DJEMBES_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys()) if order_depths[Product.DJEMBES].sell_orders else float("inf")
        implied_bid = (JAMS_best_bid * BASKET_WEIGHTS[Product.JAMS] +
                       CROISSANTS_best_bid * BASKET_WEIGHTS[Product.CROISSANTS] +
                       DJEMBES_best_bid * BASKET_WEIGHTS[Product.DJEMBES])
        implied_ask = (JAMS_best_ask * BASKET_WEIGHTS[Product.JAMS] +
                       CROISSANTS_best_ask * BASKET_WEIGHTS[Product.CROISSANTS] +
                       DJEMBES_best_ask * BASKET_WEIGHTS[Product.DJEMBES])
        if implied_bid > 0:
            JAMS_bid_vol = order_depths[Product.JAMS].buy_orders[JAMS_best_bid] // BASKET_WEIGHTS[Product.JAMS]
            CROISSANTS_bid_vol = order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid] // BASKET_WEIGHTS[Product.CROISSANTS]
            DJEMBES_bid_vol = order_depths[Product.DJEMBES].buy_orders[DJEMBES_best_bid] // BASKET_WEIGHTS[Product.DJEMBES]
            implied_bid_volume = min(JAMS_bid_vol, CROISSANTS_bid_vol, DJEMBES_bid_vol)
            synthetic_order.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask < float("inf"):
            JAMS_ask_vol = -order_depths[Product.JAMS].sell_orders[JAMS_best_ask] // BASKET_WEIGHTS[Product.JAMS]
            CROISSANTS_ask_vol = -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask] // BASKET_WEIGHTS[Product.CROISSANTS]
            DJEMBES_ask_vol = -order_depths[Product.DJEMBES].sell_orders[DJEMBES_best_ask] // BASKET_WEIGHTS[Product.DJEMBES]
            implied_ask_volume = min(JAMS_ask_vol, CROISSANTS_ask_vol, DJEMBES_ask_vol)
            synthetic_order.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_order

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {Product.JAMS: [], Product.CROISSANTS: [], Product.DJEMBES: []}
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float("inf")
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                JAMS_price = min(order_depths[Product.JAMS].sell_orders.keys())
                CROISSANTS_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                DJEMBES_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                JAMS_price = max(order_depths[Product.JAMS].buy_orders.keys())
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                DJEMBES_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                continue
            component_orders[Product.JAMS].append(Order(Product.JAMS, JAMS_price, quantity * BASKET_WEIGHTS[Product.JAMS]))
            component_orders[Product.CROISSANTS].append(Order(Product.CROISSANTS, CROISSANTS_price, quantity * BASKET_WEIGHTS[Product.CROISSANTS]))
            component_orders[Product.DJEMBES].append(Order(Product.DJEMBES, DJEMBES_price, quantity * BASKET_WEIGHTS[Product.DJEMBES]))
        return component_orders

    def execute_spread_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        if target_position == basket_position:
            return None
        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)]
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_data: Dict[str, Any]) -> Dict[str, List[Order]]:
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        if len(spread_data["spread_history"]) < self.params[Product.SPREAD]["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)
        spread_std = np.std(spread_data["spread_history"])
        zscore = (spread - self.params[Product.SPREAD]["default_spread_mean"]) / spread_std
        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(-self.params[Product.SPREAD]["target_position"], basket_position, order_depths)
        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(self.params[Product.SPREAD]["target_position"], basket_position, order_depths)
        spread_data["prev_zscore"] = zscore
        return None