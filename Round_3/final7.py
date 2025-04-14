from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math


class Product:
    # Products that are directly traded with the “make” strategy:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"   # was AMETHYSTS
    SQUID_INK = "SQUID_INK"                 # was STARFRUIT
    KELP = "KELP"
    # Basket and synthetic products:
    PICNIC_BASKET1 = "PICNIC_BASKET1"         # was GIFT_BASKET
    JAMS = "JAMS"                           # was CHOCOLATE
    CROISSANTS = "CROISSANTS"               # was STRAWBERRIES
    DJEMBES = "DJEMBES"                     # was ROSES
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


# === PARAMETERS ===
# For the three products traded as in your first algo, we use the first algo’s parameters.
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 0.8,
        "clear_width": 0,
        "disregard_edge": 1,    # used in making orders
        "join_edge": 1.5,
        "default_edge": 3,
        "soft_position_limit": 10,
        "volume_limit": 0,      # if applicable for clearing (not used in making orders)
    },
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
    Product.KELP: {
        "take_width": 1.5,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 20,
        "reversion_beta": -0.1,
        "disregard_edge": 1.5,
        "join_edge": 0.5,
        "default_edge": 1.5,
    },
    # The basket/spread parameters come from your second algo:
    Product.SPREAD: {
        "spread_mean": 379.50439988484237,
        "starting_its": 30000,
        "spread_std_window": 25,
        "zscore_threshold": 11,
        "target_position": 60,
    }
}


# Basket weights (used when constructing a synthetic basket from the components)
BASKET_WEIGHTS = {
    Product.JAMS: 3,
    Product.CROISSANTS: 6,
    Product.DJEMBES: 1,
}


# === TRADER CLASS ===
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        # Set position limits for all traded products.
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
            Product.KELP: 50,
            Product.PICNIC_BASKET1: 60,
            Product.JAMS: 250,
            Product.CROISSANTS: 350,
            Product.DJEMBES: 60
        }

    # --- Basic order book operations ---
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        # Process sell orders (buying opportunity)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if (not prevent_adverse or abs(best_ask_amount) <= adverse_volume) and (best_ask <= fair_value - take_width):
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
        # Process buy orders (selling opportunity)
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (not prevent_adverse or abs(best_bid_amount) <= adverse_volume) and (best_bid >= fair_value + take_width):
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume and best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume and best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    # --- Fair value functions for SQUID_INK and KELP ---
    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys()
                            if abs(order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys()
                            if abs(order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if traderObject.get("SQUID_INK_last_price", None) is None else traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("SQUID_INK_last_price", None) is not None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys()
                            if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys()
                            if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if traderObject.get("KELP_last_price", None) is None else traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("KELP_last_price", None) is not None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None

    # --- Order wrappers ---
    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float,
                    position: int, prevent_adverse: bool = False, adverse_volume: int = 0) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product, fair_value, take_width, orders, order_depth, position,
                buy_order_volume, sell_order_volume, adverse_volume
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product, fair_value, take_width, orders, order_depth, position,
                buy_order_volume, sell_order_volume
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int,
                     position: int, buy_order_volume: int, sell_order_volume: int) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    # --- “Make” orders functions for the three products (from the first algorithm) ---
    def make_rainforest_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        # Determine the best levels for joining/pennying
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3
        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3
        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_squid_ink_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [price for price in order_depth.sell_orders.keys() if price >= round(fair_value + min_edge)]
        bbf = [price for price in order_depth.buy_orders.keys() if price <= round(fair_value - min_edge)]
        baaf = min(aaf) if aaf else round(fair_value + min_edge)
        bbbf = max(bbf) if bbf else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.SQUID_INK, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_KELP_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        bid = round(fair_value - default_edge)
        ask = round(fair_value + default_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    # --- Basket operations (as in the second algorithm) ---
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS[Product.DJEMBES]
        synthetic_order_price = OrderDepth()
        jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
        jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float('inf')
        croissants_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else 0
        croissants_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else float('inf')
        djembes_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys()) if order_depths[Product.DJEMBES].buy_orders else 0
        djembes_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys()) if order_depths[Product.DJEMBES].sell_orders else float('inf')
        implied_bid = (jams_best_bid * JAMS_PER_BASKET +
                       croissants_best_bid * CROISSANTS_PER_BASKET +
                       djembes_best_bid * DJEMBES_PER_BASKET)
        implied_ask = (jams_best_ask * JAMS_PER_BASKET +
                       croissants_best_ask * CROISSANTS_PER_BASKET +
                       djembes_best_ask * DJEMBES_PER_BASKET)
        if implied_bid > 0:
            jams_bid_volume = order_depths[Product.JAMS].buy_orders[jams_best_bid] // JAMS_PER_BASKET
            croissants_bid_volume = order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid] // CROISSANTS_PER_BASKET
            djembes_bid_volume = order_depths[Product.DJEMBES].buy_orders[djembes_best_bid] // DJEMBES_PER_BASKET
            implied_bid_volume = min(jams_bid_volume, croissants_bid_volume, djembes_bid_volume)
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask < float('inf'):
            jams_ask_volume = -order_depths[Product.JAMS].sell_orders[jams_best_ask] // JAMS_PER_BASKET
            croissants_ask_volume = -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask] // CROISSANTS_PER_BASKET
            djembes_ask_volume = -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask] // DJEMBES_PER_BASKET
            implied_ask_volume = min(jams_ask_volume, croissants_ask_volume, djembes_ask_volume)
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_order_price

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {
            Product.JAMS: [],
            Product.CROISSANTS: [],
            Product.DJEMBES: [],
        }
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float("inf")
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                croissants_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                continue
            jams_order = Order(Product.JAMS, jams_price, quantity * BASKET_WEIGHTS[Product.JAMS])
            croissants_order = Order(Product.CROISSANTS, croissants_price, quantity * BASKET_WEIGHTS[Product.CROISSANTS])
            djembes_order = Order(Product.DJEMBES, djembes_price, quantity * BASKET_WEIGHTS[Product.DJEMBES])
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.DJEMBES].append(djembes_order)
        return component_orders

    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_history: List[float]):
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_history.append(spread)
        if len(spread_history) < self.params[Product.SPREAD]["spread_std_window"]:
            return None
        spread_mean = (np.sum(spread_history) +
                       (self.params[Product.SPREAD]["spread_mean"] * self.params[Product.SPREAD]["starting_its"])) / (
                          self.params[Product.SPREAD]["starting_its"] + len(spread_history))
        spread_std = np.std(spread_history[-self.params[Product.SPREAD]["spread_std_window"]:])
        zscore = (spread - spread_mean) / spread_std
        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position == -self.params[Product.SPREAD]["target_position"]:
                return None
            target_quantity = abs(-self.params[Product.SPREAD]["target_position"] - basket_position)
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
        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position == self.params[Product.SPREAD]["target_position"]:
                return None
            target_quantity = abs(self.params[Product.SPREAD]["target_position"] - basket_position)
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
        return None

    # --- Run method (merging the two algos) ---
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        conversions = 0

        # Trade RAINFOREST_RESIN, SQUID_INK, KELP using first algo behavior:

        # RAINFOREST_RESIN trading
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position[Product.RAINFOREST_RESIN] if Product.RAINFOREST_RESIN in state.position else 0
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                resin_position
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume, sell_order_volume
            )
            resin_make_orders, _, _ = self.make_rainforest_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume, sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["volume_limit"]
            )
            result[Product.RAINFOREST_RESIN] = resin_take_orders + resin_clear_orders + resin_make_orders

        # SQUID_INK trading
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_position = state.position[Product.SQUID_INK] if Product.SQUID_INK in state.position else 0
            squid_fair_value = self.SQUID_INK_fair_value(state.order_depths[Product.SQUID_INK], traderObject)
            squid_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_fair_value,
                self.params[Product.SQUID_INK]["take_width"],
                squid_position,
                self.params[Product.SQUID_INK]["prevent_adverse"],
                self.params[Product.SQUID_INK]["adverse_volume"]
            )
            squid_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_fair_value,
                self.params[Product.SQUID_INK]["clear_width"],
                squid_position,
                buy_order_volume, sell_order_volume
            )
            squid_make_orders, _, _ = self.make_squid_ink_orders(
                state.order_depths[Product.SQUID_INK],
                squid_fair_value,
                self.params[Product.SQUID_INK]["default_edge"],
                squid_position,
                buy_order_volume, sell_order_volume
            )
            result[Product.SQUID_INK] = squid_take_orders + squid_clear_orders + squid_make_orders

        # KELP trading
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = state.position[Product.KELP] if Product.KELP in state.position else 0
            kelp_fair_value = self.KELP_fair_value(state.order_depths[Product.KELP], traderObject)
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["take_width"],
                kelp_position,
                self.params[Product.KELP]["prevent_adverse"],
                self.params[Product.KELP]["adverse_volume"]
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["clear_width"],
                kelp_position,
                buy_order_volume, sell_order_volume
            )
            kelp_make_orders, _, _ = self.make_KELP_orders(
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume, sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"]
            )
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        # --- Basket arbitrage trading (for PICNIC_BASKET1, with conversion to JAMS, CROISSANTS, DJEMBES) ---
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {"spread_history": []}
        basket_position = state.position[Product.PICNIC_BASKET1] if Product.PICNIC_BASKET1 in state.position else 0
        spread_orders = self.spread_orders(state.order_depths, Product.PICNIC_BASKET1, basket_position, traderObject[Product.SPREAD]["spread_history"])
        if spread_orders is not None:
            result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]

        traderData = jsonpickle.encode(traderObject)
        conversions = 1
        return result, conversions, traderData