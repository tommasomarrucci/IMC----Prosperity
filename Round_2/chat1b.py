from Round_3.datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict
import jsonpickle
import numpy as np
import math

# Updated product names.
class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# Parameters for individual products and baskets.
# For individual products we use fixed fair values;
# for baskets we use spread parameters that will now be updated dynamically.
PARAMS = {
    Product.CROISSANTS: {
        "fair_value": 100,
        "take_width": 1,  # This could be later adapted based on volatility.
        "clear_width": 0.5,
        "volume_limit": 250,
    },
    Product.JAMS: {
        "fair_value": 80,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 350,
    },
    Product.DJEMBES: {
        "fair_value": 150,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 60,
    },
    Product.PICNIC_BASKET1: {
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 60,
        # Initial spread parameters; these will update dynamically.
        "starting_spread_mean": 379.5,
        "spread_std_window": 25,
        "zscore_threshold": 1.5,  # More sensitive threshold after dynamic adjustment.
        "target_position": 20,
    },
    Product.PICNIC_BASKET2: {
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 100,
        "starting_spread_mean": 379.5,
        "spread_std_window": 25,
        "zscore_threshold": 1.5,
        "target_position": 30,
    },
}

POSITION_LIMITS = {
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
}

# Basket composition definitions.
BASKET_COMPOSITION = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1,
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2,
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = POSITION_LIMITS

    ####################################################
    # Methods for individual product trading
    ####################################################
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
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
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
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
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
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
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
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
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
        position_limit = self.LIMIT[product]
        buy_qty = position_limit - (position + buy_order_volume)
        if buy_qty > 0:
            orders.append(Order(product, round(bid), buy_qty))
        sell_qty = position_limit + (position - sell_order_volume)
        if sell_qty > 0:
            orders.append(Order(product, round(ask), -sell_qty))
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
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        position_limit = self.LIMIT[product]
        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(position_limit + position, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(position_limit - position, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    ####################################################
    # Basket Synthesis and Conversion for Picnic Baskets
    ####################################################
    def get_synthetic_picnic_basket_order_depth(self, basket_type: str, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        synthetic_depth = OrderDepth()
        composition = BASKET_COMPOSITION[basket_type]
        implied_bid = 0
        implied_ask = 0
        bid_volumes = []
        ask_volumes = []
        for comp, qty in composition.items():
            if comp not in order_depths:
                continue
            comp_depth = order_depths[comp]
            if comp_depth.buy_orders:
                comp_best_bid = max(comp_depth.buy_orders.keys())
                implied_bid += comp_best_bid * qty
                bid_volumes.append(comp_depth.buy_orders[comp_best_bid] // qty)
            else:
                bid_volumes.append(0)
            if comp_depth.sell_orders:
                comp_best_ask = min(comp_depth.sell_orders.keys())
                implied_ask += comp_best_ask * qty
                ask_volumes.append((-comp_depth.sell_orders[comp_best_ask]) // qty)
            else:
                ask_volumes.append(0)
        implied_bid_volume = min(bid_volumes) if bid_volumes else 0
        implied_ask_volume = min(ask_volumes) if ask_volumes else 0
        if implied_bid > 0:
            synthetic_depth.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask > 0 and implied_ask != float('inf'):
            synthetic_depth.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_depth

    def convert_picnic_basket_orders(self, basket_type: str, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders: Dict[str, List[Order]] = {comp: [] for comp in BASKET_COMPOSITION[basket_type]}
        synthetic_depth = self.get_synthetic_picnic_basket_order_depth(basket_type, order_depths)
        best_bid = max(synthetic_depth.buy_orders.keys()) if synthetic_depth.buy_orders else 0
        best_ask = min(synthetic_depth.sell_orders.keys()) if synthetic_depth.sell_orders else float("inf")
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                for comp, comp_qty in BASKET_COMPOSITION[basket_type].items():
                    if comp in order_depths and order_depths[comp].sell_orders:
                        comp_price = min(order_depths[comp].sell_orders.keys())
                        comp_order = Order(comp, comp_price, quantity * comp_qty)
                        component_orders[comp].append(comp_order)
            elif quantity < 0 and price <= best_bid:
                for comp, comp_qty in BASKET_COMPOSITION[basket_type].items():
                    if comp in order_depths and order_depths[comp].buy_orders:
                        comp_price = max(order_depths[comp].buy_orders.keys())
                        comp_order = Order(comp, comp_price, quantity * comp_qty)
                        component_orders[comp].append(comp_order)
        return component_orders

    ####################################################
    # Dynamic Spread Statistics for Basket Arbitrage
    ####################################################
    def update_spread_stats(self, basket_type: str, spread_history: List[float]) -> (float, float):
        # Calculate a rolling mean and standard deviation over the spread history window.
        window = self.params[basket_type]["spread_std_window"]
        if len(spread_history) < window:
            # Not enough data; use starting spread mean and a default std deviation.
            return self.params[basket_type]["starting_spread_mean"], 1.0
        recent_spreads = spread_history[-window:]
        mean_spread = np.mean(recent_spreads)
        std_spread = np.std(recent_spreads)
        return mean_spread, std_spread

    # Generate spread orders for picnic baskets using dynamic z-score.
    def spread_orders(self, basket_type: str, order_depths: Dict[str, OrderDepth], basket_position: int, spread_history: List[float]):
        market_depth = order_depths[basket_type]
        synthetic_depth = self.get_synthetic_picnic_basket_order_depth(basket_type, order_depths)
        if market_depth.buy_orders and market_depth.sell_orders:
            market_mid = (max(market_depth.buy_orders.keys()) + min(market_depth.sell_orders.keys())) / 2
        else:
            market_mid = None
        if synthetic_depth.buy_orders and synthetic_depth.sell_orders:
            synthetic_mid = (max(synthetic_depth.buy_orders.keys()) + min(synthetic_depth.sell_orders.keys())) / 2
        else:
            synthetic_mid = None
        if market_mid is None or synthetic_mid is None:
            return None

        spread = market_mid - synthetic_mid
        spread_history.append(spread)
        mean_spread, std_spread = self.update_spread_stats(basket_type, spread_history)
        # Compute the z-score for the current spread.
        if std_spread > 0:
            zscore = (spread - mean_spread) / std_spread
        else:
            zscore = 0

        result = {}
        threshold = self.params[basket_type]["zscore_threshold"]

        # If the basket is overpriced (zscore high): sell basket and buy components.
        if zscore >= threshold:
            target_quantity = abs(self.params[basket_type]["target_position"] - basket_position)
            if target_quantity <= 0:
                return None
            basket_bid_price = max(market_depth.buy_orders.keys())
            basket_bid_volume = abs(market_depth.buy_orders[basket_bid_price])
            execute_volume = min(basket_bid_volume, target_quantity)
            basket_orders = [Order(basket_type, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order("SYNTHETIC_" + basket_type, min(synthetic_depth.sell_orders.keys()), execute_volume)]
            component_orders = self.convert_picnic_basket_orders(basket_type, synthetic_orders, order_depths)
            component_orders[basket_type] = basket_orders
            return component_orders

        # If the basket is underpriced (zscore low): buy basket and sell components.
        if zscore <= -threshold:
            target_quantity = abs(self.params[basket_type]["target_position"] - basket_position)
            if target_quantity <= 0:
                return None
            basket_ask_price = min(market_depth.sell_orders.keys())
            basket_ask_volume = abs(market_depth.sell_orders[basket_ask_price])
            execute_volume = min(basket_ask_volume, target_quantity)
            basket_orders = [Order(basket_type, basket_ask_price, execute_volume)]
            synthetic_orders = [Order("SYNTHETIC_" + basket_type, max(synthetic_depth.buy_orders.keys()), -execute_volume)]
            component_orders = self.convert_picnic_basket_orders(basket_type, synthetic_orders, order_depths)
            component_orders[basket_type] = basket_orders
            return component_orders

        return None

    ####################################################
    # Main run function
    ####################################################
    def run(self, state: TradingState):
        traderData = {}
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
        result = {}
        conversions = 0

        # Trade individual products: CROISSANTS, JAMS, DJEMBES.
        for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
            if product in state.order_depths:
                position = state.position[product] if product in state.position else 0
                fair_value = self.params[product]["fair_value"]
                take_width = self.params[product]["take_width"]
                clear_width = self.params[product]["clear_width"]
                orders_take, buy_vol, sell_vol = self.take_orders(product, state.order_depths[product], fair_value, take_width, position)
                orders_clear, buy_vol, sell_vol = self.clear_orders(product, state.order_depths[product], fair_value, clear_width, position, buy_vol, sell_vol)
                orders_mm = []
                self.market_make(product, orders_mm, fair_value - 1, fair_value + 1, position, buy_vol, sell_vol)
                result[product] = orders_take + orders_clear + orders_mm

        # Trade picnic baskets (PICNIC_BASKET1 and PICNIC_BASKET2) using spread arbitrage.
        for basket in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            if basket in state.order_depths:
                basket_position = state.position[basket] if basket in state.position else 0
                spread_history = traderData.get(basket, {}).get("spread_history", [])
                arb_orders = self.spread_orders(basket, state.order_depths, basket_position, spread_history)
                if arb_orders:
                    for prod, orders in arb_orders.items():
                        if prod in result:
                            result[prod].extend(orders)
                        else:
                            result[prod] = orders
                if basket not in traderData:
                    traderData[basket] = {}
                traderData[basket]["spread_history"] = spread_history

        traderData_encoded = jsonpickle.encode(traderData)
        return result, conversions, traderData_encoded