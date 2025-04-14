from Round_3.datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict
import jsonpickle
import numpy as np
import math

class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# Define basic parameters for the individual products and baskets.
PARAMS = {
    Product.CROISSANTS: {
         "fair_value": 100,
         "take_width": 1,
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
    # For the baskets we won’t use a fixed fair value – we compute their synthetic value.
    Product.PICNIC_BASKET1: {
         "take_width": 1,
         "clear_width": 0.5,
         "volume_limit": 60,
         "spread_threshold": 5,    # Arbitrary threshold for triggering arbitrage
         "spread_std_window": 25,
         "spread_mean": 0,
         "starting_its": 30000,
         "target_position": 20,
    },
    Product.PICNIC_BASKET2: {
         "take_width": 1,
         "clear_width": 0.5,
         "volume_limit": 100,
         "spread_threshold": 5,
         "spread_std_window": 25,
         "spread_mean": 0,
         "starting_its": 30000,
         "target_position": 30,
    },
}

# Define position limits as specified.
POSITION_LIMITS = {
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
}

# Basket composition for the two picnic baskets.
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
        # Initialize trader state for basket spread history.
        self.state = {
            Product.PICNIC_BASKET1: {"spread_history": []},
            Product.PICNIC_BASKET2: {"spread_history": []},
        }

    ####################################################
    # Methods for individual product trading
    ####################################################
    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, position: int) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        position_limit = self.LIMIT[product]

        # Check if the best ask is below the fair value – if so, buy.
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_volume, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

        # Similarly, if the best bid is above the fair value – sell.
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_volume, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: float, position: int, buy_order_volume: int, sell_order_volume: int) -> (List[Order], int, int):
        orders: List[Order] = []
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_bid = round(fair_value - clear_width)
        fair_ask = round(fair_value + clear_width)
        position_limit = self.LIMIT[product]

        # If the position is positive, clear by selling.
        if position_after_take > 0:
            clear_volume = sum(vol for price, vol in order_depth.buy_orders.items() if price >= fair_ask)
            clear_volume = min(clear_volume, position_after_take)
            execute_volume = min(position_limit + position, clear_volume)
            if execute_volume > 0:
                orders.append(Order(product, fair_ask, -abs(execute_volume)))
                sell_order_volume += abs(execute_volume)
        # If negative, clear by buying.
        if position_after_take < 0:
            clear_volume = sum(abs(vol) for price, vol in order_depth.sell_orders.items() if price <= fair_bid)
            clear_volume = min(clear_volume, abs(position_after_take))
            execute_volume = min(position_limit - position, clear_volume)
            if execute_volume > 0:
                orders.append(Order(product, fair_bid, abs(execute_volume)))
                buy_order_volume += abs(execute_volume)

        return orders, buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: float, ask: float, position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_limit = self.LIMIT[product]
        buy_qty = position_limit - (position + buy_order_volume)
        if buy_qty > 0:
            orders.append(Order(product, round(bid), buy_qty))
        sell_qty = position_limit + (position - sell_order_volume)
        if sell_qty > 0:
            orders.append(Order(product, round(ask), -sell_qty))
        return buy_order_volume, sell_order_volume

    ####################################################
    # Basket Synthesis and Conversion methods for Picnic Baskets
    ####################################################
    def get_synthetic_picnic_basket_order_depth(self, basket_type: str, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        """
        Compute a synthetic order depth for a picnic basket using its components.
        The synthetic bid is the weighted sum of the best bid prices and similarly for the ask.
        Volumes are limited by the scarcest component.
        """
        synthetic_order_depth = OrderDepth()
        composition = BASKET_COMPOSITION[basket_type]
        implied_bid = 0
        implied_ask = 0
        volumes_bid = []
        volumes_ask = []
        for comp, quantity in composition.items():
            comp_depth = order_depths[comp]
            if comp_depth.buy_orders:
                best_bid = max(comp_depth.buy_orders.keys())
                implied_bid += best_bid * quantity
                vol = comp_depth.buy_orders[best_bid] // quantity
                volumes_bid.append(vol)
            else:
                volumes_bid.append(0)
            if comp_depth.sell_orders:
                best_ask = min(comp_depth.sell_orders.keys())
                implied_ask += best_ask * quantity
                vol = (-comp_depth.sell_orders[best_ask]) // quantity
                volumes_ask.append(vol)
            else:
                volumes_ask.append(0)
        implied_bid_volume = min(volumes_bid) if volumes_bid else 0
        implied_ask_volume = min(volumes_ask) if volumes_ask else 0
        if implied_bid > 0:
            synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask > 0 and implied_ask != float('inf'):
            synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_order_depth

    def convert_picnic_basket_orders(self, basket_type: str, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        """
        Convert a synthetic basket order into component orders according to the basket composition.
        """
        component_orders: Dict[str, List[Order]] = {comp: [] for comp in BASKET_COMPOSITION[basket_type]}
        synthetic_basket_order_depth = self.get_synthetic_picnic_basket_order_depth(basket_type, order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float('inf')
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            # For a buy order (positive quantity) that is aggressive relative to the synthetic ask,
            # simulate buying the components at their best ask prices.
            if quantity > 0 and price >= best_ask:
                for comp, qty in BASKET_COMPOSITION[basket_type].items():
                    comp_price = min(order_depths[comp].sell_orders.keys()) if order_depths[comp].sell_orders else None
                    if comp_price is not None:
                        comp_order = Order(comp, comp_price, quantity * qty)
                        component_orders[comp].append(comp_order)
            # For a sell order (negative quantity), use the best bid for components.
            elif quantity < 0 and price <= best_bid:
                for comp, qty in BASKET_COMPOSITION[basket_type].items():
                    comp_price = max(order_depths[comp].buy_orders.keys()) if order_depths[comp].buy_orders else None
                    if comp_price is not None:
                        comp_order = Order(comp, comp_price, quantity * qty)
                        component_orders[comp].append(comp_order)
        return component_orders

    def picnic_basket_arbitrage_orders(self, basket_type: str, order_depths: Dict[str, OrderDepth], basket_position: int, spread_history: List[float]) -> Dict[str, List[Order]]:
        """
        Compare the market basket price (from the exchange) with the synthetic price computed
        from its components. If the spread exceeds a preset threshold, generate arbitrage orders.
        """
        market_order_depth = order_depths[basket_type]
        synthetic_order_depth = self.get_synthetic_picnic_basket_order_depth(basket_type, order_depths)
        if market_order_depth.buy_orders and market_order_depth.sell_orders:
            market_mid = (max(market_order_depth.buy_orders.keys()) + min(market_order_depth.sell_orders.keys())) / 2
        else:
            market_mid = None
        if synthetic_order_depth.buy_orders and synthetic_order_depth.sell_orders:
            synthetic_mid = (max(synthetic_order_depth.buy_orders.keys()) + min(synthetic_order_depth.sell_orders.keys())) / 2
        else:
            synthetic_mid = None
        if market_mid is None or synthetic_mid is None:
            return None

        spread = market_mid - synthetic_mid
        spread_history.append(spread)
        threshold = self.params[basket_type]["spread_threshold"]

        result = {}
        # If the spread is too high (basket is overpriced), sell the basket and buy its components.
        if spread >= threshold:
            target_quantity = abs(self.params[basket_type]["target_position"] - basket_position)
            if target_quantity <= 0:
                return None
            basket_bid_price = max(market_order_depth.buy_orders.keys())
            basket_bid_volume = abs(market_order_depth.buy_orders[basket_bid_price])
            execute_volume = min(basket_bid_volume, target_quantity)
            basket_orders = [Order(basket_type, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order("SYNTHETIC_" + basket_type, min(synthetic_order_depth.sell_orders.keys()), execute_volume)]
            component_orders = self.convert_picnic_basket_orders(basket_type, synthetic_orders, order_depths)
            component_orders[basket_type] = basket_orders
            result = component_orders
            return result

        # If the spread is too low (basket is underpriced), buy the basket and sell its components.
        if spread <= -threshold:
            target_quantity = abs(self.params[basket_type]["target_position"] - basket_position)
            if target_quantity <= 0:
                return None
            basket_ask_price = min(market_order_depth.sell_orders.keys())
            basket_ask_volume = abs(market_order_depth.sell_orders[basket_ask_price])
            execute_volume = min(basket_ask_volume, target_quantity)
            basket_orders = [Order(basket_type, basket_ask_price, execute_volume)]
            synthetic_orders = [Order("SYNTHETIC_" + basket_type, max(synthetic_order_depth.buy_orders.keys()), -execute_volume)]
            component_orders = self.convert_picnic_basket_orders(basket_type, synthetic_orders, order_depths)
            component_orders[basket_type] = basket_orders
            result = component_orders
            return result

        return None

    ####################################################
    # Main run function
    ####################################################
    def run(self, state: TradingState):
        # Retrieve persistent state if available.
        traderData = {}
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
        result = {}
        conversions = 0

        # Trade the individual products: CROISSANTS, JAMS, and DJEMBES.
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

        # Trade the Picnic Baskets using arbitrage between market price and synthetic (component-based) price.
        for basket in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            if basket in state.order_depths:
                basket_position = state.position[basket] if basket in state.position else 0
                spread_history = traderData.get(basket, {}).get("spread_history", [])
                arb_orders = self.picnic_basket_arbitrage_orders(basket, state.order_depths, basket_position, spread_history)
                if arb_orders:
                    for prod, orders in arb_orders.items():
                        if prod in result:
                            result[prod].extend(orders)
                        else:
                            result[prod] = orders
                # Update persistent state for the basket.
                if basket not in traderData:
                    traderData[basket] = {}
                traderData[basket]["spread_history"] = spread_history

        traderData_encoded = jsonpickle.encode(traderData)
        return result, conversions, traderData_encoded