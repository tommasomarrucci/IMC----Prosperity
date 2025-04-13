from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
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

# Updated parameters. Note that for individual products we have fair values,
# while for the baskets we rely on our synthetic spread method.
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
    # For baskets, we use parameters similar to your SPREAD products.
    Product.PICNIC_BASKET1: {
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 60,
        "spread_mean": 379.5,      # These can be adjusted by historical data
        "starting_its": 30000,
        "spread_std_window": 25,
        "zscore_threshold": 11,
        "target_position": 20,
    },
    Product.PICNIC_BASKET2: {
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 100,
        "spread_mean": 379.5,
        "starting_its": 30000,
        "spread_std_window": 25,
        "zscore_threshold": 11,
        "target_position": 30,
    },
}

# Basket composition based on the new definitions.
# PICNIC_BASKET1 contains: 6 CROISSANTS, 3 JAMS, 1 DJEMBES.
# PICNIC_BASKET2 contains: 4 CROISSANTS, 2 JAMS.
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
        self.LIMIT = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
        }

    ####################################################
    # Methods for individual product trading
    ####################################################
    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float, position: int) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        position_limit = self.LIMIT[product]

        # Check if the best ask is below fair value and take liquidity.
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_volume, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity

        # Similarly, if the best bid is above fair value, take liquidity.
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

        # Clear positions if you are long.
        if position_after_take > 0:
            clear_volume = sum(vol for price, vol in order_depth.buy_orders.items() if price >= fair_ask)
            clear_volume = min(clear_volume, position_after_take)
            execute_volume = min(position_limit + position, clear_volume)
            if execute_volume > 0:
                orders.append(Order(product, fair_ask, -abs(execute_volume)))
                sell_order_volume += abs(execute_volume)

        # Or if short, clear by buying.
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
    # Basket synthesis and conversion methods for Picnic Baskets
    ####################################################
    def get_synthetic_basket_order_depth(self, basket_type: str, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        """
        Build synthetic order depth for a picnic basket using its component order depths
        and the basket composition defined in BASKET_COMPOSITION.
        """
        synthetic_depth = OrderDepth()
        composition = BASKET_COMPOSITION[basket_type]
        implied_bid = 0
        implied_ask = 0
        bid_volumes = []
        ask_volumes = []
        for comp, qty in composition.items():
            comp_depth = order_depths.get(comp)
            if comp_depth is None:
                continue
            # For the bid side: weighted sum using the best bid.
            if comp_depth.buy_orders:
                comp_best_bid = max(comp_depth.buy_orders.keys())
                implied_bid += comp_best_bid * qty
                bid_volumes.append(comp_depth.buy_orders[comp_best_bid] // qty)
            else:
                bid_volumes.append(0)
            # For the ask side: weighted sum using the best ask.
            if comp_depth.sell_orders:
                comp_best_ask = min(comp_depth.sell_orders.keys())
                implied_ask += comp_best_ask * qty
                ask_volumes.append((-comp_depth.sell_orders[comp_best_ask]) // qty)
            else:
                ask_volumes.append(0)
        if bid_volumes:
            implied_bid_volume = min(bid_volumes)
        else:
            implied_bid_volume = 0
        if ask_volumes:
            implied_ask_volume = min(ask_volumes)
        else:
            implied_ask_volume = 0
        if implied_bid > 0:
            synthetic_depth.buy_orders[implied_bid] = implied_bid_volume
        if implied_ask > 0 and implied_ask != float('inf'):
            synthetic_depth.sell_orders[implied_ask] = -implied_ask_volume
        return synthetic_depth

    def convert_basket_orders(self, basket_type: str, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        """
        Convert a synthetic basket order into component orders using BASKET_COMPOSITION.
        """
        component_orders: Dict[str, List[Order]] = {comp: [] for comp in BASKET_COMPOSITION[basket_type]}
        synthetic_basket_depth = self.get_synthetic_basket_order_depth(basket_type, order_depths)
        best_bid = max(synthetic_basket_depth.buy_orders.keys()) if synthetic_basket_depth.buy_orders else 0
        best_ask = min(synthetic_basket_depth.sell_orders.keys()) if synthetic_basket_depth.sell_orders else float("inf")
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                # For buy orders: trade components at their best ask prices.
                for comp in BASKET_COMPOSITION[basket_type]:
                    comp_depth = order_depths.get(comp)
                    if comp_depth and comp_depth.sell_orders:
                        comp_price = min(comp_depth.sell_orders.keys())
                        comp_order = Order(comp, comp_price, quantity * BASKET_COMPOSITION[basket_type][comp])
                        component_orders[comp].append(comp_order)
            elif quantity < 0 and price <= best_bid:
                # For sell orders: trade components at their best bid prices.
                for comp in BASKET_COMPOSITION[basket_type]:
                    comp_depth = order_depths.get(comp)
                    if comp_depth and comp_depth.buy_orders:
                        comp_price = max(comp_depth.buy_orders.keys())
                        comp_order = Order(comp, comp_price, quantity * BASKET_COMPOSITION[basket_type][comp])
                        component_orders[comp].append(comp_order)
        return component_orders

    def spread_orders(self, basket_type: str, order_depths: Dict[str, OrderDepth], basket_position: int, spread_history: List[float]):
        """
        Generate orders if the spread between the market basket and its synthetic price exceeds a threshold.
        (We keep the same structure as before, but update names and use basket_type.)
        """
        market_depth = order_depths[basket_type]
        synthetic_depth = self.get_synthetic_basket_order_depth(basket_type, order_depths)
        market_mid = (max(market_depth.buy_orders.keys()) + min(market_depth.sell_orders.keys())) / 2 if (market_depth.buy_orders and market_depth.sell_orders) else None
        synthetic_mid = (max(synthetic_depth.buy_orders.keys()) + min(synthetic_depth.sell_orders.keys())) / 2 if (synthetic_depth.buy_orders and synthetic_depth.sell_orders) else None
        if market_mid is None or synthetic_mid is None:
            return None

        spread = market_mid - synthetic_mid
        spread_history.append(spread)
        threshold = self.params[basket_type]["zscore_threshold"]  # Using zscore_threshold as the trigger

        result = {}
        # If the basket is overpriced relative to its synthetic value.
        if spread >= threshold:
            target_quantity = abs(self.params[basket_type]["target_position"] - basket_position)
            if target_quantity <= 0:
                return None
            basket_bid_price = max(market_depth.buy_orders.keys())
            basket_bid_volume = abs(market_depth.buy_orders[basket_bid_price])
            execute_volume = min(basket_bid_volume, target_quantity)
            basket_orders = [Order(basket_type, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order("SYNTHETIC_" + basket_type, min(synthetic_depth.sell_orders.keys()), execute_volume)]
            component_orders = self.convert_basket_orders(basket_type, synthetic_orders, order_depths)
            component_orders[basket_type] = basket_orders
            return component_orders

        # If the basket is underpriced.
        if spread <= -threshold:
            target_quantity = abs(self.params[basket_type]["target_position"] - basket_position)
            if target_quantity <= 0:
                return None
            basket_ask_price = min(market_depth.sell_orders.keys())
            basket_ask_volume = abs(market_depth.sell_orders[basket_ask_price])
            execute_volume = min(basket_ask_volume, target_quantity)
            basket_orders = [Order(basket_type, basket_ask_price, execute_volume)]
            synthetic_orders = [Order("SYNTHETIC_" + basket_type, max(synthetic_depth.buy_orders.keys()), -execute_volume)]
            component_orders = self.convert_basket_orders(basket_type, synthetic_orders, order_depths)
            component_orders[basket_type] = basket_orders
            return component_orders

        return None

    ####################################################
    # Main run function
    ####################################################
    def run(self, state: TradingState):
        # Retrieve stored trader data.
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

        # Trade the picnic baskets (both types) using spread arbitrage.
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
                # Update persistent trader data.
                if basket not in traderData:
                    traderData[basket] = {}
                traderData[basket]["spread_history"] = spread_history

        traderData_encoded = jsonpickle.encode(traderData)
        return result, conversions, traderData_encoded