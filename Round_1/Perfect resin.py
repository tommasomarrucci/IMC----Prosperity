from Round_3.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle

class Trader:
    def __init__(self):
        pass

    def resin_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_volume = 0
        sell_volume = 0
        fair_value = 10000  # Reference fair value for RAINFOREST_RESIN

        # Check sell orders: if best ask is below 9998, buy enough to move toward +50.
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask < 9999 and position < 50:
                quantity = min(-order_depth.sell_orders[best_ask], 50 - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_volume += quantity

        # Check buy orders: if best bid is above 10003, sell enough to move toward -50.
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid > 10001 and position > -50:
                quantity = min(order_depth.buy_orders[best_bid], position + 50)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                    sell_volume += quantity

        # If after the above orders we still haven't reached +50, place an additional buy order.
        buy_quantity = 50 - (position + buy_volume)
        if buy_quantity > 0:
            # Placing an order just below the fair value (i.e. 9996 when fair_value is 10000)
            orders.append(Order("RAINFOREST_RESIN", fair_value - 4, buy_quantity))

        # If we still haven't reached -50 on the sell side, place an additional sell order.
        sell_quantity = (position + 50) - sell_volume
        if sell_quantity > 0:
            # Placing an order just above the fair value (i.e. 10004 when fair_value is 10000)
            orders.append(Order("RAINFOREST_RESIN", fair_value + 4, -sell_quantity))

        return orders

    def run(self, state: TradingState):
        result = {}
        resin_position_limit = 50  # Target limits: +50 for long, -50 for short

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            resin_order_list = self.resin_orders(state.order_depths["RAINFOREST_RESIN"], resin_position, resin_position_limit)
            result["RAINFOREST_RESIN"] = resin_order_list

        traderData = jsonpickle.encode({})
        conversions = 1

        return result, conversions, traderData