from datamodel import Order, TradingState, OrderDepth, ConversionObservation
from typing import Dict, List
import jsonpickle
import numpy as np


class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 1.0,             # tighter edge
        "make_min_edge": 0.3,         # allows more aggressive quoting
        "make_probability": 0.5,
        "init_make_edge": 0.9,        # initial spread
        "min_edge": 0.2,
        "volume_avg_timestamp": 5,
        "volume_bar": 50,
        "dec_edge_discount": 0.8,
        "step_size": 0.2
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.MAGNIFICENT_MACARONS: 75}

    def implied_bid_ask(self, obs: ConversionObservation):
        bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        ask = obs.askPrice + obs.importTariff + obs.transportFees
        return bid, ask

    def adaptive_edge(self, timestamp, curr_edge, position, traderObject):
        product = Product.MAGNIFICENT_MACARONS
        params = self.params[product]
        hist = traderObject.setdefault("volume_history", [])

        hist.append(abs(position))
        if len(hist) > params["volume_avg_timestamp"]:
            hist.pop(0)

        if len(hist) < params["volume_avg_timestamp"]:
            return curr_edge

        avg_volume = np.mean(hist)

        if avg_volume >= params["volume_bar"]:
            traderObject["volume_history"] = []
            return curr_edge + params["step_size"]

        elif params["dec_edge_discount"] * params["volume_bar"] * (curr_edge - params["step_size"]) > avg_volume * curr_edge:
            new_edge = max(curr_edge - params["step_size"], params["min_edge"])
            traderObject["volume_history"] = []
            return new_edge

        return curr_edge

    def run(self, state: TradingState):
        product = Product.MAGNIFICENT_MACARONS
        params = self.params[product]
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        position = state.position.get(product, 0)
        observation = state.observations.conversionObservations[product]
        order_depth = state.order_depths[product]
        limit = self.LIMIT[product]

        curr_edge = traderObject.get("curr_edge", params["init_make_edge"])
        curr_edge = self.adaptive_edge(state.timestamp, curr_edge, position, traderObject)
        traderObject["curr_edge"] = curr_edge

        implied_bid, implied_ask = self.implied_bid_ask(observation)

        conversions = -position  # always flatten
        orders = []

        # TAKE orders (more aggressive now)
        for ask in sorted(order_depth.sell_orders):
            if ask <= implied_bid - curr_edge:
                volume = min(abs(order_depth.sell_orders[ask]), limit - position)
                orders.append(Order(product, ask, volume))
                position += volume

        for bid in sorted(order_depth.buy_orders, reverse=True):
            if bid >= implied_ask + curr_edge:
                volume = min(abs(order_depth.buy_orders[bid]), limit + position)
                orders.append(Order(product, bid, -volume))
                position -= volume

        # MAKE orders
        make_bid = round(implied_bid - curr_edge)
        make_ask = round(implied_ask + curr_edge)

        if position < limit:
            orders.append(Order(product, make_bid, limit - position))
        if position > -limit:
            orders.append(Order(product, make_ask, -(limit + position)))

        traderData = jsonpickle.encode(traderObject)
        return {product: orders}, conversions, traderData

