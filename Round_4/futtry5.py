from datamodel import Order, TradingState, OrderDepth, ConversionObservation
from typing import List
import jsonpickle
import numpy as np


class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

# Strategy parameters including CSI, conversion limits, and momentum biases
PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 0.7,              # base spread
        "make_min_edge": 0.2,          # minimum make spread
        "init_make_edge": 0.7,         # starting edge
        "min_edge": 0.2,               # lower bound on edge
        "volume_avg_timestamp": 5,     # lookback for volume adaptation
        "volume_bar": 30,              # threshold volume
        "dec_edge_discount": 0.5,      # discount factor for shrinking edge
        "step_size": 0.2,              # edge adjustment increment
        "CSI": 30.0,                   # Critical Sunlight Index
        "sun_bias_edge": 0.2,          # edge tilt under low sun
        "conv_limit": 20,              # max conversion per timestamp
        "sugar_momentum_window": 3,    # sugar price lookback
        "sugar_bias_edge": 0.1         # edge tilt from sugar momentum
    }
}


class Trader:
    def __init__(self, params=None):
        self.params = params or PARAMS
        self.LIMIT = {Product.MAGNIFICENT_MACARONS: 75}

    def implied_bid_ask(self, obs: ConversionObservation):
        bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        ask = obs.askPrice + obs.importTariff + obs.transportFees
        return bid, ask

    def adaptive_edge(self, timestamp, curr_edge, position, traderObject):
        params = self.params[Product.MAGNIFICENT_MACARONS]
        hist = traderObject.setdefault("volume_history", [])
        hist.append(abs(position))
        if len(hist) > params["volume_avg_timestamp"]:
            hist.pop(0)

        if len(hist) < params["volume_avg_timestamp"]:
            return curr_edge

        avg_volume = np.mean(hist)
        # widen on high volume
        if avg_volume >= params["volume_bar"]:
            traderObject["volume_history"] = []
            return curr_edge + params["step_size"]
        # tighten on low volume
        if params["dec_edge_discount"] * params["volume_bar"] * (curr_edge - params["step_size"]) > avg_volume * curr_edge:
            new_edge = max(curr_edge - params["step_size"], params["min_edge"])
            traderObject["volume_history"] = []
            return new_edge
        return curr_edge

    def run(self, state: TradingState):
        product = Product.MAGNIFICENT_MACARONS
        params = self.params[product]
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        position = state.position.get(product, 0)
        obs = state.observations.conversionObservations[product]
        order_depth = state.order_depths[product]

        # 1) Adaptive edge by volume
        curr_edge = traderObject.get("curr_edge", params["init_make_edge"])
        curr_edge = self.adaptive_edge(state.timestamp, curr_edge, position, traderObject)
        traderObject["curr_edge"] = curr_edge

        # 2) Implied conversion prices
        implied_bid, implied_ask = self.implied_bid_ask(obs)

        # 3) Sugar momentum signal
        sugar_hist = traderObject.setdefault("sugar_history", [])
        sugar_hist.append(obs.sugarPrice)
        if len(sugar_hist) > params["sugar_momentum_window"]:
            sugar_hist.pop(0)
        sugar_mom = 0.0
        if len(sugar_hist) == params["sugar_momentum_window"]:
            sugar_mom = sugar_hist[-1] - sugar_hist[0]

        # Base edges
        bid_edge = ask_edge = curr_edge
        # Tilt on sugar momentum
        if sugar_mom > 0:
            bid_edge = max(curr_edge - params["sugar_bias_edge"], params["min_edge"])
        elif sugar_mom < 0:
            ask_edge = curr_edge + params["sugar_bias_edge"]

        # 4) Sunlight bias if below CSI
        if obs.sunlightIndex < params["CSI"]:
            bid_edge = max(bid_edge - params["sun_bias_edge"], params["min_edge"])
            ask_edge = ask_edge + params["sun_bias_edge"]

        orders: List[Order] = []
        limit = self.LIMIT[product]

        # TAKE: hit book orders
        for ask in sorted(order_depth.sell_orders):
            if ask <= implied_bid - bid_edge:
                vol = min(order_depth.sell_orders[ask], limit - position)
                orders.append(Order(product, ask, vol))
                position += vol
        for bid in sorted(order_depth.buy_orders, reverse=True):
            if bid >= implied_ask + ask_edge:
                vol = min(order_depth.buy_orders[bid], limit + position)
                orders.append(Order(product, bid, -vol))
                position -= vol

        # MAKE: post quotes
        make_bid = round(implied_bid - bid_edge)
        make_ask = round(implied_ask + ask_edge)
        if position < limit:
            orders.append(Order(product, make_bid, limit - position))
        if position > -limit:
            orders.append(Order(product, make_ask, -(limit + position)))

        # 5) Chunked conversions up to conv_limit
        conv_limit = params["conv_limit"]
        to_conv = min(abs(position), conv_limit)
        conversions = 0

        traderData = jsonpickle.encode(traderObject)
        return {product: orders}, conversions, traderData
