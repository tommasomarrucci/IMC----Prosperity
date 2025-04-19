from datamodel import Order, TradingState, OrderDepth, ConversionObservation
from typing import Dict, List
import jsonpickle
import numpy as np


class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


# --- PARAMETERS, now including CSI and sensitivity to sunlight deviations
PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 1.0,             # tighter edge
        "make_min_edge": 0.3,         # allows more aggressive quoting
        "make_probability": 0.5,
        "init_make_edge": 0.9,        # initial spread half‐width
        "min_edge": 0.2,
        "volume_avg_timestamp": 5,
        "volume_bar": 50,
        "dec_edge_discount": 0.8,
        "step_size": 0.2,
        # Sunlight‐based parameters:
        "CSI": 0.5,                   # Critical Sunlight Index threshold
        "sunlight_sensitivity": 0.2, # how much to shift mid‐price per unit below CSI
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        # position limits
        self.LIMIT = {Product.MAGNIFICENT_MACARONS: 75}
        # conversion limits
        self.CONVERSION_LIMIT = {Product.MAGNIFICENT_MACARONS: 10}

    def implied_bid_ask(self, obs: ConversionObservation, params: Dict):
        """
        Compute a fee‐adjusted bid/ask, *then* if sunlightIndex < CSI,
        shift both up by sensitivity * (CSI – sunlightIndex).
        """
        # base implied bid/ask
        bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        ask = obs.askPrice + obs.importTariff + obs.transportFees

        # sunlight adjustment
        csi = params["CSI"]
        si = getattr(obs, "sunlightIndex", None)
        if si is not None and si < csi:
            shift = params["sunlight_sensitivity"] * (csi - si)
            bid += shift
            ask += shift

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

        # current position
        position = state.position.get(product, 0)
        # get the enriched observation (includes sunlightIndex)
        observation: ConversionObservation = state.observations.conversionObservations[product]
        order_depth: OrderDepth = state.order_depths[product]
        limit = self.LIMIT[product]

        # 1) adapt your quoting edge based on recent volume
        curr_edge = traderObject.get("curr_edge", params["init_make_edge"])
        curr_edge = self.adaptive_edge(state.timestamp, curr_edge, position, traderObject)
        traderObject["curr_edge"] = curr_edge

        # 2) compute implied bid & ask *with* sunlight adjustment
        implied_bid, implied_ask = self.implied_bid_ask(observation, params)

        orders: List[Order] = []

        # 3) TAKE liquidity where it's extremely favorable
        #   buy if ask is better than our bid minus edge
        for ask_p in sorted(order_depth.sell_orders):
            if ask_p <= implied_bid - curr_edge:
                vol = min(abs(order_depth.sell_orders[ask_p]), limit - position)
                orders.append(Order(product, ask_p, vol))
                position += vol

        #   sell if bid is better than our ask plus edge
        for bid_p in sorted(order_depth.buy_orders, reverse=True):
            if bid_p >= implied_ask + curr_edge:
                vol = min(abs(order_depth.buy_orders[bid_p]), limit + position)
                orders.append(Order(product, bid_p, -vol))
                position -= vol

        # 4) MAKE (passive) quotes on both sides around the (shifted) mid‐price
        make_bid = round(implied_bid - curr_edge)
        make_ask = round(implied_ask + curr_edge)

        if position < limit:
            orders.append(Order(product, make_bid, limit - position))
        if position > -limit:
            orders.append(Order(product, make_ask, -(limit + position)))

        # 5) FLATTEN via conversion, *capped* at ±CONVERSION_LIMIT
        raw_conv = -position
        cap = self.CONVERSION_LIMIT[product]
        conversions = 0

        # persist state
        traderData = jsonpickle.encode(traderObject)
        return {product: orders}, conversions, traderData