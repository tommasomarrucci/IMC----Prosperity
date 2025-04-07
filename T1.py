import json
from time import time
from typing import Any, List

import numpy as np
import pandas as pd

from datamodel import ( 
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end
    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        print(
            json.dumps(
                [
                    self.compress_state(state),
                    self.compress_orders(orders),
                    conversions,
                    trader_data,
                    self.logs,
                ],
                cls=ProsperityEncoder,
                separators=(",", ":"),
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]]
            )

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed


logger = Logger()

example_price = {}


class Trader:
    def __init__(self):
        # store order book data from squid_ink and kelp
        self.squid_ink = []
        self.kelp = []

    def _update_data(self, state: TradingState):
        """
        Update the prices data in the object
        instance, recording the last prices
        """

        order_depth = state.order_depths
        for product, orders in order_depth.items():
            data = {
                "timestamp": state.timestamp,
            }

            for i, limit in enumerate(orders.buy_orders, 1):
                amount = orders.buy_orders[limit]

                data[f"bid_price_{i}"] = limit
                data[f"bid_volume_{i}"] = amount

            for i, limit in enumerate(orders.sell_orders, 1):
                amount = orders.sell_orders[limit]

                data[f"ask_price_{i}"] = limit
                data[f"ask_volume_{i}"] = amount

            if product == "SQUID_INK":
                self.squid_ink.append(data)
            elif product == "KELP":
                self.kelp.append(data)

    def _get_MACD(self, prices: pd.DataFrame, price_col: str, min_periods: int = 5):
        """
        Compute MACD for a given price column in the a prices dataframe

        Parameters
        ----------
        prices : pd.DataFrame
            The DataFrame with bid/asks for the product
        price_col : str
            The column to use for the MACD calculation.
        min_periods : int, optional
            Min periods for the MACD to make sense, by default 5.
            Periods before this will be NaN.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            The MACD and Signal lines
        """
        SHORT_WINDOW = 500
        LONG_WINDOW = 1000
        SMOOTHING_WINDOW = 100

        ema12 = prices[price_col].ewm(span=SHORT_WINDOW, adjust=False).mean()
        ema26 = prices[price_col].ewm(span=LONG_WINDOW, adjust=False).mean()
        macd = ema12 - ema26

        signal = macd.ewm(span=SMOOTHING_WINDOW, adjust=False).mean()
        macd.iloc[:min_periods] = np.nan
        signal.iloc[:min_periods] = np.nan

        return macd, signal

    def _get_orders_squid_ink(self, state: TradingState) -> Order | None:
        """
        Process orders for Squid Ink

        Used MACD to determine buy/sell signals. Buy when the MACD crosses the signal,
        and close the position when the MACD crosses under the signal.

        Parameters
        ----------
        state : TradingState
            The Trading State.

        Returns
        -------
        order : Order
            The order to submit to the market.
        """

        prices = pd.DataFrame.from_records(self.squid_ink)
        current_pos = (
            state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
        )

        macd_ask_1, signal_ask_1 = self._get_MACD(prices, "ask_price_1")
        macd_bid_1, signal_bid_1 = self._get_MACD(prices, "bid_price_1")

        if macd_ask_1.iloc[-1] > signal_ask_1.iloc[-1] and current_pos < 20:
            price = prices["ask_price_1"].iloc[-1]
            amount = 20 - current_pos
            order = Order("SQUID_INK", int(price), int(amount))
            return order

        if macd_bid_1.iloc[-1] < signal_bid_1.iloc[-1] and current_pos > 0:
            price = prices["bid_price_1"].iloc[-1]
            amount = -current_pos
            order = Order("SQUID_INK", int(price), int(amount))
            return order

    def _get_orders_kelp(self):
        pass

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        t0 = time()  # start timer
        orders = {}
        conversions = 0
        trader_data = ""

        self._update_data(state)

        order_squid_ink = self._get_orders_squid_ink(state)
        if order_squid_ink:
            orders["SQUID_INK"] = [order_squid_ink]

        order_kelp = self._get_orders_kelp()
        if order_kelp:
            orders["KELP"] = [order_kelp]

        logger.flush(state, orders, conversions, trader_data)
        print(f"Time taken: {(time() - t0) * 100}ms")

        return orders, conversions, trader_data