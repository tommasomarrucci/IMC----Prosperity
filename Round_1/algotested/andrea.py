import json
from time import time
from typing import Any

import numpy as np
import pandas as pd

from Round_3.datamodel import (Listing, Observation, Order, OrderDepth,
                       ProsperityEncoder, Symbol, Time, Trade, TradingState)


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
        return [
            [listing.symbol, listing.product, listing.denomination]
            for listing in listings.values()
        ]

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        return {
            symbol: [order_depth.buy_orders, order_depth.sell_orders]
            for symbol, order_depth in order_depths.items()
        }

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


class Trader:
    def __init__(self):
        self.squid_ink = []
        self.kelp = []

    def _vw_mid_price(self, data):
        data = {i: abs(data[i]) if not np.isnan(data[i]) else 0 for i in data}

        bid_amt = data["bid_volume_1"] + data["bid_volume_2"] + data["bid_volume_3"]
        ask_amt = data["ask_volume_1"] + data["ask_volume_2"] + data["ask_volume_3"]
        tot_amt = bid_amt + ask_amt

        vw_bid = (
            data["bid_price_1"] * data["bid_volume_1"]
            + data["bid_price_2"] * data["bid_volume_2"]
            + data["bid_price_3"] * data["bid_volume_3"]
        ) / bid_amt
        vw_ask = (
            data["ask_price_1"] * data["ask_volume_1"]
            + data["ask_price_2"] * data["ask_volume_2"]
            + data["ask_price_3"] * data["ask_volume_3"]
        ) / ask_amt

        return vw_ask * bid_amt / tot_amt + vw_bid * ask_amt / tot_amt

    def _update_data(self, state: TradingState):
        order_depth = state.order_depths
        for product, orders in order_depth.items():
            data = {
                "timestamp": state.timestamp,
                'ask_volume_1': np.nan,
                'ask_volume_2': np.nan,
                'ask_volume_3': np.nan,
                'bid_volume_1': np.nan,
                'bid_volume_2': np.nan,
                'bid_volume_3': np.nan,
                'ask_price_1': np.nan,
                'ask_price_2': np.nan,
                'ask_price_3': np.nan,
                'bid_price_1': np.nan,
                'bid_price_2': np.nan,
                'bid_price_3': np.nan,
            }

            for i, limit in enumerate(orders.buy_orders, 1):
                data[f"bid_price_{i}"] = limit
                data[f"bid_volume_{i}"] = orders.buy_orders[limit]

            for i, limit in enumerate(orders.sell_orders, 1):
                data[f"ask_price_{i}"] = limit
                data[f"ask_volume_{i}"] = orders.sell_orders[limit]

            data['mid_price'] = (data["ask_price_1"] + data["bid_price_1"]) / 2
            data['spread'] = (data["ask_price_1"] - data["bid_price_1"]) / data["mid_price"]
            data['mid_price_vw'] = self._vw_mid_price(data)

            if product == "SQUID_INK":
                self.squid_ink.append(data)
            elif product == "KELP":
                self.kelp.append(data)

    def _get_MACD(self, prices: pd.DataFrame, price_col: str, min_periods: int = 5):
        SHORT_WINDOW, LONG_WINDOW, SMOOTHING_WINDOW = 500, 1000, 100
        prices = prices.tail(1000)

        ema12 = prices[price_col].ewm(span=SHORT_WINDOW, adjust=False).mean()
        ema26 = prices[price_col].ewm(span=LONG_WINDOW, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=SMOOTHING_WINDOW, adjust=False).mean()
        macd.iloc[:min_periods] = np.nan
        signal.iloc[:min_periods] = np.nan

        return macd, signal

    def _stoikov_bidask(self, mid_price, current_pos, target_pos, timestamp, gamma, sigma, k):
        q = current_pos - target_pos
        total_timestamps = 1_000_000
        T = 1
        t = timestamp / total_timestamps
        sigma = np.sqrt(sigma)

        ref_price = mid_price - q * gamma * sigma ** 2 * (T - t)
        stoikov_spread = gamma * sigma ** 2 * (T - t) + 2 / gamma * np.log(1 + gamma / k)

        logger.print('Pricing using Stoikov-Anelladeva Model')
        logger.print(f'Ref Price: {ref_price} Spread: {stoikov_spread}')
        logger.print('-----------')

        bid = int(np.floor(ref_price - stoikov_spread / 2))
        ask = int(np.ceil(ref_price + stoikov_spread / 2))

        return bid, ask

    def _get_orders_squid_ink(self, state: TradingState):
        logger.print('--- SQUID INK ORDERS ---')

        prices = pd.DataFrame.from_records(self.squid_ink)
        if prices.empty:
            return None

        current_pos = state.position.get("SQUID_INK", 0)

        macd_ask_1, signal_ask_1 = self._get_MACD(prices, "ask_price_1")

        trend = "UP" if macd_ask_1.iloc[-1] > signal_ask_1.iloc[-1] else "DOWN"

        avg_spread = prices["spread"].mean()
        mid_price = prices['mid_price_vw'].iloc[-1]

        gamma = 0.01
        k = 0.5
        sigma = prices['mid_price'].rolling(50, min_periods=5).var().iloc[-1]
        if np.isnan(sigma):
            sigma = 3.8
        timestamp = state.timestamp

        if trend == 'UP':
            target_pos = 15
            bid_amt = 20 - current_pos
            ask_amt = min(-current_pos + 5, -5)
        else:
            target_pos = -15
            bid_amt = max(-current_pos - 5, 5)
            ask_amt = -20 - current_pos

        bid_price, ask_price = self._stoikov_bidask(mid_price, current_pos, target_pos, timestamp, gamma, sigma, k)

        return [
            Order("SQUID_INK", bid_price, bid_amt),
            Order("SQUID_INK", ask_price, ask_amt),
        ]

    def _get_orders_kelp(self, state: TradingState):
        logger.print('--- KELP ORDERS ---')

        prices = pd.DataFrame.from_records(self.kelp)
        if prices.empty:
            return None

        mid_price = prices['mid_price_vw'].iloc[-1]
        current_pos = state.position.get("KELP", 0)
        t = state.timestamp

        gamma = 0.025
        k = 0.5
        sigma = prices['mid_price'].rolling(50, min_periods=5).var().iloc[-1]
        if np.isnan(sigma):
            sigma = 2.24

        bid, ask = self._stoikov_bidask(mid_price, current_pos, 0, t, gamma, sigma, k)

        bid_amt = 20 - current_pos
        ask_amt = -20 - current_pos

        return [
            Order("KELP", bid, bid_amt),
            Order("KELP", ask, ask_amt),
        ]

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        t0 = time()
        orders = {}
        conversions = 0
        trader_data = ""

        self._update_data(state)

        squid_orders = self._get_orders_squid_ink(state)
        if squid_orders:
            orders["SQUID_INK"] = squid_orders

        kelp_orders = self._get_orders_kelp(state)
        if kelp_orders:
            orders["KELP"] = kelp_orders

        logger.flush(state, orders, conversions, trader_data)
        print(f"Time taken: {(time() - t0) * 100:.2f}ms")

        return orders, conversions, trader_data
