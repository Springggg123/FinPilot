"""Download daily prices without changing the saved demo CSV."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


def validate_prices(prices, tickers):
    # Require enough complete, positive prices for analysis and testing.
    if prices is None or prices.empty:
        raise ValueError("No price data was returned.")
    if set(prices.columns) != set(tickers):
        raise ValueError("Some assets are missing from the download.")
    prices = prices.sort_index().copy()
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price dates are invalid.")
    if prices.index.has_duplicates or prices.index.hasnans:
        raise ValueError("Price dates are missing or duplicated.")
    if len(prices) < 100:
        raise ValueError("Too few trading days were returned.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Some prices are missing or invalid.")
    return prices


def download_daily_prices(tickers):
    # Exclude today's US market date so an unfinished daily bar is not used.
    market_date = datetime.now(ZoneInfo("America/New_York")).date()
    end = pd.Timestamp(market_date)
    start = end - pd.DateOffset(years=1)
    data = yf.download(
        tickers=list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        group_by="column",
        multi_level_index=True,
        progress=False,
        timeout=20,
    )
    if data is None or data.empty:
        raise ValueError("The data provider returned no prices.")
    if "Close" not in data.columns.get_level_values(0):
        raise ValueError("Closing prices are missing.")
    prices = validate_prices(data["Close"], tickers)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    # Reject unexpected dates instead of silently using partial data.
    if prices.index.max().date() >= market_date:
        raise ValueError("The download contains an unfinished market date.")
    return prices
