# data_fetcher.py
# Responsible for downloading historical market data.

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
import time
from typing import Tuple

logger = logging.getLogger(__name__)

# --- simple in-memory TTL cache for downloaded price data (per process) ---
_DATA_CACHE: dict = {}  # key -> (expiry_ts, asset_prices_df, sentiment_series)
DATA_TTL_SECONDS = int(os.environ.get("DATA_TTL_SECONDS", "600"))  # default 10 minutes

def _cache_key(assets: dict, sentiment_ticker: str, start_date: str, end_date: str) -> Tuple:
    auto_adjust = os.environ.get("YF_AUTO_ADJUST", "0")
    return (
        tuple(sorted(assets.items())),  # stable order
        sentiment_ticker,
        start_date,
        end_date,
        auto_adjust,
    )

def _get_fallback_data(assets: dict, sentiment_ticker: str, start_date: str, end_date: str) -> (pd.DataFrame, pd.Series):
    """
    Generate fallback data when yfinance fails.
    Creates realistic sample data for development/testing.
    """
    logger.warning("get_data(): using fallback data due to yfinance failure")

    # Create date range (weekdays)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range = date_range[date_range.weekday < 5]

    # Generate realistic sample data
    np.random.seed(42)
    asset_prices = pd.DataFrame(index=date_range)
    sentiment_data = pd.Series(index=date_range, dtype="float64")

    base_prices = {
        'Equities': 15000.0,
        'Gold': 2000.0,
        'REITs': 100.0,
        'Bitcoin': 45000.0
    }

    for name in assets.keys():
        if name in base_prices:
            returns = np.random.normal(0.0005, 0.02, len(date_range))
            prices = [float(base_prices[name])]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            asset_prices[name] = prices

    sentiment_vals = np.random.normal(20, 5, len(date_range))
    sentiment_vals = np.clip(sentiment_vals, 10, 50).astype("float64")
    sentiment_data = pd.Series(sentiment_vals, index=date_range, dtype="float64")

    logger.info("get_data(): fallback generated | asset_prices=%s rows, sentiment=%s rows",
                asset_prices.shape, len(sentiment_data))
    return asset_prices, sentiment_data

def _download(all_tickers, start_date, end_date):
    """Single yfinance call with explicit options."""
    auto_adjust = os.environ.get("YF_AUTO_ADJUST", "0") in ("1", "true", "True")
    df = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=auto_adjust,
        group_by="column",
        threads=False,
    )
    return df

def _get_with_retries(all_tickers, start_date, end_date, attempts=3, backoff=1.5):
    last_err = None
    for i in range(1, attempts + 1):
        try:
            return _download(all_tickers, start_date, end_date)
        except Exception as e:
            last_err = e
            logger.warning("yfinance download retry %d/%d failed: %s", i, attempts, e)
            time.sleep(backoff ** i)
    raise RuntimeError(f"yfinance retries exhausted: {last_err}")

def get_data(assets: dict, sentiment_ticker: str, start_date: str, end_date: str) -> (pd.DataFrame, pd.Series):
    """
    Downloads historical closing prices for assets and a sentiment proxy.
    Falls back to sample data if yfinance fails. Caches successful results
    in-memory for DATA_TTL_SECONDS (per process).

    IMPORTANT: We DO NOT inner-join assets with sentiment here.
    We clean and return them separately to avoid truncating asset history
    when the sentiment series lags.
    """
    logger.debug("get_data(): assets=%s sentiment=%s start=%s end=%s", assets, sentiment_ticker, start_date, end_date)

    # --- in-memory TTL cache ---
    key = _cache_key(assets, sentiment_ticker, start_date, end_date)
    now = time.time()
    hit = _DATA_CACHE.get(key)
    if hit and hit[0] > now:
        asset_prices_cached, sentiment_cached = hit[1], hit[2]
        logger.info("get_data(): cache hit | asset_prices=%s rows, sentiment=%s rows",
                    getattr(asset_prices_cached, "shape", None), len(sentiment_cached))
        return asset_prices_cached.copy(), sentiment_cached.copy()

    try:
        all_tickers = list(assets.values()) + [sentiment_ticker]
        raw_data = _get_with_retries(all_tickers, start_date, end_date, attempts=3, backoff=1.5)
        if raw_data is None or getattr(raw_data, "empty", True):
            raise Exception("yfinance returned empty data")

        # Flatten multi-index to Close/Adj Close
        if isinstance(raw_data.columns, pd.MultiIndex):
            if "Close" in raw_data.columns.levels[0]:
                raw_data = raw_data['Close']
            elif "Adj Close" in raw_data.columns.levels[0]:
                raw_data = raw_data['Adj Close']

        # Map to asset names
        asset_prices = pd.DataFrame()
        for name, ticker in assets.items():
            if ticker in raw_data.columns:
                asset_prices[name] = pd.to_numeric(raw_data[ticker], errors="coerce")
            else:
                logger.warning("get_data(): missing column for asset | name=%s ticker=%s", name, ticker)

        # Sentiment separate
        if sentiment_ticker in raw_data.columns:
            sentiment_data = pd.to_numeric(raw_data[sentiment_ticker], errors="coerce")
        else:
            logger.warning("get_data(): missing sentiment data for ticker=%s", sentiment_ticker)
            sentiment_data = pd.Series(dtype="float64")

        # --- CLEAN SEPARATELY (no inner join) ---
        asset_prices = asset_prices.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        sentiment_data = sentiment_data.replace([np.inf, -np.inf], np.nan).dropna()

        if asset_prices.empty:
            raise Exception("Insufficient asset price data after processing")

        _DATA_CACHE[key] = (now + DATA_TTL_SECONDS, asset_prices.copy(), sentiment_data.copy())
        logger.info("get_data(): data ok | asset_prices=%s rows, sentiment=%s rows",
                    asset_prices.shape, len(sentiment_data))
        return asset_prices, sentiment_data

    except Exception as e:
        logger.warning("get_data(): yfinance failed: %s, using fallback", e)
        prices_fb, sent_fb = _get_fallback_data(assets, sentiment_ticker, start_date, end_date)
        _DATA_CACHE[key] = (now + DATA_TTL_SECONDS, prices_fb.copy(), sent_fb.copy())
        return prices_fb, sent_fb
