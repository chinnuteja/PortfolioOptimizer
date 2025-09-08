# data_fetcher.py
# This module is responsible for downloading historical market data.

import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_data(assets: dict, sentiment_ticker: str, start_date: str, end_date: str) -> (pd.DataFrame, pd.Series):
    """
    Downloads historical closing prices for assets and a sentiment proxy.

    Args:
        assets (dict): A dictionary mapping asset names to their ticker symbols.
        sentiment_ticker (str): The ticker symbol for the sentiment proxy (e.g., VIX).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        tuple(pd.DataFrame, pd.Series): A tuple containing:
            - A DataFrame with the historical asset prices.
            - A Series with the historical sentiment proxy values.
    """
    logger.debug("get_data(): starting download | assets=%s sentiment_ticker=%s start=%s end=%s", assets, sentiment_ticker, start_date, end_date)
    try:
        # Combine all tickers for a single download request
        all_tickers = list(assets.values()) + [sentiment_ticker]
        raw_data = yf.download(all_tickers, start=start_date, end=end_date)['Close']
        logger.debug("get_data(): raw_data columns=%s shape=%s", getattr(raw_data, 'columns', None), getattr(raw_data, 'shape', None))

        # --- ROBUST MAPPING LOGIC ---
        asset_prices = pd.DataFrame()
        for name, ticker in assets.items():
            if ticker in raw_data.columns:
                asset_prices[name] = raw_data[ticker]
            else:
                logger.warning("get_data(): missing downloaded column for asset | name=%s ticker=%s", name, ticker)

        sentiment_data = raw_data[sentiment_ticker]
        # ---------------------------

        # Drop any rows with missing data
        combined_data = asset_prices.join(sentiment_data).dropna()
        logger.debug("get_data(): combined_data shape after dropna=%s", getattr(combined_data, 'shape', None))

        # Separate the data back into assets and sentiment
        final_asset_prices = combined_data[list(assets.keys())]
        final_sentiment_data = combined_data[sentiment_ticker]

        logger.debug("get_data(): final_asset_prices head=\n%s", final_asset_prices.head().to_string())
        logger.info("get_data(): data download successful | prices_shape=%s sentiment_len=%s", final_asset_prices.shape, len(final_sentiment_data))
        return final_asset_prices, final_sentiment_data

    except Exception as e:
        logger.exception("get_data(): error during data download: %s", e)
        return pd.DataFrame(), pd.Series()