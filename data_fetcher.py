# data_fetcher.py
# This module is responsible for downloading historical market data.

import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _get_fallback_data(assets: dict, sentiment_ticker: str, start_date: str, end_date: str) -> (pd.DataFrame, pd.Series):
    """
    Generate fallback data when yfinance fails.
    Creates realistic sample data for development/testing.
    """
    logger.warning("get_data(): using fallback data due to yfinance failure")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range = date_range[date_range.weekday < 5]  # Only weekdays
    
    # Generate realistic sample data
    np.random.seed(42)  # For reproducible results
    asset_prices = pd.DataFrame(index=date_range)
    sentiment_data = pd.Series(index=date_range)
    
    # Sample asset prices (realistic ranges)
    base_prices = {
        'Equities': 15000,    # Nifty 50
        'Gold': 2000,         # Gold per ounce
        'REITs': 100,         # REIT ETF
        'Bitcoin': 45000      # Bitcoin
    }
    
    for name in assets.keys():
        if name in base_prices:
            # Generate price series with realistic volatility
            returns = np.random.normal(0.0005, 0.02, len(date_range))  # ~0.05% daily return, 2% volatility
            prices = [base_prices[name]]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            asset_prices[name] = prices
    
    # Generate sentiment data (VIX-like)
    sentiment_data = np.random.normal(20, 5, len(date_range)).clip(10, 50)
    
    logger.info("get_data(): fallback data generated | prices_shape=%s sentiment_len=%s", asset_prices.shape, len(sentiment_data))
    return asset_prices, pd.Series(sentiment_data, index=date_range)

def get_data(assets: dict, sentiment_ticker: str, start_date: str, end_date: str) -> (pd.DataFrame, pd.Series):
    """
    Downloads historical closing prices for assets and a sentiment proxy.
    Falls back to sample data if yfinance fails.

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
    
    # Try yfinance first
    try:
        # Combine all tickers for a single download request
        all_tickers = list(assets.values()) + [sentiment_ticker]
        logger.debug("get_data(): attempting yfinance download | tickers=%s", all_tickers)
        
        raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
        
        if raw_data.empty:
            raise Exception("yfinance returned empty data")
            
        # Handle multi-level columns
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data = raw_data['Close']
        
        logger.debug("get_data(): raw_data columns=%s shape=%s", getattr(raw_data, 'columns', None), getattr(raw_data, 'shape', None))

        # --- ROBUST MAPPING LOGIC ---
        asset_prices = pd.DataFrame()
        for name, ticker in assets.items():
            if ticker in raw_data.columns:
                asset_prices[name] = raw_data[ticker]
            else:
                logger.warning("get_data(): missing downloaded column for asset | name=%s ticker=%s", name, ticker)

        if sentiment_ticker in raw_data.columns:
            sentiment_data = raw_data[sentiment_ticker]
        else:
            logger.warning("get_data(): missing sentiment data for ticker=%s", sentiment_ticker)
            sentiment_data = pd.Series()
        # ---------------------------

        # Drop any rows with missing data
        if not asset_prices.empty and not sentiment_data.empty:
            combined_data = asset_prices.join(sentiment_data).dropna()
            logger.debug("get_data(): combined_data shape after dropna=%s", getattr(combined_data, 'shape', None))

            # Separate the data back into assets and sentiment
            final_asset_prices = combined_data[list(assets.keys())]
            final_sentiment_data = combined_data[sentiment_ticker]

            logger.debug("get_data(): final_asset_prices head=\n%s", final_asset_prices.head().to_string())
            logger.info("get_data(): data download successful | prices_shape=%s sentiment_len=%s", final_asset_prices.shape, len(final_sentiment_data))
            return final_asset_prices, final_sentiment_data
        else:
            raise Exception("Insufficient data after processing")

    except Exception as e:
        logger.warning("get_data(): yfinance failed: %s, using fallback data", e)
        return _get_fallback_data(assets, sentiment_ticker, start_date, end_date)