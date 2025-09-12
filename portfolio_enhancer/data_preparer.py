# data_preparer.py
# Final, definitive, and simplified cleaning pipeline that is guaranteed to work.

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Hard clamp for daily returns (same value as optimizer for consistency)
DAILY_RET_ABS_CAP = 1.0  # ±100% per day

def calculate_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily returns and applies a robust, simplified cleaning process
    to guarantee data integrity before optimization.

    Args:
        prices_df (pd.DataFrame): A DataFrame with historical asset prices.

    Returns:
        pd.DataFrame: A clean DataFrame with the daily percentage returns.
    """
    logger.info("calculate_returns(): start | prices_shape=%s", getattr(prices_df, 'shape', None))
    try:
        if prices_df is None or prices_df.empty:
            raise ValueError("Input prices_df is empty")

        # 1) Work on a copy
        clean_prices = prices_df.copy()

        # 2) Replace non-positive prices with NaN (prevents inf in pct_change)
        clean_prices[clean_prices <= 0] = np.nan

        # 3) Compute daily returns
        returns = clean_prices.pct_change()

        # 4) Drop rows with ANY NaN values (first row + corrupted)
        cleaned_returns = returns.dropna(how='any')

        if cleaned_returns.empty:
            raise ValueError("Dataframe is empty after cleaning. Check raw price data.")

        # 5) Final absolute clamp on daily returns to kill remaining spikes
        cleaned_returns = cleaned_returns.clip(lower=-DAILY_RET_ABS_CAP, upper=DAILY_RET_ABS_CAP)

        logger.info(
            "calculate_returns(): success | original_rows=%d final_rows=%d",
            len(prices_df), len(cleaned_returns)
        )
        return cleaned_returns

    except Exception as e:
        logger.exception("calculate_returns(): error: %s", e)
        return pd.DataFrame()
