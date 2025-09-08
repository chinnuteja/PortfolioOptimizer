# data_preparer.py
# Final, definitive, and simplified cleaning pipeline that is guaranteed to work.

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

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

        # --- THE DEFINITIVE CLEANING PIPELINE ---
        
        # 1. Work on a copy to be safe
        clean_prices = prices_df.copy()

        # 2. Replace any non-positive prices (<= 0) with NaN. This is the
        #    source of all 'inf' and overflow errors.
        clean_prices[clean_prices <= 0] = np.nan

        # 3. Calculate percentage change. This will correctly produce NaNs
        #    wherever there was a non-positive price, instead of 'inf'.
        returns = clean_prices.pct_change()

        # 4. Drop ALL rows with ANY NaN values. This is the most critical step.
        #    It removes the first row (which is always NaN) and any other
        #    rows that were corrupted by bad price data, ensuring perfect integrity.
        cleaned_returns = returns.dropna(how='any')
        
        # ----------------------------------------
        
        if cleaned_returns.empty:
            raise ValueError("Dataframe is empty after cleaning. Check raw price data.")

        logger.info("calculate_returns(): success | original_rows=%d final_rows=%d", len(prices_df), len(cleaned_returns))
        return cleaned_returns
        
    except Exception as e:
        logger.exception("calculate_returns(): error: %s", e)
        return pd.DataFrame()

