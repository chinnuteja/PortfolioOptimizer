# data_preparer.py
# Robust & simplified cleaning pipeline for daily returns.

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Hard clamp for daily returns (same value as optimizer for consistency)
DAILY_RET_ABS_CAP = 1.0  # ±100% per day

def calculate_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily returns and applies a robust cleaning process.
    Ensures finite, dense, bounded returns for optimization.
    """
    logger.info("calculate_returns(): start | prices_shape=%s", getattr(prices_df, 'shape', None))
    try:
        if prices_df is None or prices_df.empty:
            raise ValueError("Input prices_df is empty")

        # Work on a copy
        clean_prices = prices_df.copy()

        # Replace non-positive prices (<=0) with NaN (prevents inf in pct_change)
        clean_prices[clean_prices <= 0] = np.nan

        # Compute daily pct returns
        returns = clean_prices.pct_change()

        # Replace resid ±inf with NaN, enforce float64
        returns = returns.replace([np.inf, -np.inf], np.nan).astype("float64")

        # Drop rows that are all-NaN
        returns = returns.dropna(how='all')

        # Lightly repair isolated holes (1-day gaps)
        returns = returns.ffill(limit=1).bfill(limit=1)


        # Drop any remaining NaN rows
        cleaned_returns = returns.dropna(how='any')

        if cleaned_returns.empty:
            raise ValueError("Dataframe is empty after cleaning. Check raw price data.")

        # Final clamp on daily returns
        cleaned_returns = cleaned_returns.clip(lower=-DAILY_RET_ABS_CAP, upper=DAILY_RET_ABS_CAP)

        logger.info(
            "calculate_returns(): success | original_rows=%d final_rows=%d",
            len(prices_df), len(cleaned_returns)
        )
        return cleaned_returns

    except Exception as e:
        logger.exception("calculate_returns(): error: %s", e)
        return pd.DataFrame()
