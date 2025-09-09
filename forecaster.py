# forecaster.py
# Final, definitive, and hardened version that sanitizes its own outputs.

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_forecasted_returns(returns_df: pd.DataFrame, asset_sentiment_scores: dict) -> pd.Series:
    """
    Generates sentiment-adjusted forecasts and ensures the final output is
    mathematically clean and stable before returning.

    Args:
        returns_df (pd.DataFrame): DataFrame of historical daily returns.
        asset_sentiment_scores (dict): A dictionary mapping asset names to their live sentiment scores.

    Returns:
        pd.Series: A clean, stable Series of daily expected returns.
    """
    logger.info("generate_forecasted_returns(): start | returns_shape=%s", getattr(returns_df, 'shape', None))
    if returns_df is None or returns_df.empty:
        raise ValueError("returns_df is empty")

    # Start with the historical mean daily returns
    base_means = returns_df.mean()
    forecasts = base_means.copy()

    # Apply asset-specific sentiment adjustments
    for asset, sentiment_score in (asset_sentiment_scores or {}).items():
        if asset not in forecasts.index:
            logger.warning("generate_forecasted_returns(): skipping unknown asset key | asset=%s", asset)
            continue
        try:
            s_clipped = np.clip(float(sentiment_score), -0.2, 0.2)
        except Exception:
            s_clipped = 0.0
        factor = 1.0 + s_clipped
        forecasts.loc[asset] *= factor
        logger.debug("generate_forecasted_returns(): adjust | asset=%s score=%s factor=%.4f", asset, sentiment_score, factor)
            
    # --- THE DEFINITIVE FIX: Sanitize the final forecast ---
    # 1. Replace any potential non-finite numbers with zero.
    forecasts.replace([np.inf, -np.inf], np.nan, inplace=True)
    forecasts.fillna(0.0, inplace=True)
    
    # 2. Clip the final forecast to a reasonable range to prevent overflow.
    #    (e.g., max expected daily return of 50%)
    final_forecasts = forecasts.clip(lower=-0.5, upper=0.5)
    # --------------------------------------------------------

    logger.info("generate_forecasted_returns(): success | forecasts_index=%s", list(forecasts.index))
    return final_forecasts

