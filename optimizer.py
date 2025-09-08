# optimizer.py
# Hardened optimizer for PyPortfolioOpt with dtype coercion, clipping,
# robust covariance fallback, and safe fallbacks. (User-provided, correct version)

import numpy as np
import pandas as pd
import logging
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
logger = logging.getLogger(__name__)

# Tuning
WINSOR_STD_MULT = 8.0
COV_REG_EPS = 1e-8
MAX_ANNUAL_RET = 10.0      # clip annual returns to ±1000%

def _force_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to float64, replace non-finite with NaN, drop NaN rows."""
    df_num = df.copy()
    # Force numeric coercion per column
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    # Replace inf with NaN and drop rows with any NaN
    df_num = df_num.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df_num.astype(np.float64)

def _winsorize(df: pd.DataFrame, mult=WINSOR_STD_MULT) -> pd.DataFrame:
    """Clip each column to mean +/- mult * std to limit extreme outliers."""
    df_w = df.copy()
    for c in df_w.columns:
        col = df_w[c]
        mu = col.mean()
        sigma = col.std()
        if not np.isfinite(sigma) or sigma == 0:
            low, high = mu - 1e-6, mu + 1e-6
        else:
            low, high = mu - mult * sigma, mu + mult * sigma
        df_w[c] = col.clip(lower=low, upper=high)
    return df_w

def _safe_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """Try Ledoit-Wolf shrinkage, else fallback to sample cov with regularization."""
    try:
        S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    except Exception as e:
        print("Warning: CovarianceShrinkage failed:", e)
        # fallback: sample covariance on cleaned data
        S = df.cov()
    
    S_vals = np.array(S, dtype=np.float64)
    if not np.isfinite(S_vals).all():
        S_vals = np.nan_to_num(S_vals, nan=0.0, posinf=0.0, neginf=0.0)

    diag = np.diag(S_vals)
    diag = np.where(np.isfinite(diag) & (diag > 0), diag, 1e-10)
    np.fill_diagonal(S_vals, diag + COV_REG_EPS)
    
    med_var = np.median(diag) if np.isfinite(np.median(diag)) else 1e-10
    S_vals += np.eye(S_vals.shape[0]) * (med_var * 1e-6 + COV_REG_EPS)
    return pd.DataFrame(S_vals, index=df.columns, columns=df.columns)

def get_optimal_portfolio(
    returns_df: pd.DataFrame,
    expected_returns: pd.Series,
    objective: str = 'Sharpe'
) -> (pd.DataFrame, tuple):
    """
    Calculates the optimal portfolio allocation using a hardened, fail-safe process.

    Args:
        returns_df (pd.DataFrame): DataFrame with daily asset returns.
        expected_returns (pd.Series): Series with the final forecasted daily returns.
        objective (str): The optimization objective ('Sharpe', 'MinRisk', or 'MaxRet').

    Returns:
        A tuple containing the optimal weights and the portfolio performance.
    """
    logger.info("get_optimal_portfolio(): start | objective=%s returns_shape=%s expret_index=%s", objective, getattr(returns_df, 'shape', None), list(expected_returns.index))

    # 1) Force numeric and drop non-finite rows
    df_num = _force_numeric(returns_df)
    if df_num.empty:
        print("Error: returns_df empty after coercion. Returning equal-weight fallback.")
        logger.error("get_optimal_portfolio(): returns_df empty after coercion, fallback equal-weight")
        n_assets = len(returns_df.columns)
        fallback = {a: 1.0 / n_assets for a in returns_df.columns}
        return pd.DataFrame.from_dict(fallback, orient='index', columns=['Weight']), (0,0,0)

    # 2) Winsorize to suppress extreme outliers that cause overflow
    df_clean = _winsorize(df_num)

    # 3) Compute robust covariance (with fallback)
    S = _safe_covariance(df_clean)

    # 4) Prepare expected returns -> align and annualize
    mu = expected_returns.reindex(df_clean.columns).fillna(0.0).astype(float)
    mu_annual = mu * 252.0
    mu_annual = mu_annual.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu_annual = mu_annual.clip(lower=-MAX_ANNUAL_RET, upper=MAX_ANNUAL_RET)

    # 5) Optimize
    try:
        ef = EfficientFrontier(mu_annual, S, weight_bounds=(0, 1))
        cols = list(df_clean.columns)

        if 'Bitcoin' in cols:
            btc_idx = cols.index('Bitcoin')
            if objective.lower() == 'sharpe':
                ef.add_constraint(lambda w: w[btc_idx] <= 0.20)
            elif objective.lower() in ('minrisk', 'min_risk'):
                ef.add_constraint(lambda w: w[btc_idx] <= 0.05)
            elif objective.lower() in ('maxret', 'max_ret', 'maxreturn'):
                ef.add_constraint(lambda w: w[btc_idx] <= 0.30)

        if objective.lower() == 'sharpe':
            ef.max_sharpe()
        elif objective.lower() in ('minrisk', 'min_risk', 'min_volatility', 'minvol'):
            ef.min_volatility()
        elif objective.lower() in ('maxret', 'max_ret', 'maxreturn'):
            # User preference: cap Bitcoin at 70%, allocate remainder to next highest expected return
            tickers = list(mu_annual.index)
            mu_sorted = mu_annual.sort_values(ascending=False)
            weights_map = {a: 0.0 for a in tickers}

            if len(mu_sorted) == 0:
                ef.max_sharpe()
            else:
                top_asset = mu_sorted.index[0]
                if top_asset == 'Bitcoin':
                    weights_map['Bitcoin'] = 0.70
                    # allocate remaining 30% to the next best non-Bitcoin asset
                    next_assets = [a for a in mu_sorted.index if a != 'Bitcoin']
                    if next_assets:
                        weights_map[next_assets[0]] = 0.30
                else:
                    # If Bitcoin isn't top, put 100% in the top asset
                    weights_map[top_asset] = 1.0

                ef.weights = np.array([weights_map.get(a, 0.0) for a in ef.tickers])
        
        cleaned = ef.clean_weights()
        weights_df = pd.DataFrame.from_dict(cleaned, orient='index', columns=['Weight'])
        performance = ef.portfolio_performance(verbose=False)
        
        logger.info("get_optimal_portfolio(): optimization successful | weights_keys=%s", list(cleaned.keys()))
        return weights_df, performance

    except Exception as e:
        logger.exception("get_optimal_portfolio(): optimization failed: %s", e)
        # Final fallback: equal weight
        logger.error("get_optimal_portfolio(): falling back to equal-weight allocation")
        n = len(returns_df.columns)
        ew = {a: 1.0 / n for a in returns_df.columns}
        return pd.DataFrame.from_dict(ew, orient='index', columns=['Weight']), (0,0,0)

