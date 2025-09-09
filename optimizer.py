# optimizer.py
# Hardened optimizer for PyPortfolioOpt with proper annualization,
# diversified MaxRet objective (stable), and robust fallbacks.

import numpy as np
import pandas as pd
import logging
from typing import Tuple
from pypfopt import EfficientFrontier, risk_models
from pypfopt.exceptions import OptimizationError

logger = logging.getLogger(__name__)

# --- Tunables / guards ---
WINSOR_STD_MULT = 8.0
COV_REG_EPS = 1e-8
MAX_ANNUAL_RET = 4.0        # cap annual expected returns to ±400%
DEFAULT_BTC_SOFT_CAP = 0.35 # gentle BTC cap inside EF (profiles enforce later too)

# ----------------- Cleaning helpers -----------------
def _force_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to float, drop non-finite rows (any NaN/inf in a row)."""
    df_num = df.copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    df_num = df_num.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df_num.astype(np.float64)

def _winsorize(df: pd.DataFrame, mult=WINSOR_STD_MULT) -> pd.DataFrame:
    """Clip each column to mean ± mult*std to suppress extreme outliers."""
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

def _safe_covariance_annual(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit–Wolf shrinkage on daily returns (fallback to sample cov), then
    add a tiny ridge and annualize by 252.
    """
    try:
        S_daily = risk_models.CovarianceShrinkage(df_daily).ledoit_wolf()
    except Exception as e:
        logger.warning("CovarianceShrinkage failed: %s; using sample cov", e)
        S_daily = df_daily.cov()

    S = np.array(S_daily, dtype=np.float64)
    if not np.isfinite(S).all():
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    # enforce positive diagonal
    diag = np.diag(S)
    diag = np.where(np.isfinite(diag) & (diag > 0), diag, 1e-10)
    np.fill_diagonal(S, diag + COV_REG_EPS)

    # tiny ridge for numerical stability
    med_var = np.median(diag) if np.isfinite(np.median(diag)) else 1e-10
    S += np.eye(S.shape[0]) * (med_var * 1e-6 + COV_REG_EPS)

    # annualize
    S_annual = S * 252.0
    return pd.DataFrame(S_annual, index=df_daily.columns, columns=df_daily.columns)

# ----------------- Core optimizer -----------------
def get_optimal_portfolio(
    returns_df: pd.DataFrame,
    expected_returns_daily: pd.Series,
    objective: str = "Sharpe",
) -> Tuple[pd.DataFrame, tuple]:
    """
    Compute optimal weights and portfolio performance.

    Inputs:
      - returns_df: DAILY returns DataFrame
      - expected_returns_daily: DAILY expected returns with sentiment tilt
      - objective: 'Sharpe' | 'MinRisk' | 'MaxRet'

    Returns:
      (weights_df, (ann_return, ann_vol, sharpe))
      weights_df has a single column 'Weight'
    """
    logger.info("get_optimal_portfolio(): objective=%s", objective)

    # 1) Clean & winsorize
    df_num = _force_numeric(returns_df)
    if df_num.empty:
        logger.error("returns_df empty after cleaning; equal-weight fallback")
        n = len(returns_df.columns)
        ew = {a: 1.0 / n for a in returns_df.columns}
        return pd.DataFrame.from_dict(ew, orient="index", columns=["Weight"]), (0, 0, 0)

    df_clean = _winsorize(df_num)

    # 2) Annualized covariance
    S_ann = _safe_covariance_annual(df_clean)

    # 3) Annualized expected returns (clip to keep optimizer sane)
    mu_daily = expected_returns_daily.reindex(df_clean.columns).fillna(0.0).astype(float)
    mu_ann = (mu_daily * 252.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu_ann = mu_ann.clip(lower=-MAX_ANNUAL_RET, upper=MAX_ANNUAL_RET)

    try:
        ef = EfficientFrontier(mu_ann, S_ann, weight_bounds=(0, 1))

        # Gentle BTC cap to prevent corner solutions (profiles/caps are applied later too)
        cols = list(df_clean.columns)
        if "Bitcoin" in cols:
            btc_idx = cols.index("Bitcoin")
            ef.add_constraint(lambda w: w[btc_idx] <= DEFAULT_BTC_SOFT_CAP)

        obj = (objective or "Sharpe").strip().lower()
        if obj == "sharpe":
            ef.max_sharpe()

        elif obj in ("minrisk", "min_risk", "minvol", "min_volatility"):
            ef.min_volatility()

        elif obj in ("maxret", "max_ret", "maxreturn"):
            # Stable aggressive target: small positive risk aversion
            # keeps the problem well-posed and still return-seeking.
            try:
                ef.max_quadratic_utility(risk_aversion=0.05)  # 0.03–0.10 works well
            except Exception:
                # Fallback: aim for a high feasible return on the frontier
                try:
                    target = float(min(mu_ann.max() * 0.95, mu_ann.mean() + 2.0 * mu_ann.std()))
                    ef.efficient_return(target_return=target)
                except Exception:
                    ef.max_sharpe()
        else:
            ef.max_sharpe()

        cleaned = ef.clean_weights()
        weights_df = pd.DataFrame.from_dict(cleaned, orient="index", columns=["Weight"])

        # EF performance is annual (inputs were annual)
        ann_ret, ann_vol, sharpe = ef.portfolio_performance(verbose=False)
        return weights_df, (ann_ret, ann_vol, sharpe)

    except Exception as e:
        logger.exception("Optimization failed: %s", e)
        n = len(returns_df.columns)
        ew = {a: 1.0 / n for a in returns_df.columns}
        return pd.DataFrame.from_dict(ew, orient="index", columns=["Weight"]), (0, 0, 0)

# ----------------- Slider (Custom) support -----------------
def _to_series(w):
    if isinstance(w, pd.DataFrame) and "Weight" in w.columns:
        return w["Weight"]
    if isinstance(w, (pd.Series, np.ndarray, list)):
        return pd.Series(w, index=w.index if hasattr(w, "index") else None, dtype=float)
    return w.squeeze()

def _blend(w_a: pd.Series, w_b: pd.Series, t: float) -> pd.Series:
    """Linear blend of two weight vectors with renorm to sum=1."""
    t = max(0.0, min(1.0, float(t)))
    w = (1.0 - t) * w_a + t * w_b
    w = w.clip(lower=0.0)
    s = float(w.sum())
    return w / s if s > 0 else w

def _compute_performance(asset_returns_daily: pd.DataFrame,
                         expected_returns_daily: pd.Series,
                         weights: pd.Series):
    """Recompute annualized performance from blended weights."""
    mu_d = expected_returns_daily.reindex(weights.index).fillna(0.0)
    Sigma_d = asset_returns_daily.cov().reindex(index=weights.index, columns=weights.index).fillna(0.0)

    w = weights.values.reshape(-1, 1)
    mu_vec = mu_d.values.reshape(-1, 1)

    exp_daily = float((w.T @ mu_vec).item())
    var_daily = float((w.T @ Sigma_d.values @ w).item())
    vol_daily = np.sqrt(max(var_daily, 0.0))

    exp_ann = 252.0 * exp_daily
    vol_ann = np.sqrt(252.0) * vol_daily
    sharpe = (exp_ann / vol_ann) if vol_ann > 1e-12 else 0.0
    return exp_ann, vol_ann, sharpe

def get_portfolio_by_slider(asset_returns: pd.DataFrame,
                            expected_returns: pd.Series,
                            slider: float):
    """
    Build three anchor portfolios (MinRisk/Sharpe/MaxRet), then linearly
    blend weights according to slider 0..100 and recompute performance.
    """
    w_min, _ = get_optimal_portfolio(asset_returns, expected_returns, objective="MinRisk")
    w_shp, _ = get_optimal_portfolio(asset_returns, expected_returns, objective="Sharpe")
    w_max, _ = get_optimal_portfolio(asset_returns, expected_returns, objective="MaxRet")

    ws_min = _to_series(w_min)
    ws_shp = _to_series(w_shp)
    ws_max = _to_series(w_max)

    s = max(0.0, min(100.0, float(slider)))
    if s <= 50.0:
        t = s / 50.0
        ws = _blend(ws_min, ws_shp, t)
    else:
        t = (s - 50.0) / 50.0
        ws = _blend(ws_shp, ws_max, t)

    weights_df = pd.DataFrame({"Weight": ws})
    ann_ret, ann_vol, sharpe = _compute_performance(asset_returns, expected_returns, ws)
    return weights_df, (ann_ret, ann_vol, sharpe)
