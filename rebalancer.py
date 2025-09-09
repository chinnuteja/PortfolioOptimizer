from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def _align_weights(current: pd.Series, target: pd.Series, asset_order: Optional[list] = None) -> Tuple[pd.Series, pd.Series]:
    cur = current.copy().astype(float).fillna(0.0)
    tar = target.copy().astype(float).fillna(0.0)

    idx = asset_order if asset_order else sorted(set(cur.index).union(tar.index))
    cur = cur.reindex(idx).fillna(0.0)
    tar = tar.reindex(idx).fillna(0.0)

    cur_sum = float(cur.sum())
    if cur_sum <= 0:
        cur[:] = 0.0
    else:
        cur = cur / cur_sum

    tar_sum = float(tar.sum())
    if tar_sum <= 0:
        raise ValueError("Target weights sum to zero.")
    tar = tar / tar_sum

    return cur, tar


def _apply_caps(weights: pd.Series, caps: Optional[Dict[str, float]]) -> pd.Series:
    if not caps:
        return weights
    w = weights.copy()
    for a, c in caps.items():
        if a in w.index and c is not None:
            w[a] = min(w[a], float(c))
    s = float(w.sum())
    if s > 0:
        w = w / s
    return w


def rebalance_with_controls(
    current_weights: pd.Series,
    target_weights: pd.Series,
    turnover_cap: float = 0.20,
    min_trade_band: float = 0.02,
    max_caps: Optional[Dict[str, float]] = None,
    asset_order: Optional[list] = None
) -> Dict:
    cur, tar = _align_weights(current_weights, target_weights, asset_order)

    raw_delta = tar - cur

    keep = raw_delta.abs() >= min_trade_band
    adj_delta = raw_delta.where(keep, 0.0)

    base_turnover = 0.5 * float(adj_delta.abs().sum())
    scale = 1.0
    if base_turnover > turnover_cap and base_turnover > 0:
        scale = turnover_cap / base_turnover
    adj_delta = adj_delta * scale

    proposed = cur + adj_delta
    proposed = proposed.clip(lower=0.0)
    proposed = _apply_caps(proposed, max_caps)

    total = float(proposed.sum())
    if total <= 0:
        proposed = tar.copy()
    else:
        proposed = proposed / total

    final_delta = proposed - cur
    turnover_used = 0.5 * float(final_delta.abs().sum())

    plan = {
        "current": (cur * 100).round(2).to_dict(),
        "target_raw": (tar * 100).round(2).to_dict(),
        "proposed": (proposed * 100).round(2).to_dict(),
        "trades_pct": (final_delta * 100).round(2).to_dict(),
        "turnover_used_pct": round(turnover_used * 100, 2),
        "params": {
            "turnover_cap_pct": round(turnover_cap * 100, 2),
            "min_trade_band_pct": round(min_trade_band * 100, 2),
            "max_caps_pct": {k: round(v * 100, 2) for k, v in (max_caps or {}).items()}
        }
    }
    return plan


