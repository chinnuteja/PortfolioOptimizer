# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging, os, time, csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project modules
import config
import data_fetcher
import data_preparer
import forecaster
import optimizer
import rebalancer
from api_client import AssetSentimentAPI
from explainer_llm import explain_allocation

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

asset_returns = None
EXPORT_DIR = os.path.join(os.getcwd(), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Profile policies - single source of truth for risk-profile behavior
PROFILE_POLICY = {
    "MinRisk": {
        "anchor_strength": 0.60,   # pull targets toward user portfolio
        "turnover_cap_pct": 8.0,    # per rebalance
        "max_asset_cap_pct": 30.0,  # hard upper bound per asset
    },
    "Sharpe": {
        "anchor_strength": 0.30,
        "turnover_cap_pct": 20.0,
        "max_asset_cap_pct": 35.0,
    },
    "MaxRet": {
        "anchor_strength": 0.10,    # still give user some voice
        "turnover_cap_pct": 35.0,   # but not unlimited churn
        "max_asset_cap_pct": 40.0,  # **prevents 70/30 corner**
    },
}

# ---------- Init ----------
def initialize_data():
    global asset_returns
    logger.info("initialize_data(): start fetch+prepare")
    asset_prices, _ = data_fetcher.get_data(
        config.ASSETS, config.SENTIMENT_TICKER, config.START_DATE, config.END_DATE
    )
    if not asset_prices.empty:
        asset_returns = data_preparer.calculate_returns(asset_prices)
        if not asset_returns.empty:
            logger.info("initialize_data(): ready | returns_shape=%s", getattr(asset_returns, 'shape', None))
        else:
            logger.error("initialize_data(): could not prepare historical returns")
    else:
        logger.error("initialize_data(): could not download historical prices")

def project_to_caps_simplex(w: pd.Series, cap: pd.Series) -> pd.Series:
    """
    Project raw weights w onto the feasible set:
      w_i >= 0, w_i <= cap_i, sum w_i = 1.
    """
    idx = w.index
    capv = cap.reindex(idx).fillna(1.0).clip(lower=0.0, upper=1.0).values
    wv = np.maximum(0.0, np.minimum(w.reindex(idx).fillna(0.0).values, capv))

    # If sum <= 1 after clamping, we need to distribute the remainder across non-capped slots
    total = wv.sum()
    if abs(total - 1.0) < 1e-12:
        return pd.Series(wv, index=idx)

    if total > 1.0:
        # shrink proportionally on active (non-zero) coordinates until sum=1 respecting lower bounds (0)
        active = wv > 1e-12
        if active.any():
            wv[active] *= (1.0 / wv[active].sum())
            # enforce caps again (rare corner), then renormalize a second time
            wv = np.minimum(wv, capv)
            s = wv.sum()
            if s > 0: wv /= s
        else:
            # degenerate: all zero -> put mass uniformly within caps
            room = capv.sum()
            if room > 0:
                wv = capv / room
        return pd.Series(wv, index=idx)

    # total < 1.0: allocate remainder to names not at cap
    rem = 1.0 - total
    room = capv - wv
    room_pos = room > 1e-12
    if room_pos.any():
        add = np.zeros_like(wv)
        add[room_pos] = room[room_pos] / room[room_pos].sum() * rem
        wv = wv + add
    else:
        # no room (all at cap) -> minimal renorm
        wv /= (wv.sum() + 1e-12)

    return pd.Series(wv, index=idx)

def anchor_to_user(target_model: pd.Series, current_user: pd.Series, k: float) -> pd.Series:
    if current_user is None or current_user.sum() <= 1e-9:
        return target_model.copy()
    t = ((1.0 - k) * target_model + k * current_user).clip(lower=0.0)
    s = t.sum()
    return t if s <= 0 else t / s

def _normalize_profile(p: str) -> str:
    if not p: return "Sharpe"
    s = p.strip().lower()
    if "min" in s and "risk" in s: return "MinRisk"
    if "max" in s and "ret" in s:  return "MaxRet"
    return "Sharpe"


def _map_api_assets_to_config(sent_dict: dict) -> dict:
    if not isinstance(sent_dict, dict):
        return {}
    config_keys = list(config.ASSETS.keys())
    key_ci_map = {k.lower(): k for k in config_keys}
    out = {}
    for k, v in sent_dict.items():
        if not isinstance(k, str): continue
        lk = k.strip().lower()
        if lk in key_ci_map:
            out[key_ci_map[lk]] = float(v); continue
        synonyms = {
            "nifty50":"equities","nifty 50":"equities","nse":"equities","equities":"equities",
            "gold":"gold","bitcoin":"bitcoin","btc":"bitcoin","xbt":"bitcoin",
            "reit":"reits","reits":"reits"
        }
        mapped = synonyms.get(lk)
        if mapped and mapped in key_ci_map:
            out[key_ci_map[mapped]] = float(v)
    return out

def _weights_series_from_any(weights) -> pd.Series:
    if isinstance(weights, pd.DataFrame) and 'Weight' in weights.columns:
        return weights['Weight']
    return pd.Series(weights).squeeze()


def _write_proposal_csv(filename, assets, current_pct, target_pct, proposed_pct, trades_pct,
                        turnover_used_pct, sentiments, performance, policy, objective):
    path = os.path.join(EXPORT_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["portfolio_enhancer — Proposal Export"])
        w.writerow(["objective", objective])
        w.writerow(["turnover_used_pct", turnover_used_pct])
        w.writerow([])
        w.writerow(["policy"])
        for k,v in (policy or {}).items(): w.writerow([k, v])
        w.writerow([])
        w.writerow(["performance"])
        for k,v in (performance or {}).items(): w.writerow([k, v])
        w.writerow([])
        w.writerow(["sentiments"])
        for a in assets: w.writerow([a, sentiments.get(a, 0)])
        w.writerow([])
        w.writerow(["asset","current_%","target_%","proposed_%","trade_%"])
        for a in assets:
            w.writerow([
                a,
                current_pct.get(a, 0),
                target_pct.get(a, 0),
                proposed_pct.get(a, 0),
                trades_pct.get(a, 0),
            ])
    return path

# ---------- Routes ----------
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "assets_loaded": asset_returns is not None and not asset_returns.empty})

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(EXPORT_DIR, filename, as_attachment=True)

@app.route('/get_portfolio', methods=['POST'])
def get_portfolio():
    if asset_returns is None or asset_returns.empty:
        return jsonify({"error": "Historical data not loaded. Server is not ready."}), 500

    data = request.get_json(silent=True) or {}
    risk_profile_in = data.get('risk_profile', 'Sharpe')
    objective = _normalize_profile(risk_profile_in)
    export_csv = bool(data.get("export_csv", False))

    logger.info("get_portfolio(): request | profile_in=%s objective=%s", risk_profile_in, objective)

    # 1) Live sentiments (hosted API)
    try:
        api = AssetSentimentAPI()
        api_sent = api.analyze_and_get_sentiments(assets=list(config.ASSETS.keys()), wait_s=90)
        logger.info("get_portfolio(): raw_api_sent=%s", api_sent)
        asset_sentiment_scores = _map_api_assets_to_config(api_sent)
    except Exception as e:
        logger.warning("sentiment API issue: %s", e)
        asset_sentiment_scores = {}
    for k in config.ASSETS.keys():
        asset_sentiment_scores.setdefault(k, 0.0)

    # Heuristic fallback if everything is neutral/zero
    try:
        if not any(abs(float(v)) > 1e-6 for v in asset_sentiment_scores.values()):
            cols = list(config.ASSETS.keys())
            recent = asset_returns[cols].tail(60)
            mu = recent.mean()
            sigma = recent.std().replace(0, np.nan)
            z = (mu / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            z = z.clip(lower=-2.0, upper=2.0) * 0.1  # map roughly to [-0.2, 0.2]
            for a in cols:
                asset_sentiment_scores[a] = float(z.get(a, 0.0))
            logger.info("get_portfolio(): applied heuristic sentiment fallback | scores=%s", asset_sentiment_scores)
    except Exception as e:
        logger.warning("get_portfolio(): heuristic fallback failed: %s", e)

    # 2) Expected returns (sentiment tilt)
    expected_returns = forecaster.generate_forecasted_returns(asset_returns, asset_sentiment_scores)

    # 3) Optimize (model target)
    weights, performance = optimizer.get_optimal_portfolio(asset_returns, expected_returns, objective=objective)
    if weights.empty:
        return jsonify({"error": "Optimization failed."}), 500
    assets_order = list(config.ASSETS.keys())
    
    # 1) Extract raw model target
    target_raw = _weights_series_from_any(weights).reindex(assets_order).fillna(0.0)

    # 2) Profile policy
    pol = PROFILE_POLICY.get(objective, PROFILE_POLICY["Sharpe"])
    anchor_k   = pol["anchor_strength"]
    turnover_c = pol["turnover_cap_pct"]     # used in rebalancer
    cap_each   = pol["max_asset_cap_pct"]    # one cap for all assets for now

    # Optional: per-asset caps (e.g., stricter on BTC)
    # caps_dict = {"Bitcoin": 35.0, "Gold": 45.0, "Equities": 60.0, "REITs": 40.0}
    caps = pd.Series(cap_each/100.0, index=target_raw.index)
    # If you want custom per-asset caps, override:
    # for k,v in caps_dict.items(): caps[k] = v/100.0

    # 3) Project to caps
    target_capped = project_to_caps_simplex(target_raw, caps)

    # 4) Read user current weights (percent) and normalize
    current_weights_dict = data.get("current_weights", None)
    if current_weights_dict:
        cur = pd.Series(current_weights_dict, dtype=float).reindex(target_capped.index).fillna(0.0) / 100.0
        if cur.sum() > 0:
            cur = cur / cur.sum()
    else:
        cur = pd.Series(0.0, index=target_capped.index)

    # 5) Anchor to user (depends on risk profile)
    target_anchored = anchor_to_user(target_capped, cur, anchor_k)

    # 6) Rebalance plan with policy turnover/bands (and keep your fixed min band ±2%, cap 10% if you like)
    plan = rebalancer.rebalance_with_controls(
        current_weights=cur,
        target_weights=target_anchored,
        turnover_cap=turnover_c/100.0,
        min_trade_band=0.02,               # fixed ±2% as you wanted
        max_caps=None,                     # already enforced via target projection
        asset_order=assets_order
    )

    # Recompute performance using anchored & capped targets (what the user will actually hold)
    mu = expected_returns.reindex(target_anchored.index).fillna(0.0)
    cov = asset_returns.cov().reindex(index=mu.index, columns=mu.index).fillna(0.0)
    ret = float((mu @ target_anchored) * 252)
    vol = float(np.sqrt(target_anchored.T @ cov.values @ target_anchored) * np.sqrt(252))
    shp = ret/vol if vol>0 else 0

    performance_dict = {
        "Expected annual return": f"{ret*100:.2f}%",
        "Annual volatility": f"{vol*100:.2f}%",
        "Sharpe Ratio": f"{shp:.2f}"
    }
    sentiments_out = {k: round(float(v), 3) for k, v in asset_sentiment_scores.items()}

    response = {
        "weights_target_model_pct": (target_raw*100).round(2).to_dict(),
        "weights_target_capped_pct": (target_capped*100).round(2).to_dict(),
        "weights_target_anchored_pct": (target_anchored*100).round(2).to_dict(),
        "proposal": plan | {
            "policy": {
                "objective": objective,
                "anchor_strength": anchor_k,
                "turnover_cap_pct": turnover_c,
                "max_asset_cap_pct": cap_each
            }
        },
        "performance": performance_dict,
        "sentiments": sentiments_out
    }
    
    # Add narrative explanation
    response["narrative"] = explain_allocation(response)

    # Optional CSV export
    if export_csv:
        proposed_pct = plan.get("proposed", {})
        trades_pct   = plan.get("trades_pct", {})
        turnover_used_pct = float(plan.get("turnover_used_pct", 0))
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"proposal_{objective}_{ts}.csv"
        csv_path = _write_proposal_csv(
            filename=fname,
            assets=assets_order,
            current_pct=(cur*100).round(2).to_dict(),
            target_pct=(target_raw*100).round(2).to_dict(),
            proposed_pct=proposed_pct,
            trades_pct=trades_pct,
            turnover_used_pct=turnover_used_pct,
            sentiments=sentiments_out,
            performance=performance_dict,
            policy=pol,
            objective=objective
        )
        response["csv_filename"] = os.path.basename(csv_path)
        response["csv_download_url"] = f"/download/{os.path.basename(csv_path)}"

    return jsonify(response)

if __name__ == '__main__':
    initialize_data()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
