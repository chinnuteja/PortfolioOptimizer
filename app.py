from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging, os, time, csv, requests
from dotenv import load_dotenv

import config
import data_fetcher
import data_preparer
import forecaster
import optimizer
import rebalancer
from api_client import AssetSentimentAPI

load_dotenv()
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

asset_returns = None
EXPORT_DIR = os.path.join(os.getcwd(), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

LATEST_KPIS_JOB_ID = None
EXPLAINER_URL = "https://portfolioexplain-production.up.railway.app/generate-portfolio-explanation"

PROFILE_POLICY = {
    "MinRisk": {"anchor_strength": 0.60, "turnover_cap_pct": 8.0,  "max_asset_cap_pct": 30.0},
    "Sharpe":  {"anchor_strength": 0.30, "turnover_cap_pct": 20.0, "max_asset_cap_pct": 35.0},
    "MaxRet":  {"anchor_strength": 0.10, "turnover_cap_pct": 35.0, "max_asset_cap_pct": 40.0},
}

def initialize_data():
    global asset_returns
    logger.info("initialize_data(): fetching historical data…")
    prices, _ = data_fetcher.get_data(
        config.ASSETS, config.SENTIMENT_TICKER, config.START_DATE, config.END_DATE
    )
    if prices.empty:
        logger.error("initialize_data(): could not download historical prices")
        return
    asset_returns = data_preparer.calculate_returns(prices)
    if asset_returns.empty:
        logger.error("initialize_data(): could not compute returns")
    else:
        logger.info("initialize_data(): returns ready %s", getattr(asset_returns, "shape", None))

def _normalize_profile(p: str) -> str:
    if not p: return "Sharpe"
    s = p.strip().lower()
    if "min" in s and "risk" in s: return "MinRisk"
    if "max" in s and "ret" in s:  return "MaxRet"
    return "Sharpe"

def project_to_caps_simplex(w: pd.Series, cap: pd.Series) -> pd.Series:
    idx = w.index
    capv = cap.reindex(idx).fillna(1.0).clip(0.0, 1.0).values
    wv   = np.maximum(0.0, np.minimum(w.reindex(idx).fillna(0.0).values, capv))
    total = float(wv.sum())
    if abs(total - 1.0) < 1e-12:
        return pd.Series(wv, index=idx)
    if total > 1.0:
        active = wv > 1e-12
        if active.any():
            wv[active] *= (1.0 / wv[active].sum())
            wv = np.minimum(wv, capv)
            s = wv.sum()
            if s > 0: wv /= s
        else:
            room = capv.sum()
            if room > 0: wv = capv / room
        return pd.Series(wv, index=idx)
    # total < 1.0
    rem = 1.0 - total
    room = capv - wv
    mask = room > 1e-12
    if mask.any():
        add = np.zeros_like(wv)
        add[mask] = room[mask] / room[mask].sum() * rem
        wv = wv + add
    else:
        s = wv.sum()
        if s > 0: wv /= s
    return pd.Series(wv, index=idx)

def anchor_to_user(target_model: pd.Series, current_user: pd.Series, k: float) -> pd.Series:
    if current_user is None or current_user.sum() <= 1e-9:
        return target_model.copy()
    t = ((1.0 - k) * target_model + k * current_user).clip(lower=0.0)
    s = t.sum()
    return t if s <= 0 else t / s

def _weights_series_from_any(weights) -> pd.Series:
    if isinstance(weights, pd.DataFrame) and "Weight" in weights.columns:
        return weights["Weight"]
    return pd.Series(weights).squeeze()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({"status":"ok", "assets_loaded": asset_returns is not None and not asset_returns.empty})

@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(EXPORT_DIR, filename, as_attachment=True)

@app.route('/get_portfolio', methods=['POST'])
def get_portfolio():
    """
    Non-blocking: fire KPI job, return immediately with job_id, run optimization with
    quick/no sentiments. UI can click Explain later using the same job_id.
    """
    if asset_returns is None or asset_returns.empty:
        return jsonify({"error":"Historical data not loaded."}), 500

    data = request.get_json(silent=True) or {}
    risk_profile_in = data.get("risk_profile", "Sharpe")
    objective = _normalize_profile(risk_profile_in)
    risk_slider = data.get("risk_slider", None)
    export_csv  = bool(data.get("export_csv", False))
    disabled = set(data.get("disabled_assets", []))
    assets_all = list(config.ASSETS.keys())
    investable = [a for a in assets_all if a not in disabled]
    if not investable:
        return jsonify({"error":"All assets deselected. Enable at least one."}), 400

    # 1) Start KPI job (fast) and try a single quick sentiments fetch (non-blocking)
    job_id = None
    sentiments = {}
    try:
        api = AssetSentimentAPI()
        job_id = api.start_analysis(assets=None, timeout_minutes=15)
        logger.info("get_portfolio(): started KPI job_id=%s", job_id)
        if job_id:
            quick = api.try_get_sentiments_once(job_id)  # returns {} if not ready yet
            # map tickers
            map_tbl = {"NIFTY50":"Equities","GOLD":"Gold","BITCOIN":"Bitcoin","REIT":"REITs"}
            sentiments = { map_tbl.get(k,k): float(v) for k,v in (quick or {}).items() }
    except Exception as e:
        logger.warning("KPI start/quick-fetch failed: %s", e)
        sentiments = {}

    # fill neutrals for any missing
    for a in assets_all:
        sentiments.setdefault(a, 0.0)

    global LATEST_KPIS_JOB_ID
    if job_id: LATEST_KPIS_JOB_ID = job_id

    # 2) Expected returns (sentiment tilt) — on investable
    local_returns = asset_returns[investable].copy()
    expected_returns = forecaster.generate_forecasted_returns(
        local_returns,
        {a: sentiments.get(a, 0.0) for a in investable}
    )

    # 3) Optimize
    if risk_slider is not None:
        try:
            weights, performance = optimizer.get_portfolio_by_slider(local_returns, expected_returns, float(risk_slider))
            slider_used = int(round(max(0, min(100, float(risk_slider)))))
            profile_used = "Balanced" if slider_used == 50 else "Custom"
        except Exception as e:
            logger.warning("slider fallback: %s", e)
            weights, performance = optimizer.get_optimal_portfolio(local_returns, expected_returns, objective=objective)
            slider_used = {"MinRisk":0,"Sharpe":50,"MaxRet":100}.get(objective,50)
            profile_used = "Balanced" if objective=="Sharpe" else ("Conservative" if objective=="MinRisk" else "Aggressive")
    else:
        weights, performance = optimizer.get_optimal_portfolio(local_returns, expected_returns, objective=objective)
        slider_used = {"MinRisk":0,"Sharpe":50,"MaxRet":100}.get(objective,50)
        profile_used = "Balanced" if objective=="Sharpe" else ("Conservative" if objective=="MinRisk" else "Aggressive")

    if weights.empty:
        return jsonify({"error":"Optimization failed."}), 500

    # 4) Cap + anchor
    tar_local = _weights_series_from_any(weights)
    target_raw = pd.Series(0.0, index=assets_all, dtype=float)
    target_raw.loc[tar_local.index] = tar_local.values
    for a in disabled: target_raw[a] = 0.0

    pol = PROFILE_POLICY.get(objective, PROFILE_POLICY["Sharpe"])
    caps = pd.Series(pol["max_asset_cap_pct"]/100.0, index=assets_all)
    for a in disabled: caps[a] = 0.0
    target_capped = project_to_caps_simplex(target_raw, caps)

    cur_dict = data.get("current_weights", {}) or {}
    cur = pd.Series(cur_dict, dtype=float).reindex(assets_all).fillna(0.0) / 100.0
    if cur.sum() > 0: cur = cur / cur.sum()
    target_anchored = anchor_to_user(target_capped, cur, pol["anchor_strength"])

    plan = rebalancer.rebalance_with_controls(
        current_weights=cur,
        target_weights=target_anchored,
        turnover_cap=pol["turnover_cap_pct"]/100.0,
        min_trade_band=0.02,
        max_caps=None,
        asset_order=assets_all
    )

    # 5) Performance (anchored)
    mu = expected_returns.reindex(target_anchored.index).fillna(0.0)
    cov = asset_returns.cov().reindex(index=mu.index, columns=mu.index).fillna(0.0)
    ret = float((mu @ target_anchored) * 252.0)
    vol = float(np.sqrt(target_anchored.T @ cov.values @ target_anchored) * np.sqrt(252.0))
    shp = ret/vol if vol>0 else 0.0

    perf = {
        "Expected annual return": f"{ret*100:.2f}%",
        "Annual volatility": f"{vol*100:.2f}%",
        "Sharpe Ratio": f"{shp:.2f}"
    }

    return jsonify({
        "weights_target_model_pct":    (target_raw*100).round(2).to_dict(),
        "weights_target_capped_pct":   (target_capped*100).round(2).to_dict(),
        "weights_target_anchored_pct": (target_anchored*100).round(2).to_dict(),
        "proposal": plan | {"policy": pol | {"objective": objective}},
        "performance": perf,
        "sentiments": {k: round(float(v),3) for k,v in sentiments.items()},
        "risk_profile_used": profile_used,
        "risk_slider_used": slider_used,
        "investable_assets": investable,
        "disabled_assets": list(disabled),
        "kpis_job_id": job_id or LATEST_KPIS_JOB_ID  # UI uses this for /explain
    })

@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id") or data.get("kpis_job_id") or LATEST_KPIS_JOB_ID

    def _to_pct_map(key_list):
        for k in key_list:
            m = data.get(k)
            if isinstance(m, dict):
                try: return {a: float(m.get(a, 0.0)) for a in ["Gold","Equities","REITs","Bitcoin"]}
                except Exception: pass
        return {"Gold":0.0,"Equities":0.0,"REITs":0.0,"Bitcoin":0.0}

    current_portfolio   = _to_pct_map(["current_portfolio","current_weights"])
    optimized_portfolio = _to_pct_map(["optimized_portfolio","final_weights"])

    rp = (data.get("risk_profile_used") or data.get("risk_profile") or "Balanced").lower()
    if "min" in rp or "cons" in rp:   risk_profile = "Conservative"
    elif "max" in rp or "agg" in rp:  risk_profile = "Aggressive"
    else:                             risk_profile = "Balanced"

    if not job_id:
        return jsonify({"status":"error","error":"missing_job_id",
                        "message":"Metrics job is still starting. Click Get Recommendation again, then Explain."}), 200

    payload = {
        "job_id": job_id,
        "current_portfolio": current_portfolio,
        "optimized_portfolio": optimized_portfolio,
        "risk_profile": risk_profile
    }
    logger.info("explain(): POST %s | payload=%s", EXPLAINER_URL, payload)

    try:
        r = requests.post(EXPLAINER_URL, json=payload, timeout=60)
        if r.status_code != 200:
            return jsonify({"status":"error","error":f"explainer_http_{r.status_code}",
                            "message": r.text}), 200
        return jsonify(r.json())
    except Exception as e:
        logger.exception("explain(): relay failed")
        return jsonify({"status":"error","error":"relay_failed"}), 200

if __name__ == "__main__":
    initialize_data()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
