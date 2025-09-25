# app.py — repo-level /static, robust caching + SWR + safe math + yfinance fallback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging, os, time, csv, requests, threading, math
from datetime import datetime, timezone
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# project modules
from portfolio_enhancer import (
    config, data_fetcher, data_preparer, forecaster, optimizer, rebalancer
)
from portfolio_enhancer.api_client import AssetSentimentAPI

load_dotenv()

# ---------- paths & folders ----------
REPO_ROOT = os.getcwd()                               # On Render: /opt/render/project/src
STATIC_DIR = os.path.join(REPO_ROOT, "static")        # <— repo-level /static
BASE_DIR   = REPO_ROOT

EXPORT_DIR = os.path.join(BASE_DIR, "exports")
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

RETURNS_CACHE_PARQUET = os.path.join(CACHE_DIR, "asset_returns.parquet")
RETURNS_CACHE_CSV     = os.path.join(CACHE_DIR, "asset_returns.csv")

# Create Flask with repo-level /static
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)
logger.info("STATIC FOLDER = %s", app.static_folder)

asset_returns = None

# --- refresh policy (hourly by default) ---
RETURNS_MAX_AGE_SECONDS     = int(os.environ.get("RETURNS_MAX_AGE_SECONDS", "3600"))  # 1 hour
STALE_WHILE_REVALIDATE      = os.environ.get("STALE_WHILE_REVALIDATE", "1") in ("1", "true", "True")
FORCE_END_TODAY             = os.environ.get("FORCE_END_TODAY", "1") in ("1", "true", "True")
RETURNS_HARD_MAX_AGE_SECONDS= int(os.environ.get("RETURNS_HARD_MAX_AGE_SECONDS", "0"))  # 0=disabled

_RETURNS_LOCK = threading.Lock()
INIT_IN_PROGRESS = False

# KPI throttling (reuse job for 10 min)
KPI_JOB_TTL_SECONDS = int(os.environ.get("KPI_JOB_TTL_SECONDS", "600"))
LATEST_KPIS_JOB_ID = None
LATEST_KPIS_JOB_STARTED_AT = 0.0

# ---- Explainer config (env overrideable) ----
EXPLAINER_URL = os.environ.get(
    "EXPLAINER_URL",
    "https://portfolioexplain-production.up.railway.app/generate-portfolio-explanation"
)
EXPLAINER_CONNECT_TIMEOUT = float(os.environ.get("EXPLAINER_CONNECT_TIMEOUT", "10"))
EXPLAINER_READ_TIMEOUT    = float(os.environ.get("EXPLAINER_READ_TIMEOUT", "120"))
EXPLAINER_RETRIES         = int(os.environ.get("EXPLAINER_RETRIES", "3"))
EXPLAINER_BACKOFF         = float(os.environ.get("EXPLAINER_BACKOFF", "0.8"))

PROFILE_POLICY = {
    "MinRisk": {"anchor_strength": 0.60, "turnover_cap_pct": 8.0,  "max_asset_cap_pct": 30.0},
    "Sharpe":  {"anchor_strength": 0.30, "turnover_cap_pct": 20.0, "max_asset_cap_pct": 35.0},
    "MaxRet":  {"anchor_strength": 0.10, "turnover_cap_pct": 35.0, "max_asset_cap_pct": 40.0},
}


# ---------- helpers ----------

def _requests_session_with_retry(
    total=EXPLAINER_RETRIES,
    backoff_factor=EXPLAINER_BACKOFF,
    status_forcelist=(429, 500, 502, 503, 504),
):
    s = requests.Session()
    retries = Retry(
        total=total, read=total, connect=total, status=total,
        backoff_factor=backoff_factor, status_forcelist=status_forcelist,
        allowed_methods=["POST", "GET"], raise_on_status=False, respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
    return s

def _load_returns_cache() -> pd.DataFrame:
    # Try parquet first (if pyarrow/fastparquet installed)
    if os.path.exists(RETURNS_CACHE_PARQUET):
        try:
            df = pd.read_parquet(RETURNS_CACHE_PARQUET)
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info("initialize_data(): loaded returns from parquet cache %s", df.shape)
                return df
        except Exception as e:
            logger.warning("initialize_data(): parquet cache read failed: %s", e)
    # Fallback to CSV
    if os.path.exists(RETURNS_CACHE_CSV):
        try:
            df = pd.read_csv(RETURNS_CACHE_CSV, index_col=0, parse_dates=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info("initialize_data(): loaded returns from csv cache %s", df.shape)
                return df
        except Exception as e:
            logger.warning("initialize_data(): csv cache read failed: %s", e)
    return pd.DataFrame()

def _save_returns_cache(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        df.to_parquet(RETURNS_CACHE_PARQUET, index=True)
    except Exception as e:
        logger.warning("cache parquet write failed: %s", e)
    try:
        df.to_csv(RETURNS_CACHE_CSV, index=True)
    except Exception as e:
        logger.warning("cache csv write failed: %s", e)

def _get_prices_with_retries(assets, sentiment_ticker, start_date, end_date, attempts=3, backoff=1.5):
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            return data_fetcher.get_data(assets, sentiment_ticker, start_date, end_date)
        except Exception as e:
            last_exc = e
            logger.warning("get_data retry %d/%d failed: %s", i, attempts, e)
            time.sleep(backoff ** i)
    logger.error("get_data retries exhausted: %s; using internal fallback", last_exc)
    return data_fetcher._get_fallback_data(assets, sentiment_ticker, start_date, end_date)

def _rolling_dates():
    """Use END_DATE=today (UTC) if FORCE_END_TODAY or config.END_DATE missing."""
    start_date = config.START_DATE
    if FORCE_END_TODAY or not getattr(config, "END_DATE", None):
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        end_date = config.END_DATE
    return start_date, end_date

def _cache_age_seconds() -> float:
    """Age of the newest cache file; inf if no cache exists."""
    path = RETURNS_CACHE_PARQUET if os.path.exists(RETURNS_CACHE_PARQUET) else RETURNS_CACHE_CSV
    if not path or not os.path.exists(path):
        return float("inf")
    try:
        mtime = os.path.getmtime(path)
        return max(0.0, time.time() - mtime)
    except Exception:
        return float("inf")

def _refresh_returns_blocking() -> bool:
    """Recompute returns from live data, update global + caches."""
    global asset_returns
    with _RETURNS_LOCK:
        t0 = time.time()
        try:
            start_date, end_date = _rolling_dates()
            logger.info("refresh_returns(): START  range=%s→%s  cache_age=%.0fs",
                        start_date, end_date, _cache_age_seconds())
            prices, _ = _get_prices_with_retries(
                config.ASSETS, config.SENTIMENT_TICKER, start_date, end_date,
                attempts=3, backoff=1.5
            )
            if prices.empty:
                logger.error("refresh_returns(): download empty; keeping existing cache (took %.1fs)", time.time()-t0)
                return False

            new_returns = data_preparer.calculate_returns(prices)
            if new_returns.empty:
                logger.error("refresh_returns(): cleaned returns empty; keeping existing cache (took %.1fs)", time.time()-t0)
                return False

            asset_returns = new_returns
            _save_returns_cache(asset_returns)
            logger.info("refresh_returns(): DONE   shape=%s  latest_date=%s  elapsed=%.1fs",
                        getattr(asset_returns, "shape", None),
                        str(asset_returns.index.max()) if hasattr(asset_returns, "index") else "n/a",
                        time.time()-t0)
            return True
        except Exception as e:
            logger.exception("refresh_returns(): FAILED after %.1fs: %s", time.time()-t0, e)
            return False

def maybe_refresh_returns_if_stale(blocking: bool = False):
    """Refresh policy:
       - If age < RETURNS_MAX_AGE_SECONDS: do nothing.
       - If HARD_MAX_AGE_SECONDS > 0 and age >= HARD: block refresh.
       - Else: refresh per blocking flag (SWR if False)."""
    age = _cache_age_seconds()
    if age < RETURNS_MAX_AGE_SECONDS:
        return
    if RETURNS_HARD_MAX_AGE_SECONDS > 0 and age >= RETURNS_HARD_MAX_AGE_SECONDS:
        _refresh_returns_blocking()
        return
    if blocking:
        _refresh_returns_blocking()
    else:
        threading.Thread(target=_refresh_returns_blocking, daemon=True).start()

def initialize_data():
    """Load cached returns or compute them once at startup."""
    global asset_returns, INIT_IN_PROGRESS
    if INIT_IN_PROGRESS:
        return
    INIT_IN_PROGRESS = True
    try:
        # 1) Try cache first
        cached = _load_returns_cache()
        if not cached.empty:
            asset_returns = cached
            if STALE_WHILE_REVALIDATE:
                maybe_refresh_returns_if_stale(blocking=False)
            else:
                maybe_refresh_returns_if_stale(blocking=True)
            logger.info("initialize_data(): latest date in returns = %s", str(asset_returns.index.max()))
            return

        # 2) No cache: fetch and build now
        logger.info("initialize_data(): fetching historical data…")
        start_date, end_date = _rolling_dates()
        prices, _ = _get_prices_with_retries(
            config.ASSETS, config.SENTIMENT_TICKER, start_date, end_date,
            attempts=3, backoff=1.5
        )
        if prices.empty:
            logger.error("initialize_data(): could not download historical prices")
            asset_returns = pd.DataFrame()
            return

        # 3) Compute returns
        asset_returns_local = data_preparer.calculate_returns(prices)
        if asset_returns_local.empty:
            logger.error("initialize_data(): could not compute returns")
            asset_returns = pd.DataFrame()
            return

        asset_returns = asset_returns_local
        _save_returns_cache(asset_returns)
        logger.info("initialize_data(): returns ready %s | latest_date=%s",
                    getattr(asset_returns, "shape", None), str(asset_returns.index.max()))
    finally:
        INIT_IN_PROGRESS = False


# Warmup at import (gunicorn workers will each run this once)
initialize_data()

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
    # total < 1 → distribute remainder
    rem  = 1.0 - total
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
            w.writerow([a, current_pct.get(a, 0), target_pct.get(a, 0),
                        proposed_pct.get(a, 0), trades_pct.get(a, 0)])
    return path


# ---------- routes ----------

@app.route("/")
def index():
    """Serve /static/index.html, or a clear error if missing."""
    index_path = os.path.join(app.static_folder, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({
        "error": "index_not_found",
        "message": "Place index.html under the /static folder.",
        "looked_in": app.static_folder
    }), 404

@app.route("/explain.html")
def explain_html():
    """Serve /static/explain.html explicitly."""
    path = os.path.join(app.static_folder, "explain.html")
    if os.path.exists(path):
        return send_from_directory(app.static_folder, "explain.html")
    return jsonify({
        "error": "explain_not_found",
        "message": "Place explain.html under the /static folder.",
        "looked_in": app.static_folder
    }), 404

@app.route('/health')
def health():
    return jsonify({"status":"ok", "assets_loaded": asset_returns is not None and not asset_returns.empty})

@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(EXPORT_DIR, filename, as_attachment=True)

@app.route('/get_portfolio', methods=['POST'])
def get_portfolio():
    """
    Returns optimized portfolio + kpis_job_id (needed by /explain).
    KPI job is started non-blockingly to avoid delaying optimization.
    """
    global asset_returns, LATEST_KPIS_JOB_ID, LATEST_KPIS_JOB_STARTED_AT

    # Ensure data present
    if asset_returns is None or getattr(asset_returns, "empty", True):
        initialize_data()
    if asset_returns is None or getattr(asset_returns, "empty", True):
        return jsonify({"error":"Historical data not loaded. Please retry shortly."}), 503, {
            "Retry-After": "3"
        }

    # Log cache age & apply refresh policy (guard against inf)
    age_val = _cache_age_seconds()
    age_sec = int(age_val) if math.isfinite(age_val) else -1
    logger.info("get_portfolio(): cache_age=%ss  swr=%s  hard_cap=%s",
                age_sec, STALE_WHILE_REVALIDATE, RETURNS_HARD_MAX_AGE_SECONDS or "disabled")

    if STALE_WHILE_REVALIDATE:
        maybe_refresh_returns_if_stale(blocking=False)
    else:
        maybe_refresh_returns_if_stale(blocking=True)

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

    # 1) KPI pipeline: at most once per TTL, reuse recent job_id
    job_id = None
    now_ts = time.time()
    if LATEST_KPIS_JOB_ID and (now_ts - LATEST_KPIS_JOB_STARTED_AT) < KPI_JOB_TTL_SECONDS:
        job_id = LATEST_KPIS_JOB_ID
        logger.info("get_portfolio(): reusing KPI job_id=%s (age=%.1fs)", job_id, now_ts - LATEST_KPIS_JOB_STARTED_AT)
    else:
        try:
            api = AssetSentimentAPI()
            job_id = api.start_analysis(assets=None, timeout_minutes=15)
            LATEST_KPIS_JOB_ID = job_id
            LATEST_KPIS_JOB_STARTED_AT = now_ts
            logger.info("get_portfolio(): started KPI job_id=%s", job_id)
        except Exception as e:
            logger.warning("KPI pipeline start failed: %s", e)
            job_id = LATEST_KPIS_JOB_ID  # fall back to last known if any

    # 2) Heuristic sentiments (fast, local)
    try:
        recent = asset_returns[assets_all].tail(60)
        mu = recent.mean()
        sigma = recent.std().replace(0, np.nan)
        z = (mu / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0) * 0.1
        heuristic_sentiments = {a: float(z.get(a, 0.0)) for a in assets_all}
    except Exception as e:
        logger.warning("Heuristic sentiments failed: %s", e)
        heuristic_sentiments = {a: 0.0 for a in assets_all}

    # 3) Expected returns with sentiment tilt
    local_returns = asset_returns[investable].copy()
    # Double-sanitize before optimizer
    local_returns = local_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if local_returns.empty:
        return jsonify({"error":"No valid return series after sanitization."}), 500

    expected_returns = forecaster.generate_forecasted_returns(
        local_returns,
        {a: heuristic_sentiments.get(a, 0.0) for a in investable}
    )

    # 4) Optimize (guard against warnings; fallback handled inside optimizer)
    if risk_slider is not None:
        try:
            with np.errstate(invalid="ignore", over="ignore"):
                weights, performance = optimizer.get_portfolio_by_slider(local_returns, expected_returns, float(risk_slider))
            slider_used = int(round(max(0, min(100, float(risk_slider)))))
            profile_used = "Balanced" if slider_used == 50 else "Custom"
        except Exception as e:
            logger.warning("slider fallback: %s", e)
            with np.errstate(invalid="ignore", over="ignore"):
                weights, performance = optimizer.get_optimal_portfolio(local_returns, expected_returns, objective=objective)
            slider_used = {"MinRisk":0,"Sharpe":50,"MaxRet":100}.get(objective,50)
            profile_used = "Balanced" if objective=="Sharpe" else ("Conservative" if objective=="MinRisk" else "Aggressive")
    else:
        with np.errstate(invalid="ignore", over="ignore"):
            weights, performance = optimizer.get_optimal_portfolio(local_returns, expected_returns, objective=objective)
        slider_used = {"MinRisk":0,"Sharpe":50,"MaxRet":100}.get(objective,50)
        profile_used = "Balanced" if objective=="Sharpe" else ("Conservative" if objective=="MinRisk" else "Aggressive")

    if weights is None or getattr(weights, "empty", True):
        return jsonify({"error":"Optimization failed."}), 500

    # 5) Cap + anchor to user
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

    # 6) Performance (annualized)
    mu = expected_returns.reindex(target_anchored.index).fillna(0.0)
    cov = asset_returns.cov().reindex(index=mu.index, columns=mu.index).fillna(0.0)
    ret = float((mu @ target_anchored) * 252.0)
    vol = float(np.sqrt(max(0.0, target_anchored.T @ cov.values @ target_anchored)) * np.sqrt(252.0))
    shp = ret/vol if vol>0 else 0.0

    perf = {
        "Expected annual return": f"{ret*100:.2f}%",
        "Annual volatility": f"{vol*100:.2f}%",
        "Sharpe Ratio": f"{shp:.2f}"
    }

    response = {
        "weights_target_model_pct":    (target_raw*100).round(2).to_dict(),
        "weights_target_capped_pct":   (target_capped*100).round(2).to_dict(),
        "weights_target_anchored_pct": (target_anchored*100).round(2).to_dict(),
        "proposal": plan | {"policy": pol | {"objective": objective}},
        "performance": perf,
        "sentiments": {k: round(float(v),3) for k,v in heuristic_sentiments.items()},
        "risk_profile_used": profile_used,
        "risk_slider_used": slider_used,
        "investable_assets": investable,
        "disabled_assets": list(disabled),
        "kpis_job_id": job_id or LATEST_KPIS_JOB_ID
    }

    resp = jsonify(response)
    # Freshness/debug headers
    try:
        age_dbg = _cache_age_seconds()
        resp.headers["X-Returns-Cache-Age-Seconds"] = str(int(age_dbg)) if math.isfinite(age_dbg) else "inf"
        resp.headers["X-Returns-SWR"] = "1" if STALE_WHILE_REVALIDATE else "0"
        resp.headers["X-Returns-Latest-Date"] = str(asset_returns.index.max())
    except Exception:
        pass
    return resp

@app.route('/explain', methods=['POST'])
def explain():
    """
    Relay to explainer with the schema it expects.
    If the explainer is slow/busy, return 202 {status:'pending'} so the UI keeps polling.
    """
    data = request.get_json(silent=True) or {}

    job_id = data.get("job_id") or data.get("kpis_job_id") or LATEST_KPIS_JOB_ID

    def _to_pct_map(keys):
        for k in keys:
            m = data.get(k)
            if isinstance(m, dict):
                try:
                    return {a: float(m.get(a, 0.0)) for a in ["Gold","Equities","REITs","Bitcoin"]}
                except Exception:
                    pass
        return {"Gold":0.0,"Equities":0.0,"REITs":0.0,"Bitcoin":0.0}

    current_portfolio   = _to_pct_map(["current_portfolio","current_weights"])
    optimized_portfolio = _to_pct_map(["optimized_portfolio","final_weights"])

    rp_in = (data.get("risk_profile_used") or data.get("risk_profile") or "Balanced").lower()
    if "min" in rp_in or "cons" in rp_in:   risk_profile = "Conservative"
    elif "max" in rp_in or "agg" in rp_in:  risk_profile = "Aggressive"
    else:                                   risk_profile = "Balanced"

    if not job_id:
        return jsonify({"status":"error","error":"missing_job_id",
                        "message":"No KPI job_id available. Please click Get Recommendation again."}), 200

    payload = {
        "job_id": job_id,
        "current_portfolio": current_portfolio,
        "optimized_portfolio": optimized_portfolio,
        "risk_profile": risk_profile
    }
    logger.info("explain(): POST %s | payload=%s", EXPLAINER_URL, payload)

    s = _requests_session_with_retry()

    try:
        r = s.post(EXPLAINER_URL, json=payload, timeout=(EXPLAINER_CONNECT_TIMEOUT, EXPLAINER_READ_TIMEOUT))

        if r.status_code in (202, 425, 429, 500, 502, 503, 504):
            logger.warning("Explainer not ready (HTTP %s). Returning pending.", r.status_code)
            return jsonify({"status": "pending"}), 202

        if r.status_code != 200:
            logger.warning("Explainer returned %s: %s", r.status_code, r.text[:400])
            return jsonify({"status":"error",
                            "error": f"explainer_http_{r.status_code}",
                            "message": r.text}), 200

        try:
            return jsonify(r.json())
        except Exception:
            logger.exception("explain(): invalid JSON from explainer")
            return jsonify({"status":"error","error":"invalid_json"}), 200

    except requests.exceptions.ReadTimeout:
        logger.warning("Explainer read timeout after %.1fs; returning pending", EXPLAINER_READ_TIMEOUT)
        return jsonify({"status": "pending"}), 202

    except Exception as e:
        logger.exception("explain(): relay failed: %s", e)
        return jsonify({"status":"error","error":"relay_failed"}), 200


if __name__ == "__main__":
    # Local runs: optional debug and reloader control via env
    port = int(os.environ.get("PORT", 5000))
    debug_env = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    disable_reloader = os.environ.get("DISABLE_RELOADER", "1") in ("1", "true", "True")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_env,
        use_reloader=(debug_env and not disable_reloader)
    )
