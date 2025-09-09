import time
import requests
import logging
from typing import Dict, List, Optional

DEFAULT_BASE_URL = "https://assetmanagement-production-f542.up.railway.app"
logger = logging.getLogger(__name__)

class AssetSentimentAPI:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout_s: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def start_analysis(self, assets: Optional[List[str]] = None, timeout_minutes: int = 15) -> Optional[str]:
        payload = {
            "assets": assets or ["NIFTY50", "GOLD", "BITCOIN", "REIT"],
            "timeout_minutes": timeout_minutes,
            "generate_pipeline_outputs": True
        }
        logger.info("AssetSentimentAPI.start_analysis(): POST analyze/all-assets | payload=%s", payload)
        r = self.session.post(f"{self.base_url}/analyze/all-assets", json=payload, timeout=30)
        if r.status_code == 200 and "job_id" in r.json():
            job_id = r.json()["job_id"]
            logger.info("AssetSentimentAPI.start_analysis(): job_id=%s", job_id)
            return job_id
        logger.warning("start_analysis() failed: %s %s", r.status_code, r.text[:300])
        return None

    # ---------- NEW: non-blocking helpers ----------

    def get_status(self, job_id: str) -> Optional[str]:
        try:
            jd = self.session.get(f"{self.base_url}/jobs/{job_id}", timeout=10)
            if jd.status_code != 200:
                return None
            return jd.json().get("status")
        except Exception:
            return None

    def try_get_sentiments_once(self, job_id: str) -> Dict[str, float]:
        """
        If job is completed, pull the CSV once; else return {} without waiting.
        """
        status = self.get_status(job_id)
        if status != "completed":
            return {}
        try:
            r = self.session.get(f"{self.base_url}/jobs/{job_id}/csv", timeout=30)
            if r.status_code != 200:
                return {}
            data = r.json().get("data", [])
            out: Dict[str, float] = {}
            for row in data:
                if (
                    row.get("metric_type") == "sentiment"
                    and row.get("metric_name") == "overall_sentiment"
                    and row.get("asset") != "PORTFOLIO"
                ):
                    try:
                        out[row["asset"]] = float(row.get("value", 0.0))
                    except Exception:
                        pass
            if out:
                logger.info("try_get_sentiments_once(): parsed sentiments %s", out)
            return out
        except Exception:
            return {}

    # (legacy, still available if you ever need blocking)
    def wait_until_done(self, job_id: str, max_wait: Optional[int] = None) -> bool:
        max_wait = max_wait or self.timeout_s
        start = time.time()
        while (time.time() - start) < max_wait:
            st = self.get_status(job_id)
            if st == "completed":
                return True
            if st == "failed":
                return False
            time.sleep(2)
        return False

    def get_sentiment_dict(self, job_id: str) -> Dict[str, float]:
        r = self.session.get(f"{self.base_url}/jobs/{job_id}/csv", timeout=30)
        if r.status_code != 200:
            return {}
        data = r.json().get("data", [])
        out: Dict[str, float] = {}
        for row in data:
            if (
                row.get("metric_type") == "sentiment"
                and row.get("metric_name") == "overall_sentiment"
                and row.get("asset") != "PORTFOLIO"
            ):
                try:
                    out[row["asset"]] = float(row.get("value", 0.0))
                except Exception:
                    pass
        return out

    def analyze_and_get_sentiments(self, assets: Optional[List[str]] = None, wait_s: int = 60) -> Dict[str, float]:
        job_id = self.start_analysis(assets=assets)
        if not job_id:
            return {}
        self.wait_until_done(job_id, max_wait=wait_s)
        return self.get_sentiment_dict(job_id)
