# explainer_llm.py
import os, math
import requests
from typing import Dict, Any

def _topn(d: Dict[str, float], n=2):
    return sorted([(k, float(v or 0)) for k, v in (d or {}).items()], key=lambda x: x[1], reverse=True)[:n]

def _fmt_pct(x):
    try: return f"{float(x):.2f}%"
    except: return "—"

def _rule_based(payload: Dict[str, Any]) -> str:
    pol   = payload.get("policy", {}) or {}
    prof  = pol.get("profile", "Balanced")
    sent  = payload.get("sentiments", {}) or {}
    perf  = payload.get("performance", {}) or {}
    plan  = payload.get("proposal", {}) or {}
    anchored = plan.get("anchored", payload.get("weights_target_pct", {})) or {}
    trades   = plan.get("trades_pct", {}) or {}
    t_used   = plan.get("turnover_used_pct", None)

    top_w    = _topn(anchored, 2)
    top_sent = _topn(sent, 2)
    worst    = sorted(sent.items(), key=lambda kv: kv[1])[:1] if sent else []

    parts = []
    # Lead line: profile + objective
    parts.append(f"This {prof} allocation emphasizes stability and risk-adjusted return.")

    # Why these tilts (weights + sentiment)
    if top_w:
        tw_str = ", ".join([f"{a} ({_fmt_pct(w)})" for a,w in top_w])
        parts.append(f"We tilt toward {tw_str}")

    if top_sent:
        ts_str = ", ".join([f"{a} ({s:+.2f})" for a,s in top_sent])
        parts.append(f"reflecting supportive composite sentiment in {ts_str}")

    # Guardrails / constraints
    caps = pol.get("max_caps_pct", {})
    cap_notes = [f"{a} ≤ {int(v)}%" for a,v in caps.items()] if caps else []
    if cap_notes:
        parts.append(f"while respecting guardrails ({'; '.join(cap_notes)}).")

    # Rebalancing cost and discipline
    if t_used is not None:
        parts.append(f"Rebalance sized to ~{_fmt_pct(t_used)} turnover")
    mb = pol.get("min_trade_band_pct", None)
    if mb is not None:
        parts.append(f"with ±{int(mb)}% trade bands to avoid churn.")

    # Risk/return summary
    er = perf.get("Expected annual return")
    vol= perf.get("Annual volatility")
    sr = perf.get("Sharpe Ratio")
    if er and vol and sr:
        parts.append(f"Projected metrics: ER {er}, Vol {vol}, Sharpe {sr}.")

    # Hedge / underweight note if any clearly weak sentiment
    if worst:
        a, s = worst[0]
        if s < -0.15:
            parts.append(f"We keep {a} contained given softer sentiment ({s:+.2f}).")

    return " ".join(parts)

# ---- OpenRouter LLM wrapper ----
def _build_prompt(payload: Dict[str, Any]) -> str:
    """One short paragraph, portfolio-level, 70–110 words, no disclaimers."""
    return f"""
You are a CIO explaining an allocation to a non-technical investor.
Write ONE paragraph (70–110 words), neutral and professional, no bullets.

Inputs (JSON):
{payload}

Rules:
- Mention investor profile (policy.profile) and the goal (risk-adjusted return for the profile).
- Name at most two overweight assets with % (proposal.anchored or weights_target_pct).
- Briefly connect sentiment scores (sentiments) to tilts.
- Mention risk controls (policy.turnover_cap_pct, policy.max_caps_pct, policy.min_trade_band_pct).
- End with the performance line (performance fields), no extra caveats.
    """.strip()

def _call_openrouter(prompt: str) -> str:
    """
    Call OpenRouter API with the provided API key.
    Return None/'' to fall back when key is missing or call fails.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    print(f"[DEBUG] OpenRouter API key found: {bool(api_key)}")
    if not api_key:
        print("[DEBUG] No OpenRouter API key found, using fallback")
        return ""
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 200
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
        
        return ""
        
    except Exception as e:
        print(f"OpenRouter API call failed: {e}")
        return ""

def explain_allocation(payload: Dict[str, Any]) -> str:
    print("[DEBUG] explain_allocation called")
    try:
        txt = _call_openrouter(_build_prompt(payload))
        if txt and txt.strip():
            print("[DEBUG] OpenRouter API returned text, using LLM explanation")
            return txt.strip()
        else:
            print("[DEBUG] OpenRouter API returned empty, using fallback")
    except Exception as e:
        print(f"[DEBUG] OpenRouter API failed: {e}, using fallback")
        pass
    # Fallback explanation that always works
    print("[DEBUG] Using rule-based fallback explanation")
    return _rule_based(payload)
