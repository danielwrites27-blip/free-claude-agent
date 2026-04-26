"""
model_health_check.py — Self-healing model registry for free-claude-agent.

Runs at pod startup + daily via cron (entrypoint.sh).
Checks each provider's current model with 5 capability questions.
Replaces dead models via Tavily search for best coding alternatives.
Alerts via Telegram. Agent reads models.json dynamically (5-min TTL cache in router.py).

Status states per provider:
  ok              — model healthy, score >= 9/15
  busy            — 429 high traffic, skip replacement today
  quota_exhausted — 429 daily limit hit, skip replacement today
  dead            — 404/400/model-not-found, replace immediately
"""

import os
import json
import time
import logging
import re
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_JSON_PATH = Path(os.getenv("MODELS_JSON_PATH", "/app/data/models.json"))
HEALTH_LOG_PATH  = Path(os.getenv("HEALTH_LOG_PATH",  "/app/data/health_check.log"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")   # set this in pod env vars

TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY", "")

SCORE_THRESHOLD = 9   # minimum score out of 15 to accept a model
SCORE_REJECT    = 3   # score <= this = reject entirely (useless model)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_handlers = [logging.StreamHandler()]
try:
    HEALTH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_handlers.append(logging.FileHandler(str(HEALTH_LOG_PATH), encoding="utf-8"))
except Exception:
    pass  # log dir not available locally — stream only

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [health_check] %(levelname)s %(message)s",
    handlers=_log_handlers,
)
log = logging.getLogger("health_check")

# ---------------------------------------------------------------------------
# Capability questions — 5 questions, 3 pts each, max 15
# ---------------------------------------------------------------------------

CAPABILITY_QUESTIONS = [
    {
        "question": "What is the output of this Python expression: [i*i for i in range(5) if i%2!=0] ? Reply with only the list.",
        "check": lambda r: all(x in r for x in ["1", "9", "25"]),
        "points": 3,
        "label": "Q1: list comprehension",
    },
    {
        "question": (
            "A function calls itself with a smaller input until it hits a base case. "
            "What is this programming pattern called, and what specific bug occurs if the base case is missing? "
            "Answer in 2-3 sentences."
        ),
        "check": lambda r: (
            "recursion" in r.lower() and
            any(x in r.lower() for x in ["stack overflow", "recursionerror", "recursion error", "maximum recursion"])
        ),
        "points": 3,
        "label": "Q2: recursion + stack overflow",
    },
    {
        "question": (
            "What is the difference between a shallow copy and a deep copy of a nested list in Python? "
            "Which standard library module handles deep copies? Answer in 2-3 sentences."
        ),
        "check": lambda r: (
            "copy" in r.lower() and
            any(x in r.lower() for x in ["nested", "reference", "inner"])
        ),
        "points": 3,
        "label": "Q3: shallow vs deep copy",
    },
    {
        "question": (
            "Write the exact bash command to find all files modified in the last 24 hours "
            "in the /var/log directory. Output only the command."
        ),
        "check": lambda r: "find" in r.lower() and "-mtime" in r.lower(),
        "points": 3,
        "label": "Q4: find -mtime",
    },
    {
        "question": (
            "A REST API returns HTTP 429. What does this status code mean, "
            "and what should your client code do in response? Answer in 2-3 sentences."
        ),
        "check": lambda r: (
            any(x in r.lower() for x in ["rate limit", "too many requests", "rate-limit"]) and
            any(x in r.lower() for x in ["retry", "backoff", "back-off", "wait", "exponential"])
        ),
        "points": 3,
        "label": "Q5: 429 rate limit + retry",
    },
]

# ---------------------------------------------------------------------------
# Bad model patterns — reject these from Tavily results
# ---------------------------------------------------------------------------

BAD_MODEL_PATTERNS = [
    r"embed", r"embedding", r"whisper", r"\btts\b", r"vision",
    r"rerank", r"\bguard\b", r"moderat", r"reward",
    r"-r1\b", r"\bthinking\b", r"\breasoning\b", r"\bo1-", r"\bo3-",
    r"\bnano\b", r"-1b\b", r"-3b\b", r"-7b\b",
]
BAD_MODEL_RE = re.compile("|".join(BAD_MODEL_PATTERNS), re.IGNORECASE)

MIN_PARAMS_B = 30  # reject models below 30B parameters

# ---------------------------------------------------------------------------
# Provider definitions — each has: name, api_url, auth_header_fn, model, model_field
# ---------------------------------------------------------------------------

def _bearer(key: str):
    return {"Authorization": f"Bearer {key}"}

PROVIDERS = {
    "cerebras": {
        "label": "Cerebras",
        "api_url": "https://api.cerebras.ai/v1/chat/completions",
        "key_env": "CEREBRAS_API_KEY",
        "default_model": "qwen-3-235b-a22b-instruct-2507",
        "tavily_query": "Cerebras cloud best coding models 2026 SWE-bench benchmark parameters",
    },
    "sambanova": {
        "label": "SambaNova",
        "api_url": "https://api.sambanova.ai/v1/chat/completions",
        "key_env": "SAMBANOVA_API_KEY",
        "default_model": "DeepSeek-V3.1",
        "tavily_query": "SambaNova best coding models 2026 SWE-bench benchmark parameters",
    },
    "nvidia": {
        "label": "NVIDIA NIM",
        "api_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "key_env": "NVIDIA_API_KEY",
        "default_model": "minimaxai/minimax-m2.7",
        "tavily_query": "NVIDIA NIM best coding models 2026 SWE-bench benchmark parameters",
    },
    "nvidia_nemotron": {
        "label": "NVIDIA Nemotron",
        "api_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "key_env": "NVIDIA_API_KEY_NEMOTRON",
        "default_model": "nvidia/nemotron-3-nano-30b-a3b",
        "tavily_query": "NVIDIA NIM best coding models 2026 SWE-bench benchmark parameters",
    },
    "openrouter": {
        "label": "OpenRouter",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "z-ai/glm-5.1",
        "tavily_query": "OpenRouter best coding models 2026 SWE-bench benchmark parameters",
    },
    "groq": {
        "label": "Groq",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.1-8b-instant",
        "tavily_query": "Groq cloud best coding models 2026 SWE-bench benchmark parameters",
    },
}

# ---------------------------------------------------------------------------
# models.json schema helpers
# ---------------------------------------------------------------------------

def _load_models_json() -> dict:
    """Load models.json or return empty dict if missing/corrupt."""
    if not MODELS_JSON_PATH.exists():
        return {}
    try:
        return json.loads(MODELS_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"models.json load error: {e} — starting fresh")
        return {}


def _save_models_json(data: dict):
    MODELS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODELS_JSON_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(MODELS_JSON_PATH)
    log.info(f"models.json saved → {MODELS_JSON_PATH}")


def _provider_entry(model: str, status: str, score: int, prev_entry: dict) -> dict:
    """Build a provider entry for models.json."""
    now = datetime.now(timezone.utc).isoformat()
    history = prev_entry.get("score_history", [])
    history.append(score)
    history = history[-5:]  # keep last 5
    return {
        "current": model,
        "preferred": prev_entry.get("preferred", model),
        "last_verified": now,
        "status": status,
        "last_score": score,
        "score_history": history,
    }

# ---------------------------------------------------------------------------
# 429 classification
# ---------------------------------------------------------------------------

def _classify_429(body: str) -> str:
    """
    Returns 'busy' (high traffic, retry later) or 'quota_exhausted' (daily limit hit).
    """
    lower = body.lower()
    if any(x in lower for x in ["quota", "per day", "limit exceeded", "tokens per", "daily"]):
        return "quota_exhausted"
    # default: treat as transient high traffic
    return "busy"

# ---------------------------------------------------------------------------
# Single model call — returns (score, status, message)
# ---------------------------------------------------------------------------

def _call_model(api_url: str, api_key: str, model: str) -> tuple[int, str, str]:
    """
    Run 5 capability questions against model.
    Returns (score 0-15, status string, human message).
    Status: 'ok' | 'busy' | 'quota_exhausted' | 'dead'
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    score = 0
    for q in CAPABILITY_QUESTIONS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": q["question"]}],
            "max_tokens": 200,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=12)
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout on {model} ({q['label']})")
            continue
        except Exception as e:
            log.warning(f"  Request error on {model} ({q['label']}): {e}")
            continue

        if resp.status_code == 429:
            raw = resp.text
            state = _classify_429(raw)
            log.warning(f"  429 on {model}: {state} — body: {raw[:200]}")
            # Retry once after 10s for busy, not for quota
            if state == "busy":
                log.info("  Retrying once after 10s...")
                time.sleep(10)
                try:
                    resp2 = requests.post(api_url, headers=headers, json=payload, timeout=12)
                    if resp2.status_code == 200:
                        # retry succeeded, continue scoring
                        resp = resp2
                    else:
                        return score, "busy", f"429 busy after retry: {resp2.text[:100]}"
                except Exception:
                    return score, "busy", "429 busy, retry failed"
            else:
                return score, "quota_exhausted", f"Daily quota exhausted: {raw[:100]}"

        if resp.status_code in (400, 404):
            body = resp.text
            if any(x in body.lower() for x in ["model not found", "no such model", "invalid model", "does not exist"]):
                return score, "dead", f"Model dead ({resp.status_code}): {body[:150]}"
            # 400 for other reasons — treat as dead (misconfigured)
            return score, "dead", f"HTTP {resp.status_code}: {body[:150]}"

        if resp.status_code == 401 or resp.status_code == 403:
            return score, "dead", f"Auth error {resp.status_code} — check API key"

        if resp.status_code != 200:
            log.warning(f"  Unexpected {resp.status_code} on {model}: {resp.text[:100]}")
            continue

        try:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            log.warning(f"  Parse error on {model}: {e}")
            continue

        if q["check"](answer):
            score += q["points"]
            log.info(f"  ✅ {q['label']}: +{q['points']} pts")
        else:
            log.info(f"  ❌ {q['label']}: 0 pts | answer: {answer[:80]}")

        time.sleep(1)  # gentle rate limit between questions

    log.info(f"  Final score: {score}/15")
    if score >= SCORE_THRESHOLD:
        return score, "ok", f"Score {score}/15 — healthy"
    else:
        return score, "dead", f"Score {score}/15 — below threshold {SCORE_THRESHOLD}"

# ---------------------------------------------------------------------------
# Tavily — find replacement model
# ---------------------------------------------------------------------------

def _tavily_find_replacement(provider_id: str, query: str) -> Optional[str]:
    """
    Ask Tavily for best coding models for this provider.
    Returns best model name string, or None if can't find one.
    """
    if not TAVILY_API_KEY:
        log.warning("TAVILY_API_KEY not set — cannot search for replacement models")
        return None

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        log.warning(f"Tavily search failed for {provider_id}: {e}")
        return None

    # Collect all text from results
    combined = " ".join(r.get("content", "") + " " + r.get("title", "") for r in results)

    # Extract model-name-like tokens: contain slash, dash, digits
    # e.g. "meta-llama/Llama-3.3-70B" or "qwen-3-235b"
    candidates = re.findall(r'[\w\-\.]+/[\w\-\.]+|[\w]+-[\d]+[bB][\w\-]*', combined)

    scored = []
    for c in candidates:
        if BAD_MODEL_RE.search(c):
            continue
        # Try to extract parameter count
        m = re.search(r'(\d+)\s*[bB]\b', c)
        if m:
            params = int(m.group(1))
            if params < MIN_PARAMS_B:
                continue
            scored.append((params, c))

    if not scored:
        log.info(f"Tavily: no suitable replacement found for {provider_id}")
        return None

    # Pick highest param count
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    log.info(f"Tavily: best replacement candidate for {provider_id}: {best}")
    return best

# ---------------------------------------------------------------------------
# Telegram alerts
# ---------------------------------------------------------------------------

def _telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.info(f"[Telegram skipped — no token/chat_id] {msg}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")

# ---------------------------------------------------------------------------
# Main health check loop
# ---------------------------------------------------------------------------

def run_health_check():
    log.info("=" * 60)
    log.info("Starting model health check")
    log.info("=" * 60)

    registry = _load_models_json()
    alerts = []
    changed = False

    for provider_id, pdef in PROVIDERS.items():
        api_key = os.getenv(pdef["key_env"], "")
        if not api_key:
            log.info(f"[{provider_id}] No API key — skipping")
            continue

        prev_entry = registry.get(provider_id, {})
        current_model = prev_entry.get("current", pdef["default_model"])

        log.info(f"[{provider_id}] Checking model: {current_model}")

        score, status, detail = _call_model(pdef["api_url"], api_key, current_model)

        if status == "ok":
            log.info(f"[{provider_id}] ✅ OK — {current_model} (score {score}/15)")
            registry[provider_id] = _provider_entry(current_model, "ok", score, prev_entry)
            alerts.append(f"🟢 OK — {pdef['label']}: {current_model} (score {score}/15)")
            changed = True

        elif status == "busy":
            log.info(f"[{provider_id}] 🟡 BUSY — skipping replacement. {detail}")
            registry[provider_id] = _provider_entry(current_model, "busy", score, prev_entry)
            alerts.append(f"🟡 BUSY — {pdef['label']}: high traffic, skipped replacement")
            changed = True

        elif status == "quota_exhausted":
            log.info(f"[{provider_id}] 🔴 QUOTA — daily limit hit. {detail}")
            registry[provider_id] = _provider_entry(current_model, "quota_exhausted", score, prev_entry)
            alerts.append(f"🔴 QUOTA — {pdef['label']}: daily limit hit, skipped replacement")
            changed = True

        elif status == "dead":
            log.warning(f"[{provider_id}] 💀 DEAD — {current_model}. Searching replacement...")
            replacement = _tavily_find_replacement(provider_id, pdef["tavily_query"])

            if replacement and replacement != current_model:
                log.info(f"[{provider_id}] Verifying replacement: {replacement}")
                r_score, r_status, r_detail = _call_model(pdef["api_url"], api_key, replacement)

                if r_score > SCORE_REJECT:
                    log.info(f"[{provider_id}] ✅ Replacement accepted: {replacement} (score {r_score}/15)")
                    registry[provider_id] = _provider_entry(replacement, "ok" if r_status == "ok" else r_status, r_score, prev_entry)
                    alerts.append(
                        f"🔴 DEAD — {pdef['label']}: {current_model} → {replacement} (score {r_score}/15)"
                    )
                    changed = True
                else:
                    log.warning(f"[{provider_id}] Replacement also too weak (score {r_score}/15) — keeping dead entry")
                    registry[provider_id] = _provider_entry(current_model, "dead", score, prev_entry)
                    alerts.append(
                        f"🔴 DEAD — {pdef['label']}: {current_model} dead, no suitable replacement found"
                    )
                    changed = True
            else:
                log.warning(f"[{provider_id}] No replacement found — keeping dead entry")
                registry[provider_id] = _provider_entry(current_model, "dead", score, prev_entry)
                alerts.append(
                    f"🔴 DEAD — {pdef['label']}: {current_model} dead, no replacement found"
                )
                changed = True

        # Save incrementally after each provider so partial results survive crashes
        if changed:
            _save_models_json(registry)

        # Small pause between providers to avoid hammering APIs
        time.sleep(2)

    # Send Telegram summary
    if alerts:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        msg = f"<b>🤖 free-claude-agent model health check</b>\n<i>{now_str}</i>\n\n" + "\n".join(alerts)
        _telegram(msg)
        log.info("Telegram alert sent")

    log.info("Health check complete")
    log.info("=" * 60)


if __name__ == "__main__":
    run_health_check()