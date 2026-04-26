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
        "check": lambda r: ("1" in r and "9" in r) if r else False,
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
            any(x in r.lower() for x in ["stack overflow", "recursionerror", "recursion error", "maximum recursion", "infinite loop", "never terminates", "no base"])
        ) if r else False,
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
        ) if r else False,
        "points": 3,
        "label": "Q3: shallow vs deep copy",
    },
    {
        "question": (
            "Write the exact bash command to find all files modified in the last 24 hours "
            "in the /var/log directory. Output only the command."
        ),
        "check": lambda r: ("find" in r.lower() and "-mtime" in r.lower()) if r else False,
        "points": 3,
        "label": "Q4: find -mtime",
    },
    {
        "question": (
            "A REST API returns HTTP 429. What does this status code mean, "
            "and what should your client code do in response? Answer in 2-3 sentences."
        ),
        "check": lambda r: (
            any(x in r.lower() for x in ["rate limit", "too many requests", "rate-limit", "ratelimit", "throttl"]) and
            any(x in r.lower() for x in ["retry", "backoff", "back-off", "wait", "exponential", "delay"])
        ) if r else False,
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
            return score, "busy", f"Timeout on {q['label']} — provider slow/overloaded"
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

# Hardcoded fallback model lists per provider — used when /v1/models unavailable
PROVIDER_FALLBACK_MODELS = {
    "cerebras":       ["qwen-3-235b-a22b-instruct-2507", "llama-3.3-70b", "llama3.1-8b"],
    "sambanova":      ["Meta-Llama-3.3-70B-Instruct", "Meta-Llama-3.1-405B-Instruct", "Meta-Llama-3.1-8B-Instruct"],
    "nvidia":         ["nvidia/llama-3.3-nemotron-super-49b-v1", "mistralai/mistral-large-2-instruct", "meta/llama-3.3-70b-instruct"],
    "nvidia_nemotron": ["nvidia/llama-3.3-nemotron-super-49b-v1", "meta/llama-3.3-70b-instruct", "mistralai/mistral-large-2-instruct"],
    "openrouter":     ["meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-large-2411", "qwen/qwen-2.5-72b-instruct"],
    "groq":           ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
}

# Provider /v1/models endpoints
PROVIDER_MODELS_ENDPOINTS = {
    "cerebras":       "https://api.cerebras.ai/v1/models",
    "sambanova":      "https://api.sambanova.ai/v1/models",
    "nvidia":         "https://integrate.api.nvidia.com/v1/models",
    "nvidia_nemotron": "https://integrate.api.nvidia.com/v1/models",
    "openrouter":     "https://openrouter.ai/api/v1/models",
    "groq":           "https://api.groq.com/openai/v1/models",
}


def _fetch_provider_models(provider_id: str, api_key: str) -> list[str]:
    """
    Fetch available models from provider /v1/models endpoint.
    Returns list of model id strings, filtered for coding suitability.
    Empty list if endpoint unavailable.
    """
    endpoint = PROVIDER_MODELS_ENDPOINTS.get(provider_id)
    if not endpoint:
        return []
    try:
        resp = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning(f"  /v1/models returned {resp.status_code} for {provider_id}")
            return []
        data = resp.json()
        # OpenAI-compatible: {"data": [{"id": "model-name"}, ...]}
        models = [m.get("id", "") or m.get("name", "") for m in data.get("data", [])]
        # Filter bad patterns and size minimums
        filtered = []
        for m in models:
            if not m:
                continue
            if BAD_MODEL_RE.search(m):
                continue
            size_match = re.search(r'(\d+)\s*[bB]\b', m)
            if size_match and int(size_match.group(1)) < MIN_PARAMS_B:
                continue
            filtered.append(m)
        log.info(f"  /v1/models: {len(filtered)} suitable models found for {provider_id}")
        return filtered
    except Exception as e:
        log.warning(f"  /v1/models fetch failed for {provider_id}: {e}")
        return []


def _extract_param_count(model_id: str) -> int:
    """
    Extract parameter count in billions from model id string.
    Returns 0 if not found — sorts to bottom.
    Examples: "llama-3.3-70b" -> 70, "qwen-3-235b-a22b" -> 235, "gpt-4" -> 0
    """
    m = re.search(r'(?<![\d])(\d+)\s*[bB]\b', model_id)
    if m:
        return int(m.group(1))
    return 0


def _find_replacement(provider_id: str, api_key: str, api_url: str, query: str) -> Optional[str]:
    """
    Find and verify the best replacement model for a dead provider.

    Strategy:
    1. Fetch candidate list from /v1/models endpoint (filtered by BAD_MODEL_PATTERNS + MIN_PARAMS)
    2. Fall back to hardcoded list if endpoint fails
    3. Sort candidates by parameter count descending (235B > 70B > 30B)
    4. Score top 3 candidates with capability questions
    5. Return highest-scoring candidate that clears SCORE_THRESHOLD
       If none clear threshold, return highest scorer above SCORE_REJECT

    This ensures a 235B model beats an 8B model even if both answer all questions correctly.
    """
    # Try live model list from provider first
    candidates = _fetch_provider_models(provider_id, api_key)

    # Fall back to hardcoded list if endpoint fails or returns nothing
    if not candidates:
        log.info(f"  Falling back to hardcoded model list for {provider_id}")
        candidates = PROVIDER_FALLBACK_MODELS.get(provider_id, [])

    if not candidates:
        log.warning(f"  No replacement candidates found for {provider_id}")
        return None

    # Sort by parameter count descending — largest model first
    candidates_ranked = sorted(candidates, key=_extract_param_count, reverse=True)
    top3 = candidates_ranked[:3]
    log.info(f"  Top candidates for {provider_id} (by param count): {top3}")

    # Score each candidate, pick best above threshold
    best_model = None
    best_score = -1

    for candidate in top3:
        log.info(f"  Scoring candidate: {candidate}")
        c_score, c_status, c_detail = _call_model(api_url, api_key, candidate)
        log.info(f"  {candidate} scored {c_score}/15 (status: {c_status})")

        if c_score > best_score:
            best_score = c_score
            best_model = candidate

        # Stop early if we found a clearly good model
        if c_score >= SCORE_THRESHOLD:
            log.info(f"  {candidate} cleared threshold — stopping search")
            break

        time.sleep(1)

    if best_model and best_score > SCORE_REJECT:
        log.info(f"  Best replacement for {provider_id}: {best_model} (score {best_score}/15)")
        return best_model

    log.warning(f"  No suitable replacement found for {provider_id} (best score: {best_score}/15)")
    return None

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
            replacement = _find_replacement(provider_id, api_key, pdef["api_url"], pdef["tavily_query"])

            if replacement and replacement != current_model:
                log.info(f"[{provider_id}] ✅ Replacement selected: {replacement}")
                registry[provider_id] = _provider_entry(replacement, "ok", best_score_from_find := 0, prev_entry)
                alerts.append(
                    f"🔴 DEAD — {pdef['label']}: {current_model} → {replacement}"
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
    import fcntl
    lock_path = Path(os.getenv("HEALTH_LOCK_PATH", "/app/data/health_check.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(str(lock_path), "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info("Another health check instance is running — exiting")
        raise SystemExit(0)
    try:
        run_health_check()
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()