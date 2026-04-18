#!/usr/bin/env python3
"""
tests/benchmark_models.py
=========================
Session 15 — Free Claude Agent model benchmarking.

Tests all 6 candidate models across 4 dimensions:
  1. Speed      — TTFT (time to first token) + total time
  2. Tool call  — Does it return a valid JSON tool-call schema?
  3. Reasoning  — Multi-step math (must show all steps, correct final answer)
  4. Coding     — Debug a broken function + explain the fix

Usage:
  cd ~/Tiny Agent/free-claude-agent
  python tests/benchmark_models.py

  # Run specific models only:
  python tests/benchmark_models.py --models cerebras_qwen3 groq_8b

  # Run specific tests only:
  python tests/benchmark_models.py --tests speed tool_call

  # Save results to JSON:
  python tests/benchmark_models.py --output tests/results/benchmark_session15.json

Missing API keys → model is skipped (yellow warning, not a crash).
"""

import os
import sys
import json
import time
import argparse
import textwrap
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}  ✓ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ⚠ {msg}{RESET}")
def fail(msg): print(f"{RED}  ✗ {msg}{RESET}")
def info(msg): print(f"{CYAN}  → {msg}{RESET}")
def head(msg): print(f"\n{BOLD}{msg}{RESET}")


# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────
MODELS = [
    {
        "id":        "cerebras_qwen3",
        "label":     "Cerebras / Qwen3-235B",
        "provider":  "cerebras",
        "model":     "qwen-3-235b-a22b-instruct-2507",
        "base_url":  "https://api.cerebras.ai/v1",
        "key_env":   "CEREBRAS_API_KEY",
        "style":     "openai",
        "tier_note": "Current primary tool caller",
    },
    {
        "id":        "sambanova_deepseek",
        "label":     "SambaNova / DeepSeek-R1-0528",
        "provider":  "sambanova",
        "model":     "DeepSeek-R1-0528",
        "base_url":  "https://api.sambanova.ai/v1",
        "key_env":   "SAMBANOVA_API_KEY",
        "style":     "openai",
        "tier_note": "Current deep reasoning primary",
    },
    {
        "id":        "nvidia_nemotron",
        "label":     "NVIDIA NIM / Nemotron-3-Nano-30B",
        "provider":  "nvidia",
        "model":     "nvidia/nemotron-3-nano-30b-a3b",
        "base_url":  "https://integrate.api.nvidia.com/v1",
        "key_env":   "NVIDIA_API_KEY_NEMOTRON",
        "style":     "openai",
        "tier_note": "NEW — tier TBD from benchmark",
    },
    {
        "id":        "modal_glm51",
        "label":     "Modal / GLM-5.1-FP8",
        "provider":  "modal",
        "model":     "zai-org/GLM-5.1-FP8",
        "base_url":  "https://api.us-west-2.modal.direct/v1",
        "key_env":   "GLM51_MODAL_KEY",
        "style":     "openai",
        "tier_note": "NEW — coding+reasoning candidate (free until Apr 30)",
    },
    {
        "id":        "sambanova_llama70b",
        "label":     "SambaNova / Llama-3.3-70B",
        "provider":  "sambanova",
        "model":     "Meta-Llama-3.3-70B-Instruct",
        "base_url":  "https://api.sambanova.ai/v1",
        "key_env":   "SAMBANOVA_API_KEY",
        "style":     "openai",
        "tier_note": "Current fallback 1",
    },
    {
        "id":        "groq_8b",
        "label":     "Groq / Llama-3.1-8B",
        "provider":  "groq",
        "model":     "llama-3.1-8b-instant",
        "base_url":  "https://api.groq.com/openai/v1",
        "key_env":   "GROQ_API_KEY",
        "style":     "openai",
        "tier_note": "Current fallback 2 — fast, high RPM",
    },
    {
        "id":        "ollama_glm51",
        "label":     "Ollama Cloud / GLM-5.1",
        "provider":  "ollama",
        "model":     "glm-5.1:cloud",
        "base_url":  "https://ollama.com",
        "key_env":   "GLM51_OLLAMA_KEY",
        "style":     "ollama",   # /api/chat — NOT openai-compat
        "tier_note": "NEW — May 1 fallback when Modal expires",
    },
]


# ─────────────────────────────────────────────
# BENCHMARK PROMPTS
# ─────────────────────────────────────────────

SPEED_PROMPT = "Reply with exactly one word: Hello"

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]
TOOL_PROMPT = "What is the weather in Tokyo right now? Use the get_weather tool."

REASONING_PROMPT = textwrap.dedent("""
    A train leaves Station A at 9:00 AM travelling at 60 km/h toward Station B.
    Another train leaves Station B at 10:00 AM travelling at 90 km/h toward Station A.
    The distance between the two stations is 300 km.
    At what time do the trains meet? Show every step of your working.
""").strip()
# Correct answer: 11:20 AM
REASONING_KEYWORDS = ["11:20", "11:20 am", "1 hour 20", "80 minutes", "1h 20", "11h20"]

CODING_PROMPT = textwrap.dedent("""
    The following Python function is supposed to return the nth Fibonacci number
    but it has a bug. Identify the bug and provide the corrected function.

    ```python
    def fibonacci(n):
        if n <= 0:
            return 0
        if n == 1:
            return 1
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return a
    ```

    Hint: fibonacci(5) should return 5 but this function returns 3.
    Show the corrected code and explain exactly what was wrong.
""").strip()
# Fix: return b instead of a (loop runs n-1 times but ends with correct value in b)
CODING_KEYWORDS = ["return b", "off-by-one", "off by one", "b instead", "returns b"]


# ─────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────

def _openai_call(
    model_cfg: dict,
    messages: list,
    tools: list = None,
    timeout: int = 60,
) -> tuple[str, float, float, Optional[list]]:
    """
    OpenAI-compatible /v1/chat/completions call using only stdlib urllib.
    Returns: (content, ttft_s, total_s, tool_calls_or_None)
    """
    import urllib.request
    import urllib.error

    api_key = os.environ.get(model_cfg["key_env"], "")
    url = model_cfg["base_url"].rstrip("/") + "/chat/completions"

    payload: dict = {
        "model":       model_cfg["model"],
        "messages":    messages,
        "max_tokens":  1024,
        "temperature": 0,
        "stream":      False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ttft = time.time() - t0
            body = json.loads(resp.read())
            total = time.time() - t0
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err[:400]}")
    except Exception as exc:
        raise RuntimeError(str(exc))

    choice     = body["choices"][0]
    msg        = choice["message"]
    content    = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")
    return content, ttft, total, tool_calls


def _ollama_call(
    model_cfg: dict,
    messages: list,
    timeout: int = 90,
) -> tuple[str, float, float]:
    """
    Ollama Cloud remote API call.
    POST https://ollama.com/api/chat
    Auth: Authorization: Bearer <GLM51_OLLAMA_KEY>
    Response format: {"message": {"role": "assistant", "content": "..."}}

    NOTE: Ollama Cloud /api/chat does not yet expose OpenAI-style tool calling.
    Tool-call test is skipped automatically for this provider.
    """
    import urllib.request
    import urllib.error

    api_key = os.environ.get(model_cfg["key_env"], "")
    url = "https://ollama.com/api/chat"

    payload = {
        "model":    model_cfg["model"],
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0, "num_predict": 1024},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ttft  = time.time() - t0
            body  = json.loads(resp.read())
            total = time.time() - t0
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err[:400]}")
    except Exception as exc:
        raise RuntimeError(str(exc))

    content = body.get("message", {}).get("content", "")
    return content, ttft, total


# ─────────────────────────────────────────────
# INDIVIDUAL TESTS
# ─────────────────────────────────────────────

def test_speed(model_cfg: dict, run: bool = True) -> dict:
    if not run:
        return {"skipped": True}
    messages = [{"role": "user", "content": SPEED_PROMPT}]
    try:
        if model_cfg["style"] == "openai":
            content, ttft, total, _ = _openai_call(model_cfg, messages, timeout=30)
        else:
            content, ttft, total = _ollama_call(model_cfg, messages, timeout=30)
        return {
            "passed":           True,
            "ttft_s":           round(ttft, 2),
            "total_s":          round(total, 2),
            "response_preview": content.strip()[:80],
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


def test_tool_call(model_cfg: dict, run: bool = True) -> dict:
    if not run:
        return {"skipped": True}
    if model_cfg["style"] == "ollama":
        return {
            "passed": None,
            "note":   "Ollama Cloud /api/chat does not support tool_calls — skip",
        }
    messages = [{"role": "user", "content": TOOL_PROMPT}]
    try:
        content, ttft, total, tool_calls = _openai_call(
            model_cfg, messages, tools=TOOL_SCHEMA, timeout=45
        )
        if tool_calls and len(tool_calls) > 0:
            first    = tool_calls[0]
            fn_name  = first.get("function", {}).get("name", "")
            args_raw = first.get("function", {}).get("arguments", "{}")
            try:
                args         = json.loads(args_raw)
                city_present = "city" in args
            except Exception:
                city_present = False
            return {
                "passed":    fn_name == "get_weather" and city_present,
                "tool_name": fn_name,
                "args":      args_raw[:200],
                "total_s":   round(total, 2),
            }
        else:
            return {
                "passed":           False,
                "note":             "Model returned text instead of tool_call",
                "response_preview": content.strip()[:120],
                "total_s":          round(total, 2),
            }
    except Exception as e:
        return {"passed": False, "error": str(e)}


def test_reasoning(model_cfg: dict, run: bool = True) -> dict:
    if not run:
        return {"skipped": True}
    messages = [
        {"role": "system", "content": "You are a careful mathematical reasoner. Always show all steps."},
        {"role": "user",   "content": REASONING_PROMPT},
    ]
    try:
        if model_cfg["style"] == "openai":
            content, ttft, total, _ = _openai_call(model_cfg, messages, timeout=90)
        else:
            content, ttft, total = _ollama_call(model_cfg, messages, timeout=90)

        cl = content.lower()
        matched    = [kw for kw in REASONING_KEYWORDS if kw in cl]
        shows_work = any(w in cl for w in ["km", "speed", "hour", "distance", "relative"])
        return {
            "passed":           len(matched) > 0,
            "matched_keywords": matched,
            "shows_work":       shows_work,
            "total_s":          round(total, 2),
            "response_preview": content.strip()[:300],
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


def test_coding(model_cfg: dict, run: bool = True) -> dict:
    if not run:
        return {"skipped": True}
    messages = [
        {"role": "system", "content": "You are an expert Python debugger. Be precise and concise."},
        {"role": "user",   "content": CODING_PROMPT},
    ]
    try:
        if model_cfg["style"] == "openai":
            content, ttft, total, _ = _openai_call(model_cfg, messages, timeout=90)
        else:
            content, ttft, total = _ollama_call(model_cfg, messages, timeout=90)

        cl        = content.lower()
        matched   = [kw for kw in CODING_KEYWORDS if kw in cl]
        has_code  = "def fibonacci" in content or "```python" in content
        return {
            "passed":              len(matched) > 0 and has_code,
            "matched_keywords":    matched,
            "has_corrected_code":  has_code,
            "total_s":             round(total, 2),
            "response_preview":    content.strip()[:300],
        }
    except Exception as e:
        return {"passed": False, "error": str(e)}


# ─────────────────────────────────────────────
# PER-MODEL RUNNER
# ─────────────────────────────────────────────

def run_model_benchmark(model_cfg: dict, active_tests: list) -> dict:
    head(f"{'─' * 62}")
    head(f"  {model_cfg['label']}")
    print(f"  {CYAN}{model_cfg['tier_note']}{RESET}")

    api_key = os.environ.get(model_cfg["key_env"], "")
    if not api_key:
        warn(f"Env var {model_cfg['key_env']} not set — skipping this model")
        return {
            "model_id": model_cfg["id"],
            "label":    model_cfg["label"],
            "skipped":  True,
            "reason":   f"Missing env var: {model_cfg['key_env']}",
        }

    results = {
        "model_id":  model_cfg["id"],
        "label":     model_cfg["label"],
        "skipped":   False,
        "tier_note": model_cfg["tier_note"],
        "tests":     {},
    }

    run_speed    = "speed"     in active_tests
    run_tool     = "tool_call" in active_tests
    run_rsn      = "reasoning" in active_tests
    run_cod      = "coding"    in active_tests

    # ── 1. Speed ─────────────────────────────
    info("Test 1/4 — Speed")
    r = test_speed(model_cfg, run=run_speed)
    results["tests"]["speed"] = r
    if r.get("skipped"):
        warn("Skipped by --tests flag")
    elif r["passed"]:
        ok(f"TTFT {r['ttft_s']}s  |  Total {r['total_s']}s  |  '{r['response_preview']}'")
    else:
        fail(f"Speed test failed: {r.get('error', '?')[:120]}")

    # ── 2. Tool calling ───────────────────────
    info("Test 2/4 — Tool calling")
    r = test_tool_call(model_cfg, run=run_tool)
    results["tests"]["tool_call"] = r
    if r.get("skipped"):
        warn("Skipped by --tests flag")
    elif r.get("passed") is None:
        warn(r.get("note", "Skipped"))
    elif r["passed"]:
        ok(f"Tool '{r['tool_name']}' called  |  args: {r['args'][:80]}  |  {r['total_s']}s")
    else:
        if "error" in r:
            fail(f"Error: {r['error'][:120]}")
        else:
            fail(f"No tool call. {r.get('note','')} Preview: {r.get('response_preview','')[:80]}")

    # ── 3. Reasoning ──────────────────────────
    info("Test 3/4 — Reasoning (multi-step math)")
    r = test_reasoning(model_cfg, run=run_rsn)
    results["tests"]["reasoning"] = r
    if r.get("skipped"):
        warn("Skipped by --tests flag")
    elif r["passed"]:
        ok(f"Correct answer. Keywords: {r['matched_keywords']}  |  {r['total_s']}s")
    else:
        if "error" in r:
            fail(f"Error: {r['error'][:120]}")
        else:
            fail(f"Wrong/missing answer. Preview: {r.get('response_preview','')[:150]}")

    # ── 4. Coding ─────────────────────────────
    info("Test 4/4 — Coding (bug fix)")
    r = test_coding(model_cfg, run=run_cod)
    results["tests"]["coding"] = r
    if r.get("skipped"):
        warn("Skipped by --tests flag")
    elif r["passed"]:
        ok(f"Fix correct. Keywords: {r['matched_keywords']}  |  {r['total_s']}s")
    else:
        if "error" in r:
            fail(f"Error: {r['error'][:120]}")
        else:
            fail(f"Fix not found. Matched: {r.get('matched_keywords',[])}  Preview: {r.get('response_preview','')[:150]}")

    return results


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

def _sym(val) -> str:
    if val is True:  return f"{GREEN}PASS{RESET}"
    if val is False: return f"{RED}FAIL{RESET}"
    if val is None:  return f"{YELLOW}SKIP{RESET}"
    return f"{YELLOW} -- {RESET}"


def print_summary(all_results: list):
    head(f"\n{'═' * 72}")
    head("  BENCHMARK RESULTS SUMMARY")
    head(f"{'═' * 72}")

    header = f"  {'Model':<38} {'Spd':>4} {'Tool':>5} {'Rsn':>5} {'Cod':>5}  {'TTFT':>6}"
    print(header)
    print(f"  {'─'*38} {'─'*4} {'─'*5} {'─'*5} {'─'*5}  {'─'*6}")

    for r in all_results:
        if r.get("skipped"):
            print(f"  {r['label']:<38} {'─':>4} {'─':>5} {'─':>5} {'─':>5}  {'N/A':>6}  ← KEY MISSING")
            continue

        t   = r["tests"]
        spd = t.get("speed",     {})
        tc  = t.get("tool_call", {})
        rsn = t.get("reasoning", {})
        cod = t.get("coding",    {})

        ttft = f"{spd.get('ttft_s','?')}s" if spd.get("passed") else ("ERR" if spd.get("passed") is False else "--")

        print(
            f"  {r['label']:<38}"
            f" {_sym(spd.get('passed')):>4}"
            f" {_sym(tc.get('passed')):>5}"
            f" {_sym(rsn.get('passed')):>5}"
            f" {_sym(cod.get('passed')):>5}"
            f"  {ttft:>6}"
        )

    head(f"\n{'═' * 72}")
    head("  TIER ASSIGNMENT GUIDE")
    head(f"{'═' * 72}")
    print(f"""
  Tier 1 — Primary tool caller  (must PASS: speed + tool_call)
  Tier 2 — Coding / reasoning   (must PASS: coding + reasoning, speed optional)
  Tier 3 — Fast fallback        (must PASS: speed at minimum)

  Confirmed strategy (Session 15):
    Qwen3-235B    → Tier 1 primary tool caller  (keep regardless of results)
    GLM-5.1       → Tier 2 coding + deep reasoning
    DeepSeek-R1   → Tier 2 deep reasoning
    Nemotron-30B  → Assign tier based on results above
    Llama-3.3-70B → Tier 3 fallback 1
    Llama-3.1-8B  → Tier 3 fallback 2
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Free Claude Agent — Session 15 model benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Model IDs:
              cerebras_qwen3     Cerebras / Qwen3-235B
              sambanova_deepseek SambaNova / DeepSeek-R1-0528
              nvidia_nemotron    NVIDIA NIM / Nemotron-3-Nano-30B
              modal_glm51        Modal / GLM-5.1-FP8
              sambanova_llama70b SambaNova / Llama-3.3-70B
              groq_8b            Groq / Llama-3.1-8B
              ollama_glm51       Ollama Cloud / GLM-5.1
        """),
    )
    parser.add_argument(
        "--models", nargs="*",
        choices=[m["id"] for m in MODELS],
        help="Model IDs to benchmark (default: all)",
    )
    parser.add_argument(
        "--tests", nargs="*",
        choices=["speed", "tool_call", "reasoning", "coding"],
        default=["speed", "tool_call", "reasoning", "coding"],
        help="Which tests to run (default: all four)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save full JSON results to this path",
    )
    args = parser.parse_args()

    selected_ids    = args.models or [m["id"] for m in MODELS]
    selected_models = [m for m in MODELS if m["id"] in selected_ids]
    active_tests    = args.tests

    head("╔═══════════════════════════════════════════════════════╗")
    head("║    Free Claude Agent — Session 15 Model Benchmark     ║")
    head("╚═══════════════════════════════════════════════════════╝")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models    : {len(selected_models)} — {', '.join(m['id'] for m in selected_models)}")
    print(f"  Tests     : {', '.join(active_tests)}")

    all_results = []
    for model_cfg in selected_models:
        result = run_model_benchmark(model_cfg, active_tests)
        all_results.append(result)
        time.sleep(2)   # brief pause between models — avoids SambaNova sharing rate limits

    print_summary(all_results)

    if args.output:
        out = {
            "benchmark_run_at": datetime.now().isoformat(),
            "active_tests":     active_tests,
            "results":          all_results,
        }
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print()
        ok(f"Full results saved → {args.output}")


if __name__ == "__main__":
    main()
