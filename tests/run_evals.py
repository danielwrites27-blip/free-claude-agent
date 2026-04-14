#!/usr/bin/env python3
"""
Free Claude Agent — Eval Runner
================================
Runs structured evals against the live agent, grades results two ways:
  1. Hard checks  — keyword presence, tool called, must_not_contain (instant, free)
  2. Judge checks — MiniMax M2.7 via NVIDIA NIM grades quality 0-3 (uses NVIDIA_API_KEY)

Usage examples:
  # Run all evals, 3s delay between calls, judge enabled
  python3 tests/run_evals.py

  # Run only tool_use evals
  python3 tests/run_evals.py --category tool_use

  # Run without MiniMax judge (hard checks only, no NVIDIA key needed)
  python3 tests/run_evals.py --no-judge

  # Slower delay to stay well under RPM limits
  python3 tests/run_evals.py --delay 5

  # Compare two saved result files
  python3 tests/run_evals.py --compare results/2026-04-14.json results/2026-04-20.json

Results are saved to tests/results/YYYY-MM-DD_HHMMSS.json automatically.
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# ── Make sure /app src is importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
EVALS_DIR   = Path(__file__).parent / "evals"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── ANSI colours for terminal output ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✅ PASS{RESET}"
FAIL = f"{RED}❌ FAIL{RESET}"
SKIP = f"{YELLOW}⏭  SKIP{RESET}"
WARN = f"{YELLOW}⚠️  WARN{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrapper
# ─────────────────────────────────────────────────────────────────────────────

def load_agent():
    """Load a fresh FreeAgent instance using env vars."""
    from src.agent import FreeAgent
    return FreeAgent(
        api_key=os.getenv("GROQ_API_KEY"),
        daily_token_limit=int(os.getenv("DAILY_TOKEN_LIMIT", "50000")),
        memory_path=os.getenv("MEMORY_PATH", "agent_memory.db"),
    )


def run_agent(agent, prompt: str, timeout: int = 60) -> tuple[str, list[str]]:
    """
    Run one prompt through the agent. Returns (full_response, tools_called).
    Captures which tools were invoked by monkey-patching _execute_tool_call.
    """
    tools_called = []
    original_execute = agent._execute_tool_call

    def capturing_execute(tool_name, args):
        tools_called.append(tool_name)
        return original_execute(tool_name, args)

    agent._execute_tool_call = capturing_execute

    full_response = ""
    try:
        for chunk in agent.ask_stream(prompt):
            full_response += chunk
    except Exception as e:
        full_response = f"[Agent error: {e}]"
    finally:
        agent._execute_tool_call = original_execute  # always restore

    return full_response.strip(), tools_called


# ─────────────────────────────────────────────────────────────────────────────
# Hard grader (no API needed)
# ─────────────────────────────────────────────────────────────────────────────

def hard_grade(ev: dict, response: str, tools_called: list[str]) -> dict:
    """
    Grade with deterministic rules only:
      - expected_tool: was the right tool called? (None = no tool expected)
      - expected_contains: all keywords present in response (case-insensitive)?
      - must_not_contain: none of these strings in response?
    Returns dict with pass/fail per check and overall hard_pass bool.
    """
    results = {}

    # Tool check
    expected_tool = ev.get("expected_tool")
    if expected_tool is None:
        tool_ok = len(tools_called) == 0
        results["tool_check"] = {
            "expected": "no tool",
            "got": tools_called or "none",
            "pass": tool_ok,
        }
    else:
        tool_ok = expected_tool in tools_called
        results["tool_check"] = {
            "expected": expected_tool,
            "got": tools_called or "none",
            "pass": tool_ok,
        }

    # Keyword presence check
    lower_response = response.lower()
    kw_results = {}
    for kw in ev.get("expected_contains", []):
        kw_results[kw] = kw.lower() in lower_response
    results["keyword_check"] = {
        "keywords": kw_results,
        "pass": all(kw_results.values()) if kw_results else True,
    }

    # Must-not-contain check
    mnc_results = {}
    for phrase in ev.get("must_not_contain", []):
        mnc_results[phrase] = phrase.lower() not in lower_response
    results["must_not_contain_check"] = {
        "phrases": mnc_results,
        "pass": all(mnc_results.values()) if mnc_results else True,
    }

    results["hard_pass"] = (
        results["tool_check"]["pass"]
        and results["keyword_check"]["pass"]
        and results["must_not_contain_check"]["pass"]
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MiniMax judge (NVIDIA NIM)
# ─────────────────────────────────────────────────────────────────────────────

def build_judge_client():
    """Build OpenAI-compatible client pointing at NVIDIA NIM."""
    try:
        from openai import OpenAI
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            return None
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
    except ImportError:
        print(f"{YELLOW}openai package not found — install with: pip install openai{RESET}")
        return None


JUDGE_SYSTEM_PROMPT = """You are an objective AI evaluator grading agent responses.
You will be given: a user prompt, the agent's response, and a rubric question.
Grade the response on a scale of 0 to 3:
  3 = Fully correct. All rubric criteria met.
  2 = Mostly correct. Minor gaps or imprecision.
  1 = Partially correct. Some criteria met but important parts missing or wrong.
  0 = Incorrect or unhelpful. Rubric criteria not met.

Respond with ONLY a JSON object in this exact format:
{"score": <0-3>, "reason": "<one sentence explanation>"}
Do not include any other text."""


def judge_response(client, ev: dict, response: str) -> dict:
    """
    Ask MiniMax M2.7 to grade the response quality.
    Returns {"score": 0-3, "reason": "...", "judge_pass": bool}
    """
    if client is None:
        return {"score": -1, "reason": "Judge disabled (no NVIDIA_API_KEY)", "judge_pass": None}

    prompt = f"""User prompt: {ev['input']}

Agent response:
{response[:2000]}

Rubric question: {ev['judge_rubric']}"""

    try:
        completion = client.chat.completions.create(
            model="minimaxai/minimax-m2.7",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,   # low temp for consistent grading
            top_p=0.95,
            max_tokens=150,
        )
        raw = completion.choices[0].message.content.strip()
        # Strip <think>...</think> blocks (reasoning models)
        import re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        reason = parsed.get("reason", "")
        return {
            "score": score,
            "reason": reason,
            "judge_pass": score >= 2,   # 2 or 3 = passing quality
        }
    except json.JSONDecodeError as e:
        return {"score": -1, "reason": f"Judge parse error: {e} | raw: {raw[:100]}", "judge_pass": None}
    except Exception as e:
        return {"score": -1, "reason": f"Judge call failed: {e}", "judge_pass": None}


# ─────────────────────────────────────────────────────────────────────────────
# Single eval runner
# ─────────────────────────────────────────────────────────────────────────────

def run_single_eval(agent, ev: dict, judge_client, use_judge: bool, verbose: bool) -> dict:
    """Run one eval entry end-to-end. Returns full result dict."""
    print(f"\n{CYAN}{BOLD}[{ev['id']}]{RESET} {ev['description']}")
    print(f"  Prompt: {ev['input'][:80]}{'...' if len(ev['input']) > 80 else ''}")

    start = time.time()
    response, tools_called = run_agent(agent, ev["input"])
    elapsed = round(time.time() - start, 2)

    if verbose:
        print(f"  Tools called: {tools_called or 'none'}")
        print(f"  Response ({elapsed}s): {response[:200]}{'...' if len(response) > 200 else ''}")
    else:
        print(f"  Tools called: {tools_called or 'none'} ({elapsed}s)")

    # Hard grade
    hard = hard_grade(ev, response, tools_called)
    hard_status = PASS if hard["hard_pass"] else FAIL

    # Tool check detail
    tc = hard["tool_check"]
    if not tc["pass"]:
        print(f"  {FAIL} tool_check  — expected '{tc['expected']}', got {tc['got']}")
    else:
        print(f"  {PASS} tool_check  — {tc['got']}")

    # Keyword check detail
    kc = hard["keyword_check"]
    if kc["keywords"]:
        failed_kw = [k for k, v in kc["keywords"].items() if not v]
        if failed_kw:
            print(f"  {FAIL} keywords   — missing: {failed_kw}")
        else:
            print(f"  {PASS} keywords   — all present")

    # Must-not-contain detail
    mc = hard["must_not_contain_check"]
    if mc["phrases"]:
        found_bad = [k for k, v in mc["phrases"].items() if not v]
        if found_bad:
            print(f"  {FAIL} must_not   — found: {found_bad}")
        else:
            print(f"  {PASS} must_not   — clean")

    # Judge grade
    judge = {"score": -1, "reason": "Skipped", "judge_pass": None}
    if use_judge:
        judge = judge_response(judge_client, ev, response)
        score_color = GREEN if judge["judge_pass"] else (RED if judge["judge_pass"] is False else YELLOW)
        score_str = f"{score_color}{judge['score']}/3{RESET}"
        print(f"  Judge: {score_str} — {judge['reason']}")
    else:
        print(f"  Judge: {SKIP}")

    overall_pass = hard["hard_pass"] and (judge["judge_pass"] is not False)
    status_str = PASS if overall_pass else FAIL
    print(f"  Overall: {status_str}")

    return {
        "id":          ev["id"],
        "category":    ev["category"],
        "description": ev["description"],
        "input":       ev["input"],
        "response":    response,
        "tools_called": tools_called,
        "elapsed_s":   elapsed,
        "hard":        hard,
        "judge":       judge,
        "overall_pass": overall_pass,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    """Print score table grouped by category."""
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}RESULTS SUMMARY{RESET}")
    print(f"{'─'*60}")

    total_pass = 0
    total_hard = 0
    total_judge_sum = 0
    total_judged = 0

    for cat, items in categories.items():
        passed   = sum(1 for r in items if r["overall_pass"])
        hard_p   = sum(1 for r in items if r["hard"]["hard_pass"])
        judged   = [r for r in items if r["judge"]["score"] >= 0]
        j_avg    = round(sum(r["judge"]["score"] for r in judged) / len(judged), 1) if judged else "n/a"

        total_pass      += passed
        total_hard      += hard_p
        total_judge_sum += sum(r["judge"]["score"] for r in judged)
        total_judged    += len(judged)

        cat_color = GREEN if passed == len(items) else (YELLOW if passed > 0 else RED)
        print(f"\n  {BOLD}{cat.upper()}{RESET}")
        print(f"    Overall:    {cat_color}{passed}/{len(items)}{RESET}")
        print(f"    Hard checks: {hard_p}/{len(items)}")
        print(f"    Judge avg:  {j_avg}/3" if judged else "    Judge avg: n/a (judge disabled)")

        for r in items:
            icon = "✅" if r["overall_pass"] else "❌"
            j_score = f" judge={r['judge']['score']}/3" if r["judge"]["score"] >= 0 else ""
            print(f"      {icon} [{r['id']}] {r['description'][:45]}{j_score}")

    print(f"\n{'─'*60}")
    grand_pct = round(100 * total_pass / len(results)) if results else 0
    grand_color = GREEN if grand_pct >= 80 else (YELLOW if grand_pct >= 60 else RED)
    j_grand = round(total_judge_sum / total_judged, 1) if total_judged else "n/a"
    print(f"  {BOLD}GRAND TOTAL:  {grand_color}{total_pass}/{len(results)} ({grand_pct}%){RESET}")
    print(f"  Hard pass:   {total_hard}/{len(results)}")
    print(f"  Judge avg:   {j_grand}/3" if total_judged else "  Judge avg:  n/a")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Compare two result files
# ─────────────────────────────────────────────────────────────────────────────

def compare_results(file_a: str, file_b: str):
    """Print a diff table between two saved result files."""
    with open(file_a) as f: data_a = json.load(f)
    with open(file_b) as f: data_b = json.load(f)

    results_a = {r["id"]: r for r in data_a["results"]}
    results_b = {r["id"]: r for r in data_b["results"]}
    all_ids   = sorted(set(results_a) | set(results_b))

    label_a = Path(file_a).stem
    label_b = Path(file_b).stem

    print(f"\n{BOLD}COMPARISON: {label_a}  →  {label_b}{RESET}")
    print(f"{'─'*60}")
    print(f"  {'ID':<12} {'Before':>8} {'After':>8} {'Change':>8}")
    print(f"  {'─'*44}")

    improved   = 0
    regressed  = 0
    unchanged  = 0

    for eid in all_ids:
        ra = results_a.get(eid)
        rb = results_b.get(eid)
        if ra is None or rb is None:
            print(f"  {eid:<12} {'n/a':>8} {'n/a':>8} {'NEW/REMOVED':>10}")
            continue
        pa = "PASS" if ra["overall_pass"] else "FAIL"
        pb = "PASS" if rb["overall_pass"] else "FAIL"
        if pa == pb:
            change = "—"
            unchanged += 1
        elif pa == "FAIL" and pb == "PASS":
            change = f"{GREEN}▲ fixed{RESET}"
            improved += 1
        else:
            change = f"{RED}▼ broke{RESET}"
            regressed += 1
        print(f"  {eid:<12} {pa:>8} {pb:>8} {change:>8}")

    print(f"\n  Improved:  {improved}")
    print(f"  Regressed: {regressed}")
    print(f"  Unchanged: {unchanged}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Free Claude Agent — Eval Runner")
    parser.add_argument("--category",  type=str, default=None,
                        help="Run only this category: tool_use | memory | reasoning | planning")
    parser.add_argument("--id",        type=str, default=None,
                        help="Run only a single eval by ID, e.g. tool_001")
    parser.add_argument("--no-judge",  action="store_true",
                        help="Skip MiniMax judge (hard checks only, no NVIDIA key needed)")
    parser.add_argument("--delay",     type=float, default=3.0,
                        help="Seconds to sleep between eval calls (default: 3)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print full agent response for each eval")
    parser.add_argument("--compare",   nargs=2, metavar="FILE",
                        help="Compare two result JSON files instead of running evals")
    args = parser.parse_args()

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # ── Check required env vars ───────────────────────────────────────────────
    if not os.getenv("GROQ_API_KEY"):
        print(f"{RED}Error: GROQ_API_KEY not set.{RESET}")
        sys.exit(1)

    use_judge = not args.no_judge
    if use_judge and not os.getenv("NVIDIA_API_KEY"):
        print(f"{YELLOW}Warning: NVIDIA_API_KEY not set — running hard checks only (no judge).{RESET}")
        use_judge = False

    # ── Load evals ────────────────────────────────────────────────────────────
    eval_files = {
        "tool_use":  EVALS_DIR / "tool_use.json",
        "memory":    EVALS_DIR / "memory.json",
        "reasoning": EVALS_DIR / "reasoning.json",
        "planning":  EVALS_DIR / "planning.json",
    }

    all_evals = []
    for cat, path in eval_files.items():
        if args.category and cat != args.category:
            continue
        if not path.exists():
            print(f"{YELLOW}Warning: {path} not found — skipping.{RESET}")
            continue
        with open(path) as f:
            evals = json.load(f)
        if args.id:
            evals = [e for e in evals if e["id"] == args.id]
        all_evals.extend(evals)

    if not all_evals:
        print(f"{RED}No evals found. Check --category / --id flags.{RESET}")
        sys.exit(1)

    # ── Print run config ──────────────────────────────────────────────────────
    print(f"\n{BOLD}🤖 Free Claude Agent — Eval Runner{RESET}")
    print(f"{'─'*60}")
    print(f"  Evals to run:  {len(all_evals)}")
    print(f"  Delay:         {args.delay}s between calls")
    print(f"  Judge:         {'MiniMax M2.7 via NVIDIA NIM' if use_judge else 'disabled'}")
    print(f"  Category:      {args.category or 'all'}")
    if args.id:
        print(f"  ID filter:     {args.id}")
    est_min = round(len(all_evals) * (args.delay + 10) / 60, 1)
    print(f"  Est. runtime:  ~{est_min} min")
    print(f"{'─'*60}")

    # ── Set up agent and judge ────────────────────────────────────────────────
    print(f"\nLoading agent...")
    agent = load_agent()
    print(f"Agent loaded. ✅")

    judge_client = build_judge_client() if use_judge else None
    if use_judge:
        print(f"Judge client ready (NVIDIA NIM / MiniMax M2.7). ✅\n")

    # ── NVIDIA NIM smoke test ─────────────────────────────────────────────────
    if use_judge and judge_client:
        print(f"Running NVIDIA NIM smoke test...")
        test_result = judge_response(judge_client, {
            "input": "What is 2+2?",
            "judge_rubric": "Did the agent answer 4?"
        }, "The answer is 4.")
        if test_result["score"] >= 0:
            print(f"NVIDIA NIM smoke test passed. Score: {test_result['score']}/3 ✅\n")
        else:
            print(f"{RED}NVIDIA NIM smoke test FAILED: {test_result['reason']}{RESET}")
            print(f"Continuing with hard checks only.\n")
            use_judge = False
            judge_client = None

    # ── Run evals ─────────────────────────────────────────────────────────────
    all_results = []
    for i, ev in enumerate(all_evals):
        try:
            result = run_single_eval(agent, ev, judge_client, use_judge, args.verbose)
            all_results.append(result)
        except Exception as e:
            print(f"{RED}Error on {ev['id']}: {e}{RESET}")
            traceback.print_exc()
            all_results.append({
                "id": ev["id"],
                "category": ev["category"],
                "description": ev["description"],
                "input": ev["input"],
                "response": f"[Runner error: {e}]",
                "tools_called": [],
                "elapsed_s": 0,
                "hard": {"hard_pass": False},
                "judge": {"score": -1, "reason": str(e), "judge_pass": None},
                "overall_pass": False,
            })

        # Delay between calls (not after the last one)
        if i < len(all_evals) - 1:
            print(f"  Sleeping {args.delay}s...")
            time.sleep(args.delay)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_file = RESULTS_DIR / f"{timestamp}.json"
    payload = {
        "timestamp":   timestamp,
        "delay_s":     args.delay,
        "judge_model": "minimaxai/minimax-m2.7" if use_judge else "disabled",
        "total":       len(all_results),
        "passed":      sum(1 for r in all_results if r["overall_pass"]),
        "results":     all_results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved → {out_file}\n")


if __name__ == "__main__":
    main()
