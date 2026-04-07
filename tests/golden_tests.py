"""
Golden test suite for regression testing.
Run: python -m tests.golden_tests
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import FreeAgent

GOLDEN_TESTS = [
    {
        "name": "simple_math",
        "input": "What is 15% of 240?",
        "expected_contains": ["36"],
        "expected_not_contains": [],
        "is_critical": True,
        "category": "math"
    },
    {
        "name": "reasoning_syllogism",
        "input": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
        "expected_contains": ["yes"],
        "expected_not_contains": [],
        "is_critical": True,
        "category": "reasoning"
    },
    {
        "name": "code_debug",
        "input": "Fix this Python: for i in range(len(my_list)+1): print(my_list[i])",
        "expected_contains": ["range"],
        "expected_not_contains": [],
        "is_critical": True,
        "category": "code"
    },
    {
        "name": "caveman_compression",
        "input": "Tell me about Python decorators",
        "expected_contains": ["decorator"],
        "expected_not_contains": ["I'd be happy to", "Sure!", "Let me"],
        "is_critical": True,
        "category": "style"
    },
    {
        "name": "explanation",
        "input": "Explain quantum entanglement simply",
        "expected_contains": ["particle"],
        "expected_not_contains": [],
        "is_critical": False,
        "category": "explanation"
    }
]


def run_test(agent: FreeAgent, test: dict) -> dict:
    try:
        response = agent.ask(
            test["input"],
            use_reasoning=(test["category"] in ["reasoning", "code"])
        )
        response_lower = response.lower()
        passed = True
        failures = []

        for expected in test["expected_contains"]:
            if expected.lower() not in response_lower:
                passed = False
                failures.append(f"Missing: '{expected}'")

        for not_expected in test["expected_not_contains"]:
            if not_expected.lower() in response_lower:
                passed = False
                failures.append(f"Found unexpected: '{not_expected}'")

        return {
            "name": test["name"],
            "passed": passed,
            "failures": failures,
            "snippet": response[:150]
        }

    except Exception as e:
        return {
            "name": test["name"],
            "passed": False,
            "failures": [f"Exception: {str(e)}"],
            "snippet": ""
        }


def run_all_tests(api_key: str = None) -> dict:
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set")
        return {"error": "Missing API key"}

    agent = FreeAgent(
        api_key=api_key,
        daily_token_limit=10000,
        enable_reasoning=True,
        enable_code_execution=False
    )

    results = []
    critical_failed = []

    print(f"🧪 Running {len(GOLDEN_TESTS)} golden tests...\n")

    for test in GOLDEN_TESTS:
        result = run_test(agent, test)
        results.append(result)
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} {test['name']} ({test['category']})")
        for failure in result["failures"]:
            print(f"   └─ {failure}")
        if not result["passed"] and test["is_critical"]:
            critical_failed.append(test["name"])

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n📊 Results: {passed_count}/{len(results)} passed")

    if critical_failed:
        print(f"⚠️  Critical failures: {', '.join(critical_failed)}")

    return {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "critical_failed": critical_failed,
        "results": results
    }


if __name__ == "__main__":
    results = run_all_tests()
    if results.get("critical_failed"):
        print(f"\n❌ Critical tests failed: {results['critical_failed']}")
        sys.exit(1)
    elif results.get("error"):
        sys.exit(2)
    else:
        print("\n✅ All critical tests passed!")
        sys.exit(0)
