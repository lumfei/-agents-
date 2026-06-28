"""
Run golden test cases through the workflow to generate observability data.

Usage: python scripts/run_eval_for_observability.py [--limit N]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

dataset_path = os.path.join(
    os.path.dirname(__file__), "..", "tests", "golden_dataset.json"
)
with open(dataset_path, "r", encoding="utf-8") as f:
    cases = json.load(f)

limit = int(sys.argv[2]) if len(sys.argv) >= 4 and sys.argv[1] == "--limit" else len(cases)
cases = cases[:limit]

from app.graph.supervisor_graph import run_workflow
from app.observability import get_cost_tracker, get_alert_manager

ct = get_cost_tracker()
am = get_alert_manager()
ct.reset()
am.reset()

ok = 0
fail = 0
start = time.time()

for i, case in enumerate(cases):
    msg = case.get("input", "")
    expected = case.get("expected_intent", "unknown")
    try:
        result = run_workflow(
            user_message=msg,
            user_id="u-{}".format((i % 5) + 1),
            session_id="sess-{:03d}".format(i),
        )
        actual = result.get("intent", "unknown")
        score = result.get("quality_score", 0)
        correct = actual == expected
        if correct:
            ok += 1
        else:
            fail += 1
        marker = "OK" if correct else "XX"
        print(
            "[{:02d}] {} exp={:15s} act={:15s} score={:.1f}".format(
                i + 1, marker, expected, actual, score
            )
        )
    except Exception as e:
        fail += 1
        print("[{:02d}] ERR {}".format(i + 1, str(e)[:80]))

elapsed = time.time() - start
print()
print("=" * 60)
print("Result: {}/{} correct, {} errors, {:.0f}s".format(ok, len(cases), fail, elapsed))
print()

s = ct.get_summary()
print("Total Requests:    {}".format(s["total_requests"]))
print("Total Input Tokens:  {:,}".format(s["total_input_tokens"]))
print("Total Output Tokens: {:,}".format(s["total_output_tokens"]))
print("Total Tokens:        {:,}".format(s["total_input_tokens"] + s["total_output_tokens"]))
print("Total Cost:         ${:.6f}".format(s["total_cost_usd"]))
print()
print("By Agent:")
for agent, d in sorted(s["by_agent"].items()):
    total_tok = d["input"] + d["output"]
    print(
        "  {:20s}: {:3d} req, {:6,} tok, ${:.6f}".format(
            agent, d["requests"], total_tok, d["cost"]
        )
    )

active = am.get_active_alerts()
print()
print("Active Alerts: {}".format(len(active)))
for a in active:
    print("  [{:8s}] {}: {}".format(a.severity.value.upper(), a.rule_name, a.message[:80]))

print()
print("Done. Now visit http://localhost:8000/api/v1/observability/ui")
