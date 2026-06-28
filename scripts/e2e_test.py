"""E2E tests for the agent API"""
import urllib.request, json, sys

BASE = "http://127.0.0.1:8000/api/v1/agent/chat"

def test(msg, label, user_id=""):
    data = json.dumps({"message": msg, "session_id": "e2e-" + label.replace(" ", "_"), "user_id": user_id}).encode()
    req = urllib.request.Request(BASE, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=180)
    r = json.loads(resp.read())
    reply = r.get("reply", "")
    status = "FAIL" if "校验未通过" in reply or "error" in reply.lower() else "PASS"
    print(f"[{status}] {label}")
    print(f"  Intent: {r.get('intent'):15s}  Path: {r.get('agent_path')}")
    print(f"  Reply: {reply[:200]}")
    print()
    return status

results = []

# Test 1: Original bug - query order ORD00001
results.append(test("ORD00001", "Test1: Query ORD00001 (anonymous)"))

# Test 2: Logistics tracking
results.append(test("SF1234567890到哪了", "Test2: Tracking by tracking_no"))

# Test 3: Tech support
results.append(test("电脑蓝屏了怎么办", "Test3: Tech support"))

# Test 4: Order with explicit user_id
results.append(test("ORD00001", "Test4: Query with user_id", user_id="CU0015"))

print(f"\n{'='*40}")
print(f"Results: {results.count('PASS')}/{len(results)} passed")
for i, s in enumerate(results):
    print(f"  Test {i+1}: {s}")
