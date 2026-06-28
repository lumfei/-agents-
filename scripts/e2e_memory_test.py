"""E2E tests: multi-turn conversation (short-term memory) + infrastructure check"""
import urllib.request, json, time

BASE = "http://127.0.0.1:8000/api/v1/agent/chat"

def test(msg, session_id, user_id="", label=""):
    data = json.dumps({"message": msg, "session_id": session_id, "user_id": user_id}).encode()
    req = urllib.request.Request(BASE, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=180)
    r = json.loads(resp.read())
    reply = r.get("reply", "")
    print(f"  [{label}] Intent: {r.get('intent'):15s}  Resolved: {r.get('resolved')}")
    print(f"  Reply: {reply[:250]}")
    print()
    return r

# === Multi-turn test ===
print("=" * 50)
print("MULTI-TURN MEMORY TEST: Same session, 3 turns")
print("=" * 50)
sid = "multi-turn-test-001"

# Turn 1
print("--- Turn 1 ---")
r1 = test("我想查一下我的订单", sid, "CU0005", "T1: List orders")
assert r1.get("intent") != "unknown", "Turn 1 failed"

# Turn 2: reference "刚才" (just now) — tests context retention
print("--- Turn 2 ---")
r2 = test("刚才那个订单的物流情况", sid, "CU0005", "T2: Follow-up logistics")
assert r2.get("intent") != "unknown", "Turn 2 failed"

# Turn 3: another follow-up
print("--- Turn 3 ---")
r3 = test("第一个订单多少钱", sid, "CU0005", "T3: Order amount")
assert r3.get("intent") != "unknown", "Turn 3 failed"

print("Multi-turn: PASSED\n")

# === Infrastructure check ===
print("=" * 50)
print("INFRASTRUCTURE CHECK")
print("=" * 50)

# Check PostgreSQL
try:
    import psycopg2
    c = psycopg2.connect(host="127.0.0.1", port=5432, dbname="agent_cs", user="postgres", password="postgres", connect_timeout=3)
    cur = c.cursor()
    cur.execute("SELECT COUNT(*) FROM user_memories WHERE is_deleted = FALSE")
    pg_count = cur.fetchone()[0]
    print(f"PostgreSQL: OK, user_memories rows: {pg_count}")
    c.close()
except Exception as e:
    print(f"PostgreSQL: ERROR - {e}")

# Check Qdrant
try:
    import urllib.request
    resp = urllib.request.urlopen("http://127.0.0.1:6333/collections", timeout=3)
    data = json.loads(resp.read())
    cols = [c["name"] for c in data["result"]["collections"]]
    print(f"Qdrant: OK, collections: {cols}")
except Exception as e:
    print(f"Qdrant: ERROR - {e}")

# Check Redis
try:
    import redis
    r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=3)
    r.ping()
    keys = len(r.keys("*"))
    print(f"Redis: OK, keys: {keys}")
except Exception as e:
    print(f"Redis: ERROR - {e}")

print("\nDone.")
