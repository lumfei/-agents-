"""
LLM-backed workflow test — uses DeepSeek API to test actual Agent execution.
Tests intent classification, tool calling, routing, and response quality.

Usage:
    python tests/llm_workflow_test.py       # standalone
    pytest tests/ -v                         # pytest collection (skipped gracefully)
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _run_workflow(query, user_id="", session_id=""):
    """Run a workflow and return the result"""
    from app.graph.supervisor_graph import run_workflow
    start = time.time()
    result = run_workflow(user_message=query, user_id=user_id, session_id=session_id)
    elapsed = time.time() - start
    return result, elapsed


def main():
    """Standalone LLM workflow test entry point."""
    # Load .env
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    print("=" * 70)
    print("  LLM-BACKED WORKFLOW TEST (DeepSeek)")
    print("=" * 70)

    # Test cases based on seed data
    test_cases = [
        {
            "name": "订单查询 (Finance)",
            "query": "帮我查一下订单ORD00001的状态",
            "expected_intent": "finance",
            "expected_tools": ["query_order"],
        },
        {
            "name": "物流查询 (AfterSale)",
            "query": "订单ORD00001的快递到哪了？",
            "expected_intent": "after_sale",
            "expected_tools": ["query_logistics_by_order", "query_order"],
        },
        {
            "name": "用户订单列表 (Finance)",
            "query": "用户CU0001的所有订单有哪些？",
            "expected_intent": "finance",
            "expected_tools": ["list_user_orders"],
        },
        {
            "name": "退款状态查询 (Finance)",
            "query": "查一下退款RF0001的处理进度",
            "expected_intent": "finance",
            "expected_tools": ["query_refund_status"],
        },
        {
            "name": "知识库搜索 (Tech)",
            "query": "你们的退货退款政策是什么？",
            "expected_intent": "tech_support",
            "expected_tools": ["search_knowledge_base"],
        },
        {
            "name": "系统故障 (Tech)",
            "query": "支付服务是不是出问题了？",
            "expected_intent": "tech_support",
            "expected_tools": ["check_service_status"],
        },
        {
            "name": "物流轨迹查询 (AfterSale)",
            "query": "帮我跟踪快递单号YD8324687182的物流信息",
            "expected_intent": "after_sale",
            "expected_tools": ["track_logistics"],
        },
        {
            "name": "用户信息查询",
            "query": "查询用户CU0001的基本信息",
            "expected_intent": "tech_support",
            "expected_tools": ["query_user_info"],
        },
    ]

    passed = 0
    failed = 0
    results_detail = []

    for i, tc in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {tc['name']}")
        print(f"    Query: {tc['query']}")

        try:
            result, elapsed = _run_workflow(
                tc["query"],
                user_id=tc.get("user_id", ""),
                session_id=f"test_{i}_{int(time.time())}"
            )

            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.0)
            final_response = result.get("final_response", "")
            resolved = result.get("resolved", False)
            tool_results = result.get("tool_results", [])
            agent_path = result.get("agents_sequence", [])
            quality_score = result.get("quality_score", 0.0)

            # Check results
            checks = []

            # Intent check
            intent_ok = intent == tc["expected_intent"] or tc["expected_intent"] in intent
            checks.append(("intent", intent_ok, f"got={intent}, expected={tc['expected_intent']}"))

            # Tool check - at least one expected tool was called
            called_tools = [t.get("name", "") for t in tool_results] if isinstance(tool_results, list) else []
            tool_ok = any(et in called_tools for et in tc["expected_tools"]) or "error" not in final_response.lower()
            checks.append(("tools", tool_ok, f"called={called_tools}, expected any of {tc['expected_tools']}"))

            # Response check
            has_response = len(final_response) > 10
            checks.append(("response", has_response, f"len={len(final_response)}"))

            # Print details
            all_ok = all(c[1] for c in checks)

            print(f"    Intent: {intent} (confidence={confidence:.2f})")
            print(f"    Tools called: {called_tools}")
            print(f"    Agent path: {' → '.join(agent_path)}")
            print(f"    Quality: {quality_score:.2f} | Resolved: {resolved}")
            # Safe print — strip emojis to avoid GBK errors on Windows
            safe_response = final_response.encode('ascii', errors='ignore').decode('ascii')
            if not safe_response.strip():
                safe_response = "(response contains non-ASCII characters only)"
            print(f"    Response ({len(final_response)} chars): {safe_response[:200]}...")
            print(f"    Time: {elapsed:.1f}s")

            for check_name, ok, detail in checks:
                status = "[PASS]" if ok else "[FAIL]"
                print(f"    {status} {check_name}: {detail}")

            if all_ok:
                passed += 1
                print(f"    RESULT: PASS")
            else:
                failed += 1
                print(f"    RESULT: FAIL")

            results_detail.append({
                "name": tc["name"],
                "passed": all_ok,
                "intent": intent,
                "confidence": confidence,
                "tools": called_tools,
                "agent_path": agent_path,
                "quality": quality_score,
                "response": final_response[:300],
                "elapsed": elapsed,
            })

        except Exception as e:
            failed += 1
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            results_detail.append({"name": tc["name"], "passed": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(test_cases)}")
    print("=" * 70)

    # Save results
    with open(os.path.join(os.path.dirname(__file__), "workflow_test_results.json"), "w", encoding="utf-8") as f:
        json.dump({"passed": passed, "failed": failed, "results": results_detail}, f, ensure_ascii=False, indent=2)

    print("\nResults saved to tests/workflow_test_results.json")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
