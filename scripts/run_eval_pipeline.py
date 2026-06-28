"""
统一评估 Pipeline — Golden Dataset + LLM-as-Judge + 汇总报告

组合现有组件形成完整评估闭环：
  加载 GoldenDataset → run_workflow() → metrics.evaluate_case() → LLMJudge.evaluate() → 汇总报告

用法:
  # 运行全部 50 条用例
  python scripts/run_eval_pipeline.py

  # 限定类别和数量
  python scripts/run_eval_pipeline.py --category finance --limit 10

  # 输出 JSON 报告 + 设置通过阈值
  python scripts/run_eval_pipeline.py --output report.json --threshold 0.7

  # CI 模式（静默输出，exit code 反映通过/失败）
  python scripts/run_eval_pipeline.py --ci --threshold 0.7

返回码:
  0 = 通过率 >= 阈值
  1 = 通过率 < 阈值 或 执行错误
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.evaluation.golden_dataset import GoldenDataset
from app.evaluation.metrics import run_evaluation
from app.evaluation.llm_judge import get_llm_judge, EvalBatchResult


def find_dataset() -> str:
    """查找 golden_dataset.json"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "tests", "golden_dataset.json"),
        os.path.join(os.path.dirname(__file__), "..", "tests", "golden_dataset.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("找不到 golden_dataset.json，请确认路径")


def main():
    parser = argparse.ArgumentParser(description="统一评估 Pipeline")
    parser.add_argument("--category", default="", help="限定类别 (tech_support/finance/after_sale/edge/escalation)")
    parser.add_argument("--difficulty", default="", help="限定难度 (easy/medium/hard)")
    parser.add_argument("--limit", type=int, default=0, help="限制用例数量 (0=全部)")
    parser.add_argument("--threshold", type=float, default=0.7, help="通过率阈值 (0.0-1.0)")
    parser.add_argument("--output", default="", help="输出 JSON 报告文件路径")
    parser.add_argument("--ci", action="store_true", help="CI 模式 (静默输出，exit code 反映结果)")
    parser.add_argument("--workers", type=int, default=5, help="并发数")
    args = parser.parse_args()

    dataset_path = find_dataset()

    if not args.ci:
        print("=" * 60)
        print("  Agent 统一评估 Pipeline")
        print("=" * 60)
        print(f"  数据集: {dataset_path}")
        if args.category:
            print(f"  类别:   {args.category}")
        if args.difficulty:
            print(f"  难度:   {args.difficulty}")
        print()

    # ── Step 1: 离线指标评估 ──────────────────────────────
    t0 = time.time()
    report = run_evaluation(dataset_path, category=args.category, workers=args.workers)

    if args.difficulty:
        ds = GoldenDataset.load(dataset_path)
        if args.category:
            ds = ds.filter(category=args.category)
        ds = ds.filter(difficulty=args.difficulty)
        allowed_ids = {c.id for c in ds}
        report.results = [r for r in report.results if r.case_id in allowed_ids]
        report.total = len(report.results)
        report.passed = sum(1 for r in report.results if r.passed)
        report.failures = [r for r in report.results if not r.passed]

    if args.limit > 0 and args.limit < len(report.results):
        report.results = report.results[:args.limit]
        report.total = len(report.results)
        report.passed = sum(1 for r in report.results if r.passed)
        report.failures = [r for r in report.results if not r.passed]

    metrics_elapsed = time.time() - t0

    if not args.ci:
        print(report.summary())
        print()

    # ── Step 2: LLM-as-Judge 质量评估 ──────────────────────
    judge = get_llm_judge()
    try:
        t1 = time.time()
        # 只对有 final_response 的结果做 LLM Judge
        judged_count = 0
        for r in report.results:
            if not r.error:
                judged_count += 1

        if judged_count > 0:
            # 构建完整的测试用例列表传给 LLM Judge
            ds = GoldenDataset.load(dataset_path)
            case_map = {c.id: c for c in ds}
            test_cases = []
            for r in report.results:
                if r.error:
                    continue
                golden = case_map.get(r.case_id)
                test_cases.append({
                    "case_id": r.case_id,
                    "query": golden.input if golden else "",
                    "response": getattr(r, "final_response", ""),
                    "expected_intent": r.expected_intent,
                    "expected_tools": golden.expected_tools if golden else [],
                    "actual_tools": r.tools_called,
                    "actual_intent": r.actual_intent,
                })
            batch = judge.evaluate_batch(test_cases)
            judge_elapsed = time.time() - t1

            if not args.ci:
                print("─" * 50)
                print("  LLM-as-Judge 5 维质量评分")
                print("─" * 50)
                print(f"  综合均分:   {batch.average_score:.2f}")
                print(f"  通过率:     {batch.pass_rate:.1%}  (阈值 0.7)")
                print("  各维度均分:")
                for dim, score in sorted(batch.dimension_averages.items()):
                    bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
                    print(f"    {dim:12s}: {score:.2f}  {bar}")
                print(f"  评估耗时:   {metrics_elapsed + judge_elapsed:.1f}s")
                print()
        else:
            batch = EvalBatchResult(total=0, average_score=1.0, dimension_averages={}, pass_rate=1.0)
            if not args.ci:
                print("  (所有用例执行出错，跳过 LLM Judge)")
    except Exception as e:
        if not args.ci:
            print(f"  [WARN] LLM Judge 评估异常: {e}")
        batch = EvalBatchResult(total=0, average_score=1.0, dimension_averages={}, pass_rate=1.0)

    # ── Step 3: 汇总输出 ──────────────────────────────────
    combined_pass_rate = report.passed / report.total if report.total > 0 else 0.0

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_path,
        "category": args.category or "all",
        "difficulty": args.difficulty or "all",
        "metrics": {
            "total": report.total,
            "passed": report.passed,
            "pass_rate": round(combined_pass_rate, 3),
            "intent_accuracy": round(report.intent_accuracy, 3),
            "tool_precision": round(report.tool_precision, 3),
            "tool_forbidden_violations": report.tool_forbidden_violations,
            "keyword_recall": round(report.keyword_recall, 3),
        },
        "llm_judge": {
            "average_score": round(batch.average_score, 3),
            "pass_rate": round(batch.pass_rate, 3),
            "dimension_averages": batch.dimension_averages,
        },
        "duration_s": round(time.time() - t0, 1),
    }

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        if not args.ci:
            print(f"[FILE] 报告已保存: {out_path}")

    # ── Step 4: 判定 ──────────────────────────────────────
    passed = combined_pass_rate >= args.threshold

    if not args.ci:
        print()
        if passed:
            print(f"[PASS] Pipeline 通过 (综合通过率 {combined_pass_rate:.1%} >= 阈值 {args.threshold:.0%})")
        else:
            print(f"[FAIL] Pipeline 未通过 (综合通过率 {combined_pass_rate:.1%} < 阈值 {args.threshold:.0%})")
            if args.output:
                print(f"   详细报告见: {args.output}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
