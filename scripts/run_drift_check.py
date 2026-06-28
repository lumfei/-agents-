"""
行为漂移检测脚本 — CI 可用的漂移检测

用法:
  # 设置基线
  python scripts/run_drift_check.py --set-baseline v1 --limit 20

  # 检测漂移（对比当前评估结果与基线）
  python scripts/run_drift_check.py --baseline v1 --limit 20

  # 输出 JSON 报告
  python scripts/run_drift_check.py --baseline v1 --output drift_report.json

返回码:
  0 = 未检测到显著漂移
  1 = 检测到显著漂移（需要人工复核）
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.evaluation.llm_judge import get_llm_judge, EvalBatchResult
from app.evaluation.metrics import run_evaluation
from app.evaluation.golden_dataset import GoldenDataset


def find_dataset() -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "tests", "golden_dataset.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("找不到 golden_dataset.json")


def run_current_eval(dataset_path: str, category: str, limit: int) -> EvalBatchResult:
    """运行当前评估并返回 EvalBatchResult"""
    judge = get_llm_judge()
    report = run_evaluation(dataset_path, category=category)

    if limit > 0:
        report.results = report.results[:limit]
        report.total = len(report.results)
        report.passed = sum(1 for r in report.results if r.passed)

    # 构建 LLM Judge 输入
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
            "response": r.final_response,
            "expected_tools": golden.expected_tools if golden else [],
            "actual_tools": r.tools_called,
            "actual_intent": r.actual_intent,
        })

    if test_cases:
        return judge.evaluate_batch(test_cases)
    return EvalBatchResult(total=0, average_score=1.0, dimension_averages={}, pass_rate=1.0)


def main():
    parser = argparse.ArgumentParser(description="行为漂移检测")
    parser.add_argument("--set-baseline", default="", help="设置基线版本名称（如 v1）")
    parser.add_argument("--baseline", default="", help="对比的基线版本名称")
    parser.add_argument("--category", default="", help="限定类别")
    parser.add_argument("--limit", type=int, default=0, help="限制用例数")
    parser.add_argument("--output", default="", help="输出 JSON 报告路径")
    args = parser.parse_args()

    dataset_path = find_dataset()
    judge = get_llm_judge()

    if args.set_baseline:
        # ── 设置基线模式 ──────────────────────────────────
        version = args.set_baseline
        print(f"建立基线版本: {version}")
        batch = run_current_eval(dataset_path, args.category, args.limit)
        judge.set_baseline(version, batch)
        path = judge.save_baseline(version)
        print(f"基线已保存: {path}")
        print(f"  总用例:     {batch.total}")
        print(f"  综合均分:   {batch.average_score:.3f}")
        print(f"  通过率:     {batch.pass_rate:.1%}")
        print(f"  维度均分:   {batch.dimension_averages}")
        sys.exit(0)

    if args.baseline:
        # ── 漂移检测模式 ──────────────────────────────────
        version = args.baseline
        print(f"检测漂移 vs 基线: {version}")

        # 加载基线
        baseline = judge.load_baseline(version)
        if baseline is None:
            print(f"[ERROR] 基线 {version} 不存在，请先运行 --set-baseline {version}")
            sys.exit(1)

        # 运行当前评估
        current = run_current_eval(dataset_path, args.category, args.limit)

        # 检测漂移
        report = judge.detect_drift(
            current=current,
            baseline_version=version,
            current_version=f"{version}-check-{int(time.time())}",
        )

        # 输出报告
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_version": version,
            "baseline_avg_score": baseline.average_score,
            "baseline_pass_rate": baseline.pass_rate,
            "current_avg_score": current.average_score,
            "current_pass_rate": current.pass_rate,
            "overall_drift_score": report.overall_drift_score,
            "dimension_drifts": report.dimension_drifts,
            "style_changes": report.style_changes,
            "threshold_changes": report.threshold_changes,
            "requires_review": report.requires_review,
            "recommendations": report.recommendations,
        }

        print()
        print("-" * 50)
        print("  行为漂移检测报告")
        print("-" * 50)
        print(f"  基线 ({version}):        avg={baseline.average_score:.3f}, pass_rate={baseline.pass_rate:.1%}")
        print(f"  当前:                   avg={current.average_score:.3f}, pass_rate={current.pass_rate:.1%}")
        print(f"  综合漂移分数:           {report.overall_drift_score:.4f}")
        print(f"  需要人工复核:           {'是' if report.requires_review else '否'}")
        print()
        for dim, drift in sorted(report.dimension_drifts.items()):
            status = "!" if drift > 0.15 else " "
            print(f"    [{status}] {dim:14s}: drift={drift:.4f}")
        print()
        for rec in report.recommendations:
            print(f"  - {rec}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n[FILE] 报告已保存: {args.output}")

        if report.requires_review:
            print(f"\n[FAIL] 检测到显著行为漂移，需要人工复核")
            sys.exit(1)
        else:
            print(f"\n[PASS] 行为一致，未检测到显著漂移")
            sys.exit(0)

    # 无参数 → 显示帮助
    parser.print_help()
    print("\n示例:")
    print("  python scripts/run_drift_check.py --set-baseline v1 --limit 20")
    print("  python scripts/run_drift_check.py --baseline v1 --limit 20")


if __name__ == "__main__":
    main()
