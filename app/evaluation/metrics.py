"""
离线评估框架

在 Golden Dataset 上运行 Agent 工作流，计算：
  - 意图分类准确率
  - 工具调用准确率（期望调用 + 禁止调用）
  - 关键词命中率

使用方式：
  from app.evaluation.metrics import run_evaluation
  report = run_evaluation("tests/golden_dataset.json")
  print(report.summary())
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from app.evaluation.golden_dataset import GoldenDataset, GoldenTestCase
from app.graph.supervisor_graph import run_workflow


@dataclass
class CaseResult:
    """单条用例的评估结果"""
    case_id: str
    passed: bool = True
    intent_match: bool = False
    actual_intent: str = ""
    expected_intent: str = ""
    tools_called: list[str] = field(default_factory=list)
    expected_tools_hit: list[str] = field(default_factory=list)
    expected_tools_miss: list[str] = field(default_factory=list)
    forbidden_tools_hit: list[str] = field(default_factory=list)
    keywords_hit: list[str] = field(default_factory=list)
    keywords_miss: list[str] = field(default_factory=list)
    final_response: str = ""
    error: str = ""


@dataclass
class EvalReport:
    """评估报告"""
    total: int = 0
    passed: int = 0
    intent_accuracy: float = 0.0
    tool_precision: float = 0.0       # 期望工具命中率
    tool_forbidden_violations: int = 0  # 禁止工具被调用的次数
    keyword_recall: float = 0.0
    results: list[CaseResult] = field(default_factory=list)
    failures: list[CaseResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "            Agent 离线评估报告",
            "=" * 50,
            f"  总用例:         {self.total}",
            f"  全部通过:       {self.passed}/{self.total}",
            f"  意图准确率:     {self.intent_accuracy:.1%}",
            f"  工具精确率:     {self.tool_precision:.1%}",
            f"  禁止工具违规:   {self.tool_forbidden_violations} 次",
            f"  关键词召回率:   {self.keyword_recall:.1%}",
            "=" * 50,
        ]
        if self.failures:
            lines.append("\n失败用例:")
            for r in self.failures:
                reasons = []
                if not r.intent_match:
                    reasons.append(f"意图不匹配(期望:{r.expected_intent}, 实际:{r.actual_intent})")
                if r.expected_tools_miss:
                    reasons.append(f"缺少工具:{r.expected_tools_miss}")
                if r.forbidden_tools_hit:
                    reasons.append(f"禁止工具被调用:{r.forbidden_tools_hit}")
                if r.keywords_miss:
                    reasons.append(f"缺少关键词:{r.keywords_miss}")
                if r.error:
                    reasons.append(f"执行错误:{r.error}")
                lines.append(f"  [{r.case_id}] {'; '.join(reasons)}")
        return "\n".join(lines)


def evaluate_case(case: GoldenTestCase) -> CaseResult:
    """评估单条用例"""
    result = CaseResult(
        case_id=case.id,
        expected_intent=case.expected_intent,
    )

    try:
        output = run_workflow(user_message=case.input)
    except Exception as e:
        result.error = str(e)
        result.passed = False
        return result

    # ── 1. 意图匹配 ──────────────────────────────────────────
    actual_intent = output.get("intent", "unknown")
    result.actual_intent = actual_intent
    result.intent_match = (actual_intent == case.expected_intent)

    # ── 2. 工具调用 ──────────────────────────────────────────
    tool_results = output.get("tool_results", [])
    tools_called = [t.get("name", "") for t in tool_results]
    result.tools_called = tools_called

    if case.expected_tools:
        result.expected_tools_hit = [t for t in case.expected_tools if t in tools_called]
        result.expected_tools_miss = [t for t in case.expected_tools if t not in tools_called]

    if case.forbidden_tools:
        result.forbidden_tools_hit = [t for t in case.forbidden_tools if t in tools_called]

    # ── 3. 关键词 ────────────────────────────────────────────
    final_response = output.get("final_response", "")
    result.final_response = final_response
    if case.expected_keywords:
        result.keywords_hit = [kw for kw in case.expected_keywords if kw in final_response]
        result.keywords_miss = [kw for kw in case.expected_keywords if kw not in final_response]

    # ── 4. 判定通过 ──────────────────────────────────────────
    result.passed = (
        result.intent_match
        and not result.expected_tools_miss
        and not result.forbidden_tools_hit
        and not result.error
    )

    return result


def run_evaluation(dataset_path: str, category: str = "", workers: int = 5) -> EvalReport:
    """
    在黄金测试集上运行完整评估。

    参数：
      dataset_path: 黄金测试集 JSON 文件路径
      category:     限定类别（空 = 全部）
      workers:      并发数（默认 5，避免 API 限流）

    返回：
      EvalReport 对象，含 .summary() 和 .failures
    """
    ds = GoldenDataset.load(dataset_path)
    if category:
        ds = ds.filter(category=category)

    # 并发执行，保持原始顺序
    cases = list(ds)
    results_map: dict[str, CaseResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_case, case): case.id for case in cases}
        for future in as_completed(futures):
            result = future.result()
            results_map[result.case_id] = result
    results = [results_map[c.id] for c in cases]

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    # 意图准确率
    intent_cases = [r for r in results if r.expected_intent]
    intent_ok = sum(1 for r in intent_cases if r.intent_match)
    intent_accuracy = intent_ok / len(intent_cases) if intent_cases else 1.0

    # 工具精确率（命中的期望工具 / 总期望工具）
    total_expected = sum(len(r.expected_tools_hit) + len(r.expected_tools_miss) for r in results)
    total_hit = sum(len(r.expected_tools_hit) for r in results)
    tool_precision = total_hit / total_expected if total_expected > 0 else 1.0

    # 禁止工具违规
    forbidden_violations = sum(len(r.forbidden_tools_hit) for r in results)

    # 关键词召回率
    total_keywords = sum(len(r.keywords_hit) + len(r.keywords_miss) for r in results)
    total_kw_hit = sum(len(r.keywords_hit) for r in results)
    keyword_recall = total_kw_hit / total_keywords if total_keywords > 0 else 1.0

    failures = [r for r in results if not r.passed]

    return EvalReport(
        total=total,
        passed=passed,
        intent_accuracy=intent_accuracy,
        tool_precision=tool_precision,
        tool_forbidden_violations=forbidden_violations,
        keyword_recall=keyword_recall,
        results=results,
        failures=failures,
    )
