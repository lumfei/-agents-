"""
DeepEval 黄金测试集评估

使用 DeepEval pytest 集成 + 自定义指标，在 50 条黄金用例上评估 Agent。

运行方式：
  deepeval test run tests/test_golden_eval.py -n 5   # 5 并发
  pytest tests/test_golden_eval.py -v                  # 普通 pytest
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall

from app.graph.supervisor_graph import run_workflow

# ── 加载黄金数据集 ──────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")


def load_cases() -> list[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
#  自定义 DeepEval 指标
# ═══════════════════════════════════════════════════════════════

class IntentAccuracyMetric(BaseMetric):
    """意图分类准确率 —— 规则判断，不调 LLM"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        meta = test_case.metadata or {}
        # Support both single intent and list of acceptable intents
        expected = meta.get("expected_intent", "")
        expected_list = meta.get("expected_intents", [])
        actual = meta.get("actual_intent", "")
        if expected_list:
            self.score = 1.0 if actual in expected_list else 0.0
            self.reason = f"期望: {expected_list}, 实际: {actual}"
        else:
            self.score = 1.0 if actual == expected else 0.0
            self.reason = f"期望: {expected}, 实际: {actual}"
        self.success = self.score >= self.threshold

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Intent Accuracy"


class ToolPrecisionMetric(BaseMetric):
    """工具调用精确率 —— 期望工具命中 + 禁止工具不违规"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def is_successful(self) -> bool:
        return self.success

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        meta = test_case.metadata or {}
        expected = meta.get("expected_tools", [])
        forbidden = meta.get("forbidden_tools", [])
        raw_tools = test_case.tools_called or []
        actual = [t.name for t in raw_tools]

        expected_hit = sum(1 for t in expected if t in actual)
        expected_miss = [t for t in expected if t not in actual]
        forbidden_hit = [t for t in forbidden if t in actual]

        has_expected = len(expected) > 0
        has_forbidden = len(forbidden) > 0

        if not has_expected and not has_forbidden:
            self.score = 1.0
            self.reason = "无工具期望约束"
        elif not has_expected:
            # 只检查禁止工具
            self.score = 0.0 if forbidden_hit else 1.0
        elif not has_forbidden:
            # 只检查期望工具
            self.score = expected_hit / len(expected)
        else:
            # 两部分都要通过
            expected_score = expected_hit / len(expected)
            forbidden_score = 0.0 if forbidden_hit else 1.0
            self.score = min(expected_score, forbidden_score)

        self.success = self.score >= self.threshold
        parts = []
        if expected_miss:
            parts.append(f"缺少工具: {expected_miss}")
        if forbidden_hit:
            parts.append(f"违规工具: {forbidden_hit}")
        if not expected_miss and not forbidden_hit:
            parts.append("工具调用全部符合预期")
        parts.append(f"实际调用: {actual}")
        self.reason = "; ".join(parts)


class KeywordRecallMetric(BaseMetric):
    """关键词召回率"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def is_successful(self) -> bool:
        return self.success

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        meta = test_case.metadata or {}
        expected_kw = meta.get("expected_keywords", [])
        actual_output = test_case.actual_output or ""

        if not expected_kw:
            self.score = 1.0
            self.reason = "无关键词期望约束"
        else:
            hit = sum(1 for kw in expected_kw if kw in actual_output)
            self.score = hit / len(expected_kw)
            self.reason = f"命中 {hit}/{len(expected_kw)}: {expected_kw}"

        self.success = self.score >= self.threshold


class SafetyMetric(BaseMetric):
    """安全检查 —— 禁止工具违规次数"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def is_successful(self) -> bool:
        return self.success

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        meta = test_case.metadata or {}
        forbidden = meta.get("forbidden_tools", [])
        raw_tools = test_case.tools_called or []
        actual = [t.name for t in raw_tools]

        violations = [t for t in forbidden if t in actual]
        self.score = 0.0 if violations else 1.0
        self.success = self.score >= self.threshold
        if violations:
            self.reason = f"安全违规: 调用了禁止工具 {violations}"
        else:
            self.reason = "无安全违规"


# ═══════════════════════════════════════════════════════════════
#  评测辅助函数
# ═══════════════════════════════════════════════════════════════

def run_case(user_input: str) -> dict:
    """运行一条用例，返回标准化结果"""
    result = run_workflow(user_message=user_input)
    tool_results = result.get("tool_results", [])
    tools_called = [t.get("name", "") for t in tool_results]
    return {
        "actual_output": result.get("final_response", ""),
        "tools_called_raw": tools_called,
        "tools_called": [ToolCall(name=t) for t in tools_called] if tools_called else None,
        "actual_intent": result.get("intent", "unknown"),
    }


def build_test_case(case: dict, output: dict) -> LLMTestCase:
    """构建 DeepEval 测试用例"""
    expected_tools_raw = case.get("expected_tools", [])
    return LLMTestCase(
        input=case["input"],
        actual_output=output["actual_output"],
        tools_called=output.get("tools_called"),
        expected_tools=[ToolCall(name=t) for t in expected_tools_raw] if expected_tools_raw else None,
        metadata={
            "expected_intent": case.get("expected_intent", ""),
            "expected_intents": case.get("expected_intents", []),
            "actual_intent": output["actual_intent"],
            "expected_tools": case.get("expected_tools", []),
            "forbidden_tools": case.get("forbidden_tools", []),
            "expected_keywords": case.get("expected_keywords", []),
        },
    )


# ═══════════════════════════════════════════════════════════════
#  pytest 参数化测试
# ═══════════════════════════════════════════════════════════════

def _case_id(case: dict) -> str:
    return f"{case['id']}-{case['difficulty']}"


@pytest.mark.parametrize("case", load_cases(), ids=_case_id)
def test_golden_case(case: dict):
    """逐条跑黄金用例，DeepEval 多指标评估"""
    output = run_case(case["input"])
    test_case = build_test_case(case, output)

    intent_metric = IntentAccuracyMetric()
    tool_metric = ToolPrecisionMetric()
    keyword_metric = KeywordRecallMetric()
    safety_metric = SafetyMetric()

    assert_test(test_case, [intent_metric, tool_metric, keyword_metric, safety_metric], run_async=False)
