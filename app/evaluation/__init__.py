"""评估模块 — 黄金测试集 + LLM-as-Judge + 指标 + 漂移检测"""

from app.evaluation.golden_dataset import GoldenDataset, GoldenTestCase
from app.evaluation.llm_judge import LLMJudge, get_llm_judge, JudgeResult, DriftReport, EvalBatchResult
from app.evaluation.metrics import run_evaluation, evaluate_case, EvalReport, CaseResult

__all__ = [
    # Golden Dataset
    "GoldenDataset",
    "GoldenTestCase",
    # LLM Judge
    "LLMJudge",
    "get_llm_judge",
    "JudgeResult",
    "DriftReport",
    "EvalBatchResult",
    # Metrics
    "run_evaluation",
    "evaluate_case",
    "EvalReport",
    "CaseResult",
]
