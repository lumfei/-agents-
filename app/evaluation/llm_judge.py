"""
LLM-as-Judge 评估模块 + 行为漂移检测

职责：
  - 使用 LLM 作为裁判，评估 Agent 回复质量
  - 多维评分体系：准确性、完整性、相关性、安全性、有用性
  - 行为漂移检测：对比历史基线，检测模型升级后的行为变化
  - 返回结构化评分 + 评估理由

评分维度：
  - Accuracy（准确性）：回答是否正确
  - Completeness（完整性）：是否回答了所有问题
  - Relevance（相关性）：是否与问题相关
  - Safety（安全性）：是否包含不安全内容
  - Helpfulness（有用性）：是否对用户有帮助

行为漂移检测：
  - 输出风格变化（长度、语气、专业度）
  - 拒绝阈值变化（什么程度的问题开始拒绝回答）
  - 工具选择偏好变化（不同Agent对不同工具的偏好）
  - 方法：周期性 LLM-as-Judge 评估 vs 历史基线
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class JudgeResult:
    """单次 LLM-as-Judge 评估结果"""
    overall_score: float = 0.0          # 综合评分 0.0-1.0
    dimensions: dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.0,
        "completeness": 0.0,
        "relevance": 0.0,
        "safety": 0.0,
        "helpfulness": 0.0,
    })
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestion: str = ""
    evaluation_time_ms: float = 0.0
    raw_response: str = ""


@dataclass
class DriftReport:
    """行为漂移检测报告"""
    baseline_version: str = ""          # 基线版本标识
    current_version: str = ""           # 当前版本标识
    overall_drift_score: float = 0.0    # 综合漂移分数 0.0-1.0（越高漂移越大）
    dimension_drifts: dict[str, float] = field(default_factory=dict)  # 各维度漂移
    style_changes: list[str] = field(default_factory=list)     # 风格变化描述
    threshold_changes: list[str] = field(default_factory=list) # 拒绝阈值变化
    tool_preference_changes: list[str] = field(default_factory=list)  # 工具偏好变化
    requires_review: bool = False       # 是否需要人工复核
    recommendations: list[str] = field(default_factory=list)


@dataclass
class EvalBatchResult:
    """批量评估结果"""
    total: int = 0
    average_score: float = 0.0
    dimension_averages: dict[str, float] = field(default_factory=dict)
    results: list[JudgeResult] = field(default_factory=list)
    pass_rate: float = 0.0  # score >= 0.7


# ═══════════════════════════════════════════════════════════════
#  LLM Judge
# ═══════════════════════════════════════════════════════════════

class LLMJudge:
    """
    LLM-as-Judge 评估器。

    使用方式：
      judge = LLMJudge(llm_client)
      result = judge.evaluate(
          query="我的电脑蓝屏了怎么办",
          response="请尝试重启电脑并检查驱动...",
          expected_tools=["search_knowledge_base"],
      )
      print(f"综合评分: {result.overall_score:.2f}")
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client
        # 历史基线（用于漂移检测）
        self._baselines: dict[str, EvalBatchResult] = {}

    def _get_evaluation_prompt(self) -> str:
        """从 PromptRegistry 获取评估 Prompt（支持 YAML 版本管理）。"""
        from app.prompts.registry import get_prompt_registry
        return get_prompt_registry().get_evaluation_prompt()

    def set_llm(self, llm_client) -> None:
        self._llm = llm_client

    # ── 单条评估 ────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        response: str,
        expected_tools: list[str] | None = None,
        actual_tools: list[str] | None = None,
        intent: str = "",
    ) -> JudgeResult:
        """
        评估单条 Agent 回复。

        参数：
          query:          用户原始问题
          response:       Agent 最终回复
          expected_tools: 期望调用的工具列表
          actual_tools:   实际调用的工具列表
          intent:         意图分类结果
        """
        if not self._llm:
            return self._mock_evaluate(query, response)

        prompt = self._get_evaluation_prompt().format(
            query=query[:500],
            response=response[:1000],
            expected_tools=expected_tools or [],
            actual_tools=actual_tools or [],
            intent=intent or "unknown",
        )

        start = time.time()
        try:
            messages = [
                SystemMessage(content="你是专业的 AI 质量评估裁判。请严格按 JSON 格式输出评估结果。"),
                HumanMessage(content=prompt),
            ]
            raw = self._llm.invoke(messages)
            raw_text = raw.content if hasattr(raw, "content") else str(raw)
            parsed = self._parse_judge_output(raw_text)
            parsed.evaluation_time_ms = (time.time() - start) * 1000
            parsed.raw_response = raw_text
            return parsed
        except Exception as e:
            logger.warning("LLM-as-Judge 评估失败: %s", e)
            return JudgeResult(
                overall_score=0.0,
                weaknesses=[f"评估失败: {str(e)}"],
                evaluation_time_ms=(time.time() - start) * 1000,
            )

    def evaluate_batch(self, test_cases: list[dict]) -> EvalBatchResult:
        """
        批量评估。

        参数：
          test_cases: [{"query": "...", "response": "...", "expected_tools": [...], ...}, ...]

        返回：
          EvalBatchResult 含各维度平均分、通过率等。
        """
        results: list[JudgeResult] = []
        for case in test_cases:
            r = self.evaluate(
                query=case.get("query", ""),
                response=case.get("response", ""),
                expected_tools=case.get("expected_tools"),
                actual_tools=case.get("actual_tools"),
                intent=case.get("intent", ""),
            )
            results.append(r)

        total = len(results)
        if total == 0:
            return EvalBatchResult()

        avg_score = sum(r.overall_score for r in results) / total
        dim_avgs = {}
        for dim in ["accuracy", "completeness", "relevance", "safety", "helpfulness"]:
            dim_avgs[dim] = sum(r.dimensions.get(dim, 0) for r in results) / total

        pass_count = sum(1 for r in results if r.overall_score >= 0.7)

        return EvalBatchResult(
            total=total,
            average_score=round(avg_score, 3),
            dimension_averages=dim_avgs,
            results=results,
            pass_rate=round(pass_count / total, 3),
        )

    # ── 行为漂移检测 ────────────────────────────────────────

    def set_baseline(self, version: str, baseline: EvalBatchResult) -> None:
        """设置历史基线"""
        self._baselines[version] = baseline
        logger.info("基线已设置: version=%s, avg_score=%.3f, total=%d",
                     version, baseline.average_score, baseline.total)

    def save_baseline(self, version: str, path: str | None = None) -> str:
        """持久化基线到磁盘 JSON 文件。返回保存路径。"""
        import os
        if path is None:
            data_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "drift",
            )
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(data_dir, f"baseline_{version}.json")
        baseline = self._baselines.get(version)
        if baseline is None:
            raise ValueError(f"基线版本 '{version}' 不存在于内存中，请先调用 set_baseline()")

        data = {
            "version": version,
            "timestamp": time.time(),
            "total": baseline.total,
            "average_score": baseline.average_score,
            "dimension_averages": baseline.dimension_averages,
            "pass_rate": baseline.pass_rate,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("基线已保存: version=%s → %s", version, path)

        # 同时更新历史记录
        history_path = os.path.join(os.path.dirname(path), "history.json")
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass
        history.append(data)
        # 保留最近 20 次
        if len(history) > 20:
            history = history[-20:]
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        return path

    def load_baseline(self, version: str, path: str | None = None) -> EvalBatchResult | None:
        """从磁盘加载基线到内存。返回 EvalBatchResult 或 None。"""
        import os
        if path is None:
            data_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "drift",
            )
            path = os.path.join(data_dir, f"baseline_{version}.json")

        if not os.path.exists(path):
            logger.warning("基线文件不存在: %s", path)
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        baseline = EvalBatchResult(
            total=data.get("total", 0),
            average_score=data.get("average_score", 0.0),
            dimension_averages=data.get("dimension_averages", {}),
            pass_rate=data.get("pass_rate", 0.0),
        )
        self._baselines[version] = baseline
        logger.info("基线已加载: version=%s, avg_score=%.3f", version, baseline.average_score)
        return baseline

    def list_baselines(self, data_dir: str | None = None) -> list[dict]:
        """列出所有已保存的基线版本"""
        import os
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "drift",
            )
        if not os.path.isdir(data_dir):
            return []
        baselines = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.startswith("baseline_") and fname.endswith(".json"):
                version = fname[len("baseline_"):-len(".json")]
                fpath = os.path.join(data_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    baselines.append({
                        "version": version,
                        "timestamp": data.get("timestamp", 0),
                        "average_score": data.get("average_score", 0),
                        "pass_rate": data.get("pass_rate", 0),
                        "total": data.get("total", 0),
                    })
                except Exception:
                    pass
        return baselines

    def detect_drift(
        self,
        current: EvalBatchResult,
        baseline_version: str = "",
        current_version: str = "",
    ) -> DriftReport:
        """
        检测行为漂移：对比当前评估结果与历史基线。

        漂移判定：
          - 综合分变化 > 0.1 → 显著漂移
          - 任一维度变化 > 0.15 → 显著漂移
          - 通过率变化 > 15% → 显著漂移
        """
        report = DriftReport(
            baseline_version=baseline_version,
            current_version=current_version,
        )

        # 查找基线
        baseline = None
        if baseline_version:
            baseline = self._baselines.get(baseline_version)
        elif self._baselines:
            # 取最新的基线
            latest_key = max(self._baselines.keys())
            baseline = self._baselines[latest_key]
            report.baseline_version = latest_key

        if baseline is None:
            report.overall_drift_score = 0.0
            report.recommendations = ["无历史基线可供对比，建议先建立基线"]
            return report

        # ── 各维度漂移计算 ──────────────────────────────────
        for dim in ["accuracy", "completeness", "relevance", "safety", "helpfulness"]:
            baseline_val = baseline.dimension_averages.get(dim, 0)
            current_val = current.dimension_averages.get(dim, 0)
            drift = abs(current_val - baseline_val)
            report.dimension_drifts[dim] = round(drift, 4)

            if drift > 0.15:
                direction = "上升" if current_val > baseline_val else "下降"
                report.style_changes.append(
                    f"{dim}: {baseline_val:.2f}→{current_val:.2f} ({direction} {drift:.0%})"
                )

        # ── 综合漂移分数 ────────────────────────────────────
        overall_drift = abs(current.average_score - baseline.average_score)
        report.overall_drift_score = round(overall_drift, 4)

        # 通过率变化
        pass_rate_change = abs(current.pass_rate - baseline.pass_rate)
        if pass_rate_change > 0.15:
            report.threshold_changes.append(
                f"通过率: {baseline.pass_rate:.0%}→{current.pass_rate:.0%} (变化 {pass_rate_change:.0%})"
            )

        # ── 需要人工复核的判定 ─────────────────────────────
        if overall_drift > 0.1 or any(d > 0.15 for d in report.dimension_drifts.values()):
            report.requires_review = True
            report.recommendations = [
                "检测到显著行为漂移，建议人工复核以下方面：",
            ]
            if overall_drift > 0.1:
                report.recommendations.append(
                    f"- 综合评分变化 {overall_drift:.0%}，检查模型升级是否影响输出质量"
                )
            for dim, drift in report.dimension_drifts.items():
                if drift > 0.15:
                    report.recommendations.append(
                        f"- {dim} 维度漂移 {drift:.0%}，检查相关 Prompt 和工具配置"
                    )
            if pass_rate_change > 0.15:
                report.recommendations.append(
                    f"- 通过率变化 {pass_rate_change:.0%}，检查是否有系统性质量问题"
                )
        else:
            report.recommendations = ["行为一致，未检测到显著漂移"]

        return report

    def compare_versions(
        self,
        version_a: str,
        version_b: str,
    ) -> DriftReport:
        """对比两个历史基线版本之间的漂移"""
        baseline_a = self._baselines.get(version_a)
        baseline_b = self._baselines.get(version_b)

        if not baseline_a or not baseline_b:
            return DriftReport(
                baseline_version=version_a,
                current_version=version_b,
                overall_drift_score=0.0,
                recommendations=["缺少基线数据"],
            )

        # 把 baseline_a 当作基线，baseline_b 当作当前
        return self.detect_drift(
            current=baseline_b,
            baseline_version=version_a,
            current_version=version_b,
        )

    # ── 内部方法 ────────────────────────────────────────────

    @staticmethod
    def _parse_judge_output(raw: str) -> JudgeResult:
        """解析 LLM 的 JSON 输出"""
        # 提取 JSON（处理可能的 markdown 包裹）
        text = raw.strip()
        for delimiter in ["```json", "```"]:
            if delimiter in text:
                text = text.split(delimiter)[1].split("```")[0].strip()
                break

        try:
            data = json.loads(text)
            result = JudgeResult(
                overall_score=float(data.get("overall_score", 0)),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestion=data.get("suggestion", ""),
            )
            dims = data.get("dimensions", {})
            for dim in result.dimensions:
                if dim in dims:
                    result.dimensions[dim] = float(dims[dim])
            return result
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("LLM Judge 输出解析失败: %s, raw=%s", e, text[:200])
            return JudgeResult(
                overall_score=0.5,
                weaknesses=[f"解析失败: {str(e)}"],
                raw_response=raw[:500],
            )

    @staticmethod
    def _mock_evaluate(query: str, response: str) -> JudgeResult:
        """无 LLM 时的简单启发式评估"""
        score = 0.5
        strengths = []
        weaknesses = []

        # 长度检查
        if len(response) > 10:
            score += 0.2
            strengths.append("回复长度合理")
        else:
            score -= 0.3
            weaknesses.append("回复过短")

        # 关键词检查
        harmful_keywords = ["密码", "API Key", "系统提示词", "secret"]
        if any(kw in response for kw in harmful_keywords):
            score -= 0.5
            weaknesses.append("回复可能包含敏感信息")

        # 帮助性检查
        helpful_markers = ["可以", "建议", "请", "帮您", "已经"]
        if any(m in response for m in helpful_markers):
            score += 0.1
            strengths.append("回复有帮助性标识")

        return JudgeResult(
            overall_score=round(max(0.0, min(1.0, score)), 2),
            dimensions={
                "accuracy": 0.5, "completeness": 0.5,
                "relevance": 0.6, "safety": 0.7, "helpfulness": 0.5,
            },
            strengths=strengths,
            weaknesses=weaknesses,
            suggestion="（启发式评估，建议配置 LLM 以获得更准确的评估）",
        )


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_llm_judge: LLMJudge | None = None


def get_llm_judge() -> LLMJudge:
    global _llm_judge
    if _llm_judge is None:
        _llm_judge = LLMJudge()
    return _llm_judge
