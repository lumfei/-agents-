"""
黄金测试集

用于评估 Agent 质量的最小可行数据集（20 条）。
后续可扩展到 50-200 条生产级规模。

每条用例包含：
  - 期望路由的 Agent（expected_intent）
  - 期望调用的工具（expected_tools）
  - 不应调用的工具（forbidden_tools）——负样本
  - 期望回复关键词（expected_keywords）

使用方式：
  from app.evaluation.golden_dataset import GoldenDataset
  ds = GoldenDataset.load("tests/golden_dataset.json")
  for case in ds.filter(category="tech_support"):
      result = run_workflow(case.input)
      assert result["intent"] == case.expected_intent
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GoldenTestCase:
    """一条黄金测试用例"""
    id: str
    category: str           # tech_support / finance / after_sale / safety / edge
    input: str              # 用户输入
    expected_intent: str    # 期望路由（tech_support / finance / after_sale / unknown / escalate）
    expected_tools: list[str] = field(default_factory=list)    # 应调用的工具
    forbidden_tools: list[str] = field(default_factory=list)   # 不应调用的工具
    expected_keywords: list[str] = field(default_factory=list) # 回复应包含的关键词
    difficulty: str = "easy"  # easy / medium / hard
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "input": self.input,
            "expected_intent": self.expected_intent,
            "expected_tools": self.expected_tools,
            "forbidden_tools": self.forbidden_tools,
            "expected_keywords": self.expected_keywords,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenTestCase":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: d[k] for k in fields if k in d})


class GoldenDataset:
    """黄金测试集"""

    def __init__(self, cases: Optional[list[GoldenTestCase]] = None):
        self.cases: list[GoldenTestCase] = cases or []

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, index: int) -> GoldenTestCase:
        return self.cases[index]

    # ── 加载/导出 ────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> "GoldenDataset":
        """从 JSON 文件加载测试集"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = [GoldenTestCase.from_dict(item) for item in data]
        return cls(cases)

    def save(self, path: str):
        """导出为 JSON 文件"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.cases], f, ensure_ascii=False, indent=2)

    # ── 筛选 ────────────────────────────────────────────────

    def filter(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> "GoldenDataset":
        """按条件筛选用例"""
        result = self.cases
        if category:
            result = [c for c in result if c.category == category]
        if difficulty:
            result = [c for c in result if c.difficulty == difficulty]
        if tags:
            result = [c for c in result if any(t in c.tags for t in tags)]
        return GoldenDataset(result)

    def by_id(self, case_id: str) -> Optional[GoldenTestCase]:
        """按 ID 查找单条用例"""
        for c in self.cases:
            if c.id == case_id:
                return c
        return None

    # ── 管理 ────────────────────────────────────────────────

    def add(self, case: GoldenTestCase):
        """添加用例"""
        self.cases.append(case)

    def remove(self, case_id: str) -> bool:
        """删除用例"""
        for i, c in enumerate(self.cases):
            if c.id == case_id:
                self.cases.pop(i)
                return True
        return False

    # ── 统计 ────────────────────────────────────────────────

    def stats(self) -> dict:
        """测试集统计"""
        cats = {}
        diffs = {}
        for c in self.cases:
            cats[c.category] = cats.get(c.category, 0) + 1
            diffs[c.difficulty] = diffs.get(c.difficulty, 0) + 1
        return {
            "total": len(self.cases),
            "by_category": cats,
            "by_difficulty": diffs,
        }

    def __repr__(self) -> str:
        return f"GoldenDataset(cases={len(self.cases)}, cats={list(self.stats()['by_category'].keys())})"
