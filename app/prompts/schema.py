"""
Prompt 数据模型 —— 用 Pydantic 校验 YAML 文件结构

三种 Prompt 格式：
  1. 4 要素型（supervisor / worker）：role, task, boundary, output_format, extra_context
  2. 模板型（evaluation）：单个 prompt_template 字符串，含 {query} 等占位符
  3. 多子 Prompt 型（worker_roles / compression）：prompts 字典，key → 文本
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  4 要素结构
# ═══════════════════════════════════════════════════════════════

class PromptSections(BaseModel):
    """四要素 Prompt 结构：Role / Task / Boundary / Output Format"""
    role: str = ""
    task: str = ""
    boundary: str = ""
    output_format: str = ""
    extra_context: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
#  版本 & 文件
# ═══════════════════════════════════════════════════════════════

class PromptVersion(BaseModel):
    """单个版本化的 Prompt 变体

    根据 prompt 类型，以下字段之一被填充：
      - prompt:             4 要素型（supervisor / worker）
      - prompt_template:    模板型（evaluation）
      - prompts:            多子 Prompt 型（worker_roles / compression）
    """
    version: str
    created: str = ""
    author: str = "system"
    description: str = ""
    prompt: Optional[PromptSections] = None
    prompt_template: Optional[str] = None
    prompts: Optional[dict[str, Any]] = None


class PromptFile(BaseModel):
    """YAML 文件顶层结构"""
    description: str = ""
    versions: list[PromptVersion] = []


# ═══════════════════════════════════════════════════════════════
#  versions.yaml Manifest
# ═══════════════════════════════════════════════════════════════

class ManifestEntry(BaseModel):
    """versions.yaml 中的一条记录"""
    prompt_type: str
    active_version: str
    description: str = ""


class VersionsManifest(BaseModel):
    """versions.yaml 顶层结构"""
    default_version: str = "v1"
    prompts: list[ManifestEntry] = []
