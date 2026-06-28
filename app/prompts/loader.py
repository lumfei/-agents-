"""
YAML 文件加载器 —— 带内存缓存 + 优雅降级

设计：
  - 所有 YAML 读取集中在此模块，方便测试 mock
  - 加载失败返回 None（不抛异常），调用方自行回退到 defaults.py
  - 内存缓存避免重复读取磁盘
  - 支持 PROMPTS_DIR 环境变量覆盖 prompts/ 目录路径（测试用）
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.prompts.schema import PromptFile, VersionsManifest

logger = logging.getLogger(__name__)

# 项目根目录：app/prompts/loader.py → app/prompts/ → app/ → 项目根
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PROMPTS_DIR = _PROJECT_ROOT / "prompts"

# 模块级缓存（进程生命周期内有效）
_cache: dict[str, PromptFile] = {}


def _resolve_prompts_dir() -> Path:
    """解析 prompts/ 目录。支持 PROMPTS_DIR 环境变量覆盖（测试用）。"""
    custom = os.environ.get("PROMPTS_DIR")
    return Path(custom) if custom else _PROMPTS_DIR


def load_yaml(filepath: Path) -> Optional[dict]:
    """加载单个 YAML 文件。缺失/损坏返回 None，不抛异常。"""
    try:
        import yaml
    except ImportError:
        logger.error("pyyaml 未安装。请运行: pip install pyyaml")
        return None

    if not filepath.exists():
        logger.debug("Prompt YAML 文件不存在: %s（将使用硬编码默认值）", filepath)
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning("YAML 解析失败 %s: %s（回退到硬编码默认值）", filepath, e)
        return None


def load_manifest() -> Optional[VersionsManifest]:
    """加载 versions.yaml manifest。"""
    filepath = _resolve_prompts_dir() / "versions.yaml"
    data = load_yaml(filepath)
    if data is None:
        return None
    try:
        return VersionsManifest(**data)
    except Exception as e:
        logger.warning("versions.yaml schema 无效: %s（使用默认值）", e)
        return None


def load_prompt_file(prompt_type: str) -> Optional[PromptFile]:
    """按 prompt_type 加载对应的 YAML 文件。命中内存缓存。

    prompt_type 就是不带 .yaml 的文件名，如 "supervisor"、"worker"、"evaluation"。
    """
    global _cache
    filepath = _resolve_prompts_dir() / f"{prompt_type}.yaml"
    cache_key = str(filepath.resolve())
    if cache_key in _cache:
        return _cache[cache_key]

    data = load_yaml(filepath)
    if data is None:
        return None

    try:
        pf = PromptFile(**data)
        _cache[cache_key] = pf
        return pf
    except Exception as e:
        logger.warning("YAML schema 无效 %s: %s（回退到硬编码默认值）", filepath, e)
        return None


def invalidate_cache() -> None:
    """清空加载缓存（热重载用）。"""
    global _cache
    _cache.clear()
    logger.info("Prompt 缓存已清空，下次访问将重新读取 YAML 文件")
