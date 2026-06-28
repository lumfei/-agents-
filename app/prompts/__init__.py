"""
Prompt 版本管理模块

提供：
  - PromptRegistry: 核心注册表单例，按 prompt_type + version 加载 Prompt
  - YAML 优先加载，缺失时回退到硬编码默认值（defaults.py）
  - 支持 versions.yaml 控制激活版本，无需改代码即可切换 Prompt

使用示例：
  from app.prompts import get_prompt_registry
  registry = get_prompt_registry()
  prompt = registry.get_supervisor_prompt()
"""

from app.prompts.registry import PromptRegistry, get_prompt_registry

__all__ = ["PromptRegistry", "get_prompt_registry"]
