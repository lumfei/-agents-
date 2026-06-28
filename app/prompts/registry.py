"""
PromptRegistry —— 核心注册表单例

职责：
  1. 从 versions.yaml 读取每类 Prompt 的激活版本
  2. 加载对应 YAML 文件，缓存解析结果
  3. YAML 不可用时回退到 defaults.py 硬编码值
  4. 对外提供统一的 get_*() API

线程安全：模块级单例，懒初始化（首次访问时加载 manifest）。
"""

from __future__ import annotations

import logging
from typing import Optional

from app.prompts.schema import PromptVersion
from app.prompts.loader import load_manifest, load_prompt_file, invalidate_cache

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Prompt 注册表单例。

    使用方式：
      registry = get_prompt_registry()
      supervisor_prompt = registry.get_supervisor_prompt()
      worker_prompt = registry.get_worker_prompt(agent_name="售后 Agent", ...)
    """

    def __init__(self):
        self._cache: dict[str, PromptVersion] = {}  # prompt_type → active version 对象
        self._manifest = None
        self._initialized: bool = False

    # ── 懒初始化 ──────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """首次访问时加载 versions.yaml manifest。"""
        if self._initialized:
            return
        self._manifest = load_manifest()
        self._initialized = True
        if self._manifest:
            logger.info(
                "Prompt manifest 已加载，共 %d 个 prompt 类型，default_version=%s",
                len(self._manifest.prompts),
                self._manifest.default_version,
            )
        else:
            logger.info("未找到 versions.yaml，所有 Prompt 使用硬编码默认值")

    # ── 内部加载逻辑 ──────────────────────────────────────

    def _load_prompt(self, prompt_type: str) -> Optional[PromptVersion]:
        """按类型加载激活版本的 Prompt。命中缓存直接返回。"""
        self._ensure_initialized()

        # 缓存命中
        if prompt_type in self._cache:
            return self._cache[prompt_type]

        # 确定要加载的版本号
        version = self._get_active_version(prompt_type)

        # 加载 YAML 文件
        prompt_file = load_prompt_file(prompt_type)
        if prompt_file is None:
            logger.debug("prompt_type=%s 的 YAML 文件不可用，使用硬编码默认值", prompt_type)
            return None

        # 在文件版本列表中查找目标版本
        for pv in prompt_file.versions:
            if pv.version == version:
                self._cache[prompt_type] = pv
                logger.debug("prompt_type=%s → version=%s（来自 YAML）", prompt_type, version)
                return pv

        # 版本号未找到 → 使用文件中最后一个版本
        if prompt_file.versions:
            fallback_pv = prompt_file.versions[-1]
            self._cache[prompt_type] = fallback_pv
            logger.warning(
                "prompt_type=%s 中未找到 version=%s，回退到 %s",
                prompt_type, version, fallback_pv.version,
            )
            return fallback_pv

        return None

    def _get_active_version(self, prompt_type: str) -> str:
        """从 manifest 读取某类 Prompt 的激活版本号。"""
        if self._manifest is None:
            return "v1"
        for entry in self._manifest.prompts:
            if entry.prompt_type == prompt_type:
                return entry.active_version
        return self._manifest.default_version

    # ═══════════════════════════════════════════════════════
    #  公开 API
    # ═══════════════════════════════════════════════════════

    def get_supervisor_prompt(self) -> str:
        """返回调度员 Supervisor 的完整系统提示词字符串。"""
        from app.prompts.defaults import DEFAULT_SUPERVISOR_PROMPT

        pv = self._load_prompt("supervisor")
        if pv is None or pv.prompt is None:
            return DEFAULT_SUPERVISOR_PROMPT

        from app.agents.base_agent import SystemPromptBuilder
        return SystemPromptBuilder.build(
            role=pv.prompt.role,
            task=pv.prompt.task,
            boundary=pv.prompt.boundary,
            output_format=pv.prompt.output_format,
            extra_context=pv.prompt.extra_context,
        )

    def get_worker_prompt(
        self, agent_name: str, responsibilities: str, tools_desc: str,
    ) -> str:
        """返回 Worker Agent 的系统提示词（已完成模板替换）。

        参数：
          agent_name:      Agent 显示名称（如 "技术支持 Agent"）
          responsibilities: 职责描述
          tools_desc:      工具列表文本（如 "可用工具: query_order, ..."）
        """
        # 先尝试从 YAML 加载模板
        pv = self._load_prompt("worker")
        if pv is None or pv.prompt is None:
            # YAML 不可用 → 直接用 defaults.py 模板 + build() 拼装
            # 注意：不能调 SystemPromptBuilder.worker_base_prompt()（会循环回 registry）
            from app.agents.base_agent import SystemPromptBuilder
            from app.prompts.defaults import DEFAULT_WORKER_TEMPLATE
            t = DEFAULT_WORKER_TEMPLATE
            fmt = {"agent_name": agent_name, "responsibilities": responsibilities, "tools_desc": tools_desc}
            return SystemPromptBuilder.build(
                role=t["role"].format(**fmt),
                task=t["task"].format(**fmt),
                boundary=t["boundary"].format(**fmt),
                output_format=t["output_format"].format(**fmt),
            )

        # YAML 模板 → 做字符串替换
        from app.agents.base_agent import SystemPromptBuilder
        fmt_kwargs = {
            "agent_name": agent_name,
            "responsibilities": responsibilities,
            "tools_desc": tools_desc,
        }
        return SystemPromptBuilder.build(
            role=pv.prompt.role.format(**fmt_kwargs),
            task=pv.prompt.task.format(**fmt_kwargs),
            boundary=pv.prompt.boundary.format(**fmt_kwargs),
            output_format=pv.prompt.output_format.format(**fmt_kwargs),
            extra_context=(
                pv.prompt.extra_context.format(**fmt_kwargs)
                if pv.prompt.extra_context else None
            ),
        )

    def get_worker_role(self, agent_type: str) -> dict:
        """返回某个 Worker 类型的角色信息 {agent_role, responsibilities}。

        agent_type: "tech_support" | "finance" | "after_sale"
        """
        from app.prompts.defaults import DEFAULT_WORKER_ROLES

        pv = self._load_prompt("worker_roles")
        roles = pv.prompts if pv else None
        if roles is None:
            return DEFAULT_WORKER_ROLES.get(agent_type, {})

        return roles.get(agent_type, DEFAULT_WORKER_ROLES.get(agent_type, {}))

    def get_evaluation_prompt(self) -> str:
        """返回 LLM-as-Judge 评估 Prompt 模板（含 {query} 等占位符）。"""
        from app.prompts.defaults import DEFAULT_EVALUATION_PROMPT

        pv = self._load_prompt("evaluation")
        if pv is None:
            return DEFAULT_EVALUATION_PROMPT

        template = pv.prompt_template
        return template if template else DEFAULT_EVALUATION_PROMPT

    def get_compression_prompt(self, name: str = "conversation_compression") -> str:
        """返回压缩 Prompt 模板。

        name: "conversation_compression" | "dialogue_compression"
        """
        from app.prompts.defaults import DEFAULT_COMPRESSION_PROMPTS

        pv = self._load_prompt("compression")
        prompts = pv.prompts if pv else None
        if prompts is None:
            return DEFAULT_COMPRESSION_PROMPTS.get(name, "")

        return prompts.get(name, DEFAULT_COMPRESSION_PROMPTS.get(name, ""))

    def get_active_versions(self) -> dict[str, str]:
        """返回所有 prompt_type → active_version 的映射（供 LangFuse metadata 使用）。

        即使无 YAML 也能返回结果（全部标记为 "default"）。
        """
        self._ensure_initialized()
        if self._manifest is None:
            return {
                "supervisor": "default",
                "worker": "default",
                "worker_roles": "default",
                "evaluation": "default",
                "compression": "default",
            }
        result = {}
        for entry in self._manifest.prompts:
            result[entry.prompt_type] = entry.active_version
        return result

    def reload(self) -> None:
        """清空所有缓存并重新读取 YAML 文件。

        用于开发环境热重载 Prompt，无需重启服务。
        """
        self._cache.clear()
        invalidate_cache()
        self._manifest = None
        self._initialized = False
        logger.info("PromptRegistry 已重置，下次访问将重新加载所有 Prompt")


# ═══════════════════════════════════════════════════════════════
#  模块级单例
# ═══════════════════════════════════════════════════════════════

_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    """获取 PromptRegistry 单例。"""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
