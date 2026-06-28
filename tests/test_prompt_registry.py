"""
Prompt 版本管理 —— PromptRegistry 单元测试

覆盖：
  - YAML 缺失时回退到硬编码默认值
  - YAML 加载正确版本
  - 版本切换
  - Worker 模板替换
  - 回退行为（版本号不存在、YAML 格式错误）
  - get_active_versions()
  - reload() 清缓存
"""

from __future__ import annotations

import os
import pytest
from pathlib import Path


# 测试用的最小化 YAML 内容
MINIMAL_VERSIONS_YAML = """default_version: "v1"
prompts:
  - prompt_type: "supervisor"
    active_version: "v2"
    description: "test"
  - prompt_type: "worker"
    active_version: "v1"
    description: "test"
  - prompt_type: "worker_roles"
    active_version: "v1"
    description: "test"
  - prompt_type: "evaluation"
    active_version: "v1"
    description: "test"
  - prompt_type: "compression"
    active_version: "v1"
    description: "test"
"""

MINIMAL_SUPERVISOR_V1 = """
description: "test"
versions:
  - version: "v1"
    created: "2026-01-01"
    author: "test"
    description: "v1 prompt"
    prompt:
      role: "v1-role"
      task: "v1-task"
      boundary: "v1-boundary"
      output_format: "v1-format"
  - version: "v2"
    created: "2026-01-02"
    author: "test"
    description: "v2 prompt"
    prompt:
      role: "v2-role"
      task: "v2-task"
      boundary: "v2-boundary"
      output_format: "v2-format"
"""

MINIMAL_WORKER_V1 = """
description: "test"
versions:
  - version: "v1"
    created: "2026-01-01"
    author: "test"
    description: "v1 worker template"
    prompt:
      role: "你是「{agent_name}」，擅长：{responsibilities}"
      task: "task-{agent_name}"
      boundary: "tools: {tools_desc}"
      output_format: "format-{responsibilities}"
"""

MINIMAL_WORKER_ROLES_V1 = """
description: "test"
versions:
  - version: "v1"
    created: "2026-01-01"
    author: "test"
    prompts:
      tech_support:
        agent_role: "test-tech-role"
        responsibilities: "test-tech-resp"
      finance:
        agent_role: "test-finance-role"
        responsibilities: "test-finance-resp"
      after_sale:
        agent_role: "test-aftersale-role"
        responsibilities: "test-aftersale-resp"
"""

MINIMAL_EVAL_V1 = """
description: "test"
versions:
  - version: "v1"
    created: "2026-01-01"
    author: "test"
    prompt_template: "eval-{query}-{response}-{expected_tools}-{actual_tools}-{intent}"
"""

MINIMAL_COMPRESSION_V1 = """
description: "test"
versions:
  - version: "v1"
    created: "2026-01-01"
    author: "test"
    prompts:
      conversation_compression: "compress-conv-{conversation_log}"
      dialogue_compression: "compress-dialogue-{conversation_log}"
"""


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def temp_prompts_dir(tmp_path):
    """创建临时 prompts/ 目录并设置 PROMPTS_DIR 环境变量"""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    old_dir = os.environ.get("PROMPTS_DIR")
    os.environ["PROMPTS_DIR"] = str(prompts_dir)
    yield prompts_dir
    if old_dir is not None:
        os.environ["PROMPTS_DIR"] = old_dir
    else:
        os.environ.pop("PROMPTS_DIR", None)


def _write_yaml(dir_path: Path, filename: str, content: str):
    """在指定目录写入 YAML 文件"""
    filepath = dir_path / filename
    filepath.write_text(content, encoding="utf-8")
    return filepath


def _fresh_registry():
    """获取一个全新的 PromptRegistry 实例（绕过全局单例）"""
    from app.prompts.registry import PromptRegistry
    reg = PromptRegistry()
    # 强制非初始化状态
    return reg


# ═══════════════════════════════════════════════════════════════
#  Test: YAML 缺失时回退
# ═══════════════════════════════════════════════════════════════

class TestFallbackWhenYamlMissing:
    """YAML 文件不存在时，所有 API 返回硬编码默认值"""

    def test_supervisor_fallback(self, temp_prompts_dir):
        """无 YAML → 返回 DEFAULT_SUPERVISOR_PROMPT"""
        reg = _fresh_registry()
        # 清空 temp dir（只有空目录）
        prompt = reg.get_supervisor_prompt()
        assert "调度员" in prompt
        assert "路由规则" in prompt
        assert "sentiment" in prompt

    def test_worker_fallback(self, temp_prompts_dir):
        """无 YAML → 返回硬编码 worker 模板"""
        reg = _fresh_registry()
        prompt = reg.get_worker_prompt(
            agent_name="测试Agent", responsibilities="测试职责",
            tools_desc="tool1, tool2",
        )
        assert "测试Agent" in prompt
        assert "绝对禁止编造数据" in prompt
        assert "tool1, tool2" in prompt

    def test_evaluation_fallback(self, temp_prompts_dir):
        """无 YAML → 返回 DEFAULT_EVALUATION_PROMPT"""
        reg = _fresh_registry()
        prompt = reg.get_evaluation_prompt()
        assert "质量评估裁判" in prompt
        assert "{query}" in prompt
        assert "accuracy" in prompt

    def test_compression_fallback(self, temp_prompts_dir):
        """无 YAML → 返回压缩 prompt 模板"""
        reg = _fresh_registry()
        prompt = reg.get_compression_prompt("conversation_compression")
        assert "压缩要求" in prompt
        assert "{conversation_log}" in prompt

    def test_worker_roles_fallback(self, temp_prompts_dir):
        """无 YAML → 返回 DEFAULT_WORKER_ROLES"""
        reg = _fresh_registry()
        role = reg.get_worker_role("tech_support")
        assert "技术支持 Agent" in role.get("agent_role", "")
        assert "系统故障" in role.get("responsibilities", "")


# ═══════════════════════════════════════════════════════════════
#  Test: YAML 加载
# ═══════════════════════════════════════════════════════════════

class TestLoadsFromYaml:
    """YAML 文件存在时，返回 YAML 内容"""

    def test_supervisor_loads_v2(self, temp_prompts_dir):
        """manifest 指向 v2 → 返回 v2 内容"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "supervisor.yaml", MINIMAL_SUPERVISOR_V1)

        reg = _fresh_registry()
        prompt = reg.get_supervisor_prompt()
        assert "v2-role" in prompt
        assert "v1-role" not in prompt

    def test_worker_template_substitution(self, temp_prompts_dir):
        """YAML worker prompt → 占位符被正确替换"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "worker.yaml", MINIMAL_WORKER_V1)

        reg = _fresh_registry()
        prompt = reg.get_worker_prompt(
            agent_name="客服小王", responsibilities="处理退款",
            tools_desc="refund_tool, order_tool",
        )
        assert "客服小王" in prompt
        assert "处理退款" in prompt
        assert "refund_tool, order_tool" in prompt
        assert "你是「客服小王」" in prompt

    def test_worker_roles_from_yaml(self, temp_prompts_dir):
        """YAML worker_roles → 返回 YAML 定义的角色"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "worker_roles.yaml", MINIMAL_WORKER_ROLES_V1)

        reg = _fresh_registry()
        role = reg.get_worker_role("finance")
        assert role["agent_role"] == "test-finance-role"
        assert role["responsibilities"] == "test-finance-resp"

    def test_evaluation_from_yaml(self, temp_prompts_dir):
        """YAML evaluation → 返回自定义模板"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "evaluation.yaml", MINIMAL_EVAL_V1)

        reg = _fresh_registry()
        template = reg.get_evaluation_prompt()
        assert template.startswith("eval-")
        assert "{query}" in template

    def test_compression_from_yaml(self, temp_prompts_dir):
        """YAML compression → 返回自定义模板"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "compression.yaml", MINIMAL_COMPRESSION_V1)

        reg = _fresh_registry()
        template = reg.get_compression_prompt("conversation_compression")
        assert template.startswith("compress-conv-")
        assert "{conversation_log}" in template


# ═══════════════════════════════════════════════════════════════
#  Test: 版本切换
# ═══════════════════════════════════════════════════════════════

class TestVersionSwitching:
    """切换 active_version 后返回不同内容"""

    def test_switch_from_v2_to_v1(self, temp_prompts_dir):
        """manifest v2 → v1，返回内容对应变化"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "supervisor.yaml", MINIMAL_SUPERVISOR_V1)

        # 默认 manifest 指向 v2
        reg = _fresh_registry()
        prompt_v2 = reg.get_supervisor_prompt()
        assert "v2-role" in prompt_v2

        # 切换 manifest 到 v1
        new_manifest = MINIMAL_VERSIONS_YAML.replace('"v2"', '"v1"')
        _write_yaml(temp_prompts_dir, "versions.yaml", new_manifest)

        reg2 = _fresh_registry()
        prompt_v1 = reg2.get_supervisor_prompt()
        assert "v1-role" in prompt_v1

    def test_reload_picks_up_changes(self, temp_prompts_dir):
        """reload() 后重新读取 YAML"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "supervisor.yaml", MINIMAL_SUPERVISOR_V1)

        reg = _fresh_registry()
        prompt1 = reg.get_supervisor_prompt()
        assert "v2-role" in prompt1

        # 直接修改 YAML 文件内容（新增 v3）
        new_supervisor = MINIMAL_SUPERVISOR_V1.replace('"v2-role"', '"v3-role"')
        _write_yaml(temp_prompts_dir, "supervisor.yaml", new_supervisor)

        # reload → 应该拿到新内容
        reg.reload()
        prompt2 = reg.get_supervisor_prompt()
        assert "v3-role" in prompt2


# ═══════════════════════════════════════════════════════════════
#  Test: 回退行为
# ═══════════════════════════════════════════════════════════════

class TestGracefulFallback:
    """YAML 有问题时优雅降级"""

    def test_missing_version_falls_back_to_last(self, temp_prompts_dir):
        """manifest 指向不存在的版本 → 返回文件中最后一个版本"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        # supervisor.yaml 只有 v1, v2 → manifest 指向 v99
        manifest_bad = MINIMAL_VERSIONS_YAML.replace('"v2"', '"v99"')
        _write_yaml(temp_prompts_dir, "versions.yaml", manifest_bad)
        _write_yaml(temp_prompts_dir, "supervisor.yaml", MINIMAL_SUPERVISOR_V1)

        reg = _fresh_registry()
        prompt = reg.get_supervisor_prompt()
        # 回退到最后一个版本 v2
        assert "v2-role" in prompt

    def test_invalid_yaml_falls_back_to_default(self, temp_prompts_dir):
        """损坏的 YAML → 返回硬编码默认值"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "supervisor.yaml", "not: [valid: yaml: {{{")

        reg = _fresh_registry()
        prompt = reg.get_supervisor_prompt()
        # 回退到硬编码默认值
        assert "调度员" in prompt

    def test_empty_yaml_falls_back(self, temp_prompts_dir):
        """空 YAML 文件 → 返回默认值"""
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)
        _write_yaml(temp_prompts_dir, "supervisor.yaml", "")

        reg = _fresh_registry()
        prompt = reg.get_supervisor_prompt()
        assert "调度员" in prompt


# ═══════════════════════════════════════════════════════════════
#  Test: get_active_versions
# ═══════════════════════════════════════════════════════════════

class TestActiveVersions:
    """get_active_versions() 返回正确的版本映射"""

    def test_returns_yaml_versions(self, temp_prompts_dir):
        _write_yaml(temp_prompts_dir, "versions.yaml", MINIMAL_VERSIONS_YAML)

        reg = _fresh_registry()
        versions = reg.get_active_versions()
        assert versions["supervisor"] == "v2"
        assert versions["worker"] == "v1"

    def test_returns_default_when_no_yaml(self, temp_prompts_dir):
        """无 YAML → 所有版本标记为 'default'"""
        reg = _fresh_registry()
        versions = reg.get_active_versions()
        assert versions["supervisor"] == "default"
        assert versions["worker"] == "default"
        assert versions["evaluation"] == "default"


# ═══════════════════════════════════════════════════════════════
#  Test: 回归 —— default 值与当前硬编码一致
# ═══════════════════════════════════════════════════════════════

class TestDefaultsMatchOriginal:
    """确保 defaults.py 与原始硬编码 Prompt 完全一致"""

    def test_supervisor_default_matches_original_formula(self):
        """defaults.py 的 supervisor prompt 包含所有关键短语"""
        from app.prompts.defaults import DEFAULT_SUPERVISOR_PROMPT

        assert "调度员" in DEFAULT_SUPERVISOR_PROMPT
        assert "意图分类" in DEFAULT_SUPERVISOR_PROMPT
        assert "tech_support" in DEFAULT_SUPERVISOR_PROMPT
        assert "finance" in DEFAULT_SUPERVISOR_PROMPT
        assert "after_sale" in DEFAULT_SUPERVISOR_PROMPT
        assert "情绪识别规则" in DEFAULT_SUPERVISOR_PROMPT
        assert "angry" in DEFAULT_SUPERVISOR_PROMPT
        assert "frustrated" in DEFAULT_SUPERVISOR_PROMPT

    def test_worker_default_contains_key_rules(self):
        """defaults.py 的 worker 模板包含关键规则"""
        from app.prompts.defaults import DEFAULT_WORKER_TEMPLATE

        assert "绝对禁止编造数据" in DEFAULT_WORKER_TEMPLATE["boundary"]
        assert "{tools_desc}" in DEFAULT_WORKER_TEMPLATE["boundary"]
        assert "{agent_name}" in DEFAULT_WORKER_TEMPLATE["role"]

    def test_all_worker_types_have_roles(self):
        """3 个 worker 类型都有默认角色定义"""
        from app.prompts.defaults import DEFAULT_WORKER_ROLES

        assert "tech_support" in DEFAULT_WORKER_ROLES
        assert "finance" in DEFAULT_WORKER_ROLES
        assert "after_sale" in DEFAULT_WORKER_ROLES
        for key in DEFAULT_WORKER_ROLES:
            assert "agent_role" in DEFAULT_WORKER_ROLES[key]
            assert "responsibilities" in DEFAULT_WORKER_ROLES[key]

    def test_evaluation_default_has_placeholders(self):
        """评估 prompt 默认值包含所有运行时占位符"""
        from app.prompts.defaults import DEFAULT_EVALUATION_PROMPT

        assert "{query}" in DEFAULT_EVALUATION_PROMPT
        assert "{response}" in DEFAULT_EVALUATION_PROMPT
        assert "{expected_tools}" in DEFAULT_EVALUATION_PROMPT
        assert "{actual_tools}" in DEFAULT_EVALUATION_PROMPT
        assert "{intent}" in DEFAULT_EVALUATION_PROMPT
