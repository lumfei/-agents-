#!/usr/bin/env python
"""
从 app/prompts/defaults.py 导出 YAML Prompt 文件

用途：
  当 defaults.py 更新后，运行此脚本自动同步 prompts/*.yaml。
  确保 YAML 文件内容与硬编码回退值一致。

用法：
  python scripts/export_prompts.py          # 导出到 prompts/（不覆盖已有文件）
  python scripts/export_prompts.py --force  # 强制覆盖已有文件
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yaml
except ImportError:
    print("错误: 需要 pyyaml。运行: pip install pyyaml")
    sys.exit(1)


def export_all(prompts_dir: Path, force: bool = False) -> None:
    """导出所有 Prompt 到 YAML 文件。"""

    from app.prompts.defaults import (
        DEFAULT_SUPERVISOR_PROMPT,
        DEFAULT_WORKER_TEMPLATE,
        DEFAULT_WORKER_ROLES,
        DEFAULT_EVALUATION_PROMPT,
        DEFAULT_COMPRESSION_PROMPTS,
    )

    prompts_dir.mkdir(parents=True, exist_ok=True)

    # ── versions.yaml ──
    versions_path = prompts_dir / "versions.yaml"
    if force or not versions_path.exists():
        manifest = {
            "default_version": "v1",
            "prompts": [
                {"prompt_type": "supervisor", "active_version": "v1",
                 "description": "调度员意图分类与路由 Prompt"},
                {"prompt_type": "worker", "active_version": "v1",
                 "description": "Worker Agent 基础 Prompt 模板"},
                {"prompt_type": "worker_roles", "active_version": "v1",
                 "description": "各 Worker 类型角色描述"},
                {"prompt_type": "evaluation", "active_version": "v1",
                 "description": "LLM-as-Judge 质量评估 Prompt"},
                {"prompt_type": "compression", "active_version": "v1",
                 "description": "对话压缩 Prompt"},
            ],
        }
        with open(versions_path, "w", encoding="utf-8") as f:
            f.write("# Prompt Version Manifest\n")
            f.write("# 修改 active_version 即可切换 Prompt 版本\n\n")
            yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✓ {versions_path}")

    # ── supervisor.yaml ──
    # 注意: DEFAULT_SUPERVISOR_PROMPT 已经是拼好的完整字符串，无法反解回 4 要素。
    # 所以 supervisor.yaml 需要手动维护（当前 prompts/supervisor.yaml 已包含完整 4 要素）。
    sp_path = prompts_dir / "supervisor.yaml"
    if not sp_path.exists():
        print(f"⚠ {sp_path} —— supervisor prompt 需要手动编写 YAML（4 要素结构），"
              f"或从已有文件复制")

    # ── worker.yaml ──
    worker_path = prompts_dir / "worker.yaml"
    if force or not worker_path.exists():
        worker_data = worker_path.exists() and _read_yaml(worker_path) or {}
        worker_data.setdefault("description",
                               "Worker Agent 基础 Prompt 模板 —— 含 {agent_name}、{responsibilities}、{tools_desc} 占位符")
        worker_data["versions"] = [{
            "version": "v1",
            "created": "2026-06-29",
            "author": "system",
            "description": "从 defaults.py 自动导出",
            "prompt": DEFAULT_WORKER_TEMPLATE,
        }]
        with open(worker_path, "w", encoding="utf-8") as f:
            _write_yaml_header(f, worker_data["description"])
            yaml.dump(worker_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✓ {worker_path}")

    # ── worker_roles.yaml ──
    roles_path = prompts_dir / "worker_roles.yaml"
    if force or not roles_path.exists():
        roles_data = {
            "description": "各 Worker 类型的角色描述",
            "versions": [{
                "version": "v1",
                "created": "2026-06-29",
                "author": "system",
                "description": "从 defaults.py 自动导出",
                "prompts": DEFAULT_WORKER_ROLES,
            }],
        }
        with open(roles_path, "w", encoding="utf-8") as f:
            _write_yaml_header(f, roles_data["description"])
            yaml.dump(roles_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✓ {roles_path}")

    # ── evaluation.yaml ──
    eval_path = prompts_dir / "evaluation.yaml"
    if force or not eval_path.exists():
        eval_data = {
            "description": "LLM-as-Judge 质量评估 Prompt",
            "versions": [{
                "version": "v1",
                "created": "2026-06-29",
                "author": "system",
                "description": "从 defaults.py 自动导出",
                "prompt_template": DEFAULT_EVALUATION_PROMPT,
            }],
        }
        with open(eval_path, "w", encoding="utf-8") as f:
            _write_yaml_header(f, eval_data["description"])
            yaml.dump(eval_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✓ {eval_path}")

    # ── compression.yaml ──
    comp_path = prompts_dir / "compression.yaml"
    if force or not comp_path.exists():
        comp_data = {
            "description": "对话压缩 Prompt",
            "versions": [{
                "version": "v1",
                "created": "2026-06-29",
                "author": "system",
                "description": "从 defaults.py 自动导出",
                "prompts": DEFAULT_COMPRESSION_PROMPTS,
            }],
        }
        with open(comp_path, "w", encoding="utf-8") as f:
            _write_yaml_header(f, comp_data["description"])
            yaml.dump(comp_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✓ {comp_path}")

    print("\n完成！所有 Prompt YAML 文件已导出到:", prompts_dir)


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _write_yaml_header(f, description: str) -> None:
    f.write(f"# {description}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 defaults.py 导出 YAML Prompt 文件")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有文件")
    parser.add_argument("--dir", type=str, default=None, help="Prompt YAML 输出目录")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    prompts_dir = Path(args.dir) if args.dir else project_root / "prompts"

    export_all(prompts_dir, force=args.force)


if __name__ == "__main__":
    main()
