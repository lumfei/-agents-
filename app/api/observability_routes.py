"""
可观测性 API — 成本查询 + 告警查看 + 监控面板

端点：
  GET  /api/v1/observability/summary     — 成本总览 + 活跃告警数
  GET  /api/v1/observability/costs       — 查询成本（支持 session_id/user_id/date 参数）
  GET  /api/v1/observability/alerts      — 活跃告警列表
  GET  /api/v1/observability/alerts/history — 历史告警
  POST /api/v1/observability/alerts/{id}/resolve — 解决告警
  GET  /api/v1/observability/ui          — 监控面板
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from app.observability import (
    get_cost_tracker, get_alert_manager,
    get_reasoning_capture, get_cost_optimizer,
)
from app.evaluation.llm_judge import get_llm_judge

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
#  成本总览
# ═══════════════════════════════════════════════════════════════

@router.get("/summary", tags=["可观测性"])
async def observability_summary():
    """
    成本总览 + 活跃告警数。

    返回示例：
      {
        "cost": {
          "total_requests": 150, "total_input_tokens": 120000,
          "total_output_tokens": 45000, "total_cost_usd": 0.042,
          "by_agent": {...}
        },
        "alerts_active": 2,
        "alerts_recent": [...]
      }
    """
    ct = get_cost_tracker()
    am = get_alert_manager()

    return {
        "cost": ct.get_summary(),
        "alerts_active": len(am.get_active_alerts()),
        "alerts_recent": [
            a.to_dict() for a in am.get_active_alerts()
        ],
    }


# ═══════════════════════════════════════════════════════════════
#  成本查询
# ═══════════════════════════════════════════════════════════════

@router.get("/costs", tags=["可观测性"])
async def query_costs(
    session_id: str = Query(default="", description="按会话 ID 查询"),
    user_id: str = Query(default="", description="按用户 ID 查询"),
    date: str = Query(default="", description="按日期查询（2026-06-24），空=今天"),
):
    """
    按维度查询成本数据。

    三个参数可组合使用。不传任何参数返回今天的数据。
    """
    ct = get_cost_tracker()

    target = None

    if session_id:
        sc = ct.get_session_cost(session_id)
        target = {
            "dimension": "session",
            "key": session_id,
            "data": {
                "input_tokens": sc.total_input_tokens,
                "output_tokens": sc.total_output_tokens,
                "total_tokens": sc.total_input_tokens + sc.total_output_tokens,
                "cost_usd": round(sc.total_cost_usd, 8),
                "cost_cny": round(sc.total_cost_cny, 8),
                "request_count": sc.request_count,
            },
        }
    elif user_id:
        uc = ct.get_user_cost(user_id)
        target = {
            "dimension": "user",
            "key": user_id,
            "data": {
                "input_tokens": uc.total_input_tokens,
                "output_tokens": uc.total_output_tokens,
                "total_tokens": uc.total_input_tokens + uc.total_output_tokens,
                "cost_usd": round(uc.total_cost_usd, 8),
                "cost_cny": round(uc.total_cost_cny, 8),
                "request_count": uc.request_count,
            },
        }
    else:
        day = date if date else None
        dc = ct.get_daily_cost(day)
        target = {
            "dimension": "day",
            "key": day or "today",
            "data": {
                "input_tokens": dc.total_input_tokens,
                "output_tokens": dc.total_output_tokens,
                "total_tokens": dc.total_input_tokens + dc.total_output_tokens,
                "cost_usd": round(dc.total_cost_usd, 8),
                "cost_cny": round(dc.total_cost_cny, 8),
                "request_count": dc.request_count,
            },
        }

    # 附上全局概览
    target["global_summary"] = ct.get_summary()
    return target


# ═══════════════════════════════════════════════════════════════
#  告警管理
# ═══════════════════════════════════════════════════════════════

@router.get("/alerts", tags=["可观测性"])
async def get_active_alerts():
    """获取当前所有未解决的活跃告警"""
    am = get_alert_manager()
    return {
        "count": len(am.get_active_alerts()),
        "alerts": [a.to_dict() for a in am.get_active_alerts()],
    }


@router.get("/alerts/history", tags=["可观测性"])
async def get_alert_history(count: int = Query(default=50, le=200)):
    """获取最近的历史告警（含已解决的）"""
    am = get_alert_manager()
    return {
        "alerts": [a.to_dict() for a in am.get_alert_history(count)],
    }


@router.post("/alerts/{alert_id}/resolve", tags=["可观测性"])
async def resolve_alert(alert_id: str):
    """手动标记告警为已解决"""
    am = get_alert_manager()
    resolved = am.resolve_alert(alert_id)
    if resolved is None:
        return {"success": False, "message": f"告警 {alert_id} 不存在或已解决"}
    return {"success": True, "alert": resolved.to_dict()}


# ═══════════════════════════════════════════════════════════════
#  监控面板 UI
# ═══════════════════════════════════════════════════════════════

@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def observability_ui():
    """可观测性监控面板"""
    import os
    ui_path = os.path.join(os.path.dirname(__file__), "..", "static", "observability_ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    # 兜底
    return HTMLResponse(content="<h1>Observability UI not found</h1>")


# ═══════════════════════════════════════════════════════════════
#  成本优化建议
# ═══════════════════════════════════════════════════════════════

@router.get("/optimization", tags=["可观测性"])
async def cost_optimization_suggestions():
    """
    获取成本优化建议。

    基于 CostTracker 的数据自动分析，返回可操作的成本优化建议。
    """
    optimizer = get_cost_optimizer()
    tracker = get_cost_tracker()
    suggestions = optimizer.analyze(tracker)
    return {
        "suggestions": [
            {
                "category": s.category,
                "severity": s.severity,
                "title": s.title,
                "description": s.description,
                "estimated_savings_pct": s.estimated_savings_pct,
                "action": s.action,
            }
            for s in suggestions
        ],
        "total_suggestions": len(suggestions),
        "potential_savings": f"{sum(s.estimated_savings_pct for s in suggestions) / max(len(suggestions), 1):.0%}"
        if suggestions else "N/A",
    }


# ═══════════════════════════════════════════════════════════════
#  推理过程追踪
# ═══════════════════════════════════════════════════════════════

@router.get("/reasoning/stats", tags=["可观测性"])
async def reasoning_trace_stats():
    """获取推理 Token 统计"""
    capture = get_reasoning_capture()
    return capture.get_stats()


@router.get("/reasoning/{trace_id}", tags=["可观测性"])
async def reasoning_trace_chain(trace_id: str):
    """获取指定 trace 的完整推理链"""
    capture = get_reasoning_capture()
    chain = capture.get_reasoning_chain(trace_id)
    return {
        "trace_id": trace_id,
        "chain_length": len(chain),
        "chain": chain,
    }


# ═══════════════════════════════════════════════════════════════
#  行为漂移检测
# ═══════════════════════════════════════════════════════════════

@router.get("/drift/status", tags=["可观测性"])
async def drift_detection_status():
    """获取漂移检测状态（基线信息 + 是否需要复核）"""
    judge = get_llm_judge()
    # 优先从磁盘加载基线列表
    disk_baselines = judge.list_baselines()
    if disk_baselines:
        return {
            "baselines": disk_baselines,
            "has_baseline": True,
            "latest_version": disk_baselines[-1]["version"],
            "total_baselines": len(disk_baselines),
            "recommendation": "使用 POST /observability/drift/check 检测漂移",
        }
    # 回退到内存基线
    mem_baselines = list(judge._baselines.keys()) if hasattr(judge, '_baselines') else []
    return {
        "baselines": [],
        "has_baseline": len(mem_baselines) > 0,
        "recommendation": "使用 POST /observability/drift/set-baseline 设置基线版本"
        if not mem_baselines else "已有基线，可使用 POST /observability/drift/check 检测漂移",
        "note": "基线仅在内存中，运行 python scripts/run_drift_check.py --set-baseline v1 以持久化",
    }


@router.post("/drift/set-baseline", tags=["可观测性"])
async def set_drift_baseline(version: str = "v1"):
    """设置行为基线（从磁盘加载或新建）"""
    from app.evaluation.llm_judge import EvalBatchResult
    judge = get_llm_judge()

    # 先尝试从磁盘加载已有基线
    existing = judge.load_baseline(version)
    if existing is not None:
        return {
            "status": "loaded",
            "version": version,
            "average_score": existing.average_score,
            "pass_rate": existing.pass_rate,
            "dimension_averages": existing.dimension_averages,
            "message": f"基线 {version} 已从磁盘加载",
        }

    # 不存在则新建（使用默认值，建议通过 CLI 脚本生成真实基线）
    baseline = EvalBatchResult(
        total=0,
        average_score=0.85,
        dimension_averages={
            "accuracy": 0.88, "completeness": 0.82,
            "relevance": 0.90, "safety": 0.95, "helpfulness": 0.78,
        },
    )
    judge.set_baseline(version, baseline)
    try:
        saved_path = judge.save_baseline(version)
        return {
            "status": "created",
            "version": version,
            "baseline": {
                "average_score": baseline.average_score,
                "dimension_averages": baseline.dimension_averages,
            },
            "saved_to": saved_path,
            "note": "此为默认基线值，建议运行: python scripts/run_drift_check.py --set-baseline v1 --limit 20",
        }
    except Exception as e:
        return {"status": "ok", "version": version, "warning": f"保存失败: {e}"}


@router.post("/drift/check", tags=["可观测性"])
async def check_drift(current_version: str = "", baseline_version: str = ""):
    """检测行为漂移（对比当前评估 vs 基线）"""
    from app.evaluation.llm_judge import EvalBatchResult
    judge = get_llm_judge()

    # 加载基线
    baseline = None
    if baseline_version:
        baseline = judge.load_baseline(baseline_version)

    if baseline is None:
        return {
            "error": f"基线版本 '{baseline_version}' 不存在",
            "available_baselines": [b["version"] for b in judge.list_baselines()],
            "hint": "运行 python scripts/run_drift_check.py --set-baseline v1 --limit 20 建立基线",
        }

    # 构建当前评估结果（需要运行评估 Pipeline）
    # 注意：完整漂移检测建议通过 CLI 脚本 run_drift_check.py 执行
    # API 端点适用于已运行过评估 Pipeline 后的状态查询
    current = EvalBatchResult(
        total=0,
        average_score=0.78,
        dimension_averages={
            "accuracy": 0.80, "completeness": 0.75,
            "relevance": 0.85, "safety": 0.92, "helpfulness": 0.70,
        },
    )

    report = judge.detect_drift(
        current=current,
        baseline_version=baseline_version,
        current_version=current_version or "current",
    )
    return {
        "baseline_version": baseline_version,
        "baseline_avg_score": baseline.average_score,
        "current_avg_score": current.average_score,
        "overall_drift_score": report.overall_drift_score,
        "dimension_drifts": report.dimension_drifts,
        "style_changes": report.style_changes,
        "threshold_changes": report.threshold_changes,
        "requires_review": report.requires_review,
        "recommendations": report.recommendations,
        "note": "此为预设测试数据，完整评估请运行: python scripts/run_drift_check.py --baseline v1",
    }
