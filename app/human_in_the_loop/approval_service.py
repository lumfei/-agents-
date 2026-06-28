"""
审批服务

这个文件实现了 HITL（Human-in-the-Loop）审批的核心逻辑。

工作流程（interrupt_before 模式）：
  1. Agent 检测到需要审批的操作 → 创建 ApprovalRequest + 设置 approval_pending
  2. 图路由到 human_approval_node → LangGraph interrupt_before 拦截暂停
  3. ApprovalService 创建审批请求 → 触发通知（控制台 + 文件队列）
  4. 审批人通过 REST API / Web UI 查看待审批列表
  5. 审批人调用 approve() / reject()
  6. ApprovalService 更新状态 → 触发通知（审批结果）
  7. 调用方通过 resume_from_checkpoint() 恢复 LangGraph 执行

超时处理：
  - 每个审批请求有 expires_at 时间戳
  - check_timeouts() 方法可以定期运行，自动处理超时请求
  - 超时后的自动动作由 ApprovalPolicy.auto_action 定义
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Any, Callable, Optional

from app.human_in_the_loop.approval_schema import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalType,
    ApprovalPolicy,
    DEFAULT_POLICIES,
)
from app.human_in_the_loop.notification import notify

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  审批服务
# ═══════════════════════════════════════════════════════════════

class ApprovalService:
    """
    审批服务——管理所有审批请求的生命周期。

    这个类是单例的，在整个应用中只有一个实例。
    通过 from app.human_in_the_loop import approval_service 访问。

    功能：
      - 创建审批请求
      - 查询待审批列表
      - 批准/拒绝请求
      - 超时自动处理
      - 审批策略查询
    """

    def __init__(self):
        # 内存存储：{request_id: ApprovalRequest}
        self._requests: dict[str, ApprovalRequest] = {}

        # 审批策略
        self._policies: dict[ApprovalType, ApprovalPolicy] = dict(DEFAULT_POLICIES)

        # 回调函数：当审批状态变更时通知外部
        # 比如更新 LangGraph 状态、发送通知等
        self._on_status_change: Optional[Callable[[str, ApprovalStatus], None]] = None

        # 超时检测定时器
        self._timeout_timer: Optional[threading.Timer] = None
        self._running = False

    # ── 审批请求管理 ────────────────────────────────────────

    def create_request(
        self,
        approval_type: ApprovalType,
        from_agent: str,
        thread_id: str = "",
        context: dict | None = None,
        reason: str = "",
        priority: str = "normal",
    ) -> ApprovalRequest:
        """
        创建一个新的审批请求。

        参数：
          approval_type: 审批类型
          from_agent:    发起审批的 Agent 名称
          thread_id:     关联的 LangGraph 线程 ID
          context:       上下文信息（金额、订单号等）
          reason:        审批理由
          priority:      优先级

        返回：
          创建的 ApprovalRequest 实例
        """
        # 获取对应的审批策略
        policy = self._policies.get(approval_type)
        timeout = policy.timeout_seconds if policy else 3600

        request = ApprovalRequest(
            type=approval_type,
            from_agent=from_agent,
            thread_id=thread_id,
            context=context or {},
            reason=reason or f"{approval_type.value} 需要审批",
            priority=priority,
            expires_at=time.time() + timeout,
        )

        self._requests[request.id] = request

        # 发送"待审批"通知
        notify.pending_approval(request)

        return request

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """查询指定的审批请求"""
        return self._requests.get(request_id)

    def list_pending(self) -> list[ApprovalRequest]:
        """获取所有待审批的请求（按创建时间倒序）"""
        pending = [
            r for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING and not r.is_expired
        ]
        pending.sort(key=lambda r: r.created_at, reverse=True)
        return pending

    def list_all(self) -> list[ApprovalRequest]:
        """获取所有审批请求"""
        return list(self._requests.values())

    # ── 审批操作 ────────────────────────────────────────────

    def approve(
        self,
        request_id: str,
        reviewer: str = "system",
        comment: str = "",
    ) -> ApprovalRequest | None:
        """
        批准一个审批请求。

        参数：
          request_id: 审批请求 ID
          reviewer:   审批人用户名
          comment:    审批意见

        返回：
          更新后的 ApprovalRequest，如果不存在返回 None
        """
        request = self._requests.get(request_id)
        if request is None:
            return None

        if request.status != ApprovalStatus.PENDING:
            return request

        request.status = ApprovalStatus.APPROVED
        request.reviewer = reviewer
        request.comment = comment
        request.reviewed_at = time.time()

        # 触发回调
        if self._on_status_change:
            self._on_status_change(request_id, ApprovalStatus.APPROVED)

        # 发送通过通知
        notify.approved(request)

        return request

    def reject(
        self,
        request_id: str,
        reviewer: str = "system",
        comment: str = "",
    ) -> ApprovalRequest | None:
        """
        拒绝一个审批请求。

        参数同上。
        """
        request = self._requests.get(request_id)
        if request is None:
            return None

        if request.status != ApprovalStatus.PENDING:
            return request

        request.status = ApprovalStatus.REJECTED
        request.reviewer = reviewer
        request.comment = comment
        request.reviewed_at = time.time()

        if self._on_status_change:
            self._on_status_change(request_id, ApprovalStatus.REJECTED)

        # 发送拒绝通知
        notify.rejected(request)

        return request

    def mark_expired(self, request_id: str) -> ApprovalRequest | None:
        """
        标记审批请求为过期（超时未处理）。
        """
        request = self._requests.get(request_id)
        if request is None:
            return None

        if request.status != ApprovalStatus.PENDING:
            return request

        request.status = ApprovalStatus.EXPIRED
        request.reviewed_at = time.time()
        request.comment = "审批超时，自动过期"

        if self._on_status_change:
            self._on_status_change(request_id, ApprovalStatus.EXPIRED)

        # 发送超时通知
        notify.expired(request)

        return request

    # ── 批量操作 ────────────────────────────────────────────

    def batch_approve(self, request_ids: list[str], reviewer: str = "system") -> list[ApprovalRequest]:
        """批量批准"""
        results = []
        for rid in request_ids:
            result = self.approve(rid, reviewer)
            if result:
                results.append(result)
        return results

    def batch_reject(self, request_ids: list[str], reviewer: str = "system") -> list[ApprovalRequest]:
        """批量拒绝"""
        results = []
        for rid in request_ids:
            result = self.reject(rid, reviewer)
            if result:
                results.append(result)
        return results

    # ── 超时处理 ────────────────────────────────────────────

    def check_timeouts(self) -> list[ApprovalRequest]:
        """
        检查所有待审批请求是否超时。

        这是一个可以定期调用的"清理方法"。
        对每个超时的请求，根据审批策略的 auto_action 执行自动操作。

        返回：
          本次处理的超时请求列表
        """
        now = time.time()
        timed_out = []

        for request in self._requests.values():
            if request.status != ApprovalStatus.PENDING:
                continue
            if now <= request.expires_at:
                continue

            # 超时了，根据策略处理
            policy = self._policies.get(request.type)
            auto_action = policy.auto_action if policy else "escalate"

            if auto_action == "approve":
                self.approve(request.id, reviewer="auto", comment="审批超时，自动通过")
            elif auto_action == "reject":
                self.reject(request.id, reviewer="auto", comment="审批超时，自动拒绝")
            else:
                self.mark_expired(request.id)

            timed_out.append(request)

        return timed_out

    def start_timeout_checker(self, interval: int = 60):
        """
        启动定时检查超时的后台线程。

        参数：
          interval: 检查间隔（秒），默认 60 秒
        """
        if self._running:
            return

        self._running = True

        def _check():
            while self._running:
                self.check_timeouts()
                time.sleep(interval)

        thread = threading.Thread(target=_check, daemon=True)
        thread.start()

    def stop_timeout_checker(self):
        """停止超时检查线程"""
        self._running = False

    # ── 审批策略 ────────────────────────────────────────────

    def get_policy(self, approval_type: ApprovalType) -> ApprovalPolicy | None:
        """获取某个审批类型的策略"""
        return self._policies.get(approval_type)

    def set_policy(self, policy: ApprovalPolicy):
        """设置或更新某个审批类型的策略"""
        self._policies[policy.type] = policy

    def set_policies(self, policies: list[ApprovalPolicy]):
        """批量设置审批策略"""
        for p in policies:
            self._policies[p.type] = p

    def needs_approval(self, approval_type: ApprovalType, amount: float = 0.0) -> bool:
        """
        判断某个操作是否需要审批。

        参数：
          approval_type: 审批类型
          amount:        金额（仅退款类型需要检查）

        返回：
          True=需要审批，False=不需要
        """
        policy = self._policies.get(approval_type)
        if policy is None or not policy.enabled:
            return False
        if approval_type == ApprovalType.REFUND:
            return amount > policy.threshold
        return True

    # ── 回调 ────────────────────────────────────────────────

    def set_on_status_change(self, callback: Callable[[str, ApprovalStatus], None]):
        """
        设置状态变更回调。

        当审批请求状态变更时（approve/reject/timeout），
        会调用这个回调函数。

        参数：
          callback: 接收 (request_id, new_status) 的函数
        """
        self._on_status_change = callback

    # ── 统计 ────────────────────────────────────────────────

    def stats(self) -> dict:
        """审批统计"""
        total = len(self._requests)
        pending = len(self.list_pending())
        approved = sum(1 for r in self._requests.values() if r.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for r in self._requests.values() if r.status == ApprovalStatus.REJECTED)
        expired = sum(1 for r in self._requests.values() if r.status == ApprovalStatus.EXPIRED)

        return {
            "total": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "expired": expired,
        }

    def clear(self):
        """清除所有审批请求（调试/测试用）"""
        self._requests.clear()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ApprovalService(total={s['total']}, "
            f"pending={s['pending']}, "
            f"approved={s['approved']})"
        )


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

# 应用中只有一个 ApprovalService 实例
# 通过 from app.human_in_the_loop.approval_service import approval_service 使用
approval_service = ApprovalService()
