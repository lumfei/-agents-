"""
通知模块 — HITL 审批通知的多渠道分发

通知渠道（按优先级）：
  1. 控制台日志 — 始终可用，开发调试首选
  2. 文件队列 — JSON 文件轮询，无需额外基础设施
  3. WebSocket — 实时推送（FastAPI 集成，Phase 3）
  4. 邮件 / 企微 / 钉钉 — 生产环境扩展

通知类型：
  - pending_approval: 新审批请求到达
  - approved: 审批通过
  - rejected: 审批拒绝
  - expired: 审批超时

使用方式：
  from app.human_in_the_loop.notification import notify

  notify.pending_approval(request)
  notify.approved(request)
  notify.rejected(request)
  notify.expired(request)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

from app.human_in_the_loop.approval_schema import ApprovalRequest, ApprovalStatus

logger = logging.getLogger(__name__)

# 文件队列存储路径
QUEUE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "notifications")
os.makedirs(QUEUE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  通知渠道接口
# ═══════════════════════════════════════════════════════════════

class Notifier:
    """
    通知分发器 — 同时向所有已注册渠道发送通知。

    渠道注册：
      notifier = Notifier()
      notifier.add_channel(ConsoleChannel())
      notifier.add_channel(FileQueueChannel())
    """

    def __init__(self):
        self._channels: list[Channel] = [ConsoleChannel(), FileQueueChannel()]

    def add_channel(self, channel: "Channel"):
        self._channels.append(channel)

    def pending_approval(self, request: ApprovalRequest):
        """发送"待审批"通知"""
        for ch in self._channels:
            try:
                ch.send(
                    event="pending_approval",
                    title=f"新审批请求 — {request.type.value}",
                    body=self._format_body(request),
                    metadata=request.to_dict(),
                )
            except Exception as e:
                logger.warning("通知渠道 %s 发送失败: %s", ch.name, e)

    def approved(self, request: ApprovalRequest):
        """发送"审批通过"通知"""
        for ch in self._channels:
            try:
                ch.send(
                    event="approved",
                    title=f"审批通过 — {request.type.value}",
                    body=f"审批人: {request.reviewer}\n意见: {request.comment or '无'}",
                    metadata=request.to_dict(),
                )
            except Exception as e:
                logger.warning("通知渠道 %s 发送失败: %s", ch.name, e)

    def rejected(self, request: ApprovalRequest):
        """发送"审批拒绝"通知"""
        for ch in self._channels:
            try:
                ch.send(
                    event="rejected",
                    title=f"审批拒绝 — {request.type.value}",
                    body=f"审批人: {request.reviewer}\n意见: {request.comment or '无'}",
                    metadata=request.to_dict(),
                )
            except Exception as e:
                logger.warning("通知渠道 %s 发送失败: %s", ch.name, e)

    def expired(self, request: ApprovalRequest):
        """发送"审批超时"通知"""
        for ch in self._channels:
            try:
                ch.send(
                    event="expired",
                    title=f"审批超时 — {request.type.value}",
                    body=f"超时自动动作: {request.comment or '已自动升级'}",
                    metadata=request.to_dict(),
                )
            except Exception as e:
                logger.warning("通知渠道 %s 发送失败: %s", ch.name, e)

    def _format_body(self, request: ApprovalRequest) -> str:
        ctx = request.context
        lines = [
            f"审批类型: {request.type.value}",
            f"发起 Agent: {request.from_agent}",
            f"优先级: {request.priority}",
            f"理由: {request.reason}",
            f"过期时间: {datetime.fromtimestamp(request.expires_at).strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if ctx.get("amount"):
            lines.append(f"金额: ¥{ctx['amount']}")
        if ctx.get("order_id"):
            lines.append(f"订单号: {ctx['order_id']}")
        if ctx.get("reason"):
            lines.append(f"原因: {ctx['reason']}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  通知渠道实现
# ═══════════════════════════════════════════════════════════════

class Channel:
    """通知渠道基类"""
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def send(self, event: str, title: str, body: str, metadata: dict):
        raise NotImplementedError


class ConsoleChannel(Channel):
    """
    控制台日志渠道 — 始终可用，零配置。

    输出到 stdout/stderr，开发时可以直接在终端看到通知。
    生产环境可由日志收集系统（ELK/Loki）采集。
    """

    def send(self, event: str, title: str, body: str, metadata: dict):
        sep = "=" * 60
        logger.info(
            "\n%s\n🔔 [%s] %s\n%s\n%s\n审批ID: %s\n%s",
            sep,
            event.upper(),
            title,
            body,
            sep,
            metadata.get("id", "?"),
            sep,
        )


class FileQueueChannel(Channel):
    """
    文件队列渠道 — JSON 文件轮询，无需额外基础设施。

    每条通知写入独立 JSON 文件到 data/notifications/ 目录。
    外部系统（审批 UI、监控脚本）轮询该目录即可获取通知。

    文件命名: {timestamp}_{event}_{request_id}.json
    方便按时间排序和按事件类型过滤。
    """

    def send(self, event: str, title: str, body: str, metadata: dict):
        filename = f"{int(time.time() * 1000)}_{event}_{metadata.get('id', 'unknown')}.json"
        filepath = os.path.join(QUEUE_DIR, filename)

        payload = {
            "event": event,
            "title": title,
            "body": body,
            "metadata": metadata,
            "timestamp": time.time(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.debug("通知已写入文件队列: %s", filepath)

    @staticmethod
    def poll(since: float = 0, event_filter: Optional[str] = None) -> list[dict]:
        """
        轮询通知文件队列。

        参数:
          since: 只返回此时间戳之后的文件（基于文件名中的时间戳）
          event_filter: 可选，只返回指定事件类型

        返回:
          通知列表，按时间升序排列
        """
        results = []
        try:
            for fname in sorted(os.listdir(QUEUE_DIR)):
                if not fname.endswith(".json"):
                    continue
                filepath = os.path.join(QUEUE_DIR, fname)
                try:
                    file_mtime = os.path.getmtime(filepath)
                    if file_mtime <= since:
                        continue
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if event_filter and data.get("event") != event_filter:
                        continue
                    results.append(data)
                except (json.JSONDecodeError, OSError):
                    continue
        except FileNotFoundError:
            pass
        return results


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

notify = Notifier()
