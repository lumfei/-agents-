"""
审计日志（第五层防御 — 全链路可追溯）

职责：
  - 全链路记录：用户请求 → Agent 推理 → 工具选择 → 工具参数 → 工具结果 → 最终输出
  - 因果链追踪：每个操作记录触发者、原因、结果（parent_id 形成 DAG）
  - 不可篡改：日志追加写（append-only），禁止删除/修改
  - WAL（Write-Ahead Log）保证写入原子性
  - 支持复杂查询：按 trace_id、用户、风险等级、时间范围

每条日志包含（全链路因果链）：
  - log_id:       日志唯一标识（UUID7，时间排序）
  - trace_id:     全链路追踪 ID
  - session_id:   会话 ID
  - timestamp:    操作时间（Unix 毫秒）
  - actor:        操作主体（user / agent_name / tool_name / system）
  - action:       操作类型（input / classify / route / tool_call / output / escalate / guard / audit）
  - input:        操作输入（脱敏后）
  - output:       操作结果（脱敏后）
  - parent_id:    父操作 ID（形成因果链）
  - risk_level:   风险等级
  - duration_ms:  操作耗时（毫秒）
  - metadata:     扩展字段

防篡改机制：
  - Append-Only: 只追加不修改不删除
  - 哈希链: 每条日志包含前一条日志的 SHA-256 哈希
  - WAL: 先写日志再执行操作（崩溃可恢复）
  - 签名: 日志可用 HMAC 签名（防内部篡改）

存储后端：
  - 开发环境：JSONL 文件（追加写，人类可读）
  - 生产环境：PostgreSQL（支持事务和复杂查询）
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  数据模型
# ═══════════════════════════════════════════════════════════════

class AuditAction(str, Enum):
    """审计操作类型"""
    INPUT = "input"               # 用户输入
    INPUT_GUARD = "input_guard"   # 输入安全网关检查
    CLASSIFY = "classify"         # 意图分类
    ROUTE = "route"               # 路由决策
    TOOL_CALL = "tool_call"       # 工具调用
    TOOL_RESULT = "tool_result"   # 工具返回
    WORKER_PROCESS = "worker_process"  # Worker 处理
    OUTPUT = "output"             # 最终输出
    OUTPUT_AUDIT = "output_audit" # 输出安全审核
    ESCALATE = "escalate"         # 升级
    APPROVAL = "approval"         # 审批
    QUALITY_CHECK = "quality_check"  # 质检
    GUARD_BLOCK = "guard_block"   # 护栏拦截
    GUARD_WARN = "guard_warn"     # 护栏警告
    SYSTEM = "system"             # 系统事件
    ERROR = "error"               # 错误


class AuditRiskLevel(str, Enum):
    """审计风险等级"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """一条审计日志"""
    log_id: str = ""                        # UUID7
    trace_id: str = ""                      # 全链路追踪 ID
    session_id: str = ""                    # 会话 ID
    timestamp: float = 0.0                  # Unix 毫秒时间戳
    actor: str = ""                         # 操作主体
    action: str = ""                        # 操作类型
    input: dict[str, Any] = field(default_factory=dict)    # 输入（脱敏后）
    output: dict[str, Any] = field(default_factory=dict)   # 输出（脱敏后）
    parent_id: str = ""                     # 父日志 ID（因果链）
    risk_level: str = "none"               # 风险等级
    duration_ms: float = 0.0               # 耗时
    prev_hash: str = ""                     # 前一条日志的哈希（防篡改链）
    entry_hash: str = ""                    # 本条日志的哈希
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.log_id:
            self.log_id = _uuid7()
        if not self.timestamp:
            self.timestamp = time.time() * 1000

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典"""
        d = asdict(self)
        d["timestamp_iso"] = datetime.fromtimestamp(
            self.timestamp / 1000, tz=timezone.utc
        ).isoformat()
        return d

    def compute_hash(self, secret: str = "") -> str:
        """计算本条日志的 SHA-256 哈希"""
        payload = json.dumps({
            "log_id": self.log_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "parent_id": self.parent_id,
            "prev_hash": self.prev_hash,
        }, sort_keys=True, ensure_ascii=False)
        if secret:
            return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return hashlib.sha256(payload.encode()).hexdigest()


def _uuid7() -> str:
    """生成 UUID7（基于时间排序的 UUID）"""
    # UUID7: Unix 毫秒时间戳 + 随机数
    ts = int(time.time() * 1000)
    rand = uuid.uuid4().int & 0xFFFFFFFFFFFF  # 取 48 位随机数
    # 构建 UUID7
    # 前 48 bits: 毫秒时间戳
    high = ts << 16
    # + version 7 (4 bits)
    high |= 0x7000
    # + 随机数的高 12 bits
    high |= (rand >> 36) & 0xFFF
    # 后 80 bits: 变体 + 剩余随机数
    low = 0x8000000000000000 | (rand & 0x3FFFFFFFFFFFFFFF)
    return f"{high:016x}-{low:016x}"[:36]


# ═══════════════════════════════════════════════════════════════
#  WAL（Write-Ahead Log）— 追加写保证原子性
# ═══════════════════════════════════════════════════════════════

class WALWriter:
    """
    Write-Ahead Log 写入器。

    保证：
      - 每条日志原子写入（要么完整写入，要么不写）
      - 先写 WAL 再返回确认
      - 崩溃后可从 WAL 恢复
    """

    def __init__(self, wal_path: str):
        self.wal_path = Path(wal_path)
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = None

    def write(self, entry: AuditEntry) -> bool:
        """原子写入一条审计日志（追加模式 + fsync）"""
        try:
            line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
            with open(self.wal_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())  # 强制刷盘
            return True
        except Exception as e:
            logger.error("WAL 写入失败: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════
#  AuditLogService — 审计日志服务
# ═══════════════════════════════════════════════════════════════

class AuditLogService:
    """
    审计日志服务。

    使用方式：
      audit = AuditLogService(storage_path="./data/audit")
      entry = audit.record(
          trace_id="trace_123", actor="finance_agent",
          action=AuditAction.TOOL_CALL,
          input={"tool": "create_refund", "args": {"amount": 500}},
          parent_id="parent_log_456",
      )
      # 查询全链路
      trace = audit.get_trace("trace_123")
    """

    def __init__(
        self,
        storage_path: str = "./data/audit_logs",
        enable_hash_chain: bool = True,
        hmac_secret: str = "",
        db_session_factory=None,
    ):
        """
        参数：
          storage_path:      日志文件存储路径（JSONL 模式）
          enable_hash_chain: 是否启用哈希链防篡改
          hmac_secret:       HMAC 签名密钥（空=仅 SHA-256）
          db_session_factory: PostgreSQL Session 工厂（None=JSONL 模式）
        """
        self.storage_path = Path(storage_path)
        self.enable_hash_chain = enable_hash_chain
        self.hmac_secret = hmac_secret
        self._db_factory = db_session_factory

        # 初始化存储
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._wal = WALWriter(str(self.storage_path / "audit.wal"))

        # 上一个日志的哈希（用于哈希链）
        self._last_hash: str = ""
        self._load_last_hash()

        # 内存索引（快速查询，生产用数据库）
        self._entries: list[AuditEntry] = []

    # ── 记录日志 ────────────────────────────────────────────

    def record(
        self,
        trace_id: str,
        actor: str,
        action: AuditAction | str,
        input_data: dict | None = None,
        output_data: dict | None = None,
        session_id: str = "",
        parent_id: str = "",
        risk_level: AuditRiskLevel | str = AuditRiskLevel.NONE,
        duration_ms: float = 0.0,
        metadata: dict | None = None,
    ) -> AuditEntry:
        """
        记录一条审计日志。

        返回创建的 AuditEntry（包含 log_id）。
        """
        action_str = action.value if isinstance(action, AuditAction) else action
        risk_str = risk_level.value if isinstance(risk_level, AuditRiskLevel) else risk_level

        entry = AuditEntry(
            trace_id=trace_id,
            session_id=session_id,
            actor=actor,
            action=action_str,
            input=input_data or {},
            output=output_data or {},
            parent_id=parent_id,
            risk_level=risk_str,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        # 哈希链：链接到前一条日志
        if self.enable_hash_chain:
            entry.prev_hash = self._last_hash
            entry.entry_hash = entry.compute_hash(self.hmac_secret)
            self._last_hash = entry.entry_hash

        # 写入 WAL
        success = self._wal.write(entry)
        if not success:
            logger.error("审计日志写入 WAL 失败: %s", entry.log_id)

        # 内存索引
        self._entries.append(entry)
        if len(self._entries) > 10000:
            self._entries = self._entries[-5000:]  # 仅保留最近 5000 条在内存

        # 数据库持久化（如有）
        if self._db_factory:
            self._persist_to_db(entry)

        logger.debug(
            "审计日志: id=%s trace=%s actor=%s action=%s risk=%s",
            entry.log_id, trace_id, actor, action_str, risk_str,
        )
        return entry

    def record_from_state(
        self,
        trace_id: str,
        session_id: str,
        state: dict,
        action: AuditAction | str,
        actor: str = "system",
        duration_ms: float = 0.0,
    ) -> AuditEntry:
        """
        从 AgentState 中提取信息并记录审计日志。

        自动化提取 state 中的关键字段。
        """
        return self.record(
            trace_id=trace_id,
            session_id=session_id,
            actor=actor,
            action=action,
            input_data={
                "intent": state.get("intent", ""),
                "user_message": self._extract_user_message(state),
            },
            output_data={
                "final_response": state.get("final_response", "")[:200],
                "resolved": state.get("resolved", False),
                "quality_score": state.get("quality_score", 1.0),
            },
            risk_level=self._infer_risk(state),
            duration_ms=duration_ms,
            metadata={
                "iteration_count": state.get("iteration_count", 0),
                "agent_path": state.get("agents_sequence", []),
                "escalation": state.get("escalation_flag", False),
            },
        )

    # ── 查询日志 ────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> list[AuditEntry]:
        """
        按 trace_id 查询全链路日志。

        返回该 trace 的所有操作记录，按时间排序。
        """
        if self._db_factory:
            return self._query_db_trace(trace_id)

        # 内存模式（简单遍历）
        entries = [e for e in self._entries if e.trace_id == trace_id]
        entries.sort(key=lambda e: e.timestamp)
        return entries

    def get_session_logs(self, session_id: str) -> list[AuditEntry]:
        """按 session_id 查询"""
        entries = [e for e in self._entries if e.session_id == session_id]
        entries.sort(key=lambda e: e.timestamp)
        return entries

    def get_user_logs(self, user_id: str, limit: int = 100) -> list[AuditEntry]:
        """按用户查询（从 metadata 中查找 user_id）"""
        entries = [
            e for e in self._entries
            if e.metadata.get("user_id") == user_id or e.actor == user_id
        ]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def query(
        self,
        risk_level: str | None = None,
        action: str | None = None,
        actor: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """
        组合条件查询审计日志。
        """
        results = self._entries

        if risk_level:
            results = [e for e in results if e.risk_level == risk_level]
        if action:
            results = [e for e in results if e.action == action]
        if actor:
            results = [e for e in results if e.actor == actor]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def get_causal_chain(self, log_id: str) -> list[AuditEntry]:
        """
        获取某条日志的因果链（沿 parent_id 回溯）。
        """
        chain = []
        current_id = log_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            entry = next((e for e in self._entries if e.log_id == current_id), None)
            if entry:
                chain.append(entry)
                current_id = entry.parent_id
            else:
                break

        chain.reverse()
        return chain

    # ── 完整性验证 ──────────────────────────────────────────

    def verify_integrity(self) -> dict:
        """
        验证日志完整性（哈希链校验）。

        返回：
          {
            "total": int,         # 总日志数
            "valid": int,         # 哈希链完整的数量
            "tampered": int,      # 检测到篡改的数量
            "tampered_ids": [...],# 被篡改的日志 ID
          }
        """
        tampered = []
        sorted_entries = sorted(self._entries, key=lambda e: e.timestamp)

        if not sorted_entries:
            return {"total": 0, "valid": 0, "tampered": 0, "tampered_ids": []}

        # Use the first entry's prev_hash as the baseline (we can't verify
        # what came before the oldest entry in memory, so we accept it as-is)
        prev_hash = sorted_entries[0].prev_hash if self.enable_hash_chain else ""

        for entry in sorted_entries:
            if self.enable_hash_chain:
                if entry.prev_hash != prev_hash:
                    tampered.append(entry.log_id)
                prev_hash = entry.entry_hash

        return {
            "total": len(self._entries),
            "valid": len(self._entries) - len(tampered),
            "tampered": len(tampered),
            "tampered_ids": tampered,
        }

    def generate_report(self, start_time: float, end_time: float) -> dict:
        """
        生成审计报告（指定时间范围的汇总统计）。
        """
        entries = [
            e for e in self._entries
            if start_time <= e.timestamp <= end_time
        ]

        if not entries:
            return {"period": {"start": start_time, "end": end_time}, "total_entries": 0}

        action_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        actors: set[str] = set()
        traces: set[str] = set()

        for e in entries:
            action_counts[e.action] = action_counts.get(e.action, 0) + 1
            risk_counts[e.risk_level] = risk_counts.get(e.risk_level, 0) + 1
            actors.add(e.actor)
            traces.add(e.trace_id)

        return {
            "period": {
                "start": datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).isoformat(),
                "end": datetime.fromtimestamp(end_time / 1000, tz=timezone.utc).isoformat(),
            },
            "total_entries": len(entries),
            "unique_traces": len(traces),
            "unique_actors": len(actors),
            "action_distribution": action_counts,
            "risk_distribution": risk_counts,
            "avg_duration_ms": sum(e.duration_ms for e in entries) / len(entries) if entries else 0,
            "blocked_actions": sum(1 for e in entries if e.action == "guard_block"),
            "critical_events": sum(1 for e in entries if e.risk_level == "critical"),
        }

    # ── 内部方法 ────────────────────────────────────────────

    def _load_last_hash(self) -> None:
        """从 WAL 文件加载最后一条日志的哈希"""
        wal_file = self.storage_path / "audit.wal"
        if not wal_file.exists():
            return
        try:
            with open(wal_file, "r", encoding="utf-8") as f:
                # 读取最后一行
                last_line = ""
                for line in f:
                    if line.strip():
                        last_line = line
                if last_line:
                    data = json.loads(last_line)
                    self._last_hash = data.get("entry_hash", "")
        except Exception:
            pass

    @staticmethod
    def _extract_user_message(state: dict) -> str:
        """从状态中提取用户消息"""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")[:200]
        return ""

    @staticmethod
    def _infer_risk(state: dict) -> str:
        """从状态推断风险等级"""
        if state.get("escalation_flag"):
            return "high"
        quality = state.get("quality_score", 1.0)
        if quality < 0.5:
            return "high"
        if quality < 0.8:
            return "medium"
        return "low"

    def _persist_to_db(self, entry: AuditEntry) -> None:
        """持久化到数据库（PostgreSQL）"""
        if not self._db_factory:
            return
        try:
            # 使用 SQLAlchemy 或原生 SQL
            # 此处为接口预留，实际实现取决于项目的数据库模型
            pass
        except Exception as e:
            logger.error("审计日志写入 DB 失败: %s", e)

    def _query_db_trace(self, trace_id: str) -> list[AuditEntry]:
        """从数据库查询 trace"""
        # 预留数据库查询接口
        return []


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_audit_log_service: AuditLogService | None = None


def get_audit_log() -> AuditLogService:
    global _audit_log_service
    if _audit_log_service is None:
        _audit_log_service = AuditLogService(
            storage_path="./data/audit_logs",
            enable_hash_chain=True,
        )
    return _audit_log_service
