"""
Checkpoint 持久化生命周期管理

这个模块管理 LangGraph checkpoint 存储后端的完整生命周期：
  startup  → init_checkpointer()   创建 PostgresSaver + 建表
  runtime  → get_checkpointer()    获取全局单例
  shutdown → close_checkpointer()  释放连接池

设计要点：
  - PostgresSaver 需要 asyncpg 异步驱动，init/setup/close 都是 async
  - MemorySaver 作为 fallback：PG 不可用或 CHECKPOINT_BACKEND=memory 时自动回退
  - 单例模式保证整个进程共享同一个连接池
  - 懒初始化：首次 get_checkpointer() 时若未 init，自动回退 MemorySaver

使用方式：
  # 在 FastAPI lifespan 中
  from app.graph.checkpoint import init_checkpointer, close_checkpointer
  await init_checkpointer(settings.postgres_dsn)

  # 在图编译时
  from app.graph.checkpoint import get_checkpointer
  checkpointer = get_checkpointer()
  app = graph.compile(checkpointer=checkpointer, ...)
"""

from __future__ import annotations

import logging
from typing import Union

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# PostgresSaver 是可选依赖，导入失败时自动回退 MemorySaver
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    _POSTGRES_AVAILABLE = True
except ImportError:
    PostgresSaver = None  # type: ignore[assignment]
    _POSTGRES_AVAILABLE = False

# 全局单例
_checkpointer: Union[MemorySaver, "PostgresSaver", None] = None


async def init_checkpointer(conn_string: str) -> Union[MemorySaver, "PostgresSaver"]:
    """
    启动时调用：创建 PostgresSaver 连接池 + 自动建表。

    参数：
      conn_string: PostgreSQL 连接字符串，格式 postgresql://user:pass@host:port/db

    返回：
      PostgresSaver 实例（PG 可用时）或 MemorySaver 实例（回退时）

    异常处理：
      PG 连接失败 → 自动回退 MemorySaver，不阻塞应用启动
    """
    global _checkpointer

    if not _POSTGRES_AVAILABLE:
        logger.warning(
            "langgraph-checkpoint-postgres 未安装，回退到 MemorySaver"
        )
        _checkpointer = MemorySaver()
        return _checkpointer

    try:
        cp = PostgresSaver.from_conn_string(conn_string)  # type: ignore[union-attr]
        await cp.setup()
        _checkpointer = cp
        logger.info(
            "PostgresSaver 已就绪: %s (checkpoint 表已创建)",
            _mask_conn_string(conn_string),
        )
        return cp
    except Exception as e:
        logger.warning(
            "PostgresSaver 初始化失败 (%s)，回退到 MemorySaver。"
            "服务重启后对话状态将丢失。",
            e,
        )
        _checkpointer = MemorySaver()
        return _checkpointer


async def close_checkpointer() -> None:
    """
    关闭时调用：释放 PostgresSaver 连接池。

    MemorySaver 无需关闭（no-op）。
    """
    global _checkpointer
    if _checkpointer is not None and hasattr(_checkpointer, "close"):
        try:
            await _checkpointer.close()  # type: ignore[union-attr]
            logger.info("PostgresSaver 连接池已释放")
        except Exception as e:
            logger.warning("PostgresSaver 关闭时出错: %s", e)
    _checkpointer = None


def get_checkpointer() -> Union[MemorySaver, "PostgresSaver"]:
    """
    获取全局 checkpointer 单例。

    图编译时调用。如果 init_checkpointer() 尚未调用（如测试环境），
    自动回退到 MemorySaver。

    返回：
      PostgresSaver（生产环境）或 MemorySaver（开发/测试/回退）
    """
    global _checkpointer
    if _checkpointer is None:
        logger.info("Checkpointer 尚未初始化，使用 MemorySaver（开发/测试模式）")
        _checkpointer = MemorySaver()
    return _checkpointer


def reset_checkpointer() -> None:
    """
    重置 checkpointer 单例（仅用于测试）。

    测试中切换后端时使用：先 reset，再 init 新的后端。
    """
    global _checkpointer
    _checkpointer = None


def _mask_conn_string(conn_string: str) -> str:
    """隐藏连接字符串中的密码，安全输出到日志"""
    import re
    return re.sub(r':([^:@]+)@', ':****@', conn_string)
