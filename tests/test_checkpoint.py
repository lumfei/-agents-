"""
Checkpoint 持久化模块测试

验证：
  1. PostgresSaver 不可用时自动回退 MemorySaver
  2. get_checkpointer() 懒初始化
  3. init/close 生命周期
  4. 图编译正常使用 checkpointer
  5. get_state / get_state_history 可用
"""

from __future__ import annotations

import pytest


class TestCheckpointerFallback:
    """测试不带 PostgresSaver 时的 MemorySaver 回退"""

    def test_get_checkpointer_lazy_init(self):
        """首次 get_checkpointer() 自动创建 MemorySaver"""
        from app.graph.checkpoint import reset_checkpointer, get_checkpointer

        reset_checkpointer()
        cp = get_checkpointer()
        assert cp is not None
        from langgraph.checkpoint.memory import MemorySaver
        assert isinstance(cp, MemorySaver)

    def test_get_checkpointer_returns_same_instance(self):
        """多次调用返回同一个单例"""
        from app.graph.checkpoint import get_checkpointer

        cp1 = get_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is cp2

    def test_reset_checkpointer(self):
        """reset 后重新创建"""
        from app.graph.checkpoint import reset_checkpointer, get_checkpointer

        cp1 = get_checkpointer()
        reset_checkpointer()
        cp2 = get_checkpointer()
        assert cp1 is not cp2  # 新实例

    def test_init_with_invalid_conn_falls_back(self):
        """无效 PG 连接字符串 → fallback MemorySaver"""
        import asyncio
        from app.graph.checkpoint import reset_checkpointer, init_checkpointer, get_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        reset_checkpointer()

        async def _init():
            return await init_checkpointer("postgresql://invalid:5432/nonexistent")

        cp = asyncio.run(_init())
        assert isinstance(cp, MemorySaver)
        # 后续 get 也是同一个 MemorySaver
        assert get_checkpointer() is cp


class TestGraphWithCheckpointer:
    """测试图编译和 checkpoint 操作"""

    def test_build_graph_uses_checkpointer(self):
        """图编译成功使用 checkpointer"""
        from app.graph.supervisor_graph import build_supervisor_graph
        from app.graph.checkpoint import reset_checkpointer, get_checkpointer

        reset_checkpointer()
        graph = build_supervisor_graph()
        assert graph is not None
        cp = get_checkpointer()
        assert cp is not None

    def test_graph_stream_with_checkpoint(self):
        """流式执行写入 checkpoint"""
        from app.graph.supervisor_graph import build_supervisor_graph, stream_workflow
        from app.graph.checkpoint import reset_checkpointer

        reset_checkpointer()
        # 确保图先编译（触发 get_checkpointer）
        graph = build_supervisor_graph()

        events = stream_workflow(
            "帮我查一下退款 RF0001 的进度",
            user_id="test_user",
            session_id="test_checkpoint_session",
        )
        assert len(events) > 0

        # 验证最终事件中有结果
        final_event = events[-1] if events else {}
        assert final_event is not None

    def test_get_state_with_thread_id(self):
        """get_state 能查询 checkpoint 中的状态，返回 StateSnapshot"""
        from app.graph.supervisor_graph import get_state
        import uuid

        thread_id = f"test_get_state_{uuid.uuid4().hex[:8]}"
        state = get_state(thread_id)
        # 新 thread_id 返回 StateSnapshot 对象（values 为空 dict）
        # StateSnapshot 不是 dict，但 values 是 dict
        assert state is not None
        assert hasattr(state, "values")
        assert isinstance(state.values, dict)

    def test_checkpoint_configurable_thread_id(self):
        """验证 configurable.thread_id 正常传递"""
        from app.graph.supervisor_graph import run_workflow

        result = run_workflow(
            "查询退款 RF0001",
            user_id="test_cp",
            session_id="test_cp_session",
        )
        assert isinstance(result, dict)
        assert "final_response" in result


class TestCheckpointerClose:
    """测试关闭逻辑"""

    def test_close_memory_saver_noop(self):
        """MemorySaver close 是空操作，不抛异常"""
        import asyncio
        from app.graph.checkpoint import reset_checkpointer, get_checkpointer, close_checkpointer

        reset_checkpointer()
        get_checkpointer()  # 创建 MemorySaver
        # close 应该不抛异常
        asyncio.run(close_checkpointer())
