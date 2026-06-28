"""
pytest 共享配置

职责：
  - 定义所有测试共享的 fixture
  - 提供 mock 对象（MockLLM、MockRedis 等）
  - 测试配置管理
  - LongTermMemory fixture（供 test_semantic_search / test_user_profile 使用）
"""

import os
import sys
import pytest

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 环境变量设置（绕过代理 + Qdrant 配置） ──
os.environ.setdefault("NO_PROXY", "")
_no_proxy = os.environ["NO_PROXY"]
_hosts = "dashscope.aliyuncs.com,api.deepseek.com,localhost,127.0.0.1"
for _h in _hosts.split(","):
    if _h not in _no_proxy:
        _no_proxy = f"{_no_proxy},{_h}" if _no_proxy else _h
os.environ["NO_PROXY"] = _no_proxy


# ═══════════════════════════════════════════════════════════════
#  LongTermMemory fixture — 为语义搜索/用户画像测试提供共享实例
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def ltm():
    """
    创建 LongTermMemory 实例并预填充测试数据。

    Function 级别：每个测试函数获取独立实例（避免 Qdrant 本地模式文件锁冲突）。
    测试结束后自动清理测试用户数据。

    注意：QdrantClient 在模块级别是全局单例（_qdrant_client），
    多个 LongTermMemory() 实例共享同一个底层连接，不会创建重复客户端。
    """
    from app.memory.long_term import LongTermMemory, MemoryCategory

    instance = LongTermMemory()

    # 清理残留的测试数据
    for uid in ["USR-TEST-001", "USR-TEST-002"]:
        try:
            instance.forget_user(uid)
        except Exception:
            pass

    # 存储偏好记忆
    instance.store(
        user_id="USR-TEST-001",
        category=MemoryCategory.PREFERENCE,
        content="用户偏好通过微信接收通知，不喜欢短信和邮件",
        key="contact_preference",
        value="wechat",
        weight=0.9,
    )

    # 存储关键事实
    instance.store(
        user_id="USR-TEST-001",
        category=MemoryCategory.KEY_FACT,
        content="用户最近订单 ORD-2024-0615 购买了 MacBook Pro 16寸",
        key="recent_order",
        value="ORD-2024-0615",
        weight=0.8,
    )

    # 存储决策
    instance.store(
        user_id="USR-TEST-001",
        category=MemoryCategory.DECISION,
        content="客服已同意为用户升级到 VIP 等级，补偿 100 元优惠券",
        key="vip_upgrade_decision",
        value="vip_upgrade",
        weight=0.7,
    )

    # 存储另一个用户的记忆（用于隔离测试）
    instance.store(
        user_id="USR-TEST-002",
        category=MemoryCategory.PREFERENCE,
        content="用户喜欢电话沟通，工作日 9-18 点可联系",
        key="contact_preference",
        weight=0.8,
    )

    # 等待后台 Qdrant 写入完成，确保后续测试能读到数据
    instance.flush(timeout=10.0)

    yield instance

    # 清理
    for uid in ["USR-TEST-001", "USR-TEST-002"]:
        try:
            instance.forget_user(uid)
        except Exception:
            pass
