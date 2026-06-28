"""
长期记忆系统 — 端到端测试

测试范围：
  1. Embedding 服务连通性（阿里云百炼 text-embedding-v4）
  2. LongTermMemory 存储 & 语义检索
  3. 用户画像获取
  4. 时间衰减权重
  5. 用户遗忘（数据删除）
  6. MemoryManager 三层记忆集成
  7. 降级搜索（无查询时按时间排序）

运行方式：
  cd multi-agent-cs
  python tests/test_long_term_memory.py
"""

from __future__ import annotations

import os
import sys

# ── 必须在最顶部设置：绕过 Windows 系统代理 ──────────────
os.environ.setdefault("NO_PROXY", "")
_no_proxy = os.environ["NO_PROXY"]
_hosts = "dashscope.aliyuncs.com,localhost,127.0.0.1,openaipublic.blob.core.windows.net,raw.githubusercontent.com"
for _h in _hosts.split(","):
    if _h not in _no_proxy:
        _no_proxy = f"{_no_proxy},{_h}" if _no_proxy else _h
os.environ["NO_PROXY"] = _no_proxy

# ── Qdrant 模式选择：优先本地文件模式（无需 Docker）──
# 如果有 Qdrant Docker，设置环境变量 QDRANT_HOST=127.0.0.1 QDRANT_PORT=6333 启用服务器模式
# 默认使用本地文件模式，无需任何外部服务
_LOCAL_MODE = not (os.environ.get("QDRANT_HOST") and os.environ.get("QDRANT_PORT"))
if _LOCAL_MODE:
    os.environ.setdefault("QDRANT_HOST", "")
    os.environ.setdefault("QDRANT_PORT", "0")

import time
import logging

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_long_term")


# ═══════════════════════════════════════════════════════════════
#  测试 1: Embedding 服务连通性
# ═══════════════════════════════════════════════════════════════

def test_embedding_connectivity():
    """验证阿里云百炼 Embedding 服务是否可用"""
    logger.info("=== 测试 1: Embedding 连通性 ===")

    import pytest
    import httpx
    from langchain_openai import OpenAIEmbeddings

    # 从环境变量读取
    api_key = os.environ.get("EMBEDDING_API_KEY", "")
    base_url = os.environ.get(
        "EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if not api_key:
        pytest.skip("EMBEDDING_API_KEY 未配置，跳过 Embedding 连通性测试")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
        http_client=httpx.Client(trust_env=False),  # 绕过 Windows 系统代理
    )

    # 测试单个向量化
    vec = embeddings.embed_query("你好世界")
    assert len(vec) == 1024, f"期望 1024 维向量，实际 {len(vec)} 维"
    assert all(isinstance(v, float) for v in vec), "向量元素应为浮点数"
    logger.info("  ✓ 单条向量化: dim=%d, first_3=%s", len(vec), vec[:3])

    # 测试批量向量化
    docs = ["用户喜欢微信沟通", "订单号 ORD-2024-001", "退款金额 199 元"]
    vecs = embeddings.embed_documents(docs)
    assert len(vecs) == 3, f"期望 3 个向量，实际 {len(vecs)}"
    assert all(len(v) == 1024 for v in vecs), "所有向量应为 1024 维"
    logger.info("  ✓ 批量向量化: count=%d, dim=%d", len(vecs), len(vecs[0]))

    logger.info("  ✅ Embedding 服务正常")
    return embeddings


# ═══════════════════════════════════════════════════════════════
#  测试 2: LongTermMemory 存储
# ═══════════════════════════════════════════════════════════════

def test_long_term_store():
    """验证长期记忆存储功能"""
    logger.info("=== 测试 2: LongTermMemory 存储 ===")

    from app.memory.long_term import LongTermMemory, MemoryCategory

    ltm = LongTermMemory()

    # 清理之前的测试残留
    for uid in ["USR-TEST-001", "USR-TEST-002"]:
        try:
            ltm.forget_user(uid)
        except Exception:
            pass

    initial_count = ltm.count()

    # 存储一条偏好记忆
    mem = ltm.store(
        user_id="USR-TEST-001",
        category=MemoryCategory.PREFERENCE,
        content="用户偏好通过微信接收通知，不喜欢短信和邮件",
        key="contact_preference",
        value="wechat",
        weight=0.9,
    )
    assert mem.user_id == "USR-TEST-001"
    assert mem.category == MemoryCategory.PREFERENCE
    assert mem.content == "用户偏好通过微信接收通知，不喜欢短信和邮件"
    assert mem.weight == 0.9
    logger.info("  ✓ 存储偏好: id=%s, content=%s", mem.id[:16], mem.content[:50])

    # 存储关键事实
    mem2 = ltm.store(
        user_id="USR-TEST-001",
        category=MemoryCategory.KEY_FACT,
        content="用户最近订单 ORD-2024-0615 购买了 MacBook Pro 16寸",
        key="recent_order",
        value="ORD-2024-0615",
        weight=0.8,
    )
    logger.info("  ✓ 存储关键事实: id=%s", mem2.id[:16])

    # 存储决策
    mem3 = ltm.store(
        user_id="USR-TEST-001",
        category=MemoryCategory.DECISION,
        content="客服已同意为用户升级到 VIP 等级，补偿 100 元优惠券",
        key="vip_upgrade_decision",
        value="vip_upgrade",
        weight=0.7,
    )
    logger.info("  ✓ 存储决策: id=%s", mem3.id[:16])

    # 存储另一个用户的记忆（测试隔离性）
    mem4 = ltm.store(
        user_id="USR-TEST-002",
        category=MemoryCategory.PREFERENCE,
        content="用户喜欢电话沟通，工作日 9-18 点可联系",
        key="contact_preference",
        weight=0.8,
    )
    logger.info("  ✓ 存储另一用户记忆: id=%s", mem4.id[:16])

    # 等待后台 Qdrant 写入完成
    ltm.flush(timeout=10.0)

    # 验证总数（只验证本次写入的测试用户数据）
    usr1_count = ltm.count("USR-TEST-001")
    usr2_count = ltm.count("USR-TEST-002")
    assert usr1_count >= 3, f"USR-TEST-001 预期 >=3 条，实际 {usr1_count}"
    assert usr2_count >= 1, f"USR-TEST-002 预期 >=1 条，实际 {usr2_count}"
    logger.info("  ✓ USR-TEST-001: %d 条, USR-TEST-002: %d 条", usr1_count, usr2_count)

    logger.info("  ✅ 存储功能正常")


# ═══════════════════════════════════════════════════════════════
#  测试 3: 语义搜索
# ═══════════════════════════════════════════════════════════════

def test_semantic_search(ltm):
    """验证语义搜索功能"""
    logger.info("=== 测试 3: 语义搜索 ===")

    # ── 3a: 按用户 + 语义搜索 ──────────────────────────────
    results = ltm.search(
        user_id="USR-TEST-001",
        query="联系方式偏好，通知渠道",
        top_k=3,
    )
    assert len(results) > 0, "应该至少有一个搜索结果"
    logger.info("  ✓ 语义搜索 '联系方式偏好': %d 条结果", len(results))
    for r in results:
        logger.info("    - [score=%.3f] %s", r.score, r.memory.content[:60])

    # 验证第一条结果的 content 与我们之前的写入相关
    # （语义搜索应该把"联系方式"相关的结果排在前面）
    top_content = results[0].memory.content
    assert "微信" in top_content or "偏好" in top_content or "通知" in top_content, \
        f"期望顶部结果与联系方式相关，实际: {top_content[:50]}"

    # ── 3b: 按类别过滤搜索 ──────────────────────────────────
    from app.memory.long_term import MemoryCategory
    results_filtered = ltm.search(
        user_id="USR-TEST-001",
        query="订单购买记录",
        category=MemoryCategory.KEY_FACT,
        top_k=2,
    )
    logger.info("  ✓ 类别过滤搜索 '订单购买记录' (KEY_FACT): %d 条结果", len(results_filtered))
    for r in results_filtered:
        logger.info("    - [%s] %s", r.memory.category.value, r.memory.content[:60])
        assert r.memory.category == MemoryCategory.KEY_FACT, \
            f"过滤结果应全是 KEY_FACT，实际: {r.memory.category}"

    # ── 3c: 用户隔离测试 ───────────────────────────────────
    results_usr2 = ltm.search(
        user_id="USR-TEST-002",
        query="联系方式",
        top_k=3,
    )
    logger.info("  ✓ 用户隔离搜索 (USR-TEST-002): %d 条结果", len(results_usr2))
    for r in results_usr2:
        assert r.memory.user_id == "USR-TEST-002", \
            f"隔离失败: 期望 USR-TEST-002，实际 {r.memory.user_id}"
        logger.info("    - %s", r.memory.content[:60])

    # ── 3d: 无查询降级（按时间排序） ────────────────────────
    results_no_query = ltm.search(
        user_id="USR-TEST-001",
        query="",
        top_k=2,
    )
    logger.info("  ✓ 无查询降级搜索: %d 条结果", len(results_no_query))

    logger.info("  ✅ 语义搜索功能正常")


# ═══════════════════════════════════════════════════════════════
#  测试 4: 用户画像
# ═══════════════════════════════════════════════════════════════

def test_user_profile(ltm):
    """验证用户画像获取"""
    logger.info("=== 测试 4: 用户画像 ===")

    profile = ltm.get_user_profile("USR-TEST-001")

    assert profile["user_id"] == "USR-TEST-001"
    assert len(profile["preferences"]) > 0, "应该至少有偏好记录"
    assert len(profile["key_facts"]) > 0, "应该至少有关键事实"
    assert len(profile["decisions"]) > 0, "应该至少有决策记录"

    logger.info("  ✓ 偏好: %d 条", len(profile["preferences"]))
    for p in profile["preferences"]:
        logger.info("    - %s (weight=%.1f)", p["content"][:50], p["weight"])

    logger.info("  ✓ 关键事实: %d 条", len(profile["key_facts"]))
    for f in profile["key_facts"]:
        logger.info("    - %s", f["content"][:60])

    logger.info("  ✓ 决策: %d 条", len(profile["decisions"]))

    # 验证摘要
    assert profile["summary"], "摘要不应为空"
    logger.info("  ✓ 摘要: %s", profile["summary"])

    logger.info("  ✅ 用户画像功能正常")


# ═══════════════════════════════════════════════════════════════
#  测试 5: 时间衰减权重
# ═══════════════════════════════════════════════════════════════

def test_memory_decay():
    """验证时间衰减权重计算"""
    logger.info("=== 测试 5: 时间衰减权重 ===")

    from app.memory.long_term import MemoryItem, MemoryCategory

    # 创建一条"很久以前"的记忆
    old_mem = MemoryItem(
        id="test-decay-001",
        user_id="USR-TEST-001",
        category="preference",
        content="旧偏好",
        weight=1.0,
        timestamp=time.time() - 86400 * 30,  # 30 天前
    )

    # 创建一条"最近"的记忆
    new_mem = MemoryItem(
        id="test-decay-002",
        user_id="USR-TEST-001",
        category="preference",
        content="新偏好",
        weight=1.0,
        timestamp=time.time(),  # 现在
    )

    old_weight = old_mem.get_decayed_weight()
    new_weight = new_mem.get_decayed_weight()

    logger.info("  ✓ 旧记忆 (30天前) 衰减权重: %.4f", old_weight)
    logger.info("  ✓ 新记忆 (现在) 衰减权重: %.4f", new_weight)
    assert old_weight < new_weight, f"旧记忆权重 ({old_weight:.4f}) 应低于新记忆 ({new_weight:.4f})"
    assert old_weight < 1.0, "30天前的记忆应该有衰减"
    assert new_weight > 0.99, f"新记忆权重应接近 1.0，实际 {new_weight:.4f}"

    logger.info("  ✅ 时间衰减功能正常")


# ═══════════════════════════════════════════════════════════════
#  测试 6: MemoryManager 集成测试
# ═══════════════════════════════════════════════════════════════

def test_memory_manager_integration():
    """验证 MemoryManager 与新的 LongTermMemory 集成"""
    logger.info("=== 测试 6: MemoryManager 集成 ===")

    from app.memory.memory_manager import MemoryManager

    mm = MemoryManager()

    # 创建会话
    session = mm.create_session("SESS-TEST-001")
    assert session is not None
    logger.info("  ✓ 会话已创建: SESS-TEST-001")

    # 模拟一轮对话
    mm.store_interaction(
        session_id="SESS-TEST-001",
        user_id="USR-MM-001",
        user_message="我不喜欢收垃圾邮件，以后都用微信联系我",
        assistant_message="好的，已记录您的偏好。以后我们会优先通过微信与您联系。",
        tool_calls=[{"name": "update_preference", "args": {"channel": "wechat"}}],
    )

    # 第二轮
    mm.store_interaction(
        session_id="SESS-TEST-001",
        user_id="USR-MM-001",
        user_message="我的订单 ORD-2024-0615 还没发货",
        assistant_message="已为您查询到订单状态：物流已揽收，预计明天送达。",
    )

    # 检索上下文
    ctx = mm.retrieve_context(
        session_id="SESS-TEST-001",
        user_id="USR-MM-001",
        current_message="物流状态",
    )

    assert "short_term" in ctx
    assert "long_term" in ctx
    assert "working_memory" in ctx

    logger.info("  ✓ 短期记忆: %d 条消息", len(ctx["short_term"]))
    logger.info("  ✓ 长期记忆: %s", ctx["long_term"][:100] if ctx["long_term"] else "(空)")
    logger.info("  ✓ 工作记忆: %s", ctx["working_memory"][:80] if ctx["working_memory"] else "(空)")

    # 验证 stats
    stats = mm.stats()
    logger.info("  ✓ 统计:")
    for k, v in stats.items():
        logger.info("    - %s: %s", k, v)

    # 清理
    mm.clear_all()
    logger.info("  ✓ MemoryManager 清理完成")

    logger.info("  ✅ MemoryManager 集成正常")


# ═══════════════════════════════════════════════════════════════
#  测试 7: 去重逻辑
# ═══════════════════════════════════════════════════════════════

def test_deduplication():
    """验证同 key 去重功能"""
    logger.info("=== 测试 7: 去重逻辑 ===")

    from app.memory.long_term import LongTermMemory, MemoryCategory

    ltm = LongTermMemory()
    user = "USR-DEDUP-001"

    # 第一次存储
    ltm.store(
        user_id=user,
        category=MemoryCategory.PREFERENCE,
        content="用户偏好微信",
        key="contact_pref",
        weight=0.5,
    )
    ltm.flush(timeout=5.0)
    count1 = ltm.count(user)

    # 第二次存储同 key（应该覆盖而非新增）
    ltm.store(
        user_id=user,
        category=MemoryCategory.PREFERENCE,
        content="用户偏好电话（已更新）",
        key="contact_pref",
        weight=0.9,
    )
    ltm.flush(timeout=5.0)
    count2 = ltm.count(user)

    logger.info("  ✓ 第一次存储后: %d 条", count1)
    logger.info("  ✓ 第二次存储后: %d 条", count2)
    assert count2 == count1, f"同 key 应该是覆盖而非新增，预期 {count1}，实际 {count2}"

    # 验证内容已更新
    results = ltm.search(user_id=user, query="联系方式偏好", top_k=1)
    if results:
        assert "电话" in results[0].memory.content, \
            f"应该是更新后的内容（含'电话'），实际: {results[0].memory.content}"

    ltm.forget_user(user)
    logger.info("  ✅ 去重逻辑正常")


# ═══════════════════════════════════════════════════════════════
#  测试 8: 批量操作
# ═══════════════════════════════════════════════════════════════

def test_batch_operations():
    """验证批量存储和遗忘"""
    logger.info("=== 测试 8: 批量操作 ===")

    from app.memory.long_term import LongTermMemory, MemoryCategory

    ltm = LongTermMemory()
    user = "USR-BATCH-001"

    # 批量存储
    items = [
        {"user_id": user, "category": MemoryCategory.PREFERENCE,
         "content": "偏好 1", "key": "pref_1", "weight": 0.6},
        {"user_id": user, "category": MemoryCategory.PREFERENCE,
         "content": "偏好 2", "key": "pref_2", "weight": 0.7},
        {"user_id": user, "category": MemoryCategory.KEY_FACT,
         "content": "事实 1", "key": "fact_1", "weight": 0.8},
    ]
    results = ltm.store_batch(items)
    assert len(results) == 3
    logger.info("  ✓ 批量存储: %d 条", len(results))

    ltm.flush(timeout=10.0)
    count = ltm.count(user)
    assert count == 3, f"预期 3 条，实际 {count}"
    logger.info("  ✓ 用户记忆数: %d", count)

    # 遗忘
    ltm.forget_user(user)
    ltm.flush(timeout=10.0)
    count_after = ltm.count(user)
    assert count_after == 0, f"遗忘后应为 0，实际 {count_after}"
    logger.info("  ✓ 遗忘后: %d 条", count_after)

    logger.info("  ✅ 批量操作正常")


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  长期记忆系统 — 端到端测试                    ║")
    logger.info("║  Qdrant + LangChain VectorStore + 百炼 v4    ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # 加载 .env
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    passed = 0
    failed = 0

    # 创建一个共享的 LongTermMemory 实例用于全部测试
    ltm_instance = test_long_term_store()

    tests = [
        ("Embedding 连通性", test_embedding_connectivity),
        ("LongTermMemory 存储", lambda: None),  # 已在上面完成
        ("语义搜索", lambda: test_semantic_search(ltm_instance)),
        ("用户画像", lambda: test_user_profile(ltm_instance)),
        ("时间衰减权重", test_memory_decay),
        ("MemoryManager 集成", test_memory_manager_integration),
        ("去重逻辑", test_deduplication),
        ("批量操作", test_batch_operations),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            logger.error("  ❌ %s 失败: %s", name, e)
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info("")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info(f"║  结果: {passed} 通过, {failed} 失败                        ║")
    logger.info("╚══════════════════════════════════════════════╝")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
