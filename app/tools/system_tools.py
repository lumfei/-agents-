"""
系统查询工具

功能：查询系统服务状态、用户信息、系统公告。
数据来源：用户信息来自 data/seed/customers.json（30 位客户）
         服务状态和公告仍为硬编码（seed 中无对应数据）
"""

from __future__ import annotations

from typing import Any, Annotated

from langchain_core.tools import tool

from app.data.loader import get_loader


MOCK_SERVICES: dict[str, str] = {
    "order_service": "正常", "payment_service": "正常",
    "logistics_service": "正常", "knowledge_base": "正常",
}

MOCK_ANNOUNCEMENTS: list[dict[str, str]] = [
    {"id": "ANN-001", "title": "系统维护通知",
     "content": "2024年1月25日凌晨2:00-4:00进行系统升级维护，期间部分服务可能不稳定。",
     "time": "2024-01-20 10:00:00", "level": "info"},
    {"id": "ANN-002", "title": "春节物流调整",
     "content": "春节期间（2月9日-2月17日）物流配送可能延迟，敬请谅解。",
     "time": "2024-01-18 09:00:00", "level": "warning"},
]


@tool
def check_service_status(
    service_name: Annotated[str, "服务名称（可选，不传则返回全部服务状态），如 order_service、payment_service"] = "",
) -> dict[str, Any]:
    """查询系统各服务的运行状态。

使用场景：
  - 排查问题时检查订单/支付/物流等服务是否正常
  - 用户反馈"用不了"时先验证服务状态

返回字段：
  - services: 服务状态字典
  - all_normal: 是否全部正常运行"""
    if service_name:
        status = MOCK_SERVICES.get(service_name)
        if status is None:
            return {"error": f"未知服务 '{service_name}'"}
        return {"services": {service_name: status}, "all_normal": status == "正常"}
    return {"services": dict(MOCK_SERVICES), "all_normal": all(v == "正常" for v in MOCK_SERVICES.values())}


@tool
def query_user_info(
    user_id: Annotated[str, "用户 ID，格式如 CU0001（必填）"],
) -> dict[str, Any]:
    """查询用户的基本信息和会员等级。

使用场景：
  - 需要验证用户身份时
  - 需要了解用户的会员等级（普通会员/银卡会员/金卡会员/钻石会员）
  - 查看用户的标签（如"高价值"、"投诉倾向"、"企业客户"等）

返回字段：
  - name: 用户姓名
  - phone: 手机号
  - city: 所在城市
  - level: 会员等级
  - total_orders: 历史订单总数
  - total_spent: 历史消费总额
  - tags: 用户标签列表
  - register_date: 注册日期

安全约束：
  - 只返回脱敏后的手机号
  - 不返回密码等敏感信息"""
    loader = get_loader()
    customer = loader.get_customer(user_id)
    if customer is None:
        # 兼容旧格式 USR-xxx → 尝试查找
        for cid, c in loader.customers.items():
            if c.get("name") == user_id or cid == user_id.replace("USR-", "CU"):
                customer = c
                break
    if customer is None:
        return {"error": f"未找到用户 {user_id}"}
    return dict(customer)


@tool
def get_system_announcements() -> dict[str, Any]:
    """获取当前的系统公告和通知。

使用场景：
  - 用户问"有什么通知"、"系统是不是在维护"
  - Agent 回复用户前，先查看是否有影响服务的公告

返回字段：
  - announcements: 公告列表（标题、内容、时间、级别）"""
    return {"total": len(MOCK_ANNOUNCEMENTS), "announcements": list(MOCK_ANNOUNCEMENTS)}
