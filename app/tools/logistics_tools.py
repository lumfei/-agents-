"""
物流工具

功能：查询物流轨迹、按订单号查物流。
数据来源：data/seed/logistics.json（全量物流轨迹，含详细 tracking_events）
"""

from __future__ import annotations

from typing import Any, Annotated

from langchain_core.tools import tool

from app.data.loader import get_loader


@tool
def track_logistics(
    tracking_no: Annotated[str, "运单号/快递单号，由快递公司提供（必填）"],
) -> dict[str, Any]:
    """根据运单号（快递单号）查询物流轨迹。

使用场景：
  - 用户提供了运单号，想知道快递到哪里了
  - 用户问"我的快递怎么还没到"、"物流信息是什么"

返回字段：
  - company: 快递公司名称
  - current_status: 当前状态（已揽收/运输中/到达中转站/派送中/已签收/异常）
  - current_location: 当前所在位置
  - estimated_delivery: 预计送达日期
  - tracking_events: 轨迹列表，包含时间、地点、事件描述和操作人
  - recipient_name: 收件人
  - recipient_address: 收件地址

边界条件：
  - 如果运单号不存在，返回 error 字段"""
    loader = get_loader()
    data = loader.get_logistics(tracking_no)
    if data is None:
        return {"error": f"未找到运单号 {tracking_no} 的物流信息", "tracking_no": tracking_no}
    return dict(data)


@tool
def query_logistics_by_order(
    order_id: Annotated[str, "订单号，格式如 ORD00001（必填）"],
) -> dict[str, Any]:
    """根据订单号查询物流信息。

使用场景：
  - 用户只知道订单号，不知道运单号，但想查快递到哪了
  - 用户说"我的订单 ORDxxx 到哪了"

返回字段：
  - tracking_no: 关联的运单号
  - company: 快递公司
  - current_status: 当前状态
  - tracking_events: 物流轨迹

边界条件：
  - 如果订单未发货（无物流信息），返回提示信息"""
    loader = get_loader()
    data = loader.get_logistics_by_order(order_id)
    if data is None:
        return {"error": f"订单 {order_id} 暂无物流信息（可能未发货）", "order_id": order_id}
    return dict(data)
