"""
订单查询工具

功能：按订单号或用户 ID 查询订单信息。
数据来源：data/seed/orders.json（114 条订单）

使用 @tool 装饰器——函数签名 + docstring 自动生成 JSON Schema。
"""

from __future__ import annotations

from typing import Any, Annotated

from langchain_core.tools import tool

from app.data.loader import get_loader


@tool
def query_order(
    order_id: Annotated[str, "订单号，格式如 ORD00001（必填）"],
    user_id: Annotated[str, "用户 ID，格式如 CUxxxx（可选，不传则不校验）"] = "",
) -> dict[str, Any]:
    """根据订单号查询订单的完整信息。

使用场景：
  - 当用户询问"我的订单怎么样了"、"查一下订单"、"东西发货了吗"时
  - 需要获取订单的详细信息：状态、商品清单、金额、收货地址、快递单号等

返回字段：
  - status: 订单状态（pending_payment=待支付 / paid=已支付 / shipped=已发货 / in_transit=运输中
             out_for_delivery=派送中 / delivered=已签收 / cancelled=已取消 / refunded=已退款
             return_requested=申请退货 / returning=退货中）
  - products: 商品列表（包含名称、数量、单价）
  - total_amount: 总金额
  - actual_paid: 实付金额
  - shipping_address: 收货地址
  - tracking_no: 快递单号
  - shipping_company: 快递公司

边界条件：
  - 如果订单不存在，返回 error 字段
  - 如果 user_id 和订单归属不匹配，返回越权错误"""
    loader = get_loader()
    order = loader.get_order(order_id)
    if order is None:
        return {"error": f"未找到订单 {order_id}", "order_id": order_id}
    # Demo/匿名模式：不校验归属，允许查看任意订单（方便演示）
    is_anon = (not user_id) or user_id.startswith("ANON_")
    if not is_anon and order.get("customer_id") != user_id:
        return {"error": f"订单 {order_id} 不属于用户 {user_id}（越权查询）", "order_id": order_id}
    result = dict(order)
    if is_anon:
        result.pop("customer_id", None)
        result.pop("customer_name", None)
    return result


@tool
def list_user_orders(
    user_id: Annotated[str, "用户 ID，格式如 CUxxxx（必填）"],
    page: Annotated[int, "页码，从 1 开始（可选，默认 1）"] = 1,
    page_size: Annotated[int, "每页返回的订单数量（可选，默认 10）"] = 10,
) -> dict[str, Any]:
    """查询指定用户的所有订单列表。

使用场景：
  - 当用户问"我有哪些订单"、"我买过什么"、"我的购物记录"时
  - 需要获取用户订单的概览（不含完整商品明细）

返回字段：
  - total: 订单总数
  - orders: 订单摘要列表（包含订单号、状态、金额、下单时间）

边界条件：
  - 如果用户没有订单，total 为 0，orders 为空列表"""
    loader = get_loader()
    # Demo/匿名模式：自动使用 CU0001（李伟）作为默认用户展示效果
    if (not user_id) or user_id.startswith("ANON_"):
        user_id = "CU0001"
    return loader.list_orders_by_user_paginated(user_id, page, page_size)
