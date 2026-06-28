"""
统一数据加载器 — 从 data/seed/ JSON 文件加载 Mock 数据

替代各 tools/*.py 中硬编码的 Python 字典，好处：
  - 数据与代码分离，编辑 JSON 即可增删改数据
  - 种子数据更丰富：114 订单 / 30 客户 / 全量物流 / 40 退款 / 25 商品 / 15 KB
  - 启动时一次性加载到内存，查询 O(1)

使用方式：
  from app.data.loader import get_loader
  loader = get_loader()
  order = loader.get_order("ORD00001")
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Optional


# JSON 文件相对于本文件的路径
_SEED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "seed")


class DataLoader:
    """统一数据加载器，启动时加载所有 seed JSON 到内存。"""

    def __init__(self):
        self.orders: dict[str, dict] = {}           # order_id → order
        self.customers: dict[str, dict] = {}         # customer_id → customer
        self.logistics: dict[str, dict] = {}          # tracking_no → logistics
        self.logistics_by_order: dict[str, str] = {}  # order_id → tracking_no
        self.refunds: dict[str, dict] = {}            # refund_id → refund
        self.products: dict[str, dict] = {}           # product_id → product
        self.kb_articles: list[dict] = []             # knowledge base articles

        self._refund_counter = 0
        self._loaded = False

    # ══════════════════════════════════════════════════════════
    #  加载
    # ══════════════════════════════════════════════════════════

    def load_all(self):
        """加载所有 JSON 文件（幂等，已加载则跳过）。"""
        if self._loaded:
            return
        self._load_orders()
        self._load_customers()
        self._load_logistics()
        self._load_refunds()
        self._load_products()
        self._load_knowledge_base()
        self._loaded = True

    def _read_json(self, filename: str):
        path = os.path.join(_SEED_DIR, filename)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_orders(self):
        for o in self._read_json("orders.json"):
            self.orders[o["order_id"]] = o

    def _load_customers(self):
        for c in self._read_json("customers.json"):
            self.customers[c["customer_id"]] = c

    def _load_logistics(self):
        for l in self._read_json("logistics.json"):
            self.logistics[l["tracking_no"]] = l
            if l.get("order_id"):
                self.logistics_by_order[l["order_id"]] = l["tracking_no"]

    def _load_refunds(self):
        for r in self._read_json("refunds.json"):
            self.refunds[r["refund_id"]] = r
            # 追踪退款计数器（用于生成新 refund_id）
            try:
                num = int(r["refund_id"][2:])  # "RF0001" → 1
                if num > self._refund_counter:
                    self._refund_counter = num
            except (ValueError, IndexError):
                pass

    def _load_products(self):
        for p in self._read_json("products.json"):
            self.products[p["product_id"]] = p

    def _load_knowledge_base(self):
        self.kb_articles = self._read_json("knowledge_base.json")

    # ══════════════════════════════════════════════════════════
    #  订单查询
    # ══════════════════════════════════════════════════════════

    def get_order(self, order_id: str) -> Optional[dict]:
        """根据订单号查询订单。"""
        return self.orders.get(order_id)

    def list_orders_by_user(self, customer_id: str) -> list[dict]:
        """查询指定用户的所有订单（简化摘要）。"""
        return [o for o in self.orders.values() if o.get("customer_id") == customer_id]

    def list_orders_by_user_paginated(
        self, customer_id: str, page: int = 1, page_size: int = 10
    ) -> dict:
        """分页查询用户订单。"""
        user_orders = self.list_orders_by_user(customer_id)
        total = len(user_orders)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "orders": [
                {
                    "order_id": o["order_id"],
                    "status": o["status"],
                    "total_amount": o["total_amount"],
                    "order_time": o.get("created_at", ""),
                }
                for o in user_orders[start:end]
            ],
        }

    # ══════════════════════════════════════════════════════════
    #  客户查询
    # ══════════════════════════════════════════════════════════

    def get_customer(self, customer_id: str) -> Optional[dict]:
        """根据 customer_id 查询客户信息。"""
        return self.customers.get(customer_id)

    # ══════════════════════════════════════════════════════════
    #  物流查询
    # ══════════════════════════════════════════════════════════

    def get_logistics(self, tracking_no: str) -> Optional[dict]:
        """根据运单号查询物流信息。"""
        return self.logistics.get(tracking_no)

    def get_logistics_by_order(self, order_id: str) -> Optional[dict]:
        """根据订单号查询物流信息。"""
        tracking_no = self.logistics_by_order.get(order_id)
        if tracking_no:
            return self.logistics.get(tracking_no)
        return None

    # ══════════════════════════════════════════════════════════
    #  退款管理
    # ══════════════════════════════════════════════════════════

    def get_refund(self, refund_id: str) -> Optional[dict]:
        """查询退款状态。"""
        return self.refunds.get(refund_id)

    def get_refund_by_order(self, order_id: str) -> Optional[dict]:
        """查询某订单的退款（返回第一个匹配的 pending/refunding）。"""
        for r in self.refunds.values():
            if r.get("order_id") == order_id and r.get("status") in ("pending_approval", "refunding", "approved"):
                return r
        return None

    def create_refund(self, order_id: str, amount: float, reason: str, customer_id: str) -> dict:
        """创建退款申请（写入内存）。"""
        self._refund_counter += 1
        refund_id = f"RF{self._refund_counter:04d}"
        needs_approval = amount > 1000.0
        refund = {
            "refund_id": refund_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "amount": amount,
            "reason": reason,
            "status": "pending_approval" if needs_approval else "approved",
            "type": "仅退款",
            "hitl_required": needs_approval,
            "hitl_approved": None,
            "created_at": "",  # 由调用方补充
            "processed_at": None,
            "solution_note": "",
        }
        self.refunds[refund_id] = refund
        return dict(refund)

    # ══════════════════════════════════════════════════════════
    #  知识库检索
    # ══════════════════════════════════════════════════════════

    def search_kb(
        self, query: str, category: str = "", top_k: int = 3
    ) -> dict:
        """关键词匹配搜索知识库。"""
        query_lower = query.lower()

        def _score(article: dict) -> float:
            score = 0.0
            tags = article.get("tags", [])
            title = article.get("title", "")
            content = article.get("content", "")
            # 标题匹配权重更高
            for tag in tags:
                if tag in query:
                    score += 0.4
            for word in query_lower.split():
                if word in title:
                    score += 0.3
                if word in content[:500]:
                    score += 0.1
            # 类别过滤
            if category and article.get("category") != category:
                return -1.0
            return min(score, 1.0)

        scored = [(a, _score(a)) for a in self.kb_articles if _score(a) > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        return {
            "query": query,
            "total_found": min(len(scored), top_k),
            "results": [
                {
                    "title": a["title"],
                    "content": a["content"],
                    "category": a.get("category", ""),
                    "score": round(s, 2),
                }
                for a, s in scored[:top_k]
            ],
        }

    # ══════════════════════════════════════════════════════════
    #  商品查询
    # ══════════════════════════════════════════════════════════

    def get_product(self, product_id: str) -> Optional[dict]:
        """根据 product_id 查询商品信息。"""
        return self.products.get(product_id)

    # ══════════════════════════════════════════════════════════
    #  统计
    # ══════════════════════════════════════════════════════════

    def stats(self) -> dict:
        return {
            "orders": len(self.orders),
            "customers": len(self.customers),
            "logistics": len(self.logistics),
            "refunds": len(self.refunds),
            "products": len(self.products),
            "kb_articles": len(self.kb_articles),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"DataLoader(orders={s['orders']}, customers={s['customers']}, "
            f"logistics={s['logistics']}, refunds={s['refunds']}, "
            f"products={s['products']}, kb={s['kb_articles']})"
        )


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_loader: Optional[DataLoader] = None


@lru_cache(maxsize=1)
def get_loader() -> DataLoader:
    """获取全局唯一的 DataLoader 实例（自动加载数据）。"""
    global _loader
    if _loader is None:
        _loader = DataLoader()
        _loader.load_all()
    return _loader
