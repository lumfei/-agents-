"""
知识图谱构建器 — 从知识库文章中提取实体和关系

轻量设计:
  - 使用 networkx 构建内存图（不引入 Neo4j）
  - 实体来源: 文章 tags + 标题中的关键名词 + 类别
  - 关系来源: 共享实体、同类别、内容交叉引用
  - 支持 JSON 序列化/反序列化

持久化: 图数据保存为 JSON，启动时加载即可
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── 实体类型 ─────────────────────────────────────────────

ENTITY_PRODUCT = "product"       # 产品/商品类别
ENTITY_POLICY = "policy"         # 政策/规则
ENTITY_PROCESS = "process"       # 流程/步骤
ENTITY_DEPT = "department"       # 部门/角色
ENTITY_STATUS = "status"         # 状态/条件

# ── 关系类型 ─────────────────────────────────────────────

REL_INVOLVES = "involves"        # 涉及（政策涉及退款、产品涉及保修）
REL_DEPENDS_ON = "depends_on"    # 依赖（退款依赖退货完成）
REL_TRIGGERS = "triggers"        # 触发（质量问题触发退货流程）
REL_BELONGS_TO = "belongs_to"    # 属于（文章属于某类别）
REL_RELATED_TO = "related_to"    # 相关（两篇文章共享实体）


class KnowledgeGraph:
    """知识图谱 — networkx Graph 的轻量封装"""

    def __init__(self):
        self._graph = None  # type: Optional[Any]  # networkx.Graph, 懒加载
        self._articles: dict[str, dict] = {}       # kb_id → article
        self._entity_to_articles: dict[str, set] = {}  # 实体名 → {kb_id, ...}

    @property
    def graph(self):
        if self._graph is None:
            import networkx as nx
            self._graph = nx.Graph()
        return self._graph

    def is_built(self) -> bool:
        """是否已构建图"""
        return self._graph is not None and self.graph.number_of_nodes() > 0

    # ══════════════════════════════════════════════════════════
    #  实体提取（基于规则，不调 LLM）
    # ══════════════════════════════════════════════════════════

    # 常见产品名词（用于从 tag/title/content 提取产品实体）
    _PRODUCT_PATTERNS = [
        "手机", "笔记本", "电脑", "平板", "iPad", "iPhone", "MacBook",
        "耳机", "充电器", "手表", "音箱", "投影仪", "咖啡机", "吸尘器",
        "空气净化器", "电动牙刷", "灯带", "鼠标", "键盘", "显示器",
        "冲锋衣", "跑鞋", "夹克", "智能", "穿戴", "家电", "配件",
    ]

    # 常见政策/流程关键词
    _POLICY_PATTERNS = [
        "退货", "退款", "保修", "质保", "价格保护", "价保", "发票",
        "会员", "积分", "优惠券", "跨境", "清关", "关税",
        "无理由退货", "质量问题", "赔偿", "补偿", "投诉",
    ]

    # 常见流程关键词
    _PROCESS_PATTERNS = [
        "维修", "返厂", "退换", "售后", "配送", "物流", "发货",
        "支付", "分期", "审核", "审批", "签收", "拒收",
    ]

    # 常见部门/角色
    _DEPT_PATTERNS = [
        "客服", "法务", "管理层", "主管", "维修中心", "仓库",
        "财务", "海关",
    ]

    def extract_entities(self, article: dict) -> list[dict]:
        """从一篇文章中提取实体列表（规则匹配）"""
        entities: list[dict] = []
        kb_id = article.get("kb_id", "")
        title = article.get("title", "")
        category = article.get("category", "")
        tags = article.get("tags", [])
        content = article.get("content", "")

        text = f"{title} {category} {' '.join(tags)} {content[:500]}"

        # 1. 从 tags 提取产品实体
        for tag in tags:
            for pattern in self._PRODUCT_PATTERNS:
                if pattern in tag or pattern in title:
                    entities.append({
                        "name": tag, "type": ENTITY_PRODUCT,
                        "source": kb_id, "confidence": 0.8,
                    })
                    break

        # 2. 从内容提取政策实体
        for pattern in self._POLICY_PATTERNS:
            if pattern in title or pattern in content[:300]:
                entities.append({
                    "name": pattern, "type": ENTITY_POLICY,
                    "source": kb_id, "confidence": 0.7,
                })

        # 3. 从内容提取流程实体
        for pattern in self._PROCESS_PATTERNS:
            if pattern in title or pattern in content[:300]:
                entities.append({
                    "name": pattern, "type": ENTITY_PROCESS,
                    "source": kb_id, "confidence": 0.7,
                })

        # 4. 从内容提取部门实体
        for pattern in self._DEPT_PATTERNS:
            if pattern in title or pattern in content[:300]:
                entities.append({
                    "name": pattern, "type": ENTITY_DEPT,
                    "source": kb_id, "confidence": 0.6,
                })

        # 5. 类别本身也是实体
        if category:
            entities.append({
                "name": category, "type": "category",
                "source": kb_id, "confidence": 1.0,
            })

        # 去重（按 name + type）
        seen: set[tuple] = set()
        unique: list[dict] = []
        for e in entities:
            key = (e["name"], e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    # ══════════════════════════════════════════════════════════
    #  关系提取
    # ══════════════════════════════════════════════════════════

    def extract_relations(
        self, articles: list[dict], article_entities: dict[str, list[dict]],
    ) -> list[tuple[str, str, str]]:
        """提取实体间关系（返回 (source, target, relation_type) 三元组）"""
        relations: list[tuple[str, str, str]] = []
        kb_ids = [a["kb_id"] for a in articles]

        # 1. 同类别文章 → belongs_to 关系
        for a in articles:
            cat = a.get("category", "")
            if cat:
                relations.append((a["kb_id"], cat, REL_BELONGS_TO))

        # 2. 共享实体的文章 → related_to 关系
        # 构建实体→文章索引
        entity_to_kb: dict[str, set] = {}
        for kb_id, entities in article_entities.items():
            for e in entities:
                name = e["name"]
                if name not in entity_to_kb:
                    entity_to_kb[name] = set()
                entity_to_kb[name].add(kb_id)

        # 两个不同文章共享至少一个实体 → related_to
        self._entity_to_articles = {}
        for entity_name, kb_set in entity_to_kb.items():
            self._entity_to_articles[entity_name] = kb_set
            kb_list = sorted(kb_set)
            for i in range(len(kb_list)):
                for j in range(i + 1, len(kb_list)):
                    relations.append((kb_list[i], kb_list[j], REL_RELATED_TO))

        # 3. 互补实体关系（退款→退货、保修→维修 等）
        complementary_pairs = [
            ("退款", "退货", REL_DEPENDS_ON),
            ("退货", "退款", REL_TRIGGERS),
            ("保修", "维修", REL_INVOLVES),
            ("维修", "保修", REL_DEPENDS_ON),
            ("投诉", "赔偿", REL_TRIGGERS),
            ("售后", "维修", REL_INVOLVES),
            ("发票", "支付", REL_INVOLVES),
            ("支付", "退款", REL_INVOLVES),
            ("清关", "跨境", REL_DEPENDS_ON),
            ("会员", "积分", REL_INVOLVES),
            ("物流", "配送", REL_INVOLVES),
            ("价保", "退款", REL_TRIGGERS),
        ]
        for src, tgt, rel_type in complementary_pairs:
            if src in entity_to_kb and tgt in entity_to_kb:
                for sid in entity_to_kb[src]:
                    for tid in entity_to_kb[tgt]:
                        if sid != tid:
                            relations.append((sid, tid, rel_type))

        # 去重
        seen: set[tuple] = set()
        unique: list[tuple[str, str, str]] = []
        for r in relations:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique

    # ══════════════════════════════════════════════════════════
    #  图构建
    # ══════════════════════════════════════════════════════════

    def build_from_articles(self, articles: list[dict]) -> dict:
        """从知识库文章列表构建知识图谱。返回统计信息。"""
        logger.info("开始构建知识图谱，文章数: %d", len(articles))

        # 1. 提取所有文章的实体
        article_entities: dict[str, list[dict]] = {}
        for a in articles:
            kb_id = a.get("kb_id", "")
            article_entities[kb_id] = self.extract_entities(a)

        total_entities = sum(len(v) for v in article_entities.values())
        logger.info("提取实体: %d 个", total_entities)

        # 2. 提取关系
        relations = self.extract_relations(articles, article_entities)
        logger.info("提取关系: %d 条", len(relations))

        # 3. 构建图
        import networkx as nx

        # 添加文章节点
        for a in articles:
            kb_id = a["kb_id"]
            self._articles[kb_id] = {
                "kb_id": kb_id,
                "title": a.get("title", ""),
                "category": a.get("category", ""),
                "tags": a.get("tags", []),
            }
            self.graph.add_node(
                kb_id,
                type="article",
                title=a.get("title", ""),
                category=a.get("category", ""),
                tags=a.get("tags", []),
            )

        # 添加实体节点
        all_entities: dict[str, dict] = {}
        for kb_id, entities in article_entities.items():
            for e in entities:
                name = e["name"]
                etype = e["type"]
                node_id = f"entity:{name}"
                if node_id not in all_entities:
                    all_entities[node_id] = {"name": name, "type": etype, "articles": set()}
                all_entities[node_id]["articles"].add(kb_id)
                # entity ↔ article 边
                self.graph.add_node(node_id, type="entity", name=name, entity_type=etype)
                self.graph.add_edge(node_id, kb_id, relation=REL_INVOLVES)

        # 添加关系边（文章 ↔ 文章）
        for src, tgt, rel_type in relations:
            if self.graph.has_node(src) and self.graph.has_node(tgt):
                self.graph.add_edge(src, tgt, relation=rel_type)

        stats = {
            "article_nodes": len(articles),
            "entity_nodes": len(all_entities),
            "total_relations": len(relations) + len(articles),  # + 文章-实体边
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }
        logger.info("知识图谱构建完成: %s", stats)
        return stats

    # ══════════════════════════════════════════════════════════
    #  图搜索
    # ══════════════════════════════════════════════════════════

    def expand_from_articles(
        self, kb_ids: list[str], max_depth: int = 1, max_results: int = 10,
    ) -> list[str]:
        """
        从种子文章出发，通过图关系扩展找到相关文章。

        参数:
          kb_ids: 种子文章 ID 列表
          max_depth: BFS 最大深度
          max_results: 最多返回多少篇相关文章

        返回:
          相关文章 ID 列表（按距离排序，不含种子文章）
        """
        if not self.is_built():
            return []

        import networkx as nx

        related: dict[str, int] = {}  # kb_id → distance
        visited: set = set(kb_ids)

        for seed in kb_ids:
            if not self.graph.has_node(seed):
                continue
            # 1-hop：所有邻居
            for neighbor in self.graph.neighbors(seed):
                if neighbor.startswith("entity:"):
                    # 通过实体节点跳到其他文章
                    for two_hop in self.graph.neighbors(neighbor):
                        if two_hop not in visited and two_hop in self._articles:
                            if two_hop not in related or related[two_hop] > 1:
                                related[two_hop] = 1
                elif neighbor in self._articles and neighbor not in visited:
                    if neighbor not in related or related[neighbor] > 1:
                        related[neighbor] = 1

        # 按距离排序，限制数量
        sorted_related = sorted(related.items(), key=lambda x: (x[1], x[0]))
        return [kb_id for kb_id, _ in sorted_related[:max_results]]

    # ══════════════════════════════════════════════════════════
    #  序列化
    # ══════════════════════════════════════════════════════════

    def to_dict(self) -> dict:
        """序列化为 dict（不含 networkx graph）"""
        import networkx as nx
        return {
            "article_count": len(self._articles),
            "node_count": self.graph.number_of_nodes() if self._graph else 0,
            "edge_count": self.graph.number_of_edges() if self._graph else 0,
            "nodes": [
                {"id": n, **self.graph.nodes[n]}
                for n in self.graph.nodes()
            ] if self._graph else [],
            "edges": [
                {"source": u, "target": v, **self.graph.edges[u, v]}
                for u, v in self.graph.edges()
            ] if self._graph else [],
        }

    def save(self, path: str):
        """保存图数据到 JSON 文件"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("知识图谱已保存: %s", path)

    def load(self, path: str):
        """从 JSON 文件加载图数据"""
        if not os.path.exists(path):
            logger.warning("知识图谱文件不存在: %s", path)
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        import networkx as nx
        self._graph = nx.Graph()
        for node in data.get("nodes", []):
            nid = node.pop("id")
            self._graph.add_node(nid, **node)
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            self._graph.add_edge(src, tgt, **edge)

        # 重建 _articles 索引
        for nid, attrs in self._graph.nodes(data=True):
            if attrs.get("type") == "article":
                self._articles[nid] = {
                    "kb_id": nid,
                    "title": attrs.get("title", ""),
                    "category": attrs.get("category", ""),
                    "tags": attrs.get("tags", []),
                }

        logger.info(
            "知识图谱已加载: %s, %d 节点, %d 边",
            path, self._graph.number_of_nodes(), self._graph.number_of_edges(),
        )
        return True


# ═══════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════

_kg: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """获取全局唯一的 KnowledgeGraph 实例"""
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg


def build_knowledge_graph(articles: list[dict] | None = None) -> dict:
    """构建知识图谱（如果未构建）"""
    kg = get_knowledge_graph()
    if kg.is_built():
        return {
            "status": "already_built",
            "nodes": kg.graph.number_of_nodes(),
            "edges": kg.graph.number_of_edges(),
        }
    if articles is None:
        from app.data.loader import get_loader
        articles = get_loader().kb_articles
    return kg.build_from_articles(articles)
