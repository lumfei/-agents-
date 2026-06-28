"""
上下文工程引擎 — 多轮对话上下文的分层组织与 Token 预算管理

这个文件是整个对话系统的"上下文指挥官"。
它决定了：每轮 LLM 调用时，prompt 里应该放什么、放多少、以什么顺序放。

核心问题：
  LLM 的上下文窗口是有限的（16384 token）。
  如果把所有东西都塞进去 → Token 超限、回复被截断、质量下降。
  如果塞得太少 → LLM 缺乏上下文、答非所问。

解决方案：五层上下文分层（按优先级从高到低）：

  [1] System（系统层）— 最高优先级
      Agent 的角色、行为边界、输出格式。
      每轮都完整保留，不截断。

  [2] Memory（记忆层）— 高质量
      用户的长期画像、偏好、历史决策。
      根据当前问题语义检索，只注入最相关的 top_k 条。

  [3] Knowledge（知识层）— 按需
      知识库检索到的 FAQ 或政策文档。
      仅当当前问题需要查询知识库时才注入。

  [4] Conversation（对话层）— 最占空间
      最近 N 轮的对话历史（滑动窗口）。
      这是最大的 Token 消耗源，需要优先压缩。

  [5] Tool Results（工具结果层）— 最低优先级
      最近一次工具调用的返回数据。
      如果预算不够，先截断这一层。

为什么是 5 层而不是 3 层？
  原来记忆管理器的 3 类（短期/长期/工作）是按"存储方式"分的。
  这里的 5 层是按"在 prompt 中的用途"分的——同样一份短期记忆，
  对话历史和工具调用结果在 prompt 中扮演不同的角色，
  应该有不同的截断策略。

Token 预算管理策略：
  1. 总预算 = min(模型 max_tokens * 0.7, 配置值)
     (留出 30% 给模型回复)
  2. 按优先级分配预算比例：
     System:      15%
     Memory:      15%
     Knowledge:   20%
     Conversation: 35% (最占空间)
     Tool Results: 15%
  3. 每层实际使用超过预算时，从下层截断
  4. 低优先级层可以借用高优先级层未用完的预算

使用方式：
  engine = ContextEngine()
  layers = engine.build_layers(system_prompt="...", ...)
  messages = engine.assemble_messages(layers)
  # messages 可以直接传给 LLM.invoke()

Token 估算：
  中文字符 ≈ 1 token
  英文字符 ≈ 0.3 token
  这里用简单估算：len(text) * 0.5（保守估计）
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
#  配置常量
# ═══════════════════════════════════════════════════════════════

# 默认上下文窗口预算（Token）
# 这是每次 LLM 调用时，所有上下文层加在一起的总预算上限
# 注：留出 15% 给 LLM 的回复
DEFAULT_TOTAL_BUDGET = 16384

# 回复预留比例：总窗口的 15% 留给 LLM 输出
# 回复通常 300-800 token，15% 对 16384 就是 ~2458 token，绰绰有余
RESPONSE_RESERVE_RATIO = 0.15

# 各层预算分配比例（按优先级）
# 加起来应该 = 1.0
# Conversation 拿最多（45%），因为是"多轮对话"的核心
LAYER_ALLOCATION: dict[str, float] = {
    "system": 0.15,        # 系统提示词（固定，不变）
    "memory": 0.10,        # 长期记忆（用户画像，本身不大）
    "knowledge": 0.20,     # 知识库结果（按需）
    "conversation": 0.45,  # 对话历史（多轮对话核心，拿最多）
    "tool_results": 0.10,  # 工具调用结果（最低优先级）
}

# 各层的最大硬性上限（Token），防止某层异常膨胀
LAYER_HARD_CAPS: dict[str, int] = {
    "system": 2048,
    "memory": 2048,
    "knowledge": 4096,
    "conversation": 8192,
    "tool_results": 2048,
}

# 估算 Token 的系数（中英文混合文本）
TOKEN_ESTIMATE_RATIO = 0.5


# ═══════════════════════════════════════════════════════════════
#  枚举
# ═══════════════════════════════════════════════════════════════

class ContextLayer(str, Enum):
    """上下文层级"""
    SYSTEM = "system"
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    CONVERSATION = "conversation"
    TOOL_RESULTS = "tool_results"


class TruncationStrategy(str, Enum):
    """截断策略"""
    TRUNCATE_HEAD = "truncate_head"  # 截掉头部（对话历史用——去掉最早的轮次）
    TRUNCATE_TAIL = "truncate_tail"  # 截掉尾部（工具结果用——去掉最旧的结果）
    SUMMARIZE = "summarize"          # 压缩为摘要（对话历史用）
    DROP = "drop"                    # 全部丢弃（最低优先级层）


# ═══════════════════════════════════════════════════════════════
#  分层数据结构
# ═══════════════════════════════════════════════════════════════

class LayerContent:
    """
    单层上下文的内容和元数据。

    每个层包含：
      - name: 层名
      - priority: 优先级（1=最高, 5=最低）
      - content: 该层的文本内容（最终要拼到 prompt 里的）
      - tokens: 该层占用的 Token 数
      - budget: 该层的预算上限
      - truncated: 是否被截断过
    """

    def __init__(self, name: str, priority: int):
        self.name = name
        self.priority = priority
        self.content: str = ""
        self.tokens: int = 0
        self.budget: int = 0
        self.truncated: bool = False
        self.original_size: int = 0

    def set_content(self, content: str):
        """设置内容并计算 Token 数"""
        self.content = content
        self.original_size = len(content)
        self.tokens = estimate_tokens(content)

    def exceeds_budget(self) -> bool:
        """是否超出预算"""
        return self.tokens > self.budget

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "priority": self.priority,
            "tokens": self.tokens,
            "budget": self.budget,
            "truncated": self.truncated,
        }

    def __repr__(self) -> str:
        status = "TRUNCATED" if self.truncated else "OK"
        return (
            f"Layer({self.name}, tokens={self.tokens}/{self.budget}, "
            f"len={len(self.content)}, {status})"
        )


class ContextAssembly:
    """
    完整的上下文组装结果。

    包含 5 个层的内容 + 组装后的 messages 列表 + Token 统计。
    """

    def __init__(self):
        self.layers: dict[str, LayerContent] = {}
        self.messages: list[dict] = []
        self.total_tokens: int = 0
        self.total_budget: int = 0

    def add_layer(self, layer: LayerContent):
        """添加一个层"""
        self.layers[layer.name] = layer

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_budget": self.total_budget,
            "layers": {
                name: layer.to_dict()
                for name, layer in self.layers.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"ContextAssembly(tokens={self.total_tokens}/"
            f"{self.total_budget}, "
            f"layers={list(self.layers.keys())}, "
            f"messages={len(self.messages)})"
        )


# ═══════════════════════════════════════════════════════════════
#  Token 估算工具
# ═══════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """
    估算文本占用的 Token 数。

    这不是精确计算（精确计算需要 tokenizer），
    但足够用来做预算分配决策。
    保守估算：每个中文字 ≈ 1 token，每个英文词 ≈ 0.5 token
    混合取 0.5 系数。
    """
    if not text:
        return 0
    return max(1, int(len(text) * TOKEN_ESTIMATE_RATIO))


def truncate_to_budget(text: str, budget: int, strategy: TruncationStrategy) -> str:
    """
    按预算截断文本。

    参数：
      text:     原始文本
      budget:   目标 Token 预算
      strategy: 截断策略

    返回：
      截断后的文本（长度不一定精确等于 budget，但不超）
    """
    if estimate_tokens(text) <= budget:
        return text

    # 反推：budget token 对应的字符数
    target_chars = int(budget / TOKEN_ESTIMATE_RATIO)

    if strategy == TruncationStrategy.TRUNCATE_HEAD:
        # 保留尾部（对话历史用——保留最近的轮次）
        return text[-target_chars:] if len(text) > target_chars else text

    elif strategy == TruncationStrategy.TRUNCATE_TAIL:
        # 保留头部（工具结果用）
        return text[:target_chars] if len(text) > target_chars else text

    elif strategy == TruncationStrategy.DROP:
        # 预算不足，全部丢弃
        return ""

    elif strategy == TruncationStrategy.SUMMARIZE:
        # 摘要策略：取头部 + 尾部，中间省略
        if len(text) <= target_chars:
            return text
        half = target_chars // 2
        return (
            f"{text[:half]}"
            f"\n...【中间省略 {len(text) - target_chars} 字符】...\n"
            f"{text[-half:]}"
        )

    return text


# ═══════════════════════════════════════════════════════════════
#  上下文工程引擎
# ═══════════════════════════════════════════════════════════════

class ContextEngine:
    """
    上下文工程引擎 — 五层上下文分层 + Token 预算管理。

    这是连接"记忆系统"和"LLM 调用"之间的桥梁：
      记忆系统    → 存储和管理各种数据
      ↓
      ContextEngine → 从记忆系统中取出数据，分层、分配预算、截断
      ↓
      LLM 调用   → 收到组装好的 messages，直接 invoke

    核心方法：
      build_layers():    从各数据源收集内容，分配到 5 个层
      assemble_messages(): 把 5 个层组装成 LLM 可用的 messages 列表
    """

    def __init__(
        self,
        total_budget: int = DEFAULT_TOTAL_BUDGET,
        layer_allocation: Optional[dict[str, float]] = None,
    ):
        """
        初始化上下文引擎。

        参数：
          total_budget:    总 Token 预算（所有层之和不超过此值）
          layer_allocation: 各层分配比例覆盖（可选）
        """
        self.total_budget = total_budget
        self.layer_allocation = layer_allocation or LAYER_ALLOCATION

    # ══════════════════════════════════════════════════════════
    #  Token 预算计算
    # ══════════════════════════════════════════════════════════

    def _calculate_budgets(self) -> dict[str, int]:
        """
        按分配比例计算各层的 Token 预算。

        计算方式：
          1. 从总预算中扣减留给回复的部分（RESPONSE_RESERVE_RATIO）
          2. 剩余预算按 LAYER_ALLOCATION 比例分配
          3. 每层不超过 LAYER_HARD_CAPS

        返回：
          {层名: 预算Token数}
        """
        available = int(self.total_budget * (1 - RESPONSE_RESERVE_RATIO))
        budgets: dict[str, int] = {}

        for layer, ratio in self.layer_allocation.items():
            allocated = int(available * ratio)
            # 不超过硬性上限
            capped = min(allocated, LAYER_HARD_CAPS.get(layer, 9999))
            budgets[layer] = capped

        return budgets

    # ══════════════════════════════════════════════════════════
    #  各层构建
    # ══════════════════════════════════════════════════════════

    def build_system_layer(
        self, system_prompt: str, extra_context: str = ""
    ) -> LayerContent:
        """
        构建系统层（最高优先级）。

        内容：
          - Agent 的角色定义、行为边界、输出格式
          - 可选：由记忆管理器提供的额外上下文（用户画像、任务状态）

        截断策略：
          如果超出预算，用 TRUNCATE_TAIL 截断额外上下文，
          但系统提示词本身不会截断。
        """
        layer = LayerContent(name="system", priority=1)

        content_parts = [system_prompt]
        if extra_context:
            content_parts.append("")
            content_parts.append(extra_context)

        layer.set_content("\n".join(content_parts))
        return layer

    def build_memory_layer(
        self,
        long_term_context: str = "",
        include_header: bool = True,
    ) -> LayerContent:
        """
        构建记忆层。

        内容：
          - 用户的长期画像、偏好（来自 LongTermMemory）
          - 已经由记忆管理器做了语义检索，只取最相关的 top_k

        截断策略：
          如果超出预算，逐条丢弃得分最低的记忆。
        """
        layer = LayerContent(name="memory", priority=2)

        if not long_term_context:
            layer.set_content("")
            return layer

        # 如果已经用 header 格式，直接使用
        if include_header and not long_term_context.startswith("【"):
            content = f"【用户画像】\n{long_term_context}"
        else:
            content = long_term_context

        layer.set_content(content)
        return layer

    def build_knowledge_layer(
        self, knowledge_text: str = ""
    ) -> LayerContent:
        """
        构建知识层（按需注入）。

        内容：
          - 知识库检索到的 FAQ、政策文档等
          - 仅当当前问题需要时才注入

        注意：
          本系统知识库是辅助功能（客服核心操作为结构化数据）。
          如果 knowledge_text 为空，这一层不占用任何预算。
        """
        layer = LayerContent(name="knowledge", priority=3)

        if not knowledge_text:
            layer.set_content("")
            return layer

        content = f"【参考知识】\n{knowledge_text}"
        layer.set_content(content)
        return layer

    def build_conversation_layer(
        self,
        conversation_messages: list[dict],
        max_rounds: int = 10,
    ) -> LayerContent:
        """
        构建对话层（最占 Token 的一层）。

        内容：
          - 最近 N 轮的对话历史（来自 ShortTermMemory）
          - 以 [用户消息, AI回复, 用户消息, AI回复...] 格式排列

        截断策略（按优先级）：
          1. 先限制轮次（滑动窗口已经做了）
          2. 如果还超出预算 → 只保留最近一半的轮次
          3. 如果还不够 → 每轮消息截取前一半内容
          4. 如果还不够 → 全部丢弃（对话层降级为摘要）

        对话层是唯一可以使用 TRUNCATE_HEAD 策略的层，
        因为"最近的消息"比"最早的消息"更重要。
        """
        layer = LayerContent(name="conversation", priority=4)

        if not conversation_messages:
            layer.set_content("")
            return layer

        # 格式化为连续文本（用于 Token 估算）
        lines = []
        for msg in conversation_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        text = "\n".join(lines)
        layer.set_content(text)
        return layer

    def build_tool_results_layer(
        self,
        tool_results_text: str = "",
        recent_tool_calls: Optional[list[dict]] = None,
    ) -> LayerContent:
        """
        构建工具结果层（最低优先级）。

        内容：
          - 本轮或上一轮的工具调用返回数据
          - 如果预算不够，优先截断这一层

        截断策略：
          如果超出预算 → 只保留工具名和关键摘要，丢弃详细返回。
          如果还不够 → 全部丢弃（LLM 可以问"查到了什么"来重新请求）。
        """
        layer = LayerContent(name="tool_results", priority=5)

        parts = []
        if tool_results_text:
            parts.append(tool_results_text)
        if recent_tool_calls:
            for tc in recent_tool_calls:
                name = tc.get("name", "?")
                args = tc.get("args", {})
                result = tc.get("result", "")
                parts.append(f"工具 [{name}] 调用: {json.dumps(args, ensure_ascii=False)}")
                if result:
                    result_str = str(result)[:200]
                    parts.append(f"返回: {result_str}")

        if not parts:
            layer.set_content("")
            return layer

        content = f"【工具调用结果】\n" + "\n".join(parts)
        layer.set_content(content)
        return layer

    # ══════════════════════════════════════════════════════════
    #  预算执行与截断
    # ══════════════════════════════════════════════════════════

    def _apply_budgets(self, layers: list[LayerContent]) -> list[LayerContent]:
        """
        对各层执行预算截断。

        执行策略：
          1. 按优先级从高到低处理
          2. 高优先级层未用完的预算可以留给低优先级层
          3. 低优先级层超出预算时逐级截断

        参数：
          layers: 已填入内容的层列表（未截断）

        返回：
          截断后的层列表
        """
        # 按优先级排序（1=最高）
        layers.sort(key=lambda l: l.priority)

        budgets = self._calculate_budgets()
        remaining = sum(budgets.values())

        for layer in layers:
            # 分配该层的预算
            layer.budget = budgets.get(layer.name, 0)

            # 如果该层没有内容，预算留给后面的层
            if not layer.content:
                continue

            # 检查是否超出预算
            if layer.tokens > layer.budget:
                layer.truncated = True

                # 根据层级选择截断策略
                strategy_map = {
                    "system": TruncationStrategy.TRUNCATE_TAIL,
                    "memory": TruncationStrategy.TRUNCATE_TAIL,
                    "knowledge": TruncationStrategy.TRUNCATE_TAIL,
                    "conversation": TruncationStrategy.SUMMARIZE,
                    "tool_results": TruncationStrategy.DROP,
                }
                strategy = strategy_map.get(layer.name, TruncationStrategy.TRUNCATE_TAIL)

                truncated = truncate_to_budget(layer.content, layer.budget, strategy)
                layer.set_content(truncated)
                if not layer.content:
                    # 如果全部丢弃，告诉调用方
                    pass

        return layers

    # ══════════════════════════════════════════════════════════
    #  核心 API
    # ══════════════════════════════════════════════════════════

    def build_layers(
        self,
        system_prompt: str = "",
        extra_context: str = "",
        long_term_context: str = "",
        knowledge_text: str = "",
        conversation_messages: Optional[list[dict]] = None,
        tool_results_text: str = "",
        recent_tool_calls: Optional[list[dict]] = None,
    ) -> ContextAssembly:
        """
        构建完整的五层上下文。

        这是核心方法——从所有数据源收集内容，
        分配到 5 个层，执行预算截断，返回组装结果。

        参数：
          system_prompt:        系统提示词（Agent 角色、边界）
          extra_context:        额外上下文（用户画像文字、工作记忆等）
          long_term_context:    长期记忆检索结果（已由记忆管理器处理）
          knowledge_text:       知识库检索结果
          conversation_messages: 对话历史（最近的 messages）
          tool_results_text:    工具结果文本
          recent_tool_calls:    最近调用的工具列表

        返回：
          ContextAssembly 对象，包含各层内容和元数据
        """
        assembly = ContextAssembly()

        # 构建各层
        layers = [
            self.build_system_layer(system_prompt, extra_context),
            self.build_memory_layer(long_term_context),
            self.build_knowledge_layer(knowledge_text),
            self.build_conversation_layer(conversation_messages or []),
            self.build_tool_results_layer(tool_results_text, recent_tool_calls),
        ]

        # 执行预算截断
        layers = self._apply_budgets(layers)

        for layer in layers:
            assembly.add_layer(layer)

        # 统计
        assembly.total_tokens = sum(l.tokens for l in layers)
        assembly.total_budget = sum(l.budget for l in layers)

        return assembly

    def assemble_messages(
        self,
        assembly: ContextAssembly,
        system_role: str = "system",
        user_role: str = "user",
        assistant_role: str = "assistant",
    ) -> list[dict]:
        """
        把组装好的上下文转成 LLM 可用的 messages 列表。

        Lost in the Middle 优化：
          LLM 对 context 开头和结尾的信息关注度最高。
          安排策略：
            开头（高注意力）→ System Prompt + Memory + Knowledge
            中间（低注意力）→ 较早的对话历史
            结尾（高注意力）→ 最近的对话 + Tool Results

        每层内容用 XML 标签包裹，让 LLM 能快速定位信息类型：
          <system_prompt>  <memory>  <knowledge>
          <conversation_history>  <tool_results>
        """
        messages: list[dict] = []

        # ── 1. 构建 System Message（开头——高注意力区域） ────
        system_parts = []

        sys_layer = assembly.layers.get("system")
        if sys_layer and sys_layer.content:
            system_parts.append(
                f"<system_prompt>\n{sys_layer.content}\n</system_prompt>"
            )

        mem_layer = assembly.layers.get("memory")
        if mem_layer and mem_layer.content:
            system_parts.append(
                f"<memory>\n{mem_layer.content}\n</memory>"
            )

        know_layer = assembly.layers.get("knowledge")
        if know_layer and know_layer.content:
            system_parts.append(
                f"<knowledge>\n{know_layer.content}\n</knowledge>"
            )

        if system_parts:
            messages.append({
                "role": system_role,
                "content": "\n\n".join(system_parts),
            })

        # ── 2. 展开对话历史（中间——低注意力区域） ────────────
        conv_layer = assembly.layers.get("conversation")
        if conv_layer and conv_layer.content:
            lines = conv_layer.content.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                for role in [user_role, assistant_role, system_role, "tool"]:
                    prefix = f"{role}: "
                    if line.startswith(prefix):
                        messages.append({
                            "role": role,
                            "content": line[len(prefix):],
                        })
                        break

        # ── 3. 工具结果（结尾——高注意力区域） ─────────────────
        tool_layer = assembly.layers.get("tool_results")
        if tool_layer and tool_layer.content:
            messages.append({
                "role": system_role,
                "content": (
                    f"<tool_results>\n{tool_layer.content}\n</tool_results>"
                ),
            })

        assembly.messages = messages
        return messages

    def build_and_assemble(
        self,
        system_prompt: str = "",
        extra_context: str = "",
        long_term_context: str = "",
        knowledge_text: str = "",
        conversation_messages: Optional[list[dict]] = None,
        tool_results_text: str = "",
        recent_tool_calls: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        一键完成：构建各层 → 预算截断 → 组装 messages。

        这是最常用的 API —— 所有参数传给 build_layers，
        拿到 assembly 后自动调 assemble_messages。

        返回：
          可直接传给 LLM.invoke() 的 messages 列表
        """
        assembly = self.build_layers(
            system_prompt=system_prompt,
            extra_context=extra_context,
            long_term_context=long_term_context,
            knowledge_text=knowledge_text,
            conversation_messages=conversation_messages,
            tool_results_text=tool_results_text,
            recent_tool_calls=recent_tool_calls,
        )
        return self.assemble_messages(assembly)

    # ══════════════════════════════════════════════════════════
    #  预算查询工具
    # ══════════════════════════════════════════════════════════

    def estimate_message_tokens(self, messages: list[dict]) -> int:
        """估算 messages 列表的总 Token 数"""
        total = 0
        for msg in messages:
            total += estimate_tokens(msg.get("content", ""))
            # 每条消息的 role 也占少量 token
            total += 2
        return total

    def budget_summary(self, total_model_window: int = 8192) -> str:
        """打印预算分配摘要"""
        lines = [f"上下文预算分配（模型窗口: {total_model_window}）"]
        lines.append(f"总预算: {self.total_budget} (预留 {int(RESPONSE_RESERVE_RATIO * 100)}% 给回复)")

        budgets = self._calculate_budgets()
        for layer, budget in budgets.items():
            pct = self.layer_allocation.get(layer, 0) * 100
            lines.append(f"  {layer:15s}: {budget:5d} token ({pct:.0f}%)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ContextEngine(budget={self.total_budget})"
