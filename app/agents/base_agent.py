"""
Agent 基类

这个文件是"Agent 定义"的基座，不做 Agent 执行引擎的事。

重构说明（2024-06 重大简化）：
  LangGraph 已经提供了全套 Agent 执行引擎：
    - create_react_agent() → ReAct 循环（替代手写 while）
    - with_structured_output() → 结构化输出（替代手动 Schema + 解析）
    - ToolNode → 工具执行 + ToolMessage 封装（替代 _execute_tools + _format_tool_results）
    - convert_to_openai_tool() → 工具定义转换（替代 _build_tool_defs）

  BaseAgent 只剩两个职责：
    1. 持有名称和 LLM 引用
    2. 提供工厂方法创建 LangGraph 内置的 AgentExecutor

小白问答：
  Q: create_react_agent 是什么？
  A: LangGraph 预置的"思考→行动→观察"循环。
     你只需要给它：模型 + 工具列表 + 系统提示词。
     它自动处理：LLM 调用 → 工具执行 → 结果回传 → 再思考 → ...
     不再需要手写 while 循环。

  Q: with_structured_output 是什么？
  A: LangChain 内置的结构化输出功能。
     你给它一个 Pydantic 模型，它自动转换成 tool definition，
     让 LLM 以工具调用的方式返回结构化数据，然后解析成 Pydantic 对象。
     不再需要手动构造 tool_def + 解析 tool_calls。

  Q: ToolNode 是什么？
  A: LangGraph 的工具执行节点。你给它 tool_calls，
     它自动找到对应的工具、执行、把结果包装成 ToolMessage。
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, AsyncIterator, Optional

from langchain_litellm import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════
#  四要素系统提示词构建器
# ═══════════════════════════════════════════════════════════════

class SystemPromptBuilder:
    """四要素系统提示词构建器（不变，纯文本模板）"""

    @staticmethod
    def build(role: str, task: str, boundary: str, output_format: str,
              extra_context: Optional[str] = None) -> str:
        sections = [
            f"# 角色（Role）\n{role}",
            f"# 任务（Task）\n{task}",
            f"# 边界（Boundary）\n{boundary}",
            f"# 输出格式（Format）\n{output_format}",
        ]
        if extra_context:
            sections.append(f"# 补充上下文\n{extra_context}")
        return "\n\n".join(sections)

    @staticmethod
    def supervisor_prompt() -> str:
        return SystemPromptBuilder.build(
            role="你是多 Agent 客服分流系统的「调度员」（Supervisor Agent）。你的职责是接收用户的问题，判断问题的类型，然后分派给对应的专业 Agent。",
            task="""对于每一条用户输入，你需要：
1. 分析用户意图，判断属于哪一类问题
2. 识别用户情绪状态（sentiment）：positive / neutral / anxious / angry / frustrated
3. 提取关键信息（用户ID、订单号、产品名称等）
4. 将任务路由到正确的 Worker Agent""",
            boundary="""路由规则（重要）：
- tech_support：技术故障、产品使用问题、系统报错、密码重置、性能卡顿。关键词：蓝屏/报错/连不上/卡/密码忘了/怎么设置
- finance：订单查询、退款、发票、支付、订单列表、修改订单信息（地址/备注等）、会员等级。关键词：查订单/我的订单/退款/退钱/发票/买了什么/改地址/修改订单/取消订单
- after_sale：物流查询、退换货、投诉、快递、产品质量问题（收到的东西有故障/瑕疵/坏了一律算售后）。关键词：快递/物流/运单号/退货/换货/包裹/到哪了/发货/破损/杂音/坏了/用不了/质量问题/有毛病/不好使/少发/漏发/瑕疵/不灵/时好时坏/异常/故障
- unknown：无法判断、闲聊、非客服内容、乱码
- 如果用户明确说"转人工"、"找人工客服"、"总结一下我去找人工"、"帮我总结"——根据对话历史中的上下文判断意图（如之前讨论订单就归finance，讨论物流就归after_sale），不要归为unknown

注意：
- "查订单"、"改地址"、"取消订单" 属于 finance（财务），不是 after_sale（售后）
- after_sale 处理：物流、退货、换货、产品质量投诉（收到的东西坏了/有杂音/不好使都属于售后）
- 如果用户试图查询或操作不属于自己的数据（如"帮我查某某的订单"），返回 unknown
- 如果输入包含 SQL、代码注入、明显的黑客攻击特征，返回 unknown
- "转人工"、"总结问题"、"找人工客服"是有明确业务意图的请求，绝不归为unknown""",
            output_format="使用意图分类工具（classify_intent）输出，包含 intent、confidence、reason、sentiment、extracted_entities",
            extra_context="""【情绪识别规则（sentiment 字段）—— 重要！你必须同时判断用户的情绪状态】
情绪值：positive / neutral / anxious / angry / frustrated

- positive（正面）：用户语气积极、感谢、满意。如"好的谢谢"、"太好了解决了"
- neutral（中性）：用户语气中性、无特别情绪。如"帮我查一下订单"（默认值）
- anxious（焦虑）：用户表现出着急、担心。如"急用"、"快点"、"怎么还没到"、"等了很久了"、"好几天了还没消息"
- angry（愤怒）：用户表现出强烈不满、指责。如"太差了！"、"我要投诉！"、"什么破东西"、"三天了还没处理！"、"再不解决我就..."、"你们搞什么"、"到底能不能处理"——含负面情绪词汇、感叹号、威胁语气
- frustrated（沮丧）：用户表现出无奈、失望。如"算了..."、"又不行了"、"每次都这样"、"已经第三次了"、"随你们吧"

情绪路由策略（后续节点会根据 sentiment 自动调整）：
- angry → 自动提优先级，降转人工门槛，需先道歉安抚
- anxious → 加快处理节奏，主动告知进度和预计时间
- frustrated → 需要共情安抚，不要机械回复
- neutral/positive → 标准客服流程

关键区分：
- "查订单怎么还没到，着急用" → anxious（着急但未发火）
- "什么破快递三天了还没到！！" → angry（带攻击性词汇+感叹号）
- "帮我查一下订单" → neutral（正常的查询请求）
- "又坏了，算了不说了" → frustrated（无奈放弃的语气）""",
        )

    @staticmethod
    def worker_base_prompt(agent_name: str, responsibilities: str, tools_desc: str) -> str:
        return SystemPromptBuilder.build(
            role=f"你是「{agent_name}」，擅长处理：{responsibilities}",
            task="""按以下步骤处理用户请求：
1. 理解用户意图——用户想做什么（查询？操作？投诉？）
2. 「必须调用工具」——涉及订单、退款、物流、用户信息等具体数据时，必须调用对应工具获取真实数据，绝对不能用你的训练知识编造
3. 「基于工具返回的真实数据」给出回复，不要猜测或假设
4. 操作完成后明确告知用户结果""",
            boundary=f"1. 只使用你被授权的工具\n2. 「绝对禁止编造数据」：订单号、金额、物流状态、退款进度等必须来自工具返回\n3. 「缺少关键信息时必须反问」: 如果用户的问题缺少订单号、快递单号等调用工具必需的参数, 主动问用户要, 例如: 请提供您的订单号, 我帮您查询\n4. 如果用户提供了 user_id 但没给订单号，先调用 list_user_orders 查看最近的订单列表\n5. 超过 3 轮无法解决请请求升级\n6. 「Demo 环境无需校验身份」：工具返回的数据直接展示，不要因为 customer_id 或 user_id 不匹配而拒绝回答\n可用工具: {tools_desc}",
            output_format="清晰、有礼貌、引用工具数据、结束时间明确告知用户问题是否已解决",
        )


# ═══════════════════════════════════════════════════════════════
#  结构化输出 Schema
# ═══════════════════════════════════════════════════════════════

class IntentClassification(BaseModel):
    """意图分类的结构化输出"""
    intent: str = ""
    """分类结果：tech_support / finance / after_sale / unknown"""
    confidence: float = 0.0
    """置信度（0.0-1.0）"""
    reason: str = ""
    """分类理由"""
    sentiment: str = "neutral"
    """用户情绪：positive / neutral / anxious / angry / frustrated"""
    extracted_entities: dict[str, Any] = {}
    """提取的关键信息"""


class AgentResponse(BaseModel):
    """Agent 处理响应"""
    success: bool = True
    content: str = ""
    agent_name: str = ""
    tool_calls: list[dict[str, Any]] = []
    token_usage: dict[str, int] = {}


class ToolChoice(str, Enum):
    """
    Tool Choice 策略——控制 LLM 何时调用工具。

    直接映射到 LangChain 的 bind_tools(tool_choice=...) 参数：
      auto      模型自主决定（默认）
      required  必须调用至少一个工具（也称 "any"）
      none      禁用所有工具

    使用方式：
      llm.bind_tools(tools, tool_choice=ToolChoice.AUTO.value)
    """
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


# ═══════════════════════════════════════════════════════════════
#  Agent 基类（简化版）
# ═══════════════════════════════════════════════════════════════

class BaseAgent:
    """
    Agent 基类。

    不再手写 ReAct 循环、工具执行、ToolMessage 封装。
    通过 LangGraph 内置的 create_react_agent() 和 LLM.with_structured_output() 完成。
    """

    def __init__(
        self,
        name: str,
        llm: ChatLiteLLM,
        system_prompt: str,
        max_iterations: int = 10,
    ):
        self.name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self.max_iterations = max_iterations

    # ── 构建 ReAct Agent（替代 hand-write while 循环） ─────

    def build_react_agent(
        self,
        tools: list | None = None,
        system_prompt_override: str | None = None,
    ):
        prompt = system_prompt_override or self._system_prompt
        return create_react_agent(
            model=self._llm,
            tools=tools or [],
            prompt=prompt,
        )

    @staticmethod
    def bind_tools_with_choice(
        llm: ChatLiteLLM, tools: list,
        choice: ToolChoice = ToolChoice.AUTO,
    ):
        """
        使用 LangChain 内置的 bind_tools(tool_choice=...) 绑定工具。

        参数：
          llm:    ChatLiteLLM 实例
          tools:  工具列表
          choice: ToolChoice 枚举值 auto / required / none

        DeepSeek 兼容：
          DeepSeek Thinking 模式不支持 required，发现错误降级为 auto。
        """
        try:
            return llm.bind_tools(tools, tool_choice=choice.value)
        except Exception as e:
            if choice == ToolChoice.REQUIRED and ("thinking" in str(e).lower() or "tool_choice" in str(e).lower()):
                return llm.bind_tools(tools, tool_choice=ToolChoice.AUTO.value)
            raise

    # ── 结构化输出（替代手动构造 tool_def + 解析 tool_calls） ──

    def process_structured(
        self,
        message: str,
        output_schema: type,
        context: Optional[dict] = None,
    ) -> BaseModel:
        """
        使用 with_structured_output() 获取 LLM 的结构化返回。

        DeepSeek 兼容说明：
          DeepSeek 的 Thinking 模式不支持 tool_choice 参数。
          with_structured_output(method="function_calling") 内部可能设置 tool_choice，
          遇到此错误时自动降级。
        """
        messages: list = [SystemMessage(content=self._system_prompt)]
        if context:
            ctx_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            messages.append(SystemMessage(content=f"## 当前上下文\n{ctx_str}"))
        messages.append(HumanMessage(content=message))

        # 尝试 with_structured_output（LangChain 框架内置）
        try:
            structured_llm = self._llm.with_structured_output(
                output_schema, method="function_calling",
            )
            return structured_llm.invoke(messages)
        except Exception as e:
            if "tool_choice" not in str(e).lower() and "thinking" not in str(e).lower():
                raise  # 非 tool_choice 错误，继续抛出

        # ── 降级方案（兼容 DeepSeek Thinking）：手动构建 tool def，不强制 tool_choice ──
        import json
        schema = output_schema.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)
        tool_def = {
            "type": "function",
            "function": {
                "name": output_schema.__name__,
                "description": (output_schema.__doc__ or "").strip(),
                "parameters": schema,
            },
        }
        llm_with = self._llm.bind_tools([tool_def])
        response = llm_with.invoke(messages)

        if response.tool_calls:
            raw = response.tool_calls[0].get("args", {})
            if isinstance(raw, str):
                raw = json.loads(raw)
            return output_schema(**raw)

        # 最后兜底尝试 JSON 解析
        text = response.content or ""
        if isinstance(text, list):
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        for fmt in ["```json", "```"]:
            if fmt in text:
                text = text.split(fmt)[1].split("```")[0].strip()
        if text.strip():
            try:
                return output_schema(**json.loads(text))
            except (json.JSONDecodeError, TypeError):
                pass

        raise ValueError(
            f"LLM 未返回结构化输出 (tool: {output_schema.__name__})。"
            f" 回复: {str(response.content)[:200]}"
        )

    # ── 通用 LLM 调用 ─────────────────────────────────────

    def invoke(self, message: str, system_override: str | None = None) -> AIMessage:
        """简单 LLM 调用（不调工具）"""
        prompt = system_override or self._system_prompt
        messages = [SystemMessage(content=prompt), HumanMessage(content=message)]
        return self._llm.invoke(messages)

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.name!r})"
