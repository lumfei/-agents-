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
        from app.prompts.registry import get_prompt_registry
        return get_prompt_registry().get_supervisor_prompt()

    @staticmethod
    def worker_base_prompt(agent_name: str, responsibilities: str, tools_desc: str) -> str:
        from app.prompts.registry import get_prompt_registry
        return get_prompt_registry().get_worker_prompt(agent_name, responsibilities, tools_desc)


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
