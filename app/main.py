"""
FastAPI 应用启动入口

这个文件的作用：
  = 创建并配置 FastAPI 应用
  = 管理程序的"生命周期"（启动时做什么、关闭时做什么）
  = 注册 API 路由（URL 对应的处理函数）
  = 定义请求和响应的数据格式（Pydantic 模型）

FastAPI 是什么？
  - 一个 Python 的 Web 框架，用来写 API 接口
  - 特点是：快（性能高）、自动生成接口文档、类型安全
  - 我们用它来对外提供 HTTP 接口，让网页/App能调用我们的系统

程序的生命周期（lifespan）：
  启动 → 初始化 LLM 客户端 → 运行中处理请求 → 关闭 → 清理资源
  这就像开餐厅：开门 → 准备好食材 → 客人点菜 → 打烊 → 收拾厨房

小白问答：
  Q: @app.get("/health") 这种写法是什么意思？
  A: 这叫"装饰器"。它在函数上面，给函数增加额外功能。
     @app.get("/health") 的意思是：
     "当用户通过 HTTP GET 方法访问 /health 这个路径时，
     执行下面的 health_check() 函数。"
     FastAPI 把函数和 URL 绑定在一起，这叫"路由"（Routing）。

  Q: lifespan 和之前的 app.state 是什么关系？
  A: lifespan 是"生命周期管理器"，里面 yield 之前的代码是"启动时执行"，
     yield 之后的代码是"关闭时执行"。
     在启动时我们把 LLM 客户端放进 app.state（"应用全局储物柜"），
     这样路由函数就能从 app.state 取出 LLM 来用了。

  Q: Pydantic BaseModel 在这里干啥？
  A: 它定义了 API 的数据格式，比如：
     - ChatRequest 定义了"客户端发来的请求应该包含什么字段"
     - ChatResponse 定义了"服务器返回的数据包含什么字段"
     好处是 FastAPI 会自动校验请求数据、自动生成接口文档。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager  # 异步上下文管理器，用于管理生命周期

from fastapi import FastAPI, HTTPException
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel  # Pydantic 的数据模型基类

from app.config import settings  # 我们写的配置管理
from app.dependencies import get_llm  # LLM 客户端工厂

# logging：Python 自带的日志系统
# 作用：在控制台输出带时间戳的运行信息，方便调试和排错
# 比如：logger.info("正在启动应用...")
logger = logging.getLogger(__name__)  # __name__ 自动取得当前模块名


# ═══════════════════════════════════════════════════════════════
#  应用生命周期管理
#
#  asynccontextmanager 把下面的函数变成一个"异步上下文管理器"。
#  简单理解：
#    - yield 之前的代码 = 启动时执行（初始化 LLM、连接数据库等）
#    - yield 本身 = 应用运行中（等待和处理请求）
#    - yield 之后的代码 = 关闭时执行（断开数据库、清理资源等）
#
#  这种模式确保资源一定会被正确创建和销毁，不会泄露。
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理函数

    FastAPI 会在启动时自动调用这个函数，
    yield 让应用开始处理请求，
    应用关闭时继续执行 yield 后面的代码。
    """
    # ── 启动阶段（应用启动时执行一次） ────────────────────────
    logger.info(
        "正在启动 %s (运行环境=%s, 调试模式=%s)",
        settings.APP_NAME,
        settings.APP_ENV,
        settings.APP_DEBUG,
    )

    # 尝试初始化 LLM 客户端
    # 如果没配置 API Key，会抛出 RuntimeError，但我们用 try 接住，
    # 记个警告日志，不阻塞应用启动（这样至少健康检查能返回结果）
    try:
        # get_llm() 是带缓存的，会返回全局唯一 LLM 实例
        llm = get_llm()
        # 把 LLM 存到 app.state 里，相当于"放进展柜"
        # 后续路由函数可以通过 request.app.state.llm 拿到它
        app.state.llm = llm
        logger.info(
            "LLM 客户端已初始化: 模型=%s, 地址=%s, 温度=%.1f, 最大Token=%d",
            settings.LLM_MODEL,
            settings.LLM_BASE_URL,
            settings.LLM_TEMPERATURE,
            settings.LLM_MAX_TOKENS,
        )
    except RuntimeError as e:
        # 这种错误通常是没配 API Key，不致命，只是 LLM 功能不能用
        logger.warning("LLM 初始化跳过: %s", e)
        app.state.llm = None  # 标记为未初始化

    # ── 初始化可观测性模块 ────────────────────────────────
    from app.observability import (
        get_tracing_handler, get_cost_tracker, get_alert_manager,
    )
    from app.observability.tracing import get_langfuse_client

    langfuse_client = get_langfuse_client()
    if langfuse_client is not None:
        logger.info(
            "LangFuse 追踪已启用: host=%s",
            settings.LANGFUSE_HOST,
        )
    else:
        logger.info("LangFuse 追踪未配置（密钥为空），使用 no-op 模式")

    app.state.cost_tracker = get_cost_tracker()
    app.state.alert_manager = get_alert_manager()
    logger.info("可观测性模块已初始化（成本追踪 + 告警）")

    # ── 初始化知识库向量索引 ────────────────────────────────
    from app.data.kb_vector import index_kb_articles
    kb_count = index_kb_articles()
    if kb_count > 0:
        logger.info("知识库向量索引已构建: %d 篇", kb_count)
    else:
        logger.info("知识库向量索引已跳过（已有数据或 Qdrant 不可用）")

    # yield：应用开始处理请求
    # 程序会停在这里，直到收到关闭信号才继续往下走
    yield

    # ── 关闭阶段（应用关闭时执行一次） ────────────────────────
    logger.info("正在关闭 %s", settings.APP_NAME)

    # 刷新 LangFuse 追踪数据（发送最后一批 pending spans）
    from app.observability import flush_traces
    try:
        flush_traces()
        logger.info("LangFuse 追踪数据已刷新")
    except Exception:
        pass

    # 清理 LLM 客户端（释放内存和连接）
    app.state.llm = None


# ═══════════════════════════════════════════════════════════════
#  创建 FastAPI 应用实例
#
#  这是整个程序的心脏，所有功能都挂在这个 app 上。
#  uvicorn（运行 FastAPI 的服务器）会启动这个 app。
#
#  启动命令（在终端运行）：
#    uvicorn app.main:app --reload
#    解释：uvicorn 是服务器，app.main 是文件名，app 是变量名
#    --reload 表示修改代码后自动重启（开发时很方便）
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.APP_NAME,           # 接口文档的标题
    description="基于 LangGraph 的多 Agent 客服分流系统",  # 接口文档的描述
    version="0.1.0",                   # 当前版本号
    lifespan=lifespan,                 # 注册生命周期管理器
)


# ═══════════════════════════════════════════════════════════════
#  数据模型（定义 API 的请求和回复格式）
#
#  为什么需要这些？
#  - 当客户端访问 /chat 接口时，需要告诉服务器要发什么数据
#  - ChatRequest 定义"用户的请求格式"
#  - ChatResponse 定义"服务器返回的格式"
#  - FastAPI 会用这些模型自动校验数据是否正确
#
#  比如：
#    客户端发 {"message": "你好", "temperature": 0.5}
#    FastAPI 会检查：
#      - message 字段存在并且是字符串 ✓
#      - temperature 字段可选，如果有的话必须是数值 ✓
#      如果客户端漏了 message 字段，服务器会返回错误提示
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """
    聊天请求的数据格式

    客户端调用 /chat 接口时必须（或可选）提供以下字段
    """
    message: str                     # 用户的消息内容（必填，字符串）
    temperature: float | None = None # 可选：覆盖默认的随机性
    max_tokens: int | None = None    # 可选：覆盖默认的最大Token数


class ChatResponse(BaseModel):
    """
    聊天响应的数据格式

    服务器会按这个格式返回给客户端
    """
    reply: str                       # AI 的回复内容
    model: str                       # 使用的模型名称
    token_usage: dict[str, int] = {} # Token 使用情况（用了多少输入/输出Token）


class HealthResponse(BaseModel):
    """
    健康检查响应的数据格式
    """
    status: str                      # "ok" 表示一切正常
    version: str                     # 当前版本号
    llm_ready: bool                  # LLM 客户端是否已就绪（True 表示可以聊天）
    config: dict[str, str] = {}      # 当前配置信息摘要


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════


def _extract_content_text(content) -> str:
    """
    从 LLM 返回的 content 中提取纯文本。

    DeepSeek 等模型返回 content 为 list 格式：
      [{'type': 'thinking', 'thinking': '...'}, {'type': 'text', 'text': '...'}]
    普通模型返回 str。此函数兼容两种格式。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        if parts:
            return "".join(parts)
        # 如果没有 text 类型的 block，返回字符串表示
        return str(content) if content else ""
    return str(content)


# ═══════════════════════════════════════════════════════════════
#  API 路由（URL → 处理函数）
#
#  什么是"路由"？
#    - 用户访问不同的 URL，服务器执行不同的函数
#    - /health → 健康检查函数
#    - /chat → 聊天函数
#    - 就像餐厅的菜单：点不同的菜，厨房做不同的菜
# ═══════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查端点

    用途：
      - 检测服务是否正常运行
      - 用于 Docker 的健康检查、负载均衡器的存活探测
      - 也用于在浏览器里快速确认服务启动了

    访问方式：
      GET /health

    返回示例：
      {
        "status": "ok",
        "version": "0.1.0",
        "llm_ready": true,
        "config": {
          "model": "deepseek-v4-flash",
          "base_url": "https://api.deepseek.com/v1",
          "temperature": "0.1",
          "max_tokens": str(settings.LLM_MAX_TOKENS),
          "env": "development"
        }
      }

    参数说明：
      response_model=HealthResponse：告诉 FastAPI 用这个格式来格式化返回值
      tags=["系统"]：在接口文档里把接口分组归类
    """
    # 从 app.state 里取出 LLM 客户端（启动时存进去的）
    llm: ChatLiteLLM | None = getattr(app.state, "llm", None)

    return HealthResponse(
        status="ok",
        version="0.1.0",
        llm_ready=llm is not None,  # True=LLM已就绪, False=没配Key
        config={
            "model": settings.LLM_MODEL,
            "base_url": settings.LLM_BASE_URL,
            "temperature": str(settings.LLM_TEMPERATURE),
            "max_tokens": str(settings.LLM_MAX_TOKENS),
            "env": settings.APP_ENV,
        },
    )


@app.post("/chat", response_model=ChatResponse, tags=["调试"])
async def chat(request: ChatRequest):
    """
    基础聊天端点（用于测试 LLM 连通性）

    这个接口直接调用 LLM，不做任何 Agent 处理。
    主要用于调试——验证 LLM 配置是否正确、连接是否正常。

    访问方式：
      POST /chat
      Body（JSON格式）: {"message": "你好"}

    参数说明：
      request: FastAPI 自动根据 ChatRequest 模型解析请求体
              （你不用自己写 JSON 解析代码，FastAPI 全自动干了）

    请求示例：
      curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "你好"}'

    返回示例：
      {
        "reply": "你好！有什么我可以帮助你的吗？",
        "model": "deepseek-v4-flash",
        "token_usage": {
          "input_tokens": 14,
          "output_tokens": 24,
          "total_tokens": 38
        }
      }
    """
    # 检查 LLM 是否已初始化
    llm: ChatLiteLLM | None = getattr(app.state, "llm", None)
    if llm is None:
        # HTTP 503 = Service Unavailable（服务暂不可用）
        # 客户端收到这个状态码就知道"现在用不了，稍后再试"
        raise HTTPException(status_code=503, detail="LLM 客户端未初始化。请检查 API Key 配置。")

    # 如果请求里传了 temperature 或 max_tokens，创建临时实例覆盖参数
    if request.temperature is not None or request.max_tokens is not None:
        from app.dependencies import get_llm_with_overrides
        llm = get_llm_with_overrides(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

    # ── 调用 LLM ──────────────────────────────────────────────
    # invoke 是 LangChain 里"调用模型"的标准方法
    # 参数是用户消息字符串，返回的是 AI 的回复（AIMessage 对象）
    response = llm.invoke(request.message)

    # ── 提取文本内容（兼容 DeepSeek 列表格式） ──────────────
    # DeepSeek 等模型返回 content 为 list（如 [{'type': 'text', 'text': '...'}]）
    # 需要从中提取纯文本；普通模型返回 str
    reply_text = _extract_content_text(response.content)

    # ── 提取 Token 使用量 ────────────────────────────────────
    # AIMessage 有一个 usage_metadata 属性，记录了 Token 用量
    # 但不是所有模型都返回这个数据，所以用 hasattr 先检查
    usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        # usage_metadata 是一个字典，类似：
        # {"input_tokens": 14, "output_tokens": 24, "total_tokens": 38}
        usage = {
            "input_tokens": response.usage_metadata.get("input_tokens", 0),
            "output_tokens": response.usage_metadata.get("output_tokens", 0),
            "total_tokens": response.usage_metadata.get("total_tokens", 0),
        }

    # ── 返回结果 ──────────────────────────────────────────────
    # FastAPI 会自动把 ChatResponse 对象转成 JSON
    return ChatResponse(
        reply=reply_text,
        model=settings.LLM_MODEL,
        token_usage=usage,
    )


# ═══════════════════════════════════════════════════════════════
#  后续阶段路由占位
#
#  Phase 2 以后会添加更多路由，这里先留下 TODO 标记。
#  使用 include_router 可以把路由分散到不同文件里管理（模块化）。
#  比如：
#    app/agents/router.py 里定义所有 Agent 相关的路由
#    app/main.py 里通过 app.include_router(agent_router) 注册
# ═══════════════════════════════════════════════════════════════

# Phase 3: 审批管理路由（REST API + Web UI）
from app.human_in_the_loop.approval_routes import router as approval_router
app.include_router(approval_router, prefix="/api/v1/approval", tags=["审批"])

# Phase 2: Agent 工作流 API + Chat UI
from app.api.agent_routes import router as agent_router
app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])

# Phase 3: 可观测性 API + 监控面板
from app.api.observability_routes import router as observability_router
app.include_router(observability_router, prefix="/api/v1/observability", tags=["可观测性"])
