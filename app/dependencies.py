"""
依赖注入模块

这个文件的作用：
  = 统一管理程序中各种"服务"的创建和获取
  = 避免到处 new 对象（new LLM()、new Redis() 散落在代码里）
  = 通过 FastAPI 的"依赖注入"机制，自动把服务传给需要的函数

什么是"依赖注入"（Dependency Injection）？
  FastAPI 依赖注入：路由函数通过 Depends() 声明所需依赖，FastAPI 自动调用
  对应的工厂函数（如 get_llm()）并注入。好处：集中管理、便于测试 Mock。
  get_llm() 使用 @lru_cache 实现单例模式，async generator 格式适配 FastAPI 异步框架。
"""

from __future__ import annotations

from functools import lru_cache  # 缓存装饰器：让函数"记住"之前的结果

from fastapi import Request  # FastAPI 的请求对象
from langchain_litellm import ChatLiteLLM  # LiteLLM × LangChain 统一客户端（支持 100+ 模型）
import httpx  # 用于创建绕过代理的 HTTP 客户端
from langchain_openai import OpenAIEmbeddings  # OpenAI 兼容的 Embedding 客户端

from app.config import settings  # 从 config.py 导入全局配置


# ═══════════════════════════════════════════════════════════════
#  LLM 客户端工厂
#
#  "工厂"（Factory）是设计模式的一种——不直接 new 对象，而是通过工厂函数创建。
#  好处：创建逻辑统一（比如需要检查 Key 是否为空、处理代理等），
#       如果以后要换模型提供商，只需要改这一个函数。
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_llm() -> ChatLiteLLM:
    """
    获取 LLM 客户端实例。

    为什么用 LiteLLM？
      传统做法：ChatOpenAI 只能调 OpenAI（和兼容的 DeepSeek），
      换模型要改代码。LiteLLM 统一了 100+ 模型的 API 格式，
      你只需要改 .env 文件里的 LLM_MODEL 就能切换模型：
        deepseek/deepseek-v4-flash  → DeepSeek
        openai/gpt-4o               → OpenAI
        anthropic/claude-sonnet-4   → Anthropic

    @lru_cache(maxsize=1) = 全局唯一实例，重复利用连接池。

    返回的 ChatLiteLLM 对象可以：
      - 直接对话：llm.invoke("你好")
      - 绑定工具：llm.bind_tools([...])
      - 结构化输出：llm.with_structured_output(Schema)
    """
    # ── 检查 API Key 是否配置 ──────────────────────────────────
    if not settings.LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY 未配置。请检查 .env 文件或环境变量。\n"
            "操作步骤：\n"
            "  1. 在项目根目录找到 .env 文件\n"
            "  2. 在对应模型官网获取你的 API Key\n"
            "  3. 填入 LLM_API_KEY=你的key"
        )

    # ── 自动处理代理问题 ──────────────────────────────────────
    # Windows VPN/代理软件会影响 Python 的 HTTPS 请求。
    # LiteLLM 通过 httpx 发请求，同样受影响。
    # 解决方案：把常见 LLM API 域名加入 NO_PROXY，让这些请求直连。
    if settings.llm_no_proxy:
        import os
        # 已有的 NO_PROXY 值
        no_proxy = os.environ.get("NO_PROXY", "")
        existing = set(no_proxy.split(",")) if no_proxy else set()

        # 从 LLM_BASE_URL 提取域名加入 NO_PROXY（如果设置了自定义地址）
        if settings.LLM_BASE_URL:
            host = settings.LLM_BASE_URL.split("://")[1].split("/")[0].strip()
            if host:
                existing.add(host)

        # 常见 LLM API 域名（覆盖 LiteLLM 可能路由到的目标）
        common_hosts = [
            "api.deepseek.com",     # DeepSeek
            "api.openai.com",       # OpenAI
            "api.anthropic.com",    # Anthropic
            "generativelanguage.googleapis.com",  # Google Gemini
        ]
        existing.update(common_hosts)

        os.environ["NO_PROXY"] = ",".join(filter(None, existing))

    # ── 创建并返回 ChatLiteLLM 实例 ────────────────────────────
    # **settings.llm_kwargs 是"解包"操作：
    #   相当于 ChatLiteLLM(model="deepseek/deepseek-v4-flash", api_key="sk-xxx", ...)
    return ChatLiteLLM(**settings.llm_kwargs)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """
    获取 Embedding 客户端实例（全局唯一，带缓存）。

    阿里云百炼 DashScope 提供 OpenAI 兼容的 Embedding API，
    使用 langchain-openai 的 OpenAIEmbeddings 直接调用。

    返回的 OpenAIEmbeddings 对象可以：
      - embed_query("你好") → list[float]  单个查询向量化
      - embed_documents(["文本1", "文本2"]) → list[list[float]]  批量向量化
    """
    if not settings.EMBEDDING_API_KEY:
        raise RuntimeError(
            "EMBEDDING_API_KEY 未配置。请检查 .env 文件或环境变量。\n"
            "阿里云百炼 API Key 获取地址: https://bailian.console.aliyun.com/"
        )

    return OpenAIEmbeddings(
        **settings.embedding_kwargs,
        check_embedding_ctx_length=False,
        http_client=httpx.Client(trust_env=False),  # 绕过 Windows 系统代理
    )


def get_llm_with_overrides(
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatLiteLLM:
    """
    创建带参数覆盖的 LLM 实例（不缓存）。

    和 get_llm() 的区别：
      - get_llm() 返回全局唯一的缓存实例，节省资源
      - get_llm_with_overrides() 每次调用都创建新实例，但可以临时调整参数

    使用场景：
      - 某个功能需要更低的 temperature（更确定）
      - 某个功能需要更长的回复（更大 max_tokens）

    参数说明：
      temperature: 可选，覆盖默认的随机性（0.0-2.0）
      max_tokens: 可选，覆盖默认的最大输出 Token 数

    返回：
      一个新的 ChatLiteLLM 实例（不会缓存）
    """
    kwargs = settings.llm_kwargs.copy()
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatLiteLLM(**kwargs)


# ═══════════════════════════════════════════════════════════════
#  FastAPI 依赖项（用于路由注入）
#
#  以下函数配合 FastAPI 的 Depends() 使用。
#  路由函数写：
#    @app.post("/chat")
#    async def chat(llm: ChatLiteLLM = Depends(get_llm_from_request)):
#        response = llm.invoke("你好")
#
#  FastAPI 会自动调用 get_llm_from_request()，把 LLM 传进来。
#  这样路由函数不需要关心 LLM 怎么创建的，只需要"使用"。
# ═══════════════════════════════════════════════════════════════

async def get_llm_from_request(request: Request) -> ChatLiteLLM:
    """
    FastAPI 依赖项：从请求中获取 LLM 客户端。

    工作原理：
      - app.state 是 FastAPI 应用的"全局储物柜"
      - 在 main.py 的 lifespan 启动阶段，我们把 LLM 客户端存进
        app.state.llm 里（app 是整个程序的 FastAPI 实例）
      - 这里再从储物柜里取出来
      - 为什么不用全局变量？因为 FastAPI 是异步框架，
        用 app.state 是官方推荐的方式，更规范

    参数：
      request: FastAPI 自动传入的当前请求对象

    返回：
      ChatLiteLLM 实例

    可能报错：
      RuntimeError: LLM 客户端未初始化（启动时出错了）
    """
    # getattr(request.app.state, "llm", None) 相当于：
    #   尝试从 app.state 获取 "llm" 这个属性，如果不存在返回 None
    llm: ChatLiteLLM | None = getattr(request.app.state, "llm", None)
    if llm is None:
        raise RuntimeError("LLM 客户端未初始化（应用尚未完成启动）")
    return llm


async def get_settings_from_request(request: Request):
    """
    FastAPI 依赖项：获取应用配置。

    为什么还要包一层？不直接用 from app.config import settings？
    - 直接导入也可以，但用 Depends 可以让依赖关系更清晰
    - 测试时可以 mock（模拟）这个依赖，注入测试配置
    """
    return settings
