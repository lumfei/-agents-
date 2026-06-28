"""
配置管理模块

这个文件的作用：
  = 程序的所有配置都集中在这里管理（API密钥、模型名称、数据库地址等）
  = 配置来自 .env 文件（环境变量文件），不需要改代码就能改配置
  = Pydantic Settings 会自动从 .env 文件读取并校验配置是否合法

为什么要用 Pydantic Settings？
  - 传统做法：到处写 os.getenv("LLM_API_KEY")，散落在代码各个角落
  - Pydantic 做法：集中定义在 Settings 类中，谁要用就 from app.config import settings
  - 好处：配置一目了然，类型安全（比如端口写成了字符串会报错），自动校验

小白问答：
  Q: Field() 是什么？
  A: Field() 是 Pydantic 提供的"字段定义工具"。它给每个配置项加上：
     - default=...   → 默认值（如果 .env 没写就用这个）
     - description=... → 说明这个配置是干嘛的
     - ge=/le=...    → 数值的范围限制（比如Temperature必须在0-2之间）
     简单说，Field() 就是给配置项加上"规矩"和"说明"

  Q: @property 是什么？
  A: 把类的方法变成"像属性一样访问"。
     比如 settings.redis_dsn 看起来是变量，实际是函数计算出来的。
     好处是调用时不用加括号，写起来更自然。

  Q: 最后一行 settings = Settings() 是什么？
  A: 创建 Settings 类的"唯一实例"（单例模式）。
     程序启动时只创建一次，后面所有地方都导入这个实例。
     就像"宿舍楼只有一台饮水机，大家共用"。
"""

from __future__ import annotations  # 让类型注解支持更灵活的写法（Python 3.7+）

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
# BaseSettings：Pydantic 的配置基类，能自动从 .env 文件和环境变量读取配置
# SettingsConfigDict：配置 Settings 本身的行为（比如从哪个文件读）


class Settings(BaseSettings):
    """
    应用全局配置类

    这个类定义了整个程序会用到的所有配置项。
    每个配置项都对应 .env 文件里的一行，或者环境变量里的一个值。

    使用方式：
      from app.config import settings
      print(settings.LLM_MODEL)  # 直接访问，就像访问普通变量一样

    配置加载顺序（后面的覆盖前面的）：
      1. 类的默认值（写在 Field(default=...) 里的）
      2. .env 文件里的值
      3. 系统环境变量（比如在命令行里 export LLM_MODEL=xxx）
      越后面的优先级越高，所以可以通过临时设环境变量来覆盖 .env
    """

    # ── model_config：配置 Settings 类自身的行为 ──────────────────
    model_config = SettingsConfigDict(
        env_file=".env",          # 从项目根目录的 .env 文件读取配置
        env_file_encoding="utf-8",# .env 文件的编码格式（中文不乱码）
        case_sensitive=False,     # 环境变量名不区分大小写（LLM_API_KEY 和 llm_api_key 都行）
        extra="ignore",           # 如果 .env 文件里有本类没定义的变量，忽略掉（不报错）
    )

    # ── AI 配置 ──────────────────────────────────────────────
    LLM_API_KEY: str = Field(default="", description="LLM API 密钥（DeepSeek / OpenAI）")
    LLM_MODEL: str = Field(default="deepseek/deepseek-v4-flash", description="模型名，LiteLLM 格式")
    LLM_BASE_URL: str = Field(default="https://api.deepseek.com/v1", description="API 地址")
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0, description="随机性")
    LLM_MAX_TOKENS: int = Field(default=16384, ge=1, le=128_000, description="最大输出 Token")

    # ── Embedding 配置（向量化模型） ───────────────────────────
    EMBEDDING_API_KEY: str = Field(default="", description="Embedding API 密钥（阿里云百炼）")
    EMBEDDING_MODEL: str = Field(default="text-embedding-v4", description="Embedding 模型名")
    EMBEDDING_BASE_URL: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Embedding API 地址"
    )


    # ── Redis 配置（缓存数据库） ────────────────────────────────
    # Redis：把数据存在内存里，读写超快，用来存短期数据（会话、缓存）
    REDIS_HOST: str = Field(default="localhost", description="Redis 服务的地址。localhost 表示本机")
    REDIS_PORT: int = Field(default=6379, description="Redis 服务的端口号（类似于门牌号，默认 6379）")
    REDIS_PASSWORD: str = Field(default="", description="Redis 的密码，如果没设密码就留空")

    # ── Qdrant 配置（向量数据库） ───────────────────────────────
    # Qdrant：存"向量"的数据库，用来做语义搜索（找出"意思相近"的内容）
    # 设 QDRANT_HOST 为空 → 本地文件模式（无需 Docker）
    QDRANT_HOST: str = Field(default="", description="Qdrant 地址（空=本地文件模式）")
    QDRANT_PORT: int = Field(default=0, description="Qdrant 端口（0=本地文件模式）")
    QDRANT_API_KEY: str = Field(default="")

    @field_validator("QDRANT_PORT", mode="before")
    @classmethod
    def _empty_port_to_zero(cls, v):
        """空字符串 → 0，避免 Pydantic 解析报错"""
        if v == "" or v is None:
            return 0
        return v

    # ── PostgreSQL 配置（关系型数据库） ──────────────────────────
    # PostgreSQL：存结构化数据，比如用户信息、工单记录、操作日志等
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="agent_cs", description="数据库名称")
    POSTGRES_USER: str = Field(default="postgres", description="数据库用户名")
    POSTGRES_PASSWORD: str = Field(default="postgres", description="数据库密码")

    # ── LangFuse 配置（可观测性平台） ────────────────────────────
    # LangFuse：追踪 AI 程序的运行情况，看用了多少 Token、每个请求花了多久等
    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="LangFuse 公钥（用于数据追踪）")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="LangFuse 私钥")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com", description="LangFuse 服务地址")

    # ── 应用配置 ────────────────────────────────────────────────
    APP_NAME: str = Field(default="multi-agent-cs", description="应用名称（显示在日志和监控里）")
    APP_ENV: str = Field(default="development", description="运行环境：development（开发）/ production（生产）")
    APP_DEBUG: bool = Field(default=True, description="是否开启调试模式（True 时出错会显示详细信息）")
    LOG_LEVEL: str = Field(default="INFO", description="日志级别：DEBUG（最详细）/ INFO / WARNING / ERROR")

    # ── 安全配置 ────────────────────────────────────────────────
    SECRET_KEY: str = Field(default="", description="应用密钥，用于加密 Session 等敏感数据")
    API_RATE_LIMIT: int = Field(default=100, description="API 限流：每分钟最多允许 100 次请求")
    MAX_TOKENS_PER_SESSION: int = Field(
        default=100_000,
        description="单个会话最多消耗 10 万个 Token，防止有人恶意消耗你的余额"
    )

    # ── Prompt 版本管理 ──────────────────────────────────────────
    PROMPT_VERSIONS_FILE: str = Field(
        default="prompts/versions.yaml",
        description="Prompt 版本清单文件路径（相对于项目根目录）"
    )
    PROMPT_ACTIVE_VERSION_OVERRIDE: str = Field(
        default="",
        description="全局 Prompt 版本覆盖（如 'v2'）。留空则使用 versions.yaml 中各类型的独立版本。"
    )

    # ═══════════════════════════════════════════════════════════════
    #  便捷属性（@property 把方法"伪装"成变量）
    #
    #  为什么要写这些？
    #  有些配置需要"组合"使用。比如 ChatLiteLLM 需要把 model、api_key 等
    #  打包成一个字典传进去。直接写一个属性每次取就好了，不用到处写拼代码。
    # ═══════════════════════════════════════════════════════════════

    @property
    def llm_kwargs(self) -> dict:
        """
        返回 LLM 客户端需要的参数字典。

        这个属性把 LLM 相关的几个配置打包成一个字典，
        方便直接传给 ChatLiteLLM（LangChain × LiteLLM 的统一客户端）。

        LiteLLM 的关系：
          LiteLLM 是一个"统一 API 网关"，它把 OpenAI / Anthropic / DeepSeek /
          Google / Azure 等所有主流模型的 API 格式统一成一套接口。
          你只需要改 model 名字就能切换模型，不需要改任何代码。

        模型名格式：[提供商]/[模型名]
          例如：deepseek/deepseek-v4-flash、openai/gpt-4o、anthropic/claude-sonnet-4

        使用场景：
          llm = ChatLiteLLM(**settings.llm_kwargs)
          # ** 表示"解包"，把字典展开成 ChatLiteLLM(api_key=..., model=..., ...)
        """
        return {
            "model": self.LLM_MODEL,           # LiteLLM 模型名（含提供商前缀）
            "api_key": self.LLM_API_KEY,        # API 密钥
            "api_base": self.LLM_BASE_URL,      # API 地址
            "temperature": self.LLM_TEMPERATURE, # 随机性
            "max_tokens": self.LLM_MAX_TOKENS,   # 最大 Token 数
            "model_kwargs": {                    # 额外模型参数
                "thinking": {"type": "disabled"}, # 关闭 DeepSeek V4 thinking 模式，避免 tool_choice 冲突
            },
        }

    @property
    def llm_no_proxy(self) -> bool:
        """
        是否让 LLM API 不走代理。

        很多 Windows 用户开了 VPN / 翻墙软件，这些软件会设置"系统代理"。
        但 Python 的 httpx 库（负责发网络请求）通过代理访问 HTTPS 时可能会 SSL 握手失败。
        设为 True 后，程序会在环境变量里把 api.deepseek.com 加入 NO_PROXY 名单，
        让 DeepSeek 的请求直连，不走代理。不影响你浏览器使用 VPN。
        """
        return True

    @property
    def embedding_kwargs(self) -> dict:
        """
        返回 Embedding 客户端需要的参数字典。

        阿里云百炼 DashScope 兼容 OpenAI API 格式，
        所以可以直接使用 langchain-openai 的 OpenAIEmbeddings。
        """
        return {
            "model": self.EMBEDDING_MODEL,
            "api_key": self.EMBEDDING_API_KEY,
            "base_url": self.EMBEDDING_BASE_URL,
        }

    @property
    def redis_dsn(self) -> str:
        """
        Redis 连接字符串。

        DSN = Data Source Name（数据源名称），就是告诉程序"怎么连到 Redis"。
        格式：redis://:密码@地址:端口/数据库编号
        比如：redis://:mypass@localhost:6379/0
        """
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def is_development(self) -> bool:
        """当前是否在开发环境。方便其他地方判断要不要输出调试信息"""
        return self.APP_ENV == "development"


# ═══════════════════════════════════════════════════════════════
#  创建设置实例（全局唯一的配置对象）
#
#  settings 是一个"全局单例"。
#  其他文件想读配置时，只需要：
#    from app.config import settings
#    print(settings.LLM_MODEL)
#  不需要重新读取 .env 文件，不用重复创建对象。
# ═══════════════════════════════════════════════════════════════
settings = Settings()
