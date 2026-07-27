import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "10"))
MAX_REFLECTIONS = int(os.getenv("MAX_REFLECTIONS", "1"))  # 自我反思最多触发的修订轮数
# 每次反思修订额外追加的推理步数配额 —— 让反思预算与工具循环预算互不侵占
REFLECTION_STEP_BONUS = int(os.getenv("REFLECTION_STEP_BONUS", "2"))

# LLM 调用的健壮性配置
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60.0"))      # 单次请求超时（秒）
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))   # 瞬时错误自动重试次数

# ── 全局运行预算（跨嵌套 Agent 生效）────────────────────────────
# 一次用户请求内允许的 LLM 调用总数与总时限，防止 多Agent×ReAct 嵌套导致成本失控。
MAX_LLM_CALLS_PER_RUN = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "60"))
RUN_DEADLINE_SECONDS = float(os.getenv("RUN_DEADLINE_SECONDS", "600"))

# ── 上下文窗口 ────────────────────────────────────────────────
# Agent 保留的最近消息条数；溢出部分交给短期记忆压缩成摘要。
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "24"))

# ── 并行度 ────────────────────────────────────────────────────
TOOL_MAX_WORKERS = int(os.getenv("TOOL_MAX_WORKERS", "4"))       # 单轮多 tool_call 并行度
ORCHESTRATOR_MAX_WORKERS = int(os.getenv("ORCHESTRATOR_MAX_WORKERS", "3"))  # 同层子任务并行度

_ROOT = os.path.dirname(os.path.dirname(__file__))
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(_ROOT, "data", "chroma_db"),
)
PAPERS_DIR = os.getenv("PAPERS_DIR", os.path.join(_ROOT, "data", "papers"))
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(_ROOT, "data", "memory.db"))

# ── 记忆系统 ──────────────────────────────────────────────────
# 长期记忆语义检索：开启后复用 RAG 的嵌入模型 + ChromaDB 做向量召回，
# 关闭（默认）则使用内置的分词打分检索（无额外依赖，离线可用）。
MEMORY_SEMANTIC_SEARCH = _env_bool("MEMORY_SEMANTIC_SEARCH", False)
MEMORY_DEDUP_THRESHOLD = float(os.getenv("MEMORY_DEDUP_THRESHOLD", "0.92"))

# Gradio 部署：默认仅本机；对外暴露请配置 GRADIO_AUTH_USERNAME / GRADIO_AUTH_PASSWORD
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_AUTH_USERNAME = os.getenv("GRADIO_AUTH_USERNAME") or None
GRADIO_AUTH_PASSWORD = os.getenv("GRADIO_AUTH_PASSWORD") or None

# ── FastAPI 服务 ──────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
# 启动时预加载 Agent 与工具，避免首个请求等待数秒才吐出第一个事件
API_WARMUP = _env_bool("API_WARMUP", True)
# 允许跨域的前端来源，逗号分隔；为空则不启用 CORS 中间件
API_CORS_ORIGINS = [
    o.strip() for o in os.getenv("API_CORS_ORIGINS", "").split(",") if o.strip()
]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 外部 MCP（Model Context Protocol）Server 配置。
# 默认空 —— 不接入任何外部 MCP 工具，运行行为与之前一致。
# 每一项：{"name": 唯一名, "command": 启动命令, "args": [参数...], "env": {可选}, "category": 可选}
# 示例（接入官方文件系统 MCP server）：
#   MCP_SERVERS = [
#       {"name": "filesystem", "command": "npx",
#        "args": ["-y", "@modelcontextprotocol/server-filesystem", PAPERS_DIR]},
#   ]
MCP_SERVERS: list[dict] = []
