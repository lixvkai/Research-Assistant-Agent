import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "10"))
MAX_REFLECTIONS = int(os.getenv("MAX_REFLECTIONS", "1"))  # 自我反思最多触发的修订轮数

# LLM 调用的健壮性配置
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60.0"))      # 单次请求超时（秒）
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))   # 瞬时错误自动重试次数

_ROOT = os.path.dirname(os.path.dirname(__file__))
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(_ROOT, "data", "chroma_db"),
)
PAPERS_DIR = os.getenv("PAPERS_DIR", os.path.join(_ROOT, "data", "papers"))
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(_ROOT, "data", "memory.db"))

# Gradio 部署：默认仅本机；对外暴露请配置 GRADIO_AUTH_USERNAME / GRADIO_AUTH_PASSWORD
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_AUTH_USERNAME = os.getenv("GRADIO_AUTH_USERNAME") or None
GRADIO_AUTH_PASSWORD = os.getenv("GRADIO_AUTH_PASSWORD") or None

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
