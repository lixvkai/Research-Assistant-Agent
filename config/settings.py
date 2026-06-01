import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

MAX_REACT_STEPS = 10
MAX_REFLECTIONS = 1  # 自我反思最多触发的修订轮数

# LLM 调用的健壮性配置
LLM_TIMEOUT = 60.0      # 单次请求超时（秒）
LLM_MAX_RETRIES = 3     # 瞬时错误（超时/限流/5xx/连接）自动重试次数（指数退避，由 SDK 实现）

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "papers")

MEMORY_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "memory.db")

# 外部 MCP（Model Context Protocol）Server 配置。
# 默认空 —— 不接入任何外部 MCP 工具，运行行为与之前一致。
# 每一项：{"name": 唯一名, "command": 启动命令, "args": [参数...], "env": {可选}, "category": 可选}
# 示例（接入官方文件系统 MCP server）：
#   MCP_SERVERS = [
#       {"name": "filesystem", "command": "npx",
#        "args": ["-y", "@modelcontextprotocol/server-filesystem", PAPERS_DIR]},
#   ]
MCP_SERVERS: list[dict] = []
