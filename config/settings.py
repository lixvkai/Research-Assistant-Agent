import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

MAX_REACT_STEPS = 10
MAX_CONTEXT_LENGTH = 8000

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "papers")

MEMORY_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "memory.db")
