# 科研助手 Agent

> 基于 **ReAct + RAG + Multi-Agent** 的端到端科研助手，集成论文检索、知识库问答、多专家协作和长期记忆，提供流式可视化的推理过程。

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange?logo=gradio)](https://gradio.app/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-1d4ed8)](https://platform.deepseek.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-10b981)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

---

## ✨ 项目亮点

- 🧠 **自实现 ReAct Agent** — Thought → Action → Observation 循环，所有推理步骤通过 generator 流式吐出，UI 端实时渲染
- 🔌 **MCP 风格工具协议层** — 工具自动从模块发现并注册，新增工具零侵入，对 Agent 透明
- 📚 **进阶 RAG 流水线** — 查询改写 → 多召回 → Cross‑Encoder（`BAAI/bge-reranker-base`）重排序 + 基于句子嵌入相似度的语义分块
- 🤝 **Multi‑Agent 协作** — Planner LLM 拆分子任务并构建依赖图，4 个专家 Agent（文献 / 数据 / 写作 / 审查）按拓扑顺序执行后由总编 Agent 融合结果
- 🧩 **三层记忆系统** — 短期摘要压缩 + 长期 SQLite 持久化（用户偏好 / 研究发现）+ 多会话 ChatHistory
- 🎯 **Skills 系统** — 把常见科研工作流（文献综述、论文精读、研究方案设计…）固化为可检索、可调用的可复用模板
- 🎨 **流式可视化 UI** — Gradio + 自定义 CSS，支持知识库管理、历史会话切换、对话导出

---

## 🖼️ 界面预览

> 把截图放到 `docs/` 下，然后取消下面注释即可。
>
> ```markdown
> ![Main UI](docs/screenshot-main.png)
> ![ReAct Trace](docs/screenshot-trace.png)
> ```

界面三栏布局：

- **左侧**：品牌栏 / 新建对话 / 历史会话列表（支持加载、删除）
- **中间**：聊天区，流式展示「步骤 → 思考 → 工具调用 → 返回结果」全过程
- **右侧**：知识库管理 / 工具箱（按分类折叠展开）/ 技能列表

---

## 🏗️ 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Gradio Web UI (app.py)                    │
│        流式推理过程展示 · 知识库管理 · 历史会话 · 文件上传       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │       ReAct Agent (核心)     │
              │   Thought → Action → Observe │
              │        (run_iter 生成器)     │
              └──────┬─────────────────┬─────┘
                     │                 │
        ┌────────────▼─────┐    ┌──────▼─────────┐
        │  MCP Tool Server │    │ Memory Manager │
        │  (工具注册/绑定) │    │ (短期+长期记忆)│
        └────────┬─────────┘    └──────┬─────────┘
                 │                     │
       ┌─────────┼─────────┬───────────┼──────────┐
       ▼         ▼         ▼           ▼          ▼
   ┌────────┐ ┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │ Arxiv  │ │ RAG  │ │ Multi- │ │ Skills │ │ Memory │
   │  Tool  │ │Engine│ │ Agent  │ │Manager │ │ SQLite │
   └────────┘ └──┬───┘ └───┬────┘ └────────┘ └────────┘
                 │         │
        ┌────────▼──┐ ┌────▼───────────────────────┐
        │ ChromaDB  │ │ Orchestrator + 4 Experts:  │
        │ + Embed   │ │ Literature/Data/Write/Review│
        │ + Rerank  │ └────────────────────────────┘
        └───────────┘
                 │
        ┌────────▼──────────┐
        │ DeepSeek LLM API  │
        │ (OpenAI 兼容)     │
        └───────────────────┘
```

### 目录结构

```text
.
├── app.py                  # Gradio Web UI 入口
├── config/settings.py      # 全局配置（API key 读取 / 路径 / 上限）
├── core/
│   ├── llm.py              # DeepSeek LLM 客户端（chat / chat_stream）
│   ├── react_agent.py      # ReAct Agent + ToolRegistry
│   └── mcp.py              # MCP Server 工具协议层
├── agents/
│   ├── base_agent.py       # ExpertAgent 基类
│   ├── specialists.py      # 4 个专家 Agent
│   └── orchestrator.py     # 任务规划与多 Agent 协调器
├── memory/
│   ├── memory_store.py     # 短期(摘要压缩) + 长期(SQLite) 记忆
│   └── chat_history.py     # 多会话历史持久化
├── rag/
│   ├── document_loader.py  # PDF/TXT 加载 + 语义分块
│   ├── vector_store.py     # ChromaDB 封装
│   └── rag_engine.py       # 改写 → 召回 → 重排序 → 拼装
├── skills/skill_manager.py # Skill 加载 / 检索 / 持久化
├── tools/                  # 10+ 工具
│   ├── basic_tools.py      # calculator / get_current_time
│   ├── arxiv_tool.py       # search_arxiv / import_arxiv_paper
│   ├── web_tool.py         # fetch_webpage
│   ├── summarize_tool.py   # summarize_text
│   ├── rag_tool.py         # search_knowledge_base / ingest_paper
│   ├── memory_tool.py      # save_research_finding / recall_memories
│   ├── multi_agent_tool.py # multi_agent_collaborate
│   ├── skill_tool.py       # list/get/create_skill
│   ├── compare_tool.py     # compare_papers
│   └── trend_tool.py       # research_trend
├── utils/audio.py          # 语音转文字（可选）
├── static/                 # 抽离的 CSS / JS
│   ├── style.css
│   └── history.js
├── data/                   # 运行时产物（已 gitignore）
│   ├── chroma_db/          # 向量库
│   ├── papers/             # 用户上传的论文
│   ├── memory.db           # 长期记忆
│   └── chat_history.db     # 会话历史
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- DeepSeek API Key（[官网申请](https://platform.deepseek.com/)，新用户有免费额度）
- 首次运行会自动下载约 400MB 的 reranker 模型 `BAAI/bge-reranker-base`

### 2. 安装

```bash
git clone https://github.com/<your-username>/research-assistant-agent.git
cd research-assistant-agent

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 然后编辑 .env，填入你自己的 DEEPSEEK_API_KEY
```

`.env` 示例：

```env
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 4. 启动

```bash
python app.py
```

浏览器打开 `http://localhost:7860` 即可。

---

## 💡 使用示例

试着输入：

| 类型 | 示例 |
|------|------|
| 📄 论文检索 | `帮我搜索关于 retrieval-augmented generation 的最新论文` |
| 📈 趋势分析 | `分析 diffusion model 近 5 年的研究趋势` |
| 📚 知识库 QA | （先在右侧上传 PDF）`这篇论文的核心创新是什么？` |
| 🤝 多专家协作 | `帮我设计一个关于 LLM 推理能力的研究方案` |
| 🧮 工具组合 | `帮我搜索 transformer 论文，下载第 1 篇并总结摘要` |

观察聊天区底部的 **「🔍 推理过程」** 折叠块，可以看到 Agent 调用了哪些工具、传了什么参数。

---

## 🔬 核心设计亮点

### 1. ReAct Agent — generator + 流式 token

`core/react_agent.py` 的 `run_iter()` 是一个 generator，对外吐出结构化事件：

```text
step_start → thought → action → observation → ... → answer_token* → answer
```

UI 层把这些事件实时渲染成 markdown，用户直接看到「思考 → 工具调用 → 返回 → 再思考」的完整链条，而不是只等到最终结果。

### 2. MCP 工具协议层

`core/mcp.py` 提供了一个**工具自描述协议**：每个工具模块只需暴露一个 `TOOL_DEFINITION` / `TOOL_DEFINITIONS` 列表，`MCPServer` 用 `importlib` 自动加载并注册，新增工具**完全不用动 Agent 代码**。

```python
# tools/your_tool.py
def your_tool(query: str) -> str: ...

TOOL_DEFINITION = {
    "name": "your_tool",
    "description": "...",
    "parameters": {"type": "object", "properties": {...}, "required": [...]},
    "func": your_tool,
}
```

### 3. RAG：查询改写 → 多召回 → 重排序

`rag/rag_engine.py`：

1. **查询改写**：用 LLM 把用户的口语化问题改成更适合语义检索的学术查询（补充同义词 / 英文术语）
2. **多召回**：从 ChromaDB 召回 `top_k * 4`（最多 20）个候选
3. **重排序**：用 `BAAI/bge-reranker-base` Cross-Encoder 精排，输出真正相关的 top-k

加上 `rag/document_loader.py` 的**语义分块**（基于句子嵌入相似度断点），整体检索质量比单纯的固定窗口分块 + 单次召回有显著提升。

### 4. Multi-Agent 协作

`agents/orchestrator.py` 中 Planner LLM 输出结构化的 JSON 任务图：

```json
{
  "plan_summary": "...",
  "subtasks": [
    {"expert": "literature",    "task": "...", "depends_on": []},
    {"expert": "data_analysis", "task": "...", "depends_on": [0]},
    {"expert": "writing",       "task": "...", "depends_on": [0, 1]}
  ]
}
```

按 `depends_on` 拓扑顺序执行，依赖结果作为下游 Agent 的 context，最后由总编 Agent 融合。

### 5. 三层记忆系统

- **ShortTermMemory**：滑动窗口，超过阈值自动用 LLM 压缩为摘要
- **LongTermMemory**：SQLite，存 `preference / research_topic / interaction / finding` 四类
- **ChatHistoryStore**：会话级持久化，支持历史会话加载、删除

注入 prompt 时会拼上摘要 + 相关历史记忆 + 用户偏好三段。

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| LLM | DeepSeek（OpenAI 兼容） |
| Embedding | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Reranker | `BAAI/bge-reranker-base` (Cross-Encoder) |
| 向量库 | ChromaDB（`hnsw:space=cosine` 持久化） |
| 持久化 | SQLite |
| Web UI | Gradio 4.x + 自定义 CSS |
| HTTP | httpx |
| PDF | PyPDF2 |

---

## 🗺️ Roadmap

- [ ] 工具调用并行化（多个 tool_call 一次执行）
- [ ] 接入更多 LLM 后端（OpenAI / Qwen / Claude）
- [ ] 论文图谱与引用关系可视化
- [ ] 接入 Semantic Scholar / Google Scholar 数据源
- [ ] Docker 化部署 + 多用户隔离

---

## 📄 License

MIT License — 仅供学习和个人项目使用。
