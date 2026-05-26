"""科研助手 Agent — Gradio Web UI 入口。"""

import datetime
import json
import os
import shutil
from pathlib import Path

import gradio as gr

from config.settings import PAPERS_DIR
from core.mcp import create_default_mcp_server
from core.react_agent import ReActAgent
from memory.chat_history import ChatHistoryStore
from memory.memory_store import MemoryManager

STATIC_DIR = Path(__file__).parent / "static"


# ── Agent lifecycle ──────────────────────────────────────────────

agent: ReActAgent | None = None
mcp_tools_info: list[dict] = []
history_store = ChatHistoryStore()
current_session_id: int | None = None


def init_agent() -> ReActAgent:
    global agent, mcp_tools_info
    memory = MemoryManager()
    agent = ReActAgent(memory_manager=memory)
    mcp = create_default_mcp_server()
    mcp.bind_to_agent(agent)
    mcp_tools_info = mcp.list_tools()
    return agent


# ── ReAct event → Markdown rendering ─────────────────────────────

def format_streaming(events: list[dict]) -> str:
    """While the agent is still reasoning: show every intermediate step."""
    parts = []
    for ev in events:
        t = ev["type"]
        if t == "step_start":
            parts.append(f"\n---\n\n**步骤 {ev['step']}** / {ev['max_steps']}\n")
        elif t == "thought":
            parts.append(f"\n> 💭 {ev['content']}\n")
        elif t == "action":
            args_str = json.dumps(ev["args"], ensure_ascii=False, indent=2)
            parts.append(f"\n🔧 调用 **`{ev['tool']}`**\n\n```json\n{args_str}\n```\n")
        elif t == "observation":
            result = ev["result"]
            if len(result) > 500:
                result = result[:500] + "\n…(已截断)"
            parts.append(f"\n📋 **返回结果**\n\n{result}\n")
        elif t == "error":
            parts.append(f"\n❌ **错误**: {ev['content']}\n")
    parts.append("\n\n⏳ *正在推理…*")
    return "\n".join(parts)


def format_final(events: list[dict]) -> str:
    """When done: collapse the full trace under a `<details>` block."""
    answer = ""
    step_parts = []
    step_count = 0
    for ev in events:
        t = ev["type"]
        if t == "step_start":
            step_count += 1
            step_parts.append(f"\n**步骤 {ev['step']}**\n")
        elif t == "thought":
            step_parts.append(f"> 💭 {ev['content']}\n")
        elif t == "action":
            args_str = json.dumps(ev["args"], ensure_ascii=False)
            if len(args_str) > 80:
                args_str = args_str[:80] + "…"
            step_parts.append(f"🔧 `{ev['tool']}` · `{args_str}`\n")
        elif t == "observation":
            result = ev["result"]
            if len(result) > 200:
                result = result[:200] + "…"
            step_parts.append(f"📋 {result}\n")
        elif t == "answer":
            answer = ev["content"]
        elif t == "error":
            if not answer:
                answer = f"❌ 处理出错: {ev['content']}"
    if step_parts and answer:
        steps_md = "\n".join(step_parts)
        return (
            f"{answer}\n\n"
            f"<details>\n<summary>🔍 推理过程（{step_count} 步）</summary>\n\n"
            f"{steps_md}\n</details>"
        )
    return answer or "未能生成回答。"


def _build_trace_details(step_events: list[dict]) -> str:
    """Reasoning trace block appended to streamed answers."""
    if not step_events:
        return ""
    parts = []
    step_count = 0
    for ev in step_events:
        t = ev["type"]
        if t == "step_start":
            step_count += 1
            parts.append(f"\n**步骤 {ev['step']}**\n")
        elif t == "thought":
            parts.append(f"> 💭 {ev['content']}\n")
        elif t == "action":
            args_str = json.dumps(ev["args"], ensure_ascii=False)
            if len(args_str) > 80:
                args_str = args_str[:80] + "…"
            parts.append(f"🔧 `{ev['tool']}` · `{args_str}`\n")
        elif t == "observation":
            result = ev["result"][:200] + "…" if len(ev["result"]) > 200 else ev["result"]
            parts.append(f"📋 {result}\n")
    if not parts:
        return ""
    return (
        f"\n\n<details>\n<summary>🔍 推理过程（{step_count} 步）</summary>\n\n"
        + "\n".join(parts) + "\n</details>"
    )


# ── Chat handlers ────────────────────────────────────────────────

def user_submit(message: str, history: list):
    if not message.strip():
        return "", history
    global current_session_id
    if current_session_id is None:
        title = message.strip()[:30] + ("…" if len(message.strip()) > 30 else "")
        current_session_id = history_store.create_session(title)
    history_store.save_message(current_session_id, "user", message.strip())
    return "", history + [{"role": "user", "content": message.strip()}]


def bot_respond(history: list):
    """Stream agent events into the chatbot, then persist the final reply."""
    if not history or history[-1]["role"] != "user":
        yield history
        return
    user_msg = history[-1]["content"]
    history = history + [{"role": "assistant", "content": "⏳ 正在思考…"}]
    yield history

    events: list[dict] = []
    step_events: list[dict] = []
    streaming_answer = ""
    try:
        for event in agent.run_iter(user_msg):
            etype = event["type"]
            if etype == "answer_token":
                streaming_answer = event["partial"]
                trace = _build_trace_details(step_events)
                history[-1] = {"role": "assistant", "content": streaming_answer + trace}
                yield history
                continue
            events.append(event)
            if etype == "answer":
                history[-1] = {"role": "assistant", "content": format_final(events)}
            elif etype in ("step_start", "thought", "action", "observation", "error"):
                step_events.append(event)
                history[-1] = {"role": "assistant", "content": format_streaming(step_events)}
            yield history
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ 发生错误: {e}"}
        yield history
        return

    if not any(e["type"] == "answer" for e in events):
        history[-1] = {"role": "assistant", "content": format_final(events)}
        yield history

    if current_session_id and history and history[-1]["role"] == "assistant":
        history_store.save_message(current_session_id, "assistant", history[-1]["content"])


def handle_reset():
    global current_session_id
    if agent:
        agent.reset()
    current_session_id = None
    return [], ""


# ── History sidebar handlers ────────────────────────────────────

def build_history_html() -> str:
    sessions = history_store.list_sessions(limit=20)
    if not sessions:
        return '<div class="history-empty">暂无历史对话</div>'
    html = '<div class="history-section-title">历史对话</div><div class="history-list">'
    for s in sessions:
        ts = s["updated_at"][:16].replace("T", " ")
        title = s["title"] if len(s["title"]) <= 22 else s["title"][:20] + "…"
        html += (
            f'<div class="history-item" data-sid="{s["id"]}">'
            f'<span class="history-title">{title}</span>'
            f'<span class="history-meta">{ts}</span>'
            f'<button class="history-del" data-del-sid="{s["id"]}" title="删除">×</button>'
            f'</div>'
        )
    html += '</div>'
    return html


def _parse_sid(value: str) -> int | None:
    """The hidden textbox carries `<sid>|<timestamp>`; extract the sid."""
    if not value:
        return None
    try:
        return int(value.split("|")[0])
    except (ValueError, TypeError):
        return None


def load_session(session_id_str: str):
    global current_session_id
    sid = _parse_sid(session_id_str)
    if sid is None:
        return [], build_history_html()
    messages = history_store.get_messages(sid)
    current_session_id = sid
    if agent:
        agent.reset()
    return messages, build_history_html()


def delete_session(session_id_str: str):
    sid = _parse_sid(session_id_str)
    if sid is None:
        return build_history_html()
    history_store.delete_session(sid)
    return build_history_html()


# ── Knowledge-base handlers ──────────────────────────────────────

SUPPORTED_DOC_EXTS = (".pdf", ".txt", ".md", ".tex")


def _build_kb_html(upload_results: list[str] | None = None) -> str:
    from tools.rag_tool import _get_engine
    engine = _get_engine()
    stats = engine.get_stats()
    doc_count = stats["document_count"]

    files: list[str] = []
    if os.path.exists(PAPERS_DIR):
        files = sorted(
            f for f in os.listdir(PAPERS_DIR)
            if f.lower().endswith(SUPPORTED_DOC_EXTS)
        )

    html = '<div class="kb-panel">'
    html += (
        f'<div class="kb-stats">'
        f'<span class="stat-item">📁 {len(files)} 个文件</span>'
        f'<span class="stat-dot">·</span>'
        f'<span class="stat-item">📊 {doc_count} 个向量块</span>'
        f'</div>'
    )
    if upload_results:
        for r in upload_results:
            html += f'<div class="kb-upload-ok">✅ {r}</div>'
    if files:
        html += '<div class="kb-file-list">'
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            icon = "📄" if ext == ".pdf" else "📝"
            name = f if len(f) <= 28 else f[:25] + "…" + ext
            html += f'<div class="kb-file"><span class="kb-file-icon">{icon}</span> {name}</div>'
        html += '</div>'
    else:
        html += '<div class="kb-empty">上传论文</div>'
    html += '</div>'
    return html


def handle_upload(files) -> str:
    if not files:
        return _build_kb_html()
    from tools.rag_tool import ingest_paper
    results = []
    file_list = files if isinstance(files, list) else [files]
    for file_path in file_list:
        filename = os.path.basename(file_path)
        dest = os.path.join(PAPERS_DIR, filename)
        os.makedirs(PAPERS_DIR, exist_ok=True)
        shutil.copy2(file_path, dest)
        result = ingest_paper(dest)
        results.append(f"**{filename}** — {result.splitlines()[1] if len(result.splitlines()) > 1 else result}")
    return _build_kb_html(results)


def handle_delete_file(filename: str) -> str:
    if not filename:
        return _build_kb_html()
    from tools.rag_tool import _get_engine
    engine = _get_engine()
    n = engine.delete_file(filename)
    fpath = os.path.join(PAPERS_DIR, filename)
    if os.path.exists(fpath):
        os.remove(fpath)
    return _build_kb_html([f"已删除 **{filename}**（{n} 个向量块）"])


def get_kb_file_choices() -> list[str]:
    if not os.path.exists(PAPERS_DIR):
        return []
    return sorted(
        f for f in os.listdir(PAPERS_DIR)
        if f.lower().endswith(SUPPORTED_DOC_EXTS)
    )


# ── Export handler ───────────────────────────────────────────────

def handle_export(history: list) -> str | None:
    if not history:
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_export_{ts}.md"
    export_path = os.path.join(os.path.dirname(__file__), "data", filename)
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    lines = [
        "# 科研助手对话记录\n",
        f"导出时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n",
    ]
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"\n## 🧑 用户\n\n{content}\n")
        elif role == "assistant":
            # 去掉折叠的推理过程，只导出最终答复。
            clean = content.split("<details>")[0].strip()
            lines.append(f"\n## 🤖 助手\n\n{clean}\n")
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return export_path


# ── HTML builders for the right sidebar ──────────────────────────

CATEGORY_ICONS = {
    "基础工具": "⚡", "论文检索": "📄", "网络工具": "🌐",
    "文本处理": "📝", "知识库": "🗄️", "记忆系统": "💾",
    "多Agent协作": "🤝", "技能系统": "🎯",
}
CAT_COLORS = {
    "基础工具": "#818cf8", "论文检索": "#ec4899", "网络工具": "#0ea5e9",
    "文本处理": "#8b5cf6", "知识库": "#10b981", "记忆系统": "#f59e0b",
    "多Agent协作": "#f97316", "技能系统": "#ef4444",
}


def build_tools_html(tools: list[dict]) -> str:
    categories: dict[str, list[dict]] = {}
    for t in tools:
        categories.setdefault(t.get("category", "通用"), []).append(t)
    html = '<div class="tools-accordion">'
    for cat, items in categories.items():
        icon = CATEGORY_ICONS.get(cat, "🔧")
        color = CAT_COLORS.get(cat, "#94a3b8")
        html += '<details class="tool-group">'
        html += (
            f'<summary>'
            f'<span class="cat-dot" style="background:{color}"></span>'
            f' {icon} {cat}'
            f'<span class="tg-count">{len(items)}</span>'
            f'</summary>'
        )
        html += '<div class="tool-group-body">'
        for t in items:
            desc = t["description"]
            if len(desc) > 40:
                desc = desc[:40] + "…"
            html += (
                f'<div class="tool-row">'
                f'<span class="tool-fn">{t["name"]}</span>'
                f'<span class="tool-help">{desc}</span>'
                f'</div>'
            )
        html += '</div></details>'
    html += '</div>'
    return html


def build_skills_html() -> str:
    try:
        from skills.skill_manager import SkillManager
        mgr = SkillManager()
        skills = mgr.list_skills()
    except Exception:
        skills = []
    if not skills:
        return (
            '<div class="skills-empty">'
            '<p class="skills-empty-title">暂无可用技能</p>'
            '<p class="skills-empty-hint">通过对话创建可复用的研究流程模板</p>'
            '</div>'
        )
    html = '<div class="skills-list">'
    for s in skills:
        html += (
            f'<div class="skill-card">'
            f'<div class="skill-header">'
            f'<span class="skill-name">{s.name}</span>'
            f'<span class="skill-badge">{s.usage_count}次</span>'
            f'</div>'
            f'<div class="skill-desc">{s.description}</div>'
            f'<div class="skill-cat">{s.category}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


# ── Static assets & layout snippets ──────────────────────────────

PLACEHOLDER_HTML = """
<div style="
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    padding:48px 24px;height:100%;min-height:400px;
    font-family:'Inter',system-ui,sans-serif;
">
    <div style="font-size:2.2em;margin-bottom:6px;">🔬</div>
    <p style="font-size:1.8em;font-weight:700;color:#1E293B;margin:0 0 4px;
        letter-spacing:-0.02em;">科研助手</p>
    <p style="font-size:0.88em;color:#94A3B8;margin:0 0 32px;font-weight:400;">
        基于 ReAct + DeepSeek 的智能研究助手，在下方输入框提问即可开始</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;
        max-width:520px;width:100%;">
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
            padding:22px 16px;text-align:center;">
            <div style="font-size:1.6em;margin-bottom:6px;">📄</div>
            <div style="font-size:0.9em;font-weight:600;color:#334155;">论文检索</div>
            <div style="font-size:0.72em;color:#94A3B8;margin-top:4px;">搜索 arXiv 学术数据库</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
            padding:22px 16px;text-align:center;">
            <div style="font-size:1.6em;margin-bottom:6px;">📝</div>
            <div style="font-size:0.9em;font-weight:600;color:#334155;">智能摘要</div>
            <div style="font-size:0.72em;color:#94A3B8;margin-top:4px;">快速提炼论文核心内容</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
            padding:22px 16px;text-align:center;">
            <div style="font-size:1.6em;margin-bottom:6px;">📚</div>
            <div style="font-size:0.9em;font-weight:600;color:#334155;">知识库问答</div>
            <div style="font-size:0.72em;color:#94A3B8;margin-top:4px;">基于本地论文库语义检索</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
            padding:22px 16px;text-align:center;">
            <div style="font-size:1.6em;margin-bottom:6px;">🤝</div>
            <div style="font-size:0.9em;font-weight:600;color:#334155;">多专家协作</div>
            <div style="font-size:0.72em;color:#94A3B8;margin-top:4px;">多 Agent 协同深度分析</div>
        </div>
    </div>
    <div style="margin-top:28px;display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">
        <span style="font-size:0.68em;color:#CBD5E1;padding:4px 12px;
            border:1px solid #E2E8F0;border-radius:12px;">ReAct</span>
        <span style="font-size:0.68em;color:#CBD5E1;padding:4px 12px;
            border:1px solid #E2E8F0;border-radius:12px;">DeepSeek</span>
        <span style="font-size:0.68em;color:#CBD5E1;padding:4px 12px;
            border:1px solid #E2E8F0;border-radius:12px;">RAG</span>
        <span style="font-size:0.68em;color:#CBD5E1;padding:4px 12px;
            border:1px solid #E2E8F0;border-radius:12px;">Memory</span>
    </div>
</div>
"""

BRAND_HTML = (
    '<div class="brand-bar">'
    '<span class="brand-logo"></span>'
    '<span class="brand-text">科研助手</span>'
    '<span class="brand-ver">Research Agent</span>'
    '</div>'
)

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


# ── Layout ───────────────────────────────────────────────────────

def create_demo() -> gr.Blocks:
    init_agent()
    tools_html = build_tools_html(mcp_tools_info)
    skills_html = build_skills_html()

    with gr.Blocks(title="科研助手 Agent", fill_height=True) as demo:
        with gr.Row(equal_height=True, elem_id="main-row"):

            # Left sidebar: brand + new chat + history
            with gr.Column(scale=1, min_width=200, elem_id="left-sidebar"):
                gr.HTML(BRAND_HTML)
                reset_btn = gr.Button("+ 新建对话", variant="primary", size="sm", elem_id="new-chat-btn")
                history_html = gr.HTML(value=build_history_html(), elem_id="history-panel")
                # Hidden textboxes — JS pushes session ids here.
                session_loader = gr.Textbox(visible=False, elem_id="session-loader")
                session_deleter = gr.Textbox(visible=False, elem_id="session-deleter")

            # Center: chat area
            with gr.Column(scale=5, elem_id="center-col"):
                chatbot = gr.Chatbot(
                    height="calc(100vh - 180px)",
                    show_label=False,
                    placeholder=PLACEHOLDER_HTML,
                )
                with gr.Row(elem_id="input-row"):
                    msg = gr.Textbox(
                        placeholder="输入你的问题，按 Enter 发送…",
                        show_label=False, scale=10, container=False,
                        autofocus=True, elem_id="msg-box",
                    )
                    send_btn = gr.Button("发送 →", variant="primary", scale=1, min_width=80, elem_id="send-btn")
                    export_btn = gr.DownloadButton("📥 导出", variant="secondary", scale=1, min_width=72, elem_id="export-btn")
                gr.Examples(
                    examples=[
                        "帮我搜索关于大语言模型的最新论文",
                        "分析 RAG 领域的研究趋势",
                        "计算 sqrt(144) + log(100)",
                        "帮我对知识库中的论文做个总结",
                    ],
                    inputs=msg, label="💡 快速提问", elem_id="examples-row",
                )

            # Right sidebar: knowledge / tools / skills
            with gr.Column(scale=2, min_width=240, elem_id="right-sidebar"):
                with gr.Tabs():
                    with gr.Tab("📚 知识库"):
                        kb_status = gr.HTML(value=_build_kb_html())
                        upload = gr.File(
                            label=None, show_label=False,
                            file_types=list(SUPPORTED_DOC_EXTS),
                            file_count="multiple", type="filepath",
                            elem_classes=["upload-area"],
                        )
                        with gr.Row(elem_classes=["kb-delete-row"]):
                            del_dropdown = gr.Dropdown(
                                choices=get_kb_file_choices(), label="",
                                container=False, scale=3, elem_id="del-dropdown",
                            )
                            del_btn = gr.Button(
                                "🗑️", variant="secondary", size="sm",
                                scale=1, min_width=36, elem_id="del-btn",
                            )

                    with gr.Tab("⚡ 工具箱"):
                        gr.HTML(tools_html)

                    with gr.Tab("🎯 技能"):
                        gr.HTML(skills_html)

        gr.HTML('<div class="app-footer">Powered by DeepSeek · ReAct Framework · RAG Engine</div>')

        # ── Event wiring ──
        submit_ev = msg.submit(user_submit, [msg, chatbot], [msg, chatbot])
        submit_ev.then(bot_respond, chatbot, chatbot).then(
            lambda: build_history_html(), outputs=[history_html],
        )

        click_ev = send_btn.click(user_submit, [msg, chatbot], [msg, chatbot])
        click_ev.then(bot_respond, chatbot, chatbot).then(
            lambda: build_history_html(), outputs=[history_html],
        )

        reset_btn.click(handle_reset, outputs=[chatbot, msg]).then(
            lambda: build_history_html(), outputs=[history_html],
        )

        upload.change(handle_upload, inputs=[upload], outputs=[kb_status]).then(
            lambda: gr.update(choices=get_kb_file_choices()), outputs=[del_dropdown],
        )
        del_btn.click(handle_delete_file, inputs=[del_dropdown], outputs=[kb_status]).then(
            lambda: gr.update(choices=get_kb_file_choices(), value=None), outputs=[del_dropdown],
        )

        export_btn.click(handle_export, inputs=[chatbot], outputs=[export_btn])

        session_loader.change(load_session, inputs=[session_loader], outputs=[chatbot, history_html])
        session_deleter.change(delete_session, inputs=[session_deleter], outputs=[history_html])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=THEME,
        css=_read_static("style.css"),
        head=f"<script>{_read_static('history.js')}</script>",
    )
