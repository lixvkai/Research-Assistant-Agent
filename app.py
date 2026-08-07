"""科研助手 Agent — Gradio Web UI 入口。

本模块只负责「渲染 + 事件接线」，业务逻辑都在 `services/`（与 FastAPI 共用同一层）。
当前会话 id 存在 `gr.State` 里而不是模块级变量，因此多个浏览器标签/用户各自独立。
"""

import datetime
import html
import json
import logging
import os
from pathlib import Path

import gradio as gr

from config.settings import (
    GRADIO_AUTH_PASSWORD,
    GRADIO_AUTH_USERNAME,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    LOG_LEVEL,
)
from services import (
    SessionBusyError,
    SessionNotFoundError,
    get_agent_service,
    get_kb_service,
)
from services.kb_service import SUPPORTED_DOC_EXTS

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        elif t == "reflection":
            parts.append(f"\n🔁 **自我反思**：{ev['critique']}\n")
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
        elif t == "reflection":
            step_parts.append(f"🔁 自我反思：{ev['critique']}\n")
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


# ── Chat handlers ────────────────────────────────────────────────

def user_submit(message: str, history: list, session_id: int | None):
    """把用户消息显示到界面上，并确保有一个可用的会话。

    消息的落库交给服务层（`stream_chat` 内完成），这里不再重复写库。
    """
    text = (message or "").strip()
    if not text:
        return "", history, session_id
    sid = get_agent_service().ensure_session(session_id, text)
    return "", history + [{"role": "user", "content": text}], sid


def bot_respond(history: list, session_id: int | None):
    """把 Agent 的事件流渲染进对话框。落库由服务层负责。"""
    if not history or history[-1]["role"] != "user":
        yield history
        return
    user_msg = history[-1]["content"]
    service = get_agent_service()
    history = history + [{"role": "assistant", "content": "⏳ 正在思考…"}]
    yield history

    try:
        guard = service.acquire_session(session_id)
    except SessionBusyError:
        history[-1] = {"role": "assistant", "content": "⏳ 上一条消息还在处理中，请稍候再发送。"}
        yield history
        return

    events: list[dict] = []
    step_events: list[dict] = []
    try:
        for event in service.stream_chat(session_id, user_msg):
            etype = event["type"]
            events.append(event)
            if etype == "answer":
                history[-1] = {"role": "assistant", "content": format_final(events)}
            elif etype in ("step_start", "thought", "action", "observation", "reflection", "error"):
                step_events.append(event)
                history[-1] = {"role": "assistant", "content": format_streaming(step_events)}
            yield history
    except Exception as e:
        logger.exception("会话 %s 推理失败", session_id)
        history[-1] = {"role": "assistant", "content": f"❌ 发生错误: {e}"}
        yield history
        return
    finally:
        guard.release()

    if not any(e["type"] == "answer" for e in events):
        history[-1] = {"role": "assistant", "content": format_final(events)}
        yield history


def handle_reset(session_id: int | None):
    """新建对话：先把上一段会话固化进长期记忆，再清空当前会话状态。"""
    if session_id is not None:
        get_agent_service().reset_session(session_id)
    return [], "", None


# ── History sidebar handlers ────────────────────────────────────

def build_history_html() -> str:
    sessions = get_agent_service().list_sessions(limit=20)
    if not sessions:
        return '<div class="history-empty">暂无历史对话</div>'
    parts = ['<div class="history-section-title">历史对话</div><div class="history-list">']
    for s in sessions:
        ts = html.escape(s["updated_at"][:16].replace("T", " "))
        raw_title = s["title"] if len(s["title"]) <= 22 else s["title"][:20] + "…"
        title = html.escape(raw_title)
        sid = html.escape(str(s["id"]))
        parts.append(
            f'<div class="history-item" data-sid="{sid}">'
            f'<span class="history-title">{title}</span>'
            f'<span class="history-meta">{ts}</span>'
            f'<button class="history-del" data-del-sid="{sid}" title="删除">×</button>'
            f'</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _parse_sid(value: str) -> int | None:
    """The hidden textbox carries `<sid>|<timestamp>`; extract the sid."""
    if not value:
        return None
    try:
        return int(value.split("|")[0])
    except (ValueError, TypeError):
        return None


def load_session(session_id_str: str):
    """切换历史会话。

    Agent 侧不需要在这里做任何事：服务层会在该会话下一次真正对话时，
    按需把落库的消息回灌进图状态（见 `AgentService._ensure_hydrated`）。
    """
    sid = _parse_sid(session_id_str)
    if sid is None:
        return [], build_history_html(), None
    try:
        messages = get_agent_service().get_messages(sid)
    except SessionNotFoundError:
        return [], build_history_html(), None
    return messages, build_history_html(), sid


def delete_session(session_id_str: str, current: int | None, history: list):
    """删除某个会话；若删的正是当前打开的会话，同时把界面清空。"""
    sid = _parse_sid(session_id_str)
    if sid is None:
        return build_history_html(), history, current
    try:
        get_agent_service().delete_session(sid)
    except SessionNotFoundError:
        pass
    if sid == current:
        return build_history_html(), [], None
    return build_history_html(), history, current


# ── Knowledge-base handlers ──────────────────────────────────────

def _build_kb_html(notices: list[str] | None = None) -> str:
    kb = get_kb_service()
    stats = kb.stats()
    files = kb.list_files()

    parts = ['<div class="kb-panel">']
    parts.append(
        f'<div class="kb-stats">'
        f'<span class="stat-item">📁 {stats["files"]} 个文件</span>'
        f'<span class="stat-dot">·</span>'
        f'<span class="stat-item">📊 {stats["chunks"]} 个向量块</span>'
        f'</div>'
    )
    for notice in notices or []:
        parts.append(f'<div class="kb-upload-ok">{html.escape(notice)}</div>')
    if files:
        parts.append('<div class="kb-file-list">')
        for f in files:
            name = f["name"]
            ext = os.path.splitext(name)[1].lower()
            icon = "📄" if ext == ".pdf" else "📝"
            shown = name if len(name) <= 28 else name[:25] + "…" + ext
            parts.append(
                f'<div class="kb-file"><span class="kb-file-icon">{icon}</span> '
                f'{html.escape(shown)}</div>'
            )
        parts.append("</div>")
    else:
        parts.append('<div class="kb-empty">上传论文</div>')
    parts.append("</div>")
    return "".join(parts)


def handle_upload(files) -> str:
    if not files:
        return _build_kb_html()
    kb = get_kb_service()
    notices = []
    for file_path in files if isinstance(files, list) else [files]:
        result = kb.ingest_path(file_path)
        if result["ok"]:
            notices.append(f'✅ {result["filename"]} — 生成 {result["chunks"]} 个文本块')
        else:
            notices.append(f'❌ {result["filename"]} — {result["error"]}')
    return _build_kb_html(notices)


def handle_delete_file(filename: str) -> str:
    if not filename:
        return _build_kb_html()
    result = get_kb_service().delete_file(filename)
    if result["ok"]:
        return _build_kb_html([f'✅ 已删除 {result["filename"]}（{result["chunks"]} 个向量块）'])
    return _build_kb_html([f'❌ 删除失败：{result["error"]}'])


def get_kb_file_choices() -> list[str]:
    return [f["name"] for f in get_kb_service().list_files()]


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
    skills = get_agent_service().list_skills()
    if not skills:
        return (
            '<div class="skills-empty">'
            '<p class="skills-empty-title">暂无可用技能</p>'
            '<p class="skills-empty-hint">通过对话创建可复用的研究流程模板</p>'
            '</div>'
        )
    parts = ['<div class="skills-list">']
    for s in skills:
        used = s["usage_count"]
        badge = f'{used}次 · 成功率{s["success_rate"]:.0%}' if used > 0 else "未执行"
        parts.append(
            f'<div class="skill-card">'
            f'<div class="skill-header">'
            f'<span class="skill-name">{html.escape(s["name"])}</span>'
            f'<span class="skill-badge">{html.escape(badge)}</span>'
            f"</div>"
            f'<div class="skill-desc">{html.escape(s["description"])}</div>'
            f'<div class="skill-cat">{html.escape(s["category"])}</div>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


# ── Static assets & layout snippets ──────────────────────────────

PLACEHOLDER_HTML = """
<div class="empty-state">
    <div class="empty-state-icon">🔬</div>
    <p class="empty-state-title">科研助手</p>
    <p class="empty-state-subtitle">
        基于 LangGraph 与 Multi-Agent 架构的智能科研助手，在下方输入框提问即可开始</p>
    <div class="capability-grid">
        <div class="capability-card">
            <div class="capability-icon">📄</div>
            <div class="capability-title">论文检索</div>
            <div class="capability-desc">搜索 arXiv 学术数据库</div>
        </div>
        <div class="capability-card">
            <div class="capability-icon">📝</div>
            <div class="capability-title">智能摘要</div>
            <div class="capability-desc">快速提炼论文核心内容</div>
        </div>
        <div class="capability-card">
            <div class="capability-icon">📚</div>
            <div class="capability-title">知识库问答</div>
            <div class="capability-desc">基于本地论文库语义检索</div>
        </div>
        <div class="capability-card">
            <div class="capability-icon">🤝</div>
            <div class="capability-title">多专家协作</div>
            <div class="capability-desc">多 Agent 协同深度分析</div>
        </div>
    </div>
    <div class="tech-tags">
        <span>ReAct</span><span>DeepSeek</span><span>RAG</span><span>Memory</span>
    </div>
</div>
"""

BRAND_HTML = (
    '<div class="brand-bar">'
    '<span class="brand-logo">🔬</span>'
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
    service = get_agent_service()
    tools_html = build_tools_html(service.list_tools())
    skills_html = build_skills_html()

    with gr.Blocks(
        title="科研助手 Agent",
        fill_height=True,
        theme=THEME,
        css=_read_static("style.css"),
        head=f"<script>{_read_static('history.js')}</script>",
    ) as demo:
        # 当前会话 id —— 每个浏览器会话一份，取代原来的模块级全局变量
        session_state = gr.State(None)

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
                    type="messages",
                    height="100%",
                    show_label=False,
                    placeholder=PLACEHOLDER_HTML,
                    elem_id="chatbot",
                )
                with gr.Column(elem_id="composer"):
                    with gr.Row(elem_id="input-row"):
                        msg = gr.Textbox(
                            placeholder="输入你的问题，按 Enter 发送…",
                            show_label=False, scale=10, container=False,
                            autofocus=True, elem_id="msg-box",
                        )
                        send_btn = gr.Button(
                            "发送", variant="primary", scale=1,
                            min_width=72, elem_id="send-btn",
                        )
                        export_btn = gr.DownloadButton(
                            "导出", variant="secondary", scale=1,
                            min_width=64, elem_id="export-btn",
                        )
                    gr.HTML('<div class="quick-prompts-title">快速提问</div>')
                    gr.Examples(
                        examples=[
                            "帮我搜索关于大语言模型的最新论文",
                            "分析 RAG 领域的研究趋势",
                            "计算 sqrt(144) + log(100)",
                            "帮我对知识库中的论文做个总结",
                        ],
                        inputs=msg, label=None, elem_id="examples-row",
                    )

            # Right sidebar: knowledge / tools / skills
            with gr.Column(scale=2, min_width=240, elem_id="right-sidebar"):
                with gr.Tabs(elem_id="resource-tabs"):
                    with gr.Tab("📚 知识库"):
                        kb_status = gr.HTML(value=_build_kb_html())
                        gr.HTML('<div class="panel-section-title">添加文档</div>')
                        upload = gr.File(
                            label=None, show_label=False,
                            file_types=list(SUPPORTED_DOC_EXTS),
                            file_count="multiple", type="filepath",
                            elem_id="kb-upload", elem_classes=["upload-area"],
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
        for trigger in (msg.submit, send_btn.click):
            ev = trigger(
                user_submit,
                [msg, chatbot, session_state],
                [msg, chatbot, session_state],
            )
            ev.then(bot_respond, [chatbot, session_state], chatbot).then(
                lambda: build_history_html(), outputs=[history_html],
            )

        reset_btn.click(
            handle_reset, [session_state], [chatbot, msg, session_state],
        ).then(lambda: build_history_html(), outputs=[history_html])

        upload.change(handle_upload, inputs=[upload], outputs=[kb_status]).then(
            lambda: gr.update(choices=get_kb_file_choices()), outputs=[del_dropdown],
        )
        del_btn.click(handle_delete_file, inputs=[del_dropdown], outputs=[kb_status]).then(
            lambda: gr.update(choices=get_kb_file_choices(), value=None), outputs=[del_dropdown],
        )

        export_btn.click(handle_export, inputs=[chatbot], outputs=[export_btn])

        session_loader.change(
            load_session,
            inputs=[session_loader],
            outputs=[chatbot, history_html, session_state],
        )
        session_deleter.change(
            delete_session,
            inputs=[session_deleter, session_state, chatbot],
            outputs=[history_html, chatbot, session_state],
        )

    return demo


if __name__ == "__main__":
    import atexit

    from core.observability import shutdown_observability

    atexit.register(shutdown_observability)
    demo = create_demo()
    auth = None
    if GRADIO_AUTH_USERNAME and GRADIO_AUTH_PASSWORD:
        auth = (GRADIO_AUTH_USERNAME, GRADIO_AUTH_PASSWORD)
        logger.info("已启用 Gradio Basic Auth（用户：%s）", GRADIO_AUTH_USERNAME)
    elif GRADIO_SERVER_NAME not in ("127.0.0.1", "localhost"):
        logger.warning(
            "Gradio 监听 %s 且未配置 GRADIO_AUTH_*，局域网可匿名访问；"
            "生产环境请设置鉴权或改为 127.0.0.1",
            GRADIO_SERVER_NAME,
        )
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False,
        auth=auth,
    )
