"""Arxiv 论文搜索与导入工具。"""

import os
import httpx
import xml.etree.ElementTree as ET
from config.settings import PAPERS_DIR


def search_arxiv(query: str, max_results: int = 5) -> str:
    """通过 Arxiv API 搜索学术论文。"""
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        resp = httpx.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return f"Arxiv 搜索出错：{e}"

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ns)

    if not entries:
        return f"未找到与 '{query}' 相关的论文。"

    results = []
    for i, entry in enumerate(entries, 1):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:200]
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        published = entry.find("atom:published", ns).text[:10]
        link = entry.find("atom:id", ns).text

        results.append(
            f"{i}. 【{title}】\n"
            f"   作者：{', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}\n"
            f"   日期：{published}\n"
            f"   摘要：{summary}...\n"
            f"   链接：{link}"
        )

    return f"找到 {len(results)} 篇相关论文：\n\n" + "\n\n".join(results)


def import_arxiv_paper(arxiv_url: str) -> str:
    """下载 Arxiv 论文 PDF 并导入知识库。"""
    arxiv_id = arxiv_url.rstrip("/").split("/")[-1]
    if arxiv_id.startswith("abs"):
        arxiv_id = arxiv_id.replace("abs/", "")
    arxiv_id = arxiv_id.replace("v", ".v").split(".v")[0] if ".v" not in arxiv_id else arxiv_id.split("v")[0].rstrip(".")

    for prefix in ("http://arxiv.org/abs/", "https://arxiv.org/abs/"):
        arxiv_id = arxiv_id.replace(prefix, "")

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
    dest = os.path.join(PAPERS_DIR, filename)
    os.makedirs(PAPERS_DIR, exist_ok=True)

    try:
        resp = httpx.get(pdf_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        return f"下载论文失败：{e}"

    from tools.rag_tool import ingest_paper
    result = ingest_paper(dest)
    return f"已下载并导入论文 {arxiv_id}\n{result}"


TOOL_DEFINITIONS = [
    {
        "name": "search_arxiv",
        "description": "搜索 Arxiv 学术论文数据库。输入关键词，返回相关论文的标题、作者、摘要和链接。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，例如 'transformer attention mechanism' 或 'large language model'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        "func": search_arxiv,
    },
    {
        "name": "import_arxiv_paper",
        "description": "通过 Arxiv 论文链接或 ID 下载 PDF 并自动导入知识库。例如输入 'http://arxiv.org/abs/2005.11401' 或 '2005.11401'。",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_url": {
                    "type": "string",
                    "description": "Arxiv 论文链接或 ID",
                },
            },
            "required": ["arxiv_url"],
        },
        "func": import_arxiv_paper,
    },
]
