"""研究趋势分析工具 — 统计 Arxiv 论文数量年度趋势。"""

import httpx
import xml.etree.ElementTree as ET
from collections import Counter


def research_trend(query: str, years: int = 5) -> str:
    """统计某个研究方向在 Arxiv 上近几年的论文数量趋势。"""
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 200,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        resp = httpx.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return f"Arxiv 查询出错：{e}"

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ns)

    if not entries:
        return f"未找到与 '{query}' 相关的论文。"

    year_counts = Counter()
    for entry in entries:
        published = entry.find("atom:published", ns).text[:4]
        year_counts[published] += 1

    sorted_years = sorted(year_counts.keys())[-years:]

    lines = [f"📊 **「{query}」研究趋势**（基于 Arxiv 最新 {len(entries)} 篇论文）\n"]
    lines.append("| 年份 | 论文数 | 趋势 |")
    lines.append("|------|--------|------|")

    max_count = max(year_counts[y] for y in sorted_years) if sorted_years else 1
    for y in sorted_years:
        c = year_counts[y]
        bar = "█" * int(c / max_count * 15)
        lines.append(f"| {y} | {c} | {bar} |")

    total = sum(year_counts[y] for y in sorted_years)
    lines.append(f"\n共计 {total} 篇（{sorted_years[0]}—{sorted_years[-1]}）")

    if len(sorted_years) >= 2:
        first, last = year_counts[sorted_years[0]], year_counts[sorted_years[-1]]
        if last > first:
            lines.append(f"📈 趋势：**上升**（{sorted_years[0]}年 {first} 篇 → {sorted_years[-1]}年 {last} 篇）")
        elif last < first:
            lines.append(f"📉 趋势：**下降**（{sorted_years[0]}年 {first} 篇 → {sorted_years[-1]}年 {last} 篇）")
        else:
            lines.append("📊 趋势：**平稳**")

    return "\n".join(lines)


TOOL_DEFINITION = {
    "name": "research_trend",
    "description": "分析某个研究方向在 Arxiv 上的论文数量年度趋势，生成趋势表格和走势分析。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "研究方向关键词，例如 'large language model' 或 'diffusion model'",
            },
            "years": {
                "type": "integer",
                "description": "统计最近几年，默认 5",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    "func": research_trend,
}
