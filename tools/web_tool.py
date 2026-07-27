"""网页内容抓取工具。"""

import httpx

from utils.url_safety import validate_public_url


def fetch_webpage(url: str) -> str:
    """抓取网页并提取纯文本内容。"""
    try:
        safe_url = validate_public_url(url)
    except ValueError as e:
        return f"抓取网页出错：不安全的 URL（{e}）"

    try:
        headers = {"User-Agent": "ResearchAssistant/1.0"}
        # 不自动跟随重定向，避免跳转到内网地址绕过校验
        resp = httpx.get(safe_url, headers=headers, timeout=30, follow_redirects=False)
        if resp.is_redirect:
            location = resp.headers.get("location", "")
            try:
                # 相对重定向相对原 URL 解析
                next_url = str(resp.url.join(location)) if location else ""
                validate_public_url(next_url)
            except ValueError as e:
                return f"抓取网页出错：重定向目标不安全（{e}）"
            resp = httpx.get(next_url, headers=headers, timeout=30, follow_redirects=False)
            # 最多再跟一步已校验的跳转；多层跳转拒绝，降低绕过面
            if resp.is_redirect:
                return "抓取网页出错：重定向次数过多，已拒绝"
        resp.raise_for_status()
    except Exception as e:
        return f"抓取网页出错：{e}"

    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.texts = []
            self._skip = False
            self._skip_tags = {"script", "style", "nav", "footer", "header"}

        def handle_starttag(self, tag, attrs):
            if tag in self._skip_tags:
                self._skip = True

        def handle_endtag(self, tag):
            if tag in self._skip_tags:
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                text = data.strip()
                if text:
                    self.texts.append(text)

    extractor = TextExtractor()
    extractor.feed(resp.text)
    content = "\n".join(extractor.texts)

    if len(content) > 3000:
        content = content[:3000] + "\n...(内容已截断)"

    return content if content else "未能提取到有效内容。"


TOOL_DEFINITION = {
    "name": "fetch_webpage",
    "description": "抓取指定 URL 的网页内容，提取并返回纯文本。可用于阅读在线论文、文档等。",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的网页 URL",
            }
        },
        "required": ["url"],
    },
    "func": fetch_webpage,
}
