"""网页内容抓取工具。"""

import httpx


def fetch_webpage(url: str) -> str:
    """抓取网页并提取纯文本内容。"""
    try:
        headers = {"User-Agent": "ResearchAssistant/1.0"}
        resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
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
