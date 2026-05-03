from __future__ import annotations
import re
import httpx

from godbot.core.registry import tool


def _strip_html(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


@tool(timeout=30)
def web_fetch(url: str, max_chars: int = 20000) -> str:
    """Fetch a URL and return cleaned text (HTML tags stripped)."""
    try:
        r = httpx.get(url, timeout=20.0, follow_redirects=True, headers={"User-Agent": "GodBot/0.1"})
        r.raise_for_status()
    except Exception as e:
        return f"[error] fetch failed: {e}"
    ctype = r.headers.get("content-type", "")
    body = r.text
    if "html" in ctype:
        body = _strip_html(body)
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n... [{len(r.text) - max_chars} chars truncated]"
    return body


@tool(timeout=30)
def web_search(query: str, max_results: int = 8) -> str:
    """Web search via DuckDuckGo HTML endpoint. Returns title + url + snippet per result."""
    try:
        r = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20.0,
        )
        r.raise_for_status()
    except Exception as e:
        return f"[error] search failed: {e}"
    items = re.findall(
        r'<a class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?<a class="result__snippet"[^>]*>(.*?)</a>',
        r.text,
        flags=re.S,
    )
    if not items:
        return "(no results — DuckDuckGo HTML endpoint may have changed)"
    out = []
    for url, title, snippet in items[:max_results]:
        out.append(f"- {_strip_html(title)}\n  {url}\n  {_strip_html(snippet)}")
    return "\n".join(out)
