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


@tool(timeout=15)
def http_status(url: str) -> str:
    """HEAD an URL to check liveness without downloading the body (sub-project 66).

    Reports the HTTP status code, content-type, content-length, and
    final URL after redirects. Useful for "is this endpoint up?",
    "is this download still available?", or pre-flight checks before
    a heavier ``web_fetch``.

    Falls back to a small GET if the server rejects HEAD (some CDNs
    don't support it). Returns ``[error] ...`` on transport failure.
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=10.0,
                          headers={"User-Agent": "GodBot/0.1"}) as c:
            try:
                r = c.head(url)
                if r.status_code == 405:  # method not allowed → small GET
                    raise ValueError("head_not_supported")
            except (httpx.HTTPError, ValueError):
                r = c.get(url, headers={"Range": "bytes=0-0"})
    except Exception as e:
        return f"[error] fetch failed: {type(e).__name__}: {e}"
    parts = [
        f"status: {r.status_code}",
        f"final_url: {r.url}",
    ]
    ctype = r.headers.get("content-type")
    if ctype:
        parts.append(f"content-type: {ctype}")
    clen = r.headers.get("content-length")
    if clen:
        parts.append(f"content-length: {clen}")
    return "\n".join(parts)


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
