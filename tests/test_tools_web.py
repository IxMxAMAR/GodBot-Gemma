import respx
import httpx
import pytest
import godbot.tools
from godbot.core.registry import DEFAULT


@respx.mock
def test_web_fetch_returns_text():
    respx.get("https://example.com/").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/html"}, text="<html><body>hello</body></html>"
        )
    )
    out = DEFAULT.execute("web_fetch", {"url": "https://example.com/"})
    assert "hello" in out


@respx.mock
def test_web_fetch_strips_html_tags():
    respx.get("https://x.test/").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/html"},
            text="<html><head><script>x</script></head><body><p>visible</p></body></html>"
        )
    )
    out = DEFAULT.execute("web_fetch", {"url": "https://x.test/"})
    assert "visible" in out
    assert "<script>" not in out


@respx.mock
def test_web_search_parses_ddg_html():
    body = """
    <div class="result"><a class="result__a" href="https://a.test/">A title</a><a class="result__snippet">snip A</a></div>
    <div class="result"><a class="result__a" href="https://b.test/">B title</a><a class="result__snippet">snip B</a></div>
    """
    respx.post("https://html.duckduckgo.com/html/").mock(
        return_value=httpx.Response(200, text=body)
    )
    out = DEFAULT.execute("web_search", {"query": "test"})
    assert "A title" in out and "https://a.test/" in out
    assert "B title" in out
