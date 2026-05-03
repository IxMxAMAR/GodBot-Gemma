from godbot.core.rag import chunk_text, chunk_markdown


def test_chunk_text_basic():
    text = "\n".join(f"line {i}" for i in range(120))
    chunks = chunk_text(text, window=40, overlap=5)
    # 120 lines -> 0..40, 35..75, 70..110, 105..120
    assert len(chunks) >= 3
    first_lines = chunks[0]["content"].splitlines()
    assert first_lines[0] == "line 0"
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 40


def test_chunk_markdown_by_heading():
    md = "# Title\n\nintro\n\n## Section A\n\nbody A\n\n## Section B\n\nbody B\n"
    chunks = chunk_markdown(md)
    titles = [c.get("heading") for c in chunks]
    assert "Title" in titles or "Section A" in titles
    assert any("body A" in c["content"] for c in chunks)
