from godbot.core.session import Session, BLOB_INLINE_LIMIT


def test_small_result_inline(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("ok")
    s.append_assistant_tool_call("c1", "echo", {}, raw="{}")
    s.record_tool_result("c1", "small text")
    blob_dir = s.dir / "blobs"
    assert not list(blob_dir.glob("c1.txt"))
    msgs = s.messages_for_llm()
    assert any(m.get("content") == "small text" for m in msgs if m["role"] == "tool")


def test_large_result_spilled(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("ok")
    s.append_assistant_tool_call("c1", "echo", {}, raw="{}")
    big = "X" * (BLOB_INLINE_LIMIT + 100)
    s.record_tool_result("c1", big)
    blob = s.dir / "blobs" / "c1.txt"
    assert blob.exists()
    assert blob.read_text() == big
    msgs = s.messages_for_llm()
    tool_msg = [m for m in msgs if m["role"] == "tool"][0]
    assert "elided" in tool_msg["content"]
    assert tool_msg["content"].startswith("X" * 100)


def test_read_blob(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("ok")
    s.append_assistant_tool_call("c1", "echo", {}, raw="{}")
    s.record_tool_result("c1", "Y" * (BLOB_INLINE_LIMIT + 50))
    blob_text = s.read_blob("c1")
    assert blob_text.startswith("Y") and len(blob_text) > BLOB_INLINE_LIMIT
