from godbot.core.session import Session


def test_append_user_message(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hello")
    msgs = s.messages_for_llm()
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_append_assistant_and_tool(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    s.append_assistant_tool_call("c1", "echo", {"text": "x"}, raw='{"thought":"t","action":"echo","args":{"text":"x"}}')
    s.append_tool_result("c1", "x")
    msgs = s.messages_for_llm()
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "echo"
    assert msgs[1]["tool_calls"][0]["id"] == "c1"
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "x"}


def test_append_assistant_final(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hi")
    s.append_assistant_final("hello back")
    assert s.messages_for_llm()[-1] == {"role": "assistant", "content": "hello back"}


def test_replay_after_load(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("u1")
    s.append_assistant_final("a1")
    s.append_user("u2")
    sid = s.id
    loaded = Session.load(root=tmp_path, session_id=sid)
    msgs = loaded.messages_for_llm()
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert [m["content"] for m in user_msgs] == ["u1", "u2"]


def test_truncated_final_line_tolerated(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("ok")
    p = s.dir / "events.jsonl"
    p.write_text(p.read_text() + '{"type":"user","content":"hal\n', encoding="utf-8")
    loaded = Session.load(root=tmp_path, session_id=s.id)
    msgs = loaded.messages_for_llm()
    assert any(m["role"] == "user" and m["content"] == "ok" for m in msgs)
