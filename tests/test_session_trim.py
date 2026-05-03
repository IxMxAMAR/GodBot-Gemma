from godbot.core.session import Session, estimate_tokens


def test_estimate_tokens_simple():
    assert estimate_tokens("hello world") == len("hello world") // 4


def test_no_trim_under_limit(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    s.append_user("hello")
    msgs = s.messages_for_llm(max_context=10000)
    assert len(msgs) == 1


def test_trim_drops_oldest_when_over_limit(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    big = "x" * 4000
    for _ in range(20):
        s.append_user(big)
        s.append_assistant_final(big)
    msgs = s.messages_for_llm(max_context=5000)
    assert any(m["role"] == "system" and "elided" in m.get("content", "") for m in msgs)
    assert msgs[-1]["content"] == big


def test_trim_keeps_most_recent_pair(tmp_path):
    s = Session.create(root=tmp_path, model="m")
    for i in range(10):
        s.append_user(f"u{i}: " + "x" * 4000)
        s.append_assistant_final(f"a{i}: " + "x" * 4000)
    msgs = s.messages_for_llm(max_context=2000)
    last_user = [m for m in msgs if m["role"] == "user"][-1]["content"]
    last_assistant = [m for m in msgs if m["role"] == "assistant"][-1]["content"]
    assert last_user.startswith("u9")
    assert last_assistant.startswith("a9")
