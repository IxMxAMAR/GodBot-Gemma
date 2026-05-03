import json
import pytest
from godbot.core.events import (
    TokenEvent, ToolCallEvent, ToolResultEvent, GateEvent, ErrorEvent, DoneEvent,
)
from godbot.interfaces.discord_bot.rendering import (
    TokenAccumulator, build_tool_call_embed, build_tool_result_update,
    build_gate_embed, build_error_embed, extract_final_answer,
    GROUP_COLORS, group_for,
)


def test_group_for_known_tools():
    assert group_for("read_file") == "fs"
    assert group_for("run_powershell") == "shell"
    assert group_for("web_fetch") == "web"
    assert group_for("run_python") == "python"
    assert group_for("save_note") == "memory"
    assert group_for("todo_set") == "task"
    assert group_for("search_knowledge") == "rag"
    assert group_for("anything_else") == "default"


def test_group_colors_are_int():
    for color in GROUP_COLORS.values():
        assert isinstance(color, int)


def test_token_accumulator_first_chunk_emits_immediately():
    acc = TokenAccumulator(throttle_ms=750, char_threshold=400)
    text, should_emit = acc.feed("hello")
    assert text == "hello"
    assert should_emit is True


def test_token_accumulator_throttles_subsequent_chunks():
    acc = TokenAccumulator(throttle_ms=750, char_threshold=400, now_ms_fn=lambda: 1000)
    acc.feed("a")  # initial flush at t=1000
    acc.now_ms_fn = lambda: 1100
    text, should_emit = acc.feed("b")
    assert text == "ab"
    assert should_emit is False  # too soon


def test_token_accumulator_flushes_after_throttle_ms():
    times = iter([1000, 1100, 1900])
    acc = TokenAccumulator(throttle_ms=750, char_threshold=400, now_ms_fn=lambda: next(times))
    acc.feed("a")  # t=1000 — emit
    acc.feed("b")  # t=1100 — no
    text, should_emit = acc.feed("c")  # t=1900 — yes (>=1750)
    assert text == "abc"
    assert should_emit is True


def test_token_accumulator_flushes_on_char_threshold():
    acc = TokenAccumulator(throttle_ms=10000, char_threshold=10, now_ms_fn=lambda: 1000)
    acc.feed("xx")  # initial — emit
    acc.feed("yy")  # 4 chars new — no
    _, should_emit = acc.feed("z" * 11)  # crosses 10-char threshold
    assert should_emit is True


def test_extract_final_answer_from_react_json():
    raw = json.dumps({"thought": "x", "final_answer": "Hello, World."})
    assert extract_final_answer(raw) == "Hello, World."


def test_extract_final_answer_returns_raw_on_parse_failure():
    raw = "not json at all"
    assert extract_final_answer(raw) == raw


def test_extract_final_answer_returns_raw_when_only_action():
    raw = json.dumps({"thought": "x", "action": "echo", "args": {}})
    # No final_answer present — return the raw text so the user sees something
    assert extract_final_answer(raw) == raw


def test_build_tool_call_embed_has_color_and_title():
    ev = ToolCallEvent(id="c1", name="read_file", args={"path": "main.py"})
    emb = build_tool_call_embed(ev)
    assert "read_file" in emb["title"]
    assert emb["color"] == GROUP_COLORS["fs"]
    assert "main.py" in emb["description"]
    assert emb["footer"]["text"].lower().startswith("running")


def test_build_tool_result_update_attaches_to_embed():
    ev = ToolResultEvent(id="c1", preview="412 lines", blob=None, duration_ms=12)
    update = build_tool_result_update(ev)
    assert "12ms" in update["footer"]["text"]
    assert "412 lines" in update["fields"][0]["value"]


def test_build_tool_result_update_with_blob_mentions_it():
    ev = ToolResultEvent(id="c1", preview="x", blob="c1", duration_ms=5)
    update = build_tool_result_update(ev)
    assert "blob" in update["footer"]["text"].lower()


def test_build_gate_embed_yellow_warning():
    ev = GateEvent(id="c1", name="run_powershell", args={"cmd": "ls"})
    emb = build_gate_embed(ev)
    assert "Approve" in emb["title"]
    assert "run_powershell" in emb["title"]
    assert emb["color"] == 0xFFFF00 or emb["color"] == 0xFFA500


def test_build_error_embed_red():
    emb = build_error_embed(ErrorEvent(message="oops"))
    assert "oops" in emb["description"]
    assert emb["color"] == 0xEF4444 or emb["color"] == 0xFF0000
