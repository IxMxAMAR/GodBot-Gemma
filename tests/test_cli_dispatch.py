from godbot.interfaces.cli import handle_slash


def test_slash_exit_returns_quit():
    assert handle_slash("/exit", state={}) == ("quit", None)


def test_slash_yolo_toggles():
    state = {"yolo": False}
    cmd, _ = handle_slash("/yolo", state)
    assert cmd == "ok"
    assert state["yolo"] is True
    handle_slash("/yolo", state)
    assert state["yolo"] is False


def test_slash_unknown_returns_unknown():
    out = handle_slash("/whatever", state={})
    assert out[0] == "unknown"


def test_non_slash_passes_through():
    out = handle_slash("just a message", state={})
    assert out == ("message", "just a message")
