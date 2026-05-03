import pytest
from godbot.core.registry import tool, Registry


def test_tool_decorator_extracts_metadata():
    reg = Registry()

    @reg.tool()
    def echo(text: str, repeat: int = 1) -> str:
        """Repeat text."""
        return text * repeat

    spec = reg.spec("echo")
    assert spec.name == "echo"
    assert spec.description == "Repeat text."
    assert spec.dangerous is False
    assert spec.timeout == 60
    assert spec.schema["type"] == "object"
    assert "text" in spec.schema["properties"]
    assert "repeat" in spec.schema["properties"]
    assert spec.schema["required"] == ["text"]
    assert spec.schema["properties"]["text"]["type"] == "string"
    assert spec.schema["properties"]["repeat"]["type"] == "integer"


def test_tool_dangerous_flag():
    reg = Registry()

    @reg.tool(dangerous=True, timeout=120)
    def shoot(target: str) -> str:
        """Pew pew."""
        return "boom"

    spec = reg.spec("shoot")
    assert spec.dangerous is True
    assert spec.timeout == 120


def test_tool_missing_docstring_raises():
    reg = Registry()
    with pytest.raises(ValueError, match="docstring"):
        @reg.tool()
        def bad(x: str) -> str:
            return x


def test_tool_unknown_returns_none():
    reg = Registry()
    assert reg.spec("nope") is None


def test_tool_with_optional_arg():
    from typing import Optional
    reg = Registry()

    @reg.tool()
    def f(name: str, count: Optional[int] = None) -> str:
        """Test."""
        return name

    spec = reg.spec("f")
    # count should be in properties; whether it appears in required is up to pydantic.
    assert "count" in spec.schema["properties"]
    assert "name" in spec.schema["required"]
    assert "count" not in spec.schema["required"]


def test_tool_with_list_arg():
    reg = Registry()

    @reg.tool()
    def f(items: list[str]) -> str:
        """Test."""
        return ",".join(items)

    spec = reg.spec("f")
    assert spec.schema["properties"]["items"]["type"] == "array"


def test_execute_runs_tool_and_returns_string():
    reg = Registry()

    @reg.tool()
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    assert reg.execute("add", {"a": 2, "b": 3}) == "5"


def test_execute_unknown_tool_raises():
    reg = Registry()
    with pytest.raises(KeyError):
        reg.execute("nope", {})


def test_validate_args_ok():
    reg = Registry()

    @reg.tool()
    def f(name: str) -> str:
        """desc."""
        return name

    assert reg.validate_args("f", {"name": "x"}) is None


def test_validate_args_missing_required():
    reg = Registry()

    @reg.tool()
    def f(name: str) -> str:
        """desc."""
        return name

    err = reg.validate_args("f", {})
    assert err is not None and "name" in err


def test_execute_coerces_return_to_string():
    reg = Registry()

    @reg.tool()
    def listy(n: int) -> list:
        """desc."""
        return list(range(n))

    out = reg.execute("listy", {"n": 3})
    assert isinstance(out, str)
    assert "0" in out and "2" in out


def test_subset_none_returns_all():
    reg = Registry()

    @reg.tool()
    def a() -> str:
        """a."""
        return ""

    @reg.tool()
    def b() -> str:
        """b."""
        return ""

    assert {t.name for t in reg.subset(None)} == {"a", "b"}


def test_subset_filters():
    reg = Registry()

    @reg.tool()
    def a() -> str:
        """a."""
        return ""

    @reg.tool()
    def b() -> str:
        """b."""
        return ""

    assert {t.name for t in reg.subset(["a"])} == {"a"}
