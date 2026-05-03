from godbot.core.registry import Registry
from godbot.prompts import build_system_prompt, ROLE_CONTRACT, render_tool_catalog


def test_role_contract_mentions_react():
    assert "react" in ROLE_CONTRACT.lower() or "json" in ROLE_CONTRACT.lower()
    assert "final_answer" in ROLE_CONTRACT


def test_render_tool_catalog():
    reg = Registry()

    @reg.tool()
    def read_file(path: str) -> str:
        """Read a UTF-8 file."""
        return ""

    @reg.tool(dangerous=True)
    def write_file(path: str, content: str) -> str:
        """Write UTF-8 content to a file."""
        return ""

    catalog = render_tool_catalog(reg.all())
    assert "read_file" in catalog
    assert "Read a UTF-8 file." in catalog
    assert "write_file" in catalog
    assert "DANGEROUS" in catalog or "dangerous" in catalog


def test_build_system_prompt_contains_all_parts():
    reg = Registry()

    @reg.tool()
    def echo(text: str) -> str:
        """Echo text."""
        return text

    prompt = build_system_prompt(reg.all())
    assert ROLE_CONTRACT.split("\n")[0] in prompt
    assert "echo" in prompt
    assert "thought" in prompt.lower()
    assert "final_answer" in prompt
