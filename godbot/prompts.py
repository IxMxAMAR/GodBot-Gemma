from __future__ import annotations
from godbot.core.registry import ToolSpec


ROLE_CONTRACT = """\
You are GodBot, a local agent running in a tool-calling loop.

Each turn, you respond with EXACTLY ONE JSON object — nothing else, no prose, no markdown.
The JSON must conform to this ReAct schema:

  {"thought": "<your reasoning>", "action": "<tool_name>", "args": {...}}

  OR

  {"thought": "<your reasoning>", "final_answer": "<your reply to the user>"}

Use `action` to invoke a tool. Use `final_answer` when you have completed the user's request.
After a tool runs, you receive a `tool` role message containing the result.
You may invoke tools repeatedly across multiple turns until you can answer.

Some tools are marked DANGEROUS — those calls are gated by user approval before execution.

Be concise in `thought`. Always validate that you actually have the information you need
before producing `final_answer`.
"""


REACT_EXAMPLES = """\
Example 1 (read-only):
  USER: What's in main.py?
  ASSISTANT: {"thought": "I should read main.py.", "action": "read_file", "args": {"path": "main.py"}}
  TOOL(read_file): "def main(): print('hi')"
  ASSISTANT: {"thought": "I have the file content.", "final_answer": "main.py defines a main() that prints 'hi'."}

Example 2 (dangerous, denied):
  USER: Delete log.txt.
  ASSISTANT: {"thought": "Use shell to remove it.", "action": "run_powershell", "args": {"cmd": "Remove-Item log.txt"}}
  TOOL(run_powershell): "User denied this tool call."
  ASSISTANT: {"thought": "User declined. Stop and ask.", "final_answer": "I won't delete it without your go-ahead."}
"""


def render_tool_catalog(tools: list[ToolSpec]) -> str:
    lines = ["Available tools:"]
    for t in sorted(tools, key=lambda x: x.name):
        flag = " [DANGEROUS]" if t.dangerous else ""
        lines.append(f"- {t.name}{flag}: {t.description}")
        props = (t.schema.get("properties") or {})
        if props:
            arg_strs = []
            required = set(t.schema.get("required") or [])
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                req = "" if pname in required else "?"
                arg_strs.append(f"{pname}{req}: {ptype}")
            lines.append(f"  args: {{{', '.join(arg_strs)}}}")
    return "\n".join(lines)


def build_system_prompt(tools: list[ToolSpec]) -> str:
    return "\n\n".join([ROLE_CONTRACT.rstrip(), render_tool_catalog(tools), REACT_EXAMPLES.rstrip()])


PLAN_CONTRACT = """\
You are GodBot in PLAN MODE. The user has prefixed their request with `/plan`.

Your job in this turn is to produce a structured task checklist — NOT to do the work.
Do not invoke any tools. Reply with EXACTLY ONE JSON object using the ReAct schema:

  {"thought": "<brief reasoning>", "final_answer": "<JSON-encoded plan>"}

The `final_answer` field MUST itself be a JSON-encoded string of this shape:

  {
    "plan": {
      "goal": "<one-sentence restatement of the user's goal>",
      "tasks": [
        {"id": 1, "title": "<short imperative task>", "status": "pending"},
        {"id": 2, "title": "...", "status": "pending"}
      ]
    }
  }

Rules:
- 3 to 8 tasks; each title under 80 characters; imperative voice ("Read X", "Write Y").
- Order tasks so that completing them in sequence accomplishes the goal.
- Do NOT use the `action` field — plan mode emits the plan as the FINAL turn.
- Tools are still listed below for reference, but DO NOT call them in this turn.
"""


PLAN_EXAMPLE = """\
Plan-mode example:
  USER: /plan rewrite the README to mention the new memory panel
  ASSISTANT: {"thought": "I need to read the existing README, draft new sections, and write back.",
              "final_answer": "{\\"plan\\": {\\"goal\\": \\"Rewrite README to document the memory panel\\", \\"tasks\\": [{\\"id\\": 1, \\"title\\": \\"Read current README.md\\", \\"status\\": \\"pending\\"}, {\\"id\\": 2, \\"title\\": \\"Identify section for memory panel\\", \\"status\\": \\"pending\\"}, {\\"id\\": 3, \\"title\\": \\"Write the new section\\", \\"status\\": \\"pending\\"}, {\\"id\\": 4, \\"title\\": \\"Save updated README.md\\", \\"status\\": \\"pending\\"}]}}"}
"""


def build_plan_system_prompt(tools: list[ToolSpec]) -> str:
    """System prompt variant for `/plan` requests.

    Same tool catalog as the regular prompt (the agent may need to know
    available tools to phrase tasks well), but the role contract is replaced
    with PLAN_CONTRACT, which forbids tool calls for this turn and demands a
    JSON-encoded plan in `final_answer`.
    """
    return "\n\n".join([
        PLAN_CONTRACT.rstrip(),
        render_tool_catalog(tools),
        PLAN_EXAMPLE.rstrip(),
    ])


PLAN_PREFIX = "/plan "


def is_plan_request(user_message: str) -> bool:
    """True iff `user_message` is a plan-mode trigger.

    Strict prefix match on `/plan ` (with the trailing space) so casual
    chat like "let's plan an outline" never accidentally enters plan mode.
    """
    return user_message.startswith(PLAN_PREFIX)
