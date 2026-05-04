from __future__ import annotations
from typing import Any, Optional
import jsonschema


def build_react_schema(tool_schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the JSON schema constraining each ReAct turn.

    Each LLM turn must emit either:
      {"thought": "...", "action": "<one_of_registered_tool_names>", "args": {...}}
    OR:
      {"thought": "...", "final_answer": "..."}

    The ``action`` enum is restricted to known tool names; if no tools are
    registered, the schema accepts only ``final_answer`` responses (the
    placeholder enum value ``"__no_tools_available__"`` can never collide with
    a real tool name registered by the caller).

    The schema is consumed by LM Studio via
    ``response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": <this>}}``;
    the consumer wraps this return value with the ``name`` field.
    """
    enum_values = list(tool_schemas.keys()) or ["__no_tools_available__"]
    action_branch = {
        "type": "object",
        "required": ["thought", "action", "args"],
        "additionalProperties": False,
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string", "enum": enum_values},
            "args": {"type": "object"},
        },
    }
    final_branch = {
        "type": "object",
        "required": ["thought", "final_answer"],
        "additionalProperties": False,
        "properties": {
            "thought": {"type": "string"},
            "final_answer": {"type": "string"},
        },
    }
    return {"oneOf": [action_branch, final_branch]}


def build_native_tool_schemas(
    tool_schemas: dict[str, dict[str, Any]],
    descriptions: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Build the OpenAI ``tools=[...]`` payload from tool argument schemas.

    Each entry has shape::

        {"type": "function", "function": {
            "name": "<tool>", "description": "<short>", "parameters": {<arg-schema>},
        }}

    The agent loop calls this when the negotiated protocol is
    :data:`godbot.core.providers.NATIVE_TOOLS`. ``descriptions`` is
    optional; missing entries fall back to the tool name.
    """
    descriptions = descriptions or {}
    out: list[dict[str, Any]] = []
    for name, schema in tool_schemas.items():
        # OpenAI's function-call schema requires ``parameters`` to be an
        # object schema. Pydantic-generated schemas already are; pass through.
        params = schema if isinstance(schema, dict) else {"type": "object"}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": descriptions.get(name, name),
                    "parameters": params,
                },
            }
        )
    return out


def validate_react_response(payload: Any, schema: dict[str, Any]) -> Optional[str]:
    """Validate a parsed JSON payload against the ReAct schema.

    Returns ``None`` on success, or a human-readable error message string on
    failure. Used as a defense-in-depth check after JSON parsing in case the
    grammar-constrained generation slips up (small models occasionally do).
    """
    try:
        jsonschema.validate(payload, schema)
        return None
    except jsonschema.ValidationError as e:
        return str(e.message)
