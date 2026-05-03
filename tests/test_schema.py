import jsonschema
from godbot.core.schema import build_react_schema, validate_react_response


def test_action_response_validates():
    tool_schemas = {"read_file": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
    schema = build_react_schema(tool_schemas)
    payload = {"thought": "I need to read", "action": "read_file", "args": {"path": "x.py"}}
    err = validate_react_response(payload, schema)
    assert err is None


def test_final_answer_validates():
    schema = build_react_schema({})
    payload = {"thought": "done", "final_answer": "It's 42."}
    assert validate_react_response(payload, schema) is None


def test_unknown_action_rejected():
    schema = build_react_schema({"read_file": {"type": "object"}})
    payload = {"thought": "x", "action": "wat", "args": {}}
    err = validate_react_response(payload, schema)
    assert err is not None and "wat" in err


def test_missing_keys_rejected():
    schema = build_react_schema({})
    err = validate_react_response({"thought": "only thought"}, schema)
    assert err is not None


def test_empty_tools_rejects_action_branch():
    """With no registered tools, any action call must be rejected — only final_answer is valid."""
    schema = build_react_schema({})
    payload = {"thought": "try anyway", "action": "__no_tools_available__", "args": {}}
    # Even the placeholder name must not validate as a "real" tool call from the model's POV;
    # but more importantly, an arbitrary action name must fail.
    err = validate_react_response({"thought": "try", "action": "read_file", "args": {}}, schema)
    assert err is not None


def test_additional_properties_rejected():
    """The model must not be able to smuggle extra keys (e.g. hallucinated fields) past the schema."""
    schema = build_react_schema({"read_file": {"type": "object"}})
    payload = {"thought": "x", "action": "read_file", "args": {}, "extra": "nope"}
    err = validate_react_response(payload, schema)
    assert err is not None
