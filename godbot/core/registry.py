from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional
from pydantic import create_model


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    schema: dict[str, Any]
    dangerous: bool
    timeout: int


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    fields: dict[str, tuple] = {}
    for pname, param in sig.parameters.items():
        ann = str if param.annotation is inspect.Parameter.empty else param.annotation
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (ann, default)
    Model = create_model(fn.__name__ + "Args", **fields)
    schema = Model.model_json_schema()
    schema.setdefault("type", "object")
    schema.setdefault("additionalProperties", False)
    return schema


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(self, *, dangerous: bool = False, timeout: int = 60):
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            doc = (fn.__doc__ or "").strip()
            if not doc:
                raise ValueError(f"tool {fn.__name__} requires a docstring")
            description = doc.splitlines()[0].strip()
            spec = ToolSpec(
                name=fn.__name__,
                description=description,
                fn=fn,
                schema=_schema_from_signature(fn),
                dangerous=dangerous,
                timeout=timeout,
            )
            self._tools[spec.name] = spec
            return fn
        return deco

    def spec(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def subset(self, names: Optional[list[str]]) -> list[ToolSpec]:
        """Return ToolSpecs for the listed names, or all tools if names is None."""
        if names is None:
            return self.all()
        wanted = set(names)
        return [t for t in self._tools.values() if t.name in wanted]

    def is_dangerous(self, name: str) -> bool:
        s = self._tools.get(name)
        return bool(s and s.dangerous)

    def timeout_for(self, name: str) -> int:
        s = self._tools.get(name)
        return s.timeout if s else 60

    def validate_args(self, name: str, args: dict[str, Any]) -> Optional[str]:
        spec = self._tools.get(name)
        if spec is None:
            return f"unknown tool {name!r}"
        try:
            import jsonschema
            jsonschema.validate(args, spec.schema)
            return None
        except Exception as e:
            return str(getattr(e, "message", e))

    def execute(self, name: str, args: dict[str, Any]) -> str:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(name)
        result = spec.fn(**args)
        if isinstance(result, str):
            return result
        return str(result)


# Module-level default registry — used by tools/ auto-discovery.
DEFAULT = Registry()
tool = DEFAULT.tool
