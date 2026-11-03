"""A strands Model that plays back a fixed list of tool calls, then says "done" (no network).

An entry that is a list of calls is one assistant message holding all of them: strands runs those at once.
"""

from __future__ import annotations

import json
from typing import Any

from strands.models.model import Model

Call = tuple[str, dict[str, Any]]


class ScriptedModel(Model):
    def __init__(self, calls: list[Call | list[Call]]) -> None:
        self.calls = list(calls)
        self.seen: list[Any] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Any:
        return {}

    async def structured_output(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
        yield

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kw: Any) -> Any:
        self.seen.append(tool_specs)
        yield {"messageStart": {"role": "assistant"}}
        if self.calls:
            turn = self.calls.pop(0)
            for i, (name, inp) in enumerate(turn if isinstance(turn, list) else [turn]):
                tid = f"t{len(self.seen)}" + (f"_{i}" if i else "")
                yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tid, "name": name}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(inp)}}}}
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
