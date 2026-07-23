"""Pin the optional-dependency degradation contract of ``register_pack_state_step``.

``strands_robots.policies.lerobot_local.embodiment.register_pack_state_step``
defines and registers the ``strands_pack_state`` processor step against
lerobot's processor framework. Its documented contract is to return ``None``
(rather than raise) when that framework is unavailable, so a light install that
lacks lerobot's ``processor.pipeline`` module still imports and operates without
crashing. The full processor stack is always present in the test environment,
so this graceful-degradation branch is otherwise never exercised; this test
forces the import to fail and asserts the fail-soft contract.
"""

import logging
import sys

from strands_robots.policies.lerobot_local.embodiment import register_pack_state_step


def test_register_pack_state_step_returns_none_when_processor_framework_absent(monkeypatch, caplog):
    """A missing ``lerobot.processor.pipeline`` degrades to ``None``, never a raise."""
    # Force the in-function ``from lerobot.processor.pipeline import ...`` to
    # raise ImportError. setitem(..., None) makes Python's import machinery
    # treat the submodule as unimportable; monkeypatch reverts it after the test.
    monkeypatch.setitem(sys.modules, "lerobot.processor.pipeline", None)

    with caplog.at_level(logging.DEBUG, logger="strands_robots.policies.lerobot_local.embodiment"):
        result = register_pack_state_step()

    assert result is None
    assert any("processor framework unavailable" in record.getMessage() for record in caplog.records)
