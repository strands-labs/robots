import importlib.util

import pytest

from strands_robots.simulation import Simulation

# The agent-turn tests need strands (the [tools] extra); without it they are not collected.
collect_ignore_glob = (
    []
    if importlib.util.find_spec("strands")
    else ["tools/test_agent_turn.py", "tools/test_sim_tool.py", "tools/test_tool_gate.py"]
)


@pytest.fixture(autouse=True)
def _fresh_default_world():
    """Every test starts without a process-global default world and leaves none behind."""
    if Simulation._default is not None:
        Simulation._default.close()
    Simulation._default = None
    yield
    if Simulation._default is not None:
        Simulation._default.close()
    Simulation._default = None
