"""Shared test fixtures and configuration.

Installs a torch mock (if real torch is unavailable) so CI can run
all unit tests without PyTorch installed.

Also disables the Zenoh mesh by default during the test suite so the
``Robot()`` / ``Simulation()`` factory does not spin up real Zenoh
sessions and background heartbeat threads when ``eclipse-zenoh`` is
installed in the test environment.  Mesh-specific tests opt back in
explicitly via ``monkeypatch.delenv`` or by patching ``init_mesh``.
"""

import os

# Disable mesh BEFORE any strands_robots import below pulls in robot.py.
# Use setdefault so tests that explicitly enable the mesh (e.g. integ tests)
# can override via the environment without conftest stomping on them.
os.environ.setdefault("STRANDS_MESH", "false")

from tests.mocks.torch_mock import install_torch_mock

# Must run before any test imports policy modules
install_torch_mock()
