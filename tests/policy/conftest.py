import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

pytest.importorskip("lerobot.remote_inference", reason="needs the lerobot extra (Python >= 3.12)")


def free_endpoint() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp/127.0.0.1:{s.getsockname()[1]}"


@pytest.fixture(scope="session")
def tiny_act(tmp_path_factory):
    from tiny_act import save_tiny_act

    return str(save_tiny_act(tmp_path_factory.mktemp("ckpt") / "tiny-act-so101", 6))


@pytest.fixture(scope="session")
def tiny_act_wrist(tmp_path_factory):
    from tiny_act import save_tiny_act

    return str(save_tiny_act(tmp_path_factory.mktemp("ckpt") / "tiny-act-so101-wrist", 6, camera="wrist"))


@pytest.fixture
def deployment(tiny_act):
    """A real LeRobot policy server for the so101 tiny ACT, in this process; yields (url, cfg)."""
    from strands_robots import Description
    from strands_robots.policy.lerobot import serve, server_config

    def make(name="tiny", checkpoint=None, **kw):
        cfg = server_config(
            Description.load("so101"),
            checkpoint or tiny_act,
            deployment=name,
            endpoint=free_endpoint(),
            device="cpu",
            **kw,
        )
        return cfg, serve(cfg)

    return make
