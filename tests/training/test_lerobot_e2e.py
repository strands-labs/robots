"""End-to-end: LerobotTrainer trains a real ACT checkpoint from a recorded
LeRobotDataset, and the result is loadable via create_policy.

Slow + requires lerobot + mujoco. Skipped automatically if either is absent.
Runs CPU-only, 2 steps - just enough to prove the record->train->load loop.

Hermetic: this trains a real policy inside the *unit* suite, which is the one
required status check gating every pull request, so anything it fetches becomes a
merge dependency on a third party answering. The loop reaches the network for
nothing, and :func:`_record_outbound_connects` holds it to that.
"""

import os
import socket
import sys
from typing import Any

import pytest

lerobot = pytest.importorskip("lerobot")
pytest.importorskip("mujoco")

from strands_robots.training import TrainSpec, create_trainer  # noqa: E402


def _record_outbound_connects(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every non-loopback peer a socket is connected to from here on.

    Records rather than blocks: a refusal would surface as whatever error the
    caller makes of it, while the recorded address names the host and lets the
    failure say which dependency was contacted.

    Args:
        monkeypatch: Pytest fixture; restores the real ``connect`` on teardown.

    Returns:
        The list the spy appends peer addresses to, empty until something dials.
    """
    reached: list[Any] = []
    real = socket.socket.connect

    def spy(self: socket.socket, address: Any) -> Any:
        if not (isinstance(address, tuple) and str(address[0]).startswith(("127.", "::1", "localhost"))):
            reached.append(address)
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", spy)
    return reached


@pytest.fixture(scope="module")
def recorded_dataset(tmp_path_factory):
    """Record a tiny dataset in MuJoCo sim (one short episode)."""
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    from strands_robots import MockPolicy, Robot

    root = str(tmp_path_factory.mktemp("e2e_ds"))
    sim = Robot("so100", mesh=False)
    sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])
    start = sim.start_recording(
        repo_id="local/e2e",
        root=root,
        # Must equal the rollout's control_frequency (run_policy's default
        # 50.0 Hz below): the recorder captures one frame per control step, so
        # a differing rate would only mislabel every recorded timestamp.
        fps=50,
        task="pick up the red cube",
        overwrite=True,
    )
    assert start["status"] == "success", start
    sim.run_policy(
        robot_name="so100",
        policy_object=MockPolicy(),
        instruction="pick up the red cube",
        n_steps=20,
    )
    stop = sim.stop_recording()
    assert stop["status"] == "success", stop
    assert os.path.isfile(os.path.join(root, "meta", "info.json"))
    return root


@pytest.mark.slow
def test_record_train_load_loop(recorded_dataset, tmp_path, monkeypatch):
    reached = _record_outbound_connects(monkeypatch)
    out = str(tmp_path / "e2e_out")
    trainer = create_trainer("lerobot_local", device="cpu")

    spec = TrainSpec(
        dataset_root=recorded_dataset,
        base_model="",  # ACT from scratch - smallest CPU path
        output_dir=out,
        steps=2,
        save_freq=2,
        global_batch_size=2,
        extra={
            "policy_type": "act",
            "num_workers": 0,
            # ACT's config defaults its resnet18 backbone to ImageNet weights,
            # which is a 47 MB fetch from an external CDN on every run of this
            # suite. ``base_model=""`` above already asks for ACT from scratch,
            # and two steps prove the record->train->load seam whether or not
            # the backbone carries ImageNet priors, so the download buys the
            # assertions below nothing and makes them depend on that CDN.
            "policy.pretrained_backbone_weights": None,
        },
    )

    assert trainer.validate(spec) == []

    result = trainer.train(spec)
    assert result.status == "success", result.message
    assert result.checkpoint_dir and os.path.isdir(result.checkpoint_dir)
    assert os.path.isfile(os.path.join(result.checkpoint_dir, "model.safetensors"))

    # export() returns a create_policy-loadable path (default passthrough).
    exported = trainer.export(spec, result.checkpoint_dir)
    assert os.path.isdir(exported)

    # The trained checkpoint loads back as a Policy - loop closed.
    os.environ.setdefault("STRANDS_TRUST_REMOTE_CODE", "1")
    from strands_robots import create_policy

    policy = create_policy(exported, device="cpu")
    assert policy.provider_name == "lerobot_local"

    # Nothing above needs a third party: the dataset is local and the policy
    # trains from scratch. A reached host here is a merge gate that can go red
    # for a pull request that changed none of this.
    assert not reached, (
        f"the record->train->load loop reached {len(reached)} external host(s) {reached}; "
        f"this suite is the required check for every pull request, so a fetch here makes "
        f"the merge gate depend on that host answering"
    )
