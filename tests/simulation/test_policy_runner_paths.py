"""Targeted coverage for ``PolicyRunner`` error paths and edge cases.

Covers:
* ``replay()`` when no robots exist (``_require_default_robot`` ValueError)
* ``replay()`` when the dataset loader raises (opaque upstream error)
* ``replay()`` rejects a non-positive / non-numeric ``speed`` before the
  dataset loader (no ZeroDivisionError, no silent full-speed playback)
* ``replay()`` when lerobot is not installed (ImportError → graceful)
* ``replay()`` with actions that have ``.numpy()`` and ``.tolist()`` methods
  (tensor-backed dataset frames)
* ``_extract_frame_ndarray`` handles render blocks without images
* ``_resolve_success_fn`` "contact" - raise (NotImplementedError),
  missing-hook, positive (n_contacts / contacts-list) detection, and the
  degradation branches: an unexpected error, a zero-contact negative, and a
  non-dict result all resolve to False without propagating
* ``_maybe_sim_time`` get_state() fallback (nested ``content[].json.sim_time``)
* ``evaluate()`` "never-succeeds" default path (no success_fn)
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.policy_runner import (
    PolicyRunner,
    _extract_frame_ndarray,
)

# Import the FakeSim from the sibling test file
from tests.simulation.test_policy_runner import FakeSim as _BaseFakeSim


class _MinimalSim(_BaseFakeSim):
    """FakeSim variant with pluggable robot list + optional get_contacts."""

    def __init__(self, robots=None, raise_on_contacts=False):
        super().__init__()
        # Override robots
        if robots is not None:
            self._robots = {name: ["j0", "j1", "j2"] for name in robots}
        self._raise_on_contacts = raise_on_contacts

    def get_contacts(self):
        if self._raise_on_contacts:
            raise NotImplementedError("backend doesn't support contacts")
        return {"n_contacts": 0}


# ── replay() error paths ────────────────────────────────────────────


def test_replay_no_robots_errors_cleanly():
    sim = _MinimalSim(robots=[])  # empty
    r = PolicyRunner(sim).replay(repo_id="irrelevant")
    assert r["status"] == "error"
    assert "No robots" in r["content"][0]["text"]


def test_replay_unknown_robot_name_errors_before_loading(monkeypatch):
    """An explicit ``robot_name`` that is not in the sim must fail fast.

    Previously replay() used ``robot_name`` verbatim without checking
    membership, so an unknown name silently "replayed" onto a phantom robot
    (send_action no-ops) and reported success - unlike run_policy/eval_policy,
    which reject unknown robots. The check must run BEFORE the dataset loader,
    so a typo does not trigger a wasted dataset download; the loader is
    monkeypatched to fail loudly if it is ever reached.
    """
    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    def _must_not_load(*args, **kwargs):
        raise AssertionError("load_lerobot_episode reached despite unknown robot")

    monkeypatch.setattr(dr, "load_lerobot_episode", _must_not_load, raising=False)

    r = PolicyRunner(sim).replay(repo_id="some/dataset", robot_name="ghost")
    assert r["status"] == "error"
    assert "ghost" in r["content"][0]["text"]
    assert "not found" in r["content"][0]["text"]


def test_replay_dataset_loader_raises_is_handled(monkeypatch):
    sim = _MinimalSim(robots=["r0"])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated HF download failure")

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", boom, raising=False)

    r = PolicyRunner(sim).replay(repo_id="bad/dataset")
    assert r["status"] == "error"
    assert "simulated HF download failure" in r["content"][0]["text"]


def test_replay_speed_zero_rejected_before_loading(monkeypatch):
    """``speed=0`` must fail fast with a structured error, not ZeroDivisionError.

    ``frame_interval = 1 / (dataset_fps * speed)`` divided by zero, raising a
    bare ``ZeroDivisionError`` that escaped replay()'s documented "returns a
    status dict" contract. The guard must run BEFORE the dataset loader so an
    invalid speed does not trigger a wasted dataset download; the loader is
    monkeypatched to fail loudly if it is ever reached.
    """
    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    def _must_not_load(*args, **kwargs):
        raise AssertionError("load_lerobot_episode reached despite invalid speed")

    monkeypatch.setattr(dr, "load_lerobot_episode", _must_not_load, raising=False)

    r = PolicyRunner(sim).replay(repo_id="some/dataset", robot_name="r0", speed=0.0)
    assert r["status"] == "error"
    assert "positive" in r["content"][0]["text"]


def test_replay_negative_speed_rejected_before_loading(monkeypatch):
    """A negative ``speed`` previously played the episode forward at full speed
    and reported success with a meaningless "Speed: -1.0x". It must be rejected
    with a structured error before the dataset loader runs.
    """
    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    def _must_not_load(*args, **kwargs):
        raise AssertionError("load_lerobot_episode reached despite invalid speed")

    monkeypatch.setattr(dr, "load_lerobot_episode", _must_not_load, raising=False)

    r = PolicyRunner(sim).replay(repo_id="some/dataset", robot_name="r0", speed=-1.0)
    assert r["status"] == "error"
    assert "positive" in r["content"][0]["text"]


def test_replay_bool_speed_rejected(monkeypatch):
    """``bool`` is an ``int`` subclass; ``True`` must not slip through as 1.0x."""
    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    def _must_not_load(*args, **kwargs):
        raise AssertionError("load_lerobot_episode reached despite bool speed")

    monkeypatch.setattr(dr, "load_lerobot_episode", _must_not_load, raising=False)

    r = PolicyRunner(sim).replay(repo_id="some/dataset", robot_name="r0", speed=True)
    assert r["status"] == "error"
    assert "positive" in r["content"][0]["text"]


def test_replay_nan_speed_rejected_before_loading(monkeypatch):
    """A non-finite ``speed`` (``nan``/``inf``) must be rejected up front.

    ``nan`` is never ``<= 0``, so it slipped past a bare ``speed <= 0`` guard
    into ``frame_interval = 1 / (dataset_fps * speed)`` (= ``nan``) and then
    ``time.sleep(nan - elapsed)``, which raises a bare ``ValueError`` deep in
    the playback loop -- breaking replay()'s documented "returns a status dict"
    contract -- while also reporting a meaningless "Speed: nanx". The guard must
    reject it before the dataset loader runs.
    """
    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    def _must_not_load(*args, **kwargs):
        raise AssertionError("load_lerobot_episode reached despite non-finite speed")

    monkeypatch.setattr(dr, "load_lerobot_episode", _must_not_load, raising=False)

    for bad in (float("nan"), float("inf")):
        r = PolicyRunner(sim).replay(repo_id="some/dataset", robot_name="r0", speed=bad)
        assert r["status"] == "error"
        assert "positive" in r["content"][0]["text"]


def test_replay_numpy_scalar_speed_accepted(monkeypatch):
    """A NumPy-scalar ``speed`` (e.g. read from a config array) is a valid
    positive rate and must not be rejected.

    ``isinstance(x, (int, float))`` is ``False`` for every NumPy scalar except
    ``np.float64``, so a ``np.float32(2.0)`` speed was wrongly rejected with
    "speed must be a positive number" even though 2.0 is valid. Once accepted,
    the scalar must be coerced to a plain ``float`` so it does not raise a
    ``TypeError`` in ``time.sleep`` on the playback path, and the returned
    ``"speed"`` status field is a plain float.
    """

    class _TinyDataset:
        fps = 30

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"action": [0.1 * idx, 0.2, 0.3]}

    def loader(repo_id, episode, root):
        return _TinyDataset(), 0, 2

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    # speed=100.0 magnitude keeps the test fast; use a NumPy float32 scalar.
    r = PolicyRunner(sim).replay(repo_id="fake/tiny", robot_name="r0", speed=np.float32(100.0))
    assert r["status"] == "success"
    reported = r["content"][1]["json"]["speed"]
    assert reported == 100.0
    assert type(reported) is float


def test_replay_with_tensor_like_actions(monkeypatch):
    """Dataset actions may be torch tensors; replay must call .numpy().tolist()."""

    class _FakeTensor:
        def __init__(self, values):
            self._v = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return self._v

    class _TensorDataset:
        fps = 30

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return {"action": _FakeTensor([0.1 * idx, 0.2, 0.3])}

    def loader(repo_id, episode, root):
        return _TensorDataset(), 0, 3

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    r = PolicyRunner(sim).replay(repo_id="fake/tensor", speed=100.0)  # fast
    assert r["status"] == "success"


def test_replay_rejects_action_vector_wider_than_action_keys(monkeypatch):
    """A recorded vector wider than the action-key map is rejected, not truncated.

    Replay's contract is fidelity: reproduce the recorded trajectory. A dataset
    with 5 action dims cannot be reproduced on a 3-actuator robot -- the last two
    recorded DOFs have no key to be written through. Truncating them (the former
    behaviour) reported ``status="success"`` with ``Frames: 2/2`` for a replay
    that dropped 40% of every commanded action, so the caller had no signal that
    the dataset and the robot did not match. ``send_action`` already rejects a
    raw action vector whose length differs from the actuator count instead of
    truncating it; replay now matches that.
    """

    class _FatDataset:
        fps = 30

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            # 5 recorded values but the robot exposes only 3 action keys.
            return {"action": [0.1, 0.2, 0.3, 0.4, 0.5]}

    def loader(repo_id, episode, root):
        return _FatDataset(), 0, 2

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    r = PolicyRunner(sim).replay(repo_id="fake/fat", speed=100.0)
    assert r["status"] == "error"
    text = r["content"][0]["text"]
    assert "5 values" in text and "3 action keys" in text
    # Nothing was applied, and the report says so instead of claiming 2/2.
    assert "Applied 0/2 frames" in text
    # No action reached the sim: the mismatch is caught before the first send.
    assert not [c for c in sim.calls if c[0] == "send_action"]


def test_replay_reads_actions_without_video_decode(monkeypatch):
    """replay() must read actions from the parquet column store, not the full
    ``__getitem__`` that decodes per-frame video.

    A real ``LeRobotDataset.__getitem__`` decodes every camera's video for the
    frame; replay only needs the recorded action vector. When the video decoder
    (torchcodec / pyav) is unavailable, ``ds[idx]`` raises a raw exception even
    though the action column is perfectly readable -- breaking replay()'s
    documented "returns a status dict" contract. replay() must prefer
    ``ds.hf_dataset`` (columns only, no video decode) and succeed.
    """

    class _VideoDecodeBroken:
        """ds[idx] raises (video decode), but the action column is readable."""

        fps = 30

        class _Columns:
            def __getitem__(self, idx):
                return {"action": [0.1 * idx, 0.2, 0.3]}

        def __init__(self):
            self.hf_dataset = self._Columns()

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            # __getitem__ failing a row lookup raises LookupError (the
            # standard exception for indexing failures); the real decoder
            # error is carried in the message. replay() catches it broadly.
            raise LookupError("Could not load libtorchcodec (video decode failed)")

    def loader(repo_id, episode, root):
        return _VideoDecodeBroken(), 0, 3

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    r = PolicyRunner(sim).replay(repo_id="fake/brokenvideo", speed=100.0)
    assert r["status"] == "success"
    assert r["content"][1]["json"]["frames_applied"] == 3


def test_replay_frame_read_failure_returns_error_dict(monkeypatch):
    """A dataset with no column store whose frame read fails must still return a
    clean status dict (not raise) per replay()'s documented contract."""

    class _NoColumnStoreBroken:
        fps = 30

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            # Failed row lookup -> LookupError (standard for __getitem__).
            raise LookupError("corrupt frame")

    def loader(repo_id, episode, root):
        return _NoColumnStoreBroken(), 0, 2

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    r = PolicyRunner(sim).replay(repo_id="fake/corrupt", speed=100.0)
    assert r["status"] == "error"
    assert "corrupt frame" in r["content"][0]["text"]


def test_replay_action_none_advances_physics(monkeypatch):
    """Dataset frames with no 'action' key → physics step, still advance."""

    class _MissingActionDataset:
        fps = 30

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"observation.state": [0, 0, 0]}  # no 'action'

    def loader(repo_id, episode, root):
        return _MissingActionDataset(), 0, 2

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    r = PolicyRunner(sim).replay(repo_id="fake/noaction", speed=100.0)
    assert r["status"] == "success"


# ── _extract_frame_ndarray edge cases ───────────────────────────────


def test_extract_frame_ndarray_rejects_non_dict():
    assert _extract_frame_ndarray("not a dict") is None
    assert _extract_frame_ndarray(None) is None


def test_extract_frame_ndarray_no_image_blocks():
    assert _extract_frame_ndarray({"content": [{"text": "only text"}]}) is None


def test_extract_frame_ndarray_bad_image_structure():
    # image present but no source
    assert _extract_frame_ndarray({"content": [{"image": "string not dict"}]}) is None
    # source empty
    assert _extract_frame_ndarray({"content": [{"image": {"source": {}}}]}) is None
    # non-decodable bytes
    assert _extract_frame_ndarray({"content": [{"image": {"source": {"bytes": b"notpng"}}}]}) is None


# ── evaluate() paths ────────────────────────────────────────────────


def test_evaluate_unknown_success_fn_string_errors():
    sim = _MinimalSim(robots=["r0"])
    policy = MockPolicy()
    r = PolicyRunner(sim).evaluate(robot_name="r0", policy=policy, n_episodes=1, success_fn="made_up_string")
    assert r["status"] == "error"
    assert "Unknown success_fn" in r["content"][0]["text"]


def test_evaluate_with_callable_success_fn():
    sim = _MinimalSim(robots=["r0"])
    policy = MockPolicy()
    policy.set_robot_state_keys(["j0", "j1", "j2"])

    # Always succeed → success_rate = 1.0
    r = PolicyRunner(sim).evaluate(
        robot_name="r0",
        policy=policy,
        n_episodes=2,
        max_steps=5,
        success_fn=lambda obs: True,
    )
    assert r["status"] == "success"


def test_evaluate_contact_fn_with_backend_that_raises():
    """If the backend's ``get_contacts`` raises NotImplementedError, the
    contact success_fn just returns False (never propagates)."""
    sim = _MinimalSim(robots=["r0"], raise_on_contacts=True)
    policy = MockPolicy()
    policy.set_robot_state_keys(["j0", "j1", "j2"])
    r = PolicyRunner(sim).evaluate(robot_name="r0", policy=policy, n_episodes=1, max_steps=3, success_fn="contact")
    assert r["status"] == "success"


def test_evaluate_none_success_fn_gives_zero_success_rate():
    """success_fn=None → never succeeds (dry-run probe)."""
    sim = _MinimalSim(robots=["r0"])
    policy = MockPolicy()
    policy.set_robot_state_keys(["j0", "j1", "j2"])
    r = PolicyRunner(sim).evaluate(robot_name="r0", policy=policy, n_episodes=2, max_steps=3, success_fn=None)
    assert r["status"] == "success"
    # success_fn=None means no episode ever succeeds
    # Extract json block:
    for c in r["content"]:
        if isinstance(c, dict) and "json" in c:
            assert c["json"]["success_rate"] == 0.0
            break


# ── _maybe_sim_time get_state() fallback ─────────────────────────────


def test_maybe_sim_time_reads_from_get_state_content_json():
    """Backends with no structured ``_world`` expose sim time via the
    status-dict ``content[].json.sim_time`` shape. ``_maybe_sim_time`` must
    dig it out of that nested block.
    """

    class _StatusDictSim(_MinimalSim):
        # No ``_world`` attr → forces the get_state() fallback path.
        _world = None

        def get_state(self):
            return {"content": [{"text": "ok"}, {"json": {"sim_time": 1.25}}]}

    t = PolicyRunner(_StatusDictSim(robots=["r0"]))._maybe_sim_time()
    assert t == 1.25


def test_maybe_sim_time_get_state_without_sim_time_returns_none():
    """A status dict with no ``sim_time`` anywhere yields None (not a crash)."""

    class _NoTimeSim(_MinimalSim):
        _world = None

        def get_state(self):
            return {"content": [{"json": {"step_count": 3}}]}

    assert PolicyRunner(_NoTimeSim(robots=["r0"]))._maybe_sim_time() is None


# ── _resolve_success_fn "contact" positive detection ─────────────────


def test_contact_success_fn_true_on_n_contacts():
    """``success_fn="contact"`` reports success when the backend's
    ``get_contacts`` returns a positive ``n_contacts`` count.
    """

    class _ContactSim(_MinimalSim):
        def get_contacts(self):
            return {"n_contacts": 2}

    sim = _ContactSim(robots=["r0"])
    fn = PolicyRunner(sim)._resolve_success_fn("contact")
    assert fn is not None
    assert fn({}) is True


def test_contact_success_fn_true_on_contacts_list():
    """``success_fn="contact"`` also accepts the ``{"contacts": [...]}`` shape."""

    class _ContactListSim(_MinimalSim):
        def get_contacts(self):
            return {"contacts": [{"a": "hand", "b": "cube"}]}

    sim = _ContactListSim(robots=["r0"])
    fn = PolicyRunner(sim)._resolve_success_fn("contact")
    assert fn({}) is True


def test_contact_success_fn_false_when_no_get_contacts():
    """When the backend has no ``get_contacts`` at all, contact detection is
    a safe no-op returning False (rather than AttributeError).
    """

    class _NoContactsSim(_MinimalSim):
        get_contacts = None

    sim = _NoContactsSim(robots=["r0"])
    fn = PolicyRunner(sim)._resolve_success_fn("contact")
    assert fn({}) is False


def test_contact_success_fn_false_on_unexpected_get_contacts_error():
    """A generic (non-NotImplementedError) error from ``get_contacts`` must
    degrade the contact probe to False, never propagate.

    The NotImplementedError case (backend genuinely lacks contact support) is
    the expected miss, but a best-effort success probe must also swallow an
    unexpected fault (a transient backend/state error) rather than aborting the
    whole evaluation. This pins the ``except Exception`` fail-safe branch.
    """

    class _BrokenContactSim(_MinimalSim):
        def get_contacts(self):
            raise RuntimeError("contact query blew up mid-episode")

    sim = _BrokenContactSim(robots=["r0"])
    fn = PolicyRunner(sim)._resolve_success_fn("contact")
    assert fn is not None
    assert fn({}) is False


def test_contact_success_fn_false_on_zero_contacts():
    """When ``get_contacts`` reports zero contacts, the probe returns False.

    A dict with ``n_contacts == 0`` and no populated ``contacts`` list is the
    genuine "nothing is touching" negative - it must not be mistaken for
    success. This pins the correct-negative return that the positive-detection
    tests do not reach (they return early on the True branch).
    """
    sim = _MinimalSim(robots=["r0"])  # default get_contacts -> {"n_contacts": 0}
    fn = PolicyRunner(sim)._resolve_success_fn("contact")
    assert fn is not None
    assert fn({}) is False


def test_contact_success_fn_false_on_non_dict_result():
    """A ``get_contacts`` that returns a non-dict (e.g. a bare list or None)
    is treated as no-success rather than crashing on ``.get``. Pins the final
    fallthrough return for an unrecognised result shape.
    """

    class _ListResultSim(_MinimalSim):
        def get_contacts(self):
            return ["unexpected", "shape"]

    sim = _ListResultSim(robots=["r0"])
    fn = PolicyRunner(sim)._resolve_success_fn("contact")
    assert fn({}) is False
