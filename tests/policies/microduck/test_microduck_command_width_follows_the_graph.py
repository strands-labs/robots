"""The command width follows the graph's declared input, not ``command_names``.

Seven of the nine ONNX policies Pollen ships declare a ``command_names``
narrower than the ``obs`` input their graph consumes. Every shipped export takes
``obs [1, 61]``; only ``alpha_stand`` and ``alpha_walking`` declare the full
``twist,head_pose,body_pose`` (13). ``roulade``, ``roller``, ``roller_crouch``,
``ball_kick_left``, ``ball_kick_right`` and ``alpha_ground_pick`` declare
``twist`` (3), and ``alpha_sitstand`` declares ``twist,head_pose`` (7).

Summing those names produced a 51- or 55-wide observation for a 61-wide graph,
so onnxruntime refused the very first inference with
``INVALID_ARGUMENT ... Got: 51 Expected: 61`` and those seven policies could not
be run at all. ``command_names`` names which command slots a skill READS; it is
not a width. Pollen's reference runner emits ONE unified 13-component command
for every skill in a bundle (``infer_policy.py``: ``self.command =
np.zeros(13 if self.new_cmd_obs else 3)``) and leaves the slots a skill ignores
present and zero - the dead-weight rule ``observation.build_observation``
already documents, and reads the graph's own input shape to size it.

So the graph's declared input width is the authority when it declares one, and
the ``command_names`` sum stays the fallback for a session that declares no
usable shape. That fallback is what keeps an injected stub - which need not
describe a shape at all - behaving exactly as before, and it is graded here
alongside the fix so widening one cannot quietly narrow the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import MicroduckPolicy

#: Fixed observation blocks that are not the command: ``base_ang_vel`` (3) plus
#: ``projected_gravity`` (3), then three per-joint blocks. Stated locally so
#: these cells are an independent oracle rather than a restatement of the module.
BASE_OBS_WIDTH = 6
PER_JOINT_BLOCKS = 3

#: The 14-DOF biped every shipped export targets, and the ``obs`` width all nine
#: of them declare. 6 + 3*14 = 48 fixed, so the command block is 61 - 48 = 13.
JOINTS = 14
SHIPPED_OBS_WIDTH = 61
SHIPPED_COMMAND_WIDTH = 13

#: What the seven under-declaring exports say, and what summing it would build.
NARROW_COMMAND_NAMES = "twist"
NARROW_SUM_WIDTH = 3


def _fixed_width(n_joints: int) -> int:
    """Non-command observation width for ``n_joints`` (independent of the module)."""
    return BASE_OBS_WIDTH + PER_JOINT_BLOCKS * n_joints


class _Meta:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.custom_metadata_map = mapping


class _Session:
    """A session that declares an ``obs`` shape, the way a real export does.

    ``declared`` is the graph's input width. ``None`` models a stub that
    describes no shape at all (the pre-existing stubs in the sibling suites),
    and a ``str`` models a dynamic-axis symbol, which onnxruntime does emit.
    """

    def __init__(self, declared: int | str | None, command_names: str, n_joints: int = JOINTS) -> None:
        self._declared = declared
        self._n_joints = n_joints
        self.obs_widths: list[int] = []
        self.meta = {
            "joint_names": ",".join(f"j{i}" for i in range(n_joints)),
            "default_joint_pos": ",".join("0.0" for _ in range(n_joints)),
            "action_scale": "1.0",
        }
        if command_names:
            self.meta["command_names"] = command_names

    def get_inputs(self) -> list[Any]:
        declared = self._declared

        class _Input:
            name = "obs"

            def __init__(self) -> None:
                if declared is not None:
                    self.shape = [1, declared]

        return [_Input()]

    def get_modelmeta(self) -> _Meta:
        return _Meta(self.meta)

    def run(self, _names: Any, feed: dict[str, Any]) -> list[np.ndarray]:
        vector = np.asarray(next(iter(feed.values()))).reshape(-1)
        self.obs_widths.append(int(vector.shape[0]))
        # A real graph refuses a width it did not declare; model that so a cell
        # cannot pass on a vector onnxruntime would have rejected. Only enforced
        # for a width a real export could declare - one that leaves room for a
        # command block. A degenerate declaration is the input under test in
        # ``TestTheCommandNamesFallbackIsUnchanged``, where the policy is
        # expected to ignore it, so enforcing it there would contradict the cell.
        plausible = isinstance(self._declared, int) and not isinstance(self._declared, bool)
        if plausible and self._declared > _fixed_width(self._n_joints):  # type: ignore[operator]
            if vector.shape[0] != self._declared:
                raise ValueError(
                    f"Got invalid dimensions for input: obs. Got: {vector.shape[0]} Expected: {self._declared}"
                )
        return [np.zeros((1, self._n_joints), dtype=np.float32)]


def _obs(n_joints: int = JOINTS) -> dict[str, Any]:
    d: dict[str, Any] = {"base_ang_vel": [0.0, 0.0, 0.0], "base_quat": [1.0, 0.0, 0.0, 0.0]}
    for i in range(n_joints):
        d[f"j{i}"] = 0.1
        d[f"j{i}.vel"] = 0.2
    return d


class TestThePremisesTheFixRestsOn:
    """The arithmetic and the shipped shape, stated here rather than assumed."""

    def test_the_fixed_blocks_leave_a_thirteen_wide_command(self) -> None:
        assert _fixed_width(JOINTS) == 48
        assert SHIPPED_OBS_WIDTH - _fixed_width(JOINTS) == SHIPPED_COMMAND_WIDTH

    def test_summing_the_narrow_names_would_not_reach_the_graph(self) -> None:
        """The whole defect in one line: the sum is not the declared width."""
        assert NARROW_SUM_WIDTH != SHIPPED_COMMAND_WIDTH
        assert _fixed_width(JOINTS) + NARROW_SUM_WIDTH != SHIPPED_OBS_WIDTH


class TestAnUnderDeclaringExportStillRuns:
    """The seven shipped policies whose ``command_names`` is narrower than the graph."""

    def test_the_built_observation_matches_the_declared_input(self) -> None:
        session = _Session(SHIPPED_OBS_WIDTH, NARROW_COMMAND_NAMES)
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(), "")
        assert session.obs_widths == [SHIPPED_OBS_WIDTH]

    def test_the_command_is_as_wide_as_the_graph_expects(self) -> None:
        session = _Session(SHIPPED_OBS_WIDTH, NARROW_COMMAND_NAMES)
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(), "")
        assert policy._command is not None
        assert policy._command.shape[0] == SHIPPED_COMMAND_WIDTH

    def test_a_target_velocity_still_writes_the_twist_slots(self) -> None:
        """Widening must not move the twist block off the front of the vector."""
        session = _Session(SHIPPED_OBS_WIDTH, NARROW_COMMAND_NAMES)
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(), "", target_velocity=[0.3, 0.0, 0.0])
        assert policy._command is not None
        assert list(policy._command[:3]) == [pytest.approx(0.3), 0.0, 0.0]
        assert not np.any(policy._command[3:])

    @pytest.mark.parametrize("names", ["twist", "twist,head_pose", "twist,head_pose,body_pose"])
    def test_every_shipped_declaration_reaches_the_same_graph(self, names: str) -> None:
        """All nine exports declare obs=61; three distinct ``command_names`` do not."""
        session = _Session(SHIPPED_OBS_WIDTH, names)
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(), "")
        assert session.obs_widths == [SHIPPED_OBS_WIDTH]

    def test_an_initial_command_is_measured_against_the_declared_width(self) -> None:
        """A caller supplying the graph's width is accepted, not refused as 13 != 3."""
        session = _Session(SHIPPED_OBS_WIDTH, NARROW_COMMAND_NAMES)
        policy = MicroduckPolicy(  # type: ignore[arg-type]
            session=session, command=[0.0] * SHIPPED_COMMAND_WIDTH
        )
        policy.get_actions_sync(_obs(), "")
        assert session.obs_widths == [SHIPPED_OBS_WIDTH]


class TestTheJointCountIsNotHardcoded:
    """The fixed part is ``6 + 3 * n_joints``, so another embodiment resolves too."""

    def test_a_nine_joint_graph_resolves_its_own_command_width(self) -> None:
        n, command = 9, 5
        session = _Session(_fixed_width(n) + command, NARROW_COMMAND_NAMES, n_joints=n)
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(n), "")
        assert session.obs_widths == [_fixed_width(n) + command]
        assert policy._command is not None
        assert policy._command.shape[0] == command


class TestTheCommandNamesFallbackIsUnchanged:
    """A session that declares no usable width keeps the pre-existing behaviour.

    These hold on both trees. They are the boundary the fix must not cross: the
    sibling suites' stubs describe an input NAME and no shape, so the
    ``command_names`` sum has to stay authoritative for them.
    """

    def test_a_stub_without_a_declared_shape_sums_the_names(self) -> None:
        session = _Session(None, NARROW_COMMAND_NAMES)
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(), "")
        assert session.obs_widths == [_fixed_width(JOINTS) + NARROW_SUM_WIDTH]

    def test_a_dynamic_axis_symbol_is_not_read_as_a_width(self) -> None:
        session = _Session("batch_x_obs", NARROW_COMMAND_NAMES)
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(), "")
        assert session.obs_widths == [_fixed_width(JOINTS) + NARROW_SUM_WIDTH]

    def test_a_shapeless_stub_with_no_command_names_uses_the_thirteen_default(self) -> None:
        session = _Session(None, "")
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(), "")
        assert session.obs_widths == [_fixed_width(JOINTS) + SHIPPED_COMMAND_WIDTH]

    def test_a_declared_width_at_or_below_the_fixed_blocks_is_ignored(self) -> None:
        """A shape that leaves no room for a command is not a command width."""
        session = _Session(_fixed_width(JOINTS), NARROW_COMMAND_NAMES)
        policy = MicroduckPolicy(session=session)  # type: ignore[arg-type]
        policy.get_actions_sync(_obs(), "")
        assert session.obs_widths == [_fixed_width(JOINTS) + NARROW_SUM_WIDTH]
