"""``build_observation`` refuses a floating-base block it cannot read.

``MicroduckPolicy``'s observation vector is assembled from four inputs. Three of
them are the policy's own state and are held to a width at the policy seam:
``default_pose`` against ``len(joint_names)`` in ``_ensure_config``, a ``command``
override against the width ``command_names`` declares in ``_apply_command_kwargs``,
and the graph's returned action - which becomes the next tick's ``last_action`` -
in ``get_actions`` (#2882). The fourth arrives from the CALLER's observation dict:
the two floating-base blocks ``base_ang_vel`` (3) and ``base_quat`` (4, wxyz).
Those are read only inside the builder, so no seam ever sees them, and they were
taken by a truncating ``[:3]`` / ``[:4]`` slice rather than checked.

Measured on the pre-fix tree with the shipped alpha command width (C=13, so the
documented return width is ``48 + 13 == 61``):

=======================  ==========  =======  ==================================
input                    outcome     width    what the vector then carried
=======================  ==========  =======  ==================================
ang=3 quat=4 (contract)  success     61       correct
``base_ang_vel`` 2       success     60       one value short of the documented
``base_ang_vel`` 1       success     59       two values short
``base_quat`` 3          success     61       gravity 7.5-20.7 deg off
``base_quat`` 7          success     61       gravity 70.9 deg off
``base_quat`` 2          ValueError  -        raised from inside ``np.cross``
=======================  ==========  =======  ==================================

The ``base_quat`` rows are the ones nothing downstream can screen. One component
short, ``np.cross`` reads the 2-element ``q[1:4]`` as a planar vector, which is
exactly the quaternion with its ``z`` dropped: the gravity block keeps the
documented width and stays finite while the direction the biped is told is "down"
moves 7.5 degrees for a small-yaw pose and 20.7 for a roll-then-yaw one. Its norm
drifts with it (0.991 and 0.935), but nothing here reads a norm - the module
docstring says it normalises nothing - so at a percent off unity that is not a
signal anything acts on either. Over-long, a
7-element ``[base_pos, base_quat]`` slice - a caller handing over a floating-base
``qpos`` slice - is read as a quaternion made of positions. A short
``base_ang_vel`` instead falsifies the builder's own documented ``48 +
len(command)`` return width, handing the graph fewer values than its
``observation_names`` metadata declares - the same harm #2882 measured for the
fed-back action, reached through the other input.

Why nothing caught it: the shipped simulation backends produce these blocks from
fixed-width slices (measured below), so the policy path is exactly right today and
this is latent there. ``build_observation`` is a public export of
``strands_robots.policies.microduck`` though, and a caller assembling the dict from
a real IMU/teleop bridge - the consumer ``wbc.policy._extract_state`` documents -
has no seam at all. The sibling locomotion observation builders in this same
package already hold every sub-vector to a width via
``wbc.observation._require_len``, including ``base_ang_vel``; this builder was the
one that did not.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import build_observation, decode_action, projected_gravity
from strands_robots.policies.microduck import observation as obs_mod

#: The command width the shipped alpha policies declare (twist 3 + head 4 + body 6).
_ALPHA_COMMAND_WIDTH = 13

#: The fixed part of the layout: ang_vel 3 + gravity 3 + pos 14 + vel 14 + action 14.
_FIXED_OBS_WIDTH = 48

#: The contract joint count.
_N_JOINTS = 14


def _joints() -> list[str]:
    return [f"j{i}" for i in range(_N_JOINTS)]


def _obs_dict(
    names: list[str],
    *,
    base_ang_vel: Any = (0.1, 0.2, 0.3),
    base_quat: Any = (1.0, 0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """A runtime observation dict with both base blocks at a chosen width."""
    d: dict[str, Any] = {"base_ang_vel": base_ang_vel, "base_quat": base_quat}
    for n in names:
        d[n] = 0.0
        d[f"{n}.vel"] = 0.0
    return d


def _build(**over: Any) -> np.ndarray:
    names = _joints()
    return build_observation(
        _obs_dict(names, **over),
        joint_names=names,
        default_pose=np.zeros(_N_JOINTS, dtype=np.float32),
        last_action=np.zeros(_N_JOINTS, dtype=np.float32),
        command=np.zeros(_ALPHA_COMMAND_WIDTH, dtype=np.float32),
    )


#: Widths ``base_ang_vel`` must not be accepted at. 2 and 1 narrowed the returned
#: vector; 5 was silently truncated.
_WRONG_ANG_VEL_WIDTHS = (1, 2, 4, 5, 6)

#: Widths ``base_quat`` must not be accepted at. 3 and 7 kept the documented width
#: and changed the gravity block; 2 raised from inside numpy naming nothing.
_WRONG_QUAT_WIDTHS = (1, 2, 3, 5, 6, 7)


class TestAWrongWidthBaseBlockIsRefused:
    """The regression: each block is held to the width its layout defines."""

    @pytest.mark.parametrize("width", _WRONG_ANG_VEL_WIDTHS)
    def test_a_wrong_width_angular_velocity_is_refused(self, width: int) -> None:
        with pytest.raises(ValueError, match=re.escape("observation_dict['base_ang_vel']")):
            _build(base_ang_vel=np.zeros(width, dtype=np.float32))

    @pytest.mark.parametrize("width", _WRONG_QUAT_WIDTHS)
    def test_a_wrong_width_quaternion_is_refused(self, width: int) -> None:
        with pytest.raises(ValueError, match=re.escape("observation_dict['base_quat']")):
            _build(base_quat=np.zeros(width, dtype=np.float32))

    def test_a_scalar_block_is_refused(self) -> None:
        """A bare float ravels to one component rather than to the block width."""
        with pytest.raises(ValueError, match=re.escape("observation_dict['base_ang_vel']")):
            _build(base_ang_vel=0.1)

    def test_the_one_short_quaternion_that_kept_the_documented_width_is_refused(self) -> None:
        """The unscreenable case: right width, unit norm, every component finite."""
        with pytest.raises(ValueError, match="base_quat"):
            _build(base_quat=(0.683, 0.183, 0.183))

    def test_the_qpos_slice_a_caller_would_reach_for_is_refused(self) -> None:
        """A 7-element ``[base_pos, base_quat]`` is not a quaternion."""
        with pytest.raises(ValueError, match="base_quat"):
            _build(base_quat=(1.5, -0.25, 0.31, 0.9659, 0.2588, 0.0, 0.0))

    def test_the_first_wrong_block_is_the_one_named(self) -> None:
        """Both wrong: the angular velocity is read first, so it is reported."""
        with pytest.raises(ValueError, match="base_ang_vel"):
            _build(base_ang_vel=(0.1, 0.2), base_quat=(1.0, 0.0, 0.0))


class TestTheRefusalNamesWhatAReaderNeeds:
    """A width refusal has to say which block, how wide, and why it matters."""

    def test_the_refusal_names_the_builder(self) -> None:
        with pytest.raises(ValueError, match="build_observation"):
            _build(base_ang_vel=(0.1, 0.2))

    def test_the_refusal_names_both_widths(self) -> None:
        with pytest.raises(ValueError) as exc:
            _build(base_quat=(1.0, 0.0, 0.0))
        text = str(exc.value)
        assert "3 component(s)" in text, text
        assert "4 wide" in text, text

    def test_the_refusal_names_the_contract_it_protects(self) -> None:
        """The documented return width is what a narrow block falsifies."""
        with pytest.raises(ValueError, match=re.escape("48 + len(command)")):
            _build(base_ang_vel=(0.1, 0.2))


class TestTheAcceptedContractIsUnchanged:
    """Cells that hold on both trees: nothing legitimate was refused."""

    def test_the_exact_widths_still_build_the_documented_vector(self) -> None:
        vector = _build()
        assert vector.shape == (_FIXED_OBS_WIDTH + _ALPHA_COMMAND_WIDTH,)
        assert vector.dtype == np.float32
        assert bool(np.all(np.isfinite(vector)))

    def test_the_legacy_twist_only_command_width_still_works(self) -> None:
        names = _joints()
        vector = build_observation(
            _obs_dict(names),
            joint_names=names,
            default_pose=np.zeros(_N_JOINTS, dtype=np.float32),
            last_action=np.zeros(_N_JOINTS, dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),
        )
        assert vector.shape[0] == _FIXED_OBS_WIDTH + 3 == 51

    def test_a_row_shaped_block_is_still_accepted(self) -> None:
        """``reshape(-1)`` is kept: a (1, 3) row is three components."""
        vector = _build(base_ang_vel=np.zeros((1, 3), dtype=np.float32))
        assert vector.shape[0] == _FIXED_OBS_WIDTH + _ALPHA_COMMAND_WIDTH

    def test_a_python_list_block_is_still_accepted(self) -> None:
        """The shipped backends hand over plain lists of floats."""
        vector = _build(base_ang_vel=[0.1, 0.2, 0.3], base_quat=[1.0, 0.0, 0.0, 0.0])
        assert vector.shape[0] == _FIXED_OBS_WIDTH + _ALPHA_COMMAND_WIDTH

    def test_the_gravity_block_is_byte_identical_for_an_accepted_quaternion(self) -> None:
        """The accepted path's arithmetic did not move."""
        quat = np.array([0.683, 0.183, 0.183, 0.683], dtype=np.float32)
        assert np.array_equal(_build(base_quat=quat)[3:6], projected_gravity(quat))

    def test_a_missing_block_is_still_a_key_error(self) -> None:
        """The documented ``KeyError`` contract is not replaced by the width one."""
        names = _joints()
        d = _obs_dict(names)
        del d["base_quat"]
        with pytest.raises(KeyError):
            build_observation(
                d,
                joint_names=names,
                default_pose=np.zeros(_N_JOINTS, dtype=np.float32),
                last_action=np.zeros(_N_JOINTS, dtype=np.float32),
                command=np.zeros(_ALPHA_COMMAND_WIDTH, dtype=np.float32),
            )

    def test_decode_action_is_untouched(self) -> None:
        pose = np.array([0.1] * _N_JOINTS, dtype=np.float32)
        raw = np.array([1.0] * _N_JOINTS, dtype=np.float32)
        assert np.allclose(decode_action(raw, default_pose=pose, action_scale=0.25), 0.35)


class TestWhatIsDeliberatelyLeftAlone:
    """The boundary, pinned so a later widening is a deliberate edit."""

    def test_the_last_action_width_is_still_taken_on_trust_here(self) -> None:
        """It is checked at the policy seam instead - #2882's stated design.

        The base blocks differ precisely in having no seam: they arrive from the
        caller's dict and are read only here.
        """
        names = _joints()
        vector = build_observation(
            _obs_dict(names),
            joint_names=names,
            default_pose=np.zeros(_N_JOINTS, dtype=np.float32),
            last_action=np.zeros(1, dtype=np.float32),
            command=np.zeros(_ALPHA_COMMAND_WIDTH, dtype=np.float32),
        )
        assert vector.shape[0] == _FIXED_OBS_WIDTH - _N_JOINTS + 1 + _ALPHA_COMMAND_WIDTH

    def test_the_finiteness_axis_is_no_longer_left_alone(self) -> None:
        """The boundary this class pinned has moved, deliberately.

        This cell recorded finiteness as "a separate axis, refused at the action
        seam (#2882)". The first half was right and is why it is a separate
        change. The second half named a refusal that does exist and blames the
        wrong party: #2882 holds the graph's RETURNED action to the domain, so a
        caller's ``nan`` propagated through the graph and came back as
        ``'the ONNX action'`` - the checkpoint reported for a number the caller
        supplied. The builder now refuses it first, naming the block.

        The two axes stay independent: a wrong width is still reported by width,
        which the rest of this file holds.
        """
        with pytest.raises(ValueError, match=r"projected_gravity \(from base_quat\)"):
            _build(base_quat=(float("nan"), 0.0, 0.0, 0.0))


class TestBothBlocksRouteThroughTheOneReader:
    """Derived: a base block added later cannot re-introduce a bare slice."""

    @staticmethod
    def _builder_tree() -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(inspect.getsource(build_observation)))
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        return fn

    def test_no_base_key_is_read_by_a_bare_subscript(self) -> None:
        fn = self._builder_tree()
        literal_reads = [
            node.slice.value
            for node in ast.walk(fn)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "observation_dict"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ]
        assert [k for k in literal_reads if k.startswith("base_")] == [], (
            "a floating-base block is read straight out of the dict again; route it "
            "through _require_base_block so its width is held"
        )

    def test_every_base_block_is_read_through_the_reader(self) -> None:
        fn = self._builder_tree()
        keys: list[str] = []
        for call in ast.walk(fn):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_require_base_block"
                and len(call.args) >= 2
                and isinstance(call.args[1], ast.Constant)
                and isinstance(call.args[1].value, str)
            ):
                keys.append(call.args[1].value)
        # ``base_acc`` joined the reader in the harness#388 branch: slot two
        # is the projected-gravity block (from ``base_quat``) or the raw
        # accelerometer verbatim, chosen by ``gravity_source``.  All three
        # floating-base reads route through the width-holding reader.
        # Deduplicated: ``base_quat`` is read on BOTH branches now (the raw-accel
        # estimator falls back to the rotation), and the property here is that
        # every base key goes through the reader, not that each is read once.
        assert sorted(set(keys)) == ["base_acc", "base_ang_vel", "base_quat"], keys

    def test_the_widths_come_from_the_named_constants(self) -> None:
        """Not restated at the call site, so the layout has one account."""
        fn = self._builder_tree()
        widths = [
            ast.unparse(call.args[2])
            for call in ast.walk(fn)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_require_base_block"
            and len(call.args) >= 3
        ]
        # The ``base_acc`` branch shares ``_BASE_ACC_LEN == 3`` with the raw
        # accelerometer contract.  Each of the three keys is held to the
        # width its observation block defines through a named constant, so
        # a layout change has one place to touch.
        assert sorted(set(widths)) == ["_BASE_ACC_LEN", "_BASE_ANG_VEL_LEN", "_BASE_QUAT_LEN"], widths
        assert obs_mod._BASE_ANG_VEL_LEN == 3
        assert obs_mod._BASE_QUAT_LEN == 4
        assert obs_mod._BASE_ACC_LEN == 3


class TestThePremisesTheDefectRestedOn:
    """Facts that hold on both trees and make the measurement above legible."""

    def test_numpy_cross_accepts_a_two_element_vector_but_deprecates_it(self) -> None:
        """Why a 3-element quaternion was silent - and why it will stop being.

        NumPy 2.0 deprecated the planar form, so the truncating read was already
        heading for a raise from inside numpy naming neither block.
        """
        with pytest.warns(DeprecationWarning, match="2-dimensional vectors are deprecated"):
            planar = np.cross(np.array([0.2, 0.3]), np.array([0.0, 0.0, -1.0]))
        assert planar.shape == (3,)
        with pytest.raises(ValueError, match="dimension"):
            _ = np.cross(np.array([0.2, 0.3, 0.4, 0.5]), np.array([0.0, 0.0, -1.0]))

    @pytest.mark.filterwarnings("ignore:Arrays of 2-dimensional vectors:DeprecationWarning")
    def test_a_three_element_quaternion_reads_as_the_quaternion_with_z_dropped(self) -> None:
        short = np.array([0.683, 0.183, 0.183], dtype=np.float32)
        with_zero_z = np.array([0.683, 0.183, 0.183, 0.0], dtype=np.float32)
        assert np.array_equal(projected_gravity(short), projected_gravity(with_zero_z))

    @pytest.mark.filterwarnings("ignore:Arrays of 2-dimensional vectors:DeprecationWarning")
    @pytest.mark.parametrize(
        ("quat", "min_tilt_deg", "norm_tolerance"),
        [
            # A small-yaw pose: the norm stays inside a percent of unity, so even a
            # norm check with a sane tolerance would pass it.
            ((0.9098, -0.0665, 0.1604, 0.3769), 5.0, 0.01),
            # A roll-then-yaw pose: the worst tilt measured, still finite.
            ((0.683, 0.183, 0.183, 0.683), 15.0, 0.07),
        ],
    )
    def test_the_wrong_reading_stays_finite_and_near_unit_norm(
        self, quat: tuple[float, ...], min_tilt_deg: float, norm_tolerance: float
    ) -> None:
        """Width and finiteness are intact and the norm barely moves."""
        full = np.asarray(quat, dtype=np.float32)
        full = full / np.linalg.norm(full)
        wrong = projected_gravity(full[:3])
        right = projected_gravity(full)
        assert bool(np.all(np.isfinite(wrong)))
        assert float(np.linalg.norm(right)) == pytest.approx(1.0, abs=1e-3)
        assert float(np.linalg.norm(wrong)) == pytest.approx(1.0, abs=norm_tolerance)
        cos = float(np.clip(np.dot(right, wrong) / (np.linalg.norm(right) * np.linalg.norm(wrong)), -1.0, 1.0))
        assert np.degrees(np.arccos(cos)) > min_tilt_deg

    def test_the_only_norm_this_module_reads_is_the_accelerometer_estimator(self) -> None:
        """So the norm drift is still not a signal anything acts on.

        :func:`raw_accel_gravity` reads a norm to turn the accelerometer into the
        unit gravity direction Pollen's ``get_raw_accelerometer`` returns - the
        estimator's own arithmetic, on a block whose width the reader already
        held. Nothing reads a norm to JUDGE a block, so a wrong-width
        ``base_quat`` still produces no signal any code acts on, which is the
        premise the measurement above rests on.
        """
        source = inspect.getsource(obs_mod)
        norm_readers = sorted(
            node.name
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and "linalg.norm" in ast.unparse(node)
        )
        assert norm_readers == ["raw_accel_gravity"], norm_readers
        # Whitespace-normalised: the claim spans a line wrap in the docstring, so a
        # raw substring match would grade the wrapping rather than the sentence.
        assert "This module never rescales the assembled vector." in " ".join(source.split())

    def test_the_sibling_locomotion_builder_holds_the_same_block_to_a_width(self) -> None:
        """The in-package convention this builder was the exception to."""
        from strands_robots.policies.wbc import observation as wbc_obs

        with pytest.raises(ValueError, match="base_ang_vel must have length 3"):
            wbc_obs._require_len(np.zeros(2), 3, "base_ang_vel")

    def test_the_mujoco_backend_produces_fixed_width_base_blocks(self) -> None:
        """Why the policy path is right today, i.e. why this is latent there."""
        from strands_robots.simulation.mujoco import rendering

        source = inspect.getsource(rendering)
        assert re.search(r'obs\["base_quat"\]\s*=\s*\[[^\]]*qpos\[qadr \+ 3 : qadr \+ 7\]', source), (
            "the backend no longer slices a fixed 4 for base_quat"
        )
        assert re.search(r'obs\["base_ang_vel"\]\s*=\s*\[[^\]]*qvel\[vadr \+ 3 : vadr \+ 6\]', source), (
            "the backend no longer slices a fixed 3 for base_ang_vel"
        )

    def test_the_builder_is_a_public_export(self) -> None:
        """Why a caller with no policy seam is a first-class consumer."""
        import strands_robots.policies.microduck as pkg

        assert "build_observation" in pkg.__all__
