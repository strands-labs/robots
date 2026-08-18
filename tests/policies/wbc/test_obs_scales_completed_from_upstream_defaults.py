"""A WBC config that states some observation scales keeps the defaults for the rest.

:class:`~strands_robots.policies.wbc.config.WBCConfig` documents three
observation scales with upstream defaults (``ang_vel`` 0.5, ``dof_pos`` 1.0,
``dof_vel`` 0.05) and the frame builders read them per tick. A config that states
NO scale gets those defaults from the dataclass field; a config that states
*some* of them used to replace the whole map, leaving the rest to a second
fallback number inside the builders - a bare ``1.0`` - so the effective scale of
an unnamed key depended on which of its siblings the config happened to name.
``dof_vel`` resolved to 1.0 instead of 0.05, multiplying the 29 joint-velocity
entries of the frame by 20 while the config, the docs and the layout comment all
said 0.05.

That is the whole point of these tests: naming one scale must not change what an
unnamed sibling is scaled by. The upstream flat-key form
(``ang_vel_scale`` / ``dof_pos_scale`` / ``dof_vel_scale``) makes a partial
config the ordinary case, since a config that only wants to retune one scale
states only that one.

The expected values are read from a config that states no scales at all - the
route whose defaults were never in question - rather than from a literal table,
so these tests grade the agreement between the two routes rather than restating
the numbers.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from strands_robots.policies.wbc import WBCConfig, build_gait_frame
from strands_robots.policies.wbc.observation import build_single_frame

# The upstream flat spelling of each nested scale key, as WBCConfig.from_dict
# accepts it. Used to build every subset of "which scales did this config state".
_FLAT_KEY = {"ang_vel": "ang_vel_scale", "dof_pos": "dof_pos_scale", "dof_vel": "dof_vel_scale"}

# A config stating nothing about the scales: the documented-default route, and
# the reference every partial config below is graded against.
_NO_SCALES_STATED = {"policy_path": "p.onnx"}


def _documented_scales() -> dict[str, float]:
    """The effective scales of a config that states none of them."""
    return dict(WBCConfig.from_dict(_NO_SCALES_STATED).obs_scales)


def _frame_kwargs(no: int = 29, na: int = 15, c: int = 7) -> dict[str, np.ndarray]:
    """Sub-vectors for one observation frame, with a non-zero dqj block.

    ``dqj`` has to be non-zero for a wrong ``dof_vel`` scale to be observable at
    all - the block the defect mis-scaled is zero for a robot standing still.
    """
    return {
        "command": np.zeros(c),
        "base_ang_vel": np.full(3, 0.4),
        "proj_gravity": np.array([0.0, 0.0, -1.0]),
        "qj": np.linspace(-0.3, 0.3, no),
        "dqj": np.full(no, 2.0),
        "prev_action": np.zeros(na),
    }


class TestAnOmittedScaleKeepsItsDocumentedDefault:
    """Stating one scale must not silently change an unnamed sibling."""

    def test_a_config_that_omits_dof_vel_scale_still_scales_joint_velocity_by_it(self) -> None:
        """The realistic partial config: retune two scales, inherit the third.

        A config that states ``ang_vel_scale`` and ``dof_pos_scale`` and leaves
        ``dof_vel_scale`` to its documented 0.05 must build the same joint
        velocity block as a config that states all three.
        """
        documented = _documented_scales()
        full = WBCConfig.from_dict(
            {**_NO_SCALES_STATED, **{flat: documented[name] for name, flat in _FLAT_KEY.items()}}
        )
        partial = WBCConfig.from_dict(
            {**_NO_SCALES_STATED, "ang_vel_scale": documented["ang_vel"], "dof_pos_scale": documented["dof_pos"]}
        )
        kwargs = _frame_kwargs()
        no = full.n_obs_joints
        block = slice(full.command_dim + 6 + no, full.command_dim + 6 + 2 * no)
        expected = build_single_frame(full, **kwargs)[block]
        got = build_single_frame(partial, **kwargs)[block]
        assert expected.any(), "premise: the dqj block must be non-zero for a wrong scale to show"
        ratio = float(got[0] / expected[0])
        assert got == pytest.approx(expected), (
            f"a config that states ang_vel_scale and dof_pos_scale but leaves dof_vel_scale to its "
            f"documented default {documented['dof_vel']} scaled the joint velocity block by "
            f"{ratio:g}x what the same scales state explicitly: {got[:3]} vs {expected[:3]}"
        )

    @pytest.mark.parametrize(
        "stated",
        [
            pytest.param(subset, id="+".join(subset) or "none")
            for r in range(4)
            for subset in itertools.combinations(sorted(_FLAT_KEY), r)
        ],
    )
    def test_an_omitted_scale_resolves_the_same_way_whatever_siblings_are_stated(self, stated: tuple[str, ...]) -> None:
        """Root cause: the effective scale of an unnamed key is independent of the named ones."""
        documented = _documented_scales()
        config = WBCConfig.from_dict({**_NO_SCALES_STATED, **{_FLAT_KEY[name]: documented[name] for name in stated}})
        omitted = [name for name in documented if name not in stated]
        effective = {name: config.obs_scales.get(name, 1.0) for name in omitted}
        assert effective == pytest.approx({name: documented[name] for name in omitted}), (
            f"stating {list(stated)} changed the effective scale of the unnamed {omitted}: "
            f"{effective} instead of the documented {[documented[name] for name in omitted]}"
        )

    def test_a_partial_explicit_obs_scales_map_keeps_the_defaults_for_the_rest(self) -> None:
        """The nested-map route, reached by the constructor rather than from_dict."""
        documented = _documented_scales()
        config = WBCConfig(policy_path="p.onnx", obs_scales={"ang_vel": 0.25})
        assert config.obs_scales["ang_vel"] == 0.25, "the stated scale must survive"
        assert config.obs_scales["dof_vel"] == pytest.approx(documented["dof_vel"])
        assert config.obs_scales["dof_pos"] == pytest.approx(documented["dof_pos"])

    def test_obs_scales_reports_every_scale_the_frame_is_built_with(self) -> None:
        """``obs_scales`` is the complete effective map, so a subscript is safe."""
        config = WBCConfig.from_dict({**_NO_SCALES_STATED, "ang_vel_scale": 0.5})
        assert set(config.obs_scales) >= set(_FLAT_KEY), (
            f"obs_scales reports {sorted(config.obs_scales)}; a reader taking "
            f"config.obs_scales['dof_vel'] to learn what the frame is scaled by raises KeyError"
        )

    def test_the_gait_frame_is_built_with_the_same_completed_scales(self) -> None:
        """The gait variant reads the same three scales and must resolve them identically."""
        documented = _documented_scales()
        base = {"policy_path": "p.onnx", "command_dim": 8, "single_obs_dim": 95}
        full = WBCConfig.from_dict({**base, **{flat: documented[name] for name, flat in _FLAT_KEY.items()}})
        partial = WBCConfig.from_dict({**base, "ang_vel_scale": documented["ang_vel"]})
        kwargs = {**_frame_kwargs(c=8), "clock": np.array([0.3, -0.3])}
        expected = build_gait_frame(full, **kwargs)
        got = build_gait_frame(partial, **kwargs)
        assert got == pytest.approx(expected), (
            "the gait frame differs between a config that states every scale and one that "
            f"leaves two to their documented defaults: max delta {np.abs(got - expected).max():g}"
        )


class TestCompletingTheMapChangesNothingElse:
    """Boundaries the fill must not cross."""

    @pytest.mark.parametrize("route", ["flat", "nested"])
    def test_a_stated_scale_is_never_replaced_by_the_default(self, route: str) -> None:
        data = (
            {**_NO_SCALES_STATED, "dof_vel_scale": 0.5}
            if route == "flat"
            else {**_NO_SCALES_STATED, "obs_scales": {"dof_vel": 0.5}}
        )
        scales = WBCConfig.from_dict(data).obs_scales
        assert scales["dof_vel"] == 0.5

    def test_an_explicit_map_still_wins_over_the_flat_key_it_overlaps(self) -> None:
        config = WBCConfig.from_dict({**_NO_SCALES_STATED, "ang_vel_scale": 0.5, "obs_scales": {"ang_vel": 0.9}})
        assert config.obs_scales["ang_vel"] == 0.9

    def test_a_config_stating_no_scales_still_gets_the_upstream_defaults(self) -> None:
        scales = WBCConfig.from_dict(_NO_SCALES_STATED).obs_scales
        assert scales == pytest.approx({"ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5"])
    def test_a_scale_the_domain_refuses_is_still_refused(self, bad: object) -> None:
        """Completing the map must not smuggle an unusable stated value past validation."""
        with pytest.raises(ValueError, match=r"obs_scales\['dof_vel'\]"):
            WBCConfig(policy_path="p.onnx", obs_scales={"dof_vel": bad})  # type: ignore[dict-item]

    def test_a_scale_key_the_layout_does_not_read_is_kept(self) -> None:
        """The map is open: a caller-supplied extra key survives completion."""
        config = WBCConfig(policy_path="p.onnx", obs_scales={"height": 2.0})
        assert config.obs_scales["height"] == 2.0
        assert config.obs_scales["dof_vel"] == pytest.approx(0.05)
