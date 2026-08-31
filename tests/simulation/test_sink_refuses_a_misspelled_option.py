"""Regression: a tolerating ``**kwargs`` sink must refuse a misspelled option.

A backend constructor accepts ``**kwargs`` as a *tolerating* sink: an
unrecognised name is dropped so one call can carry another backend's options
(``num_envs`` / ``device``) and resolve against whichever backend is selected.
``tests/simulation/test_constructor_rejects_setup_kwargs.py`` pins that
tolerance deliberately.

The tolerance covers a name *some* backend reads. It said nothing about a
**misspelling of a name the receiver itself reads**, and dropping one made the
argument byte-identical to omitting it:

* ``Robot("so101", defualt_timestep=0.001)`` integrated the physics at the 2 ms
  default -- half the requested rate -- and reported success;
* ``Robot("so101", positon=[0.5, 0, 0])`` spawned the robot at the origin. That
  is the failure the ``position`` / ``orientation`` / ``keyframe`` pass-through
  in the factory was written to end ("absorbed by ``**kwargs``"), reaching the
  parameter *names* instead of their values.

These pin the corrected contract:

* a residual name close to one of the receiver's own parameters is a ``TypeError``
  naming the parameter meant, at the backend constructor and at the factory;
* every name the sink exists to carry is still tolerated and dropped -- notably
  ``timestep``, which ``difflib``'s own 0.6 default *would* have refused;
* a tolerated drop is logged, so it is visible rather than silent;
* the accepted set is derived from the receiver's signature, so it cannot go
  stale and quietly stop screening.
"""

import logging
from typing import Any

import pytest

from strands_robots.simulation.base import (
    _MISSPELLING_RATIO,
    own_keyword_names,
    reject_misspelled_kwargs,
)

# Misspellings of MuJoCoSimEngine's own parameters, and the parameter meant.
# ``defualt_timestep`` is the reported case; the others cover a dropped
# character, a transposition inside a different parameter and a wrong suffix.
MISSPELLINGS = [
    ("defualt_timestep", "default_timestep"),
    ("default_timstep", "default_timestep"),
    ("default_hieght", "default_height"),
    ("tool_nmae", "tool_name"),
    ("ros2_domian", "ros2_domain"),
]

# Names the tolerating sink exists to carry: options of another backend
# (IsaacConfig fields, Newton's own parameters) and of the registered test
# plugins. None is a misspelling of a MuJoCo parameter, so every one must still
# be dropped without complaint.
FORWARD_COMPAT = [
    "num_envs",
    "device",
    "headless",
    "render_mode",
    "physics_dt",
    "gpu_id",
    "timestep",
    "num_worlds",
    "solver",
    "substeps",
]


def _mujoco_engine_cls() -> type:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    return MuJoCoSimEngine


class TestAMisspelledOwnParameterIsRefused:
    """The receiver names the parameter the caller meant, instead of dropping it."""

    @pytest.mark.parametrize(("typo", "meant"), MISSPELLINGS)
    def test_the_backend_constructor_refuses_it_by_name(self, typo: str, meant: str) -> None:
        cls = _mujoco_engine_cls()
        with pytest.raises(TypeError) as exc:
            cls(**{typo: 0.001})
        msg = str(exc.value)
        assert typo in msg, "the refusal must quote the name the caller wrote"
        assert meant in msg, "and name the parameter it misspells, so the fix is obvious"

    def test_every_offending_name_is_reported_at_once(self) -> None:
        """One pass fixes them all, matching the setup-kwarg refusal next door."""
        cls = _mujoco_engine_cls()
        with pytest.raises(TypeError) as exc:
            cls(defualt_timestep=0.001, default_hieght=1080)
        msg = str(exc.value)
        assert "defualt_timestep" in msg
        assert "default_hieght" in msg

    def test_a_requested_value_is_still_honoured_when_spelled_correctly(self) -> None:
        """The positive control: screening must not touch a correct call."""
        cls = _mujoco_engine_cls()
        sim = cls(default_timestep=0.001, tool_name="spelled_right")
        try:
            assert sim.default_timestep == 0.001
        finally:
            sim.cleanup()


class TestTheToleratingContractIsIntact:
    """Every name the sink exists to carry is still dropped without complaint."""

    @pytest.mark.parametrize("name", FORWARD_COMPAT)
    def test_another_backends_option_is_still_tolerated(self, name: str) -> None:
        cls = _mujoco_engine_cls()
        sim = cls(**{name: 1}, tool_name="fc_probe")
        try:
            assert sim is not None
        finally:
            sim.cleanup()

    def test_the_cutoff_is_what_saves_the_plugin_option(self) -> None:
        """``timestep`` is why the ratio is stated rather than inherited.

        ``difflib.get_close_matches`` defaults to a 0.6 cutoff, and ``timestep``
        scores 0.667 against ``default_timestep`` because it is a substring of
        it. Inheriting that default would refuse a real plugin option, so this
        pins the gap the chosen ratio sits in rather than the ratio's value.
        """
        import difflib

        own = own_keyword_names(_mujoco_engine_cls())
        assert difflib.get_close_matches("timestep", list(own), n=1, cutoff=0.6), (
            "premise: difflib's own default does consider 'timestep' a match"
        )
        assert not difflib.get_close_matches("timestep", list(own), n=1, cutoff=_MISSPELLING_RATIO), (
            "so the cutoff must sit above it, or a portable call breaks"
        )
        # And it still sits below every real typo, or the fix stops working.
        for typo, _meant in MISSPELLINGS:
            assert difflib.get_close_matches(typo, list(own), n=1, cutoff=_MISSPELLING_RATIO), (
                f"{typo!r} must still be recognised as a misspelling"
            )

    def test_a_tolerated_drop_is_logged_rather_than_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.base"):
            reject_misspelled_kwargs({"num_envs": 4}, ("default_timestep",), owner="Probe")
        assert "num_envs" in caplog.text, "a dropped name must be visible in a log"


class TestTheAcceptedSetIsDerivedFromTheSignature:
    """A hand-maintained list would go stale and quietly stop screening."""

    def test_it_reads_the_parameters_and_omits_the_sinks(self) -> None:
        own = own_keyword_names(_mujoco_engine_cls())
        assert "self" not in own
        assert "kwargs" not in own
        # A floor, not an inventory: the screen is vacuous if the parameters
        # the reported bug named ever stop being screened.
        for required in ("default_timestep", "default_width", "default_height"):
            assert required in own, f"{required} must be screened, or the typo returns"

    def test_an_unintrospectable_callable_degrades_to_no_screening(self) -> None:
        """Better to drop as before than to refuse every keyword."""
        assert own_keyword_names(dict.update) == ()

    def test_a_positional_only_parameter_is_not_a_keyword_name(self) -> None:
        """It cannot be spelled as a keyword, so resembling it is not a typo of one."""
        assert own_keyword_names(len) == ()


class TestTheFactoryScreensItsOwnParameters:
    """``Robot(mode="sim")`` forwards verbatim, so it must screen its own names."""

    @pytest.mark.parametrize(
        ("typo", "meant"), [("positon", "position"), ("orientaton", "orientation"), ("keyfrmae", "keyframe")]
    )
    def test_a_misspelled_factory_parameter_is_refused(self, typo: str, meant: str) -> None:
        pytest.importorskip("mujoco")
        from strands_robots import Robot

        kwargs: dict[str, Any] = {typo: [0.5, 0.0, 0.0]}
        with pytest.raises(TypeError) as exc:
            Robot("so101", mesh=False, **kwargs)
        msg = str(exc.value)
        assert typo in msg
        assert meant in msg

    def test_a_backend_option_passes_through_the_factory_untouched(self) -> None:
        """The factory screens against its OWN names only, or it breaks the backend."""
        pytest.importorskip("mujoco")
        from strands_robots import Robot

        sim = Robot("so101", mesh=False, default_timestep=0.001)
        try:
            assert sim.default_timestep == 0.001
        finally:
            sim.destroy()


def test_the_newton_backend_shares_the_contract() -> None:
    pytest.importorskip("newton")
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    with pytest.raises(TypeError) as exc:
        NewtonSimEngine(defualt_timestep=0.001)
    assert "default_timestep" in str(exc.value)
