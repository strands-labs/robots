"""``build_lerobot_command`` must refuse a boolean mode flag it can only misread.

Every flag this builder emits selects a *posture*, not a magnitude: ``dataset_video``
picks the literal ``"true"`` or ``"false"`` on the argv, while ``record_resume``
decides whether ``--resume true`` appears at all. Both were read by truthiness,
and every non-empty string is truthy, so the words an operator reaches for when
opting out selected the opposite posture from the one they read as. Measured on
``3ce3da7``:

* ``dataset_push_to_hub="false"`` emitted ``--dataset.push_to_hub true``, so a
  detached, unattended recording uploaded its dataset to the Hub;
* ``record_resume="false"`` emitted ``--resume true``, appending into an existing
  dataset - and preserving its already-stamped repo_id - instead of creating the
  fresh one that was asked for;
* ``dagger_record_autonomous="off"`` emitted ``--strategy.record_autonomous
  true``, recording autonomous rollout episodes into a corrections dataset;
* ``display_data="false"`` emitted ``--display_data true``.

``None`` and ``[]`` took the other branch just as silently, without ever being a
declared spelling of it. None of these is reported anywhere: the argv goes to a
subprocess launched with ``start_new_session=True``, the tool returns
``status="success"`` with a pid, and the CLI parses every one of these argvs
without complaint - it is simply told the opposite posture.

The flags are checked against the shared
:func:`~strands_robots.utils.boolean_flag_error` domain, and only for the flags
the requested mode actually emits: refusing a flag a mode never puts on the argv
would be a false rejection, which is the same scoping rule the numeric knobs use
(``tests/tools/test_lerobot_teleoperate_numeric_domain.py``).

``lerobot_teleoperate``'s own ``background`` and ``auto_accept_calibration`` were
read by truthiness for the same reason and are refused by the *tool*, since
neither is a builder parameter and no argv records them - they select an
execution posture. That half is #2074, and the last class here is where its
scope, its ordering and the boundary between the two checks are pinned.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import subprocess
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
from strands_robots.utils import boolean_flag_error

build_lerobot_command = tele_mod.build_lerobot_command
lerobot_teleoperate = tele_mod.lerobot_teleoperate


def _run_tool(**kwargs: Any) -> dict[str, Any]:
    """Call the agent tool through one funnel.

    The flag value under test is deliberately outside the declared ``bool``,
    which is the point; mypy does not narrow a splatted ``dict[str, Any]``, so
    routing the call through here states that once rather than suppressing it at
    the call site.
    """
    return dict(lerobot_teleoperate(**kwargs))


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep the module-level session store inside the test's temp dir."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(tele_mod, "SESSION_DIR", session_dir)
    return session_dir


@pytest.fixture
def _rollout_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present the lerobot rollout module so the ``dagger`` preflight passes.

    ``dagger`` needs lerobot>=0.6.0 installed to reach its argv at all; the flag
    refusal is deliberately placed *before* that preflight, so these tests pin
    the refusal on both sides of it (see
    :class:`TestTheRefusalPrecedesTheLerobotVersionPreflight`).
    """
    real_find_spec = tele_mod.importlib.util.find_spec

    def _find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lerobot.scripts.lerobot_rollout":
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(tele_mod.importlib.util, "find_spec", _find_spec)


# A value in no boolean's domain. The four strings are the spellings an operator
# reaches for when opting out, and each is truthy; ``nan`` and ``0.7`` are truthy
# numbers; ``None`` and ``[]`` are falsy values that are not a declared spelling
# of the negative posture either.
NOT_A_BOOLEAN = [
    pytest.param("false", id="str-false"),
    pytest.param("no", id="str-no"),
    pytest.param("off", id="str-off"),
    pytest.param("0", id="str-zero"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(0.7, id="fractional"),
    pytest.param(1, id="int-one"),
    pytest.param(0, id="int-zero"),
    pytest.param(None, id="none"),
    pytest.param([], id="empty-list"),
]

# Both python spellings plus the numpy booleans ``boolean_flag_error`` accepts,
# which arrive from an array-shaped config or a NumPy comparison.
A_BOOLEAN = [
    pytest.param(True, True, id="true"),
    pytest.param(False, False, id="false"),
    pytest.param(np.True_, True, id="np-true"),
    pytest.param(np.False_, False, id="np-false"),
]


def _record(**overrides: Any) -> list[str]:
    """A ``lerobot-record`` argv (``start`` + a dataset repo id)."""
    kwargs: dict[str, Any] = {
        "action": "start",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "teleop_type": "so101_leader",
        "teleop_port": "/dev/ttyACM0",
        "dataset_repo_id": "user/pick",
        "dataset_single_task": "pick the cube",
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _teleop(**overrides: Any) -> list[str]:
    """A ``lerobot-teleoperate`` argv (``start`` with no dataset)."""
    kwargs: dict[str, Any] = {
        "action": "start",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "teleop_type": "so101_leader",
        "teleop_port": "/dev/ttyACM0",
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _replay(**overrides: Any) -> list[str]:
    """A ``lerobot-replay`` argv."""
    kwargs: dict[str, Any] = {
        "action": "replay",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "dataset_repo_id": "user/pick",
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _dagger(**overrides: Any) -> list[str]:
    """A ``lerobot-rollout --strategy.type=dagger`` argv."""
    kwargs: dict[str, Any] = {
        "action": "dagger",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "teleop_type": "so101_leader",
        "teleop_port": "/dev/ttyACM0",
        "dataset_repo_id": "user/pick",
        "policy_path": "lerobot/act_so101",
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _token(argv: list[str], flag: str) -> str | None:
    """The token following ``flag``, or ``None`` when the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def _text(result: dict[str, Any]) -> str:
    """The joined text blocks of a tool envelope."""
    return "\n".join(item["text"] for item in result["content"] if "text" in item)


def _never(label: str) -> Any:
    """A stand-in that fails the test if the tool ever reaches it."""

    def _call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{label} must not be reached for a refused call")

    return _call


class _FakePopen:
    """A launched process, without launching one."""

    pid = 4321

    def __init__(self) -> None:
        self.stdin = io.StringIO()


def _record_popen(calls: list[str]) -> Any:
    """A ``subprocess.Popen`` stand-in recording the kwargs the tool chose.

    Whether ``stdin`` is passed at all is the observable
    ``auto_accept_calibration`` decision - the newlines themselves are written by
    a daemon thread two seconds later, which no test should wait for.
    """

    class _Recorder:
        kwargs: dict[str, Any] = {}

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            calls.append("Popen")
            self.kwargs = kwargs
            return _FakePopen()

    return _Recorder()


def _record_run(calls: list[str]) -> Any:
    """A ``subprocess.run`` stand-in reporting a clean exit."""

    def _call(*args: Any, **kwargs: Any) -> Any:
        calls.append("run")
        return subprocess.CompletedProcess(args=list(args[0]) if args else [], returncode=0, stdout="", stderr="")

    return _call


class TestAnUnattendedRecordingCannotBeTalkedIntoUploading:
    """``dataset_push_to_hub`` is the flag with the widest blast radius.

    A recording session is detached and unattended by design, and the Hub push
    happens at the end of it. A truthy spelling of off therefore published a
    dataset with no one watching, and the call that asked for the opposite had
    already returned ``status="success"``.
    """

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_record_refuses_a_non_boolean_push_to_hub(self, value: Any) -> None:
        with pytest.raises(ValueError, match="dataset_push_to_hub"):
            _record(dataset_push_to_hub=value)

    def test_the_opt_out_spelling_no_longer_selects_the_upload(self) -> None:
        """Pre-fix this emitted ``--dataset.push_to_hub true``."""
        with pytest.raises(ValueError, match="dataset_push_to_hub"):
            _record(dataset_push_to_hub="false")

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_dagger_refuses_a_non_boolean_push_to_hub(self, value: Any, _rollout_entry_point: None) -> None:
        """DAgger appends corrections to a dataset and pushes the same way."""
        with pytest.raises(ValueError, match="dataset_push_to_hub"):
            _dagger(dataset_push_to_hub=value)

    @pytest.mark.parametrize(("value", "expected"), A_BOOLEAN)
    def test_a_boolean_still_selects_the_posture_it_names(self, value: Any, expected: bool) -> None:
        token = _token(_record(dataset_push_to_hub=value), "--dataset.push_to_hub")
        assert token == ("true" if expected else "false")


class TestAFreshRecordingCannotBeTurnedIntoAnAppend:
    """``record_resume`` chooses between two datasets, not between two verbosities.

    Resume preserves an existing, already-stamped ``repo_id`` and appends to the
    data at the resolved root; a fresh record stamps a new one. A truthy spelling
    of off silently merged one operator's episodes into another's dataset.
    """

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_record_refuses_a_non_boolean_resume(self, value: Any) -> None:
        with pytest.raises(ValueError, match="record_resume"):
            _record(record_resume=value)

    def test_the_opt_out_spelling_no_longer_selects_the_append(self) -> None:
        """Pre-fix this emitted ``--resume true``."""
        with pytest.raises(ValueError, match="record_resume"):
            _record(record_resume="false")

    def test_a_true_resume_still_emits_the_flag(self) -> None:
        assert _token(_record(record_resume=True), "--resume") == "true"

    def test_a_false_resume_still_omits_the_flag(self) -> None:
        """Absence is how a fresh record is spelled; that is unchanged."""
        assert "--resume" not in _record(record_resume=False)

    def test_a_numpy_false_resume_omits_the_flag_too(self) -> None:
        """The check accepts a numpy boolean, so the emitter must handle one."""
        assert "--resume" not in _record(record_resume=np.False_)


class TestTheRemainingFlagsShareTheSameDomain:
    """``dataset_video``, ``display_data`` and ``dagger_record_autonomous``."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_record_refuses_a_non_boolean_video_setting(self, value: Any) -> None:
        with pytest.raises(ValueError, match="dataset_video"):
            _record(dataset_video=value)

    @pytest.mark.parametrize(("value", "expected"), A_BOOLEAN)
    def test_a_boolean_video_setting_reaches_the_argv(self, value: Any, expected: bool) -> None:
        assert _token(_record(dataset_video=value), "--dataset.video") == ("true" if expected else "false")

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_record_refuses_a_non_boolean_display_data(self, value: Any) -> None:
        with pytest.raises(ValueError, match="display_data"):
            _record(display_data=value)

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_teleoperate_refuses_a_non_boolean_display_data(self, value: Any) -> None:
        with pytest.raises(ValueError, match="display_data"):
            _teleop(display_data=value)

    def test_a_true_display_data_still_emits_the_flag(self) -> None:
        assert _token(_teleop(display_data=True), "--display_data") == "true"

    def test_a_false_display_data_still_omits_the_flag(self) -> None:
        assert "--display_data" not in _teleop(display_data=False)

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_dagger_refuses_a_non_boolean_record_autonomous(self, value: Any, _rollout_entry_point: None) -> None:
        with pytest.raises(ValueError, match="dagger_record_autonomous"):
            _dagger(dagger_record_autonomous=value)

    def test_a_true_record_autonomous_still_emits_the_flag(self, _rollout_entry_point: None) -> None:
        argv = _dagger(dagger_record_autonomous=True)
        assert _token(argv, "--strategy.record_autonomous") == "true"

    def test_a_false_record_autonomous_still_omits_the_flag(self, _rollout_entry_point: None) -> None:
        assert "--strategy.record_autonomous" not in _dagger(dagger_record_autonomous=False)


class TestOnlyTheFlagsAModeEmitsAreChecked:
    """A caller must never be refused for a flag the requested mode ignores.

    This is the over-reach control for the whole change: the same unusable value
    that is refused above must be accepted here, because no argv carries it.
    """

    @pytest.mark.parametrize(
        "flag",
        ["record_resume", "dataset_push_to_hub", "dataset_video", "display_data", "dagger_record_autonomous"],
    )
    def test_replay_refuses_no_flag_and_its_argv_is_unchanged(self, flag: str) -> None:
        """``lerobot-replay`` emits no boolean flag at all."""
        assert _replay(**{flag: "false"}) == _replay()

    @pytest.mark.parametrize(
        "flag",
        ["record_resume", "dataset_push_to_hub", "dataset_video", "dagger_record_autonomous"],
    )
    def test_teleoperate_reads_only_display_data(self, flag: str) -> None:
        """Plain teleoperation emits no ``--dataset.*`` flag and no strategy."""
        assert _teleop(**{flag: "false"}) == _teleop()

    def test_record_ignores_the_dagger_strategy_flag(self) -> None:
        assert _record(dagger_record_autonomous="false") == _record()

    def test_dagger_ignores_the_record_resume_flag(self, _rollout_entry_point: None) -> None:
        """``lerobot-rollout`` has no ``--resume``; the flag is never emitted."""
        assert _dagger(record_resume="false") == _dagger()


class TestPlaySoundsReachesEveryModeThatCanHonorIt:
    """``play_sounds`` is emitted by the three modes whose entry point takes it.

    It was declared, documented and forwarded, and then emitted by nothing, so
    ``play_sounds=False`` - the request an unattended session on a shared machine
    makes - produced an argv byte-identical to the default and the audio cues
    played anyway. Measured on ``586109c``, all four modes returned the same argv
    for ``True`` and ``False``.

    The scoping is what each lerobot entry point accepts, not a preference:
    ``RecordConfig``, ``ReplayConfig`` and ``RolloutConfig`` declare the field,
    while ``TeleoperateConfig`` does not and that CLI exits with ``unrecognized
    arguments: --play_sounds``. So plain teleoperation still emits nothing for it
    and still refuses nothing for it, and the other three both emit and check.
    """

    def test_the_lerobot_entry_points_declare_what_the_table_claims(self) -> None:
        """The premise the scoping rests on, read off the installed lerobot."""
        import dataclasses

        pytest.importorskip("lerobot")
        declares = {}
        for module, name in (
            ("lerobot.scripts.lerobot_record", "RecordConfig"),
            ("lerobot.scripts.lerobot_replay", "ReplayConfig"),
            ("lerobot.scripts.lerobot_rollout", "RolloutConfig"),
            ("lerobot.scripts.lerobot_teleoperate", "TeleoperateConfig"),
        ):
            config = getattr(importlib.import_module(module), name)
            declares[name] = "play_sounds" in {f.name for f in dataclasses.fields(config)}
        assert declares == {
            "RecordConfig": True,
            "ReplayConfig": True,
            "RolloutConfig": True,
            "TeleoperateConfig": False,
        }

    @pytest.mark.parametrize("builder", [_record, _replay])
    @pytest.mark.parametrize(("supplied", "emitted"), [(True, "true"), (False, "false")])
    def test_the_requested_posture_reaches_the_argv(self, builder: Any, supplied: bool, emitted: str) -> None:
        assert _token(builder(play_sounds=supplied), "--play_sounds") == emitted

    @pytest.mark.parametrize(("supplied", "emitted"), [(True, "true"), (False, "false")])
    def test_the_dagger_argv_carries_it_too(self, _rollout_entry_point: None, supplied: bool, emitted: str) -> None:
        assert _token(_dagger(play_sounds=supplied), "--play_sounds") == emitted

    @pytest.mark.parametrize("builder", [_record, _replay])
    def test_the_two_postures_are_no_longer_the_same_argv(self, builder: Any) -> None:
        assert builder(play_sounds=True) != builder(play_sounds=False)

    def test_it_is_emitted_explicitly_rather_than_only_when_withheld(self) -> None:
        """The declared default must not depend on lerobot's own default."""
        assert "--play_sounds" in _record()

    def test_plain_teleoperation_still_omits_it(self) -> None:
        """``TeleoperateConfig`` has no such field; emitting it would be fatal."""
        argv = _teleop(play_sounds=False)
        assert [token for token in argv if "sound" in token.lower()] == []

    @pytest.mark.parametrize("builder", [_record, _replay])
    @pytest.mark.parametrize("value", ["false", "off", None, [], 0, 1])
    def test_a_value_no_posture_can_be_read_from_is_refused(self, builder: Any, value: Any) -> None:
        with pytest.raises(ValueError, match="play_sounds"):
            builder(play_sounds=value)

    def test_the_dagger_mode_refuses_it_too(self, _rollout_entry_point: None) -> None:
        with pytest.raises(ValueError, match="play_sounds"):
            _dagger(play_sounds="false")

    @pytest.mark.parametrize("value", ["false", None, [], 0])
    def test_plain_teleoperation_refuses_nothing_for_it(self, value: Any) -> None:
        """Refusing a flag a mode never emits would be a false rejection."""
        assert _teleop(play_sounds=value) == _teleop()

    @pytest.mark.parametrize(("supplied", "emitted"), [(np.True_, "true"), (np.False_, "false")])
    def test_a_numpy_boolean_is_still_honored(self, supplied: Any, emitted: str) -> None:
        assert _token(_record(play_sounds=supplied), "--play_sounds") == emitted

    def test_the_refusal_precedes_the_argv(self) -> None:
        """Nothing is built for a value the session could not have honored."""
        with pytest.raises(ValueError, match="play_sounds"):
            _replay(play_sounds="false")

    @pytest.mark.parametrize(("builder", "config"), [(_record, "RecordConfig"), (_replay, "ReplayConfig")])
    @pytest.mark.parametrize("supplied", [True, False])
    def test_lerobot_parses_the_emitted_argv_back_to_the_requested_value(
        self, builder: Any, config: str, supplied: bool
    ) -> None:
        """The round trip, through the real CLI parser rather than a stub."""
        draccus = pytest.importorskip("draccus")
        pytest.importorskip("lerobot")
        importlib.import_module("lerobot.policies")  # registers the policy choices
        module = "lerobot.scripts." + ("lerobot_record" if config == "RecordConfig" else "lerobot_replay")
        config_class = getattr(importlib.import_module(module), config)
        argv = builder(play_sounds=supplied)[3:]  # drop ["python", "-m", <module>]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parsed = draccus.parse(config_class=config_class, args=argv)
        assert parsed.play_sounds is supplied


class TestTheRefusalPrecedesEverythingItWouldOtherwiseReach:
    """Nothing may be launched, persisted or preflighted for a refused call."""

    def test_the_tool_reports_the_refusal_without_starting_a_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _never(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("subprocess.Popen must not be reached for a refused call")

        monkeypatch.setattr(tele_mod.subprocess, "Popen", _never)
        result = _run_tool(
            action="start",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM1",
            teleop_type="so101_leader",
            teleop_port="/dev/ttyACM0",
            dataset_repo_id="user/pick",
            dataset_push_to_hub="false",
            session_name="refused-flag",
        )
        assert result["status"] == "error"
        text = "\n".join(item.get("text", "") for item in result["content"] if "text" in item)
        assert "dataset_push_to_hub" in text
        assert tele_mod.SessionManager().get_session("refused-flag") is None

    def test_dagger_refuses_the_flag_without_the_rollout_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same caller mistake must report the same way on any lerobot.

        Placed before the version preflight, so an unusable flag is named rather
        than being masked by an upgrade hint on an older install.
        """
        monkeypatch.setattr(tele_mod.importlib.util, "find_spec", lambda *args, **kwargs: None)
        with pytest.raises(ValueError, match="dagger_record_autonomous"):
            _dagger(dagger_record_autonomous="false")

    def test_a_numeric_refusal_still_comes_first(self) -> None:
        """Both are refusals; the order is chosen rather than incidental.

        The numeric knobs are checked first so their messages - and the tests
        that pin them - are unchanged by this addition.
        """
        with pytest.raises(ValueError, match="dataset_fps"):
            _record(dataset_fps=0, dataset_push_to_hub="false")


class TestTheFlagTableCannotDriftFromTheBuilder:
    """The table is the record of what each mode emits; keep it measurable."""

    def test_every_flag_a_mode_emits_is_a_real_parameter(self) -> None:
        params = set(inspect.signature(build_lerobot_command).parameters)
        named = {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}
        unknown = sorted(named - params)
        assert not unknown, f"_MODE_FLAG_OPTIONS names non-parameters: {unknown}"

    def test_every_flag_a_mode_emits_is_declared_a_bool(self) -> None:
        """A flag whose annotation is not ``bool`` belongs to another domain."""
        params = inspect.signature(build_lerobot_command).parameters
        named = {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}
        # The module has no ``from __future__ import annotations``, so the
        # annotation is the ``bool`` type itself; accept the string spelling too
        # so adding that import does not silently disarm this check.
        adrift = sorted(flag for flag in named if params[flag].annotation not in (bool, "bool"))
        assert not adrift, f"_MODE_FLAG_OPTIONS names non-bool parameters: {adrift}"

    def test_every_mode_in_the_table_is_one_the_builder_dispatches(self) -> None:
        assert set(tele_mod._MODE_FLAG_OPTIONS) <= set(tele_mod._MODE_NUMERIC_OPTIONS)

    def test_replay_is_present_because_it_now_emits_one(self) -> None:
        """``replay`` emitted no flag until it emitted ``--play_sounds``.

        It was absent from the table while its argv carried no flag at all; the
        entry is what gives the one flag it does emit a domain.
        """
        assert tele_mod._MODE_FLAG_OPTIONS["replay"] == ("play_sounds",)
        assert "replay" in tele_mod._MODE_NUMERIC_OPTIONS

    def test_no_mode_names_a_flag_the_supplied_dict_cannot_answer(self) -> None:
        """Guards the ``supplied[param]`` lookup against a typo in the table."""
        every_flag_usable = dict.fromkeys(
            {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}, True
        )
        for mode in tele_mod._MODE_FLAG_OPTIONS:
            assert tele_mod._flag_error(mode, every_flag_usable) is None

    def test_a_mode_absent_from_the_table_refuses_nothing(self) -> None:
        """Every dispatched mode is in the table now, so name one that is not."""
        assert "calibrate" not in tele_mod._MODE_FLAG_OPTIONS
        assert tele_mod._flag_error("calibrate", {}) is None

    def test_the_flags_are_reported_in_the_order_the_argv_emits_them(self) -> None:
        """Two unusable flags in one call must report deterministically."""
        with pytest.raises(ValueError, match="record_resume"):
            _record(record_resume="false", dataset_video="false")

    def test_the_flag_and_numeric_tables_name_disjoint_options(self) -> None:
        """A knob is a magnitude or a posture, never both."""
        numeric = {knob for knobs in tele_mod._MODE_NUMERIC_OPTIONS.values() for knob in knobs}
        flags = {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}
        assert not numeric & flags


class TestTheSharedDomainIsTheOneApplied:
    """The refusal must be the shared one, not a local equivalent of it."""

    def test_the_message_is_the_shared_domains_message(self) -> None:
        expected = boolean_flag_error("false", "dataset_push_to_hub", "build_lerobot_command")
        assert expected is not None
        with pytest.raises(ValueError) as excinfo:
            _record(dataset_push_to_hub="false")
        assert str(excinfo.value) == expected

    def test_the_message_names_the_builder_as_the_context(self) -> None:
        with pytest.raises(ValueError, match="build_lerobot_command"):
            _record(dataset_video="false")

    def test_the_message_explains_why_it_is_not_parsed(self) -> None:
        """A caller who wrote ``"false"`` needs to know it was not read as off."""
        with pytest.raises(ValueError, match="checked rather than parsed"):
            _record(dataset_video="false")


class TestTheToolsOwnExecutionFlagsAreRefusedByTheToolInstead:
    """``background`` and ``auto_accept_calibration`` select an execution posture.

    Neither reaches an argv, so :data:`_MODE_FLAG_OPTIONS` has no entry to scope
    them by: they choose how ``lerobot_teleoperate`` *runs* the command the
    builder returned - detached with a log file and a persisted session, and
    whether two newlines are written into the child's stdin to accept whatever
    calibration prompt the robot is showing. Both were read as ``if <flag>:``, so
    every non-empty string selected the affirmative posture:

    * ``background="false"`` detached the session, rather than running the
      foreground one that was asked for;
    * ``auto_accept_calibration="false"`` accepted a calibration no operator saw.

    The second is the sharper of the two: ``background``'s resulting posture is at
    least *reported* - the envelope carries ``"background": True``, a pid and a
    log file - while nothing anywhere reports that stdin was written to.

    So they are refused by the tool, at the top of the ``start`` / ``dagger``
    branch, against the same shared domain the builder flags use. The class this
    replaces pinned the boundary while they were out of scope; the structural half
    of it is kept below, because "not a builder flag" is still what makes a
    tool-level check the right place for them.
    """

    def _start(self, **overrides: Any) -> dict[str, Any]:
        """A ``start`` call through the tool, with a named session."""
        kwargs: dict[str, Any] = {
            "action": "start",
            "session_name": "exec-flag",
            "robot_type": "so101_follower",
            "robot_port": "/dev/ttyACM1",
            "teleop_type": "so101_leader",
            "teleop_port": "/dev/ttyACM0",
        }
        kwargs.update(overrides)
        return _run_tool(**kwargs)

    # -- the refusal ------------------------------------------------------

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_an_unusable_execution_flag_is_refused(
        self, flag: str, value: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing may be launched or persisted for a refused call.

        Both branches are barred, not just the detaching one: ``background=0`` is
        falsy and would previously have reached ``subprocess.run`` in the
        foreground, which is a posture nobody declared either.
        """
        monkeypatch.setattr(tele_mod.subprocess, "Popen", _never("subprocess.Popen"))
        monkeypatch.setattr(tele_mod.subprocess, "run", _never("subprocess.run"))

        result = self._start(**{flag: value})

        assert result["status"] == "error"
        assert flag in _text(result)
        assert tele_mod.SessionManager().get_session("exec-flag") is None

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    def test_dagger_refuses_it_without_the_rollout_entry_point(
        self, flag: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same caller mistake must report the same way on any lerobot.

        The check precedes the builder, and therefore the lerobot-version
        preflight inside it, so an unusable flag is named rather than masked by an
        upgrade hint on an install that has no rollout entry point.
        """
        monkeypatch.setattr(tele_mod.importlib.util, "find_spec", lambda *a, **k: None)
        monkeypatch.setattr(tele_mod.subprocess, "Popen", _never("subprocess.Popen"))

        result = self._start(action="dagger", policy_path="lerobot/act_so101", **{flag: "false"})

        assert result["status"] == "error"
        assert flag in _text(result)

    def test_the_refusal_precedes_the_argv_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused call must not build a command line at all."""
        monkeypatch.setattr(tele_mod, "build_lerobot_command", _never("build_lerobot_command"))

        result = self._start(background="false")

        assert result["status"] == "error"
        assert "background" in _text(result)

    def test_the_refusal_precedes_the_duplicate_session_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The flag is named, not the session, when both would be reported."""
        monkeypatch.setattr(tele_mod.subprocess, "Popen", _never("subprocess.Popen"))
        tele_mod.SessionManager().add_session("exec-flag", {"pid": 1})

        result = self._start(background="false")

        assert result["status"] == "error"
        text = _text(result)
        assert "background" in text
        assert "already exists" not in text

    def test_the_refusal_is_an_error_envelope_and_not_a_raise(self) -> None:
        """Matching every other refusal on this surface; the tool returns."""
        result = self._start(auto_accept_calibration="false")
        assert result["status"] == "error"
        assert result["content"][0]["text"]

    # -- the shared domain ------------------------------------------------

    def test_the_message_is_the_shared_domains_message(self) -> None:
        expected = boolean_flag_error("false", "background", "lerobot_teleoperate")
        assert expected is not None
        assert _text(self._start(background="false")) == expected

    def test_the_message_names_the_tool_and_not_the_builder(self) -> None:
        """These are the tool's own parameters, so the context must say so."""
        text = _text(self._start(auto_accept_calibration="false"))
        assert "lerobot_teleoperate" in text
        assert "build_lerobot_command" not in text

    def test_the_message_explains_why_it_is_not_parsed(self) -> None:
        assert "checked rather than parsed" in _text(self._start(background="off"))

    # -- the postures a usable flag still selects -------------------------

    @pytest.mark.parametrize("value, expected", A_BOOLEAN)
    def test_a_usable_background_still_selects_its_posture(
        self, value: Any, expected: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``np.False_`` must still reach the foreground run, not be refused."""
        calls: list[str] = []
        monkeypatch.setattr(tele_mod.subprocess, "Popen", _record_popen(calls))
        monkeypatch.setattr(tele_mod.subprocess, "run", _record_run(calls))

        result = self._start(background=value)

        assert result["status"] == "success"
        assert calls == ["Popen" if expected else "run"]

    @pytest.mark.parametrize("value, expected", A_BOOLEAN)
    def test_a_usable_auto_accept_still_selects_its_posture(
        self, value: Any, expected: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stdin pipe is the decision; ``np.False_`` must withhold it."""
        calls: list[str] = []
        popen = _record_popen(calls)
        monkeypatch.setattr(tele_mod.subprocess, "Popen", popen)

        assert self._start(auto_accept_calibration=value)["status"] == "success"
        assert ("stdin" in popen.kwargs) is expected

    # -- scope ------------------------------------------------------------

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    def test_an_action_that_reads_neither_flag_refuses_nothing(self, flag: str) -> None:
        """Refusing a flag an action never reads would be a false rejection."""
        assert _run_tool(action="list", **{flag: "false"})["status"] == "success"

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    def test_replay_is_out_of_scope_because_it_reads_neither(self, flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replay runs in the foreground unconditionally and answers no prompt."""
        calls: list[str] = []
        monkeypatch.setattr(tele_mod.subprocess, "run", _record_run(calls))

        result = _run_tool(
            action="replay",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM1",
            dataset_repo_id="user/pick",
            **{flag: "false"},
        )

        assert result["status"] == "success"
        assert calls == ["run"]

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    def test_they_are_tool_parameters_and_not_builder_parameters(self, flag: str) -> None:
        """The reason there is no per-mode table to route them through."""
        assert flag in inspect.signature(lerobot_teleoperate).parameters
        assert flag not in inspect.signature(build_lerobot_command).parameters

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    def test_the_builder_still_neither_reads_nor_refuses_them(self, flag: str) -> None:
        """They arrive in ``**kwargs`` and are ignored, as they always were."""
        assert _record(**{flag: "false"}) == _record()

    # -- the two checks cannot drift into each other ----------------------

    def test_the_execution_flags_are_named_by_no_mode(self) -> None:
        """A flag is a posture of the tool or of the argv, never both."""
        emitted = {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}
        numeric = {knob for knobs in tele_mod._MODE_NUMERIC_OPTIONS.values() for knob in knobs}
        assert not set(tele_mod._EXECUTION_FLAGS) & (emitted | numeric)

    def test_every_execution_flag_is_a_bool_parameter_of_the_tool(self) -> None:
        params = inspect.signature(lerobot_teleoperate).parameters
        adrift = sorted(
            flag
            for flag in tele_mod._EXECUTION_FLAGS
            if flag not in params or params[flag].annotation not in (bool, "bool")
        )
        assert not adrift, f"_EXECUTION_FLAGS names non-bool non-parameters: {adrift}"

    def test_no_execution_flag_is_left_unchecked(self) -> None:
        """Both are checked, and the check is reached through the shared domain."""
        usable = dict.fromkeys(tele_mod._EXECUTION_FLAGS, True)
        assert tele_mod._execution_flag_error(usable) is None
        for flag in tele_mod._EXECUTION_FLAGS:
            error = tele_mod._execution_flag_error({**usable, flag: "false"})
            assert error == boolean_flag_error("false", flag, "lerobot_teleoperate")

    def test_every_boolean_the_tool_refuses_is_described_to_the_model(self) -> None:
        """A model choosing whether to withhold a flag reads this string.

        The generated placeholder ``"Parameter <name>"`` says nothing about the
        posture it selects, and a flag that is now refused for being mis-spelled
        is one an agent needs the description of.
        """
        params = inspect.signature(lerobot_teleoperate).parameters
        schema = lerobot_teleoperate.tool_spec["inputSchema"]["json"]["properties"]
        undocumented = sorted(
            name
            for name, param in params.items()
            if param.annotation in (bool, "bool")
            and schema[name].get("description", "").startswith(f"Parameter {name}")
        )
        assert not undocumented, f"boolean flags with a placeholder description: {undocumented}"
