"""``lerobot_train`` refuses a launch topology it cannot honor, before launching it.

``build_train_command`` reads ``num_gpus`` twice, and neither read can screen a
value a local comparison lets through:

* the ``num_gpus > 1`` selector picks ``accelerate launch --multi_gpu`` over a
  direct ``python -m lerobot.scripts.lerobot_train``, and
* the ``--num_processes=`` token sizes that accelerate launch.

A ``num_gpus < 1`` test guards neither. ``nan`` is not less than one and not
greater than one either - it compares false against every bound - so it fell
through to the single-process branch; ``True`` did the same, as ``1``. Both
produced a run on a topology nobody asked for, reported to the caller as
started. ``2.7``, ``2.0`` and ``inf`` *are* greater than one, so they selected
the multi-process path and were written into the argv verbatim, where
accelerate's own ``type=int`` parse rejects them - inside the DETACHED process,
whose only record is the training log. And ``"2"``, ``None`` and ``[2]`` could
not be ordered against ``1`` at all, so the comparison raised ``TypeError``,
which the tool reported as ``Tool execution failed: '<' not supported between
instances of 'str' and 'int'`` - naming neither the parameter nor the remedy.

The trainer surface for the same lerobot run already held this field to the
shared positive-count domain
(:func:`~strands_robots.training._validate.launch_topology_problems`, reached
from ``LerobotTrainer.validate``), so one parameter had two contracts depending
on which surface built the launch - and they disagreed on five of the thirteen
values probed below.

The tests here pin the corrected contract in both directions: every unusable
count is refused with a message naming the parameter, a usable topology still
reaches the launcher it selects, the refusal is an error envelope rather than a
launched process, the premises the domain rests on hold against the real
``accelerate`` parser, and no module re-implements the domain locally - checked
across the package rather than one subpackage, because the two spellings of the
same defect (``spec.num_gpus < 1`` on a spec field and a bare ``num_gpus < 1``
on a parameter) are the same defect and only the first was graded.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
from typing import Any

import pytest

import strands_robots
import strands_robots.tools.lerobot_train as train_mod
from tests.tools.test_lerobot_train import _write_dataset

build_train_command = train_mod.build_train_command

# Rooted at the package so the sweep below cannot silently point at one
# subpackage: this domain's owner is shared, so every consumer of it is in scope.
_PACKAGE_ROOT = pathlib.Path(inspect.getfile(strands_robots)).parent

# The fields the domain covers. ``num_nodes`` is a spec-only field today; it is
# named here so the sweep stays honest if the tool ever grows the flag.
TOPOLOGY_FIELDS = ("num_gpus", "num_nodes")

# Counts no launcher can honor, grouped by how each one failed before the gate.

# Not greater than one, so the selector routed them to the single-process branch
# and the run proceeded on a topology the caller did not ask for.
SILENTLY_SINGLE_PROCESS: tuple[Any, ...] = (float("nan"), True)

# Greater than one, so they selected accelerate and reached its int parse as the
# worker count - inside the detached process.
POISONED_THE_LAUNCH_ARGV: tuple[Any, ...] = (2.7, 2.0, float("inf"))

# Refused before the fix too, by the local comparison. Kept as rows so a
# regression that drops the floor is caught by the same table.
ALREADY_REFUSED: tuple[Any, ...] = (0, -1, False)

# Could not be ordered against ``1``, so the comparison raised out of the tool.
COULD_NOT_BE_COMPARED: tuple[Any, ...] = ("2", None, [2])

UNUSABLE_COUNTS: tuple[Any, ...] = (
    *SILENTLY_SINGLE_PROCESS,
    *POISONED_THE_LAUNCH_ARGV,
    *ALREADY_REFUSED,
    *COULD_NOT_BE_COMPARED,
)


def _build(**kwargs: Any) -> list[str]:
    """Build an argv with a minimal usable base, overridden by ``kwargs``.

    Funnelled so the deliberately off-type values above reach the runtime guard
    the way an agent supplies them, without a type checker objecting at each
    call site.
    """
    base: dict[str, Any] = {"dataset_root": "/data/cubes", "policy_type": "act"}
    base.update(kwargs)
    return build_train_command(**base)


def _start(**kwargs: Any) -> dict[str, Any]:
    """Call the tool's ``start`` action with a minimal base, overridden by ``kwargs``.

    Funnelled for the same reason as :func:`_build`: the deliberately off-type
    counts below must reach the runtime guard the way an agent supplies them,
    which a type checker cannot express against the tool's ``int`` annotation.
    """
    base: dict[str, Any] = {"action": "start"}
    base.update(kwargs)
    return train_mod.lerobot_train(**base)


@pytest.fixture(autouse=True)
def _isolated_sessions(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Keep the on-disk session store inside the test's own tmp_path."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return session_dir


class TestALaunchTopologyThatCannotBeHonoredIsRefused:
    """Every unusable count is refused through the shared positive-count domain."""

    @pytest.mark.parametrize("value", UNUSABLE_COUNTS)
    def test_an_unusable_count_never_reaches_the_launcher(self, value: Any) -> None:
        with pytest.raises(ValueError, match="num_gpus must be a positive integer"):
            _build(num_gpus=value)

    def test_the_refusal_names_the_parameter_the_tool_and_quotes_the_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _build(num_gpus=2.7)
        message = str(excinfo.value)
        assert "num_gpus" in message
        assert "2.7" in message
        assert "lerobot_train" in message

    @pytest.mark.parametrize("value", COULD_NOT_BE_COMPARED)
    def test_an_uncomparable_count_is_a_stated_refusal_not_a_type_error(self, value: Any) -> None:
        """The message must name the parameter, which a leaked ``TypeError`` did not."""
        with pytest.raises(ValueError) as excinfo:
            _build(num_gpus=value)
        assert "num_gpus" in str(excinfo.value)

    def test_a_usable_single_process_topology_selects_the_direct_module_run(self) -> None:
        cmd = _build(num_gpus=1)
        assert cmd[:4] == ["python", "-m", "lerobot.scripts.lerobot_train"][:3] + [cmd[3]]
        assert not [flag for flag in cmd if flag.startswith("--num_processes=")]

    @pytest.mark.parametrize("value", [2, 4, 8])
    def test_a_usable_multi_process_topology_still_sizes_the_accelerate_launch(self, value: int) -> None:
        cmd = _build(num_gpus=value)
        assert cmd[0] == "accelerate"
        assert f"--num_processes={value}" in cmd

    def test_the_topology_is_checked_even_when_the_size_knobs_are_usable(self) -> None:
        """A usable ``steps`` must not mask an unusable ``num_gpus``."""
        with pytest.raises(ValueError, match="num_gpus must be a positive integer"):
            _build(steps=20000, batch_size=8, num_gpus=float("nan"))


class TestTheBuilderMeetsTheTrainerSurfaceContract:
    """The two surfaces that launch the same lerobot run cannot disagree.

    ``LerobotTrainer.validate`` already held ``num_gpus`` to the shared domain.
    Any count it refuses must not be turned into a launcher selection by the
    builder either, or the accepted topology depends on which surface the caller
    reached for.
    """

    @staticmethod
    def _trainer_accepts(value: Any) -> bool:
        from strands_robots.training import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        spec = TrainSpec(dataset_root="/data/cubes", output_dir="/tmp/out", num_gpus=value)
        try:
            problems = LerobotTrainer().validate(spec)
        except TypeError:
            return False
        return not any("num_gpus" in problem for problem in problems)

    @staticmethod
    def _builder_accepts(value: Any) -> bool:
        try:
            _build(num_gpus=value)
        except (ValueError, TypeError):
            return False
        return True

    @pytest.mark.parametrize("value", [*UNUSABLE_COUNTS, 1, 4])
    def test_the_builder_accepts_no_topology_the_trainer_would_refuse(self, value: Any) -> None:
        if not self._trainer_accepts(value):
            assert not self._builder_accepts(value), (
                f"num_gpus={value!r} is refused by the trainer surface but built into the launch"
            )

    def test_the_trainer_surface_really_does_refuse_these(self) -> None:
        """Non-vacuity: the contract being matched is actually enforced there."""
        assert not self._trainer_accepts(float("nan"))
        assert not self._trainer_accepts(2.7)
        assert self._trainer_accepts(4)


class TestTheRefusalReachesTheCallerBeforeAnyProcessStarts:
    """A rejected topology must be reported, not launched and then discovered."""

    def test_the_tool_reports_an_error_envelope_rather_than_raising(self, tmp_path: pathlib.Path) -> None:
        dataset = _write_dataset(tmp_path / "cubes")
        result = _start(dataset_root=str(dataset), num_gpus=float("nan"))
        assert result["status"] == "error"
        text = "\n".join(item["text"] for item in result["content"] if "text" in item)
        assert "num_gpus must be a positive integer" in text

    def test_no_session_is_recorded_for_a_refused_topology(
        self, tmp_path: pathlib.Path, _isolated_sessions: pathlib.Path
    ) -> None:
        dataset = _write_dataset(tmp_path / "cubes")
        _start(dataset_root=str(dataset), num_gpus=2.7)
        assert list(_isolated_sessions.glob("*.json")) == []


class TestTheRefusedValuesAreOnesTheLauncherCannotHonor:
    """The refused set is grounded in what the selector and accelerate do."""

    @pytest.mark.parametrize("value", SILENTLY_SINGLE_PROCESS)
    def test_a_silent_value_reads_as_not_more_than_one(self, value: Any) -> None:
        assert not value > 1

    def test_nan_compares_false_against_both_bounds(self) -> None:
        """Which is why a ``< 1`` floor could not screen it at all."""
        nan = float("nan")
        assert not nan < 1
        assert not nan > 1

    @pytest.mark.parametrize("value", POISONED_THE_LAUNCH_ARGV)
    def test_a_launch_selecting_value_reads_as_more_than_one(self, value: Any) -> None:
        assert value > 1

    @pytest.mark.parametrize("value", POISONED_THE_LAUNCH_ARGV)
    def test_the_real_accelerate_parser_refuses_it_as_a_worker_count(self, value: Any) -> None:
        """So the pre-fix failure was late, inside the detached process."""
        launch = pytest.importorskip("accelerate.commands.launch")
        parser = launch.launch_command_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--multi_gpu", f"--num_processes={value}", "-m", "lerobot.scripts.lerobot_train"])

    def test_the_real_accelerate_parser_accepts_a_usable_count(self) -> None:
        """Non-vacuity for the assertion above."""
        launch = pytest.importorskip("accelerate.commands.launch")
        parser = launch.launch_command_parser()
        parsed = parser.parse_args(["--multi_gpu", "--num_processes=4", "-m", "lerobot.scripts.lerobot_train"])
        assert parsed.num_processes == 4

    @pytest.mark.parametrize("value", COULD_NOT_BE_COMPARED)
    def test_an_uncomparable_count_cannot_be_ordered_against_the_floor(self, value: Any) -> None:
        with pytest.raises(TypeError):
            _ = value < 1  # noqa: B015 - the raise is the assertion

    def test_the_probe_set_covers_both_non_finite_floats(self) -> None:
        """Guards the probe set itself against a future edit dropping a case."""
        floats = [value for value in UNUSABLE_COUNTS if isinstance(value, float)]
        assert any(math.isnan(value) for value in floats)
        assert any(math.isinf(value) for value in floats)


def _compared_field(node: ast.expr) -> str | None:
    """The topology field a comparison's left operand names, in either spelling.

    ``spec.num_gpus`` reads as an attribute and a bare ``num_gpus`` parameter as
    a name; both are the same field, which is the distinction a scan keyed on
    one spelling misses.
    """
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _local_domain_comparisons(source: str) -> list[str]:
    """Comparisons that re-implement the count domain on a topology field.

    Read from the AST rather than the text so prose - a docstring or a comment
    explaining the domain, of which this fix adds several - cannot register as a
    re-implementation. A ``> 1`` comparison is a different question ("is this a
    topology I must launch differently") and is deliberately not matched.
    """
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if _compared_field(node.left) not in TOPOLOGY_FIELDS:
            continue
        operator, right = node.ops[0], node.comparators[0]
        bound = right.value if isinstance(right, ast.Constant) else None
        if (isinstance(operator, ast.Lt) and bound == 1) or (isinstance(operator, ast.LtE) and bound == 0):
            offenders.append(ast.unparse(node))
    return offenders


class TestOneOwnerForTheLaunchTopologyDomain:
    """No module may re-implement the domain, in either spelling, anywhere.

    The population is the package rather than one subpackage: the domain's owner
    is shared, so a second entry point that launches from the field is in scope
    the moment it exists. That widening is the hole this closed - the field was
    graded for a re-implementation only among the trainer backends, where it
    arrives on a spec, so the tool's parameter spelling was never in the
    population.
    """

    def test_no_module_re_implements_the_domain(self) -> None:
        offenders: list[str] = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            for comparison in _local_domain_comparisons(path.read_text(encoding="utf-8")):
                offenders.append(f"{path.relative_to(_PACKAGE_ROOT.parent)}: {comparison}")
        assert offenders == [], f"local domain checks on a launch-topology field: {offenders}"

    def test_the_sweep_reads_a_populated_tree(self) -> None:
        """Non-vacuity: a mis-rooted sweep would report a clean tree of nothing."""
        modules = list(_PACKAGE_ROOT.rglob("*.py"))
        assert len(modules) > 100, f"only {len(modules)} modules found under {_PACKAGE_ROOT}"
        assert (_PACKAGE_ROOT / "tools" / "lerobot_train.py") in modules

    @pytest.mark.parametrize(
        "planted",
        [
            "if num_gpus < 1:\n    raise ValueError('bad')\n",
            "if spec.num_gpus < 1:\n    problems.append('bad')\n",
            "if num_nodes <= 0:\n    raise ValueError('bad')\n",
            "if spec.num_nodes <= 0:\n    problems.append('bad')\n",
        ],
    )
    def test_the_scan_detects_a_planted_defect_in_either_spelling(self, planted: str) -> None:
        assert _local_domain_comparisons(planted), f"missed: {planted!r}"

    @pytest.mark.parametrize(
        "accepted",
        [
            "if num_gpus > 1:\n    cmd = ['accelerate']\n",  # the launcher selector, a different question
            "error = positive_count_error(num_gpus, 'num_gpus', 'lerobot_train')\n",
            "if steps < 1:\n    raise ValueError('bad')\n",  # a different field's own guard
            "'``num_gpus < 1`` is what this replaced'\n",  # prose, not code
        ],
    )
    def test_the_scan_does_not_flag_an_accepted_form(self, accepted: str) -> None:
        assert _local_domain_comparisons(accepted) == [], f"false positive: {accepted!r}"

    def test_the_scan_reads_code_rather_than_the_comment_beside_it(self) -> None:
        """The shipped fix documents the replaced comparison in a comment."""
        source = (_PACKAGE_ROOT / "tools" / "lerobot_train.py").read_text(encoding="utf-8")
        assert "num_gpus > 1" in source, "the selector prose is what a text scan would trip on"
        assert _local_domain_comparisons(source) == []
