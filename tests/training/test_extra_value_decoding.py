"""``TrainSpec.extra`` text values must mean the same thing on both lerobot paths.

:meth:`LerobotTrainer.build_config` assigns an ``extra`` entry straight to a
field of lerobot's typed config tree, while :meth:`build_command` renders the
same entry as ``--key=value`` for lerobot's draccus CLI - which the module
docstring describes as the argv the typed config *corresponds to*. Assigning a
text value raw broke that correspondence for every non-string field:
``extra={"policy.freeze_vision_encoder": "false"}`` stored the string
``"false"``, which is truthy, so the vision encoder stayed frozen and the run
trained a fraction of the parameters the caller asked for - while the identical
``--policy.freeze_vision_encoder=false`` unfroze it. Nothing raised and nothing
warned.

These tests use lerobot's own CLI parser as the oracle, so they pin agreement
between the two paths rather than a hand-copied list of accepted spellings.
"""

import dataclasses
import json
import re
import typing

import pytest

from strands_robots.training import TrainSpec
from strands_robots.training.lerobot import LerobotTrainer

# Every spelling draccus resolves to a bool, plus the ones it refuses. ``0`` and
# ``1`` are ints to YAML, not bools, so lerobot's CLI rejects them - a fix that
# accepted them would diverge from the CLI just as raw assignment did.
BOOL_SPELLINGS = ["false", "False", "FALSE", "true", "no", "yes", "on", "off"]
NON_BOOL_SPELLINGS = ["0", "1", "maybe"]


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return str(tmp_path)


@pytest.fixture
def spec(dataset_root, tmp_path):
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="",
        output_dir=str(tmp_path / "out"),
        steps=200,
        global_batch_size=8,
        save_freq=100,
        extra={"policy_type": "smolvla"},
    )


@pytest.fixture
def act_spec(spec):
    """A spec on ACT, whose config is the one with a ``str | None`` to clear.

    ``pretrained_backbone_weights`` is the only ``str | None`` field in
    lerobot's policy configs carrying a non-``None`` default, so it is the only
    one where passing ``None`` is a request rather than a restatement - which is
    why the null rendering had no reachable caller until now.
    """
    spec.extra["policy_type"] = "act"
    return spec


def _cli_value(trainer, spec, dotted):
    """Value ``dotted`` takes when ``build_command``'s argv goes through the CLI.

    The oracle: lerobot's own draccus parser reading the argv this same spec
    produces, so the expectation comes from the CLI rather than from the code
    under test.
    """
    import draccus

    # The policy choice registry is populated by importing the config module.
    ptype = spec.extra.get("policy_type", "smolvla")
    __import__(f"lerobot.policies.{ptype}.configuration_{ptype}")
    from lerobot.configs.train import TrainPipelineConfig

    argv = trainer.build_command(spec)[1:]  # drop the "lerobot-train" head
    cfg = draccus.parse(config_class=TrainPipelineConfig, args=argv)
    target = cfg
    *heads, attr = dotted.split(".")
    for head in heads:
        target = getattr(target, head)
    return getattr(target, attr)


def _config_value(trainer, spec, dotted):
    cfg = trainer.build_config(spec)
    target = cfg
    *heads, attr = dotted.split(".")
    for head in heads:
        target = getattr(target, head)
    return getattr(target, attr)


class TestATextValueMeansTheSameOnBothPaths:
    @pytest.mark.parametrize("spelling", BOOL_SPELLINGS)
    def test_a_text_boolean_reaches_the_field_as_a_boolean(self, spec, spelling):
        """A text boolean must arrive as a ``bool``, never as its own text.

        The reported failure: the string ``"false"`` is truthy, so a spec asking
        for an unfrozen vision encoder trained the frozen path and said nothing.
        """
        pytest.importorskip("lerobot")
        spec.extra["policy.freeze_vision_encoder"] = spelling
        got = _config_value(LerobotTrainer(device="cpu"), spec, "policy.freeze_vision_encoder")
        assert isinstance(got, bool), (
            f"extra['policy.freeze_vision_encoder']={spelling!r} reached the config as "
            f"{got!r} ({type(got).__name__}); a non-empty string is truthy, so the encoder "
            f"stays frozen while --policy.freeze_vision_encoder={spelling} unfreezes it"
        )

    @pytest.mark.parametrize("spelling", BOOL_SPELLINGS)
    def test_the_value_agrees_with_lerobots_own_cli_parser(self, spec, spelling):
        """Both paths built from one spec must land on the same value."""
        pytest.importorskip("lerobot")
        pytest.importorskip("draccus")
        trainer = LerobotTrainer(device="cpu")
        spec.extra["policy.freeze_vision_encoder"] = spelling
        expected = _cli_value(trainer, spec, "policy.freeze_vision_encoder")
        got = _config_value(trainer, spec, "policy.freeze_vision_encoder")
        assert got == expected and isinstance(got, type(expected)), (
            f"spelling {spelling!r}: the CLI resolves --policy.freeze_vision_encoder to "
            f"{expected!r} but build_config produced {got!r}"
        )

    def test_a_text_integer_reaches_an_int_field_as_an_int(self, spec):
        """Non-bools decode too - the contract is the field's declared type."""
        pytest.importorskip("lerobot")
        spec.extra["num_workers"] = "6"
        got = _config_value(LerobotTrainer(device="cpu"), spec, "num_workers")
        assert got == 6 and isinstance(got, int), f"num_workers='6' reached the config as {got!r}"


class TestAValueTheCliRefusesIsRefusedHereToo:
    @pytest.mark.parametrize("spelling", NON_BOOL_SPELLINGS)
    def test_a_spelling_that_is_not_a_boolean_is_refused(self, spec, spelling):
        """Silently storing it is how a wrong type reaches training unnoticed."""
        pytest.importorskip("lerobot")
        spec.extra["policy.freeze_vision_encoder"] = spelling
        with pytest.raises(ValueError) as excinfo:
            LerobotTrainer(device="cpu").build_config(spec)
        message = str(excinfo.value)
        assert "policy.freeze_vision_encoder" in message, message
        assert "bool" in message, message

    def test_the_refusal_names_a_spelling_that_works(self, spec):
        """Following the refusal's own advice has to produce a usable value."""
        pytest.importorskip("lerobot")
        trainer = LerobotTrainer(device="cpu")
        spec.extra["policy.freeze_vision_encoder"] = "maybe"
        with pytest.raises(ValueError) as excinfo:
            trainer.build_config(spec)
        offered = re.findall(r"\b(false|true|no|yes|on|off)\b", str(excinfo.value))
        assert offered, f"the refusal names no spelling to use instead: {excinfo.value}"
        for spelling in dict.fromkeys(offered):
            spec.extra["policy.freeze_vision_encoder"] = spelling
            got = _config_value(trainer, spec, "policy.freeze_vision_encoder")
            assert isinstance(got, bool), f"the refusal offered {spelling!r}, which yields {got!r}"


class TestTheFieldTypeIsResolvedNotReadRaw:
    def test_a_string_annotated_bool_field_still_decodes(self, spec):
        """``dataclasses`` reports a *string* type for a postponed annotation.

        Six lerobot policy configs (``eo1``, ``evo1``, ``fastwam``,
        ``molmoact2``, ``vla_jepa``, ``xvla``) are compiled with ``from
        __future__ import annotations``, so ``dataclasses.fields(cls)[i].type``
        is the source text ``"bool"`` for 45 of their boolean fields. Comparing
        that against ``bool`` is False, which would skip exactly those configs,
        so the annotation has to be resolved before it is compared.
        """
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import _decode_extra_value, _extra_field_type

        @dataclasses.dataclass
        class Postponed:
            flag: "bool" = True  # noqa: UP037 - the string form is the subject

            def __post_init__(self) -> None:  # pragma: no cover - shape only
                return None

        declared = {f.name: f.type for f in dataclasses.fields(Postponed)}["flag"]
        assert declared == "bool", f"premise: expected the source text, got {declared!r}"
        assert typing.get_type_hints(Postponed)["flag"] is bool
        assert _extra_field_type(Postponed(), "flag") is bool
        assert _decode_extra_value(Postponed(), "flag", "flag", "false") is False


class TestATypedValueMeansTheSameOnBothPaths:
    """A value already of the field's type must survive the CLI rendering too.

    :func:`_decode_extra_value` returns a non-``str`` unchanged, which settles
    the in-process half - a caller who passed the field's own type never went
    through text. The other half is :meth:`build_command`, which renders every
    entry with an f-string, and Python's text for ``None`` is not YAML's:
    draccus reads ``None`` as a plain scalar, so a ``str | None`` field given
    ``--key=None`` is assigned the *string* ``"None"`` while ``build_config``
    assigns ``None``. For ACT's ``pretrained_backbone_weights`` the string then
    reaches torchvision as a weights *name* (``KeyError: 'None'``) instead of
    as "no pretrained weights", so the subprocess path crashes on the spec the
    in-process path runs.
    """

    def test_a_none_reaches_the_field_as_none(self, act_spec):
        """The in-process half: a typed ``None`` is assigned, not stringified."""
        pytest.importorskip("lerobot")
        act_spec.extra["policy.pretrained_backbone_weights"] = None
        got = _config_value(LerobotTrainer(device="cpu"), act_spec, "policy.pretrained_backbone_weights")
        assert got is None, f"a typed None reached the config as {got!r} ({type(got).__name__})"

    def test_a_none_agrees_with_lerobots_own_cli_parser(self, act_spec):
        """Both paths built from one spec must land on the same value."""
        pytest.importorskip("lerobot")
        pytest.importorskip("draccus")
        trainer = LerobotTrainer(device="cpu")
        act_spec.extra["policy.pretrained_backbone_weights"] = None
        expected = _cli_value(trainer, act_spec, "policy.pretrained_backbone_weights")
        got = _config_value(trainer, act_spec, "policy.pretrained_backbone_weights")
        assert got == expected and isinstance(got, type(expected)), (
            f"the CLI resolves --policy.pretrained_backbone_weights to {expected!r} but build_config produced {got!r}"
        )

    def test_the_rendered_token_is_the_yaml_null_not_pythons(self, act_spec):
        """``None`` renders as ``null``; ``None`` as text is a plain scalar."""
        pytest.importorskip("lerobot")
        act_spec.extra["policy.pretrained_backbone_weights"] = None
        argv = LerobotTrainer(device="cpu").build_command(act_spec)
        rendered = [a for a in argv if a.startswith("--policy.pretrained_backbone_weights=")]
        assert rendered == ["--policy.pretrained_backbone_weights=null"], rendered

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("num_workers", 6),
            ("policy.optimizer_lr", 0.0001),
            # Falsy but not null: a rule keyed on truthiness rather than on
            # ``is None`` would render these as ``null`` and clear the field.
            ("num_workers", 0),
            ("policy.freeze_vision_encoder", False),
        ],
    )
    def test_a_non_null_typed_value_still_renders_as_its_own_text(self, spec, key, value):
        """Only the null literal is translated - everything else is unchanged."""
        pytest.importorskip("lerobot")
        spec.extra[key] = value
        argv = LerobotTrainer(device="cpu").build_command(spec)
        assert f"--{key}={value}" in argv, [a for a in argv if a.startswith(f"--{key}=")]

    def test_a_string_field_still_carries_the_literal_text_none(self, spec):
        """A caller naming something ``None`` on a ``str`` field keeps that name.

        draccus reads the token as a YAML scalar and decodes it to ``str``, so
        ``"None"`` survives as ``"None"`` on both paths. Translating the *string*
        as well as the value would corrupt a legitimate name.
        """
        pytest.importorskip("lerobot")
        pytest.importorskip("draccus")
        trainer = LerobotTrainer(device="cpu")
        spec.extra["wandb.project"] = "None"
        assert "--wandb.project=None" in trainer.build_command(spec)
        assert _cli_value(trainer, spec, "wandb.project") == "None"
        assert _config_value(trainer, spec, "wandb.project") == "None"


class TestNothingElseChanges:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("policy.freeze_vision_encoder", False),
            ("policy.freeze_vision_encoder", True),
            ("num_workers", 0),
            ("num_workers", 6),
        ],
    )
    def test_a_value_already_of_the_fields_type_is_passed_through(self, spec, key, value):
        """A caller who passed a real Python value never went through text."""
        pytest.importorskip("lerobot")
        spec.extra[key] = value
        got = _config_value(LerobotTrainer(device="cpu"), spec, key)
        assert got is value if isinstance(value, bool) else got == value

    @pytest.mark.parametrize("text", ["smolvla_tictactoe", "false", "off"])
    def test_a_string_field_keeps_its_text_verbatim(self, spec, text):
        """A string field is already the caller's type, so it is not decoded.

        This is the one place the two paths differ on purpose: draccus reads a
        CLI token as a YAML scalar first, so ``--wandb.project=false`` reaches a
        ``str`` field as ``"False"``. Round-tripping a value through a boolean
        to retype it as text would corrupt a legitimate name, so a field that
        already admits text keeps exactly what the caller wrote.
        """
        pytest.importorskip("lerobot")
        spec.extra["wandb.project"] = text
        got = _config_value(LerobotTrainer(device="cpu"), spec, "wandb.project")
        assert got == text

    def test_an_unknown_extra_key_is_still_ignored(self, spec, caplog):
        """Decoding must not turn an ignored key into a refusal."""
        pytest.importorskip("lerobot")
        spec.extra["definitely_not_a_field"] = "false"
        with caplog.at_level("WARNING"):
            cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert not hasattr(cfg, "definitely_not_a_field")
        assert any("ignoring extra" in r.message for r in caplog.records)
