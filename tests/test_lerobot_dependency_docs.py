"""Keep the VLA install docs consistent with the declared dependencies.

The ``lerobot>=0.6`` bump obsoleted a body of pre-0.6 install guidance that
lived in the ``train_policy`` tool docstring and the policy/training docs:

* ``pip install 'lerobot[smolvla]==0.5.1'`` and a ``transformers==5.3.0`` pin -
  lerobot 0.6's ``[smolvla]``/``[pi]``/``[molmoact2]`` extras now require
  ``transformers>=5.4.0,<5.6.0`` (declared as ``transformers-dep``), so a
  ``transformers==5.3.0`` pin is a hard resolution conflict, not a fix. The
  historical "a newer transformers crashes the VLA import with
  ``non-default argument 'backbone_cfg' follows default argument``" note no
  longer applies to the supported range.
* "MolmoAct2 requires lerobot **from source**" - ``MolmoAct2Policy`` ships in
  lerobot >= 0.6, so ``strands-robots[molmoact2]`` (which pulls
  ``strands-robots[lerobot]`` -> ``lerobot>=0.6.1``) resolves it straight from
  PyPI; no ``git+`` install.

These assertions pin the pyproject reality and forbid the stale guidance from
creeping back into the user-facing docs.
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from strands_robots import dataset_recorder

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _bucket_cli_floor_spec() -> str:
    """The requirement string the library's bucket-sync guidance must quote."""
    return dataset_recorder._HF_BUCKET_CLI_MIN_SPEC


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads(_PYPROJECT.read_text())
    return data["project"]["optional-dependencies"]


# --- positive contract: the pyproject reality the docs must reflect ---


def test_lerobot_extra_requires_0_6_1() -> None:
    """The ``[lerobot]`` extra floors lerobot at >= 0.6.1.

    0.6.1 is the first release whose ``StreamingLeRobotDataset`` accepts
    ``repo_type``, so flooring there makes bucket streaming resolver-guaranteed
    rather than docs-guaranteed.
    """
    req = next(r for r in map(Requirement, _extras()["lerobot"]) if r.name == "lerobot")
    lower = min(Version(s.version) for s in req.specifier if s.operator == ">=")
    assert lower >= Version("0.6.1"), f"lerobot extra no longer floors >=0.6.1: {req.specifier}"


def test_smolvla_extra_defers_to_lerobot_smolvla() -> None:
    """A first-class ``smolvla`` extra must exist and defer to lerobot's own.

    SmolVLA is a documented ``lerobot_local`` policy type (``resolution.py``
    lists it, ``policy.py`` imports ``transformers`` for its tokenizer), but
    none of its aux deps arrive through the base ``[lerobot]`` extra
    (``lerobot[feetech,dataset]`` -- no transformers/num2words/accelerate). So
    a ``strands-robots[lerobot]`` install running SmolVLA fails at first use.
    The ``smolvla`` extra defers to lerobot's own ``[smolvla]`` extra (which
    pulls ``lerobot[transformers-dep]`` >=5.4.0 + num2words + accelerate),
    mirroring how ``[molmoact2]`` defers to ``lerobot[molmoact2]`` so the two
    stay in lock-step and share one transformers specifier across ``[all]``.
    """
    extras = _extras()
    assert "smolvla" in extras, "no `smolvla` optional-dependency group in pyproject"
    joined = " ".join(extras["smolvla"])
    # defers to lerobot's own extra rather than hand-mirroring transformers/num2words
    assert "lerobot[smolvla]" in joined, joined
    # and layers it on the base strands-robots[lerobot] extra
    assert "strands-robots[lerobot]" in joined, joined
    # the lerobot floor is >=0.6 (SmolVLAPolicy is a lerobot 0.6 policy), so its
    # transformers-dep (>=5.4.0) resolves - never the pre-0.6 transformers==5.3.0.
    # A bound, not a literal, so a floor raise cannot strand a still-satisfied guard.
    req = next(r for r in map(Requirement, extras["smolvla"]) if r.name == "lerobot")
    lower = min(Version(s.version) for s in req.specifier if s.operator == ">=")
    assert lower >= Version("0.6"), f"smolvla lerobot floor is below 0.6: {req.specifier}"
    # resolves from PyPI - no git-from-source URL
    assert "git+" not in joined, f"smolvla extra should not need a git URL: {joined!r}"
    # the umbrella [all] extra installs it, so `pip install strands-robots[all]`
    # can run every documented policy type (including SmolVLA).
    assert "strands-robots[smolvla]" in extras["all"], "smolvla missing from the [all] extra"


def test_molmoact2_extra_is_pure_pypi_with_transformers_5_4_plus() -> None:
    joined = " ".join(_extras()["molmoact2"])
    # The molmoact2 extra defers to lerobot's own [molmoact2] extra for its
    # transformers/peft/scipy floors instead of hand-mirroring them here (which
    # silently drifts when lerobot bumps them). lerobot[molmoact2] pulls
    # lerobot[transformers-dep] (>=5.4.0) transitively, so the >=5.4.0 guarantee
    # is preserved by construction while staying in lock-step with lerobot.
    assert "lerobot[molmoact2]" in joined, joined
    # the lerobot floor is >=0.6 (MolmoAct2Policy landed in lerobot 0.6), so its
    # transformers-dep (>=5.4.0) is what gets resolved - never the pre-0.6
    # transformers==5.3.0. Asserted as a bound, not a literal substring, so a
    # floor raise cannot strand a guard whose requirement it still satisfies.
    req = next(r for r in map(Requirement, _extras()["molmoact2"]) if r.name == "lerobot")
    lower = min(Version(s.version) for s in req.specifier if s.operator == ">=")
    assert lower >= Version("0.6"), f"molmoact2 lerobot floor is below 0.6: {req.specifier}"
    # resolves from PyPI - no git-from-source URL
    assert "git+" not in joined, f"molmoact2 extra should not need a git URL: {joined!r}"


# --- negative contract: stale pre-0.6 guidance must be gone from the docs ---

_TRAIN_POLICY = _REPO_ROOT / "strands_robots" / "tools" / "train_policy.py"
_TRAINING_OVERVIEW = _REPO_ROOT / "docs" / "training" / "overview.md"
_LEROBOT_LOCAL = _REPO_ROOT / "docs" / "policies" / "lerobot-local.md"


def test_train_policy_tool_has_no_stale_transformers_pin() -> None:
    text = _TRAIN_POLICY.read_text()
    # the "a newer transformers crashes the VLA import (backbone_cfg)" crash lore
    # only ever appeared in this stale bullet; it no longer applies to lerobot
    # 0.6's supported transformers>=5.4.0 range.
    assert "backbone_cfg" not in text
    # no longer claims lerobot's extra "pins transformers==5.3.0"
    assert "pins ``transformers==5.3.0``" not in text
    # documents the current (lerobot 0.6) floor instead
    assert "transformers>=5.4.0" in text


def test_training_overview_has_no_stale_vla_install_lore() -> None:
    text = _TRAINING_OVERVIEW.read_text()
    # the pre-0.6 "pin transformers==5.3.0 / lerobot 0.5.1" recommendation +
    # the backbone_cfg crash lore are gone; the current transformers floor stays.
    assert "backbone_cfg" not in text
    assert "lerobot[smolvla]==0.5.1" not in text
    assert "lerobot[pi]==0.5.1" not in text
    assert "transformers>=5.4.0" in text


def test_lerobot_local_docs_do_not_claim_molmoact2_needs_source() -> None:
    text = _LEROBOT_LOCAL.read_text()
    assert "requires lerobot installed **from source**" not in text
    assert "resolves lerobot 0.5.1, which does NOT" not in text
    # points at the PyPI extra instead
    assert "strands-robots[molmoact2]" in text


# --- negative contract, extended: the pre-0.6 "lerobot from source / <0.6 pin"
#     narrative also lingered in the architecture / troubleshooting / molmoact2
#     pages after the >=0.6 floor bump. These pin it out. ---

_ARCHITECTURE = _REPO_ROOT / "docs" / "architecture.md"
_TROUBLESHOOTING = _REPO_ROOT / "docs" / "troubleshooting.md"
_MOLMOACT2 = _REPO_ROOT / "docs" / "policies" / "molmoact2.md"
_INSTALLATION = _REPO_ROOT / "docs" / "getting-started" / "installation.md"


def _lerobot_floor_from_pyproject() -> str:
    """The exact version specifier the ``[lerobot]`` extra declares (e.g. ``>=0.6.0,<0.7.0``)."""
    for spec in _extras()["lerobot"]:
        if spec.startswith("lerobot["):  # lerobot[feetech,dataset]>=0.6.0,<0.7.0
            return spec.split("]", 1)[1]
    raise AssertionError("no lerobot extra spec found in pyproject")


def test_architecture_lerobot_extra_row_matches_pyproject_floor() -> None:
    text = _ARCHITECTURE.read_text()
    # the dependency-matrix row must not advertise the dead pre-0.6 cap
    assert "lerobot>=0.5.0,<0.6.0" not in text, "architecture.md still cites the dead <0.6.0 lerobot cap"
    # It must reflect the real floor, read from pyproject rather than hardcoded:
    # a floor bump would otherwise leave this page citing a dead version while
    # the guard kept passing.
    floor = _lerobot_floor_from_pyproject()
    assert f"lerobot{floor}" in text, f"architecture.md [lerobot] row should name the pyproject floor lerobot{floor}"


def test_troubleshooting_version_skew_remedy_does_not_conflict_with_floor() -> None:
    text = _TROUBLESHOOTING.read_text()
    # remedying a version-skew ImportError by pinning ``<0.6`` directly conflicts
    # with the pyproject floor (``lerobot[...]>=0.6.1``); the remedy must (re)install
    # the extra instead of a manual sub-floor pin.
    assert "lerobot>=0.5.0,<0.6" not in text, "troubleshooting remedy pins lerobot below the required >=0.6.1 floor"
    assert "strands-robots[lerobot]" in text


def test_troubleshooting_molmoact2_is_pypi_not_from_source() -> None:
    text = _TROUBLESHOOTING.read_text()
    # MolmoAct2Policy ships in lerobot >= 0.6 (PyPI); no from-source git+ remedy.
    assert "git+https://github.com/huggingface/lerobot" not in text
    assert "not in PyPI lerobot" not in text
    # the remedy is the [molmoact2] extra
    assert "strands-robots[molmoact2]" in text


def _md_heading_slugs(text: str) -> set[str]:
    """GitHub/mkdocs heading slugs: lowercase, spaces->'-', drop non-alnum/non-space/non-hyphen."""
    slugs: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        title = s.lstrip("#").strip()
        slug = "".join(c for c in title.lower() if c.isalnum() or c in " -").replace(" ", "-")
        slugs.add(slug)
    return slugs


def test_troubleshooting_jetson_anchor_resolves_to_a_real_heading() -> None:
    """The Jetson/pyav row links into installation.md; that anchor must exist (was broken)."""
    text = _TROUBLESHOOTING.read_text()
    # the stale, non-existent anchor is gone
    assert "molmoact2-on-jetson-lerobot-from-source" not in text
    # and the anchor it now links to resolves to a real installation.md heading
    slugs = _md_heading_slugs(_INSTALLATION.read_text())
    assert "molmoact2-on-jetson" in slugs, "installation.md lost the '### MolmoAct2 on Jetson' heading"
    assert "getting-started/installation.md#molmoact2-on-jetson" in text


def test_molmoact2_doc_install_line_is_not_from_source() -> None:
    text = _MOLMOACT2.read_text()
    assert "lerobot from source" not in text
    assert "[molmoact2]" in text


# --- negative contract: lerobot renamed the training entrypoint module
#     ``lerobot.scripts.train`` -> ``lerobot.scripts.lerobot_train`` (the script
#     rename wave; the old module is removed, so ``python -m
#     lerobot.scripts.train`` now raises ``ModuleNotFoundError``). The rest of the
#     codebase already uses the current name (``strands_robots.training.lerobot``,
#     ``strands_robots.tools.lerobot_train``, ``docs/training/overview.md``); these
#     two user-facing "how to train" spots lagged. Pin the dead module out and
#     require the current one. ---

_STREAMING_DATASET = _REPO_ROOT / "strands_robots" / "streaming_dataset.py"
_RECORDING = _REPO_ROOT / "docs" / "recording.md"


def test_no_userfacing_file_invokes_removed_lerobot_scripts_train() -> None:
    for path in (_STREAMING_DATASET, _RECORDING):
        text = path.read_text()
        assert "lerobot.scripts.train" not in text, (
            f"{path.name} instructs the removed `python -m lerobot.scripts.train`; "
            "lerobot renamed the trainer module to `lerobot.scripts.lerobot_train`"
        )
        # each file documents a train invocation, so the current module name must
        # be the one it points at
        assert "lerobot.scripts.lerobot_train" in text, (
            f"{path.name} lost its `lerobot.scripts.lerobot_train` reference"
        )


# --- negative contract: storage-buckets blog-draft corrections (#1507). The
#     blog draft was copied from these in-repo spots, so the wrong text has to
#     be pinned out at the source or it re-enters the next draft:
#     * README's streamed-training one-liner used the removed
#       ``python -m lerobot.scripts.train`` module AND Hydra-style ``key=value``
#       args; the trainer is the ``lerobot-train`` entry point
#       (``lerobot.scripts.lerobot_train``) with draccus ``--dotted.key=value``
#       flags.
#     * "``pip install -U huggingface_hub``" resolves to 0.x in many envs,
#       whose ``hf`` CLI has no ``buckets``/``sync`` subcommands - the install
#       line must pin ``>=1.0`` everywhere ``sync_to_bucket`` is documented.
#     * The shard-size claim understated lerobot's defaults: 100 MB is the
#       data-parquet default; video MP4 shards default to 200 MB. ---

_README = _REPO_ROOT / "docs" / "recording.md"  # the bucket / streamed-training guidance page (was README)
_DATASET_RECORDER = _REPO_ROOT / "strands_robots" / "dataset_recorder.py"


def test_readme_streamed_training_invocation_is_current() -> None:
    text = _README.read_text()
    assert "lerobot.scripts.train" not in text, (
        "docs/recording.md instructs the removed `python -m lerobot.scripts.train`; "
        "lerobot renamed the trainer module to `lerobot.scripts.lerobot_train`"
    )
    # the documented invocation is the entry point with draccus --dotted flags
    assert "lerobot-train" in text, "docs/recording.md lost its `lerobot-train` reference"
    assert "--dataset.streaming=true" in text, (
        "docs/recording.md streamed-training example must use draccus `--dotted.key=value` flags, not Hydra `key=value` args"
    )


def test_hf_cli_install_guidance_pins_the_bucket_cli_floor() -> None:
    # `pip install -U huggingface_hub` (unversioned) resolves to whatever is
    # newest, and an environment pinned below the floor resolves to a CLI
    # without the `buckets`/`sync` subcommands; every install line next to
    # `sync_to_bucket` guidance must name the floor that ships them.
    floor = _bucket_cli_floor_spec()
    for path in (_README, _DATASET_RECORDER):
        text = path.read_text()
        assert "pip install -U huggingface_hub" not in text, (
            f"{path.name} recommends an unversioned huggingface_hub install; "
            f"the `hf buckets`/`hf sync` subcommands need {floor}"
        )
        assert floor in text, f"{path.name} lost the {floor} pin"


def test_shard_size_claim_names_both_lerobot_defaults() -> None:
    text = _DATASET_RECORDER.read_text()
    # lerobot defaults: 100 MB data parquet / 200 MB video MP4 - "100 MB
    # default" alone understates the video shard size.
    assert "100 MB default" not in text, (
        "dataset_recorder.py understates the shard defaults; lerobot uses 100 MB data parquet / 200 MB video MP4"
    )
    assert "100 MB data parquet / 200 MB video" in text


# --- negative contract: lerobot 0.6 relocated + renamed the codec allowlist.
#     ``VALID_VIDEO_CODECS`` moved from ``lerobot.datasets.video_utils`` to
#     ``lerobot.configs.video``, the hardware-encoder set was renamed
#     ``HW_ENCODERS`` -> ``HW_VIDEO_CODECS``, and ``libaom-av1`` joined the set.
#     ``dataset_recorder.py``'s codec-routing header/docstring still documented
#     the pre-0.6 snapshot (a ``>=0.5.0,<0.6.0`` "supported range" and a
#     ``video_utils.VALID_VIDEO_CODECS ... | HW_ENCODERS`` allowlist) even though
#     the ``[lerobot]`` extra floors lerobot at >=0.6.0 - so the code comment
#     contradicted both the pyproject floor and the installed lerobot API. ---


def test_dataset_recorder_codec_docs_track_the_supported_lerobot_floor() -> None:
    text = _DATASET_RECORDER.read_text()
    # the dead pre-0.6 "supported range" claim is gone (the extra floors >=0.6.0)
    assert ">=0.5.0,<0.6.0" not in text, "dataset_recorder.py still cites the dead pre-0.6 supported lerobot range"
    # the hardware-encoder set was renamed in lerobot 0.6; the stale symbol must
    # not linger in the documented allowlist
    assert "HW_ENCODERS" not in text, "dataset_recorder.py cites the renamed HW_ENCODERS symbol (now HW_VIDEO_CODECS)"
    # and the current allowlist symbols/entries are the ones documented
    assert "HW_VIDEO_CODECS" in text
    assert "libaom-av1" in text


# --- positive contract: the [wbc] extra's huggingface_hub floor must guarantee
#     the `hf buckets`/`hf sync` CLI subcommands the bucket-sync docs instruct.
#     Those subcommands first ship in huggingface_hub 1.5.0; the docs (README
#     streamed-training section + dataset_recorder.sync_to_bucket, pinned by
#     test_hf_cli_install_guidance_pins_the_bucket_cli_floor) tell users to
#     `pip install -U "huggingface_hub>=1.5"`, so a fresh
#     `pip install strands-robots[wbc]` that resolves an `hf` entry point WITHOUT
#     those subcommands (0.36.x, but equally 1.0-1.4.x, satisfy a <1.5 floor)
#     silently reproduces the exact "hf CLI not found / no such subcommand"
#     failure the docs' own error message calls out. Floor the direct pin at the
#     capability version so the resolver can't drift below the documented
#     minimum. See issue #1549. ---


def _wbc_huggingface_hub_spec() -> str:
    """The exact version specifier the ``[wbc]`` extra declares for huggingface_hub."""
    for spec in _extras()["wbc"]:
        # normalize the dist name (huggingface_hub / huggingface-hub) before matching
        name = spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
        if name.replace("-", "_").strip() == "huggingface_hub":
            return spec
    raise AssertionError("no huggingface_hub pin found in the [wbc] extra")


def test_wbc_extra_huggingface_hub_floor_ships_the_bucket_cli() -> None:
    spec = _wbc_huggingface_hub_spec()
    # Assert the declared lower BOUND, not a `">=X" in spec` substring: a later
    # floor raise falsifies the substring while the property it stands for -
    # "the resolved `hf` CLI carries the buckets/sync subcommands" - still holds.
    lower = min(Version(s.version) for s in Requirement(spec).specifier if s.operator == ">=")
    minimum = Version(".".join(str(part) for part in dataset_recorder._HF_BUCKET_CLI_MIN_VERSION))
    assert lower >= minimum, (
        f"[wbc] huggingface_hub floor must be >= {minimum} (the `hf buckets`/`hf sync` "
        f"CLI subcommands the bucket-sync docs instruct first ship there); got {spec!r}"
    )
    # keep the MAJOR cap (<2.0.0) per repo convention (>=1.0 deps cap the major)
    assert "<2.0.0" in spec, f"[wbc] huggingface_hub pin lost its <2.0.0 major cap: {spec!r}"


# --- negative contract: the training docs must name a stack that can actually
#     train. LeRobot's ``train()`` calls
#     ``require_package("accelerate", extra="training")`` *before* it branches on
#     device, so "no GPU" does not mean "no extra" -- and nothing on the
#     ``lerobot_local`` path pulls ``accelerate`` in (the ``[lerobot]`` extra is
#     exactly ``lerobot[feetech,dataset]``). ``docs/training/overview.md`` called
#     that row "works out of the box", which is false for the one thing the page
#     is about: a reader following it gets ``'accelerate' is required but not
#     installed`` on the first ``train()``, on CPU and GPU alike, and -- because
#     ``train()`` reports the failure in its ``TrainResult`` rather than raising --
#     an unchecked call passes a ``checkpoint_dir`` of ``None`` on instead.
#
#     The notebooks state the requirement (pinned by
#     ``tests/test_notebook_min_version_docs.py``); the canonical training page
#     contradicted them. These keep both it and the troubleshooting table from
#     drifting back. ---

# every extra a `lerobot_local` user could install to reach `trainer.train(...)`
_LEROBOT_PATH_EXTRAS = ("lerobot", "lerobot-async", "molmoact2", "all")


def _extras_declaring_accelerate() -> set[str]:
    return {
        name
        for name, reqs in _extras().items()
        if any(Requirement(r).name.replace("-", "_").lower() == "accelerate" for r in reqs)
    }


def test_no_lerobot_path_extra_declares_accelerate() -> None:
    """The premise the docs' ``lerobot[training]`` instruction rests on.

    If a ``strands-robots`` extra on this path ever *does* vendor ``accelerate``
    -- the deferred ``training`` extra that would layer ``lerobot[training]`` the
    way ``lerobot-async`` layers ``lerobot[async]`` -- then the instruction is
    obsolete and the docs must be re-cut to name that extra instead. Failing here
    is the signal to do that, not to loosen the assertion.
    """
    declaring = _extras_declaring_accelerate()
    # non-vacuity: the matcher really does find `accelerate` where it is declared,
    # so the emptiness below is a fact about these extras and not a broken parse.
    assert declaring, "no extra declares accelerate at all - the requirement matcher is not matching"
    available = _extras()
    for name in _LEROBOT_PATH_EXTRAS:
        assert name in available, f"[{name}] extra vanished from pyproject; update _LEROBOT_PATH_EXTRAS"
        assert name not in declaring, (
            f"[{name}] now declares accelerate, so `pip install 'strands-robots[{name}]'` can train "
            "on its own; docs/training/overview.md and docs/troubleshooting.md must stop instructing "
            "`lerobot[training]` and name this extra instead"
        )


def _unwrapped(text: str) -> str:
    """Collapse whitespace runs so a prose assertion survives a reflow.

    Markdown joins the lines of a paragraph when it renders, so where a sentence
    happens to wrap is not semantic -- and a pin that reads the raw file cannot
    tell "the claim was removed" from "the paragraph was re-filled one column
    narrower". Only the rendered wording is asserted below.
    """
    return " ".join(text.split())


def test_training_overview_names_the_trainer_extra() -> None:
    text = _unwrapped(_TRAINING_OVERVIEW.read_text())
    # the false claim: [lerobot] alone cannot run train() on any device
    assert "works out of the box" not in text, (
        "docs/training/overview.md calls the lerobot_local ACT/diffusion install "
        "'works out of the box', but train() refuses without accelerate, which no "
        "extra on that path declares"
    )
    assert "extra is enough for **ACT / diffusion from" not in text, (
        "docs/training/overview.md still claims the [lerobot] extra alone is enough to train from scratch"
    )
    # the remedy, and why it applies with no GPU in sight
    assert "lerobot[training]" in text, "docs/training/overview.md lost the lerobot[training] requirement"
    assert 'require_package("accelerate", extra="training")' in text, (
        "docs/training/overview.md should name the call that refuses, so the "
        "'CPU needs it too' claim is checkable rather than asserted"
    )
    assert "on CPU as well as GPU" in text


def test_troubleshooting_has_a_remedy_for_the_missing_trainer_extra() -> None:
    text = _unwrapped(_TROUBLESHOOTING.read_text())
    # the symptom a reader actually sees, verbatim from lerobot's require_package
    assert "'accelerate' is required but not installed" in text, (
        "docs/troubleshooting.md has no row for the missing-accelerate training failure"
    )
    assert 'uv pip install "lerobot[training]"' in text, (
        "docs/troubleshooting.md names the accelerate symptom without the lerobot[training] remedy"
    )


# --- negative contract: no docs install line prescribes a numpy that the
#     lerobot the [lerobot] extra resolves cannot run on. The Jetson block told
#     readers to `uv pip install "numpy<2" "pandas==2.1.4"` before installing
#     `strands-robots[sim-mujoco,lerobot]`; lerobot >= 0.6 declares
#     numpy>=2.0.0,<2.3.0, so the resolver replaced the pin on the very next
#     line and the only thing the pre-pin conveyed was a numpy-1 requirement
#     that does not exist. Derived from lerobot's own metadata rather than
#     spelled here, so a future numpy range change re-grades the docs. ---


def _lerobot_numpy_specifier() -> SpecifierSet:
    """The numpy range the installed lerobot declares (e.g. ``>=2.0.0,<2.3.0``)."""
    reqs = [Requirement(r) for r in metadata.requires("lerobot") or ()]
    numpy_reqs = [r for r in reqs if canonicalize_name(r.name) == "numpy" and r.marker is None]
    assert numpy_reqs, "installed lerobot declares no unconditional numpy requirement"
    spec = SpecifierSet()
    for req in numpy_reqs:
        spec &= req.specifier
    return spec


def _numpy_versions_lerobot_accepts() -> list[Version]:
    """Concrete numpy versions inside lerobot's declared range.

    The lower bound of every ``>=`` is itself an accepted release, and the numpy
    resolved into this environment alongside lerobot is another when it agrees
    with the declared range. A docs pin has to admit at least one of them.
    """
    spec = _lerobot_numpy_specifier()
    probes = [Version(s.version) for s in spec if s.operator == ">="]
    installed = Version(metadata.version("numpy"))
    if spec.contains(installed):
        probes.append(installed)
    assert probes, f"cannot derive an accepted numpy version from {spec}"
    return probes


def _code_fragments(text: str) -> list[str]:
    """Every command a page presents as code: fenced-block lines and inline spans.

    A pip command reaches a reader either inside a ```bash block (the platform
    install steps) or as an inline span in a troubleshooting table row, and a
    numpy pin is only advice in those two places -- prose that merely names
    ``numpy < 2`` to warn against it must not be graded as an instruction.
    """
    fragments: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            fragments.append(line)
        else:
            fragments.extend(re.findall(r"`([^`\n]+)`", line))
    return fragments


_NUMPY_PIN = re.compile(r"numpy\s*(==|>=|<=|~=|!=|<|>)\s*([0-9][0-9a-zA-Z.*+!-]*)")


def test_no_docs_install_command_pins_a_numpy_lerobot_forbids() -> None:
    """An install step must not pin a numpy outside lerobot's declared range.

    Such a pin cannot survive the install it precedes -- the resolver replaces
    it while pulling lerobot -- so it only misinforms the reader about which
    numpy the stack needs.
    """
    accepted = _numpy_versions_lerobot_accepts()
    offenders: list[str] = []
    for path in sorted((_REPO_ROOT / "docs").rglob("*.md")):
        for fragment in _code_fragments(path.read_text()):
            if "pip install" not in fragment:
                continue
            for operator, version in _NUMPY_PIN.findall(fragment):
                pin = SpecifierSet(f"{operator}{version}")
                if not any(pin.contains(candidate) for candidate in accepted):
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}: {fragment.strip()}")
    assert not offenders, (
        "docs install command pins a numpy that the installed lerobot "
        f"({_lerobot_numpy_specifier()}) forbids, so the same command line undoes it: " + "; ".join(offenders)
    )


def test_numpy_abi_remedy_reinstalls_through_the_lerobot_extra() -> None:
    """The numpy-ABI remedy must resolve through a declared extra, not bare.

    Reinstalling the offending wheel on its own lets the resolver move numpy
    freely: a bare ``uv pip install --reinstall pandas`` next to lerobot 0.6.1
    resolves pandas 3 and numpy 2.5, which lerobot's ``numpy<2.3.0`` forbids.
    Naming the extra keeps the repair inside the ranges the project declares.
    """
    rows = [
        line.split("|")
        for line in _TROUBLESHOOTING.read_text().splitlines()
        if line.startswith("|") and "numpy" in line.split("|")[1] and "Jetson" in line.split("|")[1]
    ]
    assert len(rows) == 1, f"expected exactly one numpy-on-Jetson troubleshooting row, found {len(rows)}"
    remedy = rows[0][3]
    installs = [f for f in _code_fragments(remedy) if "pip install" in f]
    assert installs, "the numpy ABI row names no install command"
    for command in installs:
        assert "strands-robots[" in command, (
            f"the numpy ABI remedy reinstalls a package outside any declared extra: {command!r}; "
            "the resolver is then free to move numpy out of lerobot's range"
        )
