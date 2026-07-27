"""Keep the VLA install docs consistent with the declared dependencies.

The ``lerobot>=0.6.0`` bump obsoleted a body of pre-0.6 install guidance that
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
  ``strands-robots[lerobot]`` -> ``lerobot>=0.6.0``) resolves it straight from
  PyPI; no ``git+`` install.

These assertions pin the pyproject reality and forbid the stale guidance from
creeping back into the user-facing docs.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads(_PYPROJECT.read_text())
    return data["project"]["optional-dependencies"]


# --- positive contract: the pyproject reality the docs must reflect ---


def test_lerobot_extra_requires_0_6() -> None:
    joined = " ".join(_extras()["lerobot"])
    assert "lerobot" in joined
    assert ">=0.6.0" in joined, f"lerobot extra no longer pins >=0.6.0: {joined!r}"


def test_molmoact2_extra_is_pure_pypi_with_transformers_5_4_plus() -> None:
    joined = " ".join(_extras()["molmoact2"])
    # The molmoact2 extra defers to lerobot's own [molmoact2] extra for its
    # transformers/peft/scipy floors instead of hand-mirroring them here (which
    # silently drifts when lerobot bumps them). lerobot[molmoact2] pulls
    # lerobot[transformers-dep] (>=5.4.0) transitively, so the >=5.4.0 guarantee
    # is preserved by construction while staying in lock-step with lerobot.
    assert "lerobot[molmoact2]" in joined, joined
    # the lerobot floor is >=0.6.0 (MolmoAct2Policy landed in lerobot 0.6), so
    # its transformers-dep (>=5.4.0) is what gets resolved - never the pre-0.6
    # transformers==5.3.0.
    assert ">=0.6.0" in joined, joined
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
#     pages after the >=0.6.0 floor bump. These pin it out. ---

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
    # it must reflect the real floor (>=0.6.0)
    assert ">=0.6.0" in text, "architecture.md [lerobot] row should name the >=0.6.0 floor"


def test_troubleshooting_version_skew_remedy_does_not_conflict_with_floor() -> None:
    text = _TROUBLESHOOTING.read_text()
    # remedying a version-skew ImportError by pinning ``<0.6`` directly conflicts
    # with the pyproject floor (``lerobot[...]>=0.6.0``); the remedy must (re)install
    # the extra instead of a manual sub-floor pin.
    assert "lerobot>=0.5.0,<0.6" not in text, "troubleshooting remedy pins lerobot below the required >=0.6.0 floor"
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

_README = _REPO_ROOT / "README.md"
_DATASET_RECORDER = _REPO_ROOT / "strands_robots" / "dataset_recorder.py"


def test_readme_streamed_training_invocation_is_current() -> None:
    text = _README.read_text()
    assert "lerobot.scripts.train" not in text, (
        "README.md instructs the removed `python -m lerobot.scripts.train`; "
        "lerobot renamed the trainer module to `lerobot.scripts.lerobot_train`"
    )
    # the documented invocation is the entry point with draccus --dotted flags
    assert "lerobot-train" in text, "README.md lost its `lerobot-train` reference"
    assert "--dataset.streaming=true" in text, (
        "README.md streamed-training example must use draccus `--dotted.key=value` flags, not Hydra `key=value` args"
    )


def test_hf_cli_install_guidance_pins_huggingface_hub_1_0() -> None:
    # `pip install -U huggingface_hub` (unversioned) resolves to 0.36.x in many
    # envs, which has no `buckets`/`sync` subcommands; every install line next
    # to `sync_to_bucket` guidance must pin >=1.0.
    for path in (_README, _DATASET_RECORDER):
        text = path.read_text()
        assert "pip install -U huggingface_hub" not in text, (
            f"{path.name} recommends an unversioned huggingface_hub install; "
            "the `hf buckets`/`hf sync` subcommands need >=1.0"
        )
        assert "huggingface_hub>=1.0" in text, f"{path.name} lost the huggingface_hub>=1.0 pin"


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
#     Those subcommands only exist on huggingface_hub>=1.0; the docs (README
#     streamed-training section + dataset_recorder.sync_to_bucket, pinned by
#     test_hf_cli_install_guidance_pins_huggingface_hub_1_0) tell users to
#     `pip install -U "huggingface_hub>=1.0"`, so a fresh
#     `pip install strands-robots[wbc]` that resolves an `hf` entry point WITHOUT
#     those subcommands (huggingface_hub 0.36.x satisfies a <1.0 floor) silently
#     reproduces the exact "hf CLI not found / no such subcommand" failure the
#     docs' own error message calls out. Floor the direct pin at >=1.0 so the
#     resolver can't drift below the documented minimum. See issue #1549. ---


def _wbc_huggingface_hub_spec() -> str:
    """The exact version specifier the ``[wbc]`` extra declares for huggingface_hub."""
    for spec in _extras()["wbc"]:
        # normalize the dist name (huggingface_hub / huggingface-hub) before matching
        name = spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
        if name.replace("-", "_").strip() == "huggingface_hub":
            return spec
    raise AssertionError("no huggingface_hub pin found in the [wbc] extra")


def test_wbc_extra_huggingface_hub_floor_is_at_least_1_0() -> None:
    spec = _wbc_huggingface_hub_spec()
    # floor at >=1.0 so the resolved `hf` CLI carries the buckets/sync subcommands
    assert ">=1.0" in spec, (
        f"[wbc] huggingface_hub floor must be >=1.0 (the `hf buckets`/`hf sync` "
        f"CLI subcommands the bucket-sync docs instruct only exist on >=1.0); got {spec!r}"
    )
    # the dead pre-1.0 floor must not linger
    assert ">=0.20.0" not in spec, f"[wbc] still pins the dead pre-1.0 huggingface_hub floor: {spec!r}"
    # keep the MAJOR cap (<2.0.0) per repo convention (>=1.0 deps cap the major)
    assert "<2.0.0" in spec, f"[wbc] huggingface_hub pin lost its <2.0.0 major cap: {spec!r}"
