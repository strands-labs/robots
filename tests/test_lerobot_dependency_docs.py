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
    # transformers floor matches lerobot 0.6's transformers-dep (>=5.4.0), NOT 5.3.0
    assert "transformers>=5.4.0" in joined, joined
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
