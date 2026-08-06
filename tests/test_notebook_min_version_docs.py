"""Pin the minimum ``strands-robots`` version in the bucket-streaming docs.

The streaming-data-loop notebook (and the storage-buckets blog post drafted
from it) demonstrates the bucket path: ``sync_to_bucket`` and
``stream_dataset(..., repo_type="bucket")``.

The floor is 0.5.1 rather than 0.5.0, which is when those APIs landed. 0.5.0
floors lerobot at ``>=0.6.0``, and 0.6.0's ``StreamingLeRobotDataset`` takes no
``repo_type`` -- so a resolver may pair 0.5.0 with a lerobot that cannot serve a
bucket read, and the runtime guard in
:mod:`strands_robots.streaming_dataset` refuses it. 0.5.1 is the first release
whose ``[lerobot]`` extra floors lerobot at the ``0.6.1`` recorded in
``BUCKET_STREAMING_MIN_LEROBOT``, which makes the notebook's headline snippet
resolver-guaranteed rather than luck of the resolver picking the newest lerobot.

These assertions forbid the version-less install guidance from creeping back
into the notebook or the notebooks index (issue #1500).
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NOTEBOOK = _REPO_ROOT / "examples" / "notebooks" / "05_streaming_data_loop.ipynb"
_NOTEBOOKS_README = _REPO_ROOT / "examples" / "notebooks" / "README.md"

_MIN_VERSION = "strands-robots >= 0.5.1"


def _notebook_markdown() -> str:
    nb = json.loads(_NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "markdown")


def test_streaming_notebook_states_min_strands_robots_version() -> None:
    md = _notebook_markdown()
    assert _MIN_VERSION in md, (
        f"{_NOTEBOOK.name} must state the minimum strands-robots version "
        f"({_MIN_VERSION!r}) - it is the first release whose [lerobot] extra "
        "floors lerobot at the 0.6.1 that serves a bucket read (issue #1500)."
    )


def test_streaming_notebook_does_not_offer_bare_pypi_install_as_requirements() -> None:
    md = _notebook_markdown()
    assert '**Requirements:** `pip install "strands-robots' not in md, (
        f"{_NOTEBOOK.name} presents a version-less PyPI install as its "
        "requirements line; a resolve that lands below the declared floor "
        "refuses the notebook's bucket snippet (issue #1500). State the "
        "minimum version."
    )


def test_notebook_install_lines_upgrade_a_stale_environment() -> None:
    """An install line must be able to *replace* an older release, not skip it.

    ``pip install "strands-robots[...]"`` against an environment that already
    carries an older release reports ``Requirement already satisfied`` and
    upgrades nothing -- extras do not make a requirement unsatisfied. The reader
    then runs a notebook against, say, 0.4.1, whose
    ``StreamingDatasetReader.open`` has neither a ``repo_type`` parameter nor
    ``**kwargs``, so the bucket read fails with ``TypeError: open() got an
    unexpected keyword argument 'repo_type'`` -- a message that names the keyword
    rather than the stale install behind it.

    Either form fixes it: ``-U`` (upgrade), or a version floor on the requirement
    (``"strands-robots[...]>=0.5.1"``), which makes the installed release
    genuinely unsatisfying. A git/URL requirement re-resolves on its own, but is
    only reinstalled with ``-U``, so it is held to the same rule.
    """
    notebooks = sorted((_REPO_ROOT / "examples" / "notebooks").glob("0*.ipynb"))
    assert notebooks, "no notebooks found; update this test"
    offenders: list[str] = []
    for path in notebooks:
        nb = json.loads(path.read_text())
        for cell in nb["cells"]:
            for line in "".join(cell["source"]).splitlines():
                if "pip install" not in line or "strands-robots[" not in line:
                    continue
                upgrades = " -U " in line or " --upgrade " in line
                pinned = ">=" in line.split("strands-robots[", 1)[1]
                if not (upgrades or pinned):
                    offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, (
        "these install lines leave a pre-existing older strands-robots in place "
        "(pip reports 'Requirement already satisfied'); add -U or a >= floor so a "
        "stale environment is upgraded rather than silently kept:\n  " + "\n  ".join(offenders)
    )


def test_notebooks_readme_states_min_version_for_bucket_path() -> None:
    text = _NOTEBOOKS_README.read_text()
    assert _MIN_VERSION in text, (
        f"{_NOTEBOOKS_README} must note that notebook 5's bucket path needs {_MIN_VERSION!r} (issue #1500)."
    )


def _notebook_code() -> str:
    nb = json.loads(_NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")


def test_streaming_notebook_reads_the_bucket_path_it_wrote() -> None:
    """The bucket read must target the ``run_id`` folder the sync wrote to.

    ``sync_to_bucket`` uploads to ``hf://buckets/{bucket}/{run_id}``, while
    ``StreamingLeRobotDataset(repo_type="bucket")`` resolves ``meta/`` and
    ``data/`` directly under the repo id it is handed. A read of ``BUCKET``
    alone therefore looks for ``meta/info.json`` at the bucket root and raises
    ``FileNotFoundError``, because the dataset lives one level down. The two
    sides are only consistent when the read id carries the same ``run_id``.
    """
    code = _notebook_code()
    assert 'stream_dataset(BUCKET, repo_type="bucket"' not in code, (
        "notebook streams from BUCKET alone while sync_to_bucket writes to "
        "hf://buckets/{BUCKET}/{RUN_ID}; the read must include the run_id "
        "segment or it fails with FileNotFoundError on meta/info.json."
    )
    assert 'f"{BUCKET}/{RUN_ID}"' in code, (
        "notebook must build the bucket repo id from BUCKET and RUN_ID so the "
        "read targets the same path sync_to_bucket wrote to."
    )
    assert "run_id=RUN_ID" in code, (
        "notebook must pin the sync's run_id to RUN_ID so the write and read sides cannot drift apart."
    )


def test_training_notebooks_check_the_train_status_they_print() -> None:
    """A failed ``train()`` must surface its own message, not a ``None`` downstream.

    ``Trainer.train`` converts any failure into a ``TrainResult`` rather than
    raising, and only ``result.message`` carries the cause -- including lerobot's
    ``'accelerate' is required but not installed`` remedy, which no Strands
    Robots extra satisfies. A cell that prints ``status`` and moves on therefore
    hands the next cell a ``checkpoint_dir`` of ``None``, and
    ``create_policy(None)`` fails with ``TypeError: argument of type 'NoneType'
    is not iterable`` -- naming neither the missing package nor the fix.
    """
    for notebook in (_REPO_ROOT / "examples" / "notebooks" / "03_record_train_deploy.ipynb", _NOTEBOOK):
        nb = json.loads(notebook.read_text())
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        assert "trainer.train(spec)" in code, f"{notebook.name} no longer trains; update this test"
        assert 'result.status != "success"' in code, (
            f"{notebook.name} must check the status of trainer.train() before using "
            "its checkpoint_dir; train() reports failure in its result rather than "
            "raising, so an unchecked call fails one cell later on a None (issue #1500)."
        )
        assert "result.message" in code, (
            f"{notebook.name} must re-raise result.message -- it is the only field "
            "carrying the cause, including lerobot's own install remedy for the "
            "missing 'accelerate' (issue #1500)."
        )


def test_streaming_notebook_records_at_the_rate_it_rolls_out_at() -> None:
    """The rollout rate must match the recording fps, or nothing is recorded.

    The recorder writes one frame per control step with no decimation, so
    ``run_policy`` refuses a rollout whose ``control_frequency`` (50 Hz by
    default) disagrees with the recording's fps rather than writing frames at a
    distorted timestamp rate. Without an explicit ``control_frequency``, the
    30 fps recording here captured zero frames while the cell still printed
    "recorded ->", and every later cell ran against an empty dataset.
    """
    code = _notebook_code()
    assert "control_frequency=30" in code, (
        "the recording declares fps=30, so run_policy needs control_frequency=30 "
        "or the rate guard rejects the rollout and no frames are recorded."
    )
    assert 'raise RuntimeError(f"rollout failed' in code, (
        "the rollout's status must be checked; run_policy reports a rejected "
        "rollout in its result dict rather than raising, so an unchecked call "
        "prints success over an empty dataset."
    )
