"""Pin the minimum ``strands-robots`` version in the bucket-streaming docs.

The streaming-data-loop notebook (and the storage-buckets blog post drafted
from it) demonstrates the bucket path: ``stop_recording(bucket=...)``,
``sync_to_bucket``, and ``stream_dataset(..., repo_type="bucket")``. None of
those exist on the 0.4.1 PyPI release -- on 0.4.1
``StreamingDatasetReader.open()`` has no ``repo_type`` parameter, so the
notebook's headline snippet raises ``TypeError``. The docs must therefore
state the minimum ``strands-robots`` version (>= 0.4.2, the first tag that
includes the bucket path) instead of implying that any PyPI install works.

These assertions forbid the version-less install guidance from creeping back
into the notebook or the notebooks index (issue #1500).
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NOTEBOOK = _REPO_ROOT / "examples" / "notebooks" / "05_streaming_data_loop.ipynb"
_NOTEBOOKS_README = _REPO_ROOT / "examples" / "notebooks" / "README.md"

_MIN_VERSION = "strands-robots >= 0.4.2"


def _notebook_markdown() -> str:
    nb = json.loads(_NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "markdown")


def test_streaming_notebook_states_min_strands_robots_version() -> None:
    md = _notebook_markdown()
    assert _MIN_VERSION in md, (
        f"{_NOTEBOOK.name} must state the minimum strands-robots version "
        f"({_MIN_VERSION!r}) - the bucket APIs it demonstrates are not in the "
        "0.4.1 PyPI release (issue #1500)."
    )


def test_streaming_notebook_does_not_offer_bare_pypi_install_as_requirements() -> None:
    md = _notebook_markdown()
    assert '**Requirements:** `pip install "strands-robots' not in md, (
        f"{_NOTEBOOK.name} presents a version-less PyPI install as its "
        "requirements line; on PyPI 0.4.1 the notebook's bucket snippet "
        "crashes with TypeError (issue #1500). State the minimum version."
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
