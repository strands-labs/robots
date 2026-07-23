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
