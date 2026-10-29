"""Every install line the docs print names extras that exist, and `[all]` is every extra a CPU machine can install."""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
# Their own lines: the GPU worlds need CUDA, dev is the contributor's toolchain, all is the union itself.
NOT_IN_ALL = {"isaac", "newton", "dev", "all"}
PAGES = ["README.md", "notebooks/README.md", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md"))]


def test_all_is_every_extra_a_cpu_machine_can_install() -> None:
    (line,) = EXTRAS["all"]
    named = set(re.fullmatch(r"strands-robots\[([a-z,]+)\]", line).group(1).split(","))  # type: ignore[union-attr]
    assert named == set(EXTRAS) - NOT_IN_ALL


@pytest.mark.parametrize("page", PAGES)
def test_an_install_line_names_only_declared_extras(page: str) -> None:
    text = (ROOT / page).read_text()
    for group in re.findall(r'pip install -e "\.\[([^\]]+)\]"', text):
        assert set(group.split(",")) <= set(EXTRAS), f"{page}: .[{group}]"


@pytest.mark.parametrize("page", ["README.md", "docs/quickstart.md", "notebooks/README.md"])
def test_the_try_it_line_is_all_plus_lerobot(page: str) -> None:
    text = (ROOT / page).read_text()
    assert 'uv pip install -e ".[all]"' in text
    assert "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@" in text
