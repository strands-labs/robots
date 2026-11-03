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
    groups = re.findall(r'pip install -e "\.\[([^\]]+)\]"', text) + re.findall(r"strands-robots\[([a-z,-]+)\] @", text)
    for group in groups:
        assert set(group.split(",")) <= set(EXTRAS), f"{page}: [{group}]"


MAIN = "@ git+https://github.com/strands-labs/robots@main"


@pytest.mark.parametrize("page", ["README.md", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md"))])
def test_a_user_install_line_names_the_package_and_installs_main(page: str) -> None:
    """A reader's install line says strands-robots and takes it from main."""
    text = (ROOT / page).read_text()
    for line in re.findall(r"pip install [^\n]*strands-robots\[[^\n]*", text):
        assert MAIN in line, f"{page}: {line}"


LEROBOT_MAIN = "lerobot[remote,feetech,dataset] @ git+https://github.com/huggingface/lerobot@main"


def test_lerobot_comes_from_its_main_branch_everywhere() -> None:
    """One LeRobot line: its main branch, in the install hint, the hatch env and every page (no pinned commit)."""
    from strands_robots._extras import LEROBOT

    assert LEROBOT_MAIN in LEROBOT
    assert LEROBOT_MAIN in (ROOT / "pyproject.toml").read_text()
    for page in ["README.md", "docs/quickstart.md", "docs/install.md", "notebooks/README.md"]:
        for line in (ROOT / page).read_text().splitlines():
            if "huggingface/lerobot@" in line:
                assert LEROBOT_MAIN in line, f"{page}: {line.strip()}"


@pytest.mark.parametrize("page", ["README.md", "docs/quickstart.md", "notebooks/README.md"])
def test_the_try_it_line_is_all_plus_lerobot(page: str) -> None:
    text = (ROOT / page).read_text()
    clone = page == "notebooks/README.md"  # the notebooks run from a clone of the repository
    assert ('uv pip install -e ".[all]"' if clone else f'uv pip install "strands-robots[all] {MAIN}"') in text
    assert LEROBOT_MAIN in text
