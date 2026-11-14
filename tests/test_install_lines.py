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
    groups = re.findall(r'pip install -e "\.\[([^\]]+)\]"', text) + re.findall(r"strands-robots\[([a-z,-]+)\]", text)
    for group in groups:
        assert set(group.split(",")) <= set(EXTRAS), f"{page}: [{group}]"


MAIN = "@ git+https://github.com/strands-labs/robots@main"
DEV_SECTION = "## Install the development version"


def _outside_dev_section(text: str) -> str:
    """The page without its one development-version section (up to the next ``## `` heading)."""
    if DEV_SECTION not in text:
        return text
    head, rest = text.split(DEV_SECTION, 1)
    tail = rest.split("\n## ", 1)
    return head + ("\n## " + tail[1] if len(tail) == 2 else "")


@pytest.mark.parametrize("page", ["README.md", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md"))])
def test_a_user_install_line_installs_the_pypi_release(page: str) -> None:
    """A reader's install line says strands-robots and takes the PyPI release, not a git ref."""
    text = _outside_dev_section((ROOT / page).read_text())
    lines = re.findall(r"pip install [^\n]*strands-robots\[[^\n]*", text)
    for line in lines:
        assert "git+" not in line, f"{page}: {line}"
    assert "git+https://github.com/strands-labs/robots" not in text, page


def test_install_page_has_one_development_version_section() -> None:
    """Contributors still find the git line for main, in exactly one place."""
    text = (ROOT / "docs/install.md").read_text()
    assert text.count(DEV_SECTION) == 1
    assert f'uv pip install "strands-robots[all] {MAIN}"' in text
    assert "older API" not in text


LEROBOT_RELEASE = "lerobot[remote,feetech,dataset]>=0.7.0,<0.8.0"


def test_lerobot_is_a_release_dependency_not_a_git_ref() -> None:
    """LeRobot's 0.7 release carries remote_inference, so it is a normal wheel dependency: no git ref lives anywhere."""
    from strands_robots._extras import hint

    assert LEROBOT_RELEASE in EXTRAS["lerobot"][0]
    assert "pip install 'strands-robots[lerobot]'" in hint("lerobot")
    assert "huggingface/lerobot@" not in (ROOT / "pyproject.toml").read_text()
    for page in ["README.md", "docs/quickstart.md", "docs/install.md", "notebooks/README.md", "AGENTS.md"]:
        assert "huggingface/lerobot@" not in (ROOT / page).read_text(), page


@pytest.mark.parametrize("page", ["README.md", "docs/quickstart.md", "notebooks/README.md"])
def test_the_try_it_line_is_all(page: str) -> None:
    """``[all]`` now carries LeRobot, so the try-it install is one line with no separate LeRobot line."""
    text = (ROOT / page).read_text()
    clone = page == "notebooks/README.md"  # the notebooks run from a clone of the repository
    assert ('uv pip install -e ".[all]"' if clone else 'uv pip install "strands-robots[all]"') in text
