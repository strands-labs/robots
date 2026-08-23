"""A LIBERO driver's HF-token preflight answers about the token the Hub will use.

The three ``examples/libero`` drivers refuse ``--policy groot`` up front when no
HuggingFace token is available, rather than letting the gated checkpoint
download fail later inside ``gr00t_inference``. The refusal *is* the preflight,
so it carries two obligations: resolve the token the Hub resolves, and offer a
remedy that works.

``huggingface_hub`` resolves its cached login from ``HF_TOKEN_PATH``, else
``<HF_HOME>/token``, else ``<XDG_CACHE_HOME>/huggingface/token``, else
``~/.cache/huggingface/token``. Reading the last of those directly names a file
the Hub will not open on any host that relocated its cache, so a box that *is*
logged in gets refused - the same defect ``doctor`` carried until its token path
moved onto the Hub's own resolution, whose guard is scoped to ``doctor`` and
never reached these drivers.

The remedy has its own failure mode. ``huggingface-cli`` is not published as a
console script at the declared ``huggingface_hub>=1.5`` floor at all, and the
later 1.x releases that do install it exit "deprecated and no longer works", so
prescribing it is a dead end across the whole declared range. The live entry
point is ``hf auth login`` - what ``doctor``, ``dataset_recorder``, the README,
``docs/troubleshooting.md`` and ``huggingface_hub.get_token``'s own docstring
all name.

These drivers are not part of the installed package, so nothing else in CI
imports them: a regression confined to ``examples/`` stays invisible without a
test that loads them by path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_LIBERO = _REPO_ROOT / "examples" / "libero"

# The drivers carrying the preflight, and the backend each needs importable.
_DRIVERS = (
    ("run.py", None),
    ("run_isaac_agent.py", None),
    ("run_mujoco_agent.py", "mujoco"),
)

# Assembled from parts so the contiguous prescription never appears in this
# file: the rule below scans this tree, and a grader that names the phrase it
# forbids would report itself rather than the offender.
_DEAD_CLI = "huggingface-cli"
_LOGIN_VERB = "login"
_DEAD_PRESCRIPTION = f"{_DEAD_CLI} {_LOGIN_VERB}"
_LIVE_LOGIN_COMMAND = "hf auth login"

# Trees whose prose a reader is expected to act on.
_SCANNED = ("README.md", "docs", "examples", "strands_robots", "tests", "tests_integ", "changelog.d", "scripts")
_SUFFIXES = (".py", ".md", ".ipynb")


def _load_driver(filename: str):
    """Import an example driver by path under a test-unique module name."""
    path = _EXAMPLES_LIBERO / filename
    assert path.is_file(), f"expected example driver at {path}"
    module_name = f"_hf_preflight_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _preflight(filename: str, backend: str | None):
    """The driver's token preflight, skipping when its backend is absent."""
    if backend is not None:
        pytest.importorskip(backend)
    pytest.importorskip("huggingface_hub")
    module = _load_driver(filename)
    resolver = getattr(module, "_resolve_hf_token", None)
    assert callable(resolver), f"{filename} has no _resolve_hf_token preflight"
    return resolver


@pytest.fixture
def hub_sees_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate the Hub's token resolution onto an empty path under ``tmp_path``.

    ``huggingface_hub.constants.HF_TOKEN_PATH`` is computed at import time, so
    setting the relocation variables in-process cannot move it. Its reader
    (``_get_token_from_file``) takes the module attribute on every call, so
    steering that attribute is what redirects the Hub here.

    Returns:
        The (absent) path the Hub now resolves, for a test to populate.
    """
    constants = pytest.importorskip("huggingface_hub.constants")
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    relocated = tmp_path / "relocated-cache" / "huggingface" / "token"
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(relocated))
    return relocated


class TestThePreflightResolvesWhatTheHubResolves:
    """The preflight admits every token source the Hub itself would use."""

    @pytest.mark.parametrize(("filename", "backend"), _DRIVERS)
    def test_a_cached_login_the_hub_resolves_is_accepted(
        self, filename: str, backend: str | None, hub_sees_nothing: Path
    ) -> None:
        """A host that relocated its HF cache and logged in must not be refused.

        This is the regression: reading ``~/.cache/huggingface/token`` directly
        names a file that does not exist here, while the Hub resolves a token.
        """
        resolver = _preflight(filename, backend)
        hub_sees_nothing.parent.mkdir(parents=True, exist_ok=True)
        hub_sees_nothing.write_text("hf_cached_login_token\n")

        from huggingface_hub import get_token

        assert get_token() == "hf_cached_login_token", "premise: the Hub must resolve this token"
        assert resolver() == "hf_cached_login_token"

    @pytest.mark.parametrize(("filename", "backend"), _DRIVERS)
    @pytest.mark.parametrize("variable", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
    def test_an_environment_token_is_accepted(
        self,
        filename: str,
        backend: str | None,
        variable: str,
        monkeypatch: pytest.MonkeyPatch,
        hub_sees_nothing: Path,
    ) -> None:
        """Both documented environment spellings are honoured with no token file.

        ``HF_TOKEN`` is the spelling the drivers' own prose calls preferred for
        CI, so a preflight that reads only a cached file refuses exactly the
        environment it recommends.
        """
        resolver = _preflight(filename, backend)
        monkeypatch.setenv(variable, "hf_env_token")
        assert not hub_sees_nothing.exists(), "premise: no cached login on this host"
        assert resolver() == "hf_env_token"

    @pytest.mark.parametrize(("filename", "backend"), _DRIVERS)
    def test_no_token_anywhere_is_still_refused(
        self, filename: str, backend: str | None, hub_sees_nothing: Path
    ) -> None:
        """The preflight still exists: a host with no token is refused early."""
        resolver = _preflight(filename, backend)
        assert not hub_sees_nothing.exists(), "premise: no cached login on this host"
        with pytest.raises(RuntimeError):
            resolver()


class TestTheEnvironmentPathReachesTheDownload:
    """``run_mujoco_agent.py``'s preflight sits inline in its lifecycle block.

    The other two drivers expose the preflight as ``_resolve_hf_token``, so the
    resolution tests above reach them directly. This one drives the lifecycle
    function itself with the download stubbed out, which is the only way to see
    whether an ``HF_TOKEN``-only host - the environment the drivers' own prose
    calls preferred for CI - gets past the gate at all.
    """

    def test_an_environment_token_alone_reaches_the_download(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub_sees_nothing: Path
    ) -> None:
        pytest.importorskip("mujoco")
        pytest.importorskip("huggingface_hub")
        module = _load_driver("run_mujoco_agent.py")

        class _ReachedTheDownload(Exception):
            """Not a RuntimeError, so the token refusal stays distinguishable."""

        def _stub_lifecycle(**kwargs: object) -> dict[str, object]:
            raise _ReachedTheDownload(str(kwargs.get("hf_token")))

        # `_configure_gr00t_image` writes these; let monkeypatch own them.
        monkeypatch.setenv("STRANDS_GR00T_IMAGE", "gr00t:placeholder")
        monkeypatch.setenv("STRANDS_GR00T_IMAGE_ALLOW", "gr00t:*")
        monkeypatch.setattr(module, "gr00t_inference", _stub_lifecycle)
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")
        assert not hub_sees_nothing.exists(), "premise: no cached login on this host"

        args = argparse.Namespace(
            policy="groot",
            auto_server=True,
            image="gr00t:test",
            checkpoint_dir=str(tmp_path / "ckpt"),
            container="libero-test",
            port=8000,
        )
        with pytest.raises(_ReachedTheDownload) as reached:
            module._bring_up_gr00t_server(args, "libero_spatial")
        assert str(reached.value) == "hf_env_token", "the environment token must be the one forwarded"


class TestTheRefusalIsActionable:
    """The refusal names a command that runs and the file it looked in."""

    @pytest.mark.parametrize(("filename", "backend"), _DRIVERS)
    def test_the_refusal_names_the_live_login_command(
        self, filename: str, backend: str | None, hub_sees_nothing: Path
    ) -> None:
        resolver = _preflight(filename, backend)
        with pytest.raises(RuntimeError) as refused:
            resolver()
        message = str(refused.value)
        assert _LIVE_LOGIN_COMMAND in message, f"{filename} refusal offers no runnable login: {message}"
        assert _DEAD_PRESCRIPTION not in message, f"{filename} refusal prescribes a dead command: {message}"

    @pytest.mark.parametrize(("filename", "backend"), _DRIVERS)
    def test_the_refusal_names_the_path_the_hub_reads(
        self, filename: str, backend: str | None, hub_sees_nothing: Path
    ) -> None:
        """A relocated cache is only actionable if the message says where it looked."""
        resolver = _preflight(filename, backend)
        with pytest.raises(RuntimeError) as refused:
            resolver()
        assert str(hub_sees_nothing) in str(refused.value), (
            f"{filename} refusal does not name {hub_sees_nothing}, so a relocated "
            f"cache cannot be acted on: {refused.value}"
        )


def _prose_files() -> list[Path]:
    """Every tracked prose or source file a reader is expected to act on."""
    files: list[Path] = []
    for base in _SCANNED:
        target = _REPO_ROOT / base
        if target.is_file():
            files.append(target)
            continue
        for path in sorted(target.rglob("*")):
            if path.is_file() and path.suffix in _SUFFIXES and "__pycache__" not in path.parts:
                files.append(path)
    return files


def _prescriptions(path: Path) -> list[int]:
    """1-based line numbers where ``path`` prescribes the dead login command.

    Notebooks are graded on their markdown and source cells rather than the raw
    JSON, so a single claim is reported once at the file rather than repeatedly
    across the escaped document.
    """
    if path.suffix == ".ipynb":
        try:
            cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        text = "".join("".join(cell.get("source", [])) for cell in cells)
        return [1] if _DEAD_PRESCRIPTION in text else []
    return [n for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if _DEAD_PRESCRIPTION in line]


class TestNoSurfacePrescribesADeadLoginCommand:
    """No prose in the tree tells a reader to run a command that cannot run."""

    def test_the_dead_login_command_is_prescribed_nowhere(self) -> None:
        offenders = {
            str(path.relative_to(_REPO_ROOT)): lines for path in _prose_files() if (lines := _prescriptions(path))
        }
        assert not offenders, (
            f"prose prescribes `{_DEAD_PRESCRIPTION}`, which is not published as a console "
            f"script at the declared huggingface_hub>=1.5 floor and exits "
            f'"deprecated and no longer works" on the later 1.x releases that install it. '
            f"Use `{_LIVE_LOGIN_COMMAND}`: {offenders}"
        )

    def test_naming_the_dead_entry_point_without_prescribing_it_is_allowed(self) -> None:
        """Explaining *why* it is dead is not prescribing it.

        ``docs/troubleshooting.md`` and this file both name ``huggingface-cli``
        to say it does not work. The rule keys on the two-token prescription, so
        the explanation is not an offender - otherwise the only way to document
        the hazard would be to stop mentioning it.
        """
        naming = [p for p in _prose_files() if _DEAD_CLI in p.read_text(encoding="utf-8")]
        assert naming, f"premise: something in the tree still explains why {_DEAD_CLI} is dead"
        assert not {str(p.relative_to(_REPO_ROOT)) for p in naming if _prescriptions(p)}

    def test_the_scan_reaches_the_prose_it_grades(self) -> None:
        """A rule that reaches nothing would pass while the tree drifts."""
        files = _prose_files()
        assert len(files) > 500, f"prose scan collapsed to {len(files)} files"
        reached = {p.parts[len(_REPO_ROOT.parts)] for p in files}
        assert {"docs", "examples", "strands_robots", "tests"} <= reached, f"scan missed a tree: {sorted(reached)}"
        live = [p for p in files if _LIVE_LOGIN_COMMAND in p.read_text(encoding="utf-8")]
        assert len(live) >= 4, f"only {len(live)} files name the live login command"
