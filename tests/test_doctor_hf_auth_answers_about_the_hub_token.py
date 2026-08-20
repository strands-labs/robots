"""``doctor``'s HuggingFace check answers about the file the Hub will read.

``check_hf_auth`` reports whether a HuggingFace token is available. The token it
is reporting on belongs to ``huggingface_hub``, which resolves its location from
``HF_TOKEN_PATH``, else ``<HF_HOME>/token``, where ``HF_HOME`` is ``$HF_HOME``,
else ``<XDG_CACHE_HOME>/huggingface``, else ``~/.cache/huggingface``. Reading a
hardcoded ``~/.cache/huggingface/token`` describes a different file on any host
that relocated its cache, and it is wrong in both directions - a relocated token
reads as "not logged in", and a token left at the default path reads as
"logged in" for an environment where the Hub resolves nothing.

Both resolution branches are covered: the constant the Hub computed for itself,
and the transcription of the same rule a base install (no ``huggingface_hub``)
falls back to. The parity test is what makes that duplication safe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The spelling the shipped package already prescribes for an interactive login
# (``dataset_recorder`` and the README both use it). ``huggingface-cli`` is not
# published as a console script at the declared ``huggingface_hub>=1.5`` floor
# at all, and the 1.x releases that do ship it exit with "deprecated and no
# longer works", so prescribing it is a dead end across the whole declared range.
_LOGIN_COMMAND = "hf auth login"
_DEAD_LOGIN_COMMAND = "huggingface-cli"


def _plain(result: str) -> str:
    """Strip the ANSI colour wrappers ``doctor`` adds to a verdict."""
    for code in ("\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"):
        result = result.replace(code, "")
    return result


def _verdict(result: str) -> str:
    """``"PASS"`` / ``"WARN"`` / ``"FAIL"`` for a ``doctor`` check result."""
    head = _plain(result).strip().splitlines()[0].strip()
    return head.split()[0]


def _clear_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every variable that participates in token resolution."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME", "HF_TOKEN_PATH", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)


def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, with_token: bool) -> Path:
    """Point ``HOME`` at a scratch tree, optionally holding a default token.

    ``HOME`` is what makes these assertions independent of the machine running
    them: without it the default path is the developer's own, so the two
    directions of the defect are not separable.
    """
    home = tmp_path / ("home_with_token" if with_token else "home_without_token")
    default = home / ".cache" / "huggingface"
    default.mkdir(parents=True, exist_ok=True)
    if with_token:
        (default / "token").write_text("hf_defaultPathToken\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    return home


def _block_the_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``huggingface_hub.constants`` unimportable (a base install)."""
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", None)


def _token_file(tmp_path: Path, name: str) -> Path:
    """A file holding a token, standing in for a completed login."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("hf_relocatedToken\n", encoding="utf-8")
    return path


class TestTheVerdictFollowsTheHubsOwnTokenPath:
    """With a Hub installed, its own resolved path is the one consulted."""

    def test_a_token_the_hub_resolves_is_reported_as_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An authenticated host whose token is not at the default path passes.

        This is the false-negative direction: the Hub would hand every call a
        token, so reporting "No HuggingFace token found" describes a machine
        other than the one being diagnosed.
        """
        pytest.importorskip("huggingface_hub")
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        _isolated_home(monkeypatch, tmp_path, with_token=False)
        relocated = _token_file(tmp_path, "relocated/token")
        monkeypatch.setattr("huggingface_hub.constants.HF_TOKEN_PATH", str(relocated))

        result = check_hf_auth()
        assert _verdict(result) == "PASS", (
            f"the Hub reads {relocated} and would resolve a token there, but doctor reported {_plain(result).strip()!r}"
        )

    def test_no_token_where_the_hub_looks_is_reported_as_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A stale token at the default path must not read as authenticated.

        This is the false-positive direction, and the worse one: every gated
        download and every dataset push will fail with a 401 on this host, and
        the check whose job is to surface that ahead of runtime says it is fine.
        """
        pytest.importorskip("huggingface_hub")
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        home = _isolated_home(monkeypatch, tmp_path, with_token=True)
        assert (home / ".cache" / "huggingface" / "token").exists(), "premise: the default path holds a token"
        empty = tmp_path / "relocated_empty" / "token"
        empty.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("huggingface_hub.constants.HF_TOKEN_PATH", str(empty))

        result = check_hf_auth()
        assert _verdict(result) == "WARN", (
            f"the Hub reads {empty}, which does not exist, so it resolves no token; doctor reported "
            f"{_plain(result).strip()!r}"
        )

    def test_the_verdict_names_the_file_it_read(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Both branches name the path consulted, so the verdict is actionable."""
        pytest.importorskip("huggingface_hub")
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        _isolated_home(monkeypatch, tmp_path, with_token=False)
        relocated = _token_file(tmp_path, "named/token")
        monkeypatch.setattr("huggingface_hub.constants.HF_TOKEN_PATH", str(relocated))
        found = _plain(check_hf_auth())
        assert str(relocated) in found, "the PASS verdict must name the file it read"

        relocated.unlink()
        missing = _plain(check_hf_auth())
        assert str(relocated) in missing, "the WARN verdict must name the file it looked in"


class TestTheFallbackTranscribesTheHubsRule:
    """With no Hub installed there is nothing to ask, so the rule is restated."""

    @pytest.mark.parametrize("relocation", ["HF_TOKEN_PATH", "HF_HOME", "XDG_CACHE_HOME"])
    def test_each_relocation_variable_is_honored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, relocation: str
    ) -> None:
        """Every variable the Hub reads relocates the file doctor consults."""
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        _isolated_home(monkeypatch, tmp_path, with_token=False)
        _block_the_hub(monkeypatch)
        if relocation == "HF_TOKEN_PATH":
            monkeypatch.setenv("HF_TOKEN_PATH", str(_token_file(tmp_path, "explicit/hf.token")))
        elif relocation == "HF_HOME":
            token = _token_file(tmp_path, "hf_home/token")
            monkeypatch.setenv("HF_HOME", str(token.parent))
        else:
            token = _token_file(tmp_path, "xdg/huggingface/token")
            monkeypatch.setenv("XDG_CACHE_HOME", str(token.parent.parent))

        result = check_hf_auth()
        assert _verdict(result) == "PASS", f"{relocation} relocated the token, but doctor reported {_plain(result)!r}"

    def test_a_relocated_empty_home_is_not_answered_from_the_default_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A stale default token does not answer for a relocated ``HF_HOME``."""
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        _isolated_home(monkeypatch, tmp_path, with_token=True)
        _block_the_hub(monkeypatch)
        empty = tmp_path / "hf_home_empty"
        empty.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HF_HOME", str(empty))

        result = check_hf_auth()
        assert _verdict(result) == "WARN", (
            f"a token at the default path answered for HF_HOME={empty}: {_plain(result).strip()!r}"
        )

    def test_the_fallback_resolves_what_the_hub_resolves(self, tmp_path: Path) -> None:
        """The transcription and the Hub's own constant agree.

        The fallback exists only because ``huggingface_hub`` is not a base
        dependency, so it is a second statement of a rule the Hub owns. This is
        what keeps the two from drifting: the variables are read in a
        subprocess, because the Hub computes its constants at import time and no
        in-process assignment can move them.
        """
        pytest.importorskip("huggingface_hub")
        probe = tmp_path / "probe.py"
        probe.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "from huggingface_hub.constants import HF_TOKEN_PATH\n"
            "sys.modules['huggingface_hub.constants'] = None\n"
            "from strands_robots.doctor import _hf_token_path\n"
            # Both branches wrap the resolved string in ``Path``, so the primary
            # answer is ``Path(HF_TOKEN_PATH)`` - that is what the fallback has
            # to reproduce, including ``Path("")`` normalising to ``Path(".")``.
            "print(json.dumps({'hub': str(Path(HF_TOKEN_PATH)), 'fallback': str(_hf_token_path())}))\n",
            encoding="utf-8",
        )
        # Each case is one environment the Hub resolves differently, including
        # the empty-string spellings: the Hub reads every one of these by
        # presence, so "" must not be treated as unset.
        cases: list[dict[str, str]] = [
            {},
            {"HF_TOKEN_PATH": str(tmp_path / "explicit.token")},
            {"HF_HOME": str(tmp_path / "home")},
            {"XDG_CACHE_HOME": str(tmp_path / "xdg")},
            {"HF_HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "xdg")},
            {"HF_TOKEN_PATH": str(tmp_path / "wins.token"), "HF_HOME": str(tmp_path / "home")},
            {"HF_HOME": ""},
            {"XDG_CACHE_HOME": ""},
            {"HF_TOKEN_PATH": ""},
        ]
        for case in cases:
            env = {k: v for k, v in os.environ.items() if k not in ("HF_HOME", "HF_TOKEN_PATH", "XDG_CACHE_HOME")}
            env.update(case)
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path.cwd()), env.get("PYTHONPATH", "")]))
            out = subprocess.run(  # noqa: S603 - fixed interpreter, no shell
                [sys.executable, str(probe)], capture_output=True, text=True, env=env, check=True
            )
            resolved = json.loads(out.stdout)
            assert resolved["fallback"] == resolved["hub"], (
                f"with {case or '{}'} the Hub reads {resolved['hub']} but the fallback resolved {resolved['fallback']}"
            )


class TestTheDocumentedSpellingsAreUnchanged:
    """The spellings ``doctor`` already answered for keep their verdicts."""

    @pytest.mark.parametrize("variable", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
    def test_an_environment_token_still_passes(self, monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
        """An env token outranks any file, exactly as the Hub reads it."""
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        monkeypatch.setenv(variable, "hf_environmentToken")
        result = check_hf_auth()
        assert _verdict(result) == "PASS", _plain(result)

    def test_a_token_at_the_default_path_still_passes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The unrelocated host - the common case - is unaffected."""
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        _isolated_home(monkeypatch, tmp_path, with_token=True)
        _block_the_hub(monkeypatch)
        result = check_hf_auth()
        assert _verdict(result) == "PASS", _plain(result)

    def test_no_token_anywhere_still_warns(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An unauthenticated host is still reported as one."""
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        _isolated_home(monkeypatch, tmp_path, with_token=False)
        _block_the_hub(monkeypatch)
        result = check_hf_auth()
        assert _verdict(result) == "WARN", _plain(result)

    def test_the_remedy_names_a_command_the_declared_floor_ships(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The login command offered must be one that runs.

        ``huggingface-cli`` is not a console script at the declared
        ``huggingface_hub>=1.5`` floor, and the later 1.x releases that do
        install it refuse to run, so a caller who follows that advice gets
        either "command not found" or "deprecated and no longer works".
        """
        from strands_robots.doctor import check_hf_auth

        _clear_hf_env(monkeypatch)
        _isolated_home(monkeypatch, tmp_path, with_token=False)
        _block_the_hub(monkeypatch)
        note = _plain(check_hf_auth())
        assert _LOGIN_COMMAND in note, f"the remedy must offer {_LOGIN_COMMAND!r}, got {note!r}"
        assert _DEAD_LOGIN_COMMAND not in note, (
            f"{_DEAD_LOGIN_COMMAND!r} is not installed at the declared floor and refuses to run on the "
            f"releases that ship it, so it cannot be prescribed: {note!r}"
        )
