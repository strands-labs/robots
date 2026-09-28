"""An env VALUE must not be able to smuggle a second KEY past the allow-list.

``env_entry_error`` refused C0 control characters (``\\n``, ``\\r``), but the env file is
parsed with ``str.splitlines()``, which also breaks on ``U+0085``, ``U+2028`` and
``U+2029``. A page-writable key such as ``HF_TOKEN`` carrying ``"x\\u2028LD_PRELOAD=..."``
therefore wrote one physical line that read back as two entries, and the injected key
(explicitly denied, or gate-bearing) reached ``os.environ`` on the next start. The gate now
uses the parser's own predicate: a value must survive ``splitlines()`` whole.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.dashboard import config_api

SEPARATORS = ["\u0085", "\u2028", "\u2029", "\n", "\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e"]


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr(config_api, "ENV_FILE", path)
    return path


@pytest.mark.parametrize("sep", SEPARATORS, ids=[f"U+{ord(s[0]):04X}" for s in SEPARATORS])
def test_env_entry_error_refuses_every_splitlines_separator(sep: str) -> None:
    value = f"x{sep}LD_PRELOAD=/tmp/evil.so"
    assert value.splitlines() != [value], "precondition: the parser would split this value"
    problem = config_api.env_entry_error("HF_TOKEN", value)
    assert problem is not None
    assert "control characters" in problem


def test_plain_value_still_accepted() -> None:
    assert config_api.env_entry_error("HF_TOKEN", "hf_abc123 with spaces and unicode é") is None


@pytest.mark.parametrize("sep", ["\u0085", "\u2028", "\u2029"], ids=["U+0085", "U+2028", "U+2029"])
def test_upsert_refuses_unicode_line_separator_values(env_file: Path, sep: str) -> None:
    with pytest.raises(ValueError, match="control characters"):
        config_api.upsert_env_file({"HF_TOKEN": f"x{sep}LD_PRELOAD=/tmp/evil.so"})
    assert not env_file.exists(), "a refused write must not touch the file"


def test_gate_bearing_key_cannot_be_minted_through_a_value(env_file: Path) -> None:
    """The second fence: a consent grant cannot appear without a consent card."""
    with pytest.raises(ValueError, match="control characters"):
        config_api.upsert_env_file({"HF_TOKEN": "x\u2028STRANDS_DASH_AGENT_PHYSICAL_MOTION=1"})
    assert config_api.read_env_file() == {}


def test_round_trip_yields_exactly_one_key(env_file: Path) -> None:
    """A legitimate write reads back as the single key that was written."""
    config_api.upsert_env_file({"HF_TOKEN": "hf_legit"})
    assert config_api.read_env_file() == {"HF_TOKEN": "hf_legit"}
    # and the file on disk is one physical line, not two
    assert env_file.read_text(encoding="utf-8").splitlines() == ["HF_TOKEN=hf_legit"]
