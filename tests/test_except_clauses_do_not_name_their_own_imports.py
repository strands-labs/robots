"""An except clause must not name a symbol its own try body imports.

Python evaluates an ``except`` clause's exception expression at the moment the
try body raises, not when the ``try`` is entered. So a handler that names a
class the body itself imports is only evaluatable when that import already
succeeded -- and the interesting case is precisely the one where it did not::

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError

        path = hf_hub_download(...)
    except ImportError:
        return None
    except (HfHubHTTPError, OSError, ValueError) as exc:
        return None

An ``ImportError`` lands in the first handler and never reaches the second, so
the shape looks correct. Any *other* import failure -- a partially installed
distribution raising ``OSError``, or a module whose definition-time work raises
``TypeError`` -- skips the ``ImportError`` handler, and evaluating the second
handler then raises ``UnboundLocalError``. That both replaces the original
exception with one naming an unrelated local, and skips the handler written for
it: ``OSError`` is in that tuple.

The two sites this rule was written for were the Hub reads in
:mod:`strands_robots.policies.lerobot_local.processor`, whose docstrings promise
they never raise into the ACT / diffusion load path. Neither promise held for a
non-``ImportError`` failure, and the load path
(``LerobotLocalPolicy._load_processor_bridge``) catches
``(FileNotFoundError, ValueError, ImportError)`` -- ``UnboundLocalError`` is a
``NameError``, so it aborted the policy load outright for a condition the
handler was written to absorb.

Nothing else refuses the shape. ``ruff`` has no rule for it. ``mypy`` reports it
under ``possibly-undefined``, which is off by default and not enabled here. Code
scanning has a rule for it, ``py/uninitialized-local-variable``, and it did not
report either of the two sites above -- every one of its open alerts is in test
code. So the shape reached shipped source with no gate naming it, and is refused
here, in the gate that blocks a merge.

Scope is imports only. A name bound by ordinary assignment in a try body and
named in its handler is a different question with different answers, and this
rule says nothing about it: an import is the case where the binding statement
and the failure are the same statement.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

import strands_robots

#: The shipped package. Reached through the imported module so a layout change
#: cannot silently narrow the scan to nothing.
_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent

#: Lower bound on the number of ``try`` statements whose body imports something.
#: Pinned so a scan rooted somewhere unexpected fails loudly instead of
#: reporting a clean sweep over nothing. Measured at 100+ when written.
_MINIMUM_GUARDED_IMPORTS = 50


def _names_bound_by_imports(body: list[ast.stmt]) -> dict[str, int]:
    """Names the import statements directly in *body* bind, to their line."""
    bound: dict[str, int] = {}
    for stmt in body:
        if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        for alias in stmt.names:
            if alias.name == "*":
                continue
            bound.setdefault(alias.asname or alias.name.split(".")[0], stmt.lineno)
    return bound


def _names_read_by_handler(handler: ast.ExceptHandler) -> set[str]:
    """Root names an except clause's exception expression reads."""
    if handler.type is None:
        return set()
    read: set[str] = set()
    for node in ast.walk(handler.type):
        if isinstance(node, ast.Name):
            read.add(node.id)
        elif isinstance(node, ast.Attribute):
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                read.add(root.id)
    return read


def unevaluatable_handlers(tree: ast.AST) -> list[tuple[int, str, int]]:
    """Handlers naming a symbol their own try body imports.

    Args:
        tree: A parsed module.

    Returns:
        ``(handler_line, name, import_line)`` per violation.
    """
    found: list[tuple[int, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        imported = _names_bound_by_imports(node.body)
        if not imported:
            continue
        for handler in node.handlers:
            for name in sorted(_names_read_by_handler(handler) & imported.keys()):
                found.append((handler.lineno, name, imported[name]))
    return found


def _guarded_import_count(tree: ast.AST) -> int:
    """Number of ``try`` statements in *tree* whose body imports something."""
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try) and bool(_names_bound_by_imports(node.body)))


class TestTheRuleHoldsAcrossThePackage:
    def test_no_handler_names_a_symbol_its_own_try_imports(self) -> None:
        offenders: list[str] = []
        scanned = 0
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            scanned += _guarded_import_count(tree)
            rel = path.relative_to(_PACKAGE_ROOT.parent)
            for handler_line, name, import_line in unevaluatable_handlers(tree):
                offenders.append(
                    f"{rel}:{handler_line} names {name!r}, imported by its own try body at line {import_line}"
                )
        assert scanned >= _MINIMUM_GUARDED_IMPORTS, (
            f"premise: only {scanned} guarded imports found under {_PACKAGE_ROOT}; a clean result would prove nothing"
        )
        assert not offenders, (
            "An except clause is evaluated when the try body raises, so it cannot name a "
            "symbol that body imports: any non-ImportError import failure raises "
            "UnboundLocalError instead of reaching the handler. Import the exception class "
            "in its own try, then wrap the call in a second one.\n  " + "\n  ".join(offenders)
        )


class TestTheRuleIsNotVacuous:
    def test_a_planted_violation_is_reported(self) -> None:
        planted = ast.parse(
            "def f():\n"
            "    try:\n"
            "        from json import JSONDecodeError\n"
            "        return 1\n"
            "    except JSONDecodeError:\n"
            "        return None\n"
        )
        assert unevaluatable_handlers(planted) == [(5, "JSONDecodeError", 3)]

    def test_the_evaluatable_form_is_accepted(self) -> None:
        # The remedy: the class is bound by an earlier, separate try, so the
        # handler naming it can always be evaluated.
        accepted = ast.parse(
            "def f():\n"
            "    try:\n"
            "        from json import JSONDecodeError\n"
            "    except ImportError:\n"
            "        return None\n"
            "    try:\n"
            "        return 1\n"
            "    except JSONDecodeError:\n"
            "        return None\n"
        )
        assert unevaluatable_handlers(accepted) == []

    def test_a_handler_naming_an_unrelated_class_is_accepted(self) -> None:
        # Scope check: only the body's own imports are in question.
        accepted = ast.parse(
            "def f():\n    try:\n        import json\n        return 1\n    except OSError:\n        return None\n"
        )
        assert unevaluatable_handlers(accepted) == []


class _UnusableHub:
    """A ``huggingface_hub`` whose import raises, recording that it was reached."""

    def __init__(self, exc: type[BaseException]) -> None:
        self._exc = exc
        self.attempts = 0

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            self.attempts += 1
            raise self._exc("simulated: huggingface_hub is installed but unimportable")
        return None


def _install_unusable_hub(monkeypatch: pytest.MonkeyPatch, exc: type[BaseException]) -> _UnusableHub:
    """Make every ``huggingface_hub`` import raise *exc* for this test."""
    for name in [m for m in list(sys.modules) if m == "huggingface_hub" or m.startswith("huggingface_hub.")]:
        monkeypatch.delitem(sys.modules, name)
    breaker = _UnusableHub(exc)
    monkeypatch.setattr(sys, "meta_path", [breaker, *sys.meta_path])
    return breaker


class TestTheHubReadsDegradeWhenTheHubIsUnusable:
    """The two sites the rule was written for, driven end to end.

    An absent hub already degraded to ``None``. A hub that is present but
    unimportable is the same condition to the caller and must degrade the same
    way, rather than aborting the load with an ``UnboundLocalError`` naming the
    error class the failing import was supposed to bind.
    """

    @pytest.mark.parametrize("exc", [OSError, TypeError, RuntimeError, ValueError])
    def test_checkpoint_state_dict_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: type[BaseException]
    ) -> None:
        pytest.importorskip("safetensors")
        pytest.importorskip("torch")
        from strands_robots.policies.lerobot_local.processor import _load_checkpoint_state_dict

        breaker = _install_unusable_hub(monkeypatch, exc)
        assert _load_checkpoint_state_dict(str(tmp_path)) is None
        assert breaker.attempts, "premise: the Hub import was never reached, so nothing was measured"

    @pytest.mark.parametrize("exc", [OSError, TypeError, RuntimeError, ValueError])
    def test_pipeline_step_keys_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: type[BaseException]
    ) -> None:
        from strands_robots.policies.lerobot_local.processor import _pipeline_step_keys

        breaker = _install_unusable_hub(monkeypatch, exc)
        assert _pipeline_step_keys(str(tmp_path), "policy_preprocessor.json") is None
        assert breaker.attempts, "premise: the Hub import was never reached, so nothing was measured"

    def test_an_absent_hub_still_degrades(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Control: the shape that always worked. An ImportError from the guarded
        # import must keep returning None, unchanged.
        pytest.importorskip("safetensors")
        pytest.importorskip("torch")
        from strands_robots.policies.lerobot_local.processor import _load_checkpoint_state_dict

        breaker = _install_unusable_hub(monkeypatch, ImportError)
        assert _load_checkpoint_state_dict(str(tmp_path)) is None
        assert breaker.attempts

    def test_a_failing_download_is_still_handled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Control: splitting the import out must not drop the handler for the
        # call. With the hub importable, a download failure still degrades.
        pytest.importorskip("safetensors")
        pytest.importorskip("torch")
        from strands_robots.policies.lerobot_local.processor import _load_checkpoint_state_dict

        def _offline(*_a: Any, **_k: Any) -> str:
            raise OSError("offline")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _offline)
        assert _load_checkpoint_state_dict(str(tmp_path)) is None

    def test_a_hub_http_error_is_still_handled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Control: the Hub-specific class stays in the call handler. Pinned
        # separately from OSError because it is the one the split had to carry
        # across, and it is the reason the handler names anything at all.
        pytest.importorskip("safetensors")
        pytest.importorskip("torch")
        httpx = pytest.importorskip("httpx")
        from huggingface_hub.errors import HfHubHTTPError

        from strands_robots.policies.lerobot_local.processor import _load_checkpoint_state_dict

        def _http_error(*_a: Any, **_k: Any) -> str:
            response = httpx.Response(429, request=httpx.Request("GET", "https://huggingface.co/x"))
            raise HfHubHTTPError("rate limited", response=response)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _http_error)
        assert _load_checkpoint_state_dict(str(tmp_path)) is None
