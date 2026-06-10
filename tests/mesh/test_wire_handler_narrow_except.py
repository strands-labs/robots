"""Regression: ``_on_cmd`` and ``_on_response`` use the same narrow exception
tuple as ``_on_safety_estop`` / ``_on_safety_resume``.

Pre-fix: both wire handlers caught bare :class:`Exception`, masking
programmer-bug ``RuntimeError`` / ``TypeError`` / ``MemoryError`` from a
future Zenoh API change. Per AGENTS.md > Review Learnings (#86) > "Exception
Clauses Must Be Narrow", the four wire-input handlers must use the same
``(AttributeError, UnicodeDecodeError, json.JSONDecodeError)`` tuple.

Pre-fix verification:
    git stash && pytest tests/mesh/test_wire_handler_narrow_except.py -v
    -> 3 tests fail: bare-except still present in source.
"""

from __future__ import annotations

import inspect

from strands_robots.mesh import core as mesh_core


def _source_of(method) -> str:
    return inspect.getsource(method)


def test_on_cmd_uses_narrow_exception_tuple() -> None:
    src = _source_of(mesh_core.Mesh._on_cmd)
    assert "except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):" in src, (
        "_on_cmd must use the same narrow tuple as _on_safety_estop "
        "(AGENTS.md > Review Learnings #86: 'Exception Clauses Must Be Narrow')"
    )
    assert "except Exception:" not in src.split("def _on_cmd", 1)[1].split("def ", 1)[0], (
        "_on_cmd must not contain bare 'except Exception:' on the wire-input parse path"
    )


def test_on_response_uses_narrow_exception_tuple() -> None:
    src = _source_of(mesh_core.Mesh._on_response)
    assert "except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):" in src, (
        "_on_response must use the same narrow tuple as _on_safety_resume "
        "(AGENTS.md > Review Learnings #86: 'Exception Clauses Must Be Narrow')"
    )
    assert "except Exception:" not in src.split("def _on_response", 1)[1].split("def ", 1)[0], (
        "_on_response must not contain bare 'except Exception:' on the wire-input parse path"
    )


def test_all_four_wire_handlers_use_same_tuple() -> None:
    """The wire-input parse pattern is identical across the four handlers.

    Symmetric-reasoning audit: if one handler narrows, all must narrow.
    This test pins the symmetry so a future refactor cannot regress one
    without the others.
    """
    handlers = [
        mesh_core.Mesh._on_cmd,
        mesh_core.Mesh._on_response,
        mesh_core.Mesh._on_safety_estop,
        mesh_core.Mesh._on_safety_resume,
    ]
    expected = "except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):"
    for handler in handlers:
        src = _source_of(handler)
        assert expected in src, (
            f"{handler.__qualname__} must use the canonical narrow tuple; "
            f"see _on_safety_estop for the reference pattern"
        )


# === bridge_transport dedup decode path uses the same narrow tuple ===


def test_bridge_transport_dedup_decode_uses_narrow_tuple() -> None:
    """The bridge dedup decode path was a fifth wire-shaped exception
    site. the prior fix narrowed it to the same ``(AttributeError,
    UnicodeDecodeError, json.JSONDecodeError)`` tuple used by the four
    safety/cmd/response handlers.
    """
    from strands_robots.mesh.transport import bridge_transport as bt

    src = inspect.getsource(bt)
    # Look for the specific pattern AROUND the dedup decode block
    # (the comment in the source mentions the exact tuple).
    assert "except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):" in src, (
        "bridge_transport.py dedup decode path must use the same narrow "
        "tuple as the four wire handlers in core.py "
        "(AGENTS.md > Review Learnings #86)"
    )


# === _on_presence is the fifth wire-input handler ===


def test_on_presence_uses_narrow_exception_tuple() -> None:
    """``_on_presence`` is the fifth wire-input
    handler in core.py. the prior implementation it was the only one with a bare
    ``except Exception`` on the parse path -- AGENTS.md > Review
    Learnings (#86) calls that out as forbidden, and the same threat
    model applies to presence (a Zenoh sample.payload parse).

    Same tuple as the other four wire handlers."""
    src = _source_of(mesh_core.Mesh._on_presence)
    assert "except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):" in src, (
        "_on_presence must use the same narrow tuple as the four other wire handlers"
    )
