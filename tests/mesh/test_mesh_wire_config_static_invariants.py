"""Static (AST-based) invariants for the mesh wire-config modules.

Origin: PR-224 R1 review feedback.

Each test in this module pins a static-shape invariant of the mesh wire
modules (``_acl_config.py``, ``_zenoh_config.py``, ``session.py``)
against the kinds of regressions a future refactor might silently
re-introduce. The pins are AST-based where the property is structural
(exception-clause narrowness, dead-code presence, calling convention)
because a behavioural test would either require ``zenoh`` to be
importable in CI or would mutate global session singletons.

Reviewer threads pinned (from PR
https://github.com/strands-labs/robots/pull/224):

  * ``_acl_config.py:380`` -- ``except (OSError, FileNotFoundError):``
    is a tuple-collapse (``FileNotFoundError`` ⊂ ``OSError``).
    Pinned by ``test_acl_config_no_redundant_subclass_in_except``.

  * ``_acl_config.py:471`` -- same shape with
    ``(OSError, ValueError, FileNotFoundError)``.
    Same pin covers both occurrences (AST walk).

  * ``session.py:425-427`` and ``session.py:523-525`` -- the early
    ``try/except ValueError: _auth_mode = "mtls"`` swallow that the
    reviewer flagged as dead (``_build_config()`` raises again 5 lines
    later). Pinned by
    ``test_session_no_auth_mode_value_error_swallow``.

  * ``_zenoh_config.py:547-555`` -- the second ``key_path.is_symlink()``
    re-check that the loop above already executed for all three TLS
    paths. Pinned by ``test_zenoh_config_resolve_tls_paths_one_symlink_check``.

  * ``session.py:308`` -- ``is_default_acl_in_use()`` was called
    without the resolved ``namespace`` argument. Pinned by
    ``test_session_default_acl_check_passes_namespace``.

The CodeQL alert on ``tests/mesh/test_acl_example_refs.py`` (unused
``_TESTS`` global) self-resolves once the symbol is removed; not
pinned here because the alert IS the pin.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MESH = _REPO_ROOT / "strands_robots" / "mesh"


def _walk_except_handlers(src_path: Path) -> list[ast.ExceptHandler]:
    """Return every ``ExceptHandler`` AST node in *src_path*."""
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]


def test_acl_config_no_redundant_subclass_in_except() -> None:
    """``_acl_config.py`` must not list ``FileNotFoundError`` alongside
    ``OSError`` in a tuple of caught exceptions.

    ``FileNotFoundError`` is a subclass of ``OSError`` (since 3.3) so
    the redundant element silently expands the catch surface from a
    reader's perspective and misleads static analysers.

    Per AGENTS.md > Review Learnings (PR #86): ``Exception Clauses Must
    Be Narrow``.
    """
    src = _MESH / "_acl_config.py"
    redundant_pairs = {
        # subclass: superclass that already covers it
        "FileNotFoundError": "OSError",
        "PermissionError": "OSError",
        "IsADirectoryError": "OSError",
        "NotADirectoryError": "OSError",
    }
    offenders: list[str] = []
    for handler in _walk_except_handlers(src):
        if not isinstance(handler.type, ast.Tuple):
            continue
        names = {n.id for n in handler.type.elts if isinstance(n, ast.Name)}
        for sub, sup in redundant_pairs.items():
            if sub in names and sup in names:
                offenders.append(
                    f"{src.name}:{handler.lineno}: except clause "
                    f"{sorted(names)!r} contains both {sup!r} and its "
                    f"subclass {sub!r}; drop {sub!r}."
                )
    assert not offenders, "\n".join(offenders)


def test_session_no_auth_mode_value_error_swallow() -> None:
    """``session.py`` must not silently swallow ``ValueError`` from
    ``resolve_auth_mode()`` and fall back to ``_auth_mode = "mtls"``.

    Pre-fix shape (now removed)::

        try:
            _auth_mode = resolve_auth_mode()
        except ValueError:
            _auth_mode = "mtls"

    The swallow was dead -- ``_build_config()`` calls
    ``resolve_auth_mode()`` again 5 lines later and lets the
    ``ValueError`` propagate -- so the silent fallback to ``"mtls"``
    was setting up a wire-scheme that the immediately-following config
    build would then reject. The fix lets ``resolve_auth_mode()`` raise
    once at the first call site; ``Mesh.start`` then crashes with a
    clear stacktrace on a typo'd ``STRANDS_MESH_AUTH_MODE`` instead of
    silently logging three confusing fallback warnings.

    The pin is shape-based: any ``ExceptHandler`` whose body assigns
    to a name containing ``auth_mode`` and whose type catches
    ``ValueError`` is the pattern we removed.
    """
    src = _MESH / "session.py"
    offenders: list[str] = []
    for handler in _walk_except_handlers(src):
        type_node = handler.type
        if isinstance(type_node, ast.Name) and type_node.id == "ValueError":
            catches_value_error = True
        elif isinstance(type_node, ast.Tuple) and any(
            isinstance(n, ast.Name) and n.id == "ValueError" for n in type_node.elts
        ):
            catches_value_error = True
        else:
            catches_value_error = False
        if not catches_value_error:
            continue
        # Look at the handler body: if it assigns a literal to a name
        # whose lower-cased form contains "auth_mode", it is the
        # swallow we removed.
        for stmt in handler.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and "auth_mode" in tgt.id.lower():
                    offenders.append(
                        f"{src.name}:{handler.lineno}: ``except ValueError`` "
                        f"with ``{tgt.id} = ...`` body re-introduces the "
                        f"silent auth-mode fallback removed in PR-224 R1."
                    )
    assert not offenders, "\n".join(offenders)


def test_zenoh_config_resolve_tls_paths_one_symlink_check() -> None:
    """``_resolve_tls_paths`` must call ``is_symlink()`` exactly once
    (in the unified CA/cert/key reject loop), not a second time on the
    private-key path alone.

    The dead duplicate check fired at the same wall-clock instant as
    the loop above and made the security claim harder to audit; a
    future hardener might believe a TOCTOU window was mitigated when
    only the trivial half was. The lstat + mode check below remains
    -- the pin is on the ``is_symlink`` call count specifically.
    """
    src = _MESH / "_zenoh_config.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    target_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_tls_paths":
            target_fn = node
            break
    assert target_fn is not None, (
        f"{src.name}: _resolve_tls_paths function not found -- pin is stale, "
        "rename or refactor likely; update this test."
    )
    is_symlink_call_count = 0
    for inner in ast.walk(target_fn):
        if not isinstance(inner, ast.Call):
            continue
        # Match ``something.is_symlink()`` -- attribute call with no args.
        if (
            isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "is_symlink"
            and not inner.args
            and not inner.keywords
        ):
            is_symlink_call_count += 1
    assert is_symlink_call_count == 1, (
        f"{src.name}:_resolve_tls_paths has {is_symlink_call_count} "
        f"is_symlink() call(s); expected exactly 1 (the unified CA/cert/key "
        f"reject loop). A second check is dead code per PR-224 R1 review."
    )


def test_session_default_acl_check_passes_namespace() -> None:
    """``_build_config()`` must call ``is_default_acl_in_use(namespace)``
    with the resolved ``namespace`` argument, not bare.

    The function currently ignores its arg via shape-only inspection,
    so calling ``is_default_acl_in_use()`` without the arg works in
    the common case. But operators who set
    ``STRANDS_MESH_NAMESPACE=fleet-a`` will, after a future shape-check
    that does honour ``namespace``, see the gate and the wire-effective
    ACL reasoning over different namespaces. The pin locks the calling
    convention so the consistency cannot silently regress.
    """
    src = _MESH / "session.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    target_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_config":
            target_fn = node
            break
    assert target_fn is not None, (
        f"{src.name}: _build_config function not found -- pin is stale, rename or refactor likely; update this test."
    )
    bare_calls: list[int] = []
    for inner in ast.walk(target_fn):
        if not isinstance(inner, ast.Call):
            continue
        # Match ``...is_default_acl_in_use(...)`` regardless of
        # module-attribute prefix.
        called_name = None
        if isinstance(inner.func, ast.Attribute):
            called_name = inner.func.attr
        elif isinstance(inner.func, ast.Name):
            called_name = inner.func.id
        if called_name != "is_default_acl_in_use":
            continue
        # Must have at least one positional arg (the namespace).
        if not inner.args:
            bare_calls.append(inner.lineno)
    assert not bare_calls, (
        f"{src.name}:{bare_calls!r}: is_default_acl_in_use() called without "
        f"the namespace argument; pass the resolved namespace through so "
        f"the gate check and the ACL block reason over the same value "
        f"(PR-224 R1)."
    )
