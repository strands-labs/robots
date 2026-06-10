"""Regression test: AGENTS.md Key Conventions #10 -- no dead code in public API.

PR-6 is the consumer of LockoutError (raised by Mesh._dispatch when the
emergency-stop lockout is engaged), so the class is now both defined in
security.py and exported in __all__. This test pins:

- LockoutError IS defined and exported (the PR-6 consumer landed)
- Every export in __all__ is importable (catches typos / refactor breaks)
"""

from strands_robots.mesh import security


def test_lockout_error_is_exported():
    """PR-6 consumer (Mesh._dispatch) is now in scope, so LockoutError lands."""
    assert "LockoutError" in security.__all__, (
        "LockoutError must be exported now that PR-6 (Mesh._dispatch) is the consumer."
    )


def test_lockout_error_class_is_defined():
    """LockoutError class is defined alongside its consumer."""
    assert hasattr(security, "LockoutError"), "LockoutError class must be defined alongside Mesh._dispatch consumer."
    assert issubclass(security.LockoutError, security.SecurityError), "LockoutError must inherit from SecurityError."


def test_all_exports_are_importable():
    """Every name in __all__ must be importable from the module."""
    for name in security.__all__:
        assert hasattr(security, name), f"{name} in __all__ but not importable"
