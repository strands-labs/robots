"""A cached ACL is served only while the file still holds the bytes it came from.

The ACL file gates wire authorisation, and it is read through one cache keyed on
the file's identity tuple ``(path, dev, ino, size, mtime_ns)``. That tuple was
also what licensed serving an entry, and it cannot carry that weight: the kernel
stamps ``st_mtime_ns`` from a coarse clock, so two writes inside one tick are
stamped alike, and a rewrite that keeps the byte count leaves ``st_size`` alone
too. An in-place ACL edit of the same length therefore computes the identity the
*pre-edit* contents were cached under.

Rotating an authorised ``cert_common_names`` entry - revoking one peer and
authorising another - is exactly that shape, and the stale hit is permanent
rather than a window: no later stat can distinguish the two contents, so the
revoked peer stays authorised for the life of the process.

The sibling module docstrings both promised the refresh that closes this
(``snapshot_acl`` calls a rewrite's reload "the by-design refresh window") and
``tests/mesh/test_acl_cache_docstring_matches_the_cache.py`` pins that *sentence*
against the code. Nothing pinned the refresh as behaviour, which is what these
tests do. The shapes the identity tuple always did catch are pinned alongside,
so a future change cannot trade one for the other.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config


def _acl(cn: str) -> str:
    """An ACL authorising exactly one peer, by cert CN.

    Every CN of the same length yields a file of the same byte count, which is
    what makes an authorisation rotation invisible to ``st_size``.
    """
    return json.dumps(
        {
            "enabled": True,
            "default_permission": "deny",
            "rules": [
                {
                    "id": "op",
                    "permission": "allow",
                    "flows": ["ingress"],
                    "messages": ["put"],
                    "key_exprs": ["**/cmd"],
                }
            ],
            "subjects": [{"id": "op", "cert_common_names": [cn]}],
            "policies": [{"rules": ["op"], "subjects": ["op"]}],
        }
    )


@pytest.fixture
def acl_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An ACL authorising ``operator-1``, wired up and already cached."""
    path = tmp_path / "acl.json5"
    path.write_text(_acl("operator-1"), encoding="utf-8")
    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
    _acl_config._clear_acl_cache_for_test()
    _acl_config._clear_thread_snapshot()
    yield path
    _acl_config._clear_acl_cache_for_test()
    _acl_config._clear_thread_snapshot()


def _authorised() -> list[str]:
    """The cert CNs the mesh would put on the wire right now."""
    _, resolved = _acl_config.snapshot_acl("strands")
    return list(resolved["subjects"][0]["cert_common_names"])


def _rotate_within_one_tick(path: Path) -> None:
    """Rotate the authorised CN in place, inside one timestamp tick.

    The rewrite keeps the byte count and the inode; ``os.utime`` restores the
    pre-edit ``st_mtime_ns`` so the collision is pinned rather than raced. Two
    same-size writes milliseconds apart collide on their own - the tick is
    coarse - but a test must not depend on the host's clock granularity.
    """
    before = os.stat(path)
    path.write_text(_acl("operator-2"), encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


class TestARewriteTheIdentityTupleCannotSee:
    """The rewrite the stat tuple misses, and what the mesh does with it."""

    def test_the_identity_tuple_is_unchanged_by_the_rotation(self, acl_file: Path) -> None:
        """The premise: this edit is invisible to the cache key.

        Without this, a green sibling test below could mean the key happened to
        change rather than that the bytes were checked.
        """
        before = _acl_config._file_identity(acl_file)
        _rotate_within_one_tick(acl_file)
        assert _acl_config._file_identity(acl_file) == before, (
            "premise: a same-size in-place rewrite inside one timestamp tick must be invisible to the identity tuple"
        )

    def test_rotating_the_authorised_cn_revokes_the_old_peer(self, acl_file: Path) -> None:
        """The authorisation on the wire is the one written to the file."""
        assert _authorised() == ["operator-1"], "premise: the pre-edit ACL is cached"
        _rotate_within_one_tick(acl_file)
        assert _authorised() == ["operator-2"], (
            "the ACL file authorises operator-2, so operator-1 has been revoked; "
            "serving the cached pre-edit ACL leaves the revoked peer authorised, "
            "and no later stat can tell the two contents apart"
        )


class TestTheShapesTheIdentityTupleAlreadyCaught:
    """Verifying the bytes must not cost the refreshes that already worked."""

    @staticmethod
    def _size_changes(path: Path) -> None:
        path.write_text(_acl("operator-two-long"), encoding="utf-8")

    @staticmethod
    def _later_tick(path: Path) -> None:
        time.sleep(0.05)
        path.write_text(_acl("operator-2"), encoding="utf-8")

    @staticmethod
    def _atomic_replace(path: Path) -> None:
        scratch = path.with_suffix(".tmp")
        scratch.write_text(_acl("operator-2"), encoding="utf-8")
        os.replace(scratch, path)

    @pytest.mark.parametrize(
        ("rewrite", "expected"),
        [
            pytest.param(_size_changes, "operator-two-long", id="size-changes"),
            pytest.param(_later_tick, "operator-2", id="later-tick"),
            pytest.param(_atomic_replace, "operator-2", id="new-inode"),
        ],
    )
    def test_a_rewrite_the_key_can_see_still_refreshes(
        self, acl_file: Path, rewrite: Callable[[Path], None], expected: str
    ) -> None:
        assert _authorised() == ["operator-1"], "premise: the pre-edit ACL is cached"
        rewrite(acl_file)
        assert _authorised() == [expected]


class TestTheCacheStillSavesTheParse:
    """A hit is verified, not abandoned: the read is what licenses skipping work.

    Passes on both sides of the fix - it is the bound the fix has to stay
    inside, not a change in what the cache does.
    """

    def test_an_unchanged_file_is_served_without_parsing_it_again(
        self, acl_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _authorised() == ["operator-1"], "premise: the first read populates the cache"

        def refuse(raw: str, path: Path) -> object:
            raise AssertionError(f"a cache hit re-parsed {path}")

        monkeypatch.setattr(_acl_config, "_parse_json5", refuse)
        assert _authorised() == ["operator-1"]
