"""GitHub-hosted robot models are fetched at a pinned commit and must match their registry content digest."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from strands_robots.description import assets

COMMIT = "1" * 40
MODEL = (
    '<mujoco><worldbody><body name="b"><joint name="j" range="-1 1"/><geom size="0.1"/></body>'
    '</worldbody><actuator><position name="a" joint="j"/></actuator></mujoco>'
)


def _tarball(files: dict[str, str], *, top: str = f"repo-{COMMIT}") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in {f"{top}/": "", **files}.items():
            info = tarfile.TarInfo(name.rstrip("/") if name.endswith("/") else name)
            if name.endswith("/"):
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
                continue
            data = text.encode()
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _digest_of(files: dict[str, str], tmp_path: Path) -> str:
    root = tmp_path / "digest-src"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return assets.tree_digest(root)


def _entry(digest: str, commit: str = COMMIT) -> dict:
    source = {"type": "github", "repo": "owner/repo", "subdir": "models", "commit": commit, "sha256": digest}
    return {"asset": {"dir": "bot", "model_xml": "bot.xml", "source": source}}


class _Served:
    def __init__(self, blob: bytes):
        self.blob, self.urls = blob, []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        return io.BytesIO(self.blob)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_ROBOTS_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _serve(monkeypatch, blob: bytes) -> _Served:
    served = _Served(blob)
    monkeypatch.setattr(assets.urllib.request, "urlopen", served)
    return served


def test_every_github_model_in_the_registry_is_pinned():
    data = json.loads((Path(assets.__file__).parents[1] / "registry" / "robots.json").read_text())
    sources = [v["asset"]["source"] for v in data["robots"].values() if (v.get("asset") or {}).get("source")]
    assert sources
    for s in sources:
        assert assets._COMMIT_RE.match(s.get("commit", "")), s
        assert assets._DIGEST_RE.match(s.get("sha256", "")), s


def test_an_unpinned_source_is_refused_without_a_download(cache, monkeypatch):
    served = _serve(monkeypatch, b"")
    entry = _entry("0" * 64)
    del entry["asset"]["source"]["commit"]
    with pytest.raises(ValueError, match="pinned 'commit'"):
        assets.resolve(entry)
    assert served.urls == []


def test_the_pinned_commit_is_downloaded_and_cached_once(cache, monkeypatch, tmp_path):
    served = _serve(monkeypatch, _tarball({f"repo-{COMMIT}/models/bot.xml": MODEL}))
    entry = _entry(_digest_of({"bot.xml": MODEL}, tmp_path))
    assert assets.resolve(entry) == assets.resolve(entry) == cache / "bot" / "bot.xml"
    assert served.urls == [f"https://codeload.github.com/owner/repo/tar.gz/{COMMIT}"]


def test_content_that_does_not_match_the_digest_is_not_cached(cache, monkeypatch, tmp_path):
    widened = MODEL.replace('range="-1 1"', 'range="-10 10"')
    _serve(monkeypatch, _tarball({f"repo-{COMMIT}/models/bot.xml": widened}))
    with pytest.raises(assets.AssetMismatch, match="nothing was cached"):
        assets.resolve(_entry(_digest_of({"bot.xml": MODEL}, tmp_path)))
    assert not (cache / "bot").exists()


def test_a_tarball_of_another_commit_is_refused(cache, monkeypatch, tmp_path):
    other = "2" * 40
    _serve(monkeypatch, _tarball({f"repo-{other}/models/bot.xml": MODEL}, top=f"repo-{other}"))
    with pytest.raises(assets.AssetMismatch, match="not commit"):
        assets.resolve(_entry(_digest_of({"bot.xml": MODEL}, tmp_path)))


@pytest.mark.parametrize("name", [f"repo-{COMMIT}/models/../../../outside.txt", f"repo-{COMMIT}/models//abs.txt"])
def test_member_names_that_leave_the_cache_refuse_the_archive(cache, monkeypatch, tmp_path, name):
    _serve(monkeypatch, _tarball({f"repo-{COMMIT}/models/bot.xml": MODEL, name: "payload"}))
    with pytest.raises(assets.AssetMismatch, match="refusing the archive"):
        assets.resolve(_entry(_digest_of({"bot.xml": MODEL}, tmp_path)))
    assert not (cache / "bot").exists()
    assert not any(p.name in ("outside.txt", "abs.txt") for p in tmp_path.rglob("*"))
    assert not Path("/abs.txt").exists()


def test_a_cache_from_another_pin_is_fetched_again(cache, monkeypatch, tmp_path):
    stale = cache / "bot"
    stale.mkdir(parents=True)
    (stale / "bot.xml").write_text(MODEL.replace('range="-1 1"', 'range="-10 10"'))
    served = _serve(monkeypatch, _tarball({f"repo-{COMMIT}/models/bot.xml": MODEL}))
    path = assets.resolve(_entry(_digest_of({"bot.xml": MODEL}, tmp_path)))
    assert len(served.urls) == 1
    assert path.read_text() == MODEL
