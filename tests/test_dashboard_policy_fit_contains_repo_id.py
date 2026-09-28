"""``/api/robots/{peer}/policy-fit`` applies the same ``repo_id`` containment as ``/api/checkpoints/features``.

``checkpoints.declared_features`` reads ``config.json`` / ``train_config.json`` /
``norm_stats.json`` from wherever a path points. The features route always gated a
path-shaped ``repo_id`` with ``contain_checkpoint_path``; the policy-fit route did
not, so an authenticated caller could read policy declarations out of any
policy-shaped directory on disk and use ``evidence`` as an existence probe. Both
routes now go through ``training.contain_checkpoint_ref`` and answer an outside
path with the one refusal body, which names the homes and never the path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import settings, training  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.setenv(training.OUTPUT_HOME_ENV, str(tmp_path / "training"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    app = create_app()
    with TestClient(app) as c:
        yield c
    app.state.safety.store.shutdown()


@pytest.fixture()
def probe(tmp_path: Path) -> Path:
    """A policy-shaped directory OUTSIDE every checkpoint home."""
    outside = tmp_path / "elsewhere" / "run"
    outside.mkdir(parents=True)
    (outside / "config.json").write_text(
        json.dumps({"type": "act", "input_features": {"observation.state": {"shape": [6]}}, "output_features": {}}),
        encoding="utf-8",
    )
    return outside


def _peer(client: TestClient) -> str:
    """A peer id the fleet knows, so the 404 for unknown peers cannot mask the 400."""
    bridge = cast("Any", client.app).state.bridge
    bridge.peers["probe-peer"] = {"presence": {"hw": False}, "state": {"joints": {}}, "cameras": {}}
    return "probe-peer"


class TestPolicyFitContainment:
    def test_an_outside_path_is_refused_with_the_features_route_body(self, client: TestClient, probe: Path) -> None:
        peer = _peer(client)
        fit = client.get(f"/api/robots/{peer}/policy-fit", params={"repo_id": str(probe)})
        features = client.get("/api/checkpoints/features", params={"repo_id": str(probe)})
        assert fit.status_code == 400, fit.text
        assert features.status_code == 400, features.text
        assert fit.json() == features.json(), "the containment rule is spelled once; both lanes must say the same thing"
        body = fit.json()["error"]
        assert body["field"] == "checkpoint path"
        assert body["homes"], "the refusal names the homes"
        assert str(probe) not in json.dumps(body), "the refusal never echoes the path it refused"
        assert "input_features" not in fit.text and "act" not in fit.text, "no declaration leaked from the probe dir"

    @pytest.mark.parametrize("raw", ["~someone/policies/run", "../../../etc", "./relative/run"])
    def test_expanduser_and_relative_shapes_are_gated_too(self, client: TestClient, raw: str) -> None:
        peer = _peer(client)
        r = client.get(f"/api/robots/{peer}/policy-fit", params={"repo_id": raw})
        assert r.status_code == 400, (raw, r.text)
        assert r.json()["error"]["field"] == "checkpoint path"

    def test_a_hub_repo_id_passes_through_unchanged(self) -> None:
        assert training.contain_checkpoint_ref("lerobot/act_so101") == "lerobot/act_so101"
        assert training.contain_checkpoint_ref("") == ""

    def test_a_path_inside_the_training_home_is_allowed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "training"
        run = home / "run-1"
        run.mkdir(parents=True)
        monkeypatch.setenv(training.OUTPUT_HOME_ENV, str(home))
        assert training.contain_checkpoint_ref(str(run)) == str(run.resolve())

    def test_an_outside_path_raises_path_outside_with_the_shared_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(training.OUTPUT_HOME_ENV, str(tmp_path / "training"))
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
        with pytest.raises(training.PathOutside) as info:
            training.contain_checkpoint_ref(str(tmp_path / "elsewhere"))
        body = info.value.refusal()
        assert set(body) == {"error", "field", "homes"}
        assert body["field"] == "checkpoint path"
        assert "elsewhere" not in body["error"]
