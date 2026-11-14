"""X-Confirm is the operator's yes only on a signed-in request; every state-changing route acts only after it."""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

import test_drivers  # noqa: E402
from test_operator_gate import goal_writes  # noqa: E402

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

YES = {"X-Confirm": "yes"}


def _real(asked: list[object]) -> tuple[Robot, object]:
    d, serial = test_drivers.make_driver()  # a fake serial bus: nothing leaves this process
    return Robot("so101", driver=d, id="real_arm", twin=False, operator=lambda a: asked.append(a) or False), serial


def test_with_sign_in_off_x_confirm_does_not_stand_in_for_the_operator() -> None:
    sim, asked = Simulation(), []
    real, serial = _real(asked)
    server = Dashboard(sim, robots=[real], auth=False, unauthenticated_hardware=True).start()
    try:
        before = len(goal_writes(serial))
        r = httpx.post(f"{server.url}/api/robots/real_arm/act", json={"targets": {"shoulder_pan": 0.2}}, headers=YES)
        assert r.status_code == 403, r.text
        assert asked and len(goal_writes(serial)) == before  # the robot's own callback was asked, and said no
        assert asked[0].source == "dashboard"  # the operator reads where the command came from
        r = httpx.post(f"{server.url}/api/robots/real_arm/run", json={"policy": "hold"}, headers=YES)
        assert r.status_code == 403
    finally:
        server.stop()
        real.close()
        sim.close()


def test_an_unconfirmed_run_on_a_real_robot_builds_no_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    import strands_robots.policy.factory as factory

    made: list[object] = []
    monkeypatch.setattr(factory, "make", lambda spec, *a, **k: made.append(spec))
    sim, asked = Simulation(), []
    real, _ = _real(asked)
    server = Dashboard(sim, robots=[real]).start()
    try:
        r = httpx.post(f"{server.url}/api/robots/real_arm/run", json={"policy": "zenoh://10.9.9.9:7447/dep"})
        assert r.status_code == 403 and made == []  # refused before any endpoint is dialled
    finally:
        server.stop()
        real.close()
        sim.close()


@pytest.fixture
def world(tmp_path):
    sim = Simulation()
    Robot("so101", sim=sim, id="arm")
    inside = tmp_path / "objects"
    inside.mkdir()
    box = '<mujoco><worldbody><body name="b"><geom type="box" size=".02 .02 .02"/></body></worldbody></mujoco>'
    (inside / "box.xml").write_text(box)
    (tmp_path / "outside.xml").write_text(box)
    (inside / "link.xml").symlink_to(tmp_path / "outside.xml")
    server = Dashboard(sim, object_dir=inside).start()
    yield server.url, inside, tmp_path
    server.stop()
    sim.close()


def _obj(name: str, path: object) -> dict[str, object]:
    return {"rpc": "add_object", "add_object": {"name": name, "path": str(path)}}


def test_the_world_reads_freely_and_changes_only_with_the_yes(world) -> None:
    url, inside, _ = world
    assert httpx.post(f"{url}/api/world", json={"rpc": "describe"}).status_code == 200
    shape = {"rpc": "add_object", "add_object": {"name": "c", "shape": "box"}}
    assert httpx.post(f"{url}/api/world", json=shape).status_code == 403
    assert httpx.post(f"{url}/api/world", json=shape, headers=YES).status_code == 200
    assert httpx.post(f"{url}/api/world", json=_obj("in", inside / "box.xml"), headers=YES).status_code == 200


def test_a_file_object_must_sit_in_the_objects_folder(world) -> None:
    url, inside, tmp = world
    for name, path in (
        ("out", tmp / "outside.xml"),
        ("dots", inside / ".." / "outside.xml"),
        ("ln", inside / "link.xml"),
    ):
        r = httpx.post(f"{url}/api/world", json=_obj(name, path), headers=YES)
        assert r.status_code == 403 and "outside the objects folder" in r.text, (name, r.text)


def test_without_an_objects_folder_the_page_loads_no_files(tmp_path) -> None:
    sim = Simulation()
    server = Dashboard(sim).start()
    try:
        r = httpx.post(f"{server.url}/api/world", json=_obj("x", tmp_path / "a.xml"), headers=YES)
        assert r.status_code == 403 and "objects folder" in r.text
    finally:
        server.stop()
        sim.close()


def test_a_world_run_names_no_policy_at_a_network_address(world) -> None:
    url, _, _ = world
    call = {"rpc": "run", "run": {"policies": {"arm": "zenoh://10.9.9.9:7447/dep"}, "steps": 1}}
    r = httpx.post(f"{url}/api/world", json=call, headers=YES)
    assert r.status_code == 403 and "network address" in r.text, r.text


def test_connecting_a_policy_needs_the_yes() -> None:
    sim = Simulation()
    server = Dashboard(sim).start()
    try:
        r = httpx.post(f"{server.url}/api/connect/policy", json={"endpoint": "lerobot/act_so101", "kind": "lerobot"})
        assert r.status_code == 403 and "X-Confirm" in r.text
    finally:
        server.stop()
        sim.close()
