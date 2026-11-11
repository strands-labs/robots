"""A refusal names what to change: the setting, the argument or the place an option belongs."""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots import Robot


def test_unlisted_hub_repo_names_the_setting_to_allow_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from strands_robots.policy.hub import refuse_unlisted

    monkeypatch.delenv("STRANDS_POLICY_REPOS", raising=False)
    with pytest.raises(PermissionError, match=r"STRANDS_POLICY_REPOS=lerobot/smolvla_base \(or lerobot/\*"):
        refuse_unlisted("lerobot/smolvla_base")


def test_world_change_without_the_yes_names_the_header_not_a_sign_in() -> None:
    from strands_robots.dashboard.confirm import check_world_call

    with pytest.raises(PermissionError) as err:
        check_world_call({"rpc": "add_object", "add_object": {"name": "box"}}, confirmed=False, objects=None)
    assert "X-Confirm: yes on a signed-in request" in str(err.value)
    assert "sign in and" not in str(err.value)


def test_real_robot_refusal_names_the_mesh_as_a_source() -> None:
    from strands_robots.robot.safety import RequestRefused

    arm = Robot("so101", driver="memory")
    with pytest.raises(RequestRefused, match="an agent, the dashboard or the mesh: pass confirm="):
        arm.confirm_act({"gripper": 0.5})


@pytest.mark.parametrize(
    ("option", "hint"),
    [("watchdog_ms", r"lab\.mount\(robot, watchdog_ms=\.\.\.\)"), ("calibration", r"driver='feetech'")],
)
def test_option_a_driver_does_not_take_says_where_it_belongs(option: str, hint: str) -> None:
    with pytest.raises(TypeError, match=hint):
        Robot("so101", driver="memory", **{option: 250 if option == "watchdog_ms" else "my_arm.json"})


def test_a_driver_option_typo_gets_no_hint() -> None:
    with pytest.raises(TypeError) as err:
        Robot("so101", driver="memory", camras={})
    assert str(err.value).endswith("it takes ['cameras', 'port']")


def test_cert_says_when_it_replaces_an_identity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("cryptography")
    from strands_robots.mesh.__main__ import main

    ca, out = str(tmp_path / "ca"), str(tmp_path / "lab-pc")
    assert main(["ca", "--ca", ca]) in (0, None)
    assert main(["cert", "lab-pc", "--serves", "so101", "--ca", ca, "--out", out]) in (0, None)
    assert "replaced" not in capsys.readouterr().out
    assert main(["cert", "lab-pc", "--serves", "so101,arm", "--ca", ca, "--out", out]) in (0, None)
    assert "replaced the identity that was in" in capsys.readouterr().out


def test_dashboard_help_lists_its_environment(capsys: pytest.CaptureFixture[str]) -> None:
    from strands_robots.dashboard.__main__ import parse

    with pytest.raises(SystemExit):
        parse(["--help"])
    text = capsys.readouterr().out
    for name in (
        "STRANDS_ROBOTS_DASHBOARD_OBJECTS",
        "STRANDS_ROBOTS_RECORDINGS",
        "STRANDS_ROBOTS_DASHBOARD_AUTH",
        "STRANDS_ROBOTS_DASHBOARD_SESSION_TTL",
        "STRANDS_POLICY_REPOS",
    ):
        assert name in text
