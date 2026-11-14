"""A real robot on a zenoh mesh any machine may join takes no motion rpcs, unless the network is vouched for."""

from __future__ import annotations

import pytest

from strands_robots import Robot


@pytest.mark.parametrize("vouched", ["", "url", "env"])
def test_a_real_robot_on_a_mesh_anyone_may_join_is_served_observe_only(tenant, port, monkeypatch, vouched):
    pytest.importorskip("zenoh")
    from strands_robots.mesh import Mesh
    from strands_robots.robot.safety import RequestRefused

    if vouched == "env":
        monkeypatch.setenv("STRANDS_ZENOH_TRUSTED", "1")
    flag = "&trusted=1" if vouched == "url" else ""
    dev = Mesh(f"zenoh://?listen=tcp/127.0.0.1:{port}{flag}", tenant=tenant)
    op = Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant)
    arm = Robot("so101", driver="memory", operator=lambda action: True)
    try:
        dev.mount(arm, device_id="arm")
        assert op.discover("arm", min_count=1, timeout=3.0)
        assert op.invoke("arm", "observe").joints["shoulder_pan"] == 0.0  # reading is unchanged
        if vouched:
            op.invoke("arm", "act", action={"shoulder_pan": 0.1})
            assert arm.observe().joints["shoulder_pan"] == pytest.approx(0.1)
        else:
            for rpc, args in (("act", {"action": {"shoulder_pan": 0.1}}), ("hold", {})):
                with pytest.raises(RequestRefused, match="observe-only"):
                    op.invoke("arm", rpc, **args)
            assert arm.observe().joints["shoulder_pan"] == 0.0
    finally:
        op.close()
        dev.close()
        arm.close()


def test_a_simulated_robot_on_a_mesh_anyone_may_join_still_moves(tenant, port):
    pytest.importorskip("zenoh")
    from strands_robots.mesh import Mesh

    dev = Mesh(f"zenoh://?listen=tcp/127.0.0.1:{port}", tenant=tenant)
    op = Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant)
    arm = Robot("so101")
    try:
        dev.mount(arm, device_id="arm")
        assert op.discover("arm", min_count=1, timeout=3.0)
        op.invoke("arm", "act", action={"shoulder_pan": 0.1})
    finally:
        op.close()
        dev.close()
        arm.close()
