"""D08: AWS IoT Core, native. One mount option gives a device shadows, rules and a certificate of its own."""

from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d08_iot_core",
        "AWS IoT Core, natively",
        "mount(robot, iot=IotOptions()) makes a device an IoT thing; a shadow target still goes through the "
        "safety chain, and an e-stop never rides a shadow.",
        "Left, an operator with IAM writes control.desired and sends rpcs as Direct Messages. In the middle, the "
        "one green element: the IotOptions on a mount, which keeps three named shadows (manifest written once, "
        "state at most 1 Hz with the lockout, control) and applies rules as data from the manifest's events. "
        "Right, the robot: the control delta goes through confirm_act and the safety chain, then act; a refusal "
        "returns on a dashed wire into reported. Below, provisioning: thing equals device id, a fleet policy with "
        "policy variables, a certificate from a local CSR with CN equal to the device id, and a claim hook that "
        "refuses a forged CN. A dashed refused wire shows estop as a shadow field rejected.",
        h=640,
    )
    s.section(60, 120, "operator, IAM")
    s.box(60, 132, 280, 300, "operator", "boto3 iot-data, no certificate", size=14, subsize=12, id="op")
    s.chips(74, 206, ["update_thing_shadow"])
    s.chips(74, 242, ["Direct Message rpc"])
    s.chips(74, 278, ["operator_policy"])
    s.chips(74, 314, ['rpc="mqtt" for certs'])

    s.section(420, 120, "the mount option", accent=True)
    s.box(420, 132, 360, 300, "IotOptions(shadow_hz=1)", "on an iot:// transport only", accent=True, size=14,
          subsize=12, id="opts")
    s.box(440, 196, 320, 52, "manifest", "the record, written once, deleted at unmount", size=13, subsize=11)
    s.box(440, 258, 320, 52, "state", "at most 1 Hz, with the lockout", size=13, subsize=11)
    s.box(440, 320, 320, 52, "control", "desired target, refusal in reported", size=13, subsize=11, id="ctl")
    s.chips(440, 386, ["rules_for", "apply_rules"])

    s.section(860, 120, "the robot")
    s.box(860, 132, 280, 300, "so101", "same Robot, same safety chain", size=14, subsize=12, id="robot")
    s.chips(874, 206, ["confirm_act"])
    s.chips(874, 242, ["act"])
    s.chips(874, 278, ["lifecycle.Audit"])

    s.arrow([(340, 346), (440, 346)], label="desired", label_dx=-22, label_dy=-8, id="desired")
    s.arrow([(760, 346), (860, 346)], label="delta", label_dx=-16, label_dy=-8, id="delta")
    s.arrow([(860, 400), (760, 400)], dashed=True, label="refused", label_dx=-22, label_dy=14, id="refused")
    s.arrow([(200, 432), (200, 452), (1000, 452), (1000, 432)], label="rpc as a Direct Message, 404 when gone",
            label_dx=-120, label_dy=-6, id="rpc")

    s.section(60, 470, "provisioning")
    s.box(60, 482, 340, 70, "provision.register", "thing = device id, group = tenant", size=13, subsize=11)
    s.box(420, 482, 360, 70, "local CSR, CN = device id", "the key never leaves the robot", size=13, subsize=11)
    s.box(800, 482, 340, 70, "claim hook", "a forged CN is refused", size=13, subsize=11)
    s.footnote(600, "estop as a shadow field is refused: a latch must be immediate and must never replay.")
    s.motion = [("opts", "pulse"), ("desired", "flow"), ("delta", "flow"), ("refused", "flow"), ("rpc", "flow")]
    return s
