"""D07: the mesh. Devices announce themselves, a peer discovers and invokes them, one e-stop reaches them all."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d07_mesh",
        "The mesh",
        "Every mounted robot is a device: it announces itself, answers rpcs through its own safety chain, and "
        "latches on one e-stop from any peer in its tenant.",
        "Left, the one green element: a Mesh in tenant lab-a, with discover, invoke, subscribe, connect and the "
        "use_mesh tool. In the middle, a dashed layer for the transport, zenoh in peer mode or AWS IoT Core, with "
        "the keys it carries: presence, manifest, rpc, state and the tenant e-stop. Right, three devices: two arms "
        "labelled role:arm and a camera. Each device sends its presence and record to the left on a dashed wire. "
        "A query, discover device(role:arm), returns the two arms. An invoke act goes to arm so101-a and through "
        "its safety chain. One broadcast estop fans out to all three devices, which latch. Below, the watchdog "
        "holds a device whose invoker vanished, and resume is the stopping operator's or a signed key's.",
        h=660,
    )
    s.section(60, 120, "a peer", accent=True)
    s.box(60, 132, 270, 300, 'Mesh("zenoh://")', "tenant lab-a, one or more transports",
          accent=True, size=14, subsize=12, id="mesh")
    s.chips(74, 206, ['discover("device(role:arm)")'])
    s.chips(74, 242, ["invoke", "subscribe"])
    s.chips(74, 278, ["connect(id)", "broadcast"])
    s.chips(74, 314, ["mesh.tool: use_mesh"])
    s.chips(74, 350, ["resume_authority"])

    s.section(410, 120, "the transport, keys in one table")
    s.box(410, 132, 350, 300, None, None, dashed=True)
    s.box(430, 186, 310, 58, "zenoh peer mode", "scouting on 224.0.0.224:7447, zero config", size=13, subsize=11)
    s.box(430, 254, 310, 58, "AWS IoT Core", "rpcs as confirmed Direct Messages", size=13, subsize=11)
    s.chips(430, 326, ["@presence/<id>", "<id>/manifest"])
    s.chips(430, 360, ["<id>/rpc/<name>", "<id>/state"])
    s.chips(430, 394, ["@estop", "@peer/<peer>"])

    s.section(840, 120, "devices")
    s.box(840, 132, 300, 88, "so101-a", "role:arm, safety chain then body", size=14, subsize=12, id="dev_a")
    s.box(840, 238, 300, 88, "so101-b", "role:arm, safety chain then body", size=14, subsize=12, id="dev_b")
    s.box(840, 344, 300, 88, "cam-1", "role:camera", size=14, subsize=12, id="dev_c")

    s.arrow([(330, 160), (840, 160)], label="invoke act", label_dx=-34, label_dy=-8, id="invoke")
    s.arrow([(840, 200), (760, 200)], dashed=True, label="announce", label_dx=-26, label_dy=-8, id="ann_a")
    s.arrow([(840, 290), (760, 290)], dashed=True, id="ann_b")
    s.arrow([(840, 400), (760, 400)], dashed=True, id="ann_c")
    s.arrow([(410, 214), (330, 214)], dashed=True, label="2 arms", label_dx=-19, label_dy=-8, id="found")

    s.arrow([(195, 432), (195, 500), (1172, 500), (1172, 176), (1140, 176)],
            label="broadcast estop: every device latches, nothing is queued", label_dx=12, label_dy=4, id="estop")
    s.arrow([(1172, 282), (1140, 282)], id="estop_b")
    s.arrow([(1172, 388), (1140, 388)], id="estop_c")
    s.box(60, 540, 520, 56, "watchdog", "the invoker vanished or went quiet: on_disconnect (hold)", size=13,
          subsize=11)
    s.box(620, 540, 520, 56, "resume", "the operator who stopped, a named authority, or a signed key", size=13,
          subsize=11)
    s.footnote(626, "Names are claimed, not proven: off a loopback network, resume needs an operator key.")
    s.motion = [("mesh", "pulse"), ("ann_a", "flow"), ("ann_b", "flow"), ("ann_c", "flow"), ("found", "flow"),
                ("invoke", "flow"), ("estop", "flow"), ("estop_b", "flow"), ("estop_c", "flow")]
    return s
