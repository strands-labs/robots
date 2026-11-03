"""D19: the edge. A mesh device runs as a Greengrass v2 component; each camera streams over KVS."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d19_edge",
        "At the edge: a component, and its cameras",
        "Greengrass v2 runs the mesh as a component; Kinesis Video Streams carries each camera.",
        "Left, the one green element: a Greengrass v2 component com.strands.robots running greengrass.main(), "
        "which mounts Robot('so101') on one Mesh and stays up until SIGTERM; a chip notes robot to robot traffic "
        "stays on zenoh so invoke act stays at p50 0.78 ms with the cloud link cut. One wire labelled deploy() goes "
        "to greengrass.deployment(), which targets the tenant's thing group, the fleet, reaching RUNNING in 11.4 s "
        "locally and 48 s in the cloud. A second wire labelled observe(images) goes to kvs, which names one stream "
        "and one signaling channel per camera sensor, strands-tenant-device-camera, written into the manifest "
        "shadow; camera_entries() and publish_cameras() are its verbs. A dashed box to the right says that without "
        "the kvssink plugin the producer refuses by name, there being no PutMedia in boto3.",
        h=572,
    )

    s.section(60, 118, "the host", accent=True)
    s.box(60, 150, 340, 176, "com.strands.robots",
          "greengrass.main(): mount Robot('so101') on one Mesh, stay up until SIGTERM",
          accent=True, size=14, subsize=12, id="component")
    s.chips(74, 338, ["robot to robot on zenoh", "act p50 0.78 ms, link cut"])

    s.section(520, 118, "the fleet")
    s.box(520, 150, 330, 110, "greengrass.deployment()",
          "to the tenant's thing group, one host at a time", size=14, subsize=12, id="fleet")
    s.chips(534, 272, ["RUNNING in 11.4 s", "cloud 48 s"])
    s.arrow([(400, 212), (460, 212), (460, 196), (520, 196)], label="deploy()", id="w1")

    s.section(520, 332, "cameras")
    s.box(520, 364, 330, 110, "kvs: one stream and channel per camera",
          "named strands-tenant-device-camera, in the manifest shadow", size=13, subsize=12, id="kvs")
    s.chips(534, 486, ["camera_entries()", "publish_cameras()"])
    s.arrow([(400, 264), (460, 264), (460, 418), (520, 418)], label="observe(images)", id="w2")

    s.box(900, 364, 280, 110, None, None, dashed=True, id="refuse")
    s.text(914, 400, "no PutMedia in boto3: without", cls="mono fg", size=12.5)
    s.text(914, 424, "kvssink the producer refuses", cls="fg", size=12.5)
    s.text(914, 448, "by name, it does not pretend", cls="fg", size=12.5)
    s.arrow([(850, 418), (900, 418)], id="w3")

    s.footnote(540, "The mesh does not depend on the cloud link, and import strands_robots.mesh loads no "
                    "boto3, awsiot or gi.")
    s.motion = [("component", "pulse"), ("w1", "flow"), ("w2", "flow"), ("fleet", "visit"), ("kvs", "visit")]
    return s
