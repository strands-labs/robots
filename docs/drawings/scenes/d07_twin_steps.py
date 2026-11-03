"""D07: a real act() writes the bus, mirrors the twin and steps the twin's world once; every refusal is one name."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d07_twin_steps",
        "A real act() steps its twin",
        "One act() on a hardware Robot writes the bus, mirrors the twin and advances the twin's world one control "
        "step; every refusal is RequestRefused.",
        "Left, real.act(a) on a hardware SO-101. A wire runs right to the Driver, which writes the servo bus. A "
        "second wire runs down to the twin box: write the targets into so101_twin, then the green element, one "
        "control step of Simulation.default(), the same step a simulated act() takes. A dashed wire from "
        "manual_time() shows the step suppressed when the caller owns time; observe() mirrors without stepping. "
        "Right, a dashed refused path: the operator gate, a limit or the e-stop raises RequestRefused, a "
        "PermissionError imported from strands_robots, and nothing is written.",
        h=600,
    )
    s.section(60, 122, "your code")
    s.box(60, 134, 340, 120, "real.act(a)", "a hardware Robot; the same verb as on a simulated body",
          size=14, subsize=12, id="act")
    s.chips(74, 214, ["is_real", "real.twin"])
    s.section(460, 122, "the bus")
    s.box(460, 134, 300, 120, "Driver: feetech", "write() on the STS3215 servos, then read back", size=14,
          subsize=12, id="bus")
    s.arrow([(400, 194), (460, 194)], label="targets", label_dx=-24, label_dy=-8, id="w1")

    s.section(820, 122, "refused")
    s.box(820, 134, 320, 120, "RequestRefused", "a PermissionError from strands_robots; nothing is written",
          dashed=True, size=14, subsize=12, id="no")
    s.chips(834, 214, ["EStopped"])
    s.arrow([(760, 160), (820, 160)], dashed=True, label="no", label_dx=-8, label_dy=-8, id="w2")

    s.section(60, 330, "the twin, in Simulation.default()")
    s.box(60, 342, 1080, 170, None, None, dashed=True)
    s.box(80, 362, 320, 120, "so101_twin", "mirror: the same targets written, under the world lock", size=14,
          subsize=12, id="twin")
    s.box(460, 362, 320, 120, "one control step", "the step a simulated act() takes; now the twin moves",
          accent=True, size=14, subsize=12, id="step")
    s.box(840, 362, 280, 120, "manual_time()", "the caller steps instead; observe() never steps", size=14,
          subsize=12, id="manual")
    s.arrow([(160, 254), (160, 362)], label="mirror", label_dx=10, label_dy=4, id="mirror")
    s.arrow([(400, 422), (460, 422)], id="w3")
    s.arrow([(840, 422), (780, 422)], dashed=True, label="off", label_dx=-14, label_dy=-8, id="w4")

    s.footnote(556, "Before, the twin held its old pose until the script stepped real.twin.sim by hand.")
    s.motion = [("step", "pulse"), ("w1", "flow"), ("mirror", "flow"), ("w3", "flow")]
    return s
