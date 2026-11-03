"""D16: an agent's run on a real arm asks the operator once, with the whole plan, then checks every step."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d16_one_yes_per_run",
        "One yes for the whole run",
        "An agent's run on a real body asks the operator once, with the plan, before anything is acquired; "
        "every step then still passes the safety chain, without asking again.",
        "Left to right, top row: a box an agent's run leads by a wire to a green box approve(RunPlan), asked once "
        "before acquire, then by a wire to a box the loop whose steps each pass the chain. Under the loop a row of "
        "chips names the chain: e-stop, range, safe, interlocks, validators. Lower left, a dashed box shows the "
        "refused path: no approve leaves nothing on the bus. A footnote notes confirm_act stays the per-command gate.",
        h=580,
    )
    s.section(60, 122, "an agent's run")
    s.box(60, 134, 250, 120, "run(policy)", "hz, seconds, instruction", size=14, subsize=12, id="run")
    s.section(360, 122, "asked once", accent=True)
    s.box(360, 134, 300, 120, "approve(RunPlan)", "before anything is acquired", accent=True, size=14,
          subsize=12, id="approve")
    s.section(710, 122, "then every step")
    s.box(710, 134, 430, 120, "the loop", "observe, decide, write", size=14, subsize=12, id="loop")
    s.arrow([(310, 194), (360, 194)], label="one yes", label_dx=-30, label_dy=-10, id="w1")
    s.arrow([(660, 194), (710, 194)], id="w2")

    s.section(710, 320, "the chain, no operator re-ask")
    s.chips(724, 350, ["e-stop", "range", "safe", "interlocks", "validators"])
    s.arrow([(925, 254), (925, 344)], id="w3")

    s.section(60, 320, "no approve")
    s.box(60, 332, 400, 110, "refused", "nothing reaches the bus", dashed=True, size=14, subsize=12, id="refused")
    s.arrow([(185, 254), (185, 332)], dashed=True, label="no", label_dx=10, label_dy=4, id="w4")

    s.footnote(540, "Before, confirm() was asked on every tick (30 a second); confirm_act stays the per-command gate.")
    s.motion = [("approve", "pulse"), ("w1", "flow"), ("w2", "flow"), ("w3", "flow")]
    return s
