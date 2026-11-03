"""D14: the default world a policy sees; a checker ground under toned lighting instead of white on white."""

from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d14_world_a_camera_sees",
        "A world a camera sees, not a white blob",
        "The default ground is a checker texture under toned lighting, so a policy gets edges and colour "
        "instead of white on white; ground= chooses the floor and an unknown value is refused by name.",
        "Left, before, a dashed card: mean pixel 233 of 255, a white arm on a white plane, no texture and "
        "no edge cue. A dashed wire labelled was runs right to the green card, now: mean pixel 93 of 255 "
        "under a checker ground with headlight diffuse 0.5 and ambient 0.15, one texture, edges and colour. "
        "Right, a card Simulation(ground=) over three chips, checker, plain and False, the look of the floor "
        "a caller chooses.",
        h=400,
    )
    s.section(60, 122, "before")
    s.box(
        60, 134, 340, 150,
        "mean 233 of 255",
        "a white arm on a white plane; no texture, no edge or colour cue for a policy",
        dashed=True, size=14, subsize=12, id="before",
    )
    s.section(430, 122, "now", accent=True)
    s.box(
        430, 134, 360, 150,
        "mean 93 of 255",
        "a checker ground, headlight diffuse 0.5 and ambient 0.15; one texture, edges and colour",
        accent=True, size=14, subsize=12, id="now",
    )
    s.arrow([(400, 209), (430, 209)], dashed=True, label="was", label_dx=6, label_dy=-6, id="w1")

    s.section(830, 122, "the floor you choose")
    s.box(
        830, 134, 310, 150,
        "Simulation(ground=)",
        "the look of the world's floor, one lock around the world",
        size=14, subsize=12, id="opt",
    )
    s.chips(844, 232, ["checker", "plain", "False"])

    s.footnote(344, "light_scale now has headroom in both directions, up for a brighter table and down for a dim one.")
    s.motion = [("now", "pulse"), ("w1", "flow")]
    return s
