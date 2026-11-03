"""D09: an episode recorded with record= is replayed by Policy(dir); a local path is never sent to the Hub."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d09_record_replay",
        "Record an episode, replay it",
        "A run writes one episode of a LeRobot v3 dataset; the same directory, given back as a Policy, plays it.",
        "Left, robot.run with record= writes a LeRobot v3 dataset, the middle box, holding the parquet action "
        "column, meta/info.json with its fps and meta/units.json with its units. A wire runs right to the green "
        "element, Policy of the directory with an episode query, which reads that episode's actions. A dashed "
        "return wire runs back under the boxes to robot.run: the replay is a policy like any other, at the "
        "dataset's fps and in its units. Below, a dashed refused path: a directory with neither config.json nor "
        "meta/info.json is a ValueError, never a Hub request.",
        h=600,
    )
    s.section(60, 122, "a run")
    s.box(60, 134, 300, 200, "robot.run", "record= keeps the run as one episode", size=14, subsize=12, id="run")
    s.chips(74, 254, ["record="])

    s.section(450, 122, "LeRobot v3 dataset")
    s.box(450, 134, 330, 200, "data/wave", "state and action in the policy's units", dashed=True, size=14,
          subsize=12, id="ds")
    s.chips(464, 214, ["action", "meta/info.json"])
    s.chips(464, 260, ["meta/units.json"])
    s.arrow([(360, 234), (450, 234)], label="write", label_dx=-20, label_dy=-8, id="w1")

    s.section(870, 122, "replay")
    s.box(870, 134, 270, 200, "Policy(dir)", "one episode, at its fps, in its units", accent=True, size=14,
          subsize=12, id="pol")
    s.chips(884, 254, ["?episode=n"])
    s.arrow([(780, 234), (870, 234)], label="read", label_dx=-18, label_dy=-8, id="w2")
    s.arrow([(1005, 334), (1005, 380), (210, 380), (210, 334)], dashed=True, label="run(policy)", label_dx=10,
            label_dy=-8, id="w3")

    s.section(450, 418, "refused")
    s.box(450, 430, 690, 90, "ValueError", "a directory with neither config.json nor meta/info.json; no Hub request",
          dashed=True, size=14, subsize=12, id="no")

    s.footnote(566, "Before, a dataset path given back as a policy went to the Hub and came back a 404.")
    s.motion = [("pol", "pulse"), ("w1", "flow"), ("w2", "flow"), ("w3", "flow")]
    return s
