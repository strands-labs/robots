"""D01: Description.load: a name in the registry becomes a model file and the facts read from it."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d01_description",
        "Description.load",
        "A name in the registry becomes a model file, and the facts a world or a bus needs are read from the model.",
        "Left, the registry: robots.json and urdf_robots.json, with three ways to an asset: robot_descriptions, "
        "github and a file path. A wire labelled load(name) runs to the middle box, the model file, read through "
        "MuJoCo when installed or an XML walk when not. A wire runs right to the one green element, Description, "
        "with joints, action_keys, limits, keyframes and floating, and describe() as JSON. Under the registry a "
        "dashed wire ends at a dashed box: a robot with no model refuses by name with FileNotFoundError; an unknown "
        "name refuses with KeyError.",
        h=560,
    )
    s.section(60, 122, "the registry")
    s.box(60, 134, 330, 200, "registry/", "76 robots plus the URDF list; names and aliases fold to one entry",
          size=15, subsize=12, id="reg")
    s.chips(74, 226, ["robots.json", "urdf_robots.json"])
    s.chips(74, 256, ["robot_descriptions", "github"])
    s.chips(74, 286, ["model.xml path"])

    s.section(460, 122, "the model file")
    s.box(460, 134, 300, 200, "model_path", "cloned or fetched once, then read; a robot that only compiles "
                                            "inside its scene is read through the scene", size=15, subsize=12,
          id="file")
    s.chips(474, 256, ["mujoco", "XML walk"])
    s.chips(474, 286, ["scene_path"])

    s.section(830, 122, "what the robot is", accent=True)
    s.box(830, 134, 310, 200, "Description", "frozen; the same facts with or without MuJoCo installed",
          accent=True, size=15, subsize=12, id="desc")
    s.chips(844, 226, ["joints", "action_keys"])
    s.chips(844, 256, ["limits", "keyframes"])
    s.chips(844, 286, ["floating", "describe()"])

    s.arrow([(390, 234), (460, 234)], label="load(name)", label_dx=0, label_dy=-8, label_anchor="middle", id="w1")
    s.arrow([(760, 234), (830, 234)], label="read", label_dx=0, label_dy=-8, label_anchor="middle", id="w2")

    s.down(225, 334, 400, dashed=True, label="no model", label_dx=10, label_dy=4)
    s.box(60, 400, 1080, 64, None, None, dashed=True)
    s.text(74, 438, "refused by name", cls="mono fg", size=14, weight=500)
    s.text(240, 438, "hardware only: FileNotFoundError naming the robot; not in the registry: KeyError",
           cls="grot text", size=12.5)
    s.footnote(500, "One test pins all 76 registry robots: each loads a Description or refuses by name.")
    s.motion = [("w1", "flow"), ("w2", "flow"), ("desc", "pulse")]
    return s
