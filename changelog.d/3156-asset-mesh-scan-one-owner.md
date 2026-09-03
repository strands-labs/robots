### Fixed: one owner decides whether a model's meshes are on disk

`_needs_download` (should a robot's assets be fetched?) and
`MuJoCoSimEngine._ensure_meshes` (may `add_robot` proceed?) ask the same
question about the same model. `_ensure_meshes` reads the model the way MuJoCo
does -- the main file plus every `<include>`d fragment, with the `<compiler>`
mesh directory taken across fragments. `_needs_download` read the main file
alone, so a model whose meshes only an included fragment declares had no mesh
reference to check, which is indistinguishable from a model whose meshes are
all present.

Shipped Menagerie assets are built that way: `ability_hand`'s `scene.xml`
declares none of its 13 meshes, and its included hand fragment declares all 13
plus the `meshdir`. With those meshes absent MuJoCo refuses the model and
`_ensure_meshes` correctly asks for a download -- and the download it asked for
was a no-op, because the download path reported the assets present, leaving the
caller with the mesh-not-found error that check exists to prevent. `force=True`
could not reach such a model either: the early return for "no mesh references"
was taken before `force` was consulted, so
`download_robots(names=["ability_hand"], force=True)` answered "All 1 robots
already have assets. Use force=True to re-download."

The scan is now one function, `_mjcf_missing_meshes`, beside the resolution rule
it applies; both callers use it and the private copy in `_ensure_meshes` is
deleted, which also settles two smaller asymmetries between the copies (the mesh
extension set, and which fragments count as the model). The two paths keep their
opposite readings of an *unreadable* model, now stated in one place rather than
emerging from two implementations: the download path fetches, because a fetch is
what replaces a file it cannot read, while the sim path proceeds, because MuJoCo
names the unreadable file itself on the load that follows.
