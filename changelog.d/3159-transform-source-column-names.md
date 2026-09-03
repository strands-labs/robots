### Fixed: a dataset transform reads a source column's `names` as LeRobot writes them

`_SourceDataset` builds the output dataset's schema from the source's
`observation.state` / `action` names, and the pass-through fills those columns by
pairing each name with the source row -- so the name list decides how wide the
output trajectory is. It was read as `list(feature.get("names") or [])`.

LeRobot also writes that field as a mapping, and its own shipped teleoperators
do: `teleop_keyboard.action_features` declares `{"motors": list(self.arm.motors)}`
and `teleop_gamepad` declares `{"delta_x": 0, ...}`. On a mapping that call
yields the keys -- one name per group rather than per column -- so a six-motor arm
declared `{"motors": [...]}` transformed to a one-column action holding its
shoulder-pan value alone, and a bimanual `{"left": [...], "right": [...]}` went
from 12 columns to 2. Both returned `status="success"`, counted the episode in
`episodes_written`, and wrote provenance claiming it is its source episode, while
`docs/data/transforms.md` contract item 1 promises a generated episode is the
same trajectory rendered differently.

The two mapping spellings are read differently rather than flattened alike,
because they do not mean the same thing: a list value holds the components (its
key is a group label, not a column name), and an integer value is the
component's index (the keys are the column names, ordered by that index rather
than by insertion order). LeRobot's own `_flatten_feature_names` renders the
second shape's indices as the names, which would name a six-motor arm `0..5`.

A count that still disagrees with the column's declared width cannot be repaired
by reading it differently, so `_SourceDataset.open` refuses it -- the fourth
refusal there, beside the three it already makes for the same reason, that the
output could not be the source rendered differently. Names short of the width
would drop the trailing components; names past it would declare a column no
frame supplies, which `add_frame` writes as `0.0`, itself a travel-to-zero
command for an absolute-position actuator. The state and action pairings in
`_write_episode` are `strict` now, so a path that reached them with disagreeing
counts raises rather than quietly truncating a trajectory.

Sources declaring a flat list -- every recording this package writes -- are
unaffected.
