### Fixed: an episode's frame range is read from that episode's own row

Three readers in this package resolve the same fact - which slice of the global
frame index episode `N` occupies - and each walks a compatibility ladder over
the shapes LeRobot has used to record it.
`strands_robots.dataset_recorder.load_lerobot_episode` was the only one of the
three with no `dataset_from_index` rung at all: it asked for
`episode_data_index` and, on any dataset without it, recomputed the range by
accumulating `length` over every *preceding* episode row.

`episode_data_index` is not a shape any supported LeRobot exposes. The declared
range is `lerobot[feetech,dataset]>=0.6.1,<0.7.0`, and the string occurs 0 times
in `LeRobotDataset` on 0.6.2 *and* on 0.5.1 (below the floor), with
`hasattr(ds, "episode_data_index")` False on a real instance of each. So the
accumulation was not a fallback - it was the only rung an accepted index could
reach, while the row it was recomputing from already stated the answer:

    meta.episodes[1] -> {'dataset_from_index': 4, 'dataset_to_index': 11,
                         'length': 7, 'episode_index': 1, ...}

Measured on a real 300-episode dataset, resolving one episode's range:

    episode  accumulation  row read  ratio  metadata row fetches
    0        0.21 ms       0.095 ms  2x     1 vs 1
    50       2.91 ms       0.060 ms  49x    51 vs 1
    150      8.02 ms       0.053 ms  151x   151 vs 1
    299      14.63 ms      0.047 ms  314x   300 vs 1

Both spellings return the same numbers, so this was never a wrong range - it was
a linear scan standing in for a constant-time read, behind a broad
`except Exception` whose last resort decodes the dataset frame by frame. The two
fallback rungs are kept and now pinned: a pre-0.6 dataset still resolves through
the tensor index, and one carrying neither still accumulates lengths.

Two docstring claims rested on the dropped rung and were false because of it.
`load_lerobot_episode` justified the ordering of its index guard by saying an
accepted index "reaches the O(1) `episode_data_index` lookup", naming a rung no
supported LeRobot provides. `_SourceDataset._episode_frame_range` in
`strands_robots/transforms/base.py` claimed "Same ladder as
`load_lerobot_episode`", and the function it named had neither that leading rung
nor its order. Adding the rung is what makes both claims true, which is why this
is one change rather than a code fix plus two prose corrections.
