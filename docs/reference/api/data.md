---
description: Stream a LeRobot dataset without a full download, sync one to a bucket, and label an episode.
---

# Data

Datasets are LeRobot v3 datasets on disk or on the Hub. These functions stream them without a full download, sync them to a bucket, and grade episodes.

## Streaming

::: strands_robots.streaming_dataset
    options:
      heading_level: 3
      members:
        - StreamingDatasetReader
        - stream_dataset
        - has_streaming_dataset

## Transfer

::: strands_robots.dataset_transfer.sync_dataset_to_bucket
    options:
      heading_level: 3
      show_root_heading: true

## Episode judging

::: strands_robots.tools.episode_judge
    options:
      heading_level: 3
      members:
        - create_judge_agent
        - load_episode
        - sample_frames
        - read_predicate_verdict
        - write_label

::: strands_robots.tools.decider_judge.DeciderJudge
    options:
      heading_level: 3
