### Fixed: a rollout reports the video rate it wrote, not the one requested

`video={"fps": N}` is a request: a rollout renders at most one frame per applied
control step, so the writer caps the MP4 at `control_frequency` when `N` exceeds
it. The result quoted `N` anyway and the structured payload carried no rate at
all, so at `control_frequency=10` with the default `fps=30` a 2.0-second clip
reported as `20 frames, 30fps` and computed to 0.67 s. `run_policy` now reports
the rate the file plays at, in the text and as `video_fps` in the payload
(`None` when no MP4 was written).
