### Added: `DeciderJudge` labels recorded episodes with a Strands Decider, and defers what it is unsure of

`strands_robots.tools.decider_judge.DeciderJudge(url)` is an episode-judge
backend beside `create_judge_agent`. It asks a running Strands Decider vision
server (`strands-decider serve <checkpoint> --vision`) three typed questions
per episode in one request, over up to four frames from `sample_frames`: a
quality grade (a choice over `QUALITY_GRADES`, about execution and not
outcome), a failure-mode tag (a choice over `FAILURE_MODES` plus `none`,
written as `None`) and a yes/no success opinion. The deterministic verdict is
never sent, and the label is written through `write_label`, so verdict
precedence and `disputes_verdict` are unchanged. An episode is written only
when every answer's calibrated confidence exceeds `min_confidence` (default
0.6, which defers a yes/no between 0.2 and 0.8); otherwise nothing is written
and the episode is reported `deferred` with its probabilities, for the agent
judge or a person. Existing labels are kept unless `overwrite=True`.
`judge_episode` and `judge_dataset` return the `{"status", "content"}`
envelope and report a failing episode instead of raising. The client is the
standard library, so no dependency or extra is added; the decider runs in its
own environment. A live test (`pytest -m decider`, gated on
`STRANDS_DECIDER_URL`) records two MuJoCo episodes and judges them against a
real server.
