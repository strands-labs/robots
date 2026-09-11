### Added: an evaluation reports how close it came, and how often it can be relied on

`PolicyRunner.evaluate` reported a total reward and a success rate. Neither answers
the two questions a reader of a failing evaluation actually has.

**`max_step_reward` per attempt, `avg_max_step_reward` in the aggregate.** On a
dense-reward task the running total is partly a step count: a long flailing attempt
accrues more shaped reward than a short one that nearly finished, so the total
ranks them backwards. The peak single-step reward is the signal that separates
"never approached" from "approached and missed". Measured on `go2_walk_forward`
with the mock policy over 3 attempts, the total is 500.25 and the peak is 0.51 -
the same rollout, and only one of those numbers says anything about proximity.
LeRobot has reported both since it shipped (`avg_sum_reward` beside
`avg_max_reward` in `src/lerobot/scripts/lerobot_eval.py`); this package reported
only the total.

`None` for an attempt that ended before `on_step` scored anything, and the
aggregate averages only the attempts that scored, rather than reading an unscored
attempt as a peak of zero and dragging the average toward it for a reason unrelated
to proximity.

**`pass_hat_k`.** A success rate answers how often a policy works. It does not
answer whether it can be relied on repeatedly, and for anything driven in a loop
that is the deployable question: a policy at 60% clears five consecutive attempts
about 8% of the time. Two checkpoints at the same mean - one consistent, one
erratic - were indistinguishable in the payload.

Estimated as `C(c, k) / C(n, k)` for `c` successes out of `n` completed attempts,
which is the unbiased probability that a uniformly drawn `k`-subset of the attempts
observed is all successes. Deliberately **not** `success_rate ** k`. That form
assumes attempts are independent, and attempts on one policy against one scene are
correlated by construction - a systematic grasp offset fails every attempt rather
than a fixed fraction of them - so it reports a reliability the run never
demonstrated. On 3 successes out of 5 the two disagree by half, 0.30 against 0.36,
and the exponential form is the optimistic one, which is the direction that costs a
deployment decision rather than merely being wrong.

Keys above `episodes_completed` are absent rather than `0.0`: a run of 3 attempts
has not measured a 5-streak, and a zero there would read as measured unreliability
instead of an unasked question, so a reader comparing runs of different lengths
would be comparing a measurement against an absence. Capped at `k=8`, beyond which
the estimate is dominated by its own variance at the episode counts an evaluation
runs; a reader needing more can compute it from `n_success` and
`episodes_completed`, both of which are in the same payload. Keys are strings
because the payload is JSON and an int-keyed dict changes shape on serialisation -
which is where an agent reads it.

Neither this package nor LeRobot reported any repeated-trial figure before this;
`pass@k` / `pass^k` appears nowhere in LeRobot's eval path, checked at `b6ec006`.

Pinned by `tests/simulation/test_eval_reliability_and_peak_reward.py`, which uses a
`constant`-reward probe task so the peak is a value the test chose rather than a
property of a scene - a shaped term would leave a failure unattributable between
the arithmetic under test and the physics. It asserts the peak is below the total
over a multi-step attempt (a field aliasing the total passes an equality check on a
single-step episode and fails that one), the exponential form is pinned as a
difference rather than left as a comment, monotonicity in `k` is asserted because a
hand-rolled combinatorial formula is easy to get subtly wrong, and the payload's
`pass_hat_k` is round-tripped through JSON.
