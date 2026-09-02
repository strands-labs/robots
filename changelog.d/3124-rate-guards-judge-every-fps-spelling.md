### Fixed: the recording rate guards judge every `fps` spelling their own domain accepts

`recorder_dataset_fps` and `rollout_rate_mismatch_reason` classified `fps` with
`isinstance(value, int | float)`. `numpy.int64` and `numpy.float32` are neither an
`int` nor a `float` subclass, so both guards read a whole rate as "no readable
rate" and returned `None` -- which their callers treat as "do not judge". The
rate-disagreement refusal was therefore skipped and the episode was written on a
timebase that mislabels it, with `start_recording` returning `status="success"`.

The values were in domain, not out of it. `positive_whole_number_error` -- the
domain `start_recording` runs `fps` through *before* either guard is asked --
classifies on `numbers.Real` and accepts `np.int64(60)` and `np.float64(30.0)`,
pinned by `tests/simulation/test_dataset_recording_fps_contract.py`. So the
narrowing declined to judge a rate the surface had just accepted, which is the one
case these guards exist for.

Measured with `fps` spelled four ways against a rollout capturing at 50 Hz, the
three orderings of this single disagreement split apart:

    fps                recorder_dataset_fps   rollout    dataset    requested
    30                                 30     REFUSED    REFUSED    REFUSED
    np.int64(30)                     None     -          -          REFUSED
    np.float32(30.0)                 None     -          -          REFUSED
    np.float64(30.0)                   30     REFUSED    REFUSED    REFUSED

`numpy.float64` IS a `float` subclass, so the narrowing held for one numpy
spelling and not its siblings; `requested_rate_mismatch_reason` already classified
on `numbers.Real`, which is why one of the three orderings was unaffected and the
split went unnoticed.

`tests/simulation/test_recording_rate_matches_control_frequency.py` already
asserts the three orderings never disagree about a pair -- that invariant is what
broke -- but its parametrization carried `fps` as a plain `int` only, so the
disagreement sat outside its population.

Both guards now classify on `numbers.Real` and route the boolean question to the
shared `is_boolean` predicate, matching `requested_rate_mismatch_reason` in the
same module and catching `numpy.bool_`, which is not a `bool` subclass. The
cross-ordering parametrization gains the numpy spellings, and each guard gains
focused coverage asserting an in-domain rate is judged rather than passed through.

`rollout_rate_mismatch_reason` needs no `try` around the `float(fps)` that follows
its guard: it is asked only after the fps domain, which refuses a value beyond the
float64 range with a reason of its own. `recorder_dataset_fps` reads its rate off
disk, where no domain has been asked, so it does need one -- see the entry for that
guard below.
