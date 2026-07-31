### Fixed: the RTC re-query horizon is refused when the consumer cannot execute it

`LerobotLocalPolicy(rtc_execution_horizon=...)` was stored verbatim while its
sibling `actions_per_step` - the count it *replaces* whenever Real-Time Chunking
is active, in the very same `Policy.execution_horizon` property - has been
validated on the shared `chunk_count_error` domain. So the same horizon was
refused under one name and accepted under the other, and the read path then
reconciled it two different ways: `execution_horizon` floors it with
`max(1, int(...))` for the consumer, while `_predict_with_rtc` hands the model
the raw value. Consumer and model therefore disagreed about how many actions run
before the re-query, which is precisely the agreement RTC's cross-chunk blend
rests on.

`0` was the sharpest spelling. It is falsy, so the property fell through to the
full trained chunk and the consumer replayed it open-loop - the regime the
property's own docstring says "keeps that tail empty and silently collapses RTC"
- while `_init_rtc` skipped adopting the checkpoint's own
`rtc_config.execution_horizon` because `0` is not `None`, and the model was told
`0`. Measured on a checkpoint with a 100-action trained chunk and a declared RTC
horizon of 10: omitting the parameter gives 10 to both sides, `0` gave the
consumer 100 and the model 0, all under a successful construction. `-5` gave 1
and `-5`, `2.7` gave 2 and `2.7`, `True` a silent 1, and `nan` / `inf` / a list
surfaced as a bare `ValueError` / `OverflowError` / `TypeError` from a property
read inside the rollout loop.

A supplied horizon is now checked where it arrives - in the constructor, before
any checkpoint is downloaded, and in `preflight`, so a rollout entry point
reports it as a structured error rather than a raise. `None` remains the
documented request to adopt the checkpoint's own horizon and is unaffected.
