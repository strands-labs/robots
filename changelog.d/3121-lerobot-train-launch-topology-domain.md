### Fixed: `lerobot_train` refuses a launch topology it cannot honor

`build_train_command` reads `num_gpus` twice -- as the `num_gpus > 1` selector
that picks `accelerate launch --multi_gpu` over a direct
`python -m lerobot.scripts.lerobot_train`, and as the `--num_processes=` token
that sizes that accelerate launch -- and screened it with a local
`if num_gpus < 1` comparison, which guards neither read.

`nan` is not less than one and not greater than one either, so it fell through
to the single-process branch; `True` did the same, as `1`. Both built a run on a
topology nobody asked for and the tool reported it started. `2.7`, `2.0` and
`inf` *are* greater than one, so they selected the multi-process path and were
written into the argv verbatim, where `accelerate`'s own `type=int` parse
rejects them -- inside the DETACHED process, whose only record is the training
log. `"2"`, `None` and `[2]` could not be ordered against `1` at all, so the
comparison raised `TypeError` and the tool answered
`Tool execution failed: '<' not supported between instances of 'str' and 'int'`,
naming neither the parameter nor the remedy.

The trainer surface for the same lerobot run already held this field to the
shared positive-count domain (`launch_topology_problems`, reached from
`LerobotTrainer.validate`), so one parameter had two contracts depending on
which surface built the launch, and they disagreed on five of thirteen probed
values. `num_gpus` now routes through that same
`positive_count_error(value, "num_gpus", "lerobot_train")` -- the domain this
module already applies to `steps`, `batch_size`, `lora_r` and `lora_alpha` --
so every refusal names the parameter, quotes the value and reaches the caller as
an error envelope before any process starts.

The structural half of the contract moves with it. The re-implementation sweep
that owned this rule was rooted at the trainer package and keyed on
`spec.num_gpus`, so a bare `num_gpus < 1` on a tool parameter was not in its
population at all: the same defect in a second spelling, and only the first was
graded. The sweep is now package-wide, reads both spellings, and reads the AST
rather than the text so a docstring or comment explaining the domain cannot
register as a re-implementation of it.
