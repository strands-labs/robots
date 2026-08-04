### Fixed: the launch topology a TrainSpec asks for is one shared positive-count domain

`TrainSpec.num_gpus` and `TrainSpec.num_nodes` are the two process counts a
distributed run is sized from, and neither had a domain - while their run-size
neighbours `steps` and `global_batch_size` shared one. Each is read in three
places by the three supervised backends: a `spec.num_gpus > 1` /
`spec.num_nodes > 1` test that selects between the single-process and the
multi-process launch path, a `nproc_per_node` / `nnodes` argument to torch's
`elastic_launch`, and a `--nproc_per_node=` / `--nnodes=` / `--num_gpus=` argv
token. Every way an unusable count failed was silent or late:

* `0`, a negative, `nan` and `True` all compare as *not* greater than one -
  `nan` compares false against everything - so the selector routed them to the
  single-process path and the run proceeded on one process under a successful
  result. The topology the caller asked for simply was not the one that ran, and
  for `num_nodes` that also slipped past the multi-node refusal LeRobot and
  GR00T raise for a topology they cannot launch in-process: `num_nodes=8` is
  refused, `num_nodes=nan` was not.
* `2.7` and `inf` *are* greater than one, so they selected the multi-process
  path and reached `elastic_launch` as the worker count.
  `torch.distributed.launcher.api.LaunchConfig` accepts `2.7`, `inf`, `0`, `-4`
  and `nan` without complaint, so nothing downstream rejected them either.
* A string, `None` or a list raised `TypeError: '>' not supported between
  instances of 'str' and 'int'` out of the comparison itself - from inside a
  `Trainer.validate` documented to *return* problems.

Both fields are now checked by `launch_topology_problems`, a fourth shared gate
in `strands_robots.training._validate` alongside the injection-safety, run-size
and learning-rate ones, reached through `Trainer._launch_topology_problems` and
built on the same `positive_count_error` domain the run-size gate uses. Every
`Trainer.train` already fails closed on a `validate` problem, so the preflight
covers the launch path on every call route.

The gate is scoped like the run-size one rather than made universal: `TrainSpec`
documents that a backend "reads the fields it supports and ignores the rest", and
only the three supervised backends launch from either field, so the mock and RL
trainers must not report on them. That scoping is enforced structurally - the set
of backends required to route through the gate is derived from which modules
actually read `spec.num_gpus` / `spec.num_nodes`, so a fourth backend that starts
launching from one fails the parity test until it does.

LeRobot's and GR00T's multi-node refusals compare `num_nodes`, so each is now
reached only once the shared domain has established the value *is* a count - a
string would otherwise still raise out of that comparison. Both refusals still
fire for a usable count, which is pinned rather than assumed.
