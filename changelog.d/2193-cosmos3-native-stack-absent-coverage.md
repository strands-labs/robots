### Quality: drive both decisions the in-process Cosmos 3 backend makes about a missing native stack

`policies/cosmos3/policy_diffusers.py` is optional and lazy: `diffusers` +
`torch` + `transformers` is a heavy GPU stack, so `cosmos3-service` stays the
default install and every native import sits inside the function that needs it.
The module answers that one missing import two opposite ways, three lines apart.
`_install_hint()` -- the single actionable message, naming both the extra to
install and the service backend to fall back to -- is raised from three sites;
`_to_numpy` swallows the same `ImportError` and hands the value straight to
NumPy, which is what lets a `policy`-mode rollout complete through the
documented `pipeline=` / `condition_cls=` injection seams with no torch at all.

Two of the three refusal sites had a dedicated test; `_as_action_tensor` had
only a happy-path one, and the degradation branch had none, so neither decision
was held to its documented direction. A regression making `_as_action_tensor`
degrade would hand the pipeline a NumPy array where it needs a tensor, and one
making `_to_numpy` refuse would take the whole no-GPU injection path down with
it. Both are now driven end to end on a torch-free install, alongside the
asymmetry itself (`forward_dynamics` refuses where `policy` mode completes, on
one install and one backend) and an exact pin on the set of sites that report
the shared remedy, so a fourth lazy native import cannot join the untested half.

No library behaviour changes. `policy_diffusers.py` goes from 97% to 100%
statement coverage.
