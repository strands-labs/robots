### Fixed: an approved agent run loads exactly the policy files the operator approved

An approved agent `run()` on a real robot checked the policy files against `RunPlan.files` after loading them from
their original path, so a file replaced only while the loader was reading it (and put back afterwards) passed the
check. The run now takes a private copy of the files the loader reads, hashing each byte as it is written, refuses
the run before anything loads if the copy differs from the approved digest, loads the copy, and removes it when the
run ends, however it ends. Which files a spec reads is decided by the same rule the loader uses
(`policy.factory.local_root`), so a preset name keeps meaning the preset even next to a same-named folder. FIFOs and
folder cycles inside a policy folder are skipped or visited once, the same way in the digest and the copy.
