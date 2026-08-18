### Fixed

- **training/rl**: `EmpiricalNormalization.update()` is now a no-op in eval mode, so the documented eval-mode freeze holds for a direct `update()` call and not only for `forward(update=True)`. The running statistics are persistent buffers, so a batch folded in after `eval()` was written into the next checkpoint and changed how an exported policy whitened every observation, with nothing raised and nothing logged. Training-mode behavior, the `until` warmup freeze and the empty-batch no-op are unchanged.
