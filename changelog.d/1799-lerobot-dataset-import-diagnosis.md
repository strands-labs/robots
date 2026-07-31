### Fixed: start_recording names the lerobot dependency that is missing

`start_recording` gated on `has_lerobot_dataset()`, which collapses every way
importing `lerobot.datasets.lerobot_dataset` can fail into a single `False`, and
all three backends turned that `False` into one message: *"requires the lerobot
extra: pip install 'strands-robots[lerobot]'"*. In a partially-provisioned
environment - lerobot itself installed, but one of the packages its dataset
stack needs absent - that named a package the caller already had:

```python
# lerobot 0.6.1 installed; `datasets` is not
sim.start_recording(repo_id="user/ds", task="pick")
# status=error, "requires the lerobot extra: pip install 'strands-robots[lerobot]'"
# -> `pip show lerobot` says it is installed, so the advice reads as a library bug
```

The reason is now surfaced. A new `lerobot_dataset_import_error()` returns
`None` when `LeRobotDataset` imports and otherwise the diagnosis, and
`has_lerobot_dataset()` is a thin predicate over it so the two cannot disagree.
Four causes get four different instructions: lerobot absent (install the extra);
lerobot present but a dataset dependency absent (`pip install 'lerobot[dataset]'`
- plain `pip install lerobot` does not pull those in); lerobot present but no
longer exporting the module strands-robots imports (install an in-range
lerobot); and an import that failed with nothing missing, typically a
pandas/numpy binary conflict (no install fixes it). The same distinction was
already drawn for the policy-factory import by
`strands_robots.policies.lerobot_local.molmoact2._factory_import_error`.

The version string those messages quote comes from a new public
`utils.lerobot_version()`, promoted out of the streaming-dataset reader so the
two modules that name it cannot report it differently. Its handler catches
`ImportError` alone: `PackageNotFoundError` subclasses `ModuleNotFoundError` and
so is already one, and naming it as well bound it as a local the failing import
never reaches - which raised `UnboundLocalError` out of a best-effort helper if
`importlib.metadata` could not be imported.

The three backends each keep their own plain-MP4 fallback advice and report the
reason verbatim. A successful probe is still cached and a failed one still
re-attempted, and a healthy install records exactly as before.
