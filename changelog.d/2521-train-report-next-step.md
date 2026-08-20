### Fixed

- **tools/train_policy**: `action="train"` names the step that fits the result it got instead of always naming a checkpoint load. A run with no artifact - a `running` result from the managed-job backend, or a `success` result whose `latest_checkpoint` found nothing - was reported with `Load the result with: create_policy('None')`, which raises `Unknown policy provider: 'None'`. An unfinished run now names the `status` poll for its `job_id`; the tool-level status and the run status in the `{"json": ...}` block are unchanged.
