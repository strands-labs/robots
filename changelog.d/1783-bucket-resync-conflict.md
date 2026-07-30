### Fixed: re-syncing into an existing bucket failed on the create step

`sync_dataset_to_bucket` and `sync_to_bucket` run `hf buckets create` before
every sync when `create=True`, so the second sync of a run always meets a bucket
that already exists. That is the case buckets exist for - the mutable collection
target you re-sync through the day - and it failed:

```python
sync_dataset_to_bucket(root, "my-org/robot-fave")   # first call: creates, syncs
sync_dataset_to_bucket(root, "my-org/robot-fave")   # second call:
# {"status": "error", "message": "bucket create failed: ... '409 Conflict' ..."}
```

The already-exists branch matched only the substring `exist`, while the hub
reports the conflict as `You already created this bucket repo` with a 409. The
check now matches the status code and both phrasings. A genuine create failure
(403, for instance) still errors without running the sync, so tolerating the
conflict does not report success for a sync that never happened.
