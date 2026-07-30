### Fixed: notebook 5 streamed from a bucket path it never wrote to

The streaming-data-loop notebook synced its dataset with `sync_to_bucket`, which
uploads to `hf://buckets/{bucket}/{run_id}`, then read it back with
`stream_dataset(BUCKET, repo_type="bucket")`. LeRobot's bucket reader resolves
`meta/` and `data/` directly beneath the repo id it is handed, so the read looked
for `meta/info.json` at the bucket root while the dataset sat one level down
under its `run_id`:

```python
rec.sync_to_bucket(BUCKET)                                  # -> hf://buckets/org/name/nb5_demo
sim.stream_dataset(BUCKET, repo_type="bucket")              # looks in hf://buckets/org/name
# FileNotFoundError: .../buckets--org--name/meta/info.json
```

Only the bucket branch was affected; the default local-root path was always
consistent. The notebook now defines `RUN_ID` alongside `BUCKET`, pins the sync
to it, and builds the streaming repo id as `f"{BUCKET}/{RUN_ID}"` - the bucket
namespace is the first two segments of that id and the remainder is the path
within the bucket, so one id addresses both. A test asserts the write and read
sides name the same `run_id`, since the two are only consistent by convention
and drifted apart silently before.
