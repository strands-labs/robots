### Docs: notebook 12 runs the streaming data loop from the Storage Buckets article

`notebooks/12_streaming_data_loop.ipynb` records a LeRobot v3 dataset with `run(record=...)`, syncs it into a
Hugging Face Storage Bucket with `hf sync` when `HF_BUCKET` is set, streams it back with LeRobot's
`StreamingLeRobotDataset`, trains with `lerobot-train` from the stream, and loads the checkpoint as a `Policy`.
Without `HF_BUCKET` it trains from the local folder, so it runs with no credentials.
