### Added: `sagemaker` trainer provider - submit a `TrainSpec` as a managed training job

`create_trainer("sagemaker")` submits the same `TrainSpec` the local providers
consume as one managed SageMaker training job wrapping a containerized trainer
image: `dataset_root` (an `s3://` URI) becomes the `training` input channel,
`output_dir` the job's `S3OutputPath`, and the remaining spec fields plus
`extra` travel as string hyperparameters the container decodes back into a
`TrainSpec`. `validate()` runs locally before submission (a bad spec never
pays job-start latency), a failed job is an error `TrainResult` naming the job
and its `FailureReason` - never a silent success - and `status(job_name)`
polls a detached job. Gated behind the new `[sagemaker]` extra (boto3 only);
zero import cost otherwise.
