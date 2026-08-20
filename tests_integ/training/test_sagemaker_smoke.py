"""SageMaker trainer transport smoke - real AWS calls, gated on credentials.

NOT run in CI. Skips cleanly without ``boto3`` or resolvable AWS credentials;
the full submission smoke additionally needs a training image, an execution
role and an S3 prefix, named by three environment variables::

    export STRANDS_SAGEMAKER_SMOKE_IMAGE_URI=<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
    export STRANDS_SAGEMAKER_SMOKE_ROLE_ARN=arn:aws:iam::<account>:role/<role>
    export STRANDS_SAGEMAKER_SMOKE_S3_PREFIX=s3://<bucket>/<prefix>
    pytest tests_integ/training/test_sagemaker_smoke.py -v

With credentials alone, the error-propagation half runs: a ``status`` poll of
a job that does not exist must come back as an error ``TrainResult`` naming
the job (a real ``DescribeTrainingJob`` refusal, proving auth + client wiring
+ the never-silent contract). With the three variables set, ``train`` submits
one real short job and asserts the transport verdict - the job itself may
succeed or fail depending on the image; what is asserted is that the verdict
names the job and is never a silent success.
"""

from __future__ import annotations

import os

import pytest

boto3 = pytest.importorskip("boto3")

from strands_robots.training import TrainSpec, create_trainer  # noqa: E402


def _credentials_available() -> bool:
    """True when boto3's standard chain resolves any credentials."""
    session = boto3.session.Session()
    return session.get_credentials() is not None


pytestmark = pytest.mark.skipif(
    not _credentials_available(),
    reason="no AWS credentials resolvable (boto3 default chain)",
)

_IMAGE_URI = os.environ.get("STRANDS_SAGEMAKER_SMOKE_IMAGE_URI", "")
_ROLE_ARN = os.environ.get("STRANDS_SAGEMAKER_SMOKE_ROLE_ARN", "")
_S3_PREFIX = os.environ.get("STRANDS_SAGEMAKER_SMOKE_S3_PREFIX", "").rstrip("/")


def test_status_of_missing_job_is_an_error_naming_the_job() -> None:
    """Real DescribeTrainingJob refusal -> error result, never a silent verdict."""
    trainer = create_trainer("sagemaker")
    result = trainer.status("strands-robots-smoke-no-such-job")
    assert result.status == "error"
    assert "strands-robots-smoke-no-such-job" in result.message


@pytest.mark.skipif(
    not (_IMAGE_URI and _ROLE_ARN and _S3_PREFIX),
    reason="set STRANDS_SAGEMAKER_SMOKE_IMAGE_URI / _ROLE_ARN / _S3_PREFIX to run the submission smoke",
)
def test_submit_one_short_job() -> None:
    """One spec, one job: submit, poll to terminal, assert the verdict names it."""
    trainer = create_trainer(
        "sagemaker",
        image_uri=_IMAGE_URI,
        role_arn=_ROLE_ARN,
        instance_type=os.environ.get("STRANDS_SAGEMAKER_SMOKE_INSTANCE_TYPE", "ml.m5.large"),
        max_runtime_s=15 * 60,
        base_job_name="strands-smoke",
    )
    spec = TrainSpec(
        dataset_root=f"{_S3_PREFIX}/dataset",
        output_dir=f"{_S3_PREFIX}/output",
        base_model="lerobot/smolvla_base",
        steps=10,
        global_batch_size=1,
    )
    assert trainer.validate(spec) == []

    result = trainer.train(spec)
    assert result.job_id.startswith("strands-smoke-")
    assert result.status in {"success", "error", "running"}
    if result.status == "success":
        assert result.checkpoint_dir and result.checkpoint_dir.startswith("s3://")
    else:
        # Never a silent verdict: the message names the job either way.
        assert result.job_id in result.message
