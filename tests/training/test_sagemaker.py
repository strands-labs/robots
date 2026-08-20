"""Unit tests for the SageMaker trainer transport (stubbed client, no AWS).

The provider under test is transport-only: the acceptance surface is the
``TrainSpec -> CreateTrainingJob`` mapping, the validate-before-submit
contract (a bad spec never touches the API), and error propagation (a failed
job is an error ``TrainResult`` naming the job - never a silent success).
boto3 is faked through ``sys.modules`` (the same pattern the other provider
tests use for their backends), with the ``require_optional`` cache replaced
per-test so a previously imported real boto3 cannot leak in.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
import types
from typing import Any

import pytest

import strands_robots.utils as utils_module
from strands_robots.training import TrainSpec, create_trainer, list_trainers
from strands_robots.training.factory import import_trainer_class
from strands_robots.training.sagemaker import (
    _FORWARDED_FIELDS,
    SagemakerTrainer,
    build_hyperparameters,
    hyperparameter_problems,
)


def _trainer(**overrides: Any) -> SagemakerTrainer:
    """A fully configured trainer with a fast poll loop; overrides win."""
    kwargs: dict[str, Any] = {
        "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/strands-trainer:latest",
        "role_arn": "arn:aws:iam::123456789012:role/StrandsSageMakerTraining",
        "instance_type": "ml.g5.xlarge",
        "poll_interval_s": 0.001,
    }
    kwargs.update(overrides)
    return SagemakerTrainer(**kwargs)


@pytest.fixture
def spec() -> TrainSpec:
    """A launchable spec: both URI fields on S3, a base model, some extras."""
    return TrainSpec(
        dataset_root="s3://my-bucket/datasets/pick",
        base_model="lerobot/smolvla_base",
        output_dir="s3://my-bucket/checkpoints/pick",
        steps=2000,
        global_batch_size=16,
        seed=7,
        extra={"policy_type": "act"},
    )


class FakeSageMakerClient:
    """Records ``create_training_job`` calls; serves a scripted describe feed."""

    def __init__(self, describe_responses: list[dict] | None = None) -> None:
        self.create_calls: list[dict] = []
        self.describe_calls: list[str] = []
        self.describe_responses = list(describe_responses or [])
        self.create_error: Exception | None = None
        self.describe_error: Exception | None = None

    def create_training_job(self, **request: object) -> dict:
        if self.create_error is not None:
            raise self.create_error
        self.create_calls.append(request)
        return {"TrainingJobArn": f"arn:aws:sagemaker:::training-job/{request['TrainingJobName']}"}

    def describe_training_job(self, TrainingJobName: str) -> dict:  # noqa: N803 - AWS wire casing
        if self.describe_error is not None:
            raise self.describe_error
        self.describe_calls.append(TrainingJobName)
        if len(self.describe_responses) > 1:
            return self.describe_responses.pop(0)
        return self.describe_responses[0]


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeSageMakerClient:
    """Install a fake ``boto3`` whose ``client('sagemaker')`` is recorded.

    The ``require_optional`` module cache is swapped for an empty dict (and
    restored by monkeypatch) so neither a real boto3 cached by an earlier test
    nor this fake can leak across tests.
    """
    client = FakeSageMakerClient(
        describe_responses=[
            {
                "TrainingJobStatus": "Completed",
                "SecondaryStatus": "Completed",
                "ModelArtifacts": {"S3ModelArtifacts": "s3://my-bucket/checkpoints/pick/job/output/model.tar.gz"},
                "BillableTimeInSeconds": 321,
            }
        ]
    )
    fake_boto3 = types.ModuleType("boto3")

    def _client_factory(service: str, **kwargs: object) -> FakeSageMakerClient:
        assert service == "sagemaker"
        client.client_kwargs = kwargs  # type: ignore[attr-defined]
        return client

    fake_boto3.client = _client_factory  # type: ignore[attr-defined]
    monkeypatch.setattr(utils_module, "_lazy_modules", {})
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return client


class TestFactoryWiring:
    def test_create_trainer_resolves_sagemaker(self) -> None:
        """The registered loader and auto-discovery both resolve the provider."""
        trainer = create_trainer("sagemaker")
        assert isinstance(trainer, SagemakerTrainer)
        assert trainer.provider_name == "sagemaker"
        assert import_trainer_class("sagemaker") is SagemakerTrainer

    def test_listed_among_trainers(self) -> None:
        """Registration is what makes the transport discoverable."""
        assert "sagemaker" in list_trainers()

    def test_hardware_floor_is_local_and_zero(self) -> None:
        """The submitting host needs no GPU; the managed job brings its own."""
        floor = create_trainer("sagemaker").hardware_floor
        assert floor == {"min_gpus": 0, "min_vram_gb": 0, "multinode": True}

    def test_constructor_tolerates_unknown_kwargs(self) -> None:
        """The factory forwards arbitrary kwargs; unknown ones are ignored."""
        trainer = create_trainer("sagemaker", some_future_knob=1)
        assert isinstance(trainer, SagemakerTrainer)


class TestValidate:
    def test_launchable_spec_has_no_problems(self, spec: TrainSpec) -> None:
        assert _trainer().validate(spec) == []

    def test_validate_needs_no_boto3(self, spec: TrainSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate is local: it must work with the extra not installed."""
        monkeypatch.setattr(utils_module, "_lazy_modules", {})
        monkeypatch.setitem(sys.modules, "boto3", None)
        assert _trainer().validate(spec) == []

    @pytest.mark.parametrize(
        ("mutate", "expected_fragment"),
        [
            (lambda s: setattr(s, "dataset_root", ""), "dataset_root is required"),
            (lambda s: setattr(s, "dataset_root", "/tmp/local/dataset"), "must be an s3:// URI"),
            (lambda s: setattr(s, "output_dir", ""), "output_dir is required"),
            (lambda s: setattr(s, "output_dir", "/tmp/local/out"), "must be an s3:// URI"),
            (lambda s: setattr(s, "steps", 0), "steps"),
            (lambda s: setattr(s, "learning_rate", 0.0), "learning_rate"),
            (lambda s: setattr(s, "num_nodes", 0), "num_nodes"),
            (lambda s: setattr(s, "seed", -1), "seed"),
            (lambda s: s.extra.__setitem__("steps", "9"), "collides with the forwarded TrainSpec field"),
            (lambda s: s.extra.__setitem__("output_dir", "s3://x/y"), "collides with the forwarded TrainSpec field"),
            (lambda s: s.extra.__setitem__("payload", object()), "not JSON-encodable"),
            (lambda s: s.extra.__setitem__("payload", "x" * 2501), "caps a hyperparameter value"),
            (lambda s: s.extra.__setitem__("k" * 300, "1"), "caps a hyperparameter key"),
            (lambda s: s.extra.__setitem__("--evil", "1"), "extra key"),
        ],
    )
    def test_spec_problems(self, spec: TrainSpec, mutate, expected_fragment: str) -> None:
        mutate(spec)
        problems = _trainer().validate(spec)
        assert problems, f"expected a problem containing {expected_fragment!r}"
        assert any(expected_fragment in p for p in problems), problems

    @pytest.mark.parametrize(
        ("overrides", "expected_fragment"),
        [
            ({"image_uri": None}, "image_uri is required"),
            ({"role_arn": None}, "role_arn is required"),
            ({"role_arn": "not-an-arn"}, "does not look like an IAM role ARN"),
            ({"instance_type": "g5.xlarge"}, "does not look like a SageMaker"),
            ({"base_job_name": "-bad-"}, "base_job_name"),
            ({"base_job_name": "x" * 40}, "base_job_name"),
            # AWS states TrainingJobName as a pattern PLUS a separate 63-char
            # max length; the pattern alone caps only the alphanumeric count
            # (`-*` is unbounded), so a hyphen-padded name used to pass
            # validate and generate a TrainingJobName over 63, refused by the
            # API only after submission.
            ({"base_job_name": "a-" * 25 + "b"}, "base_job_name"),
            ({"base_job_name": "a" + "-" * 100 + "b"}, "base_job_name"),
            ({"volume_size_gb": 0}, "volume_size_gb"),
            ({"max_runtime_s": 0}, "max_runtime_s"),
            ({"poll_interval_s": 0.0}, "poll_interval_s"),
        ],
    )
    def test_transport_config_problems(self, spec: TrainSpec, overrides: dict, expected_fragment: str) -> None:
        problems = _trainer(**overrides).validate(spec)
        assert any(expected_fragment in p for p in problems), problems

    def test_too_many_hyperparameters_is_a_problem(self, spec: TrainSpec) -> None:
        spec.extra = {f"knob_{i}": i for i in range(101)}
        problems = hyperparameter_problems(spec)
        assert any("caps a training job" in p for p in problems), problems

    @pytest.mark.parametrize("base_job_name", ["a" * 32, "a-b-c", "strands-robots"])
    def test_max_length_base_job_name_still_fits_the_api_cap(self, spec: TrainSpec, base_job_name: str) -> None:
        """Control for the length bound: 32 chars is accepted, and the generated
        TrainingJobName (base + 25-char stamp/entropy suffix) stays within AWS's
        63-character cap."""
        trainer = _trainer(base_job_name=base_job_name)
        assert trainer.validate(spec) == []
        assert len(trainer._job_name()) <= 63


class TestHyperparameterMapping:
    def test_every_trainspec_field_is_forwarded_or_deliberately_special(self) -> None:
        """Drift guard: pins ``_FORWARDED_FIELDS`` to ``TrainSpec`` in both directions.

        A field added to ``TrainSpec`` and not forwarded would be silently
        dropped by this provider - the container never sees it, ``validate``
        reports nothing, and the job trains with a setting the caller did not
        ask for while reporting success. A name removed from ``TrainSpec``
        while ``_FORWARDED_FIELDS`` keeps it would make ``_forwarded_items``'s
        ``getattr`` raise out of a pure mapping function. Equality catches both.
        """
        # channel, artifact path, spliced in beside the forwarded fields
        special = {"dataset_root", "output_dir", "extra"}
        fields = {f.name for f in dataclasses.fields(TrainSpec)}
        assert fields - special == set(_FORWARDED_FIELDS)

    def test_values_are_all_strings(self, spec: TrainSpec) -> None:
        """SageMaker hyperparameters are str -> str; nothing else may leak in."""
        hp = build_hyperparameters(spec)
        assert hp and all(isinstance(k, str) and isinstance(v, str) for k, v in hp.items()), hp

    def test_core_fields_and_extra_are_forwarded(self, spec: TrainSpec) -> None:
        hp = build_hyperparameters(spec)
        assert hp["base_model"] == "lerobot/smolvla_base"
        assert hp["steps"] == "2000"
        assert hp["global_batch_size"] == "16"
        assert hp["seed"] == "7"
        assert hp["policy_type"] == "act"

    def test_none_and_empty_fields_are_omitted(self, spec: TrainSpec) -> None:
        """None / "" / {} mean "keep the wrapped backend's default"."""
        hp = build_hyperparameters(spec)
        assert "learning_rate" not in hp  # None
        assert "embodiment" not in hp  # ""
        assert "tune" not in hp  # {}
        assert "dataset_root" not in hp  # travels as the input channel
        assert "output_dir" not in hp  # travels as the artifact path

    def test_non_string_values_decode_with_json(self, spec: TrainSpec) -> None:
        """One documented decoder for the container side: json.loads."""
        spec.augmentation = {"brightness": 0.2}
        spec.streaming = True
        hp = build_hyperparameters(spec)
        assert json.loads(hp["augmentation"]) == {"brightness": 0.2}
        assert json.loads(hp["streaming"]) is True


class TestBuildTrainingJobRequest:
    def test_spec_to_job_mapping(self, spec: TrainSpec) -> None:
        trainer = _trainer(volume_size_gb=50, max_runtime_s=3600)
        spec.num_nodes = 2
        request = trainer.build_training_job_request(spec, "job-1")

        assert request["TrainingJobName"] == "job-1"
        assert request["AlgorithmSpecification"] == {
            "TrainingImage": trainer.image_uri,
            "TrainingInputMode": "File",
        }
        assert request["RoleArn"] == trainer.role_arn
        (channel,) = request["InputDataConfig"]
        assert channel["ChannelName"] == "training"
        assert channel["DataSource"]["S3DataSource"]["S3Uri"] == spec.dataset_root
        assert request["OutputDataConfig"] == {"S3OutputPath": spec.output_dir}
        assert request["ResourceConfig"] == {
            "InstanceType": "ml.g5.xlarge",
            "InstanceCount": 2,
            "VolumeSizeInGB": 50,
        }
        assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 3600}
        assert request["HyperParameters"] == build_hyperparameters(spec)


class TestTrain:
    def test_invalid_spec_never_touches_the_api(self, spec: TrainSpec, fake_client: FakeSageMakerClient) -> None:
        """Validate-before-submit: no job-start latency is paid for a bad spec."""
        spec.dataset_root = "/tmp/local/dataset"
        result = _trainer().train(spec)
        assert result.status == "error"
        assert "validation failed" in result.message
        assert fake_client.create_calls == []
        assert fake_client.describe_calls == []

    def test_completed_job_reports_the_artifact(self, spec: TrainSpec, fake_client: FakeSageMakerClient) -> None:
        result = _trainer().train(spec)
        assert result.status == "success"
        assert result.job_id.startswith("strands-robots-")
        assert result.checkpoint_dir == "s3://my-bucket/checkpoints/pick/job/output/model.tar.gz"
        assert result.metrics["billable_time_s"] == 321
        (request,) = fake_client.create_calls
        assert request["TrainingJobName"] == result.job_id

    def test_polls_until_terminal(self, spec: TrainSpec, fake_client: FakeSageMakerClient) -> None:
        fake_client.describe_responses = [
            {"TrainingJobStatus": "InProgress", "SecondaryStatus": "Downloading"},
            {"TrainingJobStatus": "InProgress", "SecondaryStatus": "Training"},
            {
                "TrainingJobStatus": "Completed",
                "ModelArtifacts": {"S3ModelArtifacts": "s3://my-bucket/out/model.tar.gz"},
            },
        ]
        result = _trainer().train(spec)
        assert result.status == "success"
        assert len(fake_client.describe_calls) == 3

    def test_failed_job_is_an_error_naming_the_job(self, spec: TrainSpec, fake_client: FakeSageMakerClient) -> None:
        """Acceptance: a failed job is never a silent success."""
        fake_client.describe_responses = [
            {"TrainingJobStatus": "Failed", "FailureReason": "AlgorithmError: exit code 1"}
        ]
        result = _trainer().train(spec)
        assert result.status == "error"
        assert result.job_id in result.message
        assert "AlgorithmError: exit code 1" in result.message
        assert result.checkpoint_dir is None

    def test_stopped_job_is_an_error_naming_the_job(self, spec: TrainSpec, fake_client: FakeSageMakerClient) -> None:
        fake_client.describe_responses = [{"TrainingJobStatus": "Stopped"}]
        result = _trainer().train(spec)
        assert result.status == "error"
        assert result.job_id in result.message

    def test_submission_failure_is_an_error_result(self, spec: TrainSpec, fake_client: FakeSageMakerClient) -> None:
        fake_client.create_error = RuntimeError("AccessDeniedException: not authorized")
        result = _trainer().train(spec)
        assert result.status == "error"
        assert "submission failed" in result.message
        assert "AccessDeniedException" in result.message
        assert result.job_id in result.message

    def test_polling_failure_is_an_error_result(self, spec: TrainSpec, fake_client: FakeSageMakerClient) -> None:
        fake_client.describe_error = RuntimeError("throttled")
        result = _trainer().train(spec)
        assert result.status == "error"
        assert "DescribeTrainingJob failed" in result.message
        assert result.job_id in result.message

    def test_missing_boto3_reports_the_install_hint(self, spec: TrainSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        """The extra gate: no boto3 -> an error result naming the extra."""
        monkeypatch.setattr(utils_module, "_lazy_modules", {})
        monkeypatch.setitem(sys.modules, "boto3", None)
        result = _trainer().train(spec)
        assert result.status == "error"
        assert "strands-robots[sagemaker]" in result.message

    def test_exhausted_poll_budget_reports_running_not_success(
        self, spec: TrainSpec, fake_client: FakeSageMakerClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job outliving the local budget is 'running' (pollable), never a verdict."""
        fake_client.describe_responses = [{"TrainingJobStatus": "InProgress"}]
        real_monotonic = time.monotonic
        state = {"calls": 0}

        def stepped_monotonic() -> float:
            state["calls"] += 1
            return real_monotonic() if state["calls"] == 1 else real_monotonic() + 10.0**9

        monkeypatch.setattr("strands_robots.training.sagemaker.time.monotonic", stepped_monotonic)
        result = _trainer().train(spec)
        assert result.status == "running"
        assert result.job_id in result.message
        assert "status(" in result.message


class TestStatus:
    def test_in_progress_maps_to_running(self, fake_client: FakeSageMakerClient) -> None:
        fake_client.describe_responses = [{"TrainingJobStatus": "InProgress", "SecondaryStatus": "Training"}]
        result = _trainer().status("job-7")
        assert result.status == "running"
        assert result.job_id == "job-7"
        assert result.metrics["sagemaker_status"] == "InProgress"
        assert result.metrics["secondary_status"] == "Training"

    def test_terminal_status_maps_like_train(self, fake_client: FakeSageMakerClient) -> None:
        result = _trainer().status("job-7")
        assert result.status == "success"
        assert result.checkpoint_dir == "s3://my-bucket/checkpoints/pick/job/output/model.tar.gz"

    def test_describe_failure_is_an_error_result(self, fake_client: FakeSageMakerClient) -> None:
        fake_client.describe_error = RuntimeError("ValidationException: no such job")
        result = _trainer().status("job-7")
        assert result.status == "error"
        assert "job-7" in result.message

    def test_missing_boto3_reports_the_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils_module, "_lazy_modules", {})
        monkeypatch.setitem(sys.modules, "boto3", None)
        result = _trainer().status("job-7")
        assert result.status == "error"
        assert "strands-robots[sagemaker]" in result.message
