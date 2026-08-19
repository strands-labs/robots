"""SageMaker trainer - submit a :class:`TrainSpec` as a managed AWS training job.

Transport-only provider: unlike the local backends
(:class:`~strands_robots.training.lerobot.LerobotTrainer`,
:class:`~strands_robots.training.groot.Gr00tTrainer`,
:class:`~strands_robots.training.cosmos3.Cosmos3Trainer`), which import a
training library and drive it in-process, this trainer submits the SAME
:class:`~strands_robots.training.base.TrainSpec` to Amazon SageMaker as one
managed training job and waits for its terminal verdict. It reimplements no
training logic: the behavior lives in the caller-supplied container image
(``image_uri``), which packages one of the local trainer paths; this provider
is the ``TrainSpec -> CreateTrainingJob`` mapping and nothing else.

The job contract (standard SageMaker training-container layout):

* ``spec.dataset_root`` MUST be an ``s3://`` URI; it becomes the ``training``
  input channel, which SageMaker stages at ``/opt/ml/input/data/training``
  inside the container.
* ``spec.output_dir`` MUST be an ``s3://`` URI; it becomes the job's
  ``S3OutputPath``. The container writes its checkpoint to ``/opt/ml/model``
  and SageMaker uploads it as ``<output_dir>/<job name>/output/model.tar.gz``,
  which is what :attr:`TrainResult.checkpoint_dir` reports on success.
* Every other spec field the container's trainer consumes travels as a string
  hyperparameter (SageMaker hyperparameters are ``str -> str``), materialized
  in ``/opt/ml/input/config/hyperparameters.json``: string fields verbatim,
  everything else JSON-encoded (see :func:`build_hyperparameters`). The
  container entry point rebuilds a :class:`TrainSpec` from them with
  ``dataset_root=/opt/ml/input/data/training`` and ``output_dir=/opt/ml/model``
  and runs the wrapped local trainer.

``validate`` runs locally BEFORE submission - the same fail-fast contract as
the local providers - so a bad spec is refused without paying job-start
latency. ``boto3`` is required only at :meth:`SagemakerTrainer.train` /
:meth:`SagemakerTrainer.status` time, gated behind the ``[sagemaker]`` extra
via :func:`~strands_robots.utils.require_optional`; importing this module (and
``validate`` itself) costs nothing.

Required IAM surface (documented in README > Training providers): the caller
needs ``sagemaker:CreateTrainingJob``, ``sagemaker:DescribeTrainingJob`` and
``iam:PassRole`` on the execution role; the execution role itself needs S3
read on the dataset prefix, S3 write on the output prefix, ECR pull on the
image, and CloudWatch Logs write.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec
from strands_robots.utils import (
    positive_count_error,
    positive_finite_number_error,
    require_optional,
)

logger = logging.getLogger(__name__)

_PURPOSE = "submitting SageMaker training jobs (strands_robots.training.sagemaker)"

# TrainingJobName must match ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$ AND be at most
# 63 characters - AWS states the pattern and the max length as two separate
# constraints, and the pattern alone caps only the alphanumeric count (the
# inner ``-*`` is unbounded, so a hyphen-padded name of any length matches).
# The lookahead carries the length half. The base job name keeps 32 chars so
# the appended UTC stamp + entropy suffix ("-YYYYmmdd-HHMMSS-<8 hex>",
# 25 chars) always fits under 63.
_BASE_JOB_NAME_RE = re.compile(r"^(?=.{1,32}$)[a-zA-Z0-9](-*[a-zA-Z0-9]){0,31}$")

# SageMaker training instance types are "ml.<family>.<size>", e.g.
# "ml.g5.12xlarge" / "ml.p4d.24xlarge". Format allowlist only - the live
# catalog changes too often to enumerate, and a wrong-but-well-formed type is
# refused by CreateTrainingJob itself.
_INSTANCE_TYPE_RE = re.compile(r"^ml\.[a-z0-9-]+\.[a-z0-9]+$")

_ROLE_ARN_RE = re.compile(r"^arn:aws[a-zA-Z-]*:iam::\d{12}:role/.+$")

# An input channel / output path is an S3 URI: bucket (S3 naming rules,
# loosely) plus optional key prefix. A local path cannot be mounted into a
# managed job, so it is refused here rather than failing at job start.
_S3_URI_RE = re.compile(r"^s3://[a-z0-9][a-z0-9.\-]*[a-z0-9](/.*)?$")

# CreateTrainingJob limits: at most 100 hyperparameters, each key at most 256
# characters and each value at most 2500. Checked in validate() so an
# oversized ``extra`` is refused before submission rather than by the API.
_MAX_HYPERPARAMETERS = 100
_MAX_HYPERPARAMETER_KEY_LEN = 256
_MAX_HYPERPARAMETER_VALUE_LEN = 2500

# TrainSpec fields forwarded to the container as hyperparameters, under their
# spec names. ``dataset_root`` / ``output_dir`` travel as the input channel and
# the artifact path instead, and ``extra`` is spliced in beside these - so an
# ``extra`` key colliding with one of them would silently shadow the spec field
# and is refused in validate().
_FORWARDED_FIELDS: tuple[str, ...] = (
    "base_model",
    "dataset_repo_id",
    "embodiment",
    "steps",
    "global_batch_size",
    "learning_rate",
    "save_freq",
    "num_gpus",
    "num_nodes",
    "resume",
    "seed",
    "method",
    "lora_r",
    "lora_alpha",
    "lora_target_modules",
    "tune",
    "val_episodes",
    "augmentation",
    "fps",
    "streaming",
)

# TrainingJobStatus values that end the poll loop, and their TrainResult
# reading. "Stopping" still resolves to a terminal Stopped, so it keeps polling.
_TERMINAL_STATUSES = frozenset({"Completed", "Failed", "Stopped"})

# Margin added to the local poll deadline beyond the job's own
# MaxRuntimeInSeconds: queueing + instance provisioning + artifact upload
# happen outside the billed runtime the stopping condition caps.
_POLL_GRACE_S = 30 * 60


def _encode_hyperparameter(value: Any) -> str:
    """Encode one spec value as a SageMaker hyperparameter string.

    Strings travel verbatim; everything else is JSON-encoded so the container
    side has one documented decoder (``json.loads``) for every non-string
    field. Raises ``TypeError`` for a value JSON cannot represent - callers on
    the validate path convert that into a reported problem.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _forwarded_items(spec: TrainSpec) -> list[tuple[str, Any]]:
    """Return the (field, value) pairs of :data:`_FORWARDED_FIELDS` worth sending.

    ``None`` means "omit and keep the wrapped backend's default" (the
    :class:`TrainSpec` sentinel convention), and an empty string / empty dict
    carries the same information as absence, so all three are skipped rather
    than serialized.
    """
    items: list[tuple[str, Any]] = []
    for field in _FORWARDED_FIELDS:
        value = getattr(spec, field)
        if value is None or value == "" or value == {}:
            continue
        items.append((field, value))
    return items


def build_hyperparameters(spec: TrainSpec) -> dict[str, str]:
    """Map a :class:`TrainSpec` onto SageMaker's ``str -> str`` hyperparameters.

    Pure and deterministic - the testable half of the job mapping. Core fields
    (see :data:`_FORWARDED_FIELDS`) are forwarded under their spec names, then
    ``spec.extra`` is spliced in beside them; :meth:`SagemakerTrainer.validate`
    has already refused a colliding or unencodable key by the time this runs on
    the submission path.
    """
    hyperparameters = {field: _encode_hyperparameter(value) for field, value in _forwarded_items(spec)}
    for key, value in (spec.extra or {}).items():
        hyperparameters[str(key)] = _encode_hyperparameter(value)
    return hyperparameters


def hyperparameter_problems(spec: TrainSpec) -> list[str]:
    """Return the hyperparameter-mapping problems for a :class:`TrainSpec`.

    Everything :func:`build_hyperparameters` would produce is checked here
    first, so the refusal happens in ``validate`` rather than at the API or -
    worse - silently: an ``extra`` key that collides with a forwarded spec
    field would otherwise shadow it, an unencodable value would raise from a
    method documented to return problems, and an oversized key, value or count
    would be refused by CreateTrainingJob only after submission.
    """
    problems: list[str] = []
    extra = spec.extra or {}

    for key in extra:
        if key in _FORWARDED_FIELDS or key in ("dataset_root", "output_dir"):
            problems.append(
                f"extra key {key!r} collides with the forwarded TrainSpec field of the same name - "
                f"set spec.{key} instead"
            )

    candidates = _forwarded_items(spec) + [(str(k), v) for k, v in extra.items()]
    for key, value in candidates:
        if len(key) > _MAX_HYPERPARAMETER_KEY_LEN:
            problems.append(
                f"hyperparameter key {key!r} is {len(key)} characters long "
                f"(SageMaker caps a hyperparameter key at {_MAX_HYPERPARAMETER_KEY_LEN})"
            )
        try:
            encoded = _encode_hyperparameter(value)
        except (TypeError, ValueError):
            problems.append(f"hyperparameter {key!r} is not JSON-encodable: {value!r}")
            continue
        if len(encoded) > _MAX_HYPERPARAMETER_VALUE_LEN:
            problems.append(
                f"hyperparameter {key!r} is {len(encoded)} characters long "
                f"(SageMaker caps a hyperparameter value at {_MAX_HYPERPARAMETER_VALUE_LEN})"
            )
    if len(candidates) > _MAX_HYPERPARAMETERS:
        problems.append(
            f"spec maps to {len(candidates)} hyperparameters (SageMaker caps a training job at {_MAX_HYPERPARAMETERS})"
        )
    return problems


class SagemakerTrainer(Trainer):
    """Submit a :class:`TrainSpec` as one managed SageMaker training job.

    Transport, not behavior: the training logic lives in the containerized
    trainer image this provider submits. See the module docstring for the job
    contract and the required IAM surface.

    Args:
        image_uri: ECR URI of the training container (packages one of the
            local trainer paths). Required for ``train``; its absence is a
            ``validate`` problem rather than a constructor error.
        role_arn: SageMaker execution role ARN the job runs as. Required.
        instance_type: SageMaker training instance type (``ml.<family>.<size>``).
            ``spec.num_gpus`` must agree with the GPUs this type carries: it
            travels to the container as a hyperparameter and is NOT
            cross-checked against the instance catalog here, so a mismatch
            fails inside the container only after the instance is provisioned
            (which bills).
        region: AWS region for the SageMaker client. ``None`` defers to boto3's
            standard resolution chain (env / config / instance profile).
        volume_size_gb: EBS volume attached to each training instance.
        max_runtime_s: Job ``StoppingCondition.MaxRuntimeInSeconds``; the local
            poll loop allows this budget plus a provisioning/upload margin.
        poll_interval_s: Seconds between ``DescribeTrainingJob`` polls.
        base_job_name: Prefix of the generated ``TrainingJobName`` (a UTC
            stamp + entropy suffix is appended per submission).
        **kwargs: Tolerated and ignored (factory forwards arbitrary kwargs).
    """

    def __init__(
        self,
        image_uri: str | None = None,
        role_arn: str | None = None,
        instance_type: str = "ml.g5.xlarge",
        region: str | None = None,
        volume_size_gb: int = 100,
        max_runtime_s: int = 24 * 3600,
        poll_interval_s: float = 30.0,
        base_job_name: str = "strands-robots",
        **kwargs: Any,
    ) -> None:
        self.image_uri = image_uri
        self.role_arn = role_arn
        self.instance_type = instance_type
        self.region = region
        self.volume_size_gb = volume_size_gb
        self.max_runtime_s = max_runtime_s
        self.poll_interval_s = poll_interval_s
        self.base_job_name = base_job_name

    @property
    def provider_name(self) -> str:
        """Provider identity - the managed SageMaker training-job transport."""
        return "sagemaker"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        """The LOCAL host needs no GPU - the managed job brings its own.

        ``min_gpus`` / ``min_vram_gb`` describe the machine ``train`` runs on,
        which for this provider only submits and polls; ``multinode`` is True
        because ``num_nodes`` maps directly onto the job's ``InstanceCount``.
        """
        return {"min_gpus": 0, "min_vram_gb": 0, "multinode": True}

    def validate(self, spec: TrainSpec) -> list[str]:
        """Local, pure preflight - refuse a bad spec before job-start latency.

        Checks the transport configuration (image, role, instance type, job
        name, poll budget), the S3-ness of the two URI fields, the shared
        numeric domains of every spec field this provider serializes, and the
        hyperparameter mapping itself (collisions, encodability, API limits).
        No boto3 import and no network call - ``validate`` works without the
        ``[sagemaker]`` extra installed.
        """
        problems = self._security_problems(spec)

        # --- transport configuration (constructor-supplied) ---
        if not self.image_uri:
            problems.append("image_uri is required (ECR image of the containerized trainer)")
        if not self.role_arn:
            problems.append("role_arn is required (SageMaker execution role the job runs as)")
        elif not _ROLE_ARN_RE.match(self.role_arn):
            problems.append(f"role_arn does not look like an IAM role ARN: {self.role_arn!r}")
        if not _INSTANCE_TYPE_RE.match(self.instance_type):
            problems.append(
                f"instance_type {self.instance_type!r} does not look like a SageMaker "
                "training instance type (expected 'ml.<family>.<size>', e.g. 'ml.g5.xlarge')"
            )
        if not _BASE_JOB_NAME_RE.match(self.base_job_name):
            problems.append(
                f"base_job_name {self.base_job_name!r} must be 1-32 alphanumeric-or-hyphen "
                "characters starting and ending alphanumeric (it prefixes the TrainingJobName)"
            )
        for count_param, count_value in (
            ("volume_size_gb", self.volume_size_gb),
            ("max_runtime_s", self.max_runtime_s),
        ):
            error = positive_count_error(count_value, count_param, self.provider_name)
            if error is not None:
                problems.append(error)
        error = positive_finite_number_error(self.poll_interval_s, "poll_interval_s", self.provider_name)
        if error is not None:
            problems.append(error)

        # --- the two URI fields that become the channel and the artifact path ---
        for uri_param, uri_value in (("dataset_root", spec.dataset_root), ("output_dir", spec.output_dir)):
            if not uri_value:
                problems.append(
                    f"{uri_param} is required and must be an s3:// URI (a managed job cannot mount a local path)"
                )
            elif not _S3_URI_RE.match(uri_value):
                problems.append(f"{uri_param} must be an s3:// URI for a managed job, got {uri_value!r}")

        # --- shared numeric domains of every field this provider serializes ---
        problems.extend(self._run_size_problems(spec))
        problems.extend(self._learning_rate_problems(spec))
        problems.extend(self._launch_topology_problems(spec))
        problems.extend(self._seed_problems(spec))
        problems.extend(self._validation_episodes_problems(spec))
        problems.extend(self._lora_hyperparameter_problems(spec))

        # --- the hyperparameter mapping itself ---
        problems.extend(hyperparameter_problems(spec))

        return problems

    def build_training_job_request(self, spec: TrainSpec, job_name: str) -> dict[str, Any]:
        """Build the full ``CreateTrainingJob`` request for a validated spec.

        Pure - the testable whole of the ``TrainSpec -> job`` mapping:
        ``dataset_root`` -> the ``training`` input channel, ``output_dir`` ->
        ``OutputDataConfig.S3OutputPath``, ``num_nodes`` ->
        ``ResourceConfig.InstanceCount``, spec fields + ``extra`` ->
        ``HyperParameters`` (via :func:`build_hyperparameters`).
        """
        return {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": self.image_uri,
                "TrainingInputMode": "File",
            },
            "RoleArn": self.role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": spec.dataset_root,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": spec.output_dir},
            "ResourceConfig": {
                "InstanceType": self.instance_type,
                "InstanceCount": int(spec.num_nodes),
                "VolumeSizeInGB": int(self.volume_size_gb),
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": int(self.max_runtime_s)},
            "HyperParameters": build_hyperparameters(spec),
        }

    def _client(self) -> Any:
        """Build the boto3 SageMaker client (the ``[sagemaker]`` extra gate).

        Raises ``ImportError`` with the install hint when boto3 is missing;
        ``train`` and ``status`` convert that into an error ``TrainResult``.
        """
        boto3: Any = require_optional("boto3", extra="sagemaker", purpose=_PURPOSE)
        client_kwargs: dict[str, Any] = {}
        if self.region:
            client_kwargs["region_name"] = self.region
        return boto3.client("sagemaker", **client_kwargs)

    def _job_name(self) -> str:
        """Generate a unique ``TrainingJobName`` under :attr:`base_job_name`.

        The UTC stamp is a recorded point in time (``time.time`` domain per
        the clock rule); uniqueness comes from the entropy suffix, not the
        stamp, so two submissions in one second cannot collide.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return f"{self.base_job_name}-{stamp}-{uuid.uuid4().hex[:8]}"

    def train(self, spec: TrainSpec) -> TrainResult:
        """Validate locally, submit the job, and block until its terminal verdict.

        Fail-closed on validation (nothing is submitted for a spec with
        problems - the stubbed-client tests pin that the API is never touched).
        A failed or stopped job is an error ``TrainResult`` naming the job and
        carrying SageMaker's ``FailureReason`` - never a silent success. If the
        local poll budget (``max_runtime_s`` + a provisioning margin) runs out
        first, the job is NOT assumed dead: the result is ``running`` with the
        job name, pollable via :meth:`status`.
        """
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error",
                job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        try:
            client = self._client()
        except ImportError as e:
            return TrainResult(status="error", job_id="", message=str(e))

        self.prepare(spec)

        job_name = self._job_name()
        request = self.build_training_job_request(spec, job_name)
        logger.info("Submitting SageMaker training job %s (image=%s)", job_name, self.image_uri)
        try:
            client.create_training_job(**request)
        except Exception as e:  # noqa: BLE001 - convert ANY submission failure to a result
            return TrainResult(
                status="error",
                job_id=job_name,
                message=f"SageMaker training job '{job_name}' submission failed: {e}",
            )

        return self._wait_for_job(client, job_name)

    def _wait_for_job(self, client: Any, job_name: str) -> TrainResult:
        """Poll ``DescribeTrainingJob`` until terminal, on a monotonic budget.

        The deadline is a duration, so it is measured on ``time.monotonic()``
        (a wall-clock step must not end the wait early or stretch it): the
        job's own ``MaxRuntimeInSeconds`` plus :data:`_POLL_GRACE_S` for
        provisioning and artifact upload, which bill outside that runtime.
        """
        deadline_mono = time.monotonic() + float(self.max_runtime_s) + _POLL_GRACE_S
        while True:
            try:
                description = client.describe_training_job(TrainingJobName=job_name)
            except Exception as e:  # noqa: BLE001 - convert ANY polling failure to a result
                return TrainResult(
                    status="error",
                    job_id=job_name,
                    message=f"SageMaker training job '{job_name}': DescribeTrainingJob failed: {e}",
                )
            job_status = str(description.get("TrainingJobStatus", ""))
            if job_status in _TERMINAL_STATUSES:
                return self._result_from_description(job_name, description)
            if time.monotonic() >= deadline_mono:
                return TrainResult(
                    status="running",
                    job_id=job_name,
                    metrics={"sagemaker_status": job_status},
                    message=(
                        f"SageMaker training job '{job_name}' is still {job_status or 'pending'} after the "
                        f"local poll budget ({self.max_runtime_s}s + {_POLL_GRACE_S}s margin) - "
                        "poll it with status(job_name)"
                    ),
                )
            time.sleep(self.poll_interval_s)

    def _result_from_description(self, job_name: str, description: dict[str, Any]) -> TrainResult:
        """Map a terminal ``DescribeTrainingJob`` response onto a ``TrainResult``.

        ``Completed`` -> success with ``checkpoint_dir`` at the S3 model
        artifact; ``Failed`` / ``Stopped`` -> error naming the job and carrying
        SageMaker's ``FailureReason`` - a failed job is never a silent success.
        """
        job_status = str(description.get("TrainingJobStatus", ""))
        artifacts = (description.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
        metrics: dict[str, Any] = {"sagemaker_status": job_status}
        secondary = description.get("SecondaryStatus")
        if secondary:
            metrics["secondary_status"] = secondary
        billable = description.get("BillableTimeInSeconds")
        if billable is not None:
            metrics["billable_time_s"] = billable

        if job_status == "Completed":
            return TrainResult(
                status="success",
                job_id=job_name,
                checkpoint_dir=artifacts,
                metrics=metrics,
                message=f"SageMaker training job '{job_name}' completed (artifact: {artifacts})",
            )

        reason = description.get("FailureReason", "")
        detail = f": {reason}" if reason else ""
        return TrainResult(
            status="error",
            job_id=job_name,
            metrics=metrics,
            message=f"SageMaker training job '{job_name}' {job_status.lower()}{detail}",
        )

    def status(self, job_id: str) -> TrainResult:
        """Poll a submitted job by name - the detached-job case the ABC allows.

        Unlike the local providers, a SageMaker job outlives the submitting
        process, so ``status`` is a real ``DescribeTrainingJob`` read: a
        non-terminal job reports ``running`` with its SageMaker status in
        ``metrics``; a terminal one maps exactly as :meth:`train` does.
        """
        try:
            client = self._client()
        except ImportError as e:
            return TrainResult(status="error", job_id=job_id, message=str(e))
        try:
            description = client.describe_training_job(TrainingJobName=job_id)
        except Exception as e:  # noqa: BLE001 - convert ANY polling failure to a result
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"SageMaker training job '{job_id}': DescribeTrainingJob failed: {e}",
            )
        job_status = str(description.get("TrainingJobStatus", ""))
        if job_status in _TERMINAL_STATUSES:
            return self._result_from_description(job_id, description)
        metrics: dict[str, Any] = {"sagemaker_status": job_status}
        secondary = description.get("SecondaryStatus")
        if secondary:
            metrics["secondary_status"] = secondary
        return TrainResult(
            status="running",
            job_id=job_id,
            metrics=metrics,
            message=f"SageMaker training job '{job_id}' is {job_status or 'pending'}",
        )
