"""Cosmos-Transfer-style generative backend - video2video episode augmentation.

The first generative backend of the
:class:`~strands_robots.transforms.base.DatasetTransform` surface: each camera
stream of each source episode is handed to a **video2video pipeline** (a
world-model / style-transfer generator) that re-renders the same trajectory
under a different visual distribution, steered by
:attr:`~strands_robots.transforms.base.TransformSpec.prompt`.

Deliberately vendor-neutral. NVIDIA's Cosmos-Transfer family
(``github.com/nvidia-cosmos``) is the namesake and the intended first pipeline,
but its models ship from source under the NVIDIA Open Model License - not from
PyPI - and their availability and licensing must be verified per deployment.
So this adapter does not import any single vendor's package: it binds to any
object satisfying the small pipeline protocol below, supplied either as a
constructed object or as a dotted import path resolved lazily. That is the
same dependency-injection shape :class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy`
uses for its ``client=`` / ``diffusers_backend=`` seams, and it keeps this
module importable (and its refusals testable) on a machine with no generation
model installed.

Pipeline protocol::

    class VideoToVideoPipeline(Protocol):
        def generate(
            self,
            video: np.ndarray,          # (T, H, W, 3) uint8
            prompt: str = "",
            seed: int | None = None,
        ) -> np.ndarray:                 # same shape and dtype
            ...

An optional ``version`` attribute on the pipeline is recorded into each
generated episode's provenance.

Adapting a ``__call__``-shaped pipeline
---------------------------------------

Diffusers-family video pipelines - including the diffusers-hosted Cosmos
Transfer checkpoints - do not expose ``generate``: they expose ``__call__``,
return an output object carrying ``.frames``, and seed via a
``generator=torch.Generator()`` argument rather than ``seed=``. Their
``output_type="np"`` frames are also ``float`` in ``[0, 1]``, where this seam
requires the source's ``(T, H, W, 3) uint8``. A thin adapter closes all three
gaps::

    class Adapter:
        version = "cosmos-transfer-2.5"  # recorded into provenance

        def generate(self, video, prompt="", seed=None):
            out = pipe(  # exact conditioning kwargs vary per pipeline
                video=list(video), prompt=prompt, output_type="np",
                generator=None if seed is None else torch.Generator().manual_seed(seed),
            )
            return np.clip(np.round(np.asarray(out.frames[0]) * 255.0), 0, 255).astype(np.uint8)

The conditioning kwargs are the one per-pipeline part: Cosmos Predict's
video2world pipelines take the source pixels as ``video=``, while Cosmos
Transfer conditions on ``controls=`` derived from them (edge / depth /
segmentation maps), so the adapter is also where that derivation lives.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from strands_robots.transforms.base import DatasetTransform, TransformSpec, derive_variant_seed
from strands_robots.utils import non_negative_whole_number_error

_PIPELINE_HELP = (
    "pass pipeline= a constructed video2video object exposing "
    "generate(video, prompt=..., seed=...) -> ndarray, or a dotted import path "
    "('pkg.mod:attr') naming one (or a zero-arg factory returning one). "
    "A diffusers-family pipeline is __call__-shaped, returns an object "
    "carrying .frames and seeds via generator=, so wrap it:\n"
    "    class Adapter:\n"
    "        def generate(self, video, prompt='', seed=None):\n"
    "            g = None if seed is None else torch.Generator().manual_seed(seed)\n"
    "            f = pipe(video=list(video), prompt=prompt, output_type='np', generator=g).frames[0]\n"
    "            return np.clip(np.round(np.asarray(f) * 255.0), 0, 255).astype(np.uint8)\n"
    "(per-pipeline conditioning kwargs vary; worked example in the "
    "strands_robots.transforms.cosmos_transfer module docstring). "
    "NVIDIA Cosmos-Transfer is the intended first pipeline; its models ship "
    "from source (github.com/nvidia-cosmos) under the NVIDIA Open Model "
    "License - verify availability and licensing for your deployment before "
    "wiring it in."
)

#: Sentinel for "the pipeline exposes no ``generate`` attribute at all", kept
#: distinct from a ``generate`` that is present but ``None`` so the probe below
#: classifies both exactly as ``hasattr`` / ``getattr`` did.
_UNSET: Any = object()


def _generate_surface(obj: Any, subject: str) -> tuple[Any, str | None]:
    """Read ``obj.generate`` for the probe, or report why the read failed.

    The pipeline is caller-supplied, so reading an attribute off it runs the
    caller's code: a lazy handle that initialises a CUDA context on first
    access raises here, and ``getattr``'s default only absorbs
    ``AttributeError``. Reporting keeps this probe inside
    :meth:`~strands_robots.transforms.base.DatasetTransform.validate`'s
    "return problems" contract.

    Args:
        obj: The candidate pipeline - an injected object or a resolved import
            target.
        subject: How the caller named it, for the problem text.

    Returns:
        ``(surface, None)`` where ``surface`` is the attribute or :data:`_UNSET`
        when absent, or ``(_UNSET, problem)`` when the read itself raised.
    """
    try:
        return getattr(obj, "generate", _UNSET), None
    except Exception as exc:  # noqa: BLE001 - never fatal while probing a caller's object
        return _UNSET, (
            f"{subject} raised {type(exc).__name__} while its generate() surface was read ({exc}) - {_PIPELINE_HELP}"
        )


class CosmosTransferTransform(DatasetTransform):
    """Video2video episode augmentation behind a vendor-neutral pipeline seam.

    The transform surface owns the dataset plumbing, pass-through, provenance
    and re-validation gate (see
    :class:`~strands_robots.transforms.base.DatasetTransform`); this class owns
    only the hand-off of pixels to a generation pipeline.
    """

    def __init__(self, pipeline: Any | None = None) -> None:
        """Bind the generation pipeline.

        Args:
            pipeline: One of:

                * a constructed object exposing
                  ``generate(video, prompt=..., seed=...) -> ndarray`` (the
                  pipeline protocol in the module docstring);
                * a dotted import path (``"pkg.mod:attr"`` or
                  ``"pkg.mod.attr"``) naming such an object, or a zero-arg
                  factory returning one - resolved lazily, so constructing
                  this transform never imports a generation stack;
                * ``None`` - the transform constructs but
                  :meth:`validate` reports the missing pipeline, so nothing is
                  read or written without one.
        """
        self._pipeline_spec = pipeline
        self._pipeline: Any | None = pipeline if pipeline is not None and not isinstance(pipeline, str) else None

    @property
    def provider_name(self) -> str:
        """Provider identity - the Cosmos-Transfer-style video2video backend."""
        return "cosmos_transfer"

    @property
    def transform_version(self) -> str:
        """The bound pipeline's ``version`` attribute, or ``"unversioned"``.

        Read from the pipeline (never guessed) so a provenance record pins the
        generator that actually produced the pixels.
        """
        pipeline = self._pipeline
        version = getattr(pipeline, "version", None) if pipeline is not None else None
        return str(version) if version is not None else "unversioned"

    def _pipeline_problems(self) -> list[str]:
        """Resolve the pipeline seam; return problems instead of raising.

        Side effect on success: caches the resolved pipeline object so
        :meth:`transform_frames` and :attr:`transform_version` read the same
        instance validate approved.
        """
        if self._pipeline is not None:
            subject = f"pipeline object {type(self._pipeline).__name__}"
            surface, problem = _generate_surface(self._pipeline, subject)
            if problem:
                return [problem]
            if not callable(surface):
                return [f"{subject} has no callable generate() - {_PIPELINE_HELP}"]
            return []
        if self._pipeline_spec is None:
            return [f"no video2video pipeline is bound - {_PIPELINE_HELP}"]

        # Dotted import path: "pkg.mod:attr" or "pkg.mod.attr".
        path = str(self._pipeline_spec)
        module_name, _, attr_name = path.partition(":") if ":" in path else path.rpartition(".")
        if not module_name or not attr_name:
            return [f"pipeline import path {path!r} is not 'pkg.mod:attr' or 'pkg.mod.attr' - {_PIPELINE_HELP}"]
        subject = f"pipeline import path {path!r}"
        try:
            candidate = getattr(importlib.import_module(module_name), attr_name)
        except Exception as exc:  # noqa: BLE001 - never fatal during name resolution
            # Not only ``ImportError`` / ``AttributeError``: a module body runs
            # arbitrary caller code on first import, and a module-level
            # ``__getattr__`` runs it again on the attribute lookup.
            return [f"{subject} did not resolve ({type(exc).__name__}: {exc}) - {_PIPELINE_HELP}"]
        surface, problem = _generate_surface(candidate, subject)
        if problem:
            return [problem]
        # A class target (or a factory with no generate of its own) is
        # constructed zero-arg; an unconstructed class would otherwise pass the
        # generate() probe and then receive the video as its ``self``.
        if isinstance(candidate, type) or (callable(candidate) and surface is _UNSET):
            try:
                candidate = candidate()
            except TypeError as exc:
                return [f"{subject} is not constructible zero-arg ({exc}) - {_PIPELINE_HELP}"]
            except Exception as exc:  # noqa: BLE001 - never fatal during name resolution
                # Constructing a real generation pipeline loads weights and
                # touches a device, so the failures are the deployment's, not
                # the signature's: no driver (``RuntimeError``), weights absent
                # (``OSError``), an optional dep imported inside the factory
                # body (``ImportError``), a malformed config (``ValueError``).
                # Reported with an accurate subject rather than folded into the
                # zero-arg wording above, which would name the wrong cause.
                return [f"{subject} raised {type(exc).__name__} while being constructed ({exc}) - {_PIPELINE_HELP}"]
            surface, problem = _generate_surface(candidate, subject)
            if problem:
                return [problem]
        if not callable(surface):
            return [f"{subject} resolves to no generate() surface - {_PIPELINE_HELP}"]
        self._pipeline = candidate
        return []

    def validate(self, spec: TransformSpec) -> list[str]:
        """Shared spec preflight plus the pipeline seam resolution."""
        problems = self._spec_problems(spec)
        problems.extend(self._pipeline_problems())
        return problems

    def transform_frames(
        self,
        camera_key: str,
        frames: np.ndarray,
        spec: TransformSpec,
        *,
        source_episode: int,
        variant: int,
    ) -> np.ndarray:
        """Hand one camera stream to the pipeline's ``generate``.

        Args:
            camera_key: Bare camera name (unused by the hand-off; the pipeline
                sees pixels and prompt only).
            frames: Source pixels, ``(T, H, W, 3) uint8``.
            spec: The running spec; its prompt steers the generation and its
                seed feeds the per-variant determinism key.
            source_episode: Source episode index (determinism key input).
            variant: Variant counter (determinism key input).

        Returns:
            Generated pixels, same shape and dtype (the base orchestration
            refuses anything else loudly).

        Raises:
            ValueError: ``source_episode`` or ``variant`` is outside the
                non-negative whole-number domain the determinism key needs
                (see :func:`~strands_robots.transforms.base.derive_variant_seed`).
                Both are refused ahead of the pipeline-binding check, so an
                unusable counter is not reported as a wiring problem.
            RuntimeError: No pipeline is bound. Unreachable through
                :meth:`~strands_robots.transforms.base.DatasetTransform.transform`,
                which validates first; raised for a direct caller so the
                refusal names the remedy instead of surfacing as an
                ``AttributeError`` on ``None``.
        """
        for name, value in (("source_episode", source_episode), ("variant", variant)):
            if text := non_negative_whole_number_error(value, name, "cosmos_transfer.transform_frames"):
                raise ValueError(text)
        if self._pipeline is None and self._pipeline_problems():
            raise RuntimeError(f"cosmos_transfer.transform_frames: no video2video pipeline is bound - {_PIPELINE_HELP}")
        pipeline = self._pipeline
        assert pipeline is not None  # narrowed by the guard above
        generated = pipeline.generate(
            frames,
            prompt=spec.prompt,
            seed=derive_variant_seed(spec.seed, source_episode, variant),
        )
        return np.asarray(generated)
