"""Dataset transform abstraction - synthesize episode variants from a recorded dataset.

The :class:`DatasetTransform` ABC is the third provider shape beside
:class:`~strands_robots.policies.base.Policy` (inference) and
:class:`~strands_robots.training.base.Trainer` (post-tuning): where a policy
produces actions and a trainer produces checkpoints, a dataset transform
produces **data** - a LeRobotDataset in, an augmented LeRobotDataset out.

The contract, in the order it protects the training loop:

1. **Video observation streams are transformed; everything else passes through
   byte-identical.** A backend sees only the pixel streams
   (:meth:`DatasetTransform.transform_frames`); the orchestration in
   :meth:`DatasetTransform.transform` copies the action and state columns from
   the source episode unchanged, so a generated episode is the *same
   trajectory* rendered differently, never a different trajectory.
2. **Provenance is mandatory.** Every generated episode is recorded in the
   output dataset's ``meta/provenance.json``
   (:mod:`strands_robots.transforms.provenance`) with ``synthetic=true``, the
   source episode index, and the transform's name and version - so training
   filters and evaluation treat generated pixels honestly. Silent mixing of
   generated and recorded data is the failure mode that file exists to prevent.
3. **Re-validation is the acceptance gate.** When :attr:`TransformSpec.revalidate`
   supplies a deterministic verdict function, every generated episode is scored
   against its source episode's verdict, and a generated episode that flips the
   verdict is discarded and **counted** in
   :attr:`TransformResult.episodes_discarded` - measured, not assumed. The
   gate's entire discriminating power lives in the image columns, because
   guarantee 1 holds every other column byte-identical: a verdict that reads
   no ``observation.images.*`` column cannot flip, so the orchestration
   measures which columns the verdict consulted and reports such a run as
   ungated (:attr:`TransformResult.revalidated` ``False``) instead of letting
   a vacuous gate masquerade as a clean gated pass.

Backends implement only :attr:`DatasetTransform.provider_name`,
:meth:`DatasetTransform.validate` and :meth:`DatasetTransform.transform_frames`;
the dataset plumbing lives here once so every backend inherits the same
pass-through and provenance guarantees. See
:class:`~strands_robots.transforms.mock.MockTransform` for the canonical
no-dependency reference implementation and
:class:`~strands_robots.transforms.cosmos_transfer.CosmosTransferTransform` for
the first generative backend.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strands_robots.utils import (
    boolean_flag_error,
    non_negative_whole_number_error,
    positive_whole_number_error,
)

logger = logging.getLogger(__name__)

# Columns LeRobot adds to every v3 dataset itself; their presence in the source
# feature schema is not "an extra feature the transform would drop".
_LEROBOT_AUTO_FEATURES = frozenset({"timestamp", "frame_index", "episode_index", "index", "task_index", "task"})

_IMAGE_PREFIX = "observation.images."


@dataclass
class TransformSpec:
    """Provider-agnostic episode-augmentation specification.

    Concrete transforms read the fields they support and **ignore the rest** -
    the same tolerance rule :class:`~strands_robots.training.base.TrainSpec`
    states for trainers. Backend-specific knobs live in :attr:`extra` until two
    backends share them.

    Attributes:
        source_root: Path to the LeRobotDataset v3 root to read (must contain
            ``meta/info.json``). This is exactly what
            :class:`~strands_robots.dataset_recorder.DatasetRecorder` /
            ``Robot.stop_recording`` produce.
        output_root: Directory the augmented LeRobotDataset is written to.
            Must not be the source root - with ``overwrite=True`` that spelling
            would destroy the recorded data the transform exists to multiply,
            so it is refused whatever the flag says.
        source_repo_id: Identifier written into the source dataset handle when
            it is opened by ``source_root``. Purely a label here (the root is
            the load-bearing field); recorded into each provenance record so a
            generated episode names where its trajectory came from.
        output_repo_id: Identifier for the augmented dataset. A label, like
            :attr:`source_repo_id`; the dataset lives at :attr:`output_root`.
        episodes: Subset of source episode indices to transform. ``None``
            (default) means every episode. This is a subset selector, so it is
            read by membership rather than truthiness: an EMPTY list asks for
            nothing and is refused rather than widened to "all", and a bare
            int, a repeated index or a negative index cannot be honored as
            written and are refused too.
        variants_per_episode: How many generated variants each source episode
            yields, as a positive whole number. The data-multiplication factor:
            N recorded episodes become up to ``N * variants_per_episode``
            generated episodes (minus any the re-validation gate discards).
        seed: Base seed for deterministic generation, a non-negative whole
            number or ``None`` (backend-nondeterministic). The orchestration
            derives a distinct per-``(episode, variant)`` seed from it via
            :func:`derive_variant_seed`, so two variants of one episode never
            see the same stream.
        prompt: Style / domain instruction forwarded to generative backends
            (e.g. ``"the same scene in a cluttered kitchen at night"``).
            Ignored by backends that take no instruction.
        revalidate: Deterministic verdict function for the acceptance gate, or
            ``None`` to skip gating (:attr:`TransformResult.revalidated`
            reports which happened, so an ungated output never masquerades as
            a gated one). Called with one episode's frames as a dict -
            ``"action"`` / ``"observation.state"`` as ``(T, N) float32`` arrays,
            each ``"observation.images.<cam>"`` as ``(T, H, W, 3) uint8``, and
            ``"task"`` as a list of per-frame strings - and returns a bool.
            The gate compares the verdict on the source episode with the
            verdict on each generated variant; a flip discards the variant.
            **The verdict must read at least one ``observation.images.<cam>``
            column to gate anything**: every other column passes through
            byte-identical (guarantee 1), so a verdict that consults no image
            column returns the same answer on the source and on every variant
            and can never flip. The orchestration measures which columns the
            source-verdict call consulted and reports such a run as ungated
            (:attr:`TransformResult.revalidated` ``False``, cause in the
            message) rather than as a clean gated pass.
        overwrite: When :attr:`output_root` already holds a dataset, remove it
            and write fresh (checked as a strict boolean; every other spelling
            of a posture flag is refused rather than read by truthiness).
        extra: Raw passthrough for backend-specific knobs. The escape hatch
            that keeps the ABC stable as backends evolve.
    """

    source_root: str = ""
    output_root: str = ""
    source_repo_id: str = "local/source"
    output_repo_id: str = "local/augmented"
    episodes: list[int] | None = None
    variants_per_episode: int = 1
    seed: int | None = None
    prompt: str = ""
    revalidate: Callable[[dict[str, Any]], bool] | None = None
    overwrite: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformResult:
    """Outcome of a :meth:`DatasetTransform.transform` call.

    Attributes:
        status: ``"success"`` | ``"error"``.
        output_root: Root of the augmented dataset (``None`` on validation
            failure, before anything was written).
        episodes_read: Source episodes read.
        episodes_written: Generated episodes written to the output dataset.
        episodes_discarded: Generated episodes the re-validation gate rejected
            because their deterministic verdict flipped relative to the source
            episode. Measured, never assumed: ``0`` with
            :attr:`revalidated` ``False`` means "nothing was checked", not
            "nothing flipped".
        revalidated: Whether a verdict function actually gated the output.
            ``False`` means one of two causes, and :attr:`message` names
            which: :attr:`TransformSpec.revalidate` was ``None`` (every
            generated episode was written unchecked), or the supplied verdict
            consulted no ``observation.images.*`` column - every other column
            passes through byte-identical, so such a verdict returns the same
            answer on the source and on every variant and can never flip. A
            vacuous gate is a run that was NOT gated, and it must not render
            as the reassuring "checked, nothing flipped".
        provenance_path: Path of the provenance sidecar written into the
            output dataset (``None`` on validation failure).
        message: Human-readable summary / error detail.
    """

    status: str
    output_root: str | None = None
    episodes_read: int = 0
    episodes_written: int = 0
    episodes_discarded: int = 0
    revalidated: bool = False
    provenance_path: str | None = None
    message: str = ""


def derive_variant_seed(seed: int | None, source_episode: int, variant: int) -> int | None:
    """Derive the deterministic per-variant seed a backend generates with.

    ``None`` stays ``None`` (the caller opted out of determinism); otherwise
    the triple ``(seed, source_episode, variant)`` is spread through
    ``numpy``'s :class:`~numpy.random.SeedSequence` so two variants of one
    episode - or one variant of two episodes - never share a stream while the
    same spec always reproduces the same output.

    Args:
        seed: The spec's base seed (:attr:`TransformSpec.seed`), a non-negative
            whole number or ``None`` to opt out of determinism.
        source_episode: Source episode index the variant is generated from.
        variant: Variant counter within that episode
            (``0 .. variants_per_episode - 1``), a non-negative whole number.

    Returns:
        A 32-bit seed, or ``None`` when ``seed`` is ``None``.

    Raises:
        ValueError: Any of the three inputs is outside the non-negative
            whole-number domain the key needs (``seed=None`` excepted - that
            spelling opts out of determinism rather than naming a stream).
            Each is checked because each spreads into the same
            :class:`~numpy.random.SeedSequence`, so an unusable value on any
            of them yields a key some other triple already owns: ``True`` is
            the value worth naming, since unrefused it is ``1`` to NumPy, and
            a str spelling of a whole number is coerced to it. Episode
            ``True`` therefore silently collided with episode 1, variant
            ``True`` (or ``"1"``) with variant 1, and ``seed=True`` with
            ``seed=1`` - two "distinct" variants generated from one stream,
            written as two episodes whose pixels are byte-identical. The
            values NumPy refuses on its own reached here as its internal
            ``TypeError``/``ValueError`` naming neither the parameter nor this
            surface.
    """
    for name, value in (("seed", seed), ("source_episode", source_episode), ("variant", variant)):
        if name == "seed" and value is None:
            continue  # documented opt-out, not a stream name
        if text := non_negative_whole_number_error(value, name, "derive_variant_seed"):
            raise ValueError(text)
    if seed is None:
        return None
    # int() after the guard, never before: the shared rule has already
    # compared the coercion back against the value it was given.
    return int(np.random.SeedSequence([int(seed), int(source_episode), int(variant)]).generate_state(1)[0])


class DatasetTransform(ABC):
    """Abstract base class for episode augmentation over LeRobot datasets.

    Lifecycle: :meth:`validate` (pure preflight) -> :meth:`transform` (read the
    source dataset, generate variants, gate, write the augmented dataset +
    provenance). Backends implement the pixel work
    (:meth:`transform_frames`) and inherit the dataset plumbing, so the
    pass-through, provenance and re-validation guarantees cannot drift per
    backend.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identity, the name :func:`~strands_robots.transforms.factory.create_transform` resolves."""

    @property
    def transform_version(self) -> str:
        """Version string recorded into each generated episode's provenance.

        Defaults to ``"unversioned"``; backends that can name their model /
        pipeline version override this so a provenance record pins *which*
        generator produced the pixels.
        """
        return "unversioned"

    @abstractmethod
    def validate(self, spec: TransformSpec) -> list[str]:
        """Pure, side-effect-free preflight.

        Return a list of human-readable problems; an empty list means the spec
        is runnable. Implementations MUST call :meth:`_spec_problems` first so
        the shared contract (source dataset present, output distinct from
        source, selector / count / flag domains) is checked identically on
        every backend, then append backend-specific problems.

        MUST NOT touch the filesystem beyond read-only stat / config reads,
        spawn processes, or allocate GPUs.
        """

    @abstractmethod
    def transform_frames(
        self,
        camera_key: str,
        frames: np.ndarray,
        spec: TransformSpec,
        *,
        source_episode: int,
        variant: int,
    ) -> np.ndarray:
        """Transform one camera stream of one episode variant.

        The only surface a backend owns. Called once per
        ``(episode, variant, camera)`` with the decoded source pixels; the
        orchestration writes the returned pixels into the output episode and
        copies every non-image column from the source unchanged.

        Args:
            camera_key: Bare camera name (the ``<cam>`` of
                ``observation.images.<cam>``).
            frames: Source pixels as a ``(T, H, W, 3) uint8`` array.
            spec: The running spec (for :attr:`TransformSpec.prompt` /
                :attr:`TransformSpec.extra`).
            source_episode: Source episode the variant derives from.
                Implementations refuse a value outside the non-negative
                whole-number domain (the shared rule
                :func:`derive_variant_seed` applies) rather than deriving a
                determinism key from it; this abstract declaration carries no
                body, so the guard lives in each implementation.
            variant: Variant counter within that episode. Together with
                :attr:`TransformSpec.seed` and ``source_episode`` this is the
                determinism key - see :func:`derive_variant_seed`.
                Implementations refuse a value outside the same non-negative
                whole-number domain, for the same reason they refuse one on
                ``source_episode``: an unusable counter names a stream another
                variant already owns. The guard lives in each implementation
                because a backend may never reach
                :func:`derive_variant_seed` (a fixed shift reads no key), and
                this abstract declaration carries no body.

        Returns:
            Transformed pixels with the SAME shape and dtype as ``frames``.
            The orchestration refuses any other return loudly rather than
            writing a dataset whose declared schema the pixels do not match.
        """

    # ------------------------------------------------------------------
    # Shared preflight
    # ------------------------------------------------------------------

    def _spec_problems(self, spec: TransformSpec) -> list[str]:
        """Shared spec preflight every backend's :meth:`validate` calls first.

        Checks the provider-agnostic half of the contract: the source is a
        LeRobotDataset v3 root, the output is named and distinct from the
        source, and the selector / count / flag fields are inside their value
        domains. Read-only (stat + ``isfile``); never writes.
        """
        problems: list[str] = []
        ctx = f"{self.provider_name}.validate"

        if not spec.source_root:
            problems.append("source_root is required")
        else:
            info = os.path.join(spec.source_root, "meta", "info.json")
            if not os.path.isfile(info):
                problems.append(f"source_root is not a LeRobotDataset v3 root (missing {info})")

        if not spec.output_root:
            problems.append("output_root is required")
        elif spec.source_root and os.path.abspath(spec.output_root) == os.path.abspath(spec.source_root):
            # With overwrite=True this spelling would DELETE the recorded
            # source dataset before reading it; refused whatever the flag says.
            problems.append("output_root must not be the source_root (the transform would overwrite its own input)")
        elif os.path.isdir(os.path.join(spec.output_root, "meta")) and spec.overwrite is not True:
            problems.append(
                f"output_root already holds a dataset ({spec.output_root}); pass overwrite=True to replace it"
            )

        # ``episodes`` is a SUBSET SELECTOR over the source episodes, so it is
        # read ``is None`` (all) / by membership (some), never by truthiness:
        # an empty selection asks for nothing and widening it to "all" is the
        # opposite answer, and a repeated index does its unit of work twice.
        if spec.episodes is not None:
            if not isinstance(spec.episodes, list):
                problems.append(
                    f"episodes must be a list of source episode indices or None for all "
                    f"(got {type(spec.episodes).__name__})"
                )
            elif not spec.episodes:
                problems.append(
                    "episodes selects an EMPTY subset - an empty selection cannot be honored; "
                    "pass None to transform every episode or name the episodes to transform"
                )
            else:
                for idx in spec.episodes:
                    if text := non_negative_whole_number_error(idx, "episodes[]", ctx):
                        problems.append(text)
                        break
                else:
                    if len({int(i) for i in spec.episodes}) != len(spec.episodes):
                        problems.append("episodes contains a repeated index - each source episode is named once")

        if text := positive_whole_number_error(spec.variants_per_episode, "variants_per_episode", ctx):
            problems.append(text)

        if spec.seed is not None and (text := non_negative_whole_number_error(spec.seed, "seed", ctx)):
            problems.append(text)

        if text := boolean_flag_error(spec.overwrite, "overwrite", ctx):
            problems.append(text)

        if spec.revalidate is not None and not callable(spec.revalidate):
            problems.append(
                f"revalidate must be a callable episode-verdict function or None (got {type(spec.revalidate).__name__})"
            )

        if not isinstance(spec.prompt, str):
            problems.append(f"prompt must be a string (got {type(spec.prompt).__name__})")

        return problems

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def transform(self, spec: TransformSpec) -> TransformResult:
        """Run the augmentation: dataset in, provenance-marked dataset out.

        Reads each selected source episode, calls :meth:`transform_frames` per
        camera stream per variant, copies the action / state / task columns
        from the source unchanged, applies the re-validation gate when
        :attr:`TransformSpec.revalidate` is supplied, writes surviving
        variants into a fresh LeRobotDataset at
        :attr:`TransformSpec.output_root`, and records every written episode
        in ``meta/provenance.json``.

        Args:
            spec: The augmentation specification. Validated first
                (:meth:`validate`); a spec with problems returns an ``error``
                result before anything is read or written.

        Returns:
            A :class:`TransformResult` with the measured counts. ``error``
            status on validation failure or when the ``lerobot`` extra is
            missing; a backend returning pixels of the wrong shape / dtype
            raises ``ValueError`` instead, because that is a backend bug and
            not a property of the spec.

        Raises:
            ValueError: :meth:`transform_frames` returned an array whose shape
                or dtype differs from its input (the output schema could not
                hold it).
        """
        problems = self.validate(spec)
        if problems:
            return TransformResult(
                status="error",
                message="validation failed: " + "; ".join(problems),
            )

        from strands_robots.dataset_recorder import lerobot_dataset_import_error

        if err := lerobot_dataset_import_error():
            return TransformResult(status="error", message=err)

        reader = _SourceDataset.open(spec)
        if isinstance(reader, str):
            return TransformResult(status="error", message=reader)

        selected = reader.selected_episodes(spec.episodes)
        if isinstance(selected, str):
            return TransformResult(status="error", message=selected)

        recorder = None
        provenance_records: list[dict[str, Any]] = []
        episodes_read = 0
        episodes_written = 0
        episodes_discarded = 0
        gated = spec.revalidate is not None
        pixel_blind_episodes: list[int] = []

        try:
            for source_episode in selected:
                episode = reader._read_episode(source_episode)
                episodes_read += 1
                source_verdict: bool | None = None
                if spec.revalidate is not None:
                    # Record which columns the verdict consults: a source call
                    # that read no observation.images.* column read only values
                    # every variant shares byte-identically (guarantee 1), so
                    # this episode's gate is provably vacuous and the run must
                    # not claim it was gated.
                    probe = _KeyRecordingEpisode(episode)
                    source_verdict = bool(spec.revalidate(probe))
                    if not probe.consulted_an_image_column():
                        pixel_blind_episodes.append(int(source_episode))

                for variant in range(int(spec.variants_per_episode)):
                    generated = dict(episode)
                    for cam in reader.camera_keys:
                        source_frames = episode[f"{_IMAGE_PREFIX}{cam}"]
                        out_frames = self.transform_frames(
                            cam,
                            source_frames,
                            spec,
                            source_episode=source_episode,
                            variant=variant,
                        )
                        _require_same_frame_schema(self.provider_name, cam, source_frames, out_frames)
                        generated[f"{_IMAGE_PREFIX}{cam}"] = out_frames

                    if spec.revalidate is not None and bool(spec.revalidate(generated)) != source_verdict:
                        episodes_discarded += 1
                        logger.info(
                            "transform %s: discarded episode %d variant %d (deterministic verdict flipped)",
                            self.provider_name,
                            source_episode,
                            variant,
                        )
                        continue

                    if recorder is None:
                        recorder = reader.create_output_recorder(spec)
                    _write_episode(recorder, reader, generated)
                    provenance_records.append(
                        {
                            "episode_index": episodes_written,
                            "synthetic": True,
                            "source_episode_index": int(source_episode),
                            "source_repo_id": spec.source_repo_id,
                            "transform": self.provider_name,
                            "transform_version": self.transform_version,
                            "variant": variant,
                            "prompt": spec.prompt,
                            "seed": derive_variant_seed(spec.seed, source_episode, variant),
                        }
                    )
                    episodes_written += 1
        finally:
            if recorder is not None:
                recorder.finalize()

        provenance_path: str | None = None
        if recorder is not None:
            from strands_robots.transforms.provenance import write_provenance

            provenance_path = str(write_provenance(spec.output_root, provenance_records))

        # A verdict that consulted no image column on some source episode is
        # vacuous for that episode's variants: reporting the run as gated
        # would render exactly like "checked, nothing flipped", which is the
        # silent direction. Never over-claim - the whole run refuses the
        # gated label, and the message names the cause.
        gate_vacuous = gated and bool(pixel_blind_episodes)
        if gate_vacuous:
            gate_note = (
                f" (revalidate consulted no observation.images.* column on source episode(s) "
                f"{pixel_blind_episodes} - every non-image column passes through byte-identical, "
                "so this verdict cannot flip and the output was NOT gated)"
            )
        elif gated:
            gate_note = ""
        else:
            gate_note = " (no revalidate function supplied - output was NOT gated)"

        return TransformResult(
            status="success",
            output_root=spec.output_root if recorder is not None else None,
            episodes_read=episodes_read,
            episodes_written=episodes_written,
            episodes_discarded=episodes_discarded,
            revalidated=gated and not gate_vacuous,
            provenance_path=provenance_path,
            message=(
                f"{self.provider_name}: read {episodes_read} episode(s), wrote {episodes_written} "
                f"generated episode(s), discarded {episodes_discarded} on the re-validation gate" + gate_note
            ),
        )


class _KeyRecordingEpisode(dict):
    """Episode dict that records which columns a verdict function consults.

    Wrapped around the SOURCE episode for the one verdict call per episode in
    :meth:`DatasetTransform.transform`. Guarantee 1 holds every non-image
    column byte-identical between the source and each generated variant, so a
    deterministic verdict whose source call consulted no
    ``observation.images.*`` column read only values every variant shares -
    it returns the same answer on all of them and can never flip. Recording
    the consulted keys is what lets the orchestration refuse to report that
    vacuous gate as a clean gated pass.

    Recording is over every way a mapping can hand a stored value to its
    caller, not just subscription, because the accusation is only honest if
    the probe saw everything the verdict saw:

    * ``[]`` / ``get()`` / ``pop()`` / ``popitem()`` / ``setdefault()`` name
      one key, so they record that key.
    * ``items()`` / ``values()`` / ``==`` / ``!=`` / ``copy()`` / ``|``
      conservatively record every key: a verdict that iterated, compared or
      **copied** the whole mapping received every column's value, so accusing
      it of reading no pixels would be the false refusal. ``dict(episode)``
      and ``{**episode}`` are that same bulk read - the defensive copy a
      verdict makes before touching the caller's mapping - and they are why
      ``__iter__`` is overridden: it records nothing itself (iteration yields
      keys, never values), but CPython takes its dict-merge fast path, which
      reads the C table directly and invisibly to ``__getitem__``, only while
      ``tp_iter`` is ``dict.__iter__``. Overriding it routes both spellings
      through ``keys()`` and ``__getitem__`` per key instead, which the
      totality tests pin so an interpreter change fails loudly rather than
      silently returning the probe to under-reporting.

    Overriding equality also keeps the added ``consulted`` attribute out of
    the comparison on purpose - the probe is a transparent stand-in for the
    episode dict, so it compares by episode contents exactly as the wrapped
    dict would, and ``copy()`` / ``|`` likewise return a plain ``dict`` just
    as they do for any other ``dict`` subclass.
    """

    def __init__(self, episode: dict[str, Any]) -> None:
        super().__init__(episode)
        self.consulted: set[Any] = set()

    def __getitem__(self, key: Any) -> Any:
        self.consulted.add(key)
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        self.consulted.add(key)
        return super().get(key, default)

    def _record_every_key(self) -> None:
        """Conservative record for a bulk read that hands out every value."""
        self.consulted.update(super().keys())

    def items(self) -> Any:
        self._record_every_key()
        return super().items()

    def values(self) -> Any:
        self._record_every_key()
        return super().values()

    def __eq__(self, other: Any) -> bool:
        self._record_every_key()
        return super().__eq__(other)

    def __ne__(self, other: Any) -> bool:
        self._record_every_key()
        return super().__ne__(other)

    def copy(self) -> dict[str, Any]:
        self._record_every_key()
        return dict(super().items())

    # ``Any`` return: ``dict.__or__`` / ``__ror__`` are overloaded in typeshed
    # over the operand's key and value types, and a single concrete signature
    # is not compatible with that. The runtime result is a plain ``dict``, as
    # it is for any other ``dict`` subclass, which the transparency tests pin.
    def __or__(self, other: Any) -> Any:
        self._record_every_key()
        return dict(super().items()) | dict(other)

    def __ror__(self, other: Any) -> Any:
        self._record_every_key()
        return dict(other) | dict(super().items())

    def pop(self, key: Any, *default: Any) -> Any:
        self.consulted.add(key)
        return super().pop(key, *default)

    def popitem(self) -> Any:
        key, value = super().popitem()
        self.consulted.add(key)
        return key, value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        self.consulted.add(key)
        return super().setdefault(key, default)

    def __iter__(self) -> Any:
        # Records nothing: iteration yields keys, never values. The override
        # exists so ``dict(probe)`` and ``{**probe}`` are recorded at all -
        # see the class docstring on CPython's dict-merge fast path.
        return super().__iter__()

    # dict subclasses are unhashable by default; equality above compares by
    # episode contents (``consulted`` is instrumentation, not identity), so
    # hashing stays refused exactly as it is for the wrapped dict.
    __hash__ = None  # type: ignore[assignment]

    def consulted_an_image_column(self) -> bool:
        """Whether the verdict read at least one ``observation.images.*`` value."""
        return any(isinstance(k, str) and k.startswith(_IMAGE_PREFIX) for k in self.consulted)


def _require_same_frame_schema(provider: str, camera_key: str, source: np.ndarray, out: Any) -> None:
    """Refuse a backend return the output schema cannot hold.

    The output dataset declares the SOURCE camera shape, so pixels of another
    shape or dtype would be written under a schema they do not match - a wrong
    dataset rather than an error. Raised loudly instead.
    """
    if not isinstance(out, np.ndarray) or out.shape != source.shape or out.dtype != source.dtype:
        got = f"{type(out).__name__}"
        if isinstance(out, np.ndarray):
            got = f"ndarray(shape={out.shape}, dtype={out.dtype})"
        raise ValueError(
            f"{provider}.transform_frames returned {got} for camera '{camera_key}' "
            f"where the contract requires ndarray(shape={source.shape}, dtype={source.dtype}) - "
            "a transform changes pixels, never the stream's schema"
        )


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a LeRobot frame value (torch tensor or array-like) to numpy."""
    if hasattr(value, "detach"):  # torch.Tensor without importing torch here
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_hwc_uint8(value: Any) -> np.ndarray:
    """Convert one decoded frame to the ``(H, W, 3) uint8`` transform contract.

    LeRobot read-back yields channel-first float images in ``[0, 1]``;
    image-mode datasets can yield ``uint8`` HWC directly. Both normalize here.
    """
    arr = _to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
    return arr


class _SourceDataset:
    """Read side of the orchestration: one opened source LeRobotDataset.

    Internal to :meth:`DatasetTransform.transform`; holds the schema facts
    (camera keys / dims / dtypes, state and action column names, fps, robot
    type) the output recorder is created from, so output schema parity is
    derived from the source rather than restated.

    ``use_videos`` is one flag for the whole camera set because the output
    recorder declares one dtype for every camera, so a source whose cameras
    disagree cannot be reproduced and :meth:`open` refuses it.
    """

    def __init__(self, ds: Any, spec: TransformSpec) -> None:
        self.ds = ds
        features: dict[str, Any] = dict(ds.meta.features)
        self.camera_keys: list[str] = [k[len(_IMAGE_PREFIX) :] for k in features if k.startswith(_IMAGE_PREFIX)]
        self.camera_dims: dict[str, tuple[int, int]] = {}
        self.camera_dtypes: dict[str, str] = {}
        for cam in self.camera_keys:
            feat = features[f"{_IMAGE_PREFIX}{cam}"]
            shape = tuple(feat.get("shape", ()))
            # _build_features declares (3, H, W); some datasets declare (H, W, 3).
            if len(shape) == 3 and shape[0] == 3 and shape[-1] != 3:
                self.camera_dims[cam] = (int(shape[1]), int(shape[2]))
            elif len(shape) == 3:
                self.camera_dims[cam] = (int(shape[0]), int(shape[1]))
            self.camera_dtypes[cam] = str(feat.get("dtype") or "")
        # One flag for the whole camera set, because that is what the output
        # recorder declares. Assigning it per camera inside the loop left it
        # holding whichever camera the iteration ended on, so a source mixing
        # dtypes re-encoded the others silently; open() refuses that source.
        self.use_videos = all(dtype == "video" for dtype in self.camera_dtypes.values())
        state_feat = features.get("observation.state", {})
        action_feat = features.get("action", {})
        self.state_names: list[str] = list(state_feat.get("names") or [])
        self.action_names: list[str] = list(action_feat.get("names") or [])
        self.fps = int(ds.meta.fps)
        self.robot_type = str(getattr(ds.meta, "robot_type", None) or "unknown")
        self.total_episodes = int(ds.meta.total_episodes)
        self._spec = spec

    @classmethod
    def open(cls, spec: TransformSpec) -> _SourceDataset | str:
        """Open the source dataset; return an error string instead of raising.

        A string return keeps :meth:`DatasetTransform.transform`'s error
        channel uniform (a ``TransformResult`` with ``status="error"``) for
        every source-side refusal. Three sources are refused, all for the same
        reason - the output could not be the source rendered differently: one
        declaring a feature the pass-through cannot preserve, one declaring no
        camera stream at all, and one whose cameras disagree about their dtype.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id=spec.source_repo_id, root=spec.source_root)
        reader = cls(ds, spec)

        features: dict[str, Any] = dict(ds.meta.features)
        unsupported = sorted(
            k
            for k in features
            if k not in _LEROBOT_AUTO_FEATURES
            and not k.startswith(_IMAGE_PREFIX)
            and k not in ("observation.state", "action")
        )
        if unsupported:
            # Refuse rather than silently drop a column the pass-through
            # contract promises to preserve.
            return (
                f"source dataset declares feature(s) the transform pass-through cannot preserve: {unsupported}. "
                "The contract copies observation.state / action / task and transforms observation.images.*; "
                "a column outside that set would be silently dropped from the output, so it is refused instead."
            )
        if not reader.camera_keys:
            return (
                "source dataset declares no observation.images.* stream - an episode augmentation transforms "
                "video observations, so a dataset without any has nothing to transform"
            )
        if len(set(reader.camera_dtypes.values())) > 1:
            # Same reason as the unsupported-feature refusal above: the output
            # recorder declares one dtype for every camera (use_videos is a
            # single flag), so a mixed source cannot be reproduced. Writing it
            # anyway re-encodes at least one camera - a video stream flattened
            # into still images, or an image column promoted to video - and
            # which way it goes depends on the order the features are declared
            # in. Schema parity is part of the pass-through contract, so the
            # source is refused rather than silently changed.
            declared = ", ".join(f"{_IMAGE_PREFIX}{cam}={dtype!r}" for cam, dtype in reader.camera_dtypes.items())
            return (
                f"source dataset declares camera streams with more than one dtype: {declared}. "
                "The output dataset declares one dtype for every camera, so at least one stream would be "
                "re-encoded in the output - a video stream written as still images, or an image column "
                "promoted to video - which the schema-parity half of the pass-through contract forbids. "
                "Re-record or convert the source so every observation.images.* stream shares one dtype."
            )
        return reader

    def selected_episodes(self, episodes: list[int] | None) -> list[int] | str:
        """Resolve the episode selector against the source episode count."""
        if episodes is None:
            return list(range(self.total_episodes))
        out_of_range = [int(i) for i in episodes if int(i) >= self.total_episodes]
        if out_of_range:
            return (
                f"episodes {out_of_range} out of range - the source dataset has "
                f"{self.total_episodes} episode(s) (0-{self.total_episodes - 1})"
            )
        return [int(i) for i in episodes]

    def _episode_frame_range(self, episode: int) -> tuple[int, int]:
        """Resolve one episode's ``[from, to)`` frame range, version-tolerantly.

        LeRobot 0.6 records the range in ``meta.episodes`` as
        ``dataset_from_index`` / ``dataset_to_index``; older versions expose an
        ``episode_data_index`` tensor dict; the last resort accumulates
        per-episode ``length``. Same three rungs, in the same order, as
        :func:`~strands_robots.dataset_recorder.load_lerobot_episode`, which
        adds a fourth (a frame-by-frame scan) behind a broad ``except`` that
        this reader deliberately does not have: a transform that cannot read
        the range should refuse rather than decode the whole dataset to guess
        it.
        """
        ep_info = self.ds.meta.episodes[episode]
        if "dataset_from_index" in ep_info:
            return int(ep_info["dataset_from_index"]), int(ep_info["dataset_to_index"])
        if hasattr(self.ds, "episode_data_index"):
            return (
                int(self.ds.episode_data_index["from"][episode].item()),
                int(self.ds.episode_data_index["to"][episode].item()),
            )
        start = sum(int(self.ds.meta.episodes[i]["length"]) for i in range(episode))
        return start, start + int(ep_info["length"])

    def _read_episode(self, episode: int) -> dict[str, Any]:
        """Read one source episode into the transform's in-memory frame dict."""
        from_idx, to_idx = self._episode_frame_range(episode)

        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        tasks: list[str] = []
        images: dict[str, list[np.ndarray]] = {cam: [] for cam in self.camera_keys}
        for idx in range(from_idx, to_idx):
            frame = self.ds[idx]
            if self.state_names:
                states.append(_to_numpy(frame["observation.state"]).astype(np.float32, copy=False))
            if self.action_names:
                actions.append(_to_numpy(frame["action"]).astype(np.float32, copy=False))
            tasks.append(str(frame.get("task", "")))
            for cam in self.camera_keys:
                images[cam].append(_image_to_hwc_uint8(frame[f"{_IMAGE_PREFIX}{cam}"]))

        episode_dict: dict[str, Any] = {"task": tasks}
        if states:
            episode_dict["observation.state"] = np.stack(states)
        if actions:
            episode_dict["action"] = np.stack(actions)
        for cam in self.camera_keys:
            episode_dict[f"{_IMAGE_PREFIX}{cam}"] = np.stack(images[cam])
        return episode_dict

    def create_output_recorder(self, spec: TransformSpec) -> Any:
        """Create the output dataset with the SOURCE schema (parity by construction).

        ``use_videos`` is the single dtype every source camera declares, which
        :meth:`open` has already established is unambiguous.
        """
        from strands_robots.dataset_recorder import DatasetRecorder

        return DatasetRecorder.create(
            spec.output_repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            camera_keys=list(self.camera_keys),
            camera_dims=dict(self.camera_dims),
            joint_names=list(self.state_names) or None,
            action_names=list(self.action_names) or None,
            root=spec.output_root,
            use_videos=self.use_videos,
            overwrite=spec.overwrite,
        )


def _write_episode(recorder: Any, reader: _SourceDataset, episode: dict[str, Any]) -> None:
    """Write one generated episode through the recorder, columns pass-through."""
    tasks: list[str] = episode["task"]
    frame_count = len(tasks)
    for t in range(frame_count):
        observation: dict[str, Any] = {}
        if reader.state_names and "observation.state" in episode:
            state_row = episode["observation.state"][t]
            observation.update(zip(reader.state_names, (float(v) for v in state_row)))
        for cam in reader.camera_keys:
            observation[cam] = episode[f"{_IMAGE_PREFIX}{cam}"][t]
        action: dict[str, Any] = {}
        if reader.action_names and "action" in episode:
            action_row = episode["action"][t]
            action.update(zip(reader.action_names, (float(v) for v in action_row)))
        recorder.add_frame(
            observation,
            action,
            task=tasks[t],
            camera_keys=list(reader.camera_keys),
        )
    recorder.save_episode()
