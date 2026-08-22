"""Mock dataset transform - the canonical no-dependency reference implementation.

Mirrors :class:`~strands_robots.policies.mock.MockPolicy` and
:class:`~strands_robots.training.mock.MockTrainer`: zero heavy deps,
deterministic, used in tests and as the worked example of the
:class:`~strands_robots.transforms.base.DatasetTransform` contract. It performs
a real (if trivial) pixel transform - a deterministic per-variant brightness
shift - so a round trip can assert that pixels changed while the action / state
columns passed through byte-identical, and so a pixel-verdict re-validation
gate has something to measure.
"""

from __future__ import annotations

import numpy as np

from strands_robots.transforms.base import DatasetTransform, TransformSpec, derive_variant_seed
from strands_robots.utils import non_negative_whole_number_error


class MockTransform(DatasetTransform):
    """Reference :class:`~strands_robots.transforms.base.DatasetTransform` (no deps).

    Shifts every pixel by a constant. The shift is either the explicit
    ``pixel_shift`` handed to the constructor (tests use this to make a
    re-validation verdict flip on purpose) or, by default, drawn
    deterministically from :func:`~strands_robots.transforms.base.derive_variant_seed`
    so distinct variants of one episode differ from each other and a re-run of
    the same spec reproduces the same output.
    """

    def __init__(self, pixel_shift: int | None = None) -> None:
        """Configure the reference transform.

        Args:
            pixel_shift: Fixed additive pixel shift in ``[-255, 255]``, or
                ``None`` (default) to derive a per-variant shift from the
                spec's seed. An explicit value makes the transform's effect on
                any pixel statistic exactly predictable, which is what the
                verdict-flip tests need.
        """
        if pixel_shift is not None and (
            isinstance(pixel_shift, bool) or not isinstance(pixel_shift, int) or not -255 <= pixel_shift <= 255
        ):
            raise ValueError(f"pixel_shift must be an int in [-255, 255] or None (got {pixel_shift!r})")
        self._pixel_shift = pixel_shift

    @property
    def provider_name(self) -> str:
        """Provider identity - the dependency-free ``mock`` reference transform."""
        return "mock"

    @property
    def transform_version(self) -> str:
        """Version recorded into provenance; bumped if the mock's math changes."""
        return "1"

    def validate(self, spec: TransformSpec) -> list[str]:
        """Pure preflight - the shared spec contract only (no backend needs)."""
        return self._spec_problems(spec)

    def transform_frames(
        self,
        camera_key: str,
        frames: np.ndarray,
        spec: TransformSpec,
        *,
        source_episode: int,
        variant: int,
    ) -> np.ndarray:
        """Shift every pixel by the configured / derived constant, clipped to uint8.

        Args:
            camera_key: Bare camera name (unused - the mock shifts every
                stream identically).
            frames: Source pixels, ``(T, H, W, 3) uint8``.
            spec: The running spec; its seed feeds the derived shift.
            source_episode: Source episode index (determinism key input).
            variant: Variant counter (determinism key input).

        Returns:
            Shifted pixels, same shape and dtype.

        Raises:
            ValueError: ``source_episode`` or ``variant`` is outside the
                non-negative whole-number domain the determinism key needs
                (see :func:`~strands_robots.transforms.base.derive_variant_seed`).
                Both are refused before the explicit ``pixel_shift``
                short-circuit, so the two constructor modes agree on the
                accepted domain - and ``variant`` has to be refused here as
                well as there, because that short-circuit never derives a key
                to be refused by.
        """
        for name, value in (("source_episode", source_episode), ("variant", variant)):
            if text := non_negative_whole_number_error(value, name, "mock.transform_frames"):
                raise ValueError(text)
        if self._pixel_shift is not None:
            shift = self._pixel_shift
        else:
            seed = derive_variant_seed(spec.seed, source_episode, variant)
            rng = np.random.default_rng(seed)
            shift = int(rng.integers(-64, 65))
        return np.clip(frames.astype(np.int16) + shift, 0, 255).astype(np.uint8)
