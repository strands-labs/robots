"""The pinned Isaac Sim docker tag must be one that exists on NGC.

``ISAAC_SIM_DOCKER_IMAGE`` was ``nvcr.io/nvidia/isaac-sim:6.0`` and NVIDIA
publishes no ``major.minor`` tag for that image, so it resolved to nothing.
Measured against the registry with ``docker manifest inspect``::

    isaac-sim:6.0     -> no such manifest
    isaac-sim:latest  -> no such manifest
    isaac-sim:6.0.0   -> exists
    isaac-sim:6.0.1   -> exists
    isaac-sim:5.0.0   -> exists
    isaac-sim:4.5.0   -> exists

What made that worse than a stale line in a document is *where* the constant is
read. It is the single source the RECOVERY instructions are composed from:
``IsaacSimulation.is_available()`` returns it in the hint it gives when the runtime
is absent, and ``create_world`` names it in the structured error it returns for the
same reason. So the one message a user sees when they have no Isaac Sim told them
to pull an image that cannot be pulled - and the two tests that already covered
this constant assert only that it appears *in* those messages, which a wrong tag
satisfies perfectly.

These pin the property a registry lookup would confirm, without a network call: a
resolvable tag for this image carries all three version components. That is
checkable offline, deterministic, and fails on the exact value that shipped.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac import _install  # noqa: E402

#: Tags measured to exist on NGC for ``nvidia/isaac-sim``.
_KNOWN_GOOD = ("6.0.1", "6.0.0", "5.0.0", "4.5.0")

#: Tags measured NOT to exist, including the one that shipped.
_KNOWN_BAD = ("6.0", "latest", "5.0", "4.5")


def _tag(image: str) -> str:
    return image.rsplit(":", 1)[-1]


class TestThePinnedImageTagIsResolvable:
    def test_the_image_carries_an_explicit_tag(self) -> None:
        """An untagged image means ``:latest``, which does not exist here."""
        assert ":" in _install.ISAAC_SIM_DOCKER_IMAGE.rsplit("/", 1)[-1]

    def test_the_tag_is_a_full_major_minor_patch(self) -> None:
        """The property that separates a tag NGC serves from one it does not."""
        tag = _tag(_install.ISAAC_SIM_DOCKER_IMAGE)
        assert re.fullmatch(r"\d+\.\d+\.\d+", tag), (
            f"ISAAC_SIM_DOCKER_IMAGE is tagged {tag!r}. NVIDIA publishes only full "
            f"major.minor.patch tags for nvidia/isaac-sim, so a two-component tag "
            f"fails with 'no such manifest'. Known good: {_KNOWN_GOOD}."
        )

    def test_the_tag_is_not_one_measured_absent(self) -> None:
        assert _tag(_install.ISAAC_SIM_DOCKER_IMAGE) not in _KNOWN_BAD

    def test_the_repository_is_unchanged(self) -> None:
        """Guards the fixture: the tag rules above describe *this* image."""
        assert _install.ISAAC_SIM_DOCKER_IMAGE.startswith("nvcr.io/nvidia/isaac-sim:")

    def test_the_pinned_tag_is_at_or_above_the_declared_floor(self) -> None:
        """``ISAAC_SIM_MIN_VERSION`` is the floor the hints advertise; a pinned
        image below it would tell a user to pull something this backend declares
        unsupported."""
        major_minor = ".".join(_tag(_install.ISAAC_SIM_DOCKER_IMAGE).split(".")[:2])
        assert tuple(int(p) for p in major_minor.split(".")) >= tuple(
            int(p) for p in _install.ISAAC_SIM_MIN_VERSION.split(".")
        )


class TestTheRecoveryMessagesCarryIt:
    """Why the tag being right matters: these are what a user follows.

    The pre-existing tests assert the constant appears in each message, which a
    wrong tag also satisfies. These assert the same reachability so the pairing
    stays visible next to the tag rules above - together they say the message
    names a tag *and* the tag is pullable.
    """

    def test_the_absent_runtime_hint_names_the_image(self) -> None:
        from strands_robots.simulation.isaac import IsaacSimulation

        available, reason = IsaacSimulation.is_available()
        if available:
            pytest.skip("Isaac Sim is installed here, so no hint is produced")
        assert reason is not None
        assert _install.ISAAC_SIM_DOCKER_IMAGE in reason

    def test_no_two_component_tag_survives_anywhere_in_the_hint(self) -> None:
        """A second spelling of the tag elsewhere in the composed hint would
        reintroduce the defect while the constant stayed correct.

        Matched as a whole tag rather than as a substring: ``isaac-sim:6.0.1``
        *contains* ``isaac-sim:6.0``, so a plain ``in`` check is unsatisfiable and
        would fail on the correct value.
        """
        from strands_robots.simulation.isaac import IsaacSimulation

        available, reason = IsaacSimulation.is_available()
        if available:
            pytest.skip("Isaac Sim is installed here, so no hint is produced")
        assert reason is not None
        for bad in _KNOWN_BAD:
            # (?![.\d]) so a longer, valid tag sharing this prefix does not match.
            assert not re.search(rf"isaac-sim:{re.escape(bad)}(?![.\d])", reason), (
                f"the hint offers isaac-sim:{bad}, which does not exist on NGC"
            )
