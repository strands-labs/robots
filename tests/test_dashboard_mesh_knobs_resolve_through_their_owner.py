# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every operator float the dashboard's mesh bridge reads is resolved by its owner.

The bridge reads floats out of the environment in two kinds of place, and each
kind has an owner it has to ask.

``STRANDS_MESH_CAMERA_HZ`` is not the dashboard's knob at all. It is the mesh's,
resolved by :meth:`~strands_robots.mesh.core.Mesh._resolve_camera_hz` to decide
whether the camera loop runs, and the dashboard is the surface that *writes* it:
the settings panel holds ``camera_hz`` and
:func:`~strands_robots.dashboard.settings.apply_mesh_env` pushes it into the
environment the peers read. Reading it back through a second, looser coercion
closed that loop onto a rate nothing publishes at, on the one panel where an
operator both sets the value and learns what took effect. So the two answers are
held equal here across every spelling an operator can type, rather than being
described as equal in a comment: parity is the property, and a comment is not a
grader. The oracle is the publisher's own method, called unbound because it reads
no instance state - a second copy of its rule in this file would be the defect
under test.

The dashboard's own ``STRANDS_DASHBOARD_*`` knobs are resolved at import, where
``float()`` has a failure mode a per-call resolver does not: a typo raises
``ValueError`` while the module body is executing, so the bridge does not lose
one knob but fails to import, from a frame naming ``float`` rather than the
variable. A non-finite value is accepted instead, and reaches each consumer as
one side of a comparison that it removes rather than widens - ``age > ttl`` is
``False`` for every age against both ``nan`` and ``inf``, so a robot that left
the fleet is never aged out.

Both are one missing question - "is this string a number a consumer can honor?" -
which the package answers in
:func:`~strands_robots.utils.finite_number_error` and in
:func:`~strands_robots.mesh.session.hz_from_env`. What is *not* pinned here is
each knob's floor: ``prune_peers`` reads a non-positive ttl as "never prune" and
:meth:`~strands_robots.dashboard.mesh_bridge.EventCoalescer.allow` reads a
non-positive rate as "no ceiling", and those are the consumers' decisions rather
than this domain's.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap

import pytest

from strands_robots.dashboard import mesh_bridge
from strands_robots.mesh.core import Mesh

#: Every spelling of the rate an operator can leave in the environment, from the
#: unset case through the three ``float()`` accepts and no loop can pace itself
#: with. ``"  "`` is here because whitespace is truthy: the reading this replaced
#: tested the raw string and so coerced it.
RATE_SPELLINGS = ("5", "0.5", "0", "-5", "nan", "inf", "Infinity", "1e999", "not-a-number", "  ", "")

#: Values no knob can be resolved to, and which a bare ``float()`` either returns
#: or raises on. Split from the spellings above because these are asserted
#: against a *default*, not against a second surface's answer.
UNUSABLE = ("nan", "inf", "-inf", "Infinity", "1e999", "not-a-number", "1,5", "")

#: The dashboard's own knobs and the default each documents, read from the module
#: rather than restated: the point is that these resolve, not what they default
#: to, and a knob renamed here without being renamed there fails
#: :meth:`TestEveryDashboardKnobResolvesToAUsableNumber.test_the_knobs_scanned_are_the_knobs_the_module_reads`.
DASHBOARD_KNOBS = (
    ("STRANDS_DASHBOARD_PEER_TTL_S", 300.0),
    ("STRANDS_DASHBOARD_PRESENCE_HZ", 1.0),
    ("STRANDS_DASHBOARD_CAMERA_META_HZ", 2.0),
    ("STRANDS_DASHBOARD_POSE_HZ", 1.0),
    ("STRANDS_DASHBOARD_IMU_HZ", 1.0),
    ("STRANDS_DASHBOARD_ODOM_HZ", 1.0),
    ("STRANDS_DASHBOARD_LIDAR_HZ", 1.0),
    ("STRANDS_DASHBOARD_HEALTH_HZ", 0.5),
)


def _set_rate(monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
    """Put *spelling* in the environment, or take the variable out entirely."""
    if spelling == "":
        monkeypatch.delenv("STRANDS_MESH_CAMERA_HZ", raising=False)
    else:
        monkeypatch.setenv("STRANDS_MESH_CAMERA_HZ", spelling)


def _published_rate() -> float:
    """What the mesh camera loop resolves, from the publisher's own method.

    Called unbound: :meth:`Mesh._resolve_camera_hz` reads the environment and two
    module constants and no instance state, so this is the shipped rule rather
    than a copy of it - which is the whole point, since a copy is what this file
    exists to refuse.
    """
    return Mesh._resolve_camera_hz(None)  # type: ignore[arg-type]


class TestTheReportedCameraRateIsThePublishedOne:
    """One knob, two surfaces, and the panel that writes it must not disagree."""

    @pytest.mark.parametrize("spelling", RATE_SPELLINGS)
    def test_the_dashboard_reports_the_rate_the_camera_loop_resolves(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        _set_rate(monkeypatch, spelling)
        reported = mesh_bridge.MeshBridge(peer_id="parity").mesh_info()["camera_hz"]
        assert reported == _published_rate(), (
            f"STRANDS_MESH_CAMERA_HZ={spelling!r}: the settings panel reports {reported!r} while the "
            f"camera loop resolves {_published_rate()!r}, so an operator reads a rate nothing publishes at"
        )

    @pytest.mark.parametrize("spelling", RATE_SPELLINGS)
    def test_the_posture_payload_is_json_a_client_can_parse(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        """``NaN`` and ``Infinity`` are not JSON, so ``/api/mesh/config`` must not emit them."""
        _set_rate(monkeypatch, spelling)
        rate = mesh_bridge.MeshBridge(peer_id="json").mesh_info()["camera_hz"]
        assert math.isfinite(rate), f"STRANDS_MESH_CAMERA_HZ={spelling!r} reported {rate!r}"
        json.dumps({"camera_hz": rate}, allow_nan=False)

    @pytest.mark.parametrize(("spelling", "expected"), [("5", 5.0), ("0.5", 0.5), ("30", 30.0)])
    def test_a_usable_rate_is_still_reported_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str, expected: float
    ) -> None:
        """The over-reach control: routing the read must not swallow a good value."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_HZ", spelling)
        assert mesh_bridge.MeshBridge(peer_id="ok").mesh_info()["camera_hz"] == expected

    def test_an_unusable_rate_is_reported_to_the_operator(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Substituting the default in silence leaves the operator's model wrong."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_HZ", "nan")
        with caplog.at_level("WARNING", logger=mesh_bridge.logger.name):
            mesh_bridge.MeshBridge(peer_id="warns").mesh_info()
        said = [r.getMessage() for r in caplog.records if r.name == mesh_bridge.logger.name]
        assert any("STRANDS_MESH_CAMERA_HZ" in message and "nan" in message for message in said), said


class TestEveryDashboardKnobResolvesToAUsableNumber:
    """The bridge's own knobs are read at import, where a typo costs the module."""

    @pytest.mark.parametrize(("name", "default"), DASHBOARD_KNOBS)
    @pytest.mark.parametrize("spelling", UNUSABLE)
    def test_an_unusable_value_falls_back_to_the_documented_default(
        self, monkeypatch: pytest.MonkeyPatch, name: str, default: float, spelling: str
    ) -> None:
        if spelling == "":
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, spelling)
        resolved = mesh_bridge._env_float(name, str(default))
        assert resolved == default, f"{name}={spelling!r} resolved to {resolved!r}"
        assert math.isfinite(resolved)

    @pytest.mark.parametrize(("spelling", "expected"), [("42", 42.0), ("0", 0.0), ("-1", -1.0), ("0.25", 0.25)])
    def test_a_usable_value_is_honoured_including_the_floors_its_consumer_owns(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str, expected: float
    ) -> None:
        """The over-reach control: this domain decides finiteness, not the floor.

        ``0`` and a negative are usable numbers here on purpose - ``prune_peers``
        reads a non-positive ttl as "never prune" and
        :meth:`~strands_robots.dashboard.mesh_bridge.EventCoalescer.allow` reads a
        non-positive rate as "no ceiling", so refusing them here would take a
        decision away from the consumer that documents it.
        """
        monkeypatch.setenv("STRANDS_DASHBOARD_PEER_TTL_S", spelling)
        assert mesh_bridge._env_float("STRANDS_DASHBOARD_PEER_TTL_S", "300") == expected

    @pytest.mark.parametrize("spelling", ["not-a-number", "nan", "inf", "1e999"])
    def test_the_module_still_imports_when_a_knob_is_unusable(self, spelling: str) -> None:
        """The failure this position has and a per-call resolver does not.

        Run out of process because the import is what is under test: the module is
        already in ``sys.modules`` here, so re-reading the knob in-process would
        grade a different thing.
        """
        probe = textwrap.dedent(
            """
            from strands_robots.dashboard import mesh_bridge
            import math
            assert math.isfinite(mesh_bridge.PEER_TTL_S), mesh_bridge.PEER_TTL_S
            assert all(math.isfinite(hz) for hz in mesh_bridge.COALESCE_HZ.values()), mesh_bridge.COALESCE_HZ
            print(mesh_bridge.PEER_TTL_S)
            """
        )
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            # Inherit the environment and override only the knobs: a pristine env
            # would also drop whatever puts the package on this interpreter's path,
            # so the probe would fail to import for a reason unrelated to the knob.
            env={**os.environ, "STRANDS_DASHBOARD_PEER_TTL_S": spelling, "STRANDS_DASHBOARD_POSE_HZ": spelling},
        )
        assert done.returncode == 0, f"STRANDS_DASHBOARD_PEER_TTL_S={spelling!r} broke the import: {done.stderr}"
        assert done.stdout.strip() == "300.0", done.stdout

    def test_the_knobs_scanned_are_the_knobs_the_module_reads(self) -> None:
        """Derived non-vacuity: a knob added or renamed cannot skip the table above."""
        import ast
        import inspect

        source = ast.parse(inspect.getsource(mesh_bridge))
        read: set[str] = set()
        for node in ast.walk(source):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_env_float"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                read.add(node.args[0].value)
        assert read == {name for name, _ in DASHBOARD_KNOBS}, (
            f"the module resolves {sorted(read)} but this file grades {sorted(name for name, _ in DASHBOARD_KNOBS)}"
        )
