# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared rule for reading a joint-state ordering out of an observation.

A ``Policy`` whose caller declared no ``robot_state_keys`` has to infer which
observation keys form this step's state vector, and the only ordering available
is the observation's own insertion order. The sim backends emit a velocity
companion beside every joint position (:mod:`strands_robots.simulation.mujoco.rendering`
writes ``obs[jnt_name] = qpos`` and then ``obs[f"{jnt_name}.vel"] = qvel``, calling
the second an "Additive key ... existing position-only" consumer contract), so that
order alternates ``[pos0, vel0, pos1, vel1, ...]``. Feeding it to a policy
trained on positions puts a velocity in every other slot and, once the vector is
truncated to the model's state dim, drops the trailing joints entirely - a wrong
state vector with no error raised.

:func:`drop_velocity_siblings` restores the producer's additive contract for
every consumer that infers an ordering this way. It is shared rather than
re-derived per provider because two providers conforming to the same ``Policy``
contract must read the same observation as the same state vector: the LeRobot
provider already filtered here and the Cosmos 3 provider did not, so one
interleaved velocities into its 7-joint request while the other did not, from
the same observation.

Explicitly configured ``robot_state_keys`` are never filtered - see the function
docstring for why an operator naming ``elbow.vel`` is stating the model's input.
"""

from __future__ import annotations

#: Suffix the sim backends append to a joint's additive velocity companion.
VELOCITY_SUFFIX = ".vel"


def drop_velocity_siblings(scalar_keys: list[str]) -> list[str]:
    """Drop each ``<joint>.vel`` whose ``<joint>`` position companion is present.

    Used only for an OBSERVATION-DERIVED state ordering, never for an ordering
    the caller declared. An operator naming ``elbow.vel`` in
    ``robot_state_keys`` is stating the model's input; this only cleans up an
    ordering inferred from whatever the observation happened to contain.

    Pairing is decided per key, not by suffix alone. A ``.vel`` key with NO
    position companion is KEPT, because some embodiments legitimately declare
    velocity state and dropping it would corrupt those instead:
    ``embodiments.json`` gives LeKiwi body-frame base velocities ``x.vel`` /
    ``y.vel`` / ``theta.vel`` with no ``x`` / ``y`` / ``theta`` position key.

    Args:
        scalar_keys: Candidate state keys in observation insertion order.

    Returns:
        The same list, order preserved, minus the paired velocity siblings.
    """
    present = set(scalar_keys)
    return [k for k in scalar_keys if not (k.endswith(VELOCITY_SUFFIX) and k[: -len(VELOCITY_SUFFIX)] in present)]
