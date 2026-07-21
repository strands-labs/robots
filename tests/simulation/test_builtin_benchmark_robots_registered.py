"""Every shipped built-in benchmark must name a REAL registry robot.

Each built-in locomotion benchmark spec
(:mod:`strands_robots.simulation.builtin_benchmarks`) declares a
``default_robot`` and a ``supported_robots`` list - ``unitree_go2`` for the
quadruped task, ``unitree_g1`` / ``booster_t1`` for the humanoid tasks. Those
robot names are only consumed at run time, inside
``BenchmarkProtocol.on_episode_start`` (the base impl adds the ``default_robot``
when the sim is empty and validates the loaded robot against
``supported_robots``). Nothing in ``builtin_benchmark_specs`` /
``register_builtin_benchmarks`` / ``list_benchmarks`` touches the robot
registry, and ``tests/simulation/test_builtin_benchmarks.py`` drives the
compiled predicates on an inline synthetic ``floater`` robot and never calls
``on_episode_start`` - so a built-in that pointed at a phantom robot would keep
every existing test green and only blow up the first time someone actually ran
the benchmark end to end.

A registry rename (``unitree_g1`` -> ``g1``) or a typo in a newly-shipped
built-in is exactly that silent regression. These checks close the gap: they
fail at build time if any built-in robot name no longer resolves to a real
registry entry.

They are registry-only - no MuJoCo, no GL, no asset download - so they run in
any environment (mirroring ``tests/test_registry_integrity.py``, which reads
only ``robots.json``). ``get_robot`` runs the production resolution path
(alias -> canonical -> lookup) and returns ``None`` for an unknown name;
``resolve_name`` alone is insufficient because it echoes an unknown name back
verbatim rather than reporting the miss.
"""

from __future__ import annotations

from strands_robots.registry.robots import get_robot
from strands_robots.simulation.builtin_benchmarks import builtin_benchmark_specs


def test_at_least_one_builtin_shipped():
    """Guard against the specs dict silently emptying (which would make every
    other assertion here vacuously pass)."""
    assert builtin_benchmark_specs(), "no built-in benchmark specs shipped"


def test_every_builtin_default_robot_resolves():
    """Every built-in's ``default_robot`` resolves to a real registry robot.

    Fails if a registry rename or a new built-in's typo leaves a spec pointing
    at a robot ``on_episode_start`` could not load.
    """
    offenders = {
        name: spec["default_robot"]
        for name, spec in builtin_benchmark_specs().items()
        if get_robot(spec["default_robot"]) is None
    }
    assert not offenders, f"built-in benchmarks with an unresolvable default_robot: {offenders}"


def test_every_builtin_supported_robot_resolves():
    """Every entry of every built-in's ``supported_robots`` resolves to a real
    registry robot - the list ``on_episode_start`` validates the loaded robot
    against."""
    offenders: dict[str, list[str]] = {}
    for name, spec in builtin_benchmark_specs().items():
        missing = [r for r in spec.get("supported_robots", []) if get_robot(r) is None]
        if missing:
            offenders[name] = missing
    assert not offenders, f"built-in benchmarks referencing unknown supported_robots: {offenders}"


def test_builtin_default_robot_is_in_its_supported_robots():
    """Each built-in's ``default_robot`` appears in its own non-empty
    ``supported_robots``.

    ``DeclarativeBenchmark.from_dict`` rejects a spec that violates this, but
    the shipped spec dicts exposed by ``builtin_benchmark_specs`` should already
    be internally consistent so ``register_builtin_benchmarks`` never trips that
    check - a default absent from its own supported list is a spec-authoring bug.
    """
    offenders = {
        name: (spec["default_robot"], spec.get("supported_robots"))
        for name, spec in builtin_benchmark_specs().items()
        if spec.get("supported_robots") and spec["default_robot"] not in spec["supported_robots"]
    }
    assert not offenders, f"built-in default_robot not listed in its own supported_robots: {offenders}"
