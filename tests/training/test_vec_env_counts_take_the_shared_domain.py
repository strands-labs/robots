"""The two counts ``VecSimEnv`` is constructed with take the shared count domain.

``VecSimEnv`` is the class the RL trainers hand ``RLTrainSpec.num_envs`` to, and
the one that acts on it: it calls ``env_factory`` once per environment, each
building its own physics engine, and sizes one thread pool from ``max_workers``.
Every other owner of a parallel-environment count grades it through
:func:`~strands_robots.utils.positive_count_error` - ``RLTrainSpec`` via the PPO,
FastTD3 and FastSAC backends, and the Isaac backend's config and ``replicate``.
This one hand-rolled ``if num_envs < 1`` and then coerced with ``int()``, and
``max_workers`` had no domain at all, so:

* ``2.5``, ``4.0`` and ``True`` were admitted and silently built 2, 4 and 1
  engines - a count the caller never asked for rather than a refusal.
* ``float("inf")`` and ``"4"`` left the constructor as ``OverflowError`` and
  ``TypeError``, outside the documented ``Raises: ValueError``, naming neither
  the class nor the parameter.
* ``max_workers=float("inf")`` reached ``ThreadPoolExecutor`` as its pool size.

These pin the domain, not the wording: a refusal is identified by the parameter
it names, and the reference verdict is read from the shared helper itself so the
two cannot drift apart again.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from strands_robots.training.rl import VecSimEnv  # noqa: E402 - after torch importorskip
from strands_robots.training.rl.env import SimEnv  # noqa: E402
from strands_robots.utils import positive_count_error  # noqa: E402


class _StandInEnv:
    """The narrowest thing ``VecSimEnv.__init__`` reads: three dims and a device.

    Deliberately not a ``SimEnv`` - these cells grade what happens *before* any
    environment is usable, so the stand-in exists only to be counted.
    """

    num_actor_obs = 4
    num_critic_obs = 4
    num_actions = 2

    def __init__(self) -> None:
        self.device = torch.device("cpu")


class _CountingFactory:
    """``env_factory`` that records how many environments were actually built."""

    def __init__(self) -> None:
        self.built = 0

    def __call__(self) -> SimEnv:
        self.built += 1
        # The stand-in is intentionally looser than SimEnv; that looseness is
        # the point of the cell, so the cast is deliberate.
        return cast("SimEnv", _StandInEnv())


# (value, why it is in the roster). A list of pairs rather than a mapping: as a
# dict ``0``/``False`` and ``4.0``/``4`` would collapse onto one key each, and
# those collapses are exactly the distinctions a count domain is about.
_PROBES: list[tuple[Any, str]] = [
    (2, "a plain count - the control, and it must keep working"),
    (1, "the degenerate single-environment count"),
    (True, "an int subclass a bare `< 1` admits as a silent count of 1"),
    (2.5, "a fractional count no number of engines can honour"),
    (4.0, "an integral float - spendable only after a coercion"),
    (float("nan"), "not a number, and not orderable in the way `< 1` assumes"),
    (float("inf"), "unbounded - int() of it overflows"),
    ("4", "a count as text, e.g. straight off a CLI or an env var"),
    (None, "unstated"),
    (0, "no environments at all"),
    (-1, "negative"),
]
_IDS = [repr(v) for v, _ in _PROBES]

# One value is legitimately answered differently by the two parameters:
# ``max_workers=None`` means "derive it from num_envs", so it is not a refusal
# there. Recorded rather than silently skipped.
_MAX_WORKERS_MEANS_UNSTATED = None


def _refusal(**kwargs: Any) -> str | None:
    """Construct through the public surface; return the refusal text or ``None``.

    Any exception is reported, not just ``ValueError``, so a value leaving the
    constructor outside its documented contract is visible as itself.
    """
    factory = _CountingFactory()
    try:
        vec = VecSimEnv(factory, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the type IS the observation here
        return f"{type(exc).__name__}: {exc}"
    vec.close()
    return None


@pytest.mark.parametrize(("value", "why"), _PROBES, ids=_IDS)
def test_num_envs_agrees_with_the_shared_count_domain(value: Any, why: str) -> None:
    """A count the shared domain refuses is refused here, naming the parameter."""
    expected = positive_count_error(value, "num_envs", "VecSimEnv")
    got = _refusal(num_envs=value)
    if expected is None:
        assert got is None, f"{value!r} ({why}) is a usable count but was refused: {got}"
    else:
        assert got is not None, f"{value!r} ({why}) was admitted; the shared domain refuses it"
        assert "num_envs" in got, f"the refusal of {value!r} does not name num_envs: {got}"


@pytest.mark.parametrize(("value", "why"), _PROBES, ids=_IDS)
def test_max_workers_agrees_with_the_shared_count_domain(value: Any, why: str) -> None:
    """The pool size takes the same domain; ``None`` alone means "derive it"."""
    if value is _MAX_WORKERS_MEANS_UNSTATED:
        assert _refusal(num_envs=2, max_workers=value) is None, "None means derive from num_envs"
        return
    expected = positive_count_error(value, "max_workers", "VecSimEnv")
    got = _refusal(num_envs=2, max_workers=value)
    if expected is None:
        assert got is None, f"{value!r} ({why}) is a usable pool size but was refused: {got}"
    else:
        assert got is not None, f"{value!r} ({why}) was admitted as a pool size"
        assert "max_workers" in got, f"the refusal of {value!r} does not name max_workers: {got}"


@pytest.mark.parametrize(("value", "why"), _PROBES, ids=_IDS)
def test_a_count_leaves_the_constructor_only_as_a_value_error(value: Any, why: str) -> None:
    """The documented contract is ``ValueError``; nothing else may escape.

    ``float("inf")`` reached ``int()`` and raised ``OverflowError``; ``"4"`` and
    ``None`` reached the ``< 1`` comparison and raised ``TypeError``.
    """
    for kwargs in ({"num_envs": value}, {"num_envs": 2, "max_workers": value}):
        try:
            vec = VecSimEnv(_CountingFactory(), **kwargs)
        except ValueError:
            continue
        except BaseException as exc:  # noqa: BLE001 - reporting the escape
            pytest.fail(f"{kwargs} ({why}) escaped as {type(exc).__name__}: {exc}")
        vec.close()


def test_a_refused_count_builds_no_environment() -> None:
    """The count is graded before ``env_factory`` is called even once.

    Each environment owns its own engine, so admitting a count the caller did
    not mean is not a bad number stored - it is that many live engines built.
    """
    refused = 0
    for value, why in _PROBES:
        if positive_count_error(value, "num_envs", "VecSimEnv") is None:
            continue
        refused += 1
        factory = _CountingFactory()
        with pytest.raises(ValueError):
            VecSimEnv(factory, value)
        assert factory.built == 0, f"{value!r} ({why}) built {factory.built} environment(s) before refusal"
    assert refused >= 8, f"roster no longer exercises the refusal path ({refused} refused)"


def test_an_admitted_count_is_the_number_of_environments_built() -> None:
    """An admitted count is published unchanged and spent exactly as given."""
    exercised = []
    for value, _why in _PROBES:
        if positive_count_error(value, "num_envs", "VecSimEnv") is not None:
            continue
        exercised.append(value)
        factory = _CountingFactory()
        vec = VecSimEnv(factory, value)
        assert factory.built == value
        assert vec.num_envs == value
        assert type(vec.num_envs) is int  # published as given, not coerced into shape
        vec.close()
    assert exercised == [2, 1], f"roster no longer exercises the accepting path ({exercised})"


def test_an_admitted_pool_size_reaches_the_executor_as_an_integer() -> None:
    """``max_workers`` sizes a real thread pool, so a non-integer cannot reach it."""
    vec = VecSimEnv(_CountingFactory(), 2, max_workers=3)
    try:
        assert vec._executor is not None
        assert type(vec._executor._max_workers) is int
        assert vec._executor._max_workers == 3
    finally:
        vec.close()


def test_the_constructor_asks_the_shared_domain_rather_than_re_deriving_it() -> None:
    """Graded on the call graph, not on source text - a comment may quote either form."""
    tree = ast.parse(inspect.getsource(VecSimEnv.__init__).strip())
    called = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "positive_count_error" in called, f"__init__ does not call the shared domain; calls {sorted(called)}"
    assert "int" not in called, "int() coercion is back; it is what turned a 2.5 into two live engines"
