"""Non-finite env values must not become resolved mesh knobs.

Both float-valued env resolvers in the mesh package read their value with
``float()``, which accepts ``"nan"``, ``"inf"``, ``"Infinity"`` and
``"1e999"`` (that last one overflowing to ``inf``). Neither tested
finiteness, and neither range test implies it:
``_parse_positive_float_env`` compares ``value < minimum``, which is
``False`` for ``nan``, and ``_resolve_dedup_ttl`` compares ``v > 0``,
which is ``True`` for ``inf``.

What made that reachable rather than theoretical is where the resolved
values go. Every knob ``_parse_positive_float_env`` serves is one side of
a comparison on the safety path, so a ``nan`` does not widen the bound but
removes it: the presence stale/future test is ``False`` for every envelope,
the replay-cache TTL purge keeps every stale entry, and the resume
brute-force cooldown - the throttle over the E-stop override code - is
armed to an instant no ``now < locked_until`` test can satisfy. ``inf``
fails open on the first two and closed on the third.

The rule is not new to the package. Three of its five env-float resolvers
already applied it, and :func:`strands_robots.mesh.security._env_pos_float`
both documents it ("Non-numeric / non-positive / NaN / inf values fall back
to the default") and names ``mesh.core._parse_positive_float_env`` as its
analogue. So this pins the two that did not, the parity between them, and -
structurally - that a sixth resolver cannot ship without the test.

Deliberately out of scope: the ``0`` floor. ``_parse_positive_float_env``
admits it (its ``minimum`` defaults to ``0.0``) while ``_env_pos_float``
refuses it, and what a zero means differs per knob - a zero resume backoff
is arguably "no cooldown", a zero freshness window drops every envelope.
That is a per-knob decision, and :class:`TestTheZeroFloorIsUnchanged` pins
it so this change is not read as having settled it.

One more property is pinned here, about this module rather than the mesh: its
own node ids. ``FLOAT_KNOBS`` parametrizes over resolver *function objects*,
and any id generator that stringifies its argument renders one as
``<function _resume_backoff_s at 0x...>`` - an address that belongs to the
process. pytest's default already resolves a function to its ``__name__``, so
:class:`TestTheNodeIdsAreReproducible` pins that the ids stay address-free and
keep naming the resolver.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import subprocess
import sys
import time

import pytest

from strands_robots.mesh import core as core_mod
from strands_robots.mesh import security as security_mod
from strands_robots.mesh.transport import bridge_transport as bridge_mod

#: Spellings ``float()`` accepts and no comparison-based bound can honor.
NON_FINITE = ["nan", "NaN", "  nan  ", "inf", "Inf", "Infinity", "-inf", "1e999", "-1e999"]

#: Env vars whose resolver is :func:`_parse_positive_float_env`, with the
#: documented default each falls back to.
FLOAT_KNOBS = [
    ("STRANDS_MESH_RESUME_FRESHNESS_S", core_mod._resume_freshness_window_s, 60.0),
    ("STRANDS_MESH_RESUME_FORWARD_SKEW_S", core_mod._resume_forward_skew_s, 5.0),
    ("STRANDS_MESH_RESUME_BACKOFF_S", core_mod._resume_backoff_s, 30.0),
]


#: The test whose parametrization carries the resolver function objects.
TARGET_TEST = "test_resolver_returns_the_documented_default"


class TestEveryResumeKnobFallsBackForANonFiniteValue:
    """A non-finite env value resolves to the documented default."""

    @pytest.mark.parametrize(("env_var", "resolver", "default"), FLOAT_KNOBS)
    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_resolver_returns_the_documented_default(self, monkeypatch, env_var, resolver, default, raw):
        monkeypatch.setenv(env_var, raw)
        resolved = resolver()
        assert math.isfinite(resolved), f"{env_var}={raw!r} resolved to {resolved!r}"
        assert resolved == default

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_the_fallback_is_reported(self, monkeypatch, caplog, raw):
        """This resolver warns on every rejection; a non-finite value is no
        exception. Silently substituting the default would make the one
        rejection an operator cannot see the one that disables a safety bound.
        """
        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", raw)
        with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
            core_mod._resume_freshness_window_s()
        assert any("STRANDS_MESH_RESUME_FRESHNESS_S" in r.message for r in caplog.records)


class TestThePresenceFreshnessBoundSurvives:
    """The stale/future presence test keeps refusing what it refused."""

    @staticmethod
    def _dropped(age_s: float) -> bool:
        """Replicate the production comparison at ``Mesh._on_presence``."""
        window = core_mod._resume_freshness_window_s()
        skew = core_mod._resume_forward_skew_s()
        return age_s > window or age_s < -skew

    @pytest.mark.parametrize("raw", NON_FINITE)
    @pytest.mark.parametrize(("label", "age_s"), [("a year old", 31_536_000.0), ("an hour in the future", -3600.0)])
    def test_an_out_of_window_envelope_is_still_dropped(self, monkeypatch, raw, label, age_s):
        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", raw)
        monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", raw)
        assert self._dropped(age_s), f"{label} presence accepted under {raw!r}"

    def test_an_in_window_envelope_is_still_accepted(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "nan")
        monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "nan")
        assert not self._dropped(2.0)


class TestTheResumeBruteForceCooldownEngagesAndExpires:
    """``locked_until = monotonic() + backoff`` must be a real instant."""

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_the_cooldown_both_engages_and_expires(self, monkeypatch, raw):
        monkeypatch.setenv("STRANDS_MESH_RESUME_BACKOFF_S", raw)
        backoff = core_mod._resume_backoff_s()
        locked_until = time.monotonic() + backoff
        # Engages: a nan backoff made this comparison False, so the throttle
        # over the E-stop override code never took effect.
        assert time.monotonic() < locked_until, f"cooldown never engaged under {raw!r}"
        # Expires: an inf backoff made it True forever, so a resume could
        # never be granted again.
        assert locked_until < time.monotonic() + 86_400.0, f"cooldown never expires under {raw!r}"


class TestTheReplayCacheTtlPurgeStillEvicts:
    """``cutoff = now_mono - ttl_s`` must be a real instant."""

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_stale_entries_are_evicted(self, monkeypatch, raw):
        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", raw)
        monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", raw)
        ttl = core_mod._resume_freshness_window_s() + core_mod._resume_forward_skew_s()
        cache: dict[str, float] = {f"k{i}": 1_000.0 + i for i in range(5)}
        core_mod._evict_replay_cache(cache, max_size=4096, ttl_s=ttl, now_mono=1_000_000.0)
        assert cache == {}, f"stale replay entries survived under ttl_s={ttl!r}"

    def test_a_fresh_entry_is_kept(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_RESUME_FRESHNESS_S", raising=False)
        monkeypatch.delenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", raising=False)
        ttl = core_mod._resume_freshness_window_s() + core_mod._resume_forward_skew_s()
        cache = {"recent": 999_999.0}
        core_mod._evict_replay_cache(cache, max_size=4096, ttl_s=ttl, now_mono=1_000_000.0)
        assert cache == {"recent": 999_999.0}


class TestTheDedupTtlFallsBackForANonFiniteValue:
    """``_resolve_dedup_ttl`` was holed on one side only."""

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_the_resolved_ttl_is_finite(self, monkeypatch, raw):
        monkeypatch.setenv("STRANDS_MESH_DEDUP_TTL", raw)
        resolved = bridge_mod._resolve_dedup_ttl()
        assert math.isfinite(resolved), f"STRANDS_MESH_DEDUP_TTL={raw!r} resolved to {resolved!r}"
        assert resolved == bridge_mod._DEFAULT_DEDUP_TTL_S

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_the_deduplicator_still_forgets(self, monkeypatch, raw):
        """An infinite TTL made the cache never forget, so a heartbeat that
        legitimately recurs with the same canonical triple would be dropped
        as a duplicate for the life of the process.
        """
        monkeypatch.setenv("STRANDS_MESH_DEDUP_TTL", raw)
        dedup = bridge_mod._CommandDeduplicator()
        assert math.isfinite(dedup.ttl)
        cache: dict[str, float] = {"seen": 1_000.0}
        cutoff = 1_000_000.0 - dedup.ttl
        assert cache["seen"] < cutoff, f"a 999_000s-old entry is not past the cutoff under {raw!r}"

    def test_a_usable_ttl_is_still_honored(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_DEDUP_TTL", "45.5")
        assert bridge_mod._resolve_dedup_ttl() == 45.5


class TestBothResolversAgreeOnNonFinite:
    """The finiteness axis is the one these resolvers must not diverge on."""

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_no_resolver_in_the_package_returns_a_non_finite_value(self, monkeypatch, raw):
        monkeypatch.setenv("PARITY_PROBE", raw)
        monkeypatch.setenv("STRANDS_MESH_DEDUP_TTL", raw)
        resolved = [
            core_mod._parse_positive_float_env("PARITY_PROBE", "60"),
            security_mod._env_pos_float("PARITY_PROBE", 60.0),
            bridge_mod._resolve_dedup_ttl(),
        ]
        assert all(math.isfinite(v) for v in resolved), resolved

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_both_documented_analogues_return_the_same_default(self, monkeypatch, raw):
        monkeypatch.setenv("PARITY_PROBE", raw)
        assert core_mod._parse_positive_float_env("PARITY_PROBE", "60") == security_mod._env_pos_float(
            "PARITY_PROBE", 60.0
        )


class TestTheIntResolverNeedsNoSuchTest:
    """The control: ``int()`` refuses every non-finite spelling itself."""

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_the_int_resolver_already_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("INT_PROBE", raw)
        assert core_mod._parse_positive_int_env("INT_PROBE", "4096") == 4096

    def test_the_int_resolver_carries_no_finiteness_test(self):
        """Adding one there would be dead code, so its absence is the
        contract rather than an oversight.
        """
        assert "isfinite" not in inspect.getsource(core_mod._parse_positive_int_env)


class TestTheZeroFloorIsUnchanged:
    """Pins the axis this change deliberately does not decide."""

    def test_the_float_resolver_still_admits_zero(self, monkeypatch):
        monkeypatch.setenv("ZERO_PROBE", "0")
        assert core_mod._parse_positive_float_env("ZERO_PROBE", "60") == 0.0

    def test_the_teleop_resolver_still_refuses_zero(self, monkeypatch):
        monkeypatch.setenv("ZERO_PROBE", "0")
        assert security_mod._env_pos_float("ZERO_PROBE", 60.0) == 60.0

    def test_a_negative_value_still_reports_the_range_reason(self, monkeypatch, caplog):
        """The finiteness test runs first, so a plain negative must still
        reach the range branch and name the floor rather than finiteness.
        """
        monkeypatch.setenv("ZERO_PROBE", "-10")
        with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
            assert core_mod._parse_positive_float_env("ZERO_PROBE", "60") == 60.0
        messages = [r.getMessage() for r in caplog.records]
        assert any("must be >=" in m for m in messages), messages


# --- structural guard: no sixth resolver may ship without the test --------

#: Every env-float resolver in the mesh package, as ``module::function``.
EXPECTED_ENV_FLOAT_RESOLVERS = frozenset(
    {
        "_zenoh_config.py::_float_env",
        "core.py::_parse_positive_float_env",
        "security.py::_env_pos_float",
        "session.py::hz_from_env",
        "transport/bridge_transport.py::_resolve_dedup_ttl",
    }
)


def _mesh_root() -> pathlib.Path:
    """Derive the scanned package from an imported symbol, not a path literal."""
    return pathlib.Path(inspect.getfile(core_mod)).parent


def _reads_the_environment(fn: ast.AST) -> bool:
    """True when *fn* really reads ``os.environ`` / ``os.getenv``.

    Structural rather than textual: the safety handlers mention ``os.getenv``
    in a comment explaining why they cache their knobs, and a text scan reads
    that as an env-float site.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if node.func.attr == "getenv" and isinstance(value, ast.Name) and value.id == "os":
                return True
            if node.func.attr == "get" and isinstance(value, ast.Attribute) and value.attr == "environ":
                return True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            return True
    return False


def _calls(fn: ast.AST, name: str) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == name:
                return True
    return False


def _scan(root: pathlib.Path) -> dict[str, bool]:
    """Map ``module::function`` to whether it tests finiteness."""
    found: dict[str, bool] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _calls(node, "float") and _reads_the_environment(node):
                found[f"{path.relative_to(root)}::{node.name}"] = _calls(node, "isfinite")
    return found


class TestEveryEnvFloatResolverTestsFiniteness:
    """A structural guard, because the divergence is what let this happen."""

    def test_the_scan_finds_exactly_the_known_resolvers(self):
        """Non-vacuity: a scan rooted elsewhere, or a detector that stopped
        matching, would report a clean sweep over nothing.
        """
        assert set(_scan(_mesh_root())) == set(EXPECTED_ENV_FLOAT_RESOLVERS)

    def test_no_resolver_omits_the_finiteness_test(self):
        adrift = sorted(name for name, guarded in _scan(_mesh_root()).items() if not guarded)
        assert adrift == [], f"env-float resolvers with no finiteness test: {adrift}"

    def test_the_scan_detects_a_planted_omission(self, tmp_path):
        planted = tmp_path / "planted.py"
        planted.write_text(
            'import os\n\n\ndef _resolve(name: str) -> float:\n    return float(os.getenv(name, "1"))\n',
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {"planted.py::_resolve": False}

    def test_the_scan_ignores_a_comment_mentioning_getenv(self, tmp_path):
        """The detector must not re-acquire the text scan's false positive."""
        planted = tmp_path / "commented.py"
        planted.write_text(
            "def _f(raw: str) -> float:\n"
            "    # Reading them per-use parsed os.getenv on every reference.\n"
            "    return float(raw)\n",
            encoding="utf-8",
        )
        assert _scan(tmp_path) == {}


@pytest.fixture(scope="module")
def collected_node_ids() -> list[str]:
    """This module's node ids, as a child process reports them.

    ``-q -q`` nets out the ``-v`` in the project's ``addopts`` so the output is
    the flat ``path::Class::test[id]`` form - the node id a caller pastes back
    to re-run one case, which is the thing whose reproducibility is under test.
    The two assertions are the non-vacuity guard: a collection that errored, or
    an output read that matched nothing, must not let the checks below pass by
    default.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(pathlib.Path(__file__).resolve()),
            "--collect-only",
            "-q",
            "-q",
            "--no-cov",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"collection failed:\n{proc.stdout}\n{proc.stderr}"
    ids = [line.strip() for line in proc.stdout.splitlines() if "::" in line]
    parametrized = [i for i in ids if TARGET_TEST in i]
    expected = len(NON_FINITE) * len(FLOAT_KNOBS)
    assert len(parametrized) == expected, f"read {len(parametrized)} ids for {TARGET_TEST}, expected {expected}"
    return ids


class TestTheNodeIdsAreReproducible:
    """This module's own node ids must survive a change of process.

    ``FLOAT_KNOBS`` carries resolver function objects, so ``ids=str``, the
    package's usual ``ids=repr``, or a lambda wrapping either all render one as
    ``<function _resume_backoff_s at 0x...>``. Measured over two collections of
    identical code, 27 of this module's 128 ids differed, so none of the 27
    could be pasted back to re-run one case and ``--last-failed`` could not
    match them across runs. Letting pytest's default apply resolves each
    function to its ``__name__`` instead; these tests pin that the ids stay
    address-free *and* keep naming the resolver, because an id generator that
    merely dropped the parameter would satisfy the first without the second.
    """

    def test_no_node_id_embeds_an_object_address(self, collected_node_ids):
        offenders = sorted(node_id for node_id in collected_node_ids if " at 0x" in node_id)
        assert offenders == [], (
            f"{len(offenders)} node id(s) embed a process-local address, so they change every "
            f"run and cannot be re-run; first: {offenders[0] if offenders else ''}"
        )

    def test_the_resolver_is_still_named_in_the_node_id(self, collected_node_ids):
        """Address-free is not enough. Without the resolver's name a failure
        reports an env var and a default with no clue which of the three
        functions produced it.
        """
        parametrized = [node_id for node_id in collected_node_ids if TARGET_TEST in node_id]
        for _env_var, resolver, _default in FLOAT_KNOBS:
            assert any(resolver.__name__ in node_id for node_id in parametrized), (
                f"{resolver.__name__} is named in no node id"
            )
