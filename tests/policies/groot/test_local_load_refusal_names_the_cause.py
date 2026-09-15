"""An in-process GR00T load refuses with the fact that stopped it.

``Gr00tPolicy(model_path=...)`` loads a checkpoint in the caller's own process,
which needs NVIDIA's Isaac-GR00T package. Three different facts can stop that,
and all three used to arrive as one sentence - ``ImportError("Isaac-GR00T not
installed. Use service mode (host/port).")`` - or as no sentence at all:

* **The caller's ``groot_version=`` names no release.** It selects a loader, so
  a value outside :data:`~strands_robots.utils.SUPPORTED_GROOT_VERSIONS` matched
  no dispatch branch and fell through to the "not installed" message. Measured
  against a gr00t package that *is* importable and auto-detected as ``n1.7``,
  ``groot_version="N1.7"`` reported the environment as lacking Isaac-GR00T: a
  false statement about the machine, answered with an install instruction for a
  package the caller already had, and never naming the parameter that caused it.
* **The package is absent, and the caller forced a release.** ``groot_version=``
  bypassed the check entirely: the forced spelling reached its loader's own
  ``from gr00t...`` line and surfaced ``ModuleNotFoundError: No module named
  'gr00t'``, while leaving it unset answered the identical missing package with
  the actionable message. The caller who supplied more information got the worse
  error.
* **The package is importable but its release is unidentifiable.** Reported as
  "not installed", which is false; ``groot_version=`` is what resolves it.

The message a present package earns matters here beyond wording, because
"install Isaac-GR00T" is the one instruction a ``strands-robots[all]`` caller
cannot follow. No extra declares ``gr00t`` (it installs from
``github.com/NVIDIA/Isaac-GR00T``), and it pins ``transformers==4.57.3`` while
lerobot needs ``transformers>=5``, so - in lerobot's own words - the two "cannot
be imported in the same Python process". Both routes that remain open are
reachable from a declared extra, so the refusal names them:
``create_policy("groot", host=..., port=...)`` for the Isaac-GR00T container, and
``create_policy("lerobot_local", policy_type="groot", ...)`` for lerobot's own
GR00T N1.7 in-process.
"""

import ast
import inspect

import numpy as np
import pytest

msgpack = pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[groot-service]'")
zmq = pytest.importorskip("zmq", reason="zmq not installed - pip install 'strands-robots[groot-service]'")

import strands_robots.policies.groot.policy as policy_module  # noqa: E402
from strands_robots.policies.groot import Gr00tPolicy  # noqa: E402
from strands_robots.utils import SUPPORTED_GROOT_VERSIONS, groot_version_error  # noqa: E402

#: Values that are not a release this policy has a loader for.
#:
#: ``""`` is what an unset environment variable interpolates to; ``"N1.7"`` and
#: ``"1.7"`` are the near-misses of the accepted spellings, and the one that made
#: an installed package report itself absent.
NOT_A_RELEASE = ("n1.8", "N1.7", "1.7", "n17", "", " n1.7", "n1.7 ", 17, 1.7, True, ["n1.7"])


@pytest.fixture
def no_isaac_groot(monkeypatch):
    """Detection finds nothing - the state of every install that declares gr00t nowhere."""
    monkeypatch.setattr(policy_module, "_detect_groot_version", lambda **kw: None)


@pytest.fixture
def isaac_groot_n17(monkeypatch):
    """Detection identifies an importable n1.7, without importing anything."""
    monkeypatch.setattr(policy_module, "_detect_groot_version", lambda **kw: "n1.7")


def _local(**kwargs):
    return Gr00tPolicy(data_config="so100_dualcam", model_path="/checkpoint", **kwargs)


class TestTheVersionDomainIsGradedNotDispatched:
    """A ``groot_version`` naming no release is refused by name."""

    @pytest.mark.parametrize("value", NOT_A_RELEASE)
    def test_a_value_naming_no_release_is_refused(self, value, isaac_groot_n17):
        with pytest.raises(ValueError) as excinfo:
            _local(groot_version=value)
        message = str(excinfo.value)
        assert "groot_version" in message, "the refusal must name the knob that caused it"
        assert "Gr00tPolicy" in message
        for release in SUPPORTED_GROOT_VERSIONS:
            assert release in message, "the refusal states the domain it graded against"
        message.encode("ascii")

    @pytest.mark.parametrize("value", NOT_A_RELEASE)
    def test_an_installed_package_is_not_reported_absent(self, value, isaac_groot_n17):
        """The headline: gr00t importable and detected, and the message said otherwise."""
        with pytest.raises(ValueError) as excinfo:
            _local(groot_version=value)
        assert "not installed" not in str(excinfo.value)
        assert "not importable" not in str(excinfo.value)

    @pytest.mark.parametrize("release", SUPPORTED_GROOT_VERSIONS)
    def test_a_release_with_a_loader_reaches_it(self, release, isaac_groot_n17, monkeypatch):
        loaded: list[tuple] = []
        monkeypatch.setattr(
            Gr00tPolicy, "_load_local_policy", lambda self, *a: loaded.append((self._groot_version, *a))
        )
        _local(groot_version=release)
        assert loaded == [(release, "/checkpoint", "NEW_EMBODIMENT", "cuda")]

    def test_an_omitted_version_auto_detects(self, isaac_groot_n17, monkeypatch):
        """``None`` is the not-supplied sentinel, so detection still decides."""
        monkeypatch.setattr(Gr00tPolicy, "_load_local_policy", lambda self, *a: None)
        assert _local(groot_version=None)._groot_version == "n1.7"

    def test_the_guard_is_the_shared_domain(self):
        assert groot_version_error(None, "groot_version", "Gr00tPolicy") is None
        assert all(groot_version_error(r, "groot_version", "Gr00tPolicy") is None for r in SUPPORTED_GROOT_VERSIONS)
        assert groot_version_error("N1.7", "groot_version", "Gr00tPolicy") is not None


class TestTheVersionIsGradedInBothModes:
    """Service mode reads the selector too, so it is refused there as well.

    ``_build_service_observation`` chooses the wire shape from it - ``n1.7``
    adds the time axis an N1.7 server requires - and its docstring tells
    service callers targeting that server to pass ``groot_version="n1.7"``.
    The guard used to be scoped to the local branch on the claim that service
    mode "never reads it", so ``groot_version="N1.7"`` was accepted and sent
    the legacy ``(B, ...)`` tensors: a server-side shape error for a typo the
    constructor could have named.
    """

    @pytest.mark.parametrize("value", NOT_A_RELEASE)
    def test_service_mode_refuses_a_value_that_names_no_release(self, value, no_isaac_groot):
        with pytest.raises(ValueError, match="groot_version") as excinfo:
            Gr00tPolicy(data_config="so100_dualcam", port=5555, groot_version=value)
        assert str(excinfo.value) == groot_version_error(value, "groot_version", "Gr00tPolicy")

    def test_a_forced_release_selects_the_n17_wire_shape(self, no_isaac_groot):
        """The value the guard protects: ``n1.7`` is the only spelling that adds the time axis."""
        obs = {"front": np.zeros((8, 8, 3), np.uint8), "single_arm": np.zeros(5)}
        wire = Gr00tPolicy(data_config="so100_dualcam", port=5555, groot_version="n1.7")._build_service_observation(
            obs, "pick"
        )
        assert wire["video.front"].shape == (1, 1, 8, 8, 3)
        assert wire["state.single_arm"].shape == (1, 1, 5)

    def test_the_parameter_entry_names_the_service_reader(self):
        """The ``Args:`` entry is the surface a caller reads for this parameter.

        It stated the value is "Only read in local mode, so it is validated only
        on the branch that reads it, as ``port`` is" - the claim that scoped the
        guard, and the one that told a service caller their spelling would go
        ungraded. Both halves stopped holding once the wire shape was chosen
        from it, so the entry names the service reader too. Pinned because
        nothing else grades the claim: the cell above proves service mode reads
        the value, and this keeps the documentation of it from drifting back to
        local-only while the guard stays in both modes.
        """
        doc = " ".join((Gr00tPolicy.__doc__ or "").split())
        start = doc.index("groot_version: Force one of")
        entry = doc[start : doc.index("strict:", start)]
        assert "Only read in local mode" not in entry, "the falsified claim is back in the entry a caller reads"
        assert "service mode" in entry, "the entry must name the reader the local-only claim omitted"


class TestAnAbsentPackageIsOneAnswerForEverySpellingOfTheRequest:
    """Forcing a release used to skip the check and raise ``ModuleNotFoundError``."""

    @pytest.mark.parametrize("groot_version", [None, *SUPPORTED_GROOT_VERSIONS])
    def test_the_actionable_message_reaches_the_forced_path_too(self, groot_version, no_isaac_groot):
        with pytest.raises(ImportError) as excinfo:
            _local(groot_version=groot_version)
        message = str(excinfo.value)
        assert "No module named" not in message, "a raw import failure names neither the cause nor a way on"
        assert "Isaac-GR00T" in message
        assert "model_path" in message, "the refusal names the parameter that selected in-process loading"
        message.encode("ascii")

    @pytest.mark.parametrize("groot_version", [None, *SUPPORTED_GROOT_VERSIONS])
    def test_both_open_routes_are_named(self, groot_version, no_isaac_groot):
        """An install instruction is the one thing an ``[all]`` caller cannot act on."""
        with pytest.raises(ImportError) as excinfo:
            _local(groot_version=groot_version)
        message = str(excinfo.value)
        assert 'create_policy("groot", host=..., port=...)' in message
        assert "lerobot_local" in message and 'policy_type="groot"' in message
        assert "strands-robots[groot-service]" in message
        assert "strands-robots[lerobot]" in message


class TestAnUnidentifiedReleaseIsNotAnAbsentPackage:
    """``_load_local_policy``'s own contract: the release, not the package, is missing."""

    def test_the_message_does_not_claim_the_package_is_absent(self):
        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        policy._groot_version = None
        with pytest.raises(ImportError) as excinfo:
            policy._load_local_policy("/checkpoint", "NEW_EMBODIMENT", "cuda")
        message = str(excinfo.value)
        assert "not installed" not in message
        assert "could not be identified" in message
        assert "groot_version" in message, "the parameter that resolves it"
        for release in SUPPORTED_GROOT_VERSIONS:
            assert release in message
        assert "lerobot_local" in message
        message.encode("ascii")


class TestTheDomainAndTheDispatchCannotDrift:
    """The graded domain is the set of loaders, derived from the dispatch itself.

    A release added to one and not the other reopens the defect from either
    side: a loader the domain refuses is unreachable, and a domain entry with no
    loader falls through to the unidentified-release message.
    """

    def test_every_dispatch_branch_is_a_graded_release(self):
        tree = ast.parse(inspect.getsource(Gr00tPolicy._load_local_policy).strip())
        compared = {
            node.comparators[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "_groot_version"
            and isinstance(node.comparators[0], ast.Constant)
        }
        assert compared == set(SUPPORTED_GROOT_VERSIONS)
