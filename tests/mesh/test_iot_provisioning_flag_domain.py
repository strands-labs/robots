"""A provisioning posture flag is checked, not read by truthiness.

``mesh.iot`` has exactly two public entry points carrying ``bool``-annotated
flags, and every one of them selects a *posture* rather than scaling a quantity:

* :func:`~strands_robots.mesh.iot.provision.provision_robot`'s
  ``allow_estop_publish`` chooses between the ``strands-robot`` policy, which
  grants ``AllowSafetyEstop``, and ``strands-robot-no-estop``, which withholds
  it. It is a security *opt-out*.
* :func:`~strands_robots.mesh.iot.bootstrap.bootstrap_account`'s ``confirm``
  gates a destructive account-wide create, ``dry_run`` selects preview mode and
  ``force_update`` overwrites an existing E-stop Lambda.

Read by truthiness, every non-boolean spelling of *off* selects the permissive
branch: ``"false"``, ``"no"``, ``"off"`` and ``"0"`` are all truthy, so the
opt-out fails open and the confirmation gate confirms. These tests pin that each
flag is now refused instead, before any AWS call, and that the two usable
postures are unchanged.

The fake IoT client here records what it was asked to create rather than
asserting on a mock, so "the refused call provisioned nothing" and "the deny
posture really omits the grant" are both direct observations.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import strands_robots.mesh.iot.bootstrap as bootstrap_mod
import strands_robots.mesh.iot.provision as provision_mod
from strands_robots.utils import boolean_flag_error

ESTOP_SID = "AllowSafetyEstop"

#: Values no posture flag can be built from. Each is truthy (and would select
#: the permissive branch) unless marked otherwise in the comment.
UNUSABLE: list[Any] = [
    "false",  # truthy
    "no",  # truthy
    "off",  # truthy
    "0",  # truthy
    "yes",
    "True",
    0,  # falsy, but never a declared spelling
    1,
    2.5,
    math.nan,  # truthy
    None,  # falsy
    [],  # falsy
    ["false"],
    {},
    object(),
]

#: Booleans every flag must keep honouring, including the numpy spellings a
#: comparison produces.
USABLE: list[Any] = [True, False, np.True_, np.False_, np.array(True), np.array(False)]


def _case_id(value: Any) -> str:
    """A label for a table row that is the same in every process.

    ``repr`` reads well for the literals above, but ``object()`` inherits the
    default ``__repr__``, which prints the instance's address. That address
    differs per interpreter, so the row's test ID differed between pytest-xdist
    workers and the whole suite failed to collect in parallel - "Different tests
    were collected between gw0 and gw1", before a single test ran. Truncating
    the text did not hide it: ``<object object at 0x7f5c`` is 24 characters, so
    ``repr(v)[:24]`` kept exactly the prefix that moves. That row is labelled by
    type instead, the spelling
    :mod:`tests.drivers.test_telemetry_coercion_refuses_the_same_non_readings`
    already uses.
    """
    text = repr(value)
    return type(value).__name__ if " at 0x" in text else text[:24]


class _NotFound(Exception):
    """Stands in for ``iot.exceptions.ResourceNotFoundException``."""


class _RecordingIot:
    """Minimal AWS IoT stand-in that records everything it is asked to create."""

    class _Exceptions:
        ResourceNotFoundException = _NotFound

    class _Meta:
        region_name = "us-west-2"

    exceptions = _Exceptions()
    meta = _Meta()

    def __init__(self) -> None:
        self.created_things: list[str] = []
        self.created_policies: dict[str, dict[str, Any]] = {}
        self.attached_policies: list[str] = []
        self.issued_certs = 0

    def describe_thing(self, **_kw: Any) -> Any:
        raise _NotFound()

    def create_thing(self, thingName: str, **_kw: Any) -> dict[str, str]:  # noqa: N803 - boto3 casing
        self.created_things.append(thingName)
        return {"thingArn": f"arn:aws:iot:us-west-2:1:thing/{thingName}"}

    def get_policy(self, **_kw: Any) -> Any:
        raise _NotFound()

    def create_policy(self, policyName: str, policyDocument: str, **_kw: Any) -> dict[str, str]:  # noqa: N803
        self.created_policies[policyName] = json.loads(policyDocument)
        return {"policyArn": f"arn:aws:iot:us-west-2:1:policy/{policyName}"}

    def list_thing_principals(self, **_kw: Any) -> dict[str, list[str]]:
        return {"principals": []}

    def create_keys_and_certificate(self, **_kw: Any) -> dict[str, Any]:
        self.issued_certs += 1
        return {
            "certificateArn": "arn:aws:iot:us-west-2:1:cert/abc",
            "certificateId": "abc",
            "certificatePem": "PEM",
            "keyPair": {"PrivateKey": "KEY"},
        }

    def attach_policy(self, policyName: str, target: str) -> None:  # noqa: N803
        self.attached_policies.append(policyName)

    def attach_thing_principal(self, **_kw: Any) -> None:
        return None

    def describe_endpoint(self, **_kw: Any) -> dict[str, str]:
        return {"endpointAddress": "x.iot.us-west-2.amazonaws.com"}

    def touched(self) -> bool:
        """True when any resource was created, attached or issued."""
        return bool(self.created_things or self.created_policies or self.attached_policies or self.issued_certs)


@pytest.fixture
def iot(monkeypatch: pytest.MonkeyPatch) -> _RecordingIot:
    """Wire a recording IoT client in and stub the pinned-CA download."""
    client = _RecordingIot()

    class _Boto3:
        @staticmethod
        def client(_name: str, region_name: str | None = None) -> _RecordingIot:
            return client

    monkeypatch.setattr(provision_mod, "_require_boto3", lambda: _Boto3())
    monkeypatch.setattr(provision_mod, "_ensure_ca", lambda ca_path: ca_path.write_text("CA"))
    return client


def _provision(cert_dir: Path, **kwargs: Any) -> Any:
    """Call ``provision_robot`` through one funnel.

    Splatted so a deliberately off-type flag reaches the runtime guard as an
    operator would supply it, rather than being reported by the type checker.
    """
    return provision_mod.provision_robot("probe-bot", cert_dir=cert_dir, **kwargs)


def _bootstrap(**kwargs: Any) -> Any:
    """Call ``bootstrap_account`` through one funnel, for the same reason."""
    return bootstrap_mod.bootstrap_account(**kwargs)


class TestTheDomain:
    """``boolean_flag_error`` accepts a boolean and nothing else."""

    @pytest.mark.parametrize("value", USABLE, ids=[repr(v) for v in USABLE])
    def test_a_boolean_is_accepted(self, value: Any) -> None:
        assert boolean_flag_error(value, "confirm", "ctx") is None

    @pytest.mark.parametrize("value", UNUSABLE, ids=_case_id)
    def test_a_non_boolean_is_refused(self, value: Any) -> None:
        assert boolean_flag_error(value, "confirm", "ctx") is not None

    def test_the_message_names_the_context_the_parameter_and_the_value(self) -> None:
        msg = boolean_flag_error("false", "allow_estop_publish", "provision_robot")
        assert msg is not None
        assert msg.startswith("provision_robot: allow_estop_publish must be a boolean")
        assert "'false'" in msg

    def test_it_is_the_inverse_of_the_numeric_domains(self) -> None:
        """A boolean is exactly what the numeric domains refuse and this requires."""
        from strands_robots.utils import positive_finite_number_error

        assert positive_finite_number_error(True, "hz", "ctx") is not None
        assert boolean_flag_error(True, "confirm", "ctx") is None
        assert positive_finite_number_error(1.5, "hz", "ctx") is None
        assert boolean_flag_error(1.5, "confirm", "ctx") is not None


class TestProvisionRobotHonoursBothPostures:
    """The two declared spellings still select the two policies they always did."""

    def test_true_grants_the_estop_publish_statement(self, iot: _RecordingIot, tmp_path: Path) -> None:
        result = _provision(tmp_path, allow_estop_publish=True)

        assert result.policy_name == provision_mod.ROBOT_POLICY_NAME
        sids = [st.get("Sid") for st in iot.created_policies[result.policy_name]["Statement"]]
        assert ESTOP_SID in sids
        assert iot.attached_policies == [provision_mod.ROBOT_POLICY_NAME]

    def test_false_withholds_the_estop_publish_statement(self, iot: _RecordingIot, tmp_path: Path) -> None:
        result = _provision(tmp_path, allow_estop_publish=False)

        assert result.policy_name == provision_mod.ROBOT_NO_ESTOP_POLICY_NAME
        sids = [st.get("Sid") for st in iot.created_policies[result.policy_name]["Statement"]]
        assert ESTOP_SID not in sids
        assert iot.attached_policies == [provision_mod.ROBOT_NO_ESTOP_POLICY_NAME]

    def test_the_default_is_the_grant_bearing_policy(self, iot: _RecordingIot, tmp_path: Path) -> None:
        result = _provision(tmp_path)
        assert result.policy_name == provision_mod.ROBOT_POLICY_NAME

    @pytest.mark.parametrize("value", [np.False_, np.array(False)], ids=["np.False_", "np.array(False)"])
    def test_a_numpy_false_selects_the_deny_posture_as_a_real_bool(
        self, iot: _RecordingIot, tmp_path: Path, value: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The normalisation is load-bearing: ``_robot_policy_doc`` is annotated ``bool``."""
        seen: list[Any] = []
        real = provision_mod._robot_policy_doc

        def _spy(*, allow_estop_publish: Any) -> Any:
            seen.append(allow_estop_publish)
            return real(allow_estop_publish=allow_estop_publish)

        monkeypatch.setattr(provision_mod, "_robot_policy_doc", _spy)

        result = _provision(tmp_path, allow_estop_publish=value)

        assert result.policy_name == provision_mod.ROBOT_NO_ESTOP_POLICY_NAME
        assert seen == [False]
        assert type(seen[0]) is bool


class TestProvisionRobotRefusesANonBooleanOptOut:
    """A truthy spelling of *off* must not resolve to the grant-bearing policy."""

    @pytest.mark.parametrize("value", UNUSABLE, ids=_case_id)
    def test_it_is_refused(self, iot: _RecordingIot, tmp_path: Path, value: Any) -> None:
        with pytest.raises(ValueError, match=r"allow_estop_publish must be a boolean"):
            _provision(tmp_path, allow_estop_publish=value)

    @pytest.mark.parametrize("value", ["false", "no", "off", "0"], ids=["false", "no", "off", "0"])
    def test_the_refused_call_provisions_nothing(self, iot: _RecordingIot, tmp_path: Path, value: Any) -> None:
        """No Thing, no policy, no certificate - the guard precedes every AWS call."""
        with pytest.raises(ValueError):
            _provision(tmp_path, allow_estop_publish=value)

        assert not iot.touched()
        assert list(tmp_path.glob("*.pem")) == []
        assert list(tmp_path.glob("*.key")) == []

    def test_the_refusal_does_not_need_boto3(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Placed ahead of the optional-dependency probe, so it reports the same either way."""

        def _no_boto3() -> Any:
            raise AssertionError("the refused call resolved boto3")

        monkeypatch.setattr(provision_mod, "_require_boto3", _no_boto3)

        with pytest.raises(ValueError, match=r"allow_estop_publish must be a boolean"):
            _provision(tmp_path, allow_estop_publish="false")


class TestBootstrapAccountRefusesANonBooleanFlag:
    """The confirmation gate must read a real boolean, not a truthy string."""

    @pytest.fixture(autouse=True)
    def _no_aws(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _no_boto3() -> Any:
            raise AssertionError("a refused call resolved boto3")

        monkeypatch.setattr(bootstrap_mod, "_require_boto3", _no_boto3)

    @pytest.mark.parametrize("flag", ["confirm", "dry_run", "force_update"])
    @pytest.mark.parametrize("value", ["false", "no", 1, math.nan, None], ids=["false", "no", "1", "nan", "None"])
    def test_each_flag_is_refused(self, flag: str, value: Any) -> None:
        with pytest.raises(ValueError, match=rf"{flag} must be a boolean"):
            _bootstrap(**{flag: value})

    def test_a_truthy_confirm_no_longer_satisfies_the_gate(self) -> None:
        """Pre-fix ``not confirm`` was False for ``"false"``, so the create path ran."""
        with pytest.raises(ValueError, match=r"confirm must be a boolean"):
            _bootstrap(confirm="false", dry_run=False)

    def test_a_truthy_dry_run_is_refused_instead_of_silently_previewing(self) -> None:
        """``dry_run="false"`` reads as a request to leave preview and stayed in it."""
        with pytest.raises(ValueError, match=r"dry_run must be a boolean"):
            _bootstrap(dry_run="false", confirm=False)

    def test_the_documented_refusal_is_unchanged(self) -> None:
        with pytest.raises(ValueError, match=r"creates persistent AWS resources"):
            _bootstrap(confirm=False, dry_run=False)

    def test_the_flag_check_precedes_the_gate(self) -> None:
        """A bad flag reports the flag, not the gate, even when the gate would also fire."""
        with pytest.raises(ValueError) as excinfo:
            _bootstrap(confirm="false", dry_run="false")
        assert "must be a boolean" in str(excinfo.value)
        assert "creates persistent AWS resources" not in str(excinfo.value)


class TestBootstrapAccountStillPreviews:
    """The usable postures keep working, including the default dry run."""

    def test_the_default_previews_without_creating(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class _Sts:
            class _Meta:
                region_name = "us-west-2"

            meta = _Meta()

            def get_caller_identity(self) -> dict[str, str]:
                return {"Account": "111122223333"}

        class _Boto3:
            @staticmethod
            def client(name: str, region_name: str | None = None) -> Any:
                if name == "sts":
                    return _Sts()
                raise AssertionError(f"the preview created a {name} client")

        monkeypatch.setattr(bootstrap_mod, "_require_boto3", lambda: _Boto3())

        _bootstrap()

        # The preview writes to stderr (see the ``file=sys.stderr`` in bootstrap).
        assert "[dry_run]" in capsys.readouterr().err


class TestEveryPostureFlagRoutesThroughTheDomain:
    """A future flag in this package cannot ship reading itself by truthiness."""

    @staticmethod
    def _flag_surfaces(src: str) -> dict[str, list[str]]:
        """Public top-level functions in *src* mapped to their ``bool`` parameters."""
        found: dict[str, list[str]] = {}
        for node in ast.parse(src).body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            args = node.args
            flags = [
                a.arg
                for a in (args.posonlyargs + args.args + args.kwonlyargs)
                if a.annotation is not None and ast.unparse(a.annotation) == "bool"
            ]
            if flags:
                found[node.name] = flags
        return found

    @staticmethod
    def _calls_the_domain(src: str, name: str) -> bool:
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "boolean_flag_error"
            for n in ast.walk(fn)
        )

    @classmethod
    def _package_sources(cls) -> dict[str, str]:
        root = Path(inspect.getfile(provision_mod)).parent
        return {p.name: p.read_text(encoding="utf-8") for p in sorted(root.glob("*.py"))}

    def test_the_expected_surfaces_are_discovered(self) -> None:
        """Non-vacuity: a scan root resolving elsewhere would report nothing."""
        found = {
            f"{name}::{fn}": flags
            for name, src in self._package_sources().items()
            for fn, flags in self._flag_surfaces(src).items()
        }
        assert found == {
            "bootstrap.py::bootstrap_account": ["confirm", "dry_run", "force_update"],
            "provision.py::provision_robot": ["allow_estop_publish"],
        }

    def test_no_surface_reads_a_posture_flag_by_truthiness(self) -> None:
        adrift = {
            f"{name}::{fn}": flags
            for name, src in self._package_sources().items()
            for fn, flags in self._flag_surfaces(src).items()
            if not self._calls_the_domain(src, fn)
        }
        assert adrift == {}, f"public mesh.iot flags not routed through boolean_flag_error: {adrift}"

    def test_the_scanner_detects_a_planted_surface(self) -> None:
        """A guard that matched nothing would report a clean package either way."""
        planted = "def provision_thing(*, replace_existing: bool = False) -> None:\n    return None\n"
        assert self._flag_surfaces(planted) == {"provision_thing": ["replace_existing"]}
        assert not self._calls_the_domain(planted, "provision_thing")


class TestTheFlagIsDocumented:
    """The opt-out is discoverable: a caller can look up that it takes a boolean."""

    def test_provision_robot_documents_allow_estop_publish(self) -> None:
        doc = inspect.getdoc(provision_mod.provision_robot) or ""
        entries = re.search(r"^Args:\n(.*?)(?=\n\S|\Z)", doc, re.S | re.M)
        assert entries is not None
        assert "allow_estop_publish:" in entries.group(1)
        assert "boolean" in entries.group(1)

    def test_provision_robot_documents_the_refusal(self) -> None:
        doc = inspect.getdoc(provision_mod.provision_robot) or ""
        raises = re.search(r"^Raises:\n(.*?)(?=\n\S|\Z)", doc, re.S | re.M)
        assert raises is not None
        assert "ValueError" in raises.group(1)
        assert "allow_estop_publish" in raises.group(1)
