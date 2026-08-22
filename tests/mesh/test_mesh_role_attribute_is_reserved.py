"""``strands-mesh-role`` is the provisioners' routing key, not a caller label.

``provision_robot`` / ``provision_operator`` inject that Thing attribute for
their own fleet routing: the e-stop fan-out Lambda enumerates every Thing and
publishes ``{"action": "stop"}`` only to those whose value is exactly
``"robot"``. Merging a caller entry of the same name - which is what
``setdefault`` did - let the caller's value win over the module's, and a robot
carrying any other value is skipped by the fan-out with no error on either
side: ``provision_robot`` reports success, the Thing exists, its certificate
works, it obeys ``cmd`` - and it never receives a fleet-wide stop.

The consumer premise is measured here rather than asserted, by driving the
shipped Lambda source over a Things page.
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Spelled out rather than imported so this module collects against a tree that
# has no reserved-key constant, which keeps every failure below behavioural.
RESERVED = "strands-mesh-role"


@pytest.fixture
def iot_client() -> Any:
    """A boto3 IoT client double whose calls all take the create path."""
    iot = MagicMock()
    iot.meta.region_name = "us-west-2"

    class _NotFound(Exception):
        pass

    iot.exceptions = MagicMock()
    iot.exceptions.ResourceNotFoundException = _NotFound
    iot.describe_thing.side_effect = _NotFound("absent")
    iot.create_thing.return_value = {"thingArn": "arn:aws:iot:us-west-2:1:thing/t"}
    iot.get_policy.side_effect = _NotFound("absent")
    iot.create_policy.return_value = {"policyArn": "arn:aws:iot:us-west-2:1:policy/p"}
    iot.create_keys_and_certificate.return_value = {
        "certificateArn": "arn:aws:iot:us-west-2:1:cert/c",
        "certificateId": "c",
        "certificatePem": "PEM",
        "keyPair": {"PrivateKey": "KEY"},
    }
    iot.list_thing_principals.return_value = {"principals": []}
    iot.describe_endpoint.return_value = {"endpointAddress": "x.iot.us-west-2.amazonaws.com"}
    return iot


@pytest.fixture
def provisioner(monkeypatch: pytest.MonkeyPatch, iot_client: Any) -> Any:
    """The provision module with boto3 and the CA download stood in for."""
    from strands_robots.mesh.iot import provision

    monkeypatch.setattr(provision, "_require_boto3", lambda: MagicMock(client=lambda *a, **k: iot_client))
    monkeypatch.setattr(provision, "_ensure_ca", lambda path: None)
    return provision


def _sent_attributes(iot_client: Any) -> dict[str, str]:
    """The attributes that reached ``CreateThing``."""
    kwargs = iot_client.create_thing.call_args.kwargs
    return dict(kwargs.get("attributePayload", {}).get("attributes", {}))


def _fanout_targets(monkeypatch: pytest.MonkeyPatch, things: list[dict[str, Any]]) -> list[str]:
    """Run the shipped e-stop fan-out over *things*; return the topics stopped.

    Executes ``bootstrap._ESTOP_LAMBDA_SOURCE`` verbatim, so the routing rule
    under test is the one that is deployed rather than a restatement of it,
    with the boto3 double installed in ``sys.modules`` because that is where
    the source resolves the module.
    """
    from strands_robots.mesh.iot import bootstrap

    published: list[str] = []
    iot = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"things": things}]
    iot.get_paginator.return_value = paginator
    iot_data = MagicMock()
    iot_data.publish.side_effect = lambda **kw: published.append(kw["topic"])
    ddb = MagicMock()
    ddb.exceptions.ConditionalCheckFailedException = type("CCF", (Exception,), {})

    clients = {"iot": iot, "iot-data": iot_data, "dynamodb": ddb}
    # The Lambda source does its own ``import boto3``, so it resolves the
    # module through ``sys.modules`` at call time rather than through any
    # binding this file holds. Install the double there - the same way
    # ``test_iot_provisioning_hook.py`` drives the sibling hook source in
    # this module - so a patch on a module-level binding cannot be left
    # pointing at an object the source never consults, which would reach
    # the real SDK.
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda name, *a, **k: clients[name]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    namespace: dict[str, Any] = {}
    exec(bootstrap._ESTOP_LAMBDA_SOURCE, namespace)  # noqa: S102 - deployed source under test
    namespace["lambda_handler"](
        {"peer_id": "ops", "t": "2026-01-01T00:00:00Z"},
        types.SimpleNamespace(aws_request_id="req-1"),
    )
    return published


class TestTheFanoutRoutesOnThisAttribute:
    """The premise: the attribute decides whether a stop reaches a Thing."""

    def test_only_a_robot_labelled_thing_is_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        things = [
            {"thingName": "strands-labelled", "attributes": {RESERVED: "robot"}},
            {"thingName": "strands-relabelled", "attributes": {RESERVED: "so101-arm"}},
            {"thingName": "strands-unlabelled", "attributes": {}},
        ]

        topics = _fanout_targets(monkeypatch, things)

        assert "strands/strands-labelled/cmd" in topics, (
            f"premise: a robot-labelled Thing must be stopped, got {topics}"
        )
        assert "strands/strands-relabelled/cmd" not in topics, (
            "premise: the fan-out skips a Thing whose role is not 'robot', so the "
            f"attribute is what decides reachability, got {topics}"
        )
        assert "strands/strands-unlabelled/cmd" not in topics
        assert len(topics) == 1

    def test_the_fanout_reports_success_while_skipping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A skipped robot is not an error anywhere - hence the refusal upstream."""
        topics = _fanout_targets(
            monkeypatch, [{"thingName": "strands-relabelled", "attributes": {RESERVED: "so101-arm"}}]
        )

        assert topics == []


class TestTheReservedKeyIsRefused:
    """A caller may not supply the key the fan-out routes on."""

    def test_provision_robot_refuses_the_reserved_key(self, provisioner: Any, iot_client: Any, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as caught:
            provisioner.provision_robot("strands-arm-01", cert_dir=tmp_path, attributes={RESERVED: "so101-arm"})

        message = str(caught.value)
        assert RESERVED in message, message
        assert "stop" in message.lower(), f"the refusal must say what the key decides, got {message!r}"
        assert iot_client.create_thing.call_count == 0, "a refused call must provision nothing"

    def test_provision_operator_refuses_the_reserved_key(
        self, provisioner: Any, iot_client: Any, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError) as caught:
            provisioner.provision_operator("strands-ops-01", cert_dir=tmp_path, attributes={RESERVED: "robot"})

        assert RESERVED in str(caught.value)
        assert iot_client.create_thing.call_count == 0

    @pytest.mark.parametrize("entrypoint", ["provision_robot", "provision_operator"])
    def test_the_refusal_lands_before_boto3_is_resolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entrypoint: str
    ) -> None:
        """So the verdict is the same with and without boto3 installed."""
        from strands_robots.mesh.iot import provision

        def _boom() -> Any:
            raise AssertionError("boto3 was resolved before the attributes were checked")

        monkeypatch.setattr(provision, "_require_boto3", _boom)

        with pytest.raises(ValueError) as caught:
            getattr(provision, entrypoint)("strands-thing-01", cert_dir=tmp_path, attributes={RESERVED: "robot"})

        assert RESERVED in str(caught.value)

    def test_the_remedy_the_refusal_offers_is_accepted(self, provisioner: Any, iot_client: Any, tmp_path: Path) -> None:
        """Dropping the key - what the message says to do - provisions."""
        supplied = {RESERVED: "so101-arm", "hw": "so100"}
        with pytest.raises(ValueError):
            provisioner.provision_robot("strands-arm-01", cert_dir=tmp_path, attributes=supplied)

        remaining = {k: v for k, v in supplied.items() if k != RESERVED}
        provisioner.provision_robot("strands-arm-01", cert_dir=tmp_path, attributes=remaining)

        sent = _sent_attributes(iot_client)
        assert sent == {"hw": "so100", RESERVED: "robot"}


class TestTheModuleStillOwnsTheLabel:
    """Controls: the value this module injects is unchanged and unconditional."""

    def test_a_robot_is_labelled_robot(self, provisioner: Any, iot_client: Any, tmp_path: Path) -> None:
        provisioner.provision_robot("strands-arm-01", cert_dir=tmp_path)

        assert _sent_attributes(iot_client)[RESERVED] == "robot"

    def test_an_operator_is_labelled_operator(self, provisioner: Any, iot_client: Any, tmp_path: Path) -> None:
        provisioner.provision_operator("strands-ops-01", cert_dir=tmp_path)

        assert _sent_attributes(iot_client)[RESERVED] == "operator"

    def test_caller_attributes_are_kept_alongside_the_label(
        self, provisioner: Any, iot_client: Any, tmp_path: Path
    ) -> None:
        provisioner.provision_robot("strands-arm-01", cert_dir=tmp_path, attributes={"hw": "so100", "site": "lab"})

        sent = _sent_attributes(iot_client)
        assert sent == {"hw": "so100", "site": "lab", RESERVED: "robot"}

    def test_a_labelled_robot_is_reached_by_the_fanout(
        self, provisioner: Any, iot_client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: what provisioning writes is what the fan-out routes on."""
        provisioner.provision_robot("strands-arm-01", cert_dir=tmp_path)

        sent = _sent_attributes(iot_client)
        topics = _fanout_targets(monkeypatch, [{"thingName": "strands-arm-01", "attributes": sent}])

        assert topics == ["strands/strands-arm-01/cmd"]


class TestTheDocumentedBudgetMatchesWhatIsSpent:
    """The caller's key budget accounts for the one this module spends."""

    def test_the_documented_budget_is_the_caller_budget(self) -> None:
        from strands_robots.mesh.iot import provision

        doc = provision.provision_robot.__doc__ or ""
        entry = doc.split("attributes:", 1)[1].split("allow_estop_publish:", 1)[0]
        entry = " ".join(entry.split())

        assert "three" in entry, f"the AWS per-Thing attribute limit is not stated: {entry!r}"
        assert "two for the caller" in entry, (
            "this function spends one of the three attributes on the routing key, "
            f"so the caller's budget is two, not three: {entry!r}"
        )
        assert "one value, not the dict" in entry, (
            f"AWS bounds 800 characters per attribute VALUE, not per dict: {entry!r}"
        )

    def test_the_caller_budget_is_what_reaches_aws(self, provisioner: Any, iot_client: Any, tmp_path: Path) -> None:
        """Two caller keys plus the injected one is AWS's three."""
        provisioner.provision_robot("strands-arm-01", cert_dir=tmp_path, attributes={"hw": "so100", "site": "lab"})

        assert len(_sent_attributes(iot_client)) == 3

    def test_the_reserved_key_is_named_in_both_docstrings(self) -> None:
        from strands_robots.mesh.iot import provision

        for fn in (provision.provision_robot, provision.provision_operator):
            doc = fn.__doc__ or ""
            assert RESERVED in doc, (
                f"{fn.__name__} refuses {RESERVED!r} but never names it, so a caller "
                "has no way to learn the key is reserved"
            )


class TestTheHelperIsTotal:
    """A non-mapping is left to the caller's own shape refusal downstream."""

    @pytest.mark.parametrize("value", [None, {}, {"hw": "so100"}, ["a", "b"], 5, "text"])
    def test_only_a_mapping_claiming_the_key_is_refused(self, value: Any) -> None:
        from strands_robots.mesh.iot.provision import _reserved_attribute_error

        assert _reserved_attribute_error(value) is None

    def test_the_constant_is_the_spelling_this_module_grades(self) -> None:
        from strands_robots.mesh.iot.provision import _MESH_ROLE_ATTRIBUTE

        assert _MESH_ROLE_ATTRIBUTE == RESERVED

    def test_a_mapping_claiming_the_key_is_refused_whatever_the_value(self) -> None:
        from strands_robots.mesh.iot.provision import _reserved_attribute_error

        for value in ("robot", "operator", "", "so101-arm"):
            assert _reserved_attribute_error({RESERVED: value}) is not None


class TestTheDoubleIsInstalledWhereTheSourceLooks:
    """Pin the seam, because a binding patch passes every test above.

    The Lambda source resolves ``boto3`` through ``sys.modules`` at call time
    (it carries its own ``import boto3``), so patching a module-level binding
    this file holds only works while nothing has replaced that entry. Any
    sibling that removes ``boto3`` from ``sys.modules`` leaves the patch on an
    orphaned object the source never consults, and the fan-out then runs
    against the real SDK - a live signed AWS request from a unit test. That
    shape passes every behavioural test in this file when the file runs alone,
    so only a structural check notices it.
    """

    def test_the_double_goes_through_sys_modules(self) -> None:
        """``sys.modules`` is the seam, matching the sibling hook driver."""
        src = inspect.getsource(_fanout_targets)
        assert 'setitem(sys.modules, "boto3"' in src

    def test_no_module_level_boto3_binding_is_patched(self) -> None:
        """A patch on this module's own binding is what the source ignores."""
        assert "setattr(boto3" not in inspect.getsource(_fanout_targets)

    def test_this_module_holds_no_boto3_binding_to_patch(self) -> None:
        """Graded as an import statement: naming it in prose binds nothing."""
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        bound = {
            alias.asname or alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "boto3"
            for alias in node.names
        }
        assert "boto3" not in bound, f"this module binds boto3: {sorted(bound)}"
