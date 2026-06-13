"""IoT safety-hardening regression tests.

Regression coverage for the IoT-transport:
- MQTT Last Will weaponized as dead-man switch (no-estop policy variant)
- E-Stop Lambda cost amplification (Lambda dedup + reserved concurrency)
"""

from __future__ import annotations

import ast


# --------------------------------------------------------------------------- #15
class TestNoEstopPolicyVariant:
    def _sids(self, doc):
        return [s.get("Sid") for s in doc["Statement"]]

    def test_default_policy_allows_estop_publish(self):
        from strands_robots.mesh.iot.provision import _robot_policy_doc

        doc = _robot_policy_doc(allow_estop_publish=True)
        assert "AllowSafetyEstop" in self._sids(doc)

    def test_hardened_policy_drops_estop_publish(self):
        from strands_robots.mesh.iot.provision import _robot_policy_doc

        doc = _robot_policy_doc(allow_estop_publish=False)
        assert "AllowSafetyEstop" not in self._sids(doc)

    def test_hardened_policy_still_receives_estop(self):
        """A no-estop-publish robot must STILL obey fleet stops."""
        from strands_robots.mesh.iot.provision import _robot_policy_doc

        doc = _robot_policy_doc(allow_estop_publish=False)
        receive = next(s for s in doc["Statement"] if s["Sid"] == "AllowReceiveScoped")
        subscribe = next(s for s in doc["Statement"] if s["Sid"] == "AllowOwnSubscriptions")
        assert any("safety/estop" in r for r in receive["Resource"])
        assert any("safety/estop" in r for r in subscribe["Resource"])

    def test_hardened_owntopics_does_not_cover_estop(self):
        """The remaining publish grant (AllowOwnTopics) is ThingName-scoped and
        must NOT reach the global safety/estop topic (else the Will vector
        would survive)."""
        from strands_robots.mesh.iot.provision import _robot_policy_doc

        doc = _robot_policy_doc(allow_estop_publish=False)
        own = next(s for s in doc["Statement"] if s["Sid"] == "AllowOwnTopics")
        assert not any("safety/estop" in r for r in own["Resource"])

    def test_builder_does_not_mutate_module_default(self):
        from strands_robots.mesh.iot import provision

        _ = provision._robot_policy_doc(allow_estop_publish=False)
        default_sids = [s.get("Sid") for s in provision._ROBOT_POLICY_DOC["Statement"]]
        assert "AllowSafetyEstop" in default_sids

    def test_distinct_policy_name(self):
        from strands_robots.mesh.iot.provision import (
            ROBOT_NO_ESTOP_POLICY_NAME,
            ROBOT_POLICY_NAME,
        )

        assert ROBOT_NO_ESTOP_POLICY_NAME != ROBOT_POLICY_NAME
        assert ROBOT_NO_ESTOP_POLICY_NAME == "strands-robot-no-estop"


# --------------------------------------------------------------------------- #16
class TestEstopLambdaDedup:
    def test_lambda_source_is_valid_python(self):
        from strands_robots.mesh.iot import bootstrap as b

        ast.parse(b._ESTOP_LAMBDA_SOURCE)

    def test_lambda_has_dedup_conditional_put(self):
        from strands_robots.mesh.iot import bootstrap as b

        src = b._ESTOP_LAMBDA_SOURCE
        assert "_is_duplicate" in src
        assert "ConditionExpression" in src
        assert "attribute_not_exists" in src

    def test_lambda_dedups_on_peer_id_and_t(self):
        from strands_robots.mesh.iot import bootstrap as b

        src = b._ESTOP_LAMBDA_SOURCE
        # dedup key is (sender, t)
        assert '"{}:{}".format(sender, t)' in src

    def test_lambda_skips_fanout_on_duplicate(self):
        from strands_robots.mesh.iot import bootstrap as b

        src = b._ESTOP_LAMBDA_SOURCE
        assert "SKIPPED" in src
        assert '"deduped": True' in src

    def test_lambda_fails_open_on_store_error(self):
        """A dedup-store outage must never suppress a real estop."""
        from strands_robots.mesh.iot import bootstrap as b

        assert "failing open" in b._ESTOP_LAMBDA_SOURCE

    def test_lambda_marker_has_ttl(self):
        from strands_robots.mesh.iot import bootstrap as b

        assert "expire_at" in b._ESTOP_LAMBDA_SOURCE

    def test_version_bumped(self):
        from strands_robots.mesh.iot import bootstrap as b

        assert b._LAMBDA_VERSION >= 2

    def test_reserved_concurrency_constant(self):
        from strands_robots.mesh.iot import bootstrap as b

        assert b.ESTOP_LAMBDA_RESERVED_CONCURRENCY >= 1


class TestEstopLambdaRoleGrantsPutItem:
    def test_role_policy_includes_dynamodb_putitem(self, monkeypatch):
        import json
        from unittest.mock import MagicMock

        from strands_robots.mesh.iot.bootstrap import (
            BootstrappedAccount,
            _ensure_lambda_role,
        )

        class _NoSuch(Exception):
            pass

        iam = MagicMock()
        iam.exceptions = MagicMock()
        iam.exceptions.NoSuchEntityException = _NoSuch
        iam.get_role.side_effect = _NoSuch()
        iam.create_role.return_value = {"Role": {"Arn": "arn:iam:estop"}}
        monkeypatch.setattr("strands_robots.mesh.iot.bootstrap.time.sleep", lambda *_: None)

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        _ensure_lambda_role(iam, a)

        doc = json.loads(iam.put_role_policy.call_args.kwargs["PolicyDocument"])
        actions = [
            act
            for st in doc["Statement"]
            for act in (st["Action"] if isinstance(st["Action"], list) else [st["Action"]])
        ]
        assert "dynamodb:PutItem" in actions


class TestEstopLambdaCreateConfig:
    def test_create_sets_env_and_reserved_concurrency(self, monkeypatch):
        from unittest.mock import MagicMock

        from strands_robots.mesh.iot.bootstrap import (
            BootstrappedAccount,
            _ensure_estop_lambda,
        )

        class _NoSuch(Exception):
            pass

        lam = MagicMock()
        lam.exceptions = MagicMock()
        lam.exceptions.ResourceNotFoundException = _NoSuch
        lam.get_function.side_effect = _NoSuch()
        lam.create_function.return_value = {"FunctionArn": "arn:lam:estop"}

        a = BootstrappedAccount(region="us-west-2", account_id="123")
        _ensure_estop_lambda(lam, "arn:iam:estop", a)

        kw = lam.create_function.call_args.kwargs
        env = kw["Environment"]["Variables"]
        assert env["STRANDS_SAFETY_TABLE"] == "strands-mesh-safety-events"
        assert "STRANDS_ESTOP_DEDUP_TTL_S" in env
        # Reserved concurrency was set.
        lam.put_function_concurrency.assert_called_once()
