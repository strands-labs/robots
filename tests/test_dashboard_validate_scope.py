"""What a policy preflight actually verified (see validate_scope.py).

The defect these pin: /api/policies/validate answered ok/stage=preflight for
lerobot_local with an EMPTY config and the run form rendered a green
a green "lerobot_local resolves". No model had been named, so the preflight's real
check (declared image inputs vs the observation keys the peer will send) had
nothing to inspect - the pass was a pass of an empty question, one click away
from Run on a real arm.
"""

from strands_robots.dashboard.validate_scope import validation_scope

LEROBOT_LOCAL = {
    "name": "lerobot_local",
    "wire_fields": [
        {"key": "pretrained_name_or_path", "required": False},
        {"key": "policy_type", "required": False},
    ],
    "config_keys": ["pretrained_name_or_path", "policy_type", "device"],
}


def test_empty_config_resolved_nothing() -> None:
    scope = validation_scope(LEROBOT_LOCAL, {})
    assert scope["resolved"] is False
    assert scope["identity_keys"] == ["pretrained_name_or_path"]
    # The sentence must say what was NOT checked, and name the empty key.
    assert "no model was named" in scope["scope_note"]
    assert "pretrained_name_or_path is empty" in scope["scope_note"]
    assert "provider exists" in scope["scope_note"]


def test_blank_and_whitespace_count_as_empty() -> None:
    for value in ("", "   ", None, [], {}):  # type: ignore[var-annotated]
        assert validation_scope(LEROBOT_LOCAL, {"pretrained_name_or_path": value})["resolved"] is False


def test_a_named_checkpoint_is_a_real_verdict() -> None:
    scope = validation_scope(LEROBOT_LOCAL, {"pretrained_name_or_path": "HashtagRobotics/smolvla-a"})
    assert scope["resolved"] is True
    assert scope["scope_note"] is None, "a real check must add no caveat"


def test_other_identity_key_spellings_are_covered() -> None:
    # A provider added later must be covered without editing this logic.
    for key in ("checkpoint", "model_path", "ckpt_dir", "weights_path", "pretrained_model"):
        spec = {"wire_fields": [{"key": key}]}
        assert validation_scope(spec, {})["resolved"] is False
        assert validation_scope(spec, {key: "/tmp/x"})["resolved"] is True


def test_provider_with_no_model_key_is_not_accused() -> None:
    # mock / a whole-body controller with defaults: there is no checkpoint to
    # name, so "resolves" is the honest word and gets no caveat.
    assert validation_scope({"name": "mock", "wire_fields": []}, {}) == {
        "resolved": True,
        "identity_keys": [],
        "scope_note": None,
    }


def test_remote_provider_verdict_stops_at_this_machine() -> None:
    spec = {"name": "remote", "wire_fields": [{"key": "host"}, {"key": "port"}]}
    scope = validation_scope(spec, {"host": "10.0.0.4", "port": 8000})
    assert scope["resolved"] is True
    assert "nothing here can confirm the server at the other end" in scope["scope_note"]
    # ...and with nothing set at all there is no address to over-claim about.
    assert validation_scope(spec, {})["scope_note"] is None


def test_identity_key_wins_over_a_remote_address() -> None:
    spec = {
        "name": "lerobot_async",
        "wire_fields": [
            {"key": "pretrained_name_or_path"},
            {"key": "server_address"},
        ],
    }
    assert validation_scope(spec, {"server_address": "1.2.3.4:9000"})["resolved"] is False
    assert validation_scope(spec, {"pretrained_name_or_path": "org/m"})["resolved"] is True


def test_junk_spec_and_config_cannot_raise() -> None:
    for spec in (None, {}, [], "lerobot_local", {"wire_fields": "nope"}, {"wire_fields": [None, {}]}):  # type: ignore[var-annotated]
        for cfg in (None, {}, {"pretrained_name_or_path": "x"}, "not a dict"):
            out = validation_scope(spec, cfg)  # type: ignore[arg-type]
            assert set(out) == {"resolved", "identity_keys", "scope_note"}
            assert isinstance(out["resolved"], bool)
