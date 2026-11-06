"""The two WebAuthn ceremonies (enrol a passkey, sign in with one) on ``py_webauthn``, imported on first use.

Each ``*_options`` returns ``(challenge, options)``: the options go to ``navigator.credentials.create`` / ``get`` as
JSON, the challenge stays on the server. Each ``verify_*`` checks the browser's answer against that challenge, this
relying party id and this origin, and raises ``PermissionError`` when it does not hold.
"""

from __future__ import annotations

import json
from typing import Any

from ..._extras import hint

RP_NAME = "strands-robots dashboard"


def _webauthn() -> Any:
    try:
        import webauthn
    except ImportError as e:
        raise ImportError(f"dashboard sign-in needs py_webauthn: {hint('dashboard')} ({e})") from e
    return webauthn


def _unb64(text: str) -> bytes:
    from webauthn.helpers import base64url_to_bytes

    return base64url_to_bytes(text)


def _descriptors(ids: list[str]) -> list[Any]:
    from webauthn.helpers.structs import PublicKeyCredentialDescriptor

    return [PublicKeyCredentialDescriptor(id=_unb64(i)) for i in ids]


def enrol_options(*, rp_id: str, user_id: str, exclude: list[str]) -> tuple[bytes, dict[str, Any]]:
    wa = _webauthn()
    from webauthn.helpers.structs import AuthenticatorSelectionCriteria, ResidentKeyRequirement

    opts = wa.generate_registration_options(
        rp_id=rp_id,
        rp_name=RP_NAME,
        user_id=_unb64(user_id),
        user_name="operator",
        user_display_name="strands-robots operator",
        exclude_credentials=_descriptors(exclude) or None,
        authenticator_selection=AuthenticatorSelectionCriteria(resident_key=ResidentKeyRequirement.PREFERRED),
    )
    return opts.challenge, json.loads(wa.options_to_json(opts))


def verify_enrol(credential: Any, *, challenge: bytes, rp_id: str, origin: str) -> tuple[str, str, int]:
    """``(credential id, public key, sign count)``, ids and key base64url, of a verified new passkey."""
    wa = _webauthn()
    from webauthn.helpers import bytes_to_base64url
    from webauthn.helpers.exceptions import WebAuthnException

    try:
        v = wa.verify_registration_response(
            credential=credential, expected_challenge=challenge, expected_rp_id=rp_id, expected_origin=origin
        )
    except (WebAuthnException, ValueError, KeyError, TypeError) as exc:
        raise PermissionError(f"passkey not accepted: {exc}") from exc
    return bytes_to_base64url(v.credential_id), bytes_to_base64url(v.credential_public_key), v.sign_count


def login_options(*, rp_id: str, allow: list[str]) -> tuple[bytes, dict[str, Any]]:
    wa = _webauthn()
    opts = wa.generate_authentication_options(rp_id=rp_id, allow_credentials=_descriptors(allow))
    return opts.challenge, json.loads(wa.options_to_json(opts))


def verify_login(
    credential: Any, *, challenge: bytes, rp_id: str, origin: str, public_key: str, sign_count: int
) -> int:
    """The authenticator's new sign count; a count that did not move past the stored one (a replay) is refused."""
    wa = _webauthn()
    from webauthn.helpers.exceptions import WebAuthnException

    try:
        v = wa.verify_authentication_response(
            credential=credential,
            expected_challenge=challenge,
            expected_rp_id=rp_id,
            expected_origin=origin,
            credential_public_key=_unb64(public_key),
            credential_current_sign_count=sign_count,
        )
    except (WebAuthnException, ValueError, KeyError, TypeError) as exc:
        raise PermissionError(f"sign-in refused: {exc}") from exc
    return int(v.new_sign_count)
