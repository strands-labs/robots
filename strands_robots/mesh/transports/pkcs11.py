"""An RFC 7512 ``pkcs11:`` key URI for the IoT transport: a device's private key stays in a token (a hardware
HSM, or SoftHSM under test) and is never a PEM on disk (readers E5).

A ``key=pkcs11:token=<label>;object=<key label>`` value on an ``iot://`` link selects
:func:`awsiot.mqtt5_client_builder.mtls_with_pkcs11`, so every private-key operation happens inside the token and
the key material never enters this process. ``module-path=<driver.so>`` in the URI query (or the ``PKCS11_MODULE``
env var) names the HSM's PKCS#11 driver. The user PIN comes only from the ``PKCS11_PIN`` env var (``None`` for a
token with a protected authentication path): a ``pin-value=`` is refused, because a link is rendered in reprs, errors
and tool results. ``awscrt`` is imported where used (the PKCS#11 TLS backend is Unix only)."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import unquote

SCHEME = "pkcs11:"
PIN_REFUSED = "a pkcs11: key never carries its PIN (pin-value=): set PKCS11_PIN, so no link, repr or error shows it"


def is_pkcs11(key: str | None) -> bool:
    """True when ``key`` is a ``pkcs11:`` URI rather than a path to a PEM key file."""
    return key is not None and key.startswith(SCHEME)


def refuse_pin(key: str) -> None:
    """Raise when the ``pkcs11:`` URI ``key`` carries its PIN (``pin-value=``); the error never echoes it."""
    if is_pkcs11(key) and "pin-value" in _attrs(key.partition("?")[2], "&"):
        raise ValueError(PIN_REFUSED)


def _attrs(text: str, sep: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in text.split(sep):
        name, eq, value = part.partition("=")
        if eq:
            out[unquote(name.strip())] = unquote(value.strip())
    return out


def parse(uri: str) -> dict[str, Any]:
    """An RFC 7512 ``pkcs11:`` URI to ``mtls_with_pkcs11`` keywords plus the ``module`` path. Path attributes
    ``token``, ``object`` and ``slot-id`` pick the key; ``module-path``/``module-name`` in the query (or the
    ``PKCS11_MODULE`` env var) give the driver and ``PKCS11_PIN`` the PIN (a ``pin-value=`` is refused). The
    module is required: awscrt needs an explicit path to the HSM's PKCS#11 ``.so``."""
    refuse_pin(uri)
    path, _, query = uri[len(SCHEME) :].partition("?")
    p, q = _attrs(path, ";"), _attrs(query, "&")
    module = q.get("module-path") or q.get("module-name") or os.getenv("PKCS11_MODULE")
    if not module:
        raise ValueError("a pkcs11: key needs module-path=<driver.so> in the URI (or the PKCS11_MODULE env var)")
    out: dict[str, Any] = {"module": module, "user_pin": os.getenv("PKCS11_PIN")}
    if "token" in p:
        out["token_label"] = p["token"]
    if "object" in p:
        out["private_key_label"] = p["object"]
    if "slot-id" in p:
        out["slot_id"] = int(p["slot-id"])
    return out


def client(cert: str, key: str, **kw: Any) -> Any:
    """An MQTT5 client whose private key lives in the token named by ``key`` (an RFC 7512 ``pkcs11:`` URI), with
    the device certificate ``cert``. ``kw`` are the common ``mqtt5_client_builder`` arguments (``endpoint``,
    ``client_id``, lifecycle callbacks)."""
    from awscrt.io import Pkcs11Lib
    from awsiot import mqtt5_client_builder

    params = parse(key)
    lib = Pkcs11Lib(file=params.pop("module"), behavior=Pkcs11Lib.InitializeFinalizeBehavior.STRICT)
    return mqtt5_client_builder.mtls_with_pkcs11(pkcs11_lib=lib, cert_filepath=cert, **params, **kw)
