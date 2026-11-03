"""A device key can live in a PKCS#11 token (RFC 7512 ``pkcs11:`` URI), never a PEM on disk (readers E5)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from strands_robots.mesh.transports import iot_link, pkcs11


def test_is_pkcs11_tells_a_token_uri_from_a_pem_path():
    assert pkcs11.is_pkcs11("pkcs11:token=lab;object=arm-1")
    assert not pkcs11.is_pkcs11("/etc/strands/arm-1/key.pem")
    assert not pkcs11.is_pkcs11(None)
    assert not pkcs11.is_pkcs11("")


@pytest.mark.parametrize(
    "uri, expected",
    [
        (
            "pkcs11:token=lab;object=arm-1?module-path=/opt/hsm.so",
            {"module": "/opt/hsm.so", "user_pin": None, "token_label": "lab", "private_key_label": "arm-1"},
        ),
        (
            "pkcs11:slot-id=7;object=arm-2?module-name=/opt/hsm.so",
            {"module": "/opt/hsm.so", "user_pin": None, "private_key_label": "arm-2", "slot_id": 7},
        ),
        (  # percent-encoded, as provision.register embeds it in an iot:// query
            "pkcs11:token=a%20lab;object=arm-3?module-path=%2Fopt%2Fhsm.so",
            {"module": "/opt/hsm.so", "user_pin": None, "token_label": "a lab", "private_key_label": "arm-3"},
        ),
    ],
)
def test_parse_reads_rfc7512_path_and_query_attributes(uri, expected, monkeypatch):
    monkeypatch.delenv("PKCS11_MODULE", raising=False)
    monkeypatch.delenv("PKCS11_PIN", raising=False)
    assert pkcs11.parse(uri) == expected


def test_parse_takes_the_module_and_pin_from_the_environment(monkeypatch):
    monkeypatch.setenv("PKCS11_MODULE", "/env/hsm.so")
    monkeypatch.setenv("PKCS11_PIN", "9999")
    assert pkcs11.parse("pkcs11:token=lab;object=arm-1") == {
        "module": "/env/hsm.so", "user_pin": "9999", "token_label": "lab", "private_key_label": "arm-1",
    }  # fmt: skip


def test_parse_without_a_module_is_refused(monkeypatch):
    monkeypatch.delenv("PKCS11_MODULE", raising=False)
    with pytest.raises(ValueError, match="module-path"):
        pkcs11.parse("pkcs11:token=lab;object=arm-1")


def test_a_pkcs11_key_builds_a_token_client_a_pem_key_builds_a_file_client(monkeypatch):
    import awsiot.mqtt5_client_builder as builder

    calls: dict[str, object] = {}
    monkeypatch.setattr(pkcs11, "client", lambda cert, key, **kw: calls.update(token=(cert, key)) or "TOKEN")
    monkeypatch.setattr(builder, "mtls_from_path", lambda **kw: calls.update(file=kw) or "FILE")

    uri = "pkcs11:token=lab;object=arm-1?module-path=/opt/hsm.so"
    assert iot_link._device_client("c.pem", uri, endpoint="e", client_id="arm-1") == "TOKEN"
    assert iot_link._device_client("c.pem", "/k.pem", endpoint="e", client_id="arm-1") == "FILE"
    assert calls["token"] == ("c.pem", uri)
    assert calls["file"]["pri_key_filepath"] == "/k.pem" and calls["file"]["cert_filepath"] == "c.pem"


# --- the real thing: a SoftHSM token, no PEM key on disk --------------------------------------------------------


def _softhsm() -> tuple[str, str]:
    """The PKCS#11 driver and the setup tool, or skip. ``SOFTHSM2_MODULE`` / ``SOFTHSM2_UTIL`` override."""
    module = os.getenv("SOFTHSM2_MODULE") or next(
        (
            c
            for c in (
                "/usr/lib/softhsm/libsofthsm2.so",
                "/usr/lib/aarch64-linux-gnu/softhsm/libsofthsm2.so",
                "/usr/lib/x86_64-linux-gnu/softhsm/libsofthsm2.so",
                "/usr/local/lib/softhsm/libsofthsm2.so",
            )
            if Path(c).exists()
        ),
        "",
    )
    util = os.getenv("SOFTHSM2_UTIL") or shutil.which("softhsm2-util") or ""
    if not module or not Path(module).exists() or not util:
        pytest.skip("SoftHSM (libsofthsm2.so + softhsm2-util) is not installed")
    return module, util


@pytest.fixture
def token(tmp_path, monkeypatch):
    """A SoftHSM token holding one RSA key (label ``devkey``) and a matching cert, with no private PEM on disk.

    Yields ``(pkcs11_uri, cert_path, tmp_path)``."""
    module, util = _softhsm()
    conf = tmp_path / "softhsm2.conf"
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    conf.write_text(f"directories.tokendir = {tokens}\nobjectstore.backend = file\nlog.level = ERROR\n")
    monkeypatch.setenv("SOFTHSM2_CONF", str(conf))

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_pem = tmp_path / "key.pem"
    key_pem.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "arm-1")])
    import datetime

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "cert.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    run = [util, "--module", module, "--init-token", "--free", "--label", "e5tok",
           "--so-pin", "1234", "--pin", "5678"]  # fmt: skip
    subprocess.run(run, check=True, capture_output=True)
    subprocess.run(
        [
            util,
            "--module",
            module,
            "--import",
            str(key_pem),
            "--token",
            "e5tok",
            "--label",
            "devkey",
            "--id",
            "01",
            "--pin",
            "5678",
        ],  # fmt: skip
        check=True,
        capture_output=True,
    )
    key_pem.unlink()  # the private key now lives only in the token
    monkeypatch.setenv("PKCS11_PIN", "5678")
    uri = f"pkcs11:token=e5tok;object=devkey?module-path={module}"
    return uri, str(cert_path), tmp_path


def test_a_token_key_builds_an_mqtt_client_with_no_pem_on_disk(token):
    from awscrt.mqtt5 import Client

    uri, cert, tmp_path = token
    assert not (tmp_path / "key.pem").exists()  # no private key as a file
    client = pkcs11.client(cert, uri, endpoint="example-ats.iot.us-west-2.amazonaws.com", client_id="arm-1")
    assert isinstance(client, Client)  # awscrt found and loaded the key from the token


def test_an_iot_link_hands_the_whole_percent_encoded_token_uri_to_the_client(monkeypatch):
    from urllib.parse import quote

    from strands_robots.mesh.transports import iot

    seen: dict[str, object] = {}
    monkeypatch.setattr(iot, "AwsLink", lambda endpoint, region, client, cert, key: seen.update(key=key) or "LINK")
    monkeypatch.setenv("PKCS11_PIN", "1234")
    uri = "pkcs11:token=lab-a;object=arm-1?module-path=/opt/hsm.so&module-name=x"
    base = "iot://us-west-2?endpoint=e&client=arm-1&cert=c.pem&key="
    assert iot._link(base + quote(uri, safe="")) == "LINK"
    assert seen["key"] == uri and pkcs11.parse(str(seen["key"]))["user_pin"] == "1234"
    with pytest.raises(ValueError, match="percent-encode"):  # bare: '&module-name' would leave the key
        iot._link(base + uri)


PIN = "pkcs11:token=lab-a;object=arm-1?module-path=/opt/hsm.so&pin-value=1234"


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: pkcs11.parse(PIN), id="parse"),
        pytest.param(lambda: _iot()._link(f"iot://fake?client=a&cert=c&key={_quote(PIN)}"), id="encoded-link"),
        pytest.param(lambda: _iot()._link(f"iot://fake?client=a&cert=c&key={PIN}"), id="split-link"),
        pytest.param(lambda: _iot().IotTransport(f"iot://fake?client=a&cert=c&key={_quote(PIN)}"), id="transport"),
    ],
)
def test_a_pin_on_a_pkcs11_key_is_refused_and_never_echoed(call):
    """The PIN comes only from PKCS11_PIN: a link is rendered by Mesh.__repr__, its closed error and use_mesh."""
    with pytest.raises(ValueError, match="PKCS11_PIN") as err:
        call()
    assert "1234" not in str(err.value)


def _iot():
    from strands_robots.mesh.transports import iot

    return iot


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")
