"""moto for iot / iot-data / sts; placeholder account, never a real one."""

from __future__ import annotations

import os

import pytest

ACCOUNT = "123456789012"
REGION = "us-west-2"


@pytest.fixture
def aws(monkeypatch):
    pytest.importorskip("moto")
    from moto import mock_aws

    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(k, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        import boto3

        yield boto3.client("iot", region_name=REGION)


@pytest.fixture
def no_live():
    if os.environ.get("STRANDS_IOT_LIVE") == "1":
        pytest.skip("live run")
