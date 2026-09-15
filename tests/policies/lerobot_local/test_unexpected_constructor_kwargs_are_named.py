"""A constructor kwarg ``lerobot_local`` does not read is named, not dropped.

``create_policy`` forwards one shared kwargs bag to every provider, so the
constructor tolerates keys it does not own - the contract ``lerobot_async``
already grades. Tolerating them silently is a different thing: ``rtc=True``
(the spelling ``docs/policies/lerobot-local.md`` warns against, for
``rtc_enabled=``) built a policy with RTC off and no line anywhere saying the
request was never read.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def test_unexpected_constructor_kwargs_warn_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        policy = LerobotLocalPolicy(rtc=True)
    assert isinstance(policy, LerobotLocalPolicy)
    named = [r for r in caplog.records if "ignoring unexpected constructor kwarg" in r.message]
    assert len(named) == 1, caplog.text
    # Grade the key off the record's own argument rather than searching the
    # rendered text: prose that happens to contain the probe key satisfies a
    # caplog.text search whatever the key list rendered as, which is the half
    # this file is named for.
    args = named[0].args
    assert isinstance(args, tuple), args
    assert args[0] == ["rtc"], caplog.text


def test_known_kwargs_raise_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        LerobotLocalPolicy(rtc_enabled=True, strict_keys=False)
    assert not any("ignoring unexpected constructor kwarg" in r.message for r in caplog.records)
