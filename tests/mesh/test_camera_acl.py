"""Camera-frame access-control tests.

Covers :class:`CameraOffloader` presigned-URL TTL semantics:

* The short default TTL of 60 seconds, as a posture pin.
* The 1-hour ceiling that clamps over-eager operator overrides.
* The None-vs-explicit-0 distinction (env fallback only when None;
  explicit 0 is treated as an operator value and clamped to 1).

The privacy kill-switch (``STRANDS_MESH_CAMERA_DISABLED``) and the S3
PutObject ACL hardening were dropped from PR #228 R2 because the
intended publish-side gate was never landed in production code; the
prior tests passed for incidental reasons (short-circuiting at the
inner-None guard rather than any kill-switch guard) and gave false
reassurance. Both items are tracked in the deferred follow-up issue
#249 and will land with their own pin tests there.
"""

from __future__ import annotations

from strands_robots.mesh.iot.camera_offload import (
    DEFAULT_PRESIGN_TTL_SECONDS,
    MAX_PRESIGN_TTL_SECONDS,
    CameraOffloader,
)


class TestPresignTTL:
    def test_default_is_60s(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="test-bucket")
        assert off.presign_ttl == 60
        # Pin the constant so a future regression that bumps it back to 3600
        # fails this test loudly.
        assert DEFAULT_PRESIGN_TTL_SECONDS == 60

    def test_env_override_within_cap_passes_through(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "120")
        off = CameraOffloader(bucket="test-bucket")
        assert off.presign_ttl == 120

    def test_env_override_above_cap_clamps(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "86400")  # 1 day
        off = CameraOffloader(bucket="test-bucket")
        assert off.presign_ttl == MAX_PRESIGN_TTL_SECONDS  # clamped

    def test_kwarg_override_above_cap_clamps(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="test-bucket", presign_ttl=999_999)
        assert off.presign_ttl == MAX_PRESIGN_TTL_SECONDS

    def test_zero_or_negative_clamped_up(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="test-bucket", presign_ttl=0)
        # presign_ttl=0 is explicitly passed (not None) -> clamped to floor of 1
        assert off.presign_ttl == 1

    def test_negative_clamped_up(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="test-bucket", presign_ttl=-5)
        # Negative values are clamped to 1
        assert off.presign_ttl == 1
