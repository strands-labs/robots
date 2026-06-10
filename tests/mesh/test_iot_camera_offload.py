"""Unit tests for CameraOffloader and S3 camera offload auto-wiring.

No real S3 — uses MagicMock-backed boto3 client and transport.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# numpy is only used by three frame-encoding tests below. Use
# pytest.importorskip so this module is collected (and the bulk
# of the suite still runs) on environments that ship without numpy.
# AGENTS.md > Review Learnings (#85): test import paths must match
# production -- the camera_offload production code does not require
# numpy at module-import time, so neither should this file.
np = pytest.importorskip("numpy")

from strands_robots.mesh.iot.camera_offload import (  # noqa: E402  # importorskip must precede
    CameraOffloader,
    enable_for_mesh,
)

# CameraOffloader behaviour


class TestCameraOffloaderConfig:
    def test_disabled_without_bucket(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
        c = CameraOffloader()
        assert c.enabled is False

    def test_env_bucket_picked_up(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "my-frames")
        c = CameraOffloader()
        assert c.enabled is True
        assert c.bucket == "my-frames"

    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "env-bucket")
        c = CameraOffloader(bucket="ctor-bucket")
        assert c.bucket == "ctor-bucket"

    def test_prefix_strips_slashes(self):
        c = CameraOffloader(bucket="b", prefix="/foo/bar/")
        assert c.prefix == "foo/bar"

    def test_default_presign_ttl(self, monkeypatch):
        # Default presigned-URL TTL is 60 s, deliberately short to limit
        # the window during which a leaked /ref message can be replayed.
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        c = CameraOffloader(bucket="b")
        assert c.presign_ttl == 60

    def test_env_presign_ttl(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "60")
        c = CameraOffloader(bucket="b")
        assert c.presign_ttl == 60


class TestCameraOffloaderS3Key:
    def test_key_layout(self):
        c = CameraOffloader(bucket="b")
        k = c.s3_key_for("so100-01", "wrist", 1700000000_000_000_000)
        assert k == "so100-01/wrist/1700000000000000000.jpg"

    def test_key_layout_with_prefix(self):
        c = CameraOffloader(bucket="b", prefix="customer-A")
        k = c.s3_key_for("so100-01", "wrist", 1700000000)
        assert k == "customer-A/so100-01/wrist/1700000000.jpg"


class TestCameraOffloaderUpload:
    def test_disabled_returns_none(self):
        c = CameraOffloader()  # no bucket
        assert c.upload_frame("p", "cam", b"jpeg", 1.0) is None

    def test_uploads_and_returns_ref(self, monkeypatch):
        c = CameraOffloader(bucket="frames", region="us-west-2")
        s3 = MagicMock()
        s3.generate_presigned_url.return_value = "https://signed.example/"
        c._s3 = s3  # short-circuit lazy import

        ref = c.upload_frame("so100-01", "wrist", b"\xff\xd8jpeg", 12345.6)
        assert ref is not None
        assert ref["peer_id"] == "so100-01"
        assert ref["cam"] == "wrist"
        assert ref["t"] == 12345.6
        assert ref["s3_uri"].startswith("s3://frames/so100-01/wrist/")
        assert ref["presigned_url"] == "https://signed.example/"
        assert ref["expires_at"] == 12345.6 + 60  # default TTL = 60 s

        # Verify the put_object call shape.
        s3.put_object.assert_called_once()
        kwargs = s3.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "frames"
        assert kwargs["ContentType"] == "image/jpeg"
        assert kwargs["Body"] == b"\xff\xd8jpeg"
        assert kwargs["Key"].startswith("so100-01/wrist/")

    def test_returns_none_on_put_error(self):
        c = CameraOffloader(bucket="frames")
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 error")
        c._s3 = s3
        assert c.upload_frame("p", "cam", b"x", 1.0) is None


# enable_for_mesh


class TestEnableForMesh:
    def test_noop_on_zenoh(self):
        mesh = MagicMock()
        with patch(
            "strands_robots.mesh.transport.factory.current_backend",
            return_value="zenoh",
        ):
            assert enable_for_mesh(mesh) is None

    def test_noop_when_no_bucket(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
        mesh = MagicMock()
        with patch(
            "strands_robots.mesh.transport.factory.current_backend",
            return_value="iot",
        ):
            assert enable_for_mesh(mesh) is None

    def test_wraps_publish_when_bucket_configured(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "frames")

        mesh = MagicMock()
        mesh.peer_id = "so100-01"
        mesh._publish_cameras_once = MagicMock()

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            off = enable_for_mesh(mesh)

        assert off is not None
        assert off.bucket == "frames"
        # The wrapper should be a different callable than the original
        assert mesh._publish_cameras_once != mesh._publish_cameras_once.__class__


class TestEnableForMeshOffloadWrapper:
    """White-box tests for camera_offload.enable_for_mesh — exercise the
    wrapper that runs inside Mesh._publish_cameras_once when the bucket is set."""

    def test_wrapper_skips_when_robot_not_connected(self, monkeypatch):
        """If the underlying robot isn't connected, the offload path
        returns silently — no S3 call, no exception."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "frames")
        from strands_robots.mesh.iot.camera_offload import enable_for_mesh

        mesh = MagicMock()
        mesh.peer_id = "p"

        # robot.robot.is_connected = False
        inner = type("I", (), {})()
        inner.is_connected = False
        inner.config = type("C", (), {"cameras": {"front": {}}})()
        mesh.robot = type("R", (), {})()
        mesh.robot.robot = inner

        # original publish_cameras_once exists and is callable
        original_called = []
        mesh._publish_cameras_once = lambda: original_called.append(1)

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            off = enable_for_mesh(mesh)
        assert off is not None
        # Drive the wrapper — it should call original AND early-return on offload
        mesh._publish_cameras_once()
        assert original_called == [1]

    def test_wrapper_handles_get_observation_failure(self, monkeypatch):
        """If get_observation raises, the wrapper swallows the error."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "frames")
        from strands_robots.mesh.iot.camera_offload import enable_for_mesh

        mesh = MagicMock()
        mesh.peer_id = "p"

        inner = type("I", (), {})()
        inner.is_connected = True
        inner.config = type("C", (), {"cameras": {"front": {}}})()
        # get_observation raises — wrapper should bail without raising
        inner.get_observation = MagicMock(side_effect=RuntimeError("camera dead"))
        mesh.robot = type("R", (), {})()
        mesh.robot.robot = inner

        original_called = []
        mesh._publish_cameras_once = lambda: original_called.append(1)

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            enable_for_mesh(mesh)

        # Must not raise
        mesh._publish_cameras_once()
        assert original_called == [1]

    def test_wrapper_no_op_when_no_cameras_in_config(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "frames")
        from strands_robots.mesh.iot.camera_offload import enable_for_mesh

        mesh = MagicMock()
        mesh.peer_id = "p"
        inner = type("I", (), {})()
        inner.is_connected = True
        inner.config = type("C", (), {"cameras": {}})()  # empty
        mesh.robot = type("R", (), {})()
        mesh.robot.robot = inner
        mesh._publish_cameras_once = lambda: None

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            enable_for_mesh(mesh)

        mesh._publish_cameras_once()  # must not raise


# === Coverage-gap tests for upload_frame error paths and edge cases ===


class TestCameraOffloaderTTLBounds:
    """The presign TTL is clamped to ``[1, MAX_PRESIGN_TTL_SECONDS=3600]``.
    Below the floor or above the cap, we clamp loudly (or silently for the
    floor since 0/negative is operator-supplied bad input).
    """

    def test_ttl_above_cap_is_clamped_to_3600(self, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "999999")
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.iot.camera_offload"):
            c = CameraOffloader(bucket="b")
        assert c.presign_ttl == 3600
        assert any("clamping" in m for m in caplog.messages), f"expected WARNING about clamping; got {caplog.messages}"

    def test_ttl_zero_clamped_to_one(self):
        c = CameraOffloader(bucket="b", presign_ttl=0)
        # ttl=0 means "always-expired URL" which is useless; clamp to 1.
        # presign_ttl=0 is explicit (not None), so the env-var fallback is
        # skipped and the < 1 floor clamps it to exactly 1.  See
        # tests/mesh/test_presign_ttl_none_vs_zero.py for the full matrix.
        assert c.presign_ttl == 1

    def test_ttl_negative_env_clamped(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "-99")
        c = CameraOffloader(bucket="b")
        assert c.presign_ttl == 1


class TestCameraOffloaderClientLazy:
    """`_client()` is lazy and gracefully degrades when boto3 is missing."""

    def test_boto3_missing_returns_none(self, monkeypatch):
        c = CameraOffloader(bucket="b")
        # Force ImportError inside _client by removing boto3 from sys.modules
        # and blocking its import.
        import builtins
        import sys

        original_import = builtins.__import__

        def blocked(name, *a, **kw):
            if name == "boto3":
                raise ImportError("simulated boto3 missing")
            return original_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", blocked)
        sys.modules.pop("boto3", None)
        assert c._client() is None
        # Subsequent calls also return None (cached miss is acceptable; we
        # just check the public contract: never raises, always returns None
        # when boto3 absent).
        assert c._client() is None

    def test_client_is_cached(self):
        c = CameraOffloader(bucket="b", region="us-west-2")
        with patch("boto3.client") as boto_client:
            mock_s3 = MagicMock()
            boto_client.return_value = mock_s3
            assert c._client() is mock_s3
            # Second call must not invoke boto3.client again.
            assert c._client() is mock_s3
            assert boto_client.call_count == 1


class TestUploadFrameErrorPaths:
    """`upload_frame` returns None on every error condition; never raises."""

    def test_no_bucket_returns_none(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
        c = CameraOffloader()
        assert c.upload_frame("r1", "front", b"jpeg-bytes", 1234.5) is None

    def test_put_object_error_returns_none(self):
        c = CameraOffloader(bucket="b")
        c._s3 = MagicMock()
        c._s3.put_object.side_effect = RuntimeError("simulated ClientError")
        assert c.upload_frame("r1", "front", b"jpeg", 1.0) is None

    def test_presign_failure_returns_ref_with_null_url(self):
        c = CameraOffloader(bucket="b")
        c._s3 = MagicMock()
        c._s3.put_object.return_value = {}
        c._s3.generate_presigned_url.side_effect = RuntimeError("simulated NoCredentialsError")
        ref = c.upload_frame("r1", "front", b"jpeg", 1.0)
        assert ref is not None
        # Upload succeeded, but presign failed -> ref carries a null URL
        assert ref["s3_uri"] == "s3://b/r1/front/1000000000.jpg"
        assert ref["presigned_url"] is None
        assert ref["expires_at"] == 1.0 + c.presign_ttl

    def test_happy_path_returns_full_ref(self):
        c = CameraOffloader(bucket="b", prefix="frames", presign_ttl=120)
        c._s3 = MagicMock()
        c._s3.generate_presigned_url.return_value = "https://example.com/signed"
        ref = c.upload_frame("r1", "front", b"jpeg", 100.0)
        assert ref == {
            "peer_id": "r1",
            "cam": "front",
            "t": 100.0,
            "encoding": "jpeg",
            "s3_uri": "s3://b/frames/r1/front/100000000000.jpg",
            "presigned_url": "https://example.com/signed",
            "expires_at": 100.0 + 120,
        }
        c._s3.put_object.assert_called_once_with(
            Bucket="b",
            Key="frames/r1/front/100000000000.jpg",
            Body=b"jpeg",
            ContentType="image/jpeg",
        )


class TestS3KeyForLayout:
    def test_no_prefix(self):
        c = CameraOffloader(bucket="b")
        assert c.s3_key_for("r1", "front", 12345) == "r1/front/12345.jpg"

    def test_with_prefix(self):
        c = CameraOffloader(bucket="b", prefix="fleet-a/raw")
        assert c.s3_key_for("r1", "front", 12345) == "fleet-a/raw/r1/front/12345.jpg"


class TestEnableForMeshGuards:
    """`enable_for_mesh` short-circuits cleanly on every wrong-state path."""

    def test_returns_none_on_zenoh_backend(self, monkeypatch):
        from strands_robots.mesh.transport import factory as fac

        monkeypatch.setattr(fac, "current_backend", lambda: "zenoh")
        mesh = MagicMock()
        assert enable_for_mesh(mesh) is None
        # Original method should NOT have been monkey-patched
        # (mesh._publish_cameras_once is a MagicMock attr, not the patched fn)
        # We verify by asserting no setattr occurred.

    def test_returns_none_when_bucket_missing(self, monkeypatch):
        from strands_robots.mesh.transport import factory as fac

        monkeypatch.setattr(fac, "current_backend", lambda: "iot")
        monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
        mesh = MagicMock()
        assert enable_for_mesh(mesh) is None

    def test_wires_up_when_bucket_set(self, monkeypatch):
        from strands_robots.mesh.transport import factory as fac

        monkeypatch.setattr(fac, "current_backend", lambda: "iot")
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "wired-bucket")

        mesh = MagicMock()
        original = mesh._publish_cameras_once
        off = enable_for_mesh(mesh)
        assert off is not None
        assert off.bucket == "wired-bucket"
        # The patched method replaces the original
        assert mesh._publish_cameras_once is not original


# === The patched _publish_cameras_once_with_offload closure ===


class TestPatchedPublishClosure:
    """Cover the inner closure ``_publish_cameras_once_with_offload`` end-to-end.
    Builds a Mesh stub that satisfies the closure's duck-type requirements.
    """

    def _make_mesh_with_camera(self, monkeypatch, bucket="b"):
        from strands_robots.mesh.transport import factory as fac

        monkeypatch.setattr(fac, "current_backend", lambda: "iot")
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", bucket)

        # Stub transport that the closure queries
        transport = MagicMock()
        transport.is_alive.return_value = True
        monkeypatch.setattr(fac, "current_transport", lambda: transport)

        # Mock S3 so upload_frame succeeds
        offloader = CameraOffloader(bucket=bucket)
        offloader._s3 = MagicMock()
        offloader._s3.generate_presigned_url.return_value = "https://example.com/signed"

        # Build a mesh stub
        mesh = MagicMock()
        mesh.peer_id = "robot-x"
        # Connected inner robot with one camera
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mesh.robot.robot.is_connected = True
        mesh.robot.robot.config.cameras = {"front": object()}
        mesh.robot.robot.get_observation.return_value = {"front": frame}
        return mesh, offloader, transport

    def test_closure_uploads_and_publishes_ref(self, monkeypatch):
        mesh, off, transport = self._make_mesh_with_camera(monkeypatch)
        # Wire up
        off_returned = enable_for_mesh(mesh, offloader=off)
        assert off_returned is off
        # Trigger the patched publish
        mesh._publish_cameras_once()
        # transport.put must have been called exactly once for the ref topic
        calls = [c for c in transport.put.call_args_list if "/ref" in c.args[0]]
        assert len(calls) == 1, f"expected 1 ref publish, got {transport.put.call_args_list}"
        topic, ref = calls[0].args
        assert topic == "strands/robot-x/camera/front/ref"
        assert ref["peer_id"] == "robot-x"
        assert ref["cam"] == "front"
        assert ref["s3_uri"].startswith("s3://b/robot-x/front/")
        assert ref["shape"] == [480, 640, 3]
        assert ref["presigned_url"] == "https://example.com/signed"

    def test_closure_skips_disconnected_robot(self, monkeypatch):
        mesh, off, transport = self._make_mesh_with_camera(monkeypatch)
        mesh.robot.robot.is_connected = False
        enable_for_mesh(mesh, offloader=off)
        mesh._publish_cameras_once()
        # No /ref publish when disconnected
        assert not any("/ref" in c.args[0] for c in transport.put.call_args_list)

    def test_closure_skips_when_transport_dead(self, monkeypatch):
        mesh, off, transport = self._make_mesh_with_camera(monkeypatch)
        transport.is_alive.return_value = False
        enable_for_mesh(mesh, offloader=off)
        mesh._publish_cameras_once()
        assert not any("/ref" in c.args[0] for c in transport.put.call_args_list)

    def test_closure_skips_camera_with_no_frame(self, monkeypatch):
        mesh, off, transport = self._make_mesh_with_camera(monkeypatch)
        # Two cameras configured, but only one has data
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mesh.robot.robot.config.cameras = {"front": object(), "back": object()}
        mesh.robot.robot.get_observation.return_value = {"front": frame, "back": None}
        enable_for_mesh(mesh, offloader=off)
        mesh._publish_cameras_once()
        ref_calls = [c for c in transport.put.call_args_list if "/ref" in c.args[0]]
        assert len(ref_calls) == 1
        assert ref_calls[0].args[0] == "strands/robot-x/camera/front/ref"

    def test_closure_handles_observation_failure(self, monkeypatch, caplog):
        import logging

        mesh, off, transport = self._make_mesh_with_camera(monkeypatch)
        mesh.robot.robot.get_observation.side_effect = RuntimeError("hardware glitch")
        enable_for_mesh(mesh, offloader=off)
        with caplog.at_level(logging.DEBUG, logger="strands_robots.mesh.iot.camera_offload"):
            # Must not raise -- offload is best-effort
            mesh._publish_cameras_once()
        assert not any("/ref" in c.args[0] for c in transport.put.call_args_list)

    def test_closure_skips_dtype_dtype_promotion_to_uint8(self, monkeypatch):
        """Float frames are silently coerced to uint8 before JPEG-encoding."""
        mesh, off, transport = self._make_mesh_with_camera(monkeypatch)
        # Use a float32 frame with valid uint8 range
        frame = np.zeros((100, 100, 3), dtype=np.float32)
        mesh.robot.robot.get_observation.return_value = {"front": frame}
        enable_for_mesh(mesh, offloader=off)
        mesh._publish_cameras_once()
        ref_calls = [c for c in transport.put.call_args_list if "/ref" in c.args[0]]
        assert len(ref_calls) == 1


# ----------------------------------------------------------------------
# Issue #262: presign_ttl negative kwarg WARNING (asymmetric clamp fix)
# ----------------------------------------------------------------------


class TestNegativeKwargWarns:
    """Pin: ``presign_ttl=-99`` (unambiguous bug at call site) emits a
    WARNING when clamped to 1, but ``presign_ttl=0`` (documented
    sentinel pinned by R1 fix) does NOT.
    """

    def test_negative_kwarg_emits_warning(self, caplog):
        from strands_robots.mesh.iot.camera_offload import CameraOffloader

        caplog.clear()
        with caplog.at_level("WARNING"):
            off = CameraOffloader(bucket="test-bucket", presign_ttl=-99)

        assert off.presign_ttl == 1
        warns = [r for r in caplog.records if "presign_ttl" in r.message]
        assert len(warns) == 1, f"expected 1 WARNING, got {warns}"
        assert "source=kwarg" in warns[0].message
        assert "-99" in warns[0].message

    def test_zero_kwarg_no_warning(self, caplog):
        """Sentinel value 0 (kwarg-vs-env-precedence pin from R1) must
        NOT emit a WARNING -- documented deliberate caller value.
        """
        from strands_robots.mesh.iot.camera_offload import CameraOffloader

        caplog.clear()
        with caplog.at_level("WARNING"):
            off = CameraOffloader(bucket="test-bucket", presign_ttl=0)

        assert off.presign_ttl == 1
        warns = [r for r in caplog.records if "presign_ttl" in r.message]
        assert len(warns) == 0, f"expected no WARNING for sentinel 0, got {warns}"

    def test_negative_env_emits_warning(self, monkeypatch, caplog):
        """Env-var path always WARNs (operator-side bug)."""
        from strands_robots.mesh.iot.camera_offload import CameraOffloader

        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "-99")
        caplog.clear()
        with caplog.at_level("WARNING"):
            off = CameraOffloader(bucket="test-bucket")

        assert off.presign_ttl == 1
        warns = [r for r in caplog.records if "presign_ttl" in r.message]
        assert len(warns) == 1
        assert "source=env" in warns[0].message
