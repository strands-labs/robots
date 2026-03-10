"""Tests for strands_robots/video.py — VideoEncoder + encode_frames + get_video_info.

Coverage target: ~90%+ of video.py (285 lines).
All tests run on CPU without PyAV or imageio installed.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. Syntax & Import
# ---------------------------------------------------------------------------


class TestVideoSyntax:
    """Verify module loads and exports expected names."""

    def test_module_imports(self):
        from strands_robots import video

        assert hasattr(video, "VideoEncoder")
        assert hasattr(video, "encode_frames")
        assert hasattr(video, "get_video_info")

    def test_constants(self):
        from strands_robots.video import DEFAULT_CODEC, SUPPORTED_CODECS

        assert "h264" in SUPPORTED_CODECS
        assert DEFAULT_CODEC == "h264"

    def test_check_pyav_function_exists(self):
        from strands_robots.video import _check_pyav

        # Just verify it returns a bool
        assert isinstance(_check_pyav(), bool)

    def test_check_imageio_function_exists(self):
        from strands_robots.video import _check_imageio

        assert isinstance(_check_imageio(), bool)


# ---------------------------------------------------------------------------
# 2. Import probes
# ---------------------------------------------------------------------------


class TestImportProbes:
    """Test _check_pyav / _check_imageio."""

    def test_check_pyav_without_av(self):

        with patch.dict("sys.modules", {"av": None}):
            import importlib

            import strands_robots.video as vid

            importlib.reload(vid)
            assert vid._check_pyav() is False

    def test_check_pyav_with_av(self):
        with patch.dict("sys.modules", {"av": MagicMock()}):
            import importlib

            import strands_robots.video as vid

            importlib.reload(vid)
            assert vid._check_pyav() is True

    def test_check_imageio_without(self):
        with patch.dict("sys.modules", {"imageio": None}):
            import importlib

            import strands_robots.video as vid

            importlib.reload(vid)
            assert vid._check_imageio() is False

    def test_check_imageio_with(self):
        with patch.dict("sys.modules", {"imageio": MagicMock()}):
            import importlib

            import strands_robots.video as vid

            importlib.reload(vid)
            assert vid._check_imageio() is True


# ---------------------------------------------------------------------------
# 3. VideoEncoder Init
# ---------------------------------------------------------------------------


class TestVideoEncoderInit:
    """VideoEncoder constructor tests."""

    def test_default_values(self):
        from strands_robots.video import VideoEncoder

        enc = VideoEncoder("output.mp4")
        assert enc.output_path == "output.mp4"
        assert enc.fps == 30
        assert enc.codec == "h264"
        assert enc.crf == 23
        assert enc.pix_fmt == "yuv420p"
        assert enc.frame_count == 0
        assert enc._writer is None
        assert enc._backend is None

    def test_custom_values(self):
        from strands_robots.video import VideoEncoder

        enc = VideoEncoder("out.mkv", fps=60, codec="hevc", crf=18, pix_fmt="yuv444p")
        assert enc.fps == 60
        assert enc.codec == "hevc"
        assert enc.crf == 18
        assert enc.pix_fmt == "yuv444p"

    def test_context_manager(self):
        from strands_robots.video import VideoEncoder

        with VideoEncoder("test.mp4") as enc:
            assert enc is not None
            assert enc.output_path == "test.mp4"

    def test_get_info(self):
        from strands_robots.video import VideoEncoder

        enc = VideoEncoder("output.mp4", fps=24, codec="hevc")
        info = enc.get_info()
        assert info["output_path"] == "output.mp4"
        assert info["fps"] == 24
        assert info["codec"] == "hevc"
        assert info["backend"] is None
        assert info["frame_count"] == 0

    def test_frame_count_property(self):
        from strands_robots.video import VideoEncoder

        enc = VideoEncoder("test.mp4")
        assert enc.frame_count == 0


# ---------------------------------------------------------------------------
# 4. VideoEncoder with PyAV backend (mocked)
# ---------------------------------------------------------------------------


class TestVideoEncoderPyAV:
    """Test PyAV code paths with mocked av module."""

    def _make_mock_av(self):
        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.options = {}
        mock_stream.encode.return_value = [MagicMock()]  # packets
        mock_container.add_stream.return_value = mock_stream
        mock_av.open.return_value = mock_container
        mock_av.VideoFrame = MagicMock()
        mock_av.VideoFrame.from_ndarray.return_value = MagicMock()
        return mock_av, mock_container, mock_stream

    @patch("strands_robots.video._check_imageio", return_value=False)
    @patch("strands_robots.video._check_pyav", return_value=True)
    def test_pyav_init(self, mock_pyav_check, mock_imageio_check):
        mock_av, mock_container, mock_stream = self._make_mock_av()
        with patch.dict("sys.modules", {"av": mock_av}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                enc.add_frame(frame)
                assert enc._backend == "pyav"
                assert enc.frame_count == 1

    @patch("strands_robots.video._check_imageio", return_value=False)
    @patch("strands_robots.video._check_pyav", return_value=True)
    def test_pyav_multiple_frames(self, mock_pyav_check, mock_imageio_check):
        mock_av, mock_container, mock_stream = self._make_mock_av()
        with patch.dict("sys.modules", {"av": mock_av}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path)
                for i in range(5):
                    frame = np.ones((240, 320, 3), dtype=np.uint8) * i
                    enc.add_frame(frame)
                assert enc.frame_count == 5

    @patch("strands_robots.video._check_imageio", return_value=False)
    @patch("strands_robots.video._check_pyav", return_value=True)
    def test_pyav_grayscale_conversion(self, mock_pyav_check, mock_imageio_check):
        """2D (grayscale) frames should be expanded to 3-channel."""
        mock_av, mock_container, mock_stream = self._make_mock_av()
        with patch.dict("sys.modules", {"av": mock_av}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path)
                gray = np.zeros((240, 320), dtype=np.uint8)
                enc.add_frame(gray)
                # VideoFrame.from_ndarray should have been called with a 3-channel frame
                call_args = mock_av.VideoFrame.from_ndarray.call_args
                frame_arg = call_args[0][0]
                assert frame_arg.shape == (240, 320, 3)

    @patch("strands_robots.video._check_imageio", return_value=False)
    @patch("strands_robots.video._check_pyav", return_value=True)
    def test_pyav_close_flushes(self, mock_pyav_check, mock_imageio_check):
        mock_av, mock_container, mock_stream = self._make_mock_av()
        with patch.dict("sys.modules", {"av": mock_av}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path)
                enc.add_frame(np.zeros((100, 100, 3), dtype=np.uint8))
                enc.close()
                mock_container.close.assert_called_once()

    @patch("strands_robots.video._check_imageio", return_value=False)
    @patch("strands_robots.video._check_pyav", return_value=True)
    def test_pyav_stream_options(self, mock_pyav_check, mock_imageio_check):
        mock_av, mock_container, mock_stream = self._make_mock_av()
        with patch.dict("sys.modules", {"av": mock_av}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path, fps=30, crf=20)
                enc.add_frame(np.zeros((100, 100, 3), dtype=np.uint8))
                # CRF and keyframe interval should be set
                assert mock_stream.options["crf"] == "20"
                assert "g" in mock_stream.options

    @patch("strands_robots.video._check_imageio", return_value=False)
    @patch("strands_robots.video._check_pyav", return_value=True)
    def test_pyav_svtav1_preset(self, mock_pyav_check, mock_imageio_check):
        mock_av, mock_container, mock_stream = self._make_mock_av()
        with patch.dict("sys.modules", {"av": mock_av}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path, codec="libsvtav1")
                enc.add_frame(np.zeros((100, 100, 3), dtype=np.uint8))
                assert mock_stream.options.get("preset") == "12"


# ---------------------------------------------------------------------------
# 5. VideoEncoder with imageio backend (mocked)
# ---------------------------------------------------------------------------


class TestVideoEncoderImageio:
    """Test imageio fallback path."""

    @patch("strands_robots.video._check_imageio", return_value=True)
    @patch("strands_robots.video._check_pyav", return_value=False)
    def test_imageio_fallback(self, mock_pyav_check, mock_imageio_check):
        mock_imageio = MagicMock()
        mock_writer = MagicMock()
        mock_imageio.get_writer.return_value = mock_writer

        with patch.dict("sys.modules", {"imageio": mock_imageio}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path)
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                enc.add_frame(frame)
                assert enc._backend == "imageio"
                assert enc.frame_count == 1

    @patch("strands_robots.video._check_imageio", return_value=True)
    @patch("strands_robots.video._check_pyav", return_value=False)
    def test_imageio_add_frame(self, mock_pyav_check, mock_imageio_check):
        mock_imageio = MagicMock()
        mock_writer = MagicMock()
        mock_imageio.get_writer.return_value = mock_writer

        with patch.dict("sys.modules", {"imageio": mock_imageio}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path)
                frame = np.ones((50, 50, 3), dtype=np.uint8)
                enc.add_frame(frame)
                mock_writer.append_data.assert_called_once()

    @patch("strands_robots.video._check_imageio", return_value=True)
    @patch("strands_robots.video._check_pyav", return_value=False)
    def test_imageio_close(self, mock_pyav_check, mock_imageio_check):
        mock_imageio = MagicMock()
        mock_writer = MagicMock()
        mock_imageio.get_writer.return_value = mock_writer

        with patch.dict("sys.modules", {"imageio": mock_imageio}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path)
                enc.add_frame(np.zeros((50, 50, 3), dtype=np.uint8))
                enc.close()
                mock_writer.close.assert_called_once()


# ---------------------------------------------------------------------------
# 6. VideoEncoder with no backend
# ---------------------------------------------------------------------------


class TestVideoEncoderNoBackend:
    """Test behavior when no video backend is available."""

    @patch("strands_robots.video._check_imageio", return_value=False)
    @patch("strands_robots.video._check_pyav", return_value=False)
    def test_raises_runtime_error(self, mock_pyav_check, mock_imageio_check):
        from strands_robots.video import VideoEncoder

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.mp4")
            enc = VideoEncoder(path)
            with pytest.raises(RuntimeError, match="No video encoder"):
                enc.add_frame(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_close_without_init(self):
        """Close on an uninitialized encoder should not raise."""
        from strands_robots.video import VideoEncoder

        enc = VideoEncoder("test.mp4")
        enc.close()  # Should not raise


# ---------------------------------------------------------------------------
# 7. PyAV init failure falls back to imageio
# ---------------------------------------------------------------------------


class TestVideoEncoderFallback:
    """Test fallback from PyAV to imageio on init failure."""

    @patch("strands_robots.video._check_imageio", return_value=True)
    @patch("strands_robots.video._check_pyav", return_value=True)
    def test_pyav_failure_falls_back(self, mock_pyav_check, mock_imageio_check):
        mock_av = MagicMock()
        mock_av.open.side_effect = RuntimeError("codec not found")

        mock_imageio = MagicMock()
        mock_writer = MagicMock()
        mock_imageio.get_writer.return_value = mock_writer

        with patch.dict("sys.modules", {"av": mock_av, "imageio": mock_imageio}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                enc = VideoEncoder(path)
                enc.add_frame(np.zeros((100, 100, 3), dtype=np.uint8))
                assert enc._backend == "imageio"


# ---------------------------------------------------------------------------
# 8. encode_frames()
# ---------------------------------------------------------------------------


class TestEncodeFrames:
    """Test one-shot encode_frames function."""

    def test_empty_frames(self):
        from strands_robots.video import encode_frames

        result = encode_frames([], "output.mp4")
        assert result["status"] == "error"
        assert "No frames" in result["message"]

    @patch("strands_robots.video._check_imageio", return_value=False)
    @patch("strands_robots.video._check_pyav", return_value=True)
    def test_encode_with_mock(self, mock_pyav, mock_imageio):
        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.options = {}
        mock_stream.encode.return_value = [MagicMock()]
        mock_container.add_stream.return_value = mock_stream
        mock_av.open.return_value = mock_container
        mock_av.VideoFrame = MagicMock()
        mock_av.VideoFrame.from_ndarray.return_value = MagicMock()

        with patch.dict("sys.modules", {"av": mock_av}):
            from strands_robots.video import encode_frames

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.mp4")
                frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3
                result = encode_frames(frames, path, fps=15, codec="h264")
                assert result["frames"] == 3
                assert result["fps"] == 15
                assert result["codec"] == "h264"


# ---------------------------------------------------------------------------
# 9. get_video_info()
# ---------------------------------------------------------------------------


class TestGetVideoInfo:
    """Test get_video_info function."""

    def test_with_mock_pyav(self):
        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.duration = 300
        mock_stream.time_base = 0.01
        mock_stream.average_rate = 30
        mock_stream.width = 640
        mock_stream.height = 480
        mock_stream.codec_context.name = "h264"
        mock_stream.frames = 90
        mock_container.streams.video = [mock_stream]
        mock_av.open.return_value = mock_container

        with patch.dict("sys.modules", {"av": mock_av}):
            from strands_robots.video import get_video_info

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(b"fake_data")
                f.flush()
                try:
                    info = get_video_info(f.name)
                    assert info["width"] == 640
                    assert info["height"] == 480
                    assert info["codec"] == "h264"
                    assert info["frames"] == 90
                finally:
                    os.unlink(f.name)

    def test_without_pyav_fallback(self):
        with patch.dict("sys.modules", {"av": None}):
            import importlib

            import strands_robots.video as vid

            importlib.reload(vid)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(b"fake_data")
                f.flush()
                try:
                    info = vid.get_video_info(f.name)
                    assert "file_size_kb" in info
                    assert "note" in info
                finally:
                    os.unlink(f.name)

    def test_error_handling_with_av_open_failure(self):
        """When av.open() raises, should return error dict."""
        mock_av = MagicMock()
        mock_av.open.side_effect = RuntimeError("corrupt file")

        with patch.dict("sys.modules", {"av": mock_av}):
            import importlib

            import strands_robots.video as vid

            importlib.reload(vid)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(b"not_a_real_video")
                f.flush()
                try:
                    info = vid.get_video_info(f.name)
                    assert "error" in info
                    assert "corrupt" in info["error"]
                finally:
                    os.unlink(f.name)

    def test_without_pyav_nonexistent_file(self):
        """Without av, non-existent file returns error dict (not FileNotFoundError)."""
        with patch.dict("sys.modules", {"av": None}):
            import importlib

            import strands_robots.video as vid

            importlib.reload(vid)
            # Fixed: get_video_info now checks os.path.exists() before os.path.getsize()
            info = vid.get_video_info("/nonexistent/file.mp4")
            assert "error" in info
            assert "File not found" in info["error"]
            assert info["path"] == "/nonexistent/file.mp4"
            assert "note" in info


# ---------------------------------------------------------------------------
# 10. Edge Cases
# ---------------------------------------------------------------------------


class TestVideoEdgeCases:
    """Miscellaneous edge cases."""

    def test_double_close(self):
        """Calling close() twice should not raise."""
        from strands_robots.video import VideoEncoder

        enc = VideoEncoder("test.mp4")
        enc.close()
        enc.close()  # Should not raise

    @patch("strands_robots.video._check_imageio", return_value=True)
    @patch("strands_robots.video._check_pyav", return_value=False)
    def test_directory_creation(self, mock_pyav, mock_imageio):
        mock_io = MagicMock()
        mock_io.get_writer.return_value = MagicMock()

        with patch.dict("sys.modules", {"imageio": mock_io}):
            from strands_robots.video import VideoEncoder

            with tempfile.TemporaryDirectory() as tmpdir:
                nested = os.path.join(tmpdir, "a", "b", "c", "test.mp4")
                enc = VideoEncoder(nested)
                enc.add_frame(np.zeros((50, 50, 3), dtype=np.uint8))
                # Directory should have been created
                assert os.path.isdir(os.path.dirname(nested))

    def test_context_manager_closes_on_exception(self):
        """VideoEncoder as context manager should call close even on exception."""
        from strands_robots.video import VideoEncoder

        enc = VideoEncoder("test.mp4")
        try:
            with enc:
                raise ValueError("test error")
        except ValueError:
            pass
        # close() was called (writer was None, so it's a no-op, but shouldn't crash)
        assert enc._writer is None
