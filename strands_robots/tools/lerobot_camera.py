#!/usr/bin/env python3
"""
LeRobot-based camera tool for Strands agents.
Leverages LeRobot's OpenCV and RealSense camera classes for professional camera management.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np

try:
    import cv2
    from lerobot.cameras.camera import Camera
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import (
        ColorMode,
        Cv2Rotation,
        OpenCVCameraConfig,
    )

    # Try to import RealSense camera if available.
    # Suppress lerobot's root-logger noise ("Could not import realsense")
    # that fires when pyrealsense2 is not installed.
    _root = logging.getLogger()
    _prev_level = _root.level
    _root.setLevel(logging.CRITICAL)
    try:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import (
            RealSenseCameraConfig,
        )

        REALSENSE_AVAILABLE = True
    except ImportError:
        REALSENSE_AVAILABLE = False
        RealSenseCamera = None
        RealSenseCameraConfig = None
    finally:
        _root.setLevel(_prev_level)

except ImportError as e:
    raise ImportError(f"LeRobot camera modules not available: {e}")

from strands import tool

logger = logging.getLogger(__name__)


def _frame_to_image_content(frame: np.ndarray, format: str = "jpg") -> Dict[str, Any]:
    """Convert a numpy frame to image content format for Converse API."""
    try:
        # Convert RGB to BGR for OpenCV encoding
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr_frame = frame

        # Encode frame to specified format
        if format.lower() in ["jpg", "jpeg"]:
            success, encoded_img = cv2.imencode(".jpg", bgr_frame)
            image_format = "jpeg"
        elif format.lower() == "png":
            success, encoded_img = cv2.imencode(".png", bgr_frame)
            image_format = "png"
        else:
            success, encoded_img = cv2.imencode(".jpg", bgr_frame)  # Default to JPEG
            image_format = "jpeg"

        if not success:
            raise ValueError("Failed to encode frame")

        # Convert to bytes
        image_bytes = encoded_img.tobytes()

        return {"image": {"format": image_format, "source": {"bytes": image_bytes}}}

    except Exception as e:
        logger.error("Failed to convert frame to image content: %s", e)
        return {"text": f"❌ Failed to encode image: {str(e)}"}


@tool
def lerobot_camera(
    action: str = "list",
    camera_type: str = "opencv",
    camera_id: Union[int, str] = None,
    save_path: str = "./lerobot_captures",
    filename: str = None,
    camera_ids: List[Union[int, str]] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    color_mode: str = "RGB",
    rotation: str = "NO_ROTATION",
    format: str = "jpg",
    capture_duration: float = 5.0,
    preview_duration: float = 10.0,
    async_mode: bool = False,
    timeout_ms: float = 1000,
    warmup: bool = True,
    save_config: bool = False,
) -> Dict[str, Any]:
    """Advanced LeRobot-based camera tool for professional camera management.

    Args:
        action: Action to perform
            - "discover": Discover all available cameras (OpenCV + RealSense)
            - "list": List camera details and configurations
            - "capture": Capture single image from camera
            - "capture_batch": Capture from multiple cameras simultaneously
            - "record": Record video sequence from camera
            - "preview": Show live preview from camera
            - "test": Test camera functionality and performance
            - "configure": Configure camera settings and save
        camera_type: Camera type ("opencv" or "realsense")
        camera_id: Camera device ID (int for index, str for path like "/dev/video0")
        save_path: Directory to save captured images/videos
        filename: Custom filename (without extension)
        camera_ids: List of camera IDs for batch operations
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        color_mode: Color mode ("RGB" or "BGR")
        rotation: Image rotation ("NO_ROTATION", "ROTATE_90", "ROTATE_180", "ROTATE_270")
        format: Image format ("jpg", "png", "bmp")
        capture_duration: Duration for video recording (seconds)
        preview_duration: Duration for preview display (seconds)
        async_mode: Use async reading for better performance
        timeout_ms: Timeout for async operations (milliseconds)
        warmup: Enable camera warmup on connection
        save_config: Save camera configuration to file

    Returns:
        Dict containing status and detailed camera operation results
    """

    try:
        if action == "discover":
            return _discover_cameras()
        elif action == "list":
            return _list_camera_details(camera_type, camera_id)
        elif action == "capture":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "❌ camera_id required for capture action"}],
                }
            return _capture_single_image(
                camera_type,
                camera_id,
                save_path,
                filename,
                width,
                height,
                fps,
                color_mode,
                rotation,
                format,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "capture_batch":
            if not camera_ids:
                camera_ids = [0, "/dev/video4"]  # Default robot cameras
            return _capture_batch_images(
                camera_type,
                camera_ids,
                save_path,
                filename,
                width,
                height,
                fps,
                color_mode,
                rotation,
                format,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "record":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "❌ camera_id required for record action"}],
                }
            return _record_video_sequence(
                camera_type,
                camera_id,
                save_path,
                filename,
                width,
                height,
                fps,
                color_mode,
                rotation,
                capture_duration,
                async_mode,
                warmup,
            )
        elif action == "preview":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "❌ camera_id required for preview action"}],
                }
            return _preview_camera_live(
                camera_type,
                camera_id,
                width,
                height,
                fps,
                color_mode,
                rotation,
                preview_duration,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "test":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "❌ camera_id required for test action"}],
                }
            return _test_camera_performance(
                camera_type,
                camera_id,
                width,
                height,
                fps,
                color_mode,
                rotation,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "configure":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "❌ camera_id required for configure action"}],
                }
            return _configure_camera_settings(
                camera_type,
                camera_id,
                width,
                height,
                fps,
                color_mode,
                rotation,
                save_path,
                save_config,
                warmup,
            )
        else:
            return {
                "status": "error",
                "content": [{"text": f"❌ Unknown action: {action}"}],
            }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Camera operation failed: {str(e)}"}],
        }


def _discover_cameras() -> Dict[str, Any]:
    """Discover all available cameras using LeRobot's detection methods."""
    try:
        # Discover OpenCV cameras
        opencv_cameras = OpenCVCamera.find_cameras()

        # Discover RealSense cameras if available
        realsense_cameras = []
        if REALSENSE_AVAILABLE:
            try:
                realsense_cameras = RealSenseCamera.find_cameras()
            except Exception as e:
                logger.warning("RealSense camera discovery failed: %s", e)

        total_cameras = len(opencv_cameras) + len(realsense_cameras)

        # Format discovery results
        discovery_info = []
        discovery_info.append("🔍 **Camera Discovery Results**\n")

        if opencv_cameras:
            discovery_info.append("📹 **OpenCV Cameras:**")
            for i, cam in enumerate(opencv_cameras):
                profile = cam.get("default_stream_profile", {})
                discovery_info.append(
                    f"  • **{cam.get('name', 'Unknown')}**\n"
                    f"    - ID: `{cam.get('id', 'N/A')}`\n"
                    f"    - Backend: {cam.get('backend_api', 'N/A')}\n"
                    f"    - Resolution: {profile.get('width', '?')}x{profile.get('height', '?')}\n"
                    f"    - FPS: {profile.get('fps', '?')}\n"
                    f"    - Format: {profile.get('format', '?')}"
                )
            discovery_info.append("")

        if realsense_cameras:
            discovery_info.append("🎯 **RealSense Cameras:**")
            for i, cam in enumerate(realsense_cameras):
                discovery_info.append(
                    f"  • **{cam.get('name', 'Unknown')}**\n"
                    f"    - Serial: `{cam.get('serial_number', 'N/A')}`\n"
                    f"    - Type: {cam.get('type', 'N/A')}"
                )
            discovery_info.append("")

        if total_cameras == 0:
            discovery_info.append("❌ **No cameras detected**")
        else:
            discovery_info.append(f"✅ **Total: {total_cameras} cameras found**")
            discovery_info.append(f"   - OpenCV: {len(opencv_cameras)}")
            discovery_info.append(f"   - RealSense: {len(realsense_cameras)}")

        return {"status": "success", "content": [{"text": "\n".join(discovery_info)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Camera discovery failed: {str(e)}"}],
        }


def _list_camera_details(
    camera_type: str, camera_id: Union[int, str] = None
) -> Dict[str, Any]:
    """List detailed camera information and configurations."""
    try:
        details = []
        details.append("📋 **Camera Configuration Details**\n")

        if camera_type.lower() == "opencv":
            details.append("🎥 **OpenCV Camera System:**")
            details.append(f"   - Backend: {_get_opencv_backend_name()}")
            details.append(f"   - Version: {cv2.__version__}")
            details.append("   - Available color modes: RGB, BGR")
            details.append("   - Supported rotations: 0°, 90°, 180°, 270°")
            details.append("   - Async reading: ✅ Supported")
            details.append("")

            if camera_id is not None:
                try:
                    config = OpenCVCameraConfig(
                        index_or_path=camera_id, fps=30, width=640, height=480
                    )
                    camera = OpenCVCamera(config)
                    camera.connect(warmup=False)

                    details.append(f"📸 **Camera {camera_id} Details:**")
                    details.append("   - Connection: ✅ Success")
                    details.append(f"   - Actual FPS: {camera.fps}")
                    details.append(f"   - Resolution: {camera.width}x{camera.height}")
                    details.append(f"   - Color Mode: {camera.color_mode.value}")

                    camera.disconnect()

                except Exception as e:
                    details.append(f"📸 **Camera {camera_id} Details:**")
                    details.append(f"   - Connection: ❌ Failed ({str(e)})")

        elif camera_type.lower() == "realsense" and REALSENSE_AVAILABLE:
            details.append("🎯 **RealSense Camera System:**")
            details.append("   - SDK Available: ✅ Yes")
            details.append("   - Depth Support: ✅ Yes")
            details.append("   - Multiple streams: Color, Depth, Infrared")
            details.append("   - Advanced features: Post-processing, alignment")

        else:
            if not REALSENSE_AVAILABLE and camera_type.lower() == "realsense":
                details.append("🎯 **RealSense Camera System:**")
                details.append("   - SDK Available: ❌ Not installed")
                details.append("   - Install with: `pip install pyrealsense2`")
            else:
                details.append(f"❌ **Unknown camera type: {camera_type}**")

        return {"status": "success", "content": [{"text": "\n".join(details)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Camera details failed: {str(e)}"}],
        }


def _capture_single_image(
    camera_type: str,
    camera_id: Union[int, str],
    save_path: str,
    filename: str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    format: str,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> Dict[str, Any]:
    """Capture a single image using LeRobot camera system."""
    try:
        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cam_name = str(camera_id).replace("/dev/", "").replace("/", "_")
            filename = f"lerobot_{camera_type}_{cam_name}_{timestamp}"

        file_path = os.path.join(save_path, f"{filename}.{format}")

        # Create camera configuration
        camera = _create_camera(
            camera_type, camera_id, width, height, fps, color_mode, rotation
        )

        # Connect and capture
        start_time = time.time()
        camera.connect(warmup=warmup)
        connect_time = time.time() - start_time

        start_time = time.time()
        if async_mode:
            frame = camera.async_read(timeout_ms=timeout_ms)
        else:
            frame = camera.read()
        capture_time = time.time() - start_time

        # Save image
        success = cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        camera.disconnect()

        if not success:
            return {
                "status": "error",
                "content": [{"text": f"❌ Failed to save image: {file_path}"}],
            }

        # Get image info
        img_height, img_width = frame.shape[:2]
        file_size = os.path.getsize(file_path)

        result_info = [
            "📸 **Image Capture Success!**",
            f"🎥 Camera: {camera_type.upper()} @ {camera_id}",
            f"💾 Saved: `{file_path}`",
            f"📐 Resolution: {img_width}x{img_height}",
            f"💿 File size: {file_size:,} bytes",
            f"⚡ Connect time: {connect_time:.3f}s",
            f"📷 Capture time: {capture_time:.3f}s",
            f"🔄 Async mode: {'✅' if async_mode else '❌'}",
            f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        # Create image content for Converse API
        image_content = _frame_to_image_content(frame, format)

        return {
            "status": "success",
            "content": [{"text": "\n".join(result_info)}, image_content],
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Image capture failed: {str(e)}"}],
        }


def _capture_batch_images(
    camera_type: str,
    camera_ids: List[Union[int, str]],
    save_path: str,
    filename: str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    format: str,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> Dict[str, Any]:
    """Capture images from multiple cameras simultaneously."""
    try:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = []
        successful_captures = 0
        total_time = time.time()

        def capture_single_camera(cam_id):
            try:
                # Generate unique filename for this camera
                cam_name = str(cam_id).replace("/dev/", "").replace("/", "_")
                if filename:
                    cam_filename = f"{filename}_{cam_name}_{timestamp}"
                else:
                    cam_filename = f"batch_{camera_type}_{cam_name}_{timestamp}"

                file_path = os.path.join(save_path, f"{cam_filename}.{format}")

                # Create and use camera
                camera = _create_camera(
                    camera_type, cam_id, width, height, fps, color_mode, rotation
                )

                start_time = time.time()
                camera.connect(warmup=warmup)

                if async_mode:
                    frame = camera.async_read(timeout_ms=timeout_ms)
                else:
                    frame = camera.read()

                capture_time = time.time() - start_time

                # Save image
                success = cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                camera.disconnect()

                if success:
                    file_size = os.path.getsize(file_path)
                    return {
                        "camera_id": cam_id,
                        "status": "success",
                        "file_path": file_path,
                        "file_size": file_size,
                        "capture_time": capture_time,
                        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                        "frame": frame,  # Include frame for image content
                    }
                else:
                    return {
                        "camera_id": cam_id,
                        "status": "error",
                        "message": "Failed to save image",
                    }

            except Exception as e:
                return {"camera_id": cam_id, "status": "error", "message": str(e)}

        # Use ThreadPoolExecutor for parallel capture
        with ThreadPoolExecutor(max_workers=len(camera_ids)) as executor:
            future_to_camera = {
                executor.submit(capture_single_camera, cam_id): cam_id
                for cam_id in camera_ids
            }

            for future in as_completed(future_to_camera):
                result = future.result()
                results.append(result)
                if result["status"] == "success":
                    successful_captures += 1

        total_time = time.time() - total_time

        # Format results and prepare content list
        result_info = ["📸 **Batch Camera Capture Results:**", ""]
        content_list = []

        for result in results:
            if result["status"] == "success":
                result_info.append(
                    f"✅ **{result['camera_id']}**: {result['resolution']} "
                    f"({result['file_size']:,} bytes, {result['capture_time']:.3f}s)"
                )
                # Add image content if frame is available
                if "frame" in result:
                    image_content = _frame_to_image_content(result["frame"], format)
                    content_list.append(image_content)
            else:
                result_info.append(f"❌ **{result['camera_id']}**: {result['message']}")

        result_info.extend(
            [
                "",
                "📊 **Summary:**",
                f"   - Success: {successful_captures}/{len(camera_ids)} cameras",
                f"   - Total time: {total_time:.3f}s",
                f"   - Save path: `{save_path}`",
                f"   - Async mode: {'✅' if async_mode else '❌'}",
            ]
        )

        # Add text summary first, then all images
        final_content = [{"text": "\n".join(result_info)}] + content_list

        return {
            "status": "success" if successful_captures > 0 else "error",
            "content": final_content,
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Batch capture failed: {str(e)}"}],
        }


def _record_video_sequence(
    camera_type: str,
    camera_id: Union[int, str],
    save_path: str,
    filename: str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    capture_duration: float,
    async_mode: bool,
    warmup: bool,
) -> Dict[str, Any]:
    """Record a video sequence from camera."""
    try:
        os.makedirs(save_path, exist_ok=True)

        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cam_name = str(camera_id).replace("/dev/", "").replace("/", "_")
            filename = f"lerobot_video_{camera_type}_{cam_name}_{timestamp}"

        video_path = os.path.join(save_path, f"{filename}.mp4")

        # Create camera
        camera = _create_camera(
            camera_type, camera_id, width, height, fps, color_mode, rotation
        )
        camera.connect(warmup=warmup)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        frames_captured = 0
        start_time = time.time()
        target_frames = int(fps * capture_duration)

        try:
            while frames_captured < target_frames:
                if async_mode:
                    frame = camera.async_read(timeout_ms=1000)
                else:
                    frame = camera.read()

                # Convert RGB to BGR for video writer
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
                frames_captured += 1

                # Progress update every second
                if frames_captured % fps == 0:
                    elapsed = time.time() - start_time
                    remaining = capture_duration - elapsed
                    logger.info(
                        "Recording... %.1fs / %.1fs (%.1fs remaining)",
                        elapsed, capture_duration, remaining,
                    )

        finally:
            video_writer.release()
            camera.disconnect()

        actual_duration = time.time() - start_time
        file_size = os.path.getsize(video_path)

        result_info = [
            "🎬 **Video Recording Complete!**",
            f"🎥 Camera: {camera_type.upper()} @ {camera_id}",
            f"💾 Saved: `{video_path}`",
            f"📐 Resolution: {width}x{height}",
            f"🎞️  Frames: {frames_captured} @ {fps} FPS",
            f"⏱️  Duration: {actual_duration:.2f}s (target: {capture_duration:.2f}s)",
            f"💿 File size: {file_size:,} bytes",
            f"🔄 Async mode: {'✅' if async_mode else '❌'}",
            f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        return {"status": "success", "content": [{"text": "\n".join(result_info)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Video recording failed: {str(e)}"}],
        }


def _preview_camera_live(
    camera_type: str,
    camera_id: Union[int, str],
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    preview_duration: float,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> Dict[str, Any]:
    """Show live preview from camera."""
    try:
        camera = _create_camera(
            camera_type, camera_id, width, height, fps, color_mode, rotation
        )
        camera.connect(warmup=warmup)

        frames_displayed = 0
        start_time = time.time()
        fps_counter_start = time.time()
        fps_frame_count = 0

        logger.info("Starting live preview from %s camera %s", camera_type.upper(), camera_id)
        logger.info("Duration: %ds | Press 'q' to quit early", preview_duration)

        try:
            while time.time() - start_time < preview_duration:
                frame_start = time.time()

                if async_mode:
                    frame = camera.async_read(timeout_ms=timeout_ms)
                else:
                    frame = camera.read()

                # Convert RGB to BGR for display
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Add info overlay
                info_text = (
                    f"Camera: {camera_id} | Frame: {frames_displayed} | FPS: {fps}"
                )
                cv2.putText(
                    bgr_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(f"LeRobot Camera Preview - {camera_id}", bgr_frame)

                frames_displayed += 1
                fps_frame_count += 1

                # Calculate and display FPS every second
                if time.time() - fps_counter_start >= 1.0:
                    actual_fps = fps_frame_count / (time.time() - fps_counter_start)
                    logger.info("Live FPS: %.1f | Frames: %d", actual_fps, frames_displayed)
                    fps_counter_start = time.time()
                    fps_frame_count = 0

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Preview stopped by user")
                    break

                # Maintain target FPS
                frame_time = time.time() - frame_start
                target_frame_time = 1.0 / fps
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)

        finally:
            cv2.destroyAllWindows()
            camera.disconnect()

        actual_duration = time.time() - start_time
        avg_fps = frames_displayed / actual_duration if actual_duration > 0 else 0

        result_info = [
            "📺 **Live Preview Complete!**",
            f"🎥 Camera: {camera_type.upper()} @ {camera_id}",
            f"📐 Resolution: {width}x{height}",
            f"🎞️  Frames displayed: {frames_displayed}",
            f"⏱️  Duration: {actual_duration:.2f}s",
            f"📊 Average FPS: {avg_fps:.2f}",
            f"🎯 Target FPS: {fps}",
            f"🔄 Async mode: {'✅' if async_mode else '❌'}",
        ]

        return {"status": "success", "content": [{"text": "\n".join(result_info)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Preview failed: {str(e)}"}],
        }


def _test_camera_performance(
    camera_type: str,
    camera_id: Union[int, str],
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> Dict[str, Any]:
    """Test camera performance and capabilities."""
    try:
        test_results = []
        test_results.append("🧪 **Camera Performance Test**\n")

        # Connection test
        start_time = time.time()
        camera = _create_camera(
            camera_type, camera_id, width, height, fps, color_mode, rotation
        )
        camera.connect(warmup=warmup)
        connect_time = time.time() - start_time

        test_results.append(f"✅ **Connection Test**: {connect_time:.3f}s")

        # Frame capture test (sync)
        capture_times = []
        for i in range(10):
            start_time = time.time()
            frame = camera.read()
            capture_time = time.time() - start_time
            capture_times.append(capture_time)

        avg_sync_time = np.mean(capture_times)
        min_sync_time = np.min(capture_times)
        max_sync_time = np.max(capture_times)

        test_results.append("📷 **Sync Capture (10 frames)**:")
        test_results.append(f"   - Average: {avg_sync_time:.3f}s")
        test_results.append(f"   - Min: {min_sync_time:.3f}s")
        test_results.append(f"   - Max: {max_sync_time:.3f}s")
        test_results.append(f"   - Est. FPS: {1/avg_sync_time:.1f}")

        # Frame capture test (async)
        if async_mode:
            async_times = []
            for i in range(10):
                start_time = time.time()
                frame = camera.async_read(timeout_ms=timeout_ms)
                async_time = time.time() - start_time
                async_times.append(async_time)

            avg_async_time = np.mean(async_times)
            min_async_time = np.min(async_times)
            max_async_time = np.max(async_times)

            test_results.append("⚡ **Async Capture (10 frames)**:")
            test_results.append(f"   - Average: {avg_async_time:.3f}s")
            test_results.append(f"   - Min: {min_async_time:.3f}s")
            test_results.append(f"   - Max: {max_async_time:.3f}s")
            test_results.append(f"   - Est. FPS: {1/avg_async_time:.1f}")
            test_results.append(f"   - Speedup: {avg_sync_time/avg_async_time:.2f}x")

        # Frame properties test
        test_results.append("📊 **Frame Properties**:")
        test_results.append(f"   - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        test_results.append(f"   - Channels: {frame.shape[2]}")
        test_results.append(f"   - Data type: {frame.dtype}")
        test_results.append(f"   - Memory size: {frame.nbytes:,} bytes")

        # Camera properties
        if hasattr(camera, "fps"):
            test_results.append("⚙️  **Camera Configuration**:")
            test_results.append(f"   - Configured FPS: {camera.fps}")
            test_results.append(f"   - Resolution: {camera.width}x{camera.height}")
            test_results.append(f"   - Color mode: {camera.color_mode.value}")

        camera.disconnect()

        test_results.append("\n🎯 **Performance Summary**:")
        test_results.append(
            f"   - Connection: {'✅ Fast' if connect_time < 1.0 else '⚠️ Slow'} ({connect_time:.3f}s)"
        )
        test_results.append(
            f"   - Sync capture: {'✅ Good' if avg_sync_time < 0.1 else '⚠️ Slow'} ({avg_sync_time:.3f}s)"
        )
        if async_mode:
            test_results.append(
                f"   - Async capture: {'✅ Better' if avg_async_time < avg_sync_time else '❌ Worse'}"
                f" ({avg_async_time:.3f}s)"
            )
        test_results.append(
            f"   - Frame rate: {'✅ Stable' if max_sync_time - min_sync_time < 0.05 else '⚠️ Variable'}"
        )

        return {"status": "success", "content": [{"text": "\n".join(test_results)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Performance test failed: {str(e)}"}],
        }


def _configure_camera_settings(
    camera_type: str,
    camera_id: Union[int, str],
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    save_path: str,
    save_config: bool,
    warmup: bool,
) -> Dict[str, Any]:
    """Configure camera settings and optionally save configuration."""
    try:
        camera = _create_camera(
            camera_type, camera_id, width, height, fps, color_mode, rotation
        )
        camera.connect(warmup=warmup)

        # Get actual camera properties
        actual_config = {
            "camera_type": camera_type,
            "camera_id": camera_id,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "color_mode": camera.color_mode.value,
            "warmup": warmup,
            "timestamp": datetime.now().isoformat(),
        }

        if hasattr(camera, "rotation") and camera.rotation is not None:
            actual_config["rotation"] = rotation

        config_info = [
            "⚙️  **Camera Configuration**",
            f"🎥 Camera: {camera_type.upper()} @ {camera_id}",
            f"📐 Resolution: {actual_config['width']}x{actual_config['height']}",
            f"🎞️  FPS: {actual_config['fps']}",
            f"🎨 Color mode: {actual_config['color_mode']}",
            f"🔄 Rotation: {actual_config.get('rotation', 'NO_ROTATION')}",
            f"🔧 Warmup: {'✅' if warmup else '❌'}",
        ]

        # Save configuration if requested
        if save_config:
            os.makedirs(save_path, exist_ok=True)
            cam_id_safe = str(camera_id).replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_filename = (
                f"camera_config_{camera_type}_{cam_id_safe}_{timestamp}.json"
            )
            config_path = os.path.join(save_path, config_filename)

            with open(config_path, "w") as f:
                json.dump(actual_config, f, indent=2)

            config_info.extend(
                [
                    "",
                    "💾 **Configuration Saved**:",
                    f"   - File: `{config_path}`",
                    "   - Format: JSON",
                ]
            )

        camera.disconnect()

        return {"status": "success", "content": [{"text": "\n".join(config_info)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Configuration failed: {str(e)}"}],
        }


def _create_camera(
    camera_type: str,
    camera_id: Union[int, str],
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
) -> Camera:
    """Create and configure a camera instance."""

    if camera_type.lower() == "opencv":
        # Convert string enums to proper types
        color_mode_enum = (
            ColorMode.RGB if color_mode.upper() == "RGB" else ColorMode.BGR
        )

        rotation_map = {
            "NO_ROTATION": Cv2Rotation.NO_ROTATION,
            "ROTATE_90": Cv2Rotation.ROTATE_90,
            "ROTATE_180": Cv2Rotation.ROTATE_180,
            "ROTATE_270": Cv2Rotation.ROTATE_270,
        }
        rotation_enum = rotation_map.get(rotation.upper(), Cv2Rotation.NO_ROTATION)

        config = OpenCVCameraConfig(
            index_or_path=camera_id,
            fps=fps,
            width=width,
            height=height,
            color_mode=color_mode_enum,
            rotation=rotation_enum,
        )
        return OpenCVCamera(config)

    elif camera_type.lower() == "realsense" and REALSENSE_AVAILABLE:
        config = RealSenseCameraConfig(
            serial_number_or_name=str(camera_id), fps=fps, width=width, height=height
        )
        return RealSenseCamera(config)

    else:
        raise ValueError(f"Unsupported camera type: {camera_type}")


def _get_opencv_backend_name() -> str:
    """Get the name of the current OpenCV backend."""
    backend = cv2.CAP_ANY
    backend_names = {
        cv2.CAP_V4L2: "V4L2",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_AVFOUNDATION: "AVFoundation",
        cv2.CAP_ANY: "Auto",
    }
    return backend_names.get(backend, "Unknown")
