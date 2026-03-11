#!/usr/bin/env python3
"""
GR00T Inference Service Management Tool

Manages GR00T policy inference services running in Docker containers.
Uses Isaac-GR00T's native inference service for proper ZMQ/HTTP communication.
"""

import socket
import subprocess
import time
from typing import Any, Dict

from strands import tool


@tool
def gr00t_inference(
    action: str,
    checkpoint_path: str = None,
    policy_name: str = None,
    port: int = None,
    data_config: str = "fourier_gr1_arms_only",
    embodiment_tag: str = "gr1",
    denoising_steps: int = 4,
    host: str = "127.0.0.1",
    container_name: str = None,
    timeout: int = 60,
    use_tensorrt: bool = False,
    trt_engine_path: str = "gr00t_engine",
    vit_dtype: str = "fp8",
    llm_dtype: str = "nvfp4",
    dit_dtype: str = "fp8",
    http_server: bool = False,
    api_token: str = None,
) -> Dict[str, Any]:
    """
    Manage GR00T inference services in Docker containers using Isaac-GR00T native scripts.

    Args:
        action: Action to perform
            - "start": Start inference service with checkpoint
            - "stop": Stop inference service on port
            - "status": Check status of service on port
            - "list": List all running services
            - "restart": Restart service with new checkpoint
            - "find_containers": Find available isaac-gr00t containers
        checkpoint_path: Path to model checkpoint (for start/restart)
        policy_name: Name for the policy service (for registration)
        port: Port for inference service (default: 5555 for ZMQ, 8000 for HTTP)
        data_config: GR00T data config (so100_dualcam, so100, fourier_gr1_arms_only, etc.)
        embodiment_tag: Embodiment tag for model
        denoising_steps: Number of denoising steps
        host: Host to bind service to
        container_name: Specific container name
        timeout: Timeout for operations
        use_tensorrt: Whether to use TensorRT for accelerated inference
        trt_engine_path: Path to TensorRT engine directory (default: gr00t_engine)
        vit_dtype: ViT model dtype - "fp16" or "fp8" (default: fp8, only with TensorRT)
        llm_dtype: LLM model dtype - "fp16", "nvfp4", or "fp8" (default: nvfp4, only with TensorRT)
        dit_dtype: DiT model dtype - "fp16" or "fp8" (default: fp8, only with TensorRT)
        http_server: Use HTTP server instead of ZMQ (default: False)
        api_token: API token for authentication (optional)

    Returns:
        Dict with status and information about the operation
    """

    if action == "find_containers":
        return _find_gr00t_containers()
    elif action == "list":
        return _list_running_services()
    elif action == "status":
        if port is None:
            return {"status": "error", "message": "Port required for status check"}
        return _check_service_status(port)
    elif action == "stop":
        if port is None:
            return {"status": "error", "message": "Port required to stop service"}
        return _stop_service(port)
    elif action == "start":
        if checkpoint_path is None:
            return {
                "status": "error",
                "message": "Checkpoint path required to start service",
            }
        if port is None:
            port = 8000 if http_server else 5555
        return _start_service(
            checkpoint_path=checkpoint_path,
            port=port,
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            host=host,
            container_name=container_name,
            policy_name=policy_name,
            timeout=timeout,
            use_tensorrt=use_tensorrt,
            trt_engine_path=trt_engine_path,
            vit_dtype=vit_dtype,
            llm_dtype=llm_dtype,
            dit_dtype=dit_dtype,
            http_server=http_server,
            api_token=api_token,
        )
    elif action == "restart":
        if checkpoint_path is None or port is None:
            return {
                "status": "error",
                "message": "Checkpoint path and port required for restart",
            }
        # Stop existing service and start new one
        _stop_service(port)
        time.sleep(2)  # Brief pause
        return _start_service(
            checkpoint_path=checkpoint_path,
            port=port,
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            host=host,
            container_name=container_name,
            policy_name=policy_name,
            timeout=timeout,
            use_tensorrt=use_tensorrt,
            trt_engine_path=trt_engine_path,
            vit_dtype=vit_dtype,
            llm_dtype=llm_dtype,
            dit_dtype=dit_dtype,
            http_server=http_server,
            api_token=api_token,
        )
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _find_gr00t_containers() -> Dict[str, Any]:
    """Find available Isaac-GR00T containers."""
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        containers = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    name, image, status = parts[0], parts[1], parts[2]
                    ports = parts[3] if len(parts) > 3 else ""

                    is_gr00t_container = "isaac-gr00t" in image.lower() or (
                        "isaac" in image.lower()
                        and ("gr00t" in image.lower() or "jetson" in name.lower())
                    )

                    if is_gr00t_container:
                        containers.append(
                            {
                                "name": name,
                                "image": image,
                                "status": status,
                                "ports": ports,
                            }
                        )

        return {
            "status": "success",
            "containers": containers,
            "message": f"Found {len(containers)} GR00T containers",
        }

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to find containers: {e}"}


def _list_running_services() -> Dict[str, Any]:
    """List all running GR00T inference services by checking common ports."""
    try:
        services = []
        common_ports = [5555, 5556, 5557, 5558, 8000, 8001, 8002, 8003]

        for port in common_ports:
            if _is_service_running(port):
                protocol = "HTTP" if port >= 8000 else "ZMQ"
                services.append(
                    {"port": port, "protocol": protocol, "status": "running"}
                )

        return {
            "status": "success",
            "services": services,
            "message": f"Found {len(services)} running services",
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to list services: {e}"}


def _is_service_running(port: int) -> bool:
    """Check if service is running on port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _check_service_status(port: int) -> Dict[str, Any]:
    """Check status of service on specific port."""
    if _is_service_running(port):
        protocol = "HTTP" if port >= 8000 else "ZMQ"
        return {
            "status": "success",
            "port": port,
            "service_status": "running",
            "protocol": protocol,
        }
    else:
        return {
            "status": "error",
            "port": port,
            "service_status": "not_running",
            "message": f"No service running on port {port}",
        }


def _stop_service(port: int) -> Dict[str, Any]:
    """Stop GR00T inference service running on specific port."""
    try:
        containers_result = _find_gr00t_containers()
        if containers_result["status"] == "success":
            running_containers = [
                c for c in containers_result["containers"] if "Up" in c["status"]
            ]

            for container in running_containers:
                container_name = container["name"]
                try:
                    result = subprocess.run(
                        [
                            "docker",
                            "exec",
                            container_name,
                            "pgrep",
                            "-f",
                            f"inference_service.py.*--port {port}",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        pids = result.stdout.strip().split("\n")
                        for pid in pids:
                            if pid:
                                subprocess.run(
                                    [
                                        "docker",
                                        "exec",
                                        container_name,
                                        "kill",
                                        "-TERM",
                                        pid,
                                    ],
                                    check=True,
                                )

                        time.sleep(2)

                        result = subprocess.run(
                            [
                                "docker",
                                "exec",
                                container_name,
                                "pgrep",
                                "-f",
                                f"inference_service.py.*--port {port}",
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                        )

                        if result.returncode == 0 and result.stdout.strip():
                            pids = result.stdout.strip().split("\n")
                            for pid in pids:
                                if pid:
                                    subprocess.run(
                                        [
                                            "docker",
                                            "exec",
                                            container_name,
                                            "kill",
                                            "-KILL",
                                            pid,
                                        ],
                                        check=True,
                                    )

                        return {
                            "status": "success",
                            "port": port,
                            "container": container_name,
                            "message": f"GR00T service on port {port} stopped in container {container_name}",
                        }

                except subprocess.CalledProcessError:
                    continue

        # Fallback: try host system
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"], capture_output=True, text=True
        )

        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    subprocess.run(["kill", "-TERM", pid], check=True)

            time.sleep(2)

            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"], capture_output=True, text=True
            )

            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-KILL", pid], check=True)

            return {
                "status": "success",
                "port": port,
                "message": f"Service on port {port} stopped",
            }
        else:
            return {
                "status": "success",
                "port": port,
                "message": f"No service running on port {port}",
            }

    except Exception as e:
        return {"status": "error", "message": f"Failed to stop service: {e}"}


def _start_service(
    checkpoint_path: str,
    port: int,
    data_config: str,
    embodiment_tag: str,
    denoising_steps: int,
    host: str,
    container_name: str,
    policy_name: str,
    timeout: int,
    use_tensorrt: bool,
    trt_engine_path: str,
    vit_dtype: str,
    llm_dtype: str,
    dit_dtype: str,
    http_server: bool,
    api_token: str,
) -> Dict[str, Any]:
    """Start GR00T inference service using Isaac-GR00T's native inference service."""
    try:
        # Find container if not specified
        if container_name is None:
            containers = _find_gr00t_containers()
            if containers["status"] == "error":
                return containers

            running_containers = [
                c for c in containers["containers"] if "Up" in c["status"]
            ]
            if not running_containers:
                return {
                    "status": "error",
                    "message": "No running GR00T containers found",
                }

            container_name = running_containers[0]["name"]

        # Build Isaac-GR00T inference service command
        cmd = [
            "docker",
            "exec",
            "-d",
            container_name,
            "python",
            "/opt/Isaac-GR00T/scripts/inference_service.py",
            "--server",
            "--model-path",
            checkpoint_path,
            "--port",
            str(port),
            "--host",
            host,
            "--data-config",
            data_config,
            "--embodiment-tag",
            embodiment_tag,
            "--denoising-steps",
            str(denoising_steps),
        ]

        # Add HTTP server flag if requested
        if http_server:
            cmd.append("--http-server")

        # Add TensorRT flags if enabled
        if use_tensorrt:
            cmd.extend(
                [
                    "--use-tensorrt",
                    "--trt-engine-path",
                    trt_engine_path,
                    "--vit-dtype",
                    vit_dtype,
                    "--llm-dtype",
                    llm_dtype,
                    "--dit-dtype",
                    dit_dtype,
                ]
            )

        # Add API token if provided
        if api_token:
            cmd.extend(["--api-token", api_token])

        # Start service
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Wait for service to start
        protocol = "HTTP" if http_server else "ZMQ"
        start_time = time.time()
        while time.time() - start_time < timeout:
            if _is_service_running(port):
                response = {
                    "status": "success",
                    "port": port,
                    "checkpoint_path": checkpoint_path,
                    "container_name": container_name,
                    "policy_name": policy_name,
                    "protocol": protocol,
                    "data_config": data_config,
                    "embodiment_tag": embodiment_tag,
                    "denoising_steps": denoising_steps,
                    "message": f"GR00T {protocol} service started on port {port}",
                }
                if use_tensorrt:
                    response["tensorrt"] = {
                        "enabled": True,
                        "engine_path": trt_engine_path,
                        "vit_dtype": vit_dtype,
                        "llm_dtype": llm_dtype,
                        "dit_dtype": dit_dtype,
                    }
                if http_server:
                    response["endpoint"] = f"http://{host}:{port}/act"
                return response
            time.sleep(1)

        return {
            "status": "error",
            "message": f"{protocol} service failed to start within {timeout} seconds",
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Failed to start service: {e.stderr or e}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


if __name__ == "__main__":
    print("🐳 GR00T Inference Service Manager (Isaac-GR00T Native)")
    print("Supports ZMQ, HTTP, and TensorRT inference modes")
    print()
    print("Examples:")
    print("  # Start ZMQ server (default)")
    print(
        "  gr00t_inference(action='start', checkpoint_path='/data/checkpoints/model', port=5555)"
    )
    print()
    print("  # Start HTTP server")
    print(
        "  gr00t_inference(action='start', checkpoint_path='/data/checkpoints/model', port=8000, http_server=True)"
    )
    print()
    print("  # Start with TensorRT acceleration")
    print(
        "  gr00t_inference(action='start', checkpoint_path='/data/checkpoints/model', port=5555, use_tensorrt=True)"
    )
    print()
    print("  # Start HTTP + TensorRT")
    print(
        "  gr00t_inference(action='start', checkpoint_path='/data/checkpoints/model',"
        " port=8000, http_server=True, use_tensorrt=True)"
    )
