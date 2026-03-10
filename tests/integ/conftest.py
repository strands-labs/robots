"""
Shared fixtures and markers for integration tests.

These tests are designed to run on specific hardware:
- Thor: NVIDIA Jetson AGX Thor (self-hosted runner label: thor)
- Isaac Sim: EC2 g6e instance with L40S GPU (self-hosted runner label: isaac-sim)
"""


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "thor: requires Thor device (Jetson AGX Thor)")
    config.addinivalue_line("markers", "isaac_sim: requires Isaac Sim EC2 instance")
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
