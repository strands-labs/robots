# GR00T Policy

!!! info "Coming Soon"
    This page is under active development.

NVIDIA GR00T Vision-Language-Action model integration.

## Setup

Requires Isaac GR00T Docker container on Jetson or GPU server.

```python
from strands_robots import create_policy

policy = create_policy(
    provider="groot",
    data_config="so100_dualcam",
    host="localhost",
    port=8000
)
```
