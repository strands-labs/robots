# Custom Policies

!!! info "Coming Soon"
    This page is under active development.

Implement your own policy provider for Strands Robots.

## Policy Interface

```python
from strands_robots.policies import Policy

class MyPolicy(Policy):
    @property
    def provider_name(self) -> str:
        return "my_policy"

    def get_actions(self, observation, instruction):
        # Your inference logic here
        return action_chunk
```
