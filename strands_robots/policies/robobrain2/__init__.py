# Alias — robobrain2 redirects to robobrain.
# Kept for backwards compatibility; prefer using the registry:
#   policy = PolicyRegistry.resolve("robobrain2")
from strands_robots.policies.robobrain import RobobrainPolicy

__all__ = ["RobobrainPolicy"]
