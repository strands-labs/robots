"""A Policy turns an Observation into an Action.

Anything with ``act(obs, *, instruction=None)`` and ``reset()`` is a policy; a bare
``(obs) -> action`` callable (optionally taking ``instruction=``) is adapted by ``as_policy``.
"""

from __future__ import annotations

from .base import Policy, PolicyLike
from .callable import CallablePolicy, as_policy
from .factory import from_spec
from .perturb import Perturb
from .requirements import Requirements
from .sequence import SequencePolicy
from .source import SourcePolicy

__all__ = [
    "Policy",
    "PolicyLike",
    "CallablePolicy",
    "SequencePolicy",
    "Perturb",
    "as_policy",
    "Requirements",
    "from_spec",
    "SourcePolicy",
]
