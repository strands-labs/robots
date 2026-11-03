"""The identity scheme: a policy that already speaks radians needs no conversion."""

from __future__ import annotations

import numpy as np

from ..description import Description
from .scheme import Units, register


@register("rad")
def rad(description: Description) -> Units:
    keys = description.action_keys
    lo = np.array([description.limits.get(k, (-np.pi, np.pi))[0] for k in keys])
    hi = np.array([description.limits.get(k, (-np.pi, np.pi))[1] for k in keys])
    return Units("rad", keys, lo, hi, lo.copy(), hi.copy())
