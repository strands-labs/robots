---
description: Dexterous end-effectors — Allegro, Shadow, LEAP, Robotiq, etc.
---

# Hands

Dexterous end-effectors — Allegro, Shadow, LEAP, Robotiq, etc.

```python
from strands_robots import Robot
sim = Robot("shadow_hand")      # Shadow Hand
sim = Robot("leap_hand")        # LEAP Hand
sim = Robot("robotiq_2f85")     # Robotiq 2F-85 gripper
```

## Catalog

| Name | Description | Joints | Aliases |
|------|-------------|-------:|---------|
| `ability_hand` | PSYONIC Ability Hand (5-finger prosthetic, 11-DOF) | 11 | `psyonic_ability_hand` |
| `aero_hand` | Tetheria Aero Hand Open (16-DOF dexterous) | 16 | `tetheria_aero_hand`, `aero_hand_open` |
| `allegro_hand` | Wonik Allegro Hand (16-DOF dexterous) | 16 | `wonik_allegro` |
| `leap_hand` | LEAP Hand (16-DOF dexterous) | 41 | — |
| `robotiq_2f85` | Robotiq 2F-85 Gripper (2-finger adaptive) | 16 | `robotiq` |
| `robotiq_2f85_v4` | Robotiq 2F-85 v4 Gripper (updated model) | 6 | — |
| `shadow_dexee` | Shadow DexEE Dexterous End-Effector (12-DOF) | 12 | — |
| `shadow_hand` | Shadow Dexterous Hand (24-DOF) | 45 | — |

## Featured renders

### `leap_hand`

![leap_hand](../assets/sim_render_leap_hand.png){ width=400 }

_LEAP Hand (16-DOF dexterous)_

### `robotiq_2f85`

![robotiq_2f85](../assets/sim_render_robotiq_2f85.png){ width=400 }

_Robotiq 2F-85 Gripper (2-finger adaptive)_

### `shadow_hand`

![shadow_hand](../assets/sim_render_shadow_hand.png){ width=400 }

_Shadow Dexterous Hand (24-DOF)_

## See also

- [Arms](arms.md) — pair a hand with an arm via `add_robot`.
- [Custom policies](../policies/custom-policies.md) — high-DOF hand control needs careful action-space design.
- [Bimanual](bimanual.md) — two arms each with a hand.
