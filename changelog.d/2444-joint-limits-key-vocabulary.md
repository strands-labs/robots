### Fixed

- **ros2/rtps bridges**: document `joint_limits` with the key spelling the bound is actually matched against. The inbound-command clamp looks each commanded joint's name up in the mapping, so keys have to be the `<motor>.pos` names the bridges publish in `joint_states`; the guides and docstrings wrote `{motor: (min, max)}`, which matches no commanded joint and leaves the arm exactly as unbounded as passing no limits - with no exception and no log line. A reader who copied the documented spelling got no clamp at all.
