### Fixed

- **simulation/mujoco**: the four `set_joint_positions` / `set_joint_velocities` refusals that offer a joint-discovery action now name `get_robot_state`, which the tool schema publishes, instead of `robot_joint_names`, which the agent-facing entry points refuse as a Python-only capability. Following the remedy verbatim used to produce a second refusal that named no alternative, so the discovery step of writing a pose was a dead end from a tool call. `robot_joint_names` is unchanged: still Python-only, still dispatchable.
