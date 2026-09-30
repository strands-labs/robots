# Robot and factory

`Robot(name, mode=...)` is a factory function, not a class. It returns a simulation engine in `mode="sim"` (the default) or a `strands_robots.hardware_robot.Robot` in `mode="real"`. Both expose `act`, `observe`, `run_policy` and `cleanup`, but `run_policy` differs: hardware takes a `Policy` first, sim takes a robot name and a provider, or a built `Policy` as `policy_object=`. Below: every factory keyword and every method of the returned object.

## The factory

::: strands_robots.robot.Robot
    options:
      show_root_heading: true
      heading_level: 3

## The hardware robot

Returned by `Robot(name, mode="real")`. Also usable directly when you already hold a driver.

::: strands_robots.hardware_robot.Robot
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      filters: ["!^_"]

::: strands_robots.hardware_robot.TaskStatus
    options:
      heading_level: 3
      show_root_heading: true

::: strands_robots.hardware_robot.RobotTaskState
    options:
      heading_level: 3
      show_root_heading: true

## Teleoperator

::: strands_robots.teleoperator.Teleoperator
    options:
      show_root_heading: true
      heading_level: 3
