### Fixed: the Microduck and Reachy Mini ``stop`` verbs report the halt outcome instead of asserting one

:data:`~strands_robots.drivers.base.DRIVER_SURFACE` carries two ways to halt a
robot and they are not the same contract. ``stop`` is the protocol's shutdown
hook, annotated ``-> None`` on every shipped driver, so it cannot carry a
verdict - and both daemon drivers use that freedom to swallow a failure: the
Microduck returns early for a client that is gone and logs an ``OSError`` from
robotd, the Mini logs a daemon that declined. ``stop_task`` returns an envelope
and decides.

The verb an agent reaches is ``stream({"action": "stop"})``, and on both drivers
it built its envelope beside ``await self.stop()``. An envelope written next to a
hook that carries no verdict can only restate the intent, so an agent read
``status="success"`` and text asserting the daemon had been asked - on a driver
whose client was ``None``, where nothing had been written to the socket the text
names:

```text
same driver, same state, one call apart

  stream({"action": "stop"})   status="success"   "asked robotd at /run/... to stop the robot"
  stop_task()                  status="error"     "stop_task: not connected"

  robotd errors on the request
  stream({"action": "stop"})   status="success"   (unchanged text)
  stop_task()                  status="error"     "robotd refused the stop: connection reset by peer"
```

Both branches now return ``stop_task``'s envelope, which is the shape :pr:`2828`
gave :class:`~strands_robots.drivers.g1.G1Driver` for the same reason. A stop
that landed is still ``success`` and still reaches the wire - the healthy path is
unchanged, ``robot.stop`` and ``/api/move/stop`` included - and the Mini's
``motion_stopped`` flag, which was already honest about a refused stop, now
agrees with the envelope beside it.

``tests/drivers/test_agent_stop_verb_reports_the_halt_outcome.py`` grades the
four states and adds a fleet relation so the next driver is held to it on
arrival: a driver whose ``stop_task`` can refuse must report that verdict from
the agent verb. :class:`~strands_robots.drivers.dynamixel.DynamixelDriver` and
:class:`~strands_robots.drivers.feetech.FeetechDriver` are exempt and the
exemption is derived rather than listed - their ``stop_task`` has no refusal path
at all, their serial bus being unwired, so there is no verdict to report.

Why the shipped suites were green: the Mini's own file already argues the
principle. ``test_a_daemon_that_refuses_the_stop_does_not_report_it_as_halted``
carries the comment "Reporting a halt that did not happen is an affirmative lie
on a safety path" and grades ``stop``'s effect on the ``motion_stopped`` *flag*;
``test_stop_task_reports_a_daemon_refusal`` grades ``stop_task``'s *envelope*.
The one surface where that lie was actually told to an agent was the one not
graded, and the Microduck's ``stream`` was uncovered outright.
