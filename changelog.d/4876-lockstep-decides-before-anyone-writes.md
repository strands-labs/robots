### Fixed: in lock-step no robot writes a tick another robot refused, and an agent is offered `run(on_miss=)`

`Simulation.run` wrote each robot's targets as soon as that robot had decided them, so a robot listed before the one
that refused a tick (a NaN, an agent's or a dashboard's gate refusing its targets, an e-stop while they decided) had
already put that tick's target into the world, and the world's next step applied it; an e-stop pressed and resumed in
that time let the run step the tick itself. Every robot now decides before any robot writes: a refusal, or an e-stop
on any of them while they decide (even one resumed at once), ends the run with that tick written for none. An e-stop
after the decisions still finds the robots written: a paced run waits for the world's step after the writes, so a
press in that wait is still followed by that one step, the pressed robot holding. `robot.tool`'s spec now offers
`run`'s `on_miss` (`"skip"` or `"fault"`), which it left out.
