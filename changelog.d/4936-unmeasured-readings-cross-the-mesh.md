### Fixed: a move reports success when a joint has no effort or velocity reading

ros2_control reports NaN effort for a joint with no effort state interface. The mesh refuses NaN, so the reply to a
move that happened arrived as "cannot send NaN or infinity over the mesh", and the state stream sent nothing. An
unmeasured velocity or effort now crosses the mesh as `null` and decodes back to NaN. A non-finite position or
command is still refused.
