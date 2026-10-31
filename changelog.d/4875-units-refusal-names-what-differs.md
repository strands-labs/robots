### Fixed: when `record=` refuses a dataset's units, it names each entry that differs, with both values

`record=` refuses to add a run to an existing dataset whose `meta/units.json` differs from the one the run would
write, before the run's first step. The refusal named only the unit scheme, so where the schemes agree it named no
difference: a dataset recorded before `record=` wrote each joint's LeRobot norm mode into that file was refused with
`stored in units 'lerobot'; this run records 'lerobot'`, and so were a degrees run into a dataset in LeRobot's range
modes and a simulated SO-101 into a hardware SO-101's degrees dataset. The refusal now goes on with each entry that
differs, with the dataset's value and this run's, such as `shoulder_pan.pos mode none (this run: 'range_m100_100')`.
The same runs are accepted and refused as before.
