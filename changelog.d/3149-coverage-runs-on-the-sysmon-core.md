### Changed: coverage is collected on the PEP 669 core, not the C trace function

`[tool.pytest.ini_options].addopts` carries `--cov=strands_robots`, so every run
of the suite is an instrumented one and the tracer's cost is part of the job's
wall clock. That cost had never been measured. Over 17,102 tests
(`tests/{drivers,mesh,policies,training,rendering}`), same selection, same
reports, one machine, only the core varying: 321.19 s uninstrumented, 567.36 s
on the default `ctrace` core (+76.6%), 347.47 s on `sysmon` (+8.2%). The default
core was spending more time than the tests it measured.

`[tool.coverage.run] core = "sysmon"` now selects coverage.py's
`sys.monitoring` core. The two report the same measurement, compared per file
rather than by the summary percentage: the same 303 files, the same 51,904
statements, the same 605 exclusions, 56% either way, and the same verdict
(4 failed, 17102 passed, 36 skipped). Exactly one line differed, and it differs
between two runs of the *same* core as well -- `strands_robots/drivers/g1.py`'s
FSM-refresher join-timeout branch, whose execution depends on thread
scheduling -- so it is a race in the code under test, not a disagreement
between tracers.

Nothing else changes: no dependency, no coverage lost, and
`--cov-fail-under=80` is untouched. `branch` is deliberately absent from the new
section, because the equivalence was measured for statements, which is what
that gate grades; `tests/test_coverage_runs_on_the_sysmon_core.py` pins the
selection behaviourally (coverage.py registers either as a `sys.monitoring`
tool or as a trace function) and reports if `branch` or `concurrency` ever
appears without the equivalence being re-measured.
