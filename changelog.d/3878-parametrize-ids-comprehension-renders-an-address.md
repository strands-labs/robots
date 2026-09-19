### Fixed: a `parametrize` ID rendered by a comprehension is graded like one rendered by a lambda

`tests/mesh/test_iot_provisioning_flag_domain.py` labelled its `UNUSABLE` rows
with `ids=[repr(v)[:24] for v in UNUSABLE]`, and one row is `object()`, whose
default `__repr__` is its address. `<object object at 0x7f5c` is 24 characters,
so the slice kept exactly the prefix that differs between two interpreters, and
under `pytest-xdist` the whole suite aborted at collection - "Different tests
were collected between gw0 and gw1" - before a test ran. This is the second
instance of the class #3823 fixed; its grader read `ids=repr` and
`ids=lambda ...` and was silent on a comprehension, which is a lambda spelled
inline and is now graded by the same predicate. The row is labelled by type, the
spelling `tests/drivers/test_telemetry_coercion_refuses_the_same_non_readings.py`
already uses. Towards #3869.
