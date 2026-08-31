### Fixed: a compliant new DDS domain surface no longer fails the shared-domain guard

The structural sweep in `tests/test_dds_domain_id_domain.py` grades every
surface taking a `domain_id` or `ros2_domain`: it must call
`dds_domain_id_error` or forward the value to something that does. Its
non-vacuity companion asserted set equality against a snapshot of the package,
so a surface that obeyed the rule failed the module by existing, and the only
remedy the failure suggested was appending a name to a list. It is now a floor
over `KNOWN_SURFACES`, checked in the one direction that means the scan broke,
and a new cell pins that a newcomer calling the shared guard reads as guarding
while a hand-rolled range check is still named by the sweep.
