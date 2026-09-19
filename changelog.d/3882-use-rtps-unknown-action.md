### Fixed: `use_rtps` refuses a misspelled action by name

A typo in `action` (say `pubilsh`) used to fall through to the RTPS participant: without cyclonedds installed you were told to `pip install 'strands-robots[ros2]'`, and with it a DDS participant was created before the name was checked. The tool now grades the action first and answers an unknown one with the six verbs it accepts, on any box.
