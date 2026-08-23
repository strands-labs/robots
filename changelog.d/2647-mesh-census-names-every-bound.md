### Fixed

- **mesh**: `validate_command`'s `Performed checks:` census now states every bound the
  validator enforces. A guard graded one direction of the set equality between the bounds a
  census cites and the bounds the body reads - that a cited constant is one the body loads -
  and left the other direction open, so two of the nine public bounds were enforced and
  stated nowhere. `MAX_PASSTHROUGH_LEN` bounds `turn_id` and `sender_id`, and the census named
  neither field: both are checked on every action rather than per-action, so a 129-character
  correlation id, a printable non-ASCII one and a non-string were all refused with no bound
  for a publisher to size to, on the two fields `Mesh` correlates an RPC turn and keys its
  command-replay cache with. `MAX_PEER_ID_LEN` bounds `robot_name`'s length and each
  `target_joints` key's length, where the census gave the charset without the length and
  bounded the key count without each key. No verdict, message or accepted input changes -
  with the census stripped, the module is byte-identical. The guard gains the reverse
  direction, derived from the body so a bound added later is graded on arrival; only public
  constants are graded, because the charset gates are private regexes whose admitted charset
  the census names in words.
