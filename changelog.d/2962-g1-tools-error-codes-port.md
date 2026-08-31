### Feature

- Added `strands_robots.tools.g1.g1_list_error_codes` and
  `strands_robots.tools.g1.g1_decode_error_code`, two read-only ``@tool`` verbs
  that snapshot the SDK return-code catalogue already on
  `strands_robots.tools.g1._g1_common.ERR_CODES`. A caller that receives a
  ``refusal_code`` from any other verb in this package can list the catalogue -
  or decode one code by number - through the same tool surface every other verb
  here answers on. No ``unitree_sdk2py`` submodule loads on import (the snapshot
  lives in ``._g1_common`` which never touches it either); the two verbs quote
  every text field verbatim from the catalogue so a re-word of one entry lands
  in the constant once. Refs strands-labs/robots#358.
