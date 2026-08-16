### Fixed
- `add_robot()` no longer answers a registered robot whose model file is missing with a
  suggestion to try the name it just refused. The resolution failure served two conditions
  through one message - an unknown name (a typo) and a known name whose asset is absent - and
  because the suggestion was drawn from the whole registry, `difflib` ranked the exact match
  first, so the one caller whose spelling was already correct was the one told to fix it
  (`No model found for 'google_robot'. Did you mean: google_robot, ...`). A registered robot now
  reports that it is registered, names the `<dir>/<model_xml>` the resolver looked for and every
  asset search path, and gives the remedy its registry entry implies: an `auto_download: false`
  entry (`google_robot`, `trossen_wxai`) is never fetched automatically, so the file has to be
  placed; anything else is fetchable with `download_assets`. A typo keeps the previous message.
  The suggestion now comes from the shared `close_match_hint` helper rather than a second inline
  `difflib` call, and that helper drops a suggestion identical to the requested name - promoting
  the next candidate rather than shortening the list, so every other unknown-entity message is
  unchanged.
