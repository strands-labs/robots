### Fixed

- **WBC**: the "no ONNX session loaded" refusal named `policy_session=` / `walk_session=` as its remedy, but `WBCPolicy.__init__` absorbs unknown keywords per the provider contract, so a session passed that way was dropped and the identical refusal was raised again. The message now names the `policy_session` / `walk_session` attribute assignment that installs one, and states why the constructor cannot take it.
