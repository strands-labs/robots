### Fixed: a G1 DDS decoder publishes only fields its IDL declares

`G1Driver._on_bms` read `getattr(msg, "charge", 0)` into a `charging` flag, and
`unitree_hg.msg.dds_.BmsState_` declares no charge field of any spelling
(`version_high`, `version_low`, `fn`, `cell_vol`, `bmsvoltage`, `current`,
`soc`, `soh`, `temperature`, `cycle`, `manufacturer_date`, `bmsstate`,
`reserve`). `getattr` with a typed default cannot fail, and `False` is a
well-formed answer to "is this pack charging", so every G1 reported
`charging=False` - a claim no measurement backed, indistinguishable from a pack
measured to be discharging, and published on the mesh health wire by
`SensorMixin._read_health`.

`_on_mainboard` had the same shape with the opposite symptom: it read
`sys_state` and `tick`, which `MainBoardState_` does not declare either
(`tick` is declared on `LowState_`), so both reported `None` forever, while the
two vectors the message *does* declare - `value` (`float32[6]`) and `state`
(`uint32[6]`) - were never read at all.

Both decoders now read the names their IDL declares, and every field is read
through `getattr(msg, name, None)` so an absent or renamed field lands `None`
rather than a plausible `0` / `0.0` / `False`. `g1_battery` returns `pct` /
`current` / `cycle` / `t` and no charge flag; `g1_mainboard` returns
`fan_state` / `temperature` / `value` / `state` / `t`. The fleet-health reader
sets `charging` only for a record that carries the reading, instead of
defaulting an absent key to `False`.

Three of the driver's six DDS decoders had a declared-fields grader; the BMS
decoder was one of the two without, which is why the read went unnoticed. It
has one now, in the same three-layer shape (faithful double, frozen
declaration checked against the real SDK where importable, derivation over the
decoder's source). The mainboard and pressure fidelity cells were also
comparing the frozen copy to the SDK with `<=`, which sees only the drift that
removes a field; both now assert equality, so a field declared but unread
fails too.
