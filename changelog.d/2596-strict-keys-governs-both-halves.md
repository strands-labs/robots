### Fixed

`LerobotLocalPolicy`'s `strict_keys` `Args:` entry now describes both halves of
the key binding the flag governs, not cameras alone. The entry read "raise ... if
any camera name cannot be matched to a declared policy image key by exact name
and no `camera_key_map` covers it", which was accurate when the flag shipped
(#614): both of its gates were then on the camera path. Seven days later a
joint-state refusal was gated on the same flag (#897) and twelve days after that
a second one (#1099), and the entry never followed.

So a caller who read the entry, saw every camera bind by exact name -- where the
flag is genuinely a no-op -- and set `strict_keys=True` instead turned a
joint-state *warning* into a hard `ValueError`. Measured on a real MuJoCo aloha
observation (16 named joints, 13 cameras) against generically auto-filled
`robot_state_keys`: `strict_keys=False` warns once and returns 16 values,
`strict_keys=True` raises, and the cameras bind by name in both cases.

The sibling that got the flag in the same pull request is the counter-evidence.
`Gr00tPolicy`'s entry says "if auto-inferred observation/action keys cannot be
matched", naming the whole key surface, and is still accurate. Both joint-state
methods already document their own `strict_keys` condition, and
`docs/policies/lerobot-local.md` documents the joint-state raise in its body --
so only the flag's own definition and that page's one-line gloss disagreed with
the code, and the page contradicted itself.

Two of the four gating methods never named the flag at all.
`_resolve_camera_targets` described its third routing rung as falling back with a
WARN, which is what happens only when the flag is unset, and enumerated two of
the three conditions it raises `ValueError` for; `_to_lerobot_observation`
described an unconditional positional fill. Both now say what the flag does
there.

No behaviour change: no refusal, exception class, message or gate condition is
touched. A guard derives both rules from the tree rather than listing them --
every method that reads the flag must name it, and every public attribute a gate
reads beside the flag must appear in the flag's own entry -- and the class set is
derived too, so a second class taking `strict_keys` is held to the same rule.
