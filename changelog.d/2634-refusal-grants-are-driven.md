### Fixed

- **A refusal grant is now graded by whether it lifts its refusal, not by how the variable is spelled.**
  `refusal_codes.REFUSAL_GRANTS` claims each entry is the environment variable a consumer offering the
  choice has to set. What checked that claim asserted the map's keys are the declared codes and that each
  value starts with `STRANDS_` - membership and spelling, never effect. Swapping the
  `POLICY_TYPE_NOT_ALLOWED` and `POLICY_HOST_NOT_ALLOWED` entries with each other, so a consumer
  recognising a refused policy type is told to extend the host allowlist, left the suite green; the
  swapped grant is a no-op and the refusal returns the identical message, which still names the correct
  variable, so neither the package nor a consumer could tell. Every grant is now driven: it must lift the
  refusal it is mapped to, and no other code's grant may lift it. The graded set is `REFUSAL_GRANTS`
  itself, so a sixth code is driven on arrival.
- **Each refusal code states what to set its grant to.** The variable was only half the answer.
  `HF_REPO_NOT_ALLOWED`, `POLICY_TYPE_NOT_ALLOWED` and `POLICY_HOST_NOT_ALLOWED` are allowlists the
  refusal's own `subject` is appended to, but `TRUST_REMOTE_CODE_REQUIRED` is a flag set to `1` and
  `TELEOP_VALUE_OUT_OF_RANGE` is a bound raised above the refused magnitude - and handing those two the
  subject is a silent no-op returning a byte-identical message. For the teleop bound the magnitude that
  lifts it appears only in the message text, which is recorded on the code rather than left to be
  discovered. No code, message, subject or grant value changes.
