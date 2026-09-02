### Fixed: the `policy_config` guard grades both answers a provider gives an unbacked key

The test that grades `tool_spec.json`'s per-provider `policy_config` key list
also asserted that every provider the description enumerates absorbs an unknown
constructor keyword into `**kwargs`. Five of the fifteen registered providers
refuse one instead - `cosmos3`, `kimodo`, `microduck`, `protomotions` and `vera`
declare no `**kwargs` sink - so that assertion was green only because the
description happens to enumerate three absorbers, and enumerating any of the
five turned the guard red and blamed the provider for having the stricter
contract.

`create_policy` ends in `PolicyClass(**resolved_kwargs)` without inspecting the
names, so an unbacked key gets the provider's answer: an absorber builds and
drops the value with no warning, and a provider without one raises `TypeError`
naming the keyword. Neither answer tells the caller that the key list they read
is wrong, which is the reason to grade the description against the live
constructors in the first place - and that reason holds for both shapes. The
guard now pins the two answers behaviourally at `create_policy`, derives the
refusing provider rather than naming one, and keeps a registry-wide cell that
fails if the registry ever collapses to a single shape and makes those cells
vacuous. The primary grader's failure message carried the same false premise and
now describes both outcomes.
