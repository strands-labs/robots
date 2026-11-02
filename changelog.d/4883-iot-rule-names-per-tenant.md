### Fixed: each IoT tenant has rule names of its own, so another tenant's `apply_rules` or `delete_rules` never touches them

`rules.rule_name` turned `-`, `.` and `:` into `_` and cut a name at 128 characters, so tenants such as `lab-a` and
`lab_a`, or two long tenants that start alike, shared all five rule names: the second tenant's `apply_rules` replaced
the first one's e-stop, events, fault and lifecycle rules, and `delete_rules(iot, "lab_a")` deleted `lab-a`'s. A long
tenant's `delete_rules` also left its own rules behind, and `delete_rules` read only the first page of the account's
rules, so a rule listed on a later page was never deleted. A tenant of letters and digits, at most 110 of them, keeps
`strands_<kind>_<tenant>`; any other tenant's names keep at most its first 97 characters, with `_` for `- . :`, then
end in `_` and the first 12 hex digits of its SHA-256 (`strands_estop_lab_a_1c327e55b394` for `lab-a`), and
`delete_rules` reads every page, then deletes exactly the tenant's own five names. Rules that a dev build applied for
such a tenant keep their old names until deleted by name.
