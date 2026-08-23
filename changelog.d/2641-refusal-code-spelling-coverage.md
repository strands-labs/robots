### Fixed

The rule that every refusal code reaching a consumer is a member of the closed
`refusal_codes.REFUSAL_CODES` vocabulary now reads a `code=` argument in every
spelling a raise site can use it, and reports one it cannot read rather than
skipping it. It found the codes to grade with
`isinstance(keyword.value, ast.Attribute)`, so it saw a code only when the site
reached it through the codes module -- while `test_every_code_is_exported_under_its_own_name`
pins that each code is exported under its own name and listed in `__all__`, making
a direct import a first-class way to reach one, and the sibling rule that checks
whether a refusal carries a code at all already reads the same keyword
spelling-independently.

Nothing validates the value at runtime and nothing should: `SecurityError`
stores `code` as given, and raising from an exception constructor would replace a
security refusal with a constructor error. The scan is therefore the only thing
between a mistyped code and a consumer. Measured on `main` plus two plausible
gates, one spelling its code as a from-imported name and one as a string literal
with a typo: the whole refusal-contract suite stayed green while the typo'd code
shipped, and the consumer pattern the security guide documents,
`REFUSAL_GRANTS[refusal.code]`, raised `KeyError` on it. A `code ==` switch has no
branch for it either, so the operator offer the code exists to enable disappears
and a consumer is back to reading prose -- the coupling the codes removed.

The scan now resolves a codes-module attribute, a name imported from that module
(resolving an `as` alias to the code it aliases rather than to a name this package
does not declare), or the literal string; each code being exported under its own
name is what makes those one channel. A value that is not statically readable, such
as a parameter forwarded by a shared helper, is reported with both remedies named,
because a code nothing can check is a code that may not be in the vocabulary at
all. There are no such sites today, and all seven shipped sites use the attribute
form, so the widening grades exactly the same seven and changes no verdict.
