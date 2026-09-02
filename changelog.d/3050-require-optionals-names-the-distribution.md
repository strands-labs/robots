### Fixed: `require_optionals` names the distribution, not the import name

`require_optional` (singular) already took `pip_install` for a module whose
distribution is spelled differently from its import name. The plural
`require_optionals` built its hint from import names alone, so a missing `jwt`
produced `pip install jwt` - a DIFFERENT project on PyPI than the `PyJWT` that
supplies the module, making it a remedy that reports success and leaves the
module exactly as missing. It now takes the same mapping, applied per module, so
unmapped names are still reported as-is in the same hint.
