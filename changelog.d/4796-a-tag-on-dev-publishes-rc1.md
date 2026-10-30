### Fixed: a release tag on `dev` publishes 1.0.0rc1, and only the version the package reports

The publish workflow's "Validate version" took only `X.Y.Z`, so the `v1.0.0rc1` tag M2 needs was refused before
the build, and nothing compared the version with the tag. `pyproject.toml` also held a static version beside
`strands_robots.__version__`, and the usual bump (`hatch version 1.0.0rc1`) edited only the first, so an rc1
wheel would have reported `1.0.0.dev0`. The version now has one copy, `strands_robots.__version__`, which the
wheel is built with (`[tool.hatch.version]`); Validate version takes `X.Y.Z` or `X.Y.Z{a,b,rc}N` and refuses a
tag that is not `v<version>`.
