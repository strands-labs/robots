### Fixed: GitHub-hosted robot models are fetched at a pinned commit and checked against a content digest

The six registry robots whose models live on GitHub (`reachy_mini`, `asimov_v0`, `microduck`, `open_duck_mini`,
`lekiwi`, `yahboom_m3pro`) now carry `asset.source.commit` and `asset.source.sha256`. The asset fetch downloads that
commit instead of the default branch head, refuses a tarball whose top directory names another commit, and caches
nothing unless the kept files hash to the pinned digest (`description.assets.tree_digest`). The joint limits the
safety chain enforces are read from these files, so they now change only through a registry edit. Tar member names
that are absolute or contain `..` refuse the whole archive instead of being written. A registry entry without a pin
raises `ValueError`; a cache fetched at another pin (including every cache made by 0.6.0) is fetched once again.
