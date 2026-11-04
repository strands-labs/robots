### Changed: the dashboard fetches its two browser engines on first use, so the installed package is 2.2 MB instead of 23.8 MB

`mujoco.wasm` and `ort-wasm-simd-threaded.wasm` (20 MB) are no longer in the package. The first page load downloads
each one at the pinned version, checks its sha256 and caches it in `~/.cache/strands_robots/dashboard`;
`python -m strands_robots.dashboard --fetch` does it ahead of time for a network with no internet. The notebooks are
committed without outputs, and their GIFs and screenshots moved to the `media` release.
