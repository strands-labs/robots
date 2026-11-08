### Fixed: the microduck and holosoma presets fetch their weights at a pinned commit

The presets pinned each file's sha256 but downloaded it from the Hub repo's `main`. When
`pollen-robotics/microduck-policies` published a new `velstand.onnx` under the same name, every
`Policy("microduck?skill=velstand")` refused to load ("refusing to run unknown weights"). Each preset in
`presets.json` now names the Hub `revision` its hashes were taken at, and `OnnxPolicy(..., revision=)` /
`resolve_weights(..., revision=)` pass it to `hf_hub_download`.
