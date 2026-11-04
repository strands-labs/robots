### Docs: notebook 11, the Hugging Face hub-to-hardware article on this release

The article's companion notebook lived in the 0.5 `examples/` tree, which this release replaced, so its links
404. `notebooks/11_hub_to_hardware.ipynb` (and its script) carries the same five steps on the current API: an
agent over `arm.tool` and `sim.tool`, a LeRobot v3 recording through `run(record=...)` (pushed to the Hub with
`HF_USER`), a Hub ACT checkpoint run by name, the hardware constructor with `confirm=` / `approve=`, and the arm
mounted on a mesh where an agent reaches it with `use_mesh`.
