### Docs: the v1 site, nine notebooks and a README that says what v1 is (part 12)

`docs/` is rebuilt for v1: a quickstart, one page per concept (a world, a robot, policies, hardware, GPU backends,
device and safety, the mesh, recording, agent tools, the dashboard), a status page and the API, with the drawings
rendered from `docs/drawings/scenes`. Edit links point at `dev`. `notebooks/` walks one SO-101 (with a G1 next to it)
through every part, saved with its outputs. `tests/test_docs_examples.py` runs the first python example of every page
in a fresh process; a page marks an example that needs CUDA, a real arm or AWS credentials.
