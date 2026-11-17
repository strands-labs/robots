### Docs: notebook 09 runs to the end without a model id

`notebooks/09_mesh.ipynb` read `STRANDS_MODEL_ID` directly, so without it the notebook stopped with a `KeyError` at
its one model turn and never reached the AWS IoT section. It now prints that the turn was skipped, as notebook 07
does for its chat cell, and `notebooks/README.md` says which notebooks call a model.
