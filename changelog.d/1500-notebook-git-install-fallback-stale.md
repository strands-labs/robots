### Docs: the notebook's git-install fallback outlived the release it waited for

Notebook 5 and the notebooks index both offered a git install "until v0.5.1 is
on PyPI". 0.5.1 is on PyPI, so that line now routes a reader to `main` instead
of to the release the floor asks for, and it was the first instruction in the
notebook the storage-buckets post links to. Both now give the pinned PyPI
install (`pip install -U "strands-robots[sim-mujoco,lerobot]>=0.5.1"`).

The floor and the reason for it stay, since
`test_notebook_min_version_docs.py` asserts the notebook states
`strands-robots >= 0.5.1` to keep version-less guidance from creeping back
(#1500). What went was the release archaeology around it: which release floored
lerobot at which version explains a decision the reader is not making, and the
same paragraphs had already been cut from the blog draft for the same reason.

Notebook 5's closing "Where to go from here" promised a "real, deployable
checkpoint" from raising `steps` to 500 on a GPU. The notebook records one
60-frame episode from `MockPolicy` (`n_steps=60`, and the recorder writes one
frame per control step with no decimation), so a 500-step run over it exercises
the record-train-load path and says nothing about the behaviour. Reading that
bullet as a quality claim is what makes the 132-second figure look like a claim
to have trained something useful, which it is not. The bullet now says which of
the two it is, and points at collecting real demonstrations before judging a
checkpoint.
