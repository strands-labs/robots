### Docs: the notebook install lines upgrade a stale environment instead of skipping it

Every notebook install line was a bare `pip install "strands-robots[...]"`.
Extras do not make a requirement unsatisfied, so against an environment that
already carried an older release pip reported `Requirement already satisfied`
and upgraded nothing:

```
$ pip install "strands-robots[sim-mujoco,lerobot]"   # env has 0.4.1
Requirement already satisfied: strands-robots[lerobot,sim-mujoco] ... (0.4.1)
```

The reader then ran the notebook against 0.4.1, whose
`StreamingDatasetReader.open` has neither a `repo_type` parameter nor
`**kwargs`, so notebook 5's headline bucket read failed with `TypeError: open()
got an unexpected keyword argument 'repo_type'` - naming the keyword rather than
the stale install behind it. The prerequisite text was correct and had no way to
take effect: nothing in the command could act on it.

All six notebooks, the notebooks index, and the README's bucket-sync section now
pass `-U`. A version floor on the requirement
(`"strands-robots[sim-mujoco,lerobot]>=0.5.1"`) works equally well, since it
makes the installed release genuinely unsatisfying; the new test accepts either
form and rejects a line carrying neither. The git/URL fallback keeps `-U` too:
it re-resolves the ref on its own but is only reinstalled when asked to upgrade.

Reported in #1507 (round-2 blog review) as the last gap in the install path.
