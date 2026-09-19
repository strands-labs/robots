### Fixed: the converted-USD cache keys on the files it reads, and installs without deleting a live entry

Two defects in the MJCF-to-USD cache this series introduces, both silent, both
reported on review.

**The key covered one directory, not the file set the conversion reads.**
`_asset_digest` walked `dirname(mjcf_path)`, and its own docstring gives the
reason it exists: keying on the named file alone would "hand back a stale USD
after any change that did not touch the one file named". An MJCF reaches outside
its own directory, and the shipped registry's nested layouts do it by design -
`lekiwi`'s entry point `<include>`s `../so_arm100/so_arm100.xml`, the whole arm's
joints and geoms; `asimov_v0` declares `meshdir="../assets/meshes"`, every mesh
PhysX simulates. `jvrc`, `aliengo`, `unitree_a1` and `reachy_mini` share the
shape. So an asset re-download or upstream update to the arm produced the same
key while the entry directory stayed byte-identical, and the conversion returned
the USD built from the old description under `status: success`. The wrong
`so_arm100.xml` is a wrong joint vocabulary, which defeats the joint-name parity
this module exists to provide. The collision holds in the other direction too:
two robots whose entry directories match but whose siblings differ shared one
entry. The digest now folds in the transitive closure the description references
- `<include>` targets resolved against the including file, and
`meshdir`/`assetdir`/`texturedir` assets resolved against the model file, which
are MuJoCo's own rules and the ones `loaders.py` already applies - as a superset
of the directory walk rather than a replacement, so a file no parser models is
still covered.

**The install deleted a completed entry another process was reading.** The cache
root is shared cross-process and the pid-suffixed staging directory says
concurrent converters are intended, but the install did
`_remove_tree(target_root)` before its rename. When two processes missed the
marker for one key and both converted, the loser deleted the winner's finished
entry - after the winner had returned its path and referenced that USD into a
live stage. USD composes payloads lazily, so a read landing in that window failed
or composed the robot without its meshes, in a process whose own conversion was
correct. The install now renames onto an absent target and reads the OS's refusal
as "another process published this key first", which is sound because both
converters produce identical content for a key. The marker moved inside staging
so the rename publishes a complete entry, closing a second window in which a
reader saw the directory without one. A torn entry - which nothing deletes any
more - is moved aside with a single atomic rename and retried once, so it cannot
become permanently uninstallable.

The premise test asserting the old adjacency ("the load-bearing removal is the
statement before the rename") is replaced rather than deleted: the tolerated
`unlink` in `_remove_tree` is still sound, now for the stronger reason that no
caller passes a completed entry at all, which is graded over the whole module
instead of over two adjacent statements.
