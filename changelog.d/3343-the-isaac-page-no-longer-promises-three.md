### Docs: the Isaac page no longer promises three things the backend does not do

* **`create_simulation("isaac")` does not raise when Isaac Sim is missing.** The
  page claimed it "raises a `ValueError` whose message carries the exact install
  hint", then said in the next sentence that backend discovery is lazy - which is
  what is implemented, what `tests/simulation/test_factory.py` pins, and why
  nothing raises: no runtime is imported until you build a world. The page now
  describes the real path (construction succeeds; `create_world()` returns the
  structured error; `is_available()` is the eager check that needs no world) and
  notes the deliberate contrast with Newton, which *does* raise `ImportError` from
  `create_simulation("newton")` because it imports its runtime to construct.
* **Replicator synthetic data** - "ground-truth depth, segmentation, and bounding
  boxes" - is not implemented; there is no Replicator code in the package. The
  bullet now names the RTX metric depth `get_frame` does return.
* **`enable_rtx_sensors`** was a documented `IsaacConfig` field that nothing read,
  so `enable_rtx_sensors=False` silently left RTX sensors enabled. Removed from
  both the config and the table: `IsaacConfig` refuses unknown keywords with a
  `TypeError` naming the argument, so a caller who passes it now gets told, rather
  than having it accepted and ignored.
