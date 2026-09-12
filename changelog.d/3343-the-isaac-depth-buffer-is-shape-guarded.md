### Fixed: the Isaac depth buffer is shape-guarded, like RGB already was

`_render_frame` refuses a malformed RGB buffer by shape, naming the shape:

```python
arr = np.asarray(rgba)
if arr.ndim < 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
    return None, None, {"error": f"camera {name!r} returned a malformed RGB buffer (shape ...)"}
```

Depth was `depth = np.asarray(depth_raw)` and nothing else. Three shapes got
through that:

* a **0-D scalar or 1-D** buffer reached a consumer promised `(H, W)`;
* a buffer whose size **differed from RGB's** was returned as though the two
  described one frame - the two were never compared;
* a **ragged** buffer raised NumPy's `ValueError: setting an array element with a
  sequence. The requested array has an inhomogeneous shape ...`, which the handler
  reported as a generic `Failed to render camera 'x'`, naming neither the depth
  buffer nor what shape was expected.

Depth now gets the guard RGB has, plus the RGB comparison, and the ragged case is
wrapped where it actually raises - inside `np.asarray`, which runs *before* a shape
guard can see it, so the guard alone would not have covered it.

`get_frame`'s `Raises:` block names the new refusal, per the rule that it is the
only place a caller learns which handler to write.
