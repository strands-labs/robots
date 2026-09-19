### Fixed: the Isaac backend's `get_frame` no longer hands a compositor a manufactured depth buffer

When a camera carries no depth annotator, `_render_frame` logs a WARNING and
returns `np.zeros(rgb.shape[:2])`. That is a fair degradation for the *envelope*
path it also serves - `render()` only needs pixels - but `get_frame` passed it
straight through, while its own docstring promised the opposite:

> Unlike `_render_frame` -- whose blank-frame fallbacks exist for the envelope path
> -- this method **raises** on every degraded path [...] so a compositing consumer
> can never silently receive black pixels with zero depth.

and the `SimEngine.get_frame` contract on the ABC says backends "must never
substitute silently wrong pixels -- failures raise", reserving `None` for backends
with no depth path at all (Newton).

**The consequence is not a subtly wrong image.** `HybridCompositor`'s per-pixel
rule is `valid_fg = isfinite(fg_depth) & (fg_depth > depth_epsilon) & ...`, and its
documented convention treats `0` as sky. So an all-zero foreground depth loses
*every* pixel and composites a frame containing the backdrop alone, with the
simulated robot **entirely absent** - a plausible-looking photoreal image of an
empty scene. A WARNING in a log the compositor does not read is not a refusal.

`_render_frame` now marks whether the depth it returns came from the annotator
(`meta["json"]["depth_is_real"]`), the envelope path keeps its zero-depth
degradation unchanged, and `get_frame` refuses a substituted buffer with a message
naming the camera, why zeros are not an acceptable answer, and both remedies
(re-add the camera, or use `render()` for RGB only). Raising rather than returning
`None` follows this backend's own docstring and lets the message name the
misconfiguration; the ABC reserves `None` for a backend with no depth path, which
Isaac is not.
