### Added: GS backgrounds evaluate spherical-harmonic view-dependence instead of dropping it

`_load_ply_splats` kept only the SH DC term and `_load_spz_splats` ignored the
trailing SH block, so view-dependent appearance (speculars, sheen) was baked
away for every asset that carried it -- including all `.ply` presets and any
user-supplied full-quality capture. Both loaders now decode higher-order SH
into raw `(N, K, 3)` coefficients and `GsplatBackground.render` passes the
matching `sh_degree` to gsplat for per-view evaluation; DC-only assets (such
as the default `tabletop.spz`, which is genuinely `sh_degree=0`) keep the
baked-RGB fast path. The `bonsai`/`bicycle`/`stump` presets now point at the
fully-trained 30k-iteration INRIA checkpoints instead of the visibly
undertrained `iteration_7000` ones, with per-preset provenance (source,
iteration, license) recorded next to the URL table. `bicycle` was re-measured
on the 30k checkpoint and stays excluded from skybox use: 33-66% of pixels
render near-white per view (vs bonsai's 0-5%), so the haze is the capture's
overcast sky, not undertraining.
