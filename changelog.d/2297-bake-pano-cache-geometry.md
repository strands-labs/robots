### Fixed

`bake_gsplat_panorama` now keys its cached output on the geometry it was baked
for. The default output path is `<stem>_pano_<equi_w>x<equi_h>_f<face_size>.jpg`
instead of a single `<stem>_pano.jpg` shared by every bake of one scene, so a
call asking for a different resolution or `face_size` re-renders instead of
silently returning an earlier bake's image. An explicit `out_path` is still
honored verbatim, and repeating one request still hits the cache.
