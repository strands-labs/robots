### Docs: pip-installed Isaac Sim 6.0.x collateral is documented and its red herring named

Installing Isaac Sim via the cp312 pip wheels
(`pip install 'isaacsim[all,extscache]==6.0.*' --extra-index-url
https://pypi.nvidia.com`) silently degrades an existing dev environment in
ways pip only warns about: `isaacsim-kernel` downgrades `coverage` to 7.4.4,
which breaks numba's tracer probe and surfaces far from the cause as a
robosuite OSC import failure inside the LIBERO adapter
(`module 'coverage.types' has no attribute 'Tracer'`); the torch stack is
bumped past lerobot's pins; and the first non-interactive import hangs on the
EULA prompt without `OMNI_KIT_ACCEPT_EULA=YES`.

`docs/simulation/isaac.md` now documents the pip route with the verified
sequence (install isaacsim, reinstall `coverage>=7.6.1`, set the EULA env var),
the torch-bump expectation, and the exit-134 atexit note.
`IsaacSimulation.is_available()`'s guidance no longer claims Isaac Sim is
"not via pip" - it lists the pip route first, with the caveats. The
`_ControllerInstallError` raised for the numba/coverage clash carries the
single-source remedy from #1805 (`pip install 'coverage>=7.6.1'`), now
extended with the isaacsim-kernel attribution and the note that the
resulting pip conflict warning is cosmetic. A dependency-audit test
fails loudly at collection time when `coverage<7.6.1` is installed alongside
numba and robosuite, turning the red herring into a named error.
