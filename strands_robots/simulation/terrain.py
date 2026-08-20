"""Procedural terrain generation for rough-ground simulation worlds.

A flat ground plane is enough to smoke-test a manipulator, but a locomotion
policy is only interesting on ground it can trip on. The velocity-tracking
locomotion benchmarks (``go2_walk_forward`` / ``g1_walk_forward`` /
``t1_walk_forward`` and the omnidirectional Go2 tasks) all spawn their robot on
a flat plane, so they measure command tracking but never *robustness to
terrain* -- the whole reason legged locomotion is hard. This module generates a
deterministic heightfield that
:meth:`~strands_robots.simulation.base.SimEngine.create_world` can lay down
instead of the flat plane. ``create_world(terrain="rough")`` lays smoothed
value-noise bumps; ``create_world(terrain="stairs")`` lays a flight of discrete
step plateaus rising along +x (foot-placement + climbing, which smooth bumps do
not test); ``create_world(terrain="pyramid")`` lays concentric square step
plateaus rising toward the centre from every direction (an omnidirectional climb
the +x-only staircase cannot express, matching the omnidirectional strafe/turn
velocity-tracking commands); ``create_world(terrain="slope")`` lays a
constant-grade inclined ramp (a continuous uphill pitch, which neither the
non-monotonic bumps nor the discrete steps test). All four are
ground-generation primitives a terrain *curriculum* (progressive difficulty
across resets) is built on. That curriculum knob is the ``difficulty`` scalar
(``terrain_elevation``): the heightfield the generator returns is normalized
to ``[0, 1]`` and scaled to metres by a peak elevation, so
``create_world(terrain=..., difficulty=d)`` multiplies that peak by ``d``
(``d=1.0`` full height, smaller = gentler, larger = harsher) to ramp terrain
magnitude across resets without changing the terrain *kind* -- a robot settles
onto shallower bumps/steps at a lower difficulty and taller ones as it is
raised.

The generator is intentionally backend- and MuJoCo-independent (stdlib plus the shared
validation domain in :mod:`strands_robots.utils`, no numpy / mujoco import) so the height data is trivially unit-testable and
deterministic given ``(kind, resolution, seed)`` -- a benchmark that evaluates a
policy on ``terrain="rough"`` regenerates the identical field on every reset.
"""

from __future__ import annotations

import random

from strands_robots.utils import positive_finite_number_error, positive_whole_number_error

# Supported terrain kinds. ``"rough"`` is smoothed value-noise bumps; ``"stairs"``
# is a flight of discrete parallel step plateaus rising along +x; ``"pyramid"`` is
# concentric square step plateaus rising toward the centre from every direction;
# ``"slope"`` is a constant-grade inclined ramp. The tuple is the single source
# of truth both backends validate against and is easy to extend without touching
# the create_world signature -- add a name here and a branch in
# :func:`generate_heightfield`.
SUPPORTED_TERRAINS: tuple[str, ...] = ("rough", "stairs", "pyramid", "slope")

# Heightfield geometry (metres). The field spans +/-``TERRAIN_RADIUS`` in x and y
# (matching the flat ground plane's 5 m half-size so the reachable workspace is
# unchanged), rises up to ``TERRAIN_ELEVATION`` at its highest bump, and rests on
# a ``TERRAIN_BASE``-thick solid slab so there is never a hole under the robot.
# The surface height therefore ranges over ``[0, TERRAIN_ELEVATION]`` -- flush
# with z=0 at its lowest point, so a robot never falls below the nominal floor.
TERRAIN_RADIUS = 5.0
TERRAIN_ELEVATION = 0.08
TERRAIN_BASE = 0.1
TERRAIN_RESOLUTION = 40  # nrow == ncol grid cells (25 cm cells over the 10 m field)
TERRAIN_SEED = 0

# Number of discrete step plateaus for ``terrain="stairs"``. The staircase rises
# in ``TERRAIN_STAIR_STEPS`` even levels from 0 (flush with z=0) to
# ``TERRAIN_ELEVATION`` along +x, so each riser is ``TERRAIN_ELEVATION /
# (TERRAIN_STAIR_STEPS - 1)`` tall (five plateaus -> 2 cm risers up to 8 cm).
TERRAIN_STAIR_STEPS = 5

# Number of concentric step plateaus for ``terrain="pyramid"``. The pyramid rises
# in ``TERRAIN_PYRAMID_STEPS`` even levels from 0 (the flush outer ring) up to
# ``TERRAIN_ELEVATION`` at the central plateau, so each riser is
# ``TERRAIN_ELEVATION / (TERRAIN_PYRAMID_STEPS - 1)`` tall (five plateaus -> 2 cm
# risers up to 8 cm), matching the staircase riser height.
TERRAIN_PYRAMID_STEPS = 5


def validate_terrain(kind: str | None) -> None:
    """Raise ``ValueError`` for an unsupported terrain kind (``None`` is a flat ground)."""
    if kind is None or kind in SUPPORTED_TERRAINS:
        return
    raise ValueError(
        f"Unknown terrain {kind!r}. Supported: {sorted(SUPPORTED_TERRAINS)} (or None / omit for a flat ground plane)."
    )


def validate_difficulty(difficulty: float) -> None:
    """Raise ``ValueError`` unless ``difficulty`` is a finite number ``> 0``.

    ``difficulty`` scales the terrain's peak elevation (``1.0`` = full height);
    ``0`` would collapse the heightfield to a flat plane (a degenerate hfield
    with zero elevation, which MuJoCo rejects) and a negative/NaN value is
    meaningless, so both are rejected actionably rather than silently producing
    broken ground.

    The accepted domain is
    :func:`~strands_robots.utils.positive_finite_number_error` - the shared
    positive-real domain every other continuous knob is measured against - and
    this function is the raising binding over it. All three ``create_world``
    implementations report through this one binding, so a scale one backend
    refuses cannot be honored by another and the message is identical
    everywhere.

    Delegating also closes the type axis a bare ``float(difficulty)`` left
    open. ``None`` and ``["0.5"]`` raised ``TypeError``, which escapes the
    ``{"status": "error"}`` tool-result contract entirely because the callers
    catch only ``ValueError``; a non-numeric string surfaced ``float()``'s own
    message, naming neither the parameter nor the surface; and ``bool`` was
    accepted asymmetrically - as an ``int`` subclass ``True`` passed the
    ``<= 0`` test as a silent ``1.0`` (indistinguishable from the default full
    height) while ``False`` was refused as a zero scale.
    """
    if positive_finite_number_error(difficulty, "difficulty", "terrain") is not None:
        raise ValueError(f"terrain difficulty must be a finite number > 0 (1.0 = full height), got {difficulty!r}.")


def terrain_elevation(difficulty: float = 1.0) -> float:
    """Peak terrain elevation in metres for a curriculum ``difficulty``.

    The heightfield generator returns normalized ``[0, 1]`` heights; this maps
    them to metres. At ``difficulty=1.0`` it returns :data:`TERRAIN_ELEVATION`
    (the default full-height terrain, unchanged); ``difficulty=0.5`` halves the
    peak (gentler curriculum stage), ``difficulty=2.0`` doubles it (harsher).
    The single source of truth both the surface scale and any future consumer
    agree on. Raises via :func:`validate_difficulty` for a non-positive/NaN
    value.
    """
    validate_difficulty(difficulty)
    return TERRAIN_ELEVATION * float(difficulty)


def _box_blur(grid: list[list[float]], n: int) -> list[list[float]]:
    """One 3x3 box-blur pass (edge-clamped) -- turns spiky noise into walkable bumps."""
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = 0.0
            cnt = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < n:
                        acc += grid[ii][jj]
                        cnt += 1
            out[i][j] = acc / cnt
    return out


def _rough(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    grid = [[rng.random() for _ in range(n)] for _ in range(n)]
    # Two blur passes -> smooth, walkable bumps rather than single-cell spikes
    # that would flip a robot on contact and render as noise.
    for _ in range(2):
        grid = _box_blur(grid, n)
    flat = [v for row in grid for v in row]
    lo, hi = min(flat), max(flat)
    span = hi - lo
    if span <= 0.0:  # degenerate (all-equal); flat field
        return [0.0] * (n * n)
    # Normalize to [0, 1]; MuJoCo scales it by the hfield's elevation size.
    return [(v - lo) / span for v in flat]


def _stairs(n: int) -> list[float]:
    """Discrete parallel step plateaus rising along +x.

    MuJoCo ``<hfield>`` ``userdata`` is row-major with row 0 at min-y and column
    0 at min-x, so making the level a step function of the *column* index makes
    the staircase rise along +x and stay constant across y (a flight of steps
    you climb as x increases). Returns ``TERRAIN_STAIR_STEPS`` distinct
    normalized plateaus ``{0, 1/(k-1), ..., 1}`` -- deterministic, no rng.
    """
    steps = TERRAIN_STAIR_STEPS
    out: list[float] = []
    for _i in range(n):  # row (y): every row is identical
        for j in range(n):  # column (x): step level rises with x
            level = min((j * steps) // n, steps - 1)  # 0 .. steps-1
            out.append(level / (steps - 1))
    return out


def _pyramid(n: int) -> list[float]:
    """Concentric square step plateaus rising toward the centre (a pyramid staircase).

    Unlike ``terrain="stairs"`` (whose level is a step function of the *column*
    index, so it rises only along +x and is flat across +y), the pyramid's level
    is a step function of the Chebyshev (square-ring) distance from the centre --
    it therefore rises identically from every direction, an *omnidirectional*
    climb the +x-only staircase cannot express (matching the omnidirectional
    strafe/turn velocity-tracking commands). The robot spawns at the origin on
    the highest central plateau and descends steps walking outward on ANY
    heading. Returns ``TERRAIN_PYRAMID_STEPS`` distinct normalized plateaus
    ``{1, ..., 1/(k-1), 0}`` -- highest (1.0) at the centre, flush with z=0 (0.0)
    on the outer ring so a robot never falls below the nominal floor.
    Deterministic, seed-independent (a stepped field needs no rng).
    """
    steps = TERRAIN_PYRAMID_STEPS
    c = (n - 1) / 2.0  # continuous centre index (row 0 -> min y, col 0 -> min x)
    out: list[float] = []
    for i in range(n):  # row (y)
        for j in range(n):  # column (x)
            cheb = max(abs(i - c), abs(j - c))  # square-ring distance from centre
            frac = cheb / c if c > 0 else 0.0  # 0 at centre, 1 on the outer ring
            ring = min(int(frac * steps), steps - 1)  # 0 (centre) .. steps-1 (edge)
            level = (steps - 1) - ring  # steps-1 (centre) .. 0 (edge)
            out.append(level / (steps - 1))
    return out


def _slope(n: int) -> list[float]:
    """A constant-grade inclined ramp rising linearly along +x.

    MuJoCo ``<hfield>`` ``userdata`` is row-major with row 0 at min-y and column
    0 at min-x, so making the height a *linear* function of the column index
    makes the ramp rise along +x and stay constant across y (a uniform uphill
    pitch you climb as x increases). Unlike ``stairs`` (a discrete step
    function) the surface is continuous, and unlike ``rough`` (non-monotonic
    value noise) it is strictly monotonic with a uniform grade -- the canonical
    inclined-plane locomotion terrain. Returns ``n`` distinct evenly-spaced
    normalized levels per row ``{0, 1/(n-1), ..., 1}`` -- deterministic, no rng.
    """
    out: list[float] = []
    for _i in range(n):  # row (y): every row is identical
        for j in range(n):  # column (x): height rises linearly with x
            out.append(j / (n - 1))
    return out


def generate_heightfield(
    kind: str,
    resolution: int = TERRAIN_RESOLUTION,
    seed: int = TERRAIN_SEED,
) -> list[float]:
    """Return a normalized ``[0, 1]`` heightfield as ``resolution * resolution`` floats.

    Row-major (``userdata`` order for a MuJoCo ``<hfield>``). Deterministic given
    ``(kind, resolution, seed)``. Raises ``ValueError`` for an unknown/None kind
    or an unusable resolution.

    ``resolution`` is the grid-cell count the returned length is squared from, so
    it is measured against
    :func:`~strands_robots.utils.positive_whole_number_error` - the shared
    positive-discrete domain every other pixel/cell count in the package is
    checked against - before anything is generated, exactly as ``difficulty`` is
    measured against the continuous domain by :func:`validate_difficulty`. A bare
    ``int(resolution)`` left three axes open, and each one produced a grid the
    caller did not ask for or an error a caller could not act on:

    * A **fractional** count was truncated silently, so the documented
      ``resolution * resolution`` length no longer held: ``39.7`` returned 1521
      floats for a number whose square is 1576.09, and ``2.5`` returned a 2x2
      field. Nothing raised, and the mismatch only surfaced once the caller fed
      the field to an ``<hfield>`` sized from the number they passed, where
      MuJoCo reports ``elevation data length must match nrow*ncol`` - a message
      naming neither ``resolution`` nor the truncation.
    * A **string** was accepted outright: ``"40"`` built a 40x40 field, so a
      resolution read from a config file or an argv string passed here and was
      refused nowhere.
    * ``None``, ``[]`` and ``inf`` raised ``TypeError`` / ``OverflowError`` from
      inside ``int()``. ``ValueError`` is this module's error contract - it is
      what :func:`validate_terrain` and :func:`validate_difficulty` raise and
      what the ``create_world`` implementations narrow to when turning a terrain
      refusal into a ``{"status": "error"}`` tool result - so those two classes
      escaped it entirely rather than being reported through it.

    The ``>= 2`` floor below is unchanged and stays a separate check: the shared
    domain answers whether the value is a usable count at all, and a 1x1 field
    (which MuJoCo does compile) is this module's own refusal.
    """
    validate_terrain(kind)
    if kind is None:
        raise ValueError("generate_heightfield requires a terrain kind, got None.")
    if positive_whole_number_error(resolution, "resolution", "terrain") is not None:
        raise ValueError(f"terrain resolution must be a positive whole number of grid cells, got {resolution!r}.")
    n = int(resolution)
    if n < 2:
        raise ValueError(f"terrain resolution must be >= 2, got {resolution}.")
    if kind == "rough":
        return _rough(n, seed)
    if kind == "stairs":
        return _stairs(n)
    if kind == "pyramid":
        return _pyramid(n)
    if kind == "slope":
        return _slope(n)
    # validate_terrain accepts only SUPPORTED_TERRAINS; a kind reaching here
    # means the tuple grew without a generator branch.
    raise ValueError(f"terrain kind {kind!r} has no generator implementation.")  # pragma: no cover


__all__ = [
    "SUPPORTED_TERRAINS",
    "TERRAIN_RADIUS",
    "TERRAIN_ELEVATION",
    "TERRAIN_BASE",
    "TERRAIN_RESOLUTION",
    "TERRAIN_SEED",
    "TERRAIN_STAIR_STEPS",
    "TERRAIN_PYRAMID_STEPS",
    "validate_terrain",
    "validate_difficulty",
    "terrain_elevation",
    "generate_heightfield",
]
