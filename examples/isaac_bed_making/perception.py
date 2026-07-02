"""On-robot perception for the bed-making demo, mirroring the real Unitree G1 EDU sensor geometry
from **robotics-connect** (github.com/armwaheed/robotics-connect — Arm's sanitized G1 control stack).

Two complementary sensors live on the G1's ``torso_link``, exactly as on the EDU hardware:

* **Crown Livox MID-360 LiDAR** (robotics-connect ``lidar_sight``) — mounted upright with a 2.3°
  nose-down pitch, 41.6 cm above ``torso_link``. It sees the room, so it is what DETECTS the bed:
  a large horizontal surface at bed height. Detection mirrors ``lidar_sight.find_tables`` — filter
  the cloud to a body-frame height band (so the floor drops out), keep the horizontal returns, and
  accept a cluster whose planar area clears a bed threshold (a bed is just a large "table").

* **Head Intel RealSense D435i** (robotics-connect ``depth_camera_sight``) — tilted **51.29° down**
  (the `CAMERA_TILT_DEG_DEFAULT` calibrated on the dev EDU by a floor-plane fit). At that pitch it
  sees only the floor and the near bed — *not* the room — which is exactly why the LiDAR, not the
  camera, does room-scale detection. The head cam is for CLOSE work: the bed-coverage / "good enough"
  check once at the bedside.

The LiDAR's field of view is clipped by the robot's own head geometry. ``lidar_sight`` documents two
self-occlusions, modelled here so the simulated cloud has the same blind spots as the real sensor
(the whole point of sim-to-real perception — a detector that works here works on hardware):

* the **face-frame** vertical bars blank the **±40–45° azimuth** sectors, and
* the **chin / lower jaw** blanks everything **below −10° elevation** — this hides the far *floor*,
  not a near surface: on the real robot a table reads cleanly at ~0.4 m / −7° (above the cut), so a
  bed is detectable up close as well as on approach (verified on-hardware in robotics-connect).

Not yet wired into ``demo.py`` — this module is the sim sensing for the *detect → approach → switch*
behaviour layer; the exact sim sensor poses + the bed height band are eye-calibrated when it lands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ── MID-360 LiDAR mount (robotics-connect lidar_sight, from the bundled G1 URDF) ──
LIDAR_MOUNT_LINK = "torso_link"
LIDAR_MOUNT_POS = (0.0003, 0.0, 0.416)          # metres from torso_link (mid360_joint origin)
LIDAR_MOUNT_PITCH_RAD = 0.0401                  # ~2.3° nose-down
LIDAR_VERTICAL_FOV = (-45.0, 52.0)              # deg; wide enough to see a bed below + the room
LIDAR_HORIZONTAL_FOV = (-180.0, 180.0)          # deg; 360° dome
LIDAR_CHANNELS = 64
LIDAR_HORIZONTAL_RES = 1.0                       # deg

# Self-occlusions from the robot's head geometry (lidar_sight "face-frame / chin" artefacts).
FACE_FRAME_AZIMUTH_DEG = (40.0, 45.0)            # the ± bars blank these forward azimuth stripes
CHIN_MAX_ELEVATION_DEG = -10.0                   # the chin blanks everything below this elevation
SELF_REFLECTION_MIN_RADIAL_M = 0.15             # drop dome self-returns clustered at the origin

# ── Bed detection (lidar_sight.find_tables, widened from a table to a bed) ──
# Height band (LiDAR sensor frame, +z up) that keeps furniture-height surfaces and drops the floor:
# a bed top reads near its lower edge, the floor sits well below. Approximate — eye-calibrate against
# the rendered cloud when this is wired (cf. lidar_sight's torso-frame BODY_Z_TABLE_BAND, shifted by
# the ~0.42 m sensor-above-torso mount).
BED_Z_BAND_M = (-0.60, 0.40)                     # metres, LiDAR sensor frame (+z up)
BED_HORIZONTAL_TOL_M = 0.06                      # |Δz| within a candidate plane to count as flat
BED_MIN_AREA_M2 = 1.2                            # a bed's footprint dwarfs a table's (~0.15 m²)
BED_GRID_M = 0.10                                # occupancy-cell size for the area estimate


@dataclass
class BedDetection:
    found: bool
    centroid_b: tuple[float, float, float] = (0.0, 0.0, 0.0)   # bed centre, torso body frame
    bearing_rad: float = 0.0                                    # heading to the bed centre (base +x=0)
    range_m: float = 0.0                                        # horizontal distance to the bed centre
    area_m2: float = 0.0                                        # estimated horizontal surface area
    n_points: int = 0


def make_lidar_cfg(prim_path: str, mesh_prim_paths: list[str]):
    """A RayCaster configured as the crown MID-360, attached to the robot's ``torso_link``.

    Casts against the static room meshes in ``mesh_prim_paths`` (bed, headboard, pillows, ground).
    RayCaster (mesh ray-cast) is the right fidelity here — full RTX-Livox replay is neither needed
    nor affordable; the real Livox is characterised on hardware in robotics-connect."""
    from isaaclab.sensors import RayCasterCfg
    from isaaclab.sensors.ray_caster import patterns

    half = LIDAR_MOUNT_PITCH_RAD / 2.0
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(
            pos=LIDAR_MOUNT_POS,
            rot=(math.cos(half), 0.0, math.sin(half), 0.0),  # nose-down pitch about body-Y
        ),
        ray_alignment="base",                                # rays follow the torso pose
        mesh_prim_paths=list(mesh_prim_paths),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=LIDAR_CHANNELS,
            vertical_fov_range=LIDAR_VERTICAL_FOV,
            horizontal_fov_range=LIDAR_HORIZONTAL_FOV,
            horizontal_res=LIDAR_HORIZONTAL_RES,
        ),
        max_distance=12.0,
        debug_vis=False,
    )


def occlude(points_b: np.ndarray) -> np.ndarray:
    """Drop the points the real G1's head geometry would block, given the cloud already in the LiDAR
    body frame (+x forward, +y left, +z up). Reproduces lidar_sight's face-frame + chin + dome
    self-reflection blind spots so the simulated view matches the hardware."""
    if points_b.shape[0] == 0:
        return points_b
    x, y, z = points_b[:, 0], points_b[:, 1], points_b[:, 2]
    radial = np.linalg.norm(points_b, axis=1)
    az = np.degrees(np.arctan2(y, x))                         # 0° = forward, ±180° = behind
    horiz = np.hypot(x, y)
    elev = np.degrees(np.arctan2(z, np.maximum(horiz, 1e-6)))
    lo, hi = FACE_FRAME_AZIMUTH_DEG
    face_bar = (np.abs(az) >= lo) & (np.abs(az) <= hi)        # the ± vertical bars
    chin = elev < CHIN_MAX_ELEVATION_DEG                      # lower jaw blanks low elevations
    dome = radial < SELF_REFLECTION_MIN_RADIAL_M             # self-returns at the sensor origin
    return points_b[~(face_bar | chin | dome)]


def _to_body(points_w: np.ndarray, sensor_pos_w, sensor_quat_w) -> np.ndarray:
    """World hit points → LiDAR body frame (the frame the occlusions + detection work in)."""
    w, x, y, z = sensor_quat_w
    # rotation matrix of the sensor frame in world (quat w,x,y,z)
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return (points_w - np.asarray(sensor_pos_w)) @ r          # world→body = R^T·(p−t); (p−t)·R


def detect_bed(points_w: np.ndarray, sensor_pos_w, sensor_quat_w) -> BedDetection:
    """Find a bed in a LiDAR frame. Mirrors lidar_sight.find_tables: occlude → height-band filter →
    keep the dominant horizontal plane → threshold on planar area. Returns the bed centre + bearing
    in the LiDAR body frame so the behaviour layer can steer the walk toward it."""
    pts = _to_body(points_w, sensor_pos_w, sensor_quat_w)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = occlude(pts)
    band = pts[(pts[:, 2] >= BED_Z_BAND_M[0]) & (pts[:, 2] <= BED_Z_BAND_M[1])]
    if band.shape[0] < 20:
        return BedDetection(found=False)
    # Dominant horizontal plane: the modal z (robust to a few stray returns), then the points flat
    # within tolerance of it — that is the surface (bed top), as opposed to walls / the bed's faces.
    z_med = float(np.median(band[:, 2]))
    plane = band[np.abs(band[:, 2] - z_med) <= BED_HORIZONTAL_TOL_M]
    if plane.shape[0] < 20:
        return BedDetection(found=False)
    # Area via an occupancy grid over the plane's xy footprint (robust to non-convex coverage).
    cells = np.unique(np.floor(plane[:, :2] / BED_GRID_M).astype(np.int64), axis=0)
    area = cells.shape[0] * BED_GRID_M * BED_GRID_M
    if area < BED_MIN_AREA_M2:
        return BedDetection(found=False)
    c = plane.mean(axis=0)
    return BedDetection(
        found=True,
        centroid_b=(float(c[0]), float(c[1]), float(c[2])),
        bearing_rad=float(math.atan2(c[1], c[0])),
        range_m=float(math.hypot(c[0], c[1])),
        area_m2=float(area),
        n_points=int(plane.shape[0]),
    )


def make_head_camera_cfg(prim_path: str, width: int = 640, height: int = 480):
    """The head RealSense D435i (robotics-connect depth_camera_sight): mounted on ``torso_link``,
    tilted 51.29° DOWN (the floor-plane-calibrated value). Used for the close-range bed-coverage check
    at the bedside — at that pitch it frames the near bed surface, not the room."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    t = math.radians(51.29) / 2.0                            # downward pitch (rot about body-Y; exact
    #                                                         sim convention verified when wired)
    return CameraCfg(
        prim_path=prim_path,
        update_period=0,
        width=width,
        height=height,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.0, 0.45),                           # brow, just above torso_link
            rot=(math.cos(t), 0.0, math.sin(t), 0.0),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
    )


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  HAND PLACEMENT — LiDAR-first; the depth camera is COARSE-ONLY (issue #3)
# ════════════════════════════════════════════════════════════════════════════════════════════════
# Issue #3 hardware note (the user's, from the robot): *near-field LiDAR is more accurate than the
# depth camera, because the depth camera produces stereoscopic vision artifacts that make hand
# placement less accurate.* So the behaviour layer takes the FINE hand target (where to put the
# gripper on the sheet corner / bed edge) from the NEAR-FIELD LiDAR (Livox MID-360, ~±4 cm in the
# controlled near range), and uses the head depth camera (RealSense D435i, IR stereo) only for the
# COARSE coverage check — white textureless bedding gives the stereo matcher no features, so its
# depth there is unreliable (corroborated by the depth-on-textureless literature: Seita et al. 2019
# pick on depth not RGB; stereo/structured-light degrade on textureless/specular cloth).
#
# On the real robot these are fed by robotics-connect: `lidar_sight.find_tables` + the near-field
# cloud for the grasp point, and `depth_camera_sight` for the coarse coverage frame. The functions
# below work on a body-frame LiDAR cloud (numpy), so the SAME code runs on the sim RayCaster cloud
# and the real lidar_sight cloud — the real-to-sim perception loop (RL_WHOLE_BODY_REACH §6).

NEARFIELD_RADIUS_M = 0.9        # within this range the MID-360 is hand-placement-grade (~±4 cm)
GRASP_EDGE_BAND_M = 0.10        # vertical band around the bed top to take grasp returns from
GRASP_MIN_POINTS = 8           # need this many near-field returns to trust a LiDAR grasp point


@dataclass
class GraspTarget:
    found: bool
    point_b: tuple[float, float, float] = (0.0, 0.0, 0.0)   # hand target, torso body frame
    source: str = "lidar"                                   # "lidar" (accurate) | "depth_coarse"
    confidence: float = 0.0                                 # 0..1 (n near-field returns / a soft cap)
    range_m: float = 0.0


def grasp_point_from_lidar(
    points_w: np.ndarray,
    sensor_pos_w,
    sensor_quat_w,
    bed: "BedDetection",
    side_y: float = 0.0,
) -> GraspTarget:
    """Localize a hand-placement target on the bed's near top edge from the NEAR-FIELD LiDAR.

    Given the detected bed (`bed`, from :func:`detect_bed`), keep the LiDAR returns that are (a) at
    the bed-top height (within ``GRASP_EDGE_BAND_M`` of ``bed.centroid_b`` z), (b) inside
    ``NEARFIELD_RADIUS_M`` (where the MID-360 is accurate), and (c) on the requested lateral side
    (``side_y`` > 0 = the robot's left, < 0 = right, 0 = whichever is nearest). The grasp point is
    the nearest-edge return on that side — i.e. the closest bit of bed-top surface the hand can grip.
    Returns a body-frame point the behaviour layer hands to the reach policy. LiDAR-first: prefer
    this over any depth-derived target (see the module note)."""
    if not bed.found:
        return GraspTarget(found=False)
    pts = _to_body(points_w, sensor_pos_w, sensor_quat_w)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = occlude(pts)
    if pts.shape[0] == 0:
        return GraspTarget(found=False)
    bed_z = bed.centroid_b[2]
    radial = np.linalg.norm(pts, axis=1)
    band = pts[
        (np.abs(pts[:, 2] - bed_z) <= GRASP_EDGE_BAND_M)
        & (radial <= NEARFIELD_RADIUS_M)
    ]
    if side_y > 0:
        band = band[band[:, 1] >= 0.0]
    elif side_y < 0:
        band = band[band[:, 1] < 0.0]
    if band.shape[0] < GRASP_MIN_POINTS:
        return GraspTarget(found=False)
    # The grasp point = the return closest to the robot (the near top edge) on that side.
    horiz = np.hypot(band[:, 0], band[:, 1])
    g = band[int(np.argmin(horiz))]
    return GraspTarget(
        found=True,
        point_b=(float(g[0]), float(g[1]), float(g[2])),
        source="lidar",
        confidence=float(min(1.0, band.shape[0] / 40.0)),
        range_m=float(np.hypot(g[0], g[1])),
    )


def coverage_fraction(rgb_or_mask: np.ndarray, bed_color_lo=(90, 90, 90)) -> float:
    """COARSE bed-coverage check from the head camera (the depth cam's only job here — never hand
    placement). Fraction of the framed bed that reads as 'covered' (bright sheet) vs the darker
    mattress/gap. Intentionally crude: a good-enough 'is the sheet spread?' signal for the tolerant
    'made' goal, not a precise measurement."""
    if rgb_or_mask.ndim == 3:
        bright = (rgb_or_mask[:, :, :3] >= np.array(bed_color_lo)).all(axis=2)
    else:
        bright = rgb_or_mask > 0
    return float(bright.mean()) if bright.size else 0.0


class BedPerception:
    """Behaviour-layer perception for IRL bed-making: detect → approach → hand-placement, LiDAR-first.

    Wraps the geometric primitives so the bed-making behaviour (and the human-partner coordinator)
    asks three questions without caring whether it's running in sim or on the robot:

      * ``detect(cloud, pose)``      — where is the bed? (LiDAR ``find_tables``, room-scale)
      * ``hand_target(cloud, pose, side)`` — where do I put my hand? (NEAR-FIELD LiDAR, accurate)
      * ``coverage(rgb)``            — is the sheet spread enough? (depth/RGB cam, COARSE only)

    On the real robot, feed ``detect``/``hand_target`` from robotics-connect ``lidar_sight`` and
    ``coverage`` from ``depth_camera_sight``; in sim, from the RayCaster + head Camera here. If a
    near-field LiDAR grasp point is unavailable, ``hand_target`` reports it (``found=False``) so the
    behaviour layer re-approaches rather than trusting an artifact-prone depth target."""

    def __init__(self) -> None:
        self.last_bed: BedDetection = BedDetection(found=False)

    def detect(self, points_w: np.ndarray, sensor_pos_w, sensor_quat_w) -> BedDetection:
        self.last_bed = detect_bed(points_w, sensor_pos_w, sensor_quat_w)
        return self.last_bed

    def hand_target(self, points_w: np.ndarray, sensor_pos_w, sensor_quat_w, side_y: float = 0.0) -> GraspTarget:
        if not self.last_bed.found:
            self.detect(points_w, sensor_pos_w, sensor_quat_w)
        return grasp_point_from_lidar(points_w, sensor_pos_w, sensor_quat_w, self.last_bed, side_y=side_y)

    @staticmethod
    def coverage(rgb: np.ndarray) -> float:
        return coverage_fraction(rgb)
