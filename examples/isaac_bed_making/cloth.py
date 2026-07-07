"""PhysX particle-cloth bedsheet + grasp helpers for the Isaac Sim demo.

Self-contained (only ``omni.physx`` + ``pxr``). Facts learned the hard way on the
DGX Spark and baked in here:

* The cloth mesh is built **by hand** as a numpy/USD **quad** grid. Quads matter:
  Isaac's auto particle-cloth turns every *mesh edge* into a stiff stretch spring,
  so a triangulated grid makes the cell diagonals inextensible and the sheet locks
  into a rigid plate; with quads the diagonal becomes a soft *shear* spring and the
  cloth drapes. Springs are kept elastic for the same reason.
* Particle cloth only simulates when **GPU dynamics** is enabled on the physics
  scene (:func:`enable_gpu_dynamics`). Without it the sheet is inert.
* On the **GPU pipeline the deformed particle positions never sync to the USD
  mesh / Fabric**, so a headless camera renders the flat authored mesh. Read the
  live positions from a PhysX *tensor cloth view* (:func:`make_cloth_view`,
  :func:`view_positions`) and blit them into the render mesh each frame. The demo
  runs ``SimulationCfg(use_fabric=True)`` (needed so the robot articulations render
  their motion), so the blit goes through the **Fabric** path
  (:func:`make_fabric_points` + :func:`sync_shell_fabric`), not USD.

A :func:`grasp` auto-attachment helper is provided, but the demo grips the cloth
by **friction** (rubberized hands) instead — see ``manipulation.apply_hand_friction``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, Vt


def enable_gpu_dynamics(stage, scene_path: str) -> None:
    """Enable GPU dynamics + GPU broadphase on the physics scene (required for
    particle cloth). Safe to call once after the World/Sim physics scene exists."""
    prim = stage.GetPrimAtPath(scene_path)
    psa = PhysxSchema.PhysxSceneAPI.Apply(prim)
    psa.CreateEnableGPUDynamicsAttr(True)
    psa.CreateBroadphaseTypeAttr("GPU")
    psa.CreateSolverTypeAttr("TGS")


def find_physics_scene_path(stage) -> Optional[str]:
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene):
            return p.GetPath().pathString
    return None


@dataclass
class Bedsheet:
    """Handle to a particle-cloth bedsheet and its grid topology."""

    prim_path: str
    mesh: object
    particle_system_path: str
    nx: int
    ny: int
    size: Tuple[float, float]
    # vertex ids of the four corners, keyed by compass label
    corner_vids: Dict[str, int]

    def vid(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i


def build_bedsheet(
    stage,
    scene_path: str,
    prim_path: str = "/World/Sheet",
    *,
    size: Tuple[float, float] = (2.2, 2.0),
    resolution: Tuple[int, int] = (44, 40),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.95),
    color: Tuple[float, float, float] = (0.85, 0.82, 0.72),
    # ── The "MuJoCo recipe" (validated 2026-06-06 in isolation, no robots) ──────
    # The shipped sheet draped as a stiff cantilever and "fluttered in the wind".
    # Root cause was NOT the engine — it was that the sheet was built HEAVY (~35 kg)
    # and FINE (~1750 verts), so a soft-spring lattice with that much inertia sloshes
    # in a limit cycle that no damping kills. MuJoCo's flex sheet (the one that looked
    # good) is the opposite: ~0.18 kg total, 143 verts, ~4.4 cm thick. Porting that
    # recipe to this SAME PhysX particle cloth — featherlight + coarse (set via the
    # scene's particle_mass / resolution) + low bend + more solver iters — drops the
    # frame-to-frame motion from ±5 cm to ±1 cm and the overhang folds DOWN over the
    # edge. (Quads still matter: a triangulated particle grid locks into a rigid plate
    # because every mesh edge becomes a stretch spring — see the mesh build below.)
    stretch_stiffness: float = 300.0,   # elastic (1500 was rigid; 150–300 look identical)
    bend_stiffness: float = 0.3,        # low so it folds over the edge; >2 = rigid cantilever,
                                        # <0.1 = the free edge curls/rolls
    shear_stiffness: float = 10.0,
    damping: float = 1.5,
    # PBD particle-material: cloth-surface friction kept LOW so the overhang slides
    # down the mattress side instead of catching. drag MUST be ~0 — even 0.1 acts like
    # a parachute on the flat falling flap and it never drapes (0.4 floated it dead flat).
    friction: float = 0.2,
    drag: float = 0.0,
    # More solver position iterations make each step CONVERGE — a too-low count leaves
    # the cloth in a buzzing limit cycle (the overhang "breathes" forever). 16 is the
    # Omniverse default; 48 settles it.
    solver_iterations: int = 48,
    # Per-particle mass. Keep the sheet FEATHERLIGHT: at the coarse scene resolution
    # this totals ~0.2 kg (MuJoCo was 0.18 kg). A heavy sheet sloshes; a light one
    # hangs dead still. (This is per particle, so it scales with vert count — pair it
    # with a COARSE resolution in the scene.)
    particle_mass: float = 0.001,
    thickness: float = 0.0,
    fold: bool = False,
    fold_start: float = 0.66,
    accordion: bool = False,
    accordion_start: float = 0.3,
    accordion_gather: float = 0.55,
    accordion_waves: int = 4,
    accordion_amp: float = 0.09,
) -> Bedsheet:
    """Create a draping particle-cloth sheet as a quad grid mesh.

    ``origin`` is the world translate of the sheet centre. With ``fold=False`` the
    sheet starts flat (+Z-up) and drapes under gravity. With ``fold=True`` the
    **foot end is folded back over itself**: the length up to ``fold_start`` lies
    flat, and the remaining foot fraction folds back over the top (as a turned-back
    sheet at the foot of the bed, like the Figure Helix clip) — so the bed starts
    unmade and the robots arrange it. The four cloth corners are still tracked.
    """
    from omni.physx.scripts import particleUtils, physicsUtils

    Wx, Wy = size
    nx, ny = resolution
    ox, oy, oz = origin
    layer_dz = max(thickness, 0.04)  # height the folded-back flap sits above the sheet
    x_fold = fold_start * Wx - Wx / 2.0  # local x of the fold line

    # Author the points directly in WORLD space (bake the origin in) and keep the
    # mesh transform identity — see the rendering note below: with use_fabric=True
    # the renderer reads the mesh's world transform from Fabric, so a non-identity
    # xformOp would be applied ON TOP of the world-space deformed points we blit each
    # frame, floating the sheet up by the origin. Identity transform + world points
    # means what we blit is exactly what renders.
    # Accordion: the head-side fraction up to ``accordion_start`` lies flat (the part
    # the robots grip); the foot fraction is GATHERED into a pleated ruffle — its flat
    # x-extent is compressed by ``accordion_gather`` and the slack taken up by
    # ``accordion_waves`` vertical pleats of height ``accordion_amp``. That slack is the
    # point: pulling the flat head edge toward the head simply UNSPOOLS the accordion
    # instead of dragging a sheet that is stuck flat to the mattress, so the friction
    # grip doesn't have to overpower the whole cover (and the robots aren't yanked over).
    acc_fold_x = accordion_start * Wx - Wx / 2.0
    acc_extent = (1.0 - accordion_start) * Wx

    pts: List[Gf.Vec3f] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            u, v = i / nx, j / ny
            if accordion and u > accordion_start:
                t = (u - accordion_start) / (1.0 - accordion_start)
                x = acc_fold_x + t * acc_extent * accordion_gather
                z = accordion_amp * (0.5 - 0.5 * math.cos(2.0 * math.pi * accordion_waves * t))
            elif fold and u > fold_start:
                # Fold the foot flap back over the top toward the head.
                x = x_fold - (u - fold_start) * Wx
                z = layer_dz
            else:
                x, z = u * Wx - Wx / 2.0, 0.0
            pts.append(Gf.Vec3f(x + ox, (v * Wy - Wy / 2.0) + oy, z + oz))
    idx: List[int] = []
    counts: List[int] = []

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    # Build the sheet as QUADS, not triangles. Isaac's auto particle-cloth turns
    # every *mesh edge* into a stiff stretch spring; a triangulated grid adds a
    # diagonal edge to each cell, so those diagonals become inextensible stretch
    # springs and lock the sheet into a rigid plate. With quads the diagonal is
    # NOT a mesh edge, so the solver adds it as a SOFT shear spring and the cloth
    # drapes. (MuJoCo's flex gets triangle elasticity for free; PBD does not.)
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            idx += [a, b, c, d]
            counts += [4]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateDisplayColorAttr().Set([color])
    # Identity transform — the points above are already in world space. (Authoring a
    # translate here and world-space points would double-transform the sheet under
    # Fabric and float it off the bed.)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    # Particle radius. Default: neighbours just touch at rest (radius = half the
    # grid spacing) — the canonical Omniverse recipe. A `thickness` override
    # fattens the particles into a "duvet"-like sheet with a thicker collision
    # profile the hands can catch more easily; capped just under the spacing so
    # particles don't overlap at rest (use a coarser grid for a thicker sheet).
    spacing = Wy / ny
    rest_offset = min(thickness / 2.0, 0.49 * spacing) if thickness > 0.0 else 0.5 * spacing
    contact_offset = rest_offset * 1.5
    ps_path = Sdf.Path(prim_path + "_particles")
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=ps_path,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
        particle_contact_offset=contact_offset,
        solid_rest_offset=rest_offset,
        fluid_rest_offset=0.0,
        solver_position_iterations=solver_iterations,
        simulation_owner=Sdf.Path(scene_path),
    )
    pmat = Sdf.Path(prim_path + "_material")
    particleUtils.add_pbd_particle_material(stage, pmat, drag=drag, lift=0.0, friction=friction)
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(ps_path), pmat)

    particleUtils.add_physx_particle_cloth(
        stage=stage,
        path=Sdf.Path(prim_path),
        dynamic_mesh_path=None,
        particle_system_path=ps_path,
        spring_stretch_stiffness=stretch_stiffness,
        spring_bend_stiffness=bend_stiffness,
        spring_shear_stiffness=shear_stiffness,
        spring_damping=damping,
        self_collision=True,
        self_collision_filter=True,
    )
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).GetMassAttr().Set(particle_mass * len(pts))

    # Map the four sheet corners onto compass labels (matches the swarm driver's
    # SHEET_TO_BED: A=NW, B=NE, C=SE, D=SW). +x = East, +y = North.
    corner_vids = {
        "NW": vid(0, ny),
        "NE": vid(nx, ny),
        "SE": vid(nx, 0),
        "SW": vid(0, 0),
    }
    return Bedsheet(prim_path, mesh, ps_path.pathString, nx, ny, size, corner_vids)


def grasp(stage, cloth_path: str, body_path: str, attach_path: str,
          bind_offset: float = 0.10) -> None:
    """Create a PhysX auto-attachment binding the cloth to a rigid body
    (a robot hand) where they overlap — i.e. grab whatever cloth is in the hand.

    ``bind_offset`` widens the overlap radius so cloth particles within that many
    metres of the palm are attached, letting the grasp catch the sheet even when
    the IK leaves the palm a few cm short of the surface."""
    if stage.GetPrimAtPath(attach_path):
        return
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, Sdf.Path(attach_path))
    att.GetActor0Rel().SetTargets([Sdf.Path(cloth_path)])
    att.GetActor1Rel().SetTargets([Sdf.Path(body_path)])
    api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    # Particle cloth binds through the "deformable vertex" path of the auto
    # attachment; widen its overlap offset. Wrapped defensively in case the
    # attribute set differs across PhysX schema versions.
    try:
        api.CreateEnableDeformableVertexAttachmentsAttr(True)
        api.CreateDeformableVertexOverlapOffsetAttr(float(bind_offset))
        api.CreateEnableRigidSurfaceAttachmentsAttr(True)
    except Exception:
        # These attachment attrs are absent on some PhysX schema versions; skip if so.
        pass
    # CRITICAL for the rigid GRAB TABS: auto-filter collisions between the cloth and the attached rigid
    # near the attachment. A tab sits COINCIDENT with the cloth particle it binds, and a cloth particle's
    # contact offset is several cm, so without this the tab collides with that particle — the repulsion
    # fights the attachment and a featherlight tab is flung to ~1e26 m, detonating the particle solver.
    # Eye-verified: the committed manual FilteredPairsAPI against the particle SYSTEM is NOT sufficient
    # here (both the committed baseline AND our runs exploded during the drape); this auto-attachment
    # collision filtering is the documented mechanism for exactly this. Harmless for the hand grasp.
    try:
        api.CreateEnableCollisionFilteringAttr(True)
        api.CreateCollisionFilteringOffsetAttr(max(float(bind_offset), 0.08))
    except Exception:
        # Collision-filtering attrs are optional across PhysX versions; skip if absent.
        pass


def add_spring_grip(stage, joint_path: str, wrist_path: str, tab_path: str, local_pos0,
                    *, stiffness: float = 500.0, damping: float = 30.0,
                    break_force: float = 45.0, break_torque: float = 1.0e9) -> str:
    """The **custom spring peel-off grip** — how a balancing G1 holds the cover without a kinematic
    cheat and lets go when the load gets dangerous.

    Creates a PhysX **D6 joint** (``UsdPhysics.Joint`` = a generic 6-DoF joint) between the robot's
    wrist link (``wrist_path``, body0) and a rigid grab tab sewn into the cover (``tab_path``, body1).
    The three LINEAR axes get a compliant position **drive** (a spring with ``stiffness`` N/m + ``damping``
    N·s/m) targeting zero displacement, so the tab is pulled to follow the wrist's grab-time relative
    pose; the three ANGULAR axes are left FREE so the gripped cloth/tab can pivot without fighting the
    hand. ``local_pos0`` (the tab anchor expressed in the wrist's local frame at grab time) makes the
    spring's rest length the current gap, so the grip engages with ~no jerk.

    Why this is sim-to-real valid (not a pin): PhysX applies the spring as a real bilateral force, so the
    cover's weight + mattress friction load the wrist and the whole-body policy must BALANCE against it —
    exactly the loco-manipulation work a real robot does. And ``break_force`` makes the joint **PEEL OFF**
    when the linear constraint load exceeds the threshold (requirement E): if the cover snags and the draw
    would yank the robot off its feet, the grip mechanically releases instead — a physical let-go, the
    behaviour we previously had to detect heuristically. Returns the joint prim path.
    """
    from pxr import UsdPhysics

    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)
    joint = UsdPhysics.Joint.Define(stage, Sdf.Path(joint_path))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(wrist_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tab_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2])))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))   # anchor at the tab's centre
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # A maximal-coordinate loop joint: the tab is a free rigid body, not part of the arm's reduced
    # articulation, so it must NOT be folded into the articulation solver.
    joint.CreateExcludeFromArticulationAttr(True)
    # Peel-off: PhysX disables the joint once the constraint force/torque exceeds these. We only peel on
    # the LINEAR draw load, so leave the torque threshold effectively infinite.
    joint.CreateBreakForceAttr(float(break_force))
    joint.CreateBreakTorqueAttr(float(break_torque))
    # Isotropic 3-axis linear spring toward the grab point (equal stiffness on each axis → a spring to a
    # point, independent of frame orientation). Rotation axes are left undriven/unlocked = free.
    for axis in ("transX", "transY", "transZ"):
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateStiffnessAttr(float(stiffness))
        drive.CreateDampingAttr(float(damping))
    return joint_path


def remove_grip(stage, joint_path: Optional[str]) -> None:
    """Release the spring grip (do_release / balance-loss). Idempotent — safe even if PhysX already
    broke the joint (the inert prim is simply removed) or the path is None."""
    if joint_path and stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)


def add_grab_tab(stage, cloth_path: str, tab_path: str, attach_path: str,
                 world_pos, radius: float = 0.03, mass: float = 0.004, friction: float = 12.0,
                 bind_offset: float = 0.10):
    """Sew a small rigid **grab tab** into the cloth at ``world_pos`` and return its prim path.

    WHY a rigid tab: in Isaac Sim 5.1 a particle-cloth ↔ **articulation-link** attachment is
    *unsupported* (NVIDIA IsaacLab #4291) — attaching the cloth straight to the robot's wrist link
    is exactly why the grip slips. A particle-cloth ↔ **free rigid body** attachment IS supported
    and holds. So we bind a tiny rigid sphere to the cloth (the supported direction); the robot's
    hand then physically grips THIS rigid tab (finger friction). The drag load transmits
    cloth → tab → hand → robot, so the robot still does real physical work and must balance against
    it — no kinematic pinning. The tab is a stand-in for the wad of cloth a real pinch grasp bunches
    up (which behaves semi-rigidly), and it is light enough not to distort the drape.
    """
    # A small SPHERE (NOT a box: box-shaped tabs detonate the particle solver — eye-verified head_v12;
    # spheres are stable). Roll-off is avoided by keeping tabs on the flat HEAD edge (head_only) and,
    # for the spring grip, filtering tabs from the robots (via collision groups — see
    # filter_tabs_from_robots) so the hand never plows into them.
    sph = UsdGeom.Sphere.Define(stage, tab_path)
    sph.CreateRadiusAttr(float(radius))
    sph.CreateDisplayColorAttr().Set([Gf.Vec3f(0.86, 0.86, 0.92)])
    UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3d(float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))
    prim = sph.GetPrim()
    # HIDE the tab from the renderer. It is a sim-only grip mechanism (a stand-in for the wad of cloth a
    # real pinch grasp bunches up), NOT part of the visual bed — rendered, the studded spheres read as
    # "little cubes on the sheet", an unacceptable artifact. Visibility is a pure USD/render attribute,
    # orthogonal to physics (same MakeInvisible the demo uses on the pillow colliders), so the rigid body
    # + cloth attachment + spring grip are all unchanged — the tab still does its physical job, unseen.
    # Set BEDDEMO_TAB_VISIBLE=1 to render them for debugging.
    if os.environ.get("BEDDEMO_TAB_VISIBLE", "0") not in ("1", "true", "True"):
        UsdGeom.Imageable(prim).MakeInvisible()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    # High-friction rigid material so the closed fingers hold the tab (rigid-on-rigid friction is
    # well-behaved, unlike a friction grasp on soft particle cloth which slips).
    from omni.physx.scripts import physicsUtils
    from pxr import UsdShade
    mat_path = Sdf.Path(tab_path + "_mat")
    mat = UsdShade.Material.Define(stage, mat_path)
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(float(friction))
    mapi.CreateDynamicFrictionAttr(float(friction))
    mapi.CreateRestitutionAttr(0.0)
    physicsUtils.add_physics_material_to_prim(stage, prim, mat_path)
    # FILTER tab↔cloth COLLISION: the tab is bound to the cloth by the attachment, so it must NOT
    # also collide with the cloth particles — a rigid body coincident with particles it is attached
    # to makes the collision repulsion fight the attachment and the particle solver explodes (the
    # head_v9 blow-up). Keep this rel to the cloth + its particles ONLY. Adding the ROBOT here as well
    # (to also stop the reaching hand plowing into the tabs) silently BROKE this filter and detonated
    # the solver — eye-verified spring_v1: the cover wadded up and the tabs scattered across the bed.
    # The tab↔robot exclusion is done SEPARATELY via collision groups (filter_tabs_from_robots), which
    # leaves this proven filter untouched.
    fp = UsdPhysics.FilteredPairsAPI.Apply(prim)
    rel = fp.CreateFilteredPairsRel()
    rel.AddTarget(Sdf.Path(cloth_path))
    rel.AddTarget(Sdf.Path(cloth_path + "_particles"))
    # Bind the tab to the cloth where they overlap (the SUPPORTED cloth↔rigid attachment).
    grasp(stage, cloth_path, tab_path, attach_path, bind_offset=bind_offset)
    return tab_path


def filter_tabs_from_robots(stage, tab_paths: List[str], robot_paths: List[str]) -> None:
    """Stop the grab tabs colliding with the robots, using PhysX **collision groups** rather than each
    tab's ``FilteredPairsAPI``.

    WHY collision groups and not the tab's filtered-pairs rel: the tab already filters its own cloth +
    particles through that rel, and adding the robot subtree to the SAME rel silently broke the
    cloth/particle filter and exploded the solver (eye-verified spring_v1). Collision groups are an
    orthogonal mechanism, so the proven tab↔particle filter stays intact while we separately exclude
    tab↔robot contact. With the spring peel-off grip the hand holds a tab through a D6 joint, not by
    touching it, so removing tab↔robot contact only kills the reach-time plow-topple (the head_v12 wall:
    the balancing robot toppled when its reaching hand plowed into the cluster of rigid head-edge tabs).
    Build this once at scene-build time, before the first step.
    """
    tab_grp = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path("/World/CollisionGroups/GrabTabs"))
    rob_grp = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path("/World/CollisionGroups/Robots"))
    tinc = tab_grp.GetCollidersCollectionAPI().CreateIncludesRel()
    for p in tab_paths:
        tinc.AddTarget(Sdf.Path(p))
    rinc = rob_grp.GetCollidersCollectionAPI().CreateIncludesRel()
    for rp in robot_paths:                       # a robot root includes its whole collider subtree
        rinc.AddTarget(Sdf.Path(rp))
    # Filtering is symmetric — declaring it once on the tab group excludes every tab↔robot pair.
    tab_grp.CreateFilteredGroupsRel().AddTarget(rob_grp.GetPath())


def add_perimeter_grab_tabs(stage, sheet: "Bedsheet", prefix: str = "/World/GrabTab",
                            depth: float = 0.33, stride_m: float = 0.18, radius: float = 0.025,
                            mass: float = 0.0003, friction: float = 12.0, bind_offset: float = 0.05,
                            head_only: bool = True, robot_paths: Optional[List[str]] = None):
    """Stud the cover with light rigid grab tabs, each bound to the cloth, so the hand has something
    rigid to grip wherever it closes — the motion policy's exact grab point isn't predictable, so a
    wide, deep grippable band gives tolerance. Built ONCE at scene-build time (before the first
    physics step) so the new rigid bodies are part of the initial state and don't shock a running solver.
    Each tab is filtered from ``robot_paths`` (the hand holds it via the spring grip, not by colliding).
    Returns a list of ``(tab_prim_path, vid)`` — ``vid`` is the cloth vertex the tab is sewn to, so the
    demo can read each tab's LIVE position from the cloth view (``view_positions(...)[vid]``) to find the
    nearest tab to a hand and to anchor the spring-grip joint, without a separate rigid-body view.

    ``head_only`` (default) restricts the band to the HEAD EDGE (the edge the robots draw up) at full
    ``depth``: tabs on the foot/side edges hang off the bed and drag the whole cover onto the floor
    (eye-verified head_v10/v11), and those edges are never grabbed in the bed-DRAW task. ``stride_m``
    subsamples ALONG the edge (the width) to keep the count — and the added stiffness — modest so the
    cover still crumples and drapes.
    """
    pts = list(sheet.mesh.GetPointsAttr().Get())   # authored world-space node positions
    nx, ny = sheet.nx, sheet.ny
    Wx, Wy = sheet.size
    dx, dy = Wx / nx, Wy / ny
    di = max(1, round(depth / dx))                 # band depth in node units (x = head→foot)
    dj = max(1, round(depth / dy))
    sj = max(1, round(stride_m / dy))              # subsample along the width (y)
    tabs, k = [], 0
    for j in range(0, ny + 1, sj):
        for i in range(nx + 1):
            if head_only:
                in_band = i < di                                       # head edge band, FULL depth
            else:
                in_band = (i < di or i > nx - di or j < dj or j > ny - dj)
            if not in_band:
                continue
            vid = j * (nx + 1) + i
            wp = pts[vid]
            add_grab_tab(stage, sheet.prim_path, f"{prefix}_{k}", f"{sheet.prim_path}_tab_{k}",
                         (wp[0], wp[1], wp[2]), radius=radius, mass=mass, friction=friction,
                         bind_offset=bind_offset)
            tabs.append((f"{prefix}_{k}", vid))
            k += 1
    if robot_paths:   # exclude tab↔robot collision via groups (NOT the tab's particle filter) — no plow-topple
        filter_tabs_from_robots(stage, [p for p, _ in tabs], robot_paths)
    return tabs


# ── live deformed positions + rendering ────────────────────────────────────
# On the GPU PhysX pipeline, particle-cloth deformation is NOT written back to
# the USD mesh (with Fabric on it isn't synced at all), so a headless camera
# renders the stale, flat authored mesh. The cloth IS draping in the backend —
# we read the live particle positions via the PhysX *tensor* cloth view and blit
# them into the render mesh ourselves each frame (the demo runs use_fabric=True,
# so the blit goes through the Fabric path below).
def make_cloth_view(prim_path: str = "/World/Sheet", backend: str = "torch"):
    """Create a PhysX tensor view over a particle cloth (call after the first
    sim step so the cloth is registered in the physics scene)."""
    import omni.physics.tensors as tensors

    sv = tensors.create_simulation_view(backend)
    sv.set_subspace_roots("/")
    return sv.create_particle_cloth_view(prim_path.replace(".*", "*"))


def view_positions(view, idx: int = 0):
    """Return the live deformed particle positions (world frame) as (N,3) numpy."""
    import numpy as np

    pos = view.get_positions()  # (count, max_particles*3), flat per cloth
    row = pos[idx]
    if hasattr(row, "detach"):
        row = row.detach().cpu().numpy()
    return np.asarray(row, dtype=float).reshape(-1, 3)


# ── Fabric (use_fabric=True) rendering path ─────────────────────────────────
# With use_fabric=True the RTX renderer reads geometry from Fabric, not USD, so a
# USD points blit is ignored and the cloth renders flat. We must run fabric on for
# the *robot* to render its motion: Isaac Lab's SimulationContext.forward() only
# calls update_articulations_kinematic() when fabric is enabled, so with fabric off
# the articulation renders frozen at its spawn pose. NVIDIA confirms (forums:
# "Using Fabric with Particles") that point-instancer changes can't go through
# Fabric but **mesh updates do** — our bedsheet is a UsdGeom.Mesh, so we write its
# deformed points straight into Fabric/usdrt each render. PhysX does NOT auto-sync
# particle-cloth positions to Fabric, so we still read them from the tensor cloth
# view. Fabric stores each prim's WORLD transform directly, so a non-identity mesh
# xform would be applied on top of the world-space points we write — floating the
# sheet up by the origin (the "sheet loads a few feet in the air" bug). We therefore
# build the mesh with its points already in world space and an IDENTITY transform
# (see build_bedsheet), so the deformed positions we blit render exactly where they
# are — no resetXformStack or per-frame transform fixup needed.
def make_fabric_points(prim_path: str = "/World/Sheet"):
    """Attach the Fabric/usdrt stage and return the cloth mesh's ``points``
    attribute for in-place per-frame updates (use with :func:`sync_shell_fabric`
    when running ``SimulationCfg(use_fabric=True)``). Call after ``sim.reset()`` and
    the first sim step. Returns ``None`` if the prim isn't in Fabric yet."""
    import omni.usd
    import usdrt

    rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    prim = rt_stage.GetPrimAtPath(prim_path)
    if not (prim and prim.IsValid()):
        return None
    attr = prim.GetAttribute("points")
    if attr and attr.IsValid():
        return attr
    return prim.CreateAttribute("points", usdrt.Sdf.ValueTypeNames.Point3fArray, True)


def corner_world_positions(view, sheet: "Bedsheet"):
    """Return {label: (x,y,z)} live world positions of the four sheet corners,
    read from the tensor cloth ``view`` (see :func:`make_cloth_view`)."""
    pts = view_positions(view)
    out = {}
    if pts.shape[0] == 0:
        return out
    for label, vid in sheet.corner_vids.items():
        if vid < pts.shape[0]:
            out[label] = tuple(round(float(c), 4) for c in pts[vid])
    return out


# ── Render-side VISUAL THICKNESS: a closed double-layer "shell" ─────────────
# The particle cloth is a single-layer membrane (one quad grid, N particles), so
# the renderer draws it as a zero-thickness sheet of paper — physically correct
# for the sim, but it *looks* thin. A real made bed reads as a substantial,
# thick cover. We give it visual body WITHOUT touching the physics: the membrane
# stays the simulated cloth, and we drive a SEPARATE visual mesh — the same grid
# extruded into a slab — from the same live particle positions each frame.
#
# Construction (static topology, points refreshed per frame):
#   * TOP layer    = membrane + n·(h/2)   (n = per-vertex surface normal)
#   * BOTTOM layer = membrane − n·(h/2)
#   * SIDE WALLS   = quads stitching the top and bottom rims around the perimeter
# so the result is a closed quad slab of thickness ``h`` that hugs the draped
# membrane (offset along the normal, so the thickness stays uniform even where the
# overhang folds down vertical). The membrane mesh is hidden (the slab encloses it
# anyway); hiding it is safe — particle cloth simulates off the particle system,
# not the mesh's render visibility, and if the sim ever *did* die the slab (built
# FROM the live particle positions) would render flat, so the mp4 still catches it.
@dataclass
class ShellMesh:
    """Handle to the visual double-layer slab that thickens a :class:`Bedsheet`."""

    prim_path: str
    mesh: object
    nx: int
    ny: int
    n_particles: int   # membrane vertex count = (nx+1)*(ny+1)
    thickness: float
    # How the slab straddles the membrane along the surface normal, in [0,1]:
    #   0.0 = centred (±h/2)   1.0 = fully OUTWARD (membrane is the inner face, slab
    # extends +h along +normal). Outward is the default: centred buries the bottom
    # layer h/2 BELOW the membrane, and since the membrane rests on the mattress the
    # buried half sits inside the mattress — the sharp mattress top edge then pokes
    # up through the slab (the "mattress edge clips the sheet" artifact). Extruding
    # outward keeps the whole slab on the away-from-bed side of the cloth, so the
    # mattress can never intersect it.
    bias: float = 1.0


def _grid_vertex_normals(grid):
    """Per-vertex unit normals for a draped quad grid.

    ``grid`` is (ny+1, nx+1, 3) world positions (row j = +y, col i = +x). The
    normal is cross(∂/∂i, ∂/∂j) so a flat sheet in the xy-plane yields +z; where
    the sheet folds down over an edge the normal rotates with it, keeping the
    extruded thickness perpendicular to the surface everywhere."""
    import numpy as np

    du = np.gradient(grid, axis=1)   # tangent along +x (columns)
    dv = np.gradient(grid, axis=0)   # tangent along +y (rows)
    n = np.cross(du, dv)
    mag = np.linalg.norm(n, axis=2, keepdims=True)
    return n / np.where(mag < 1e-9, 1.0, mag)


def shell_points(membrane_pts, nx: int, ny: int, thickness: float, bias: float = 1.0):
    """Map the N membrane particle positions to the 2N slab points.

    ``membrane_pts`` is (N,3) in particle/``vid`` order (vid = j*(nx+1)+i). Returns
    (2N,3): the TOP (outer) layer (indices 0..N-1) then the BOTTOM (inner) layer
    (N..2N-1), each in the same vid order. ``bias`` slides the slab along the normal:
    1.0 = fully outward (inner face = membrane), 0.0 = centred (±h/2). See
    :class:`ShellMesh` for why outward avoids the mattress-edge clip."""
    import numpy as np

    grid = np.asarray(membrane_pts, dtype=float).reshape(ny + 1, nx + 1, 3)
    n = _grid_vertex_normals(grid)
    hi = thickness * (0.5 + 0.5 * bias)        # outer offset: 0.5h..h
    lo = -thickness * 0.5 * (1.0 - bias)       # inner offset: -0.5h..0
    top = (grid + n * hi).reshape(-1, 3)
    bot = (grid + n * lo).reshape(-1, 3)
    return np.concatenate([top, bot], axis=0)


def build_shell_mesh(
    stage,
    sheet: "Bedsheet",
    *,
    thickness: float = 0.06,
    bias: float = 1.0,
    color: Tuple[float, float, float] = (0.86, 0.86, 0.92),
    prim_path: str = "/World/SheetShell",
    hide_membrane: bool = True,
) -> ShellMesh:
    """Create the closed double-layer visual slab for ``sheet`` and (by default)
    hide the thin membrane. Topology is fixed; call :func:`sync_shell_fabric` each frame to push the live
    deformed points."""
    nx, ny = sheet.nx, sheet.ny
    n_part = (nx + 1) * (ny + 1)

    def top(i: int, j: int) -> int:
        return j * (nx + 1) + i

    def bot(i: int, j: int) -> int:
        return n_part + j * (nx + 1) + i

    idx: List[int] = []
    counts: List[int] = []
    # TOP surface — same winding as the membrane (CCW from above → normal up).
    for j in range(ny):
        for i in range(nx):
            idx += [top(i, j), top(i + 1, j), top(i + 1, j + 1), top(i, j + 1)]
            counts.append(4)
    # BOTTOM surface — reversed winding so its normal points down.
    for j in range(ny):
        for i in range(nx):
            idx += [bot(i, j), bot(i, j + 1), bot(i + 1, j + 1), bot(i + 1, j)]
            counts.append(4)
    # PERIMETER side walls — march the boundary loop CCW (viewed from +z) and
    # stitch each top edge to the matching bottom edge.
    perim: List[Tuple[int, int]] = (
        [(i, 0) for i in range(nx + 1)]          # foot edge, +x
        + [(nx, j) for j in range(1, ny + 1)]    # right edge, +y
        + [(i, ny) for i in range(nx - 1, -1, -1)]  # head edge, −x
        + [(0, j) for j in range(ny - 1, 0, -1)]    # left edge, −y
    )
    for (ia, ja), (ib, jb) in zip(perim, perim[1:] + perim[:1]):
        idx += [top(ia, ja), top(ib, jb), bot(ib, jb), bot(ia, ja)]
        counts.append(4)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * (2 * n_part)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateDisplayColorAttr().Set([color])
    # doubleSided so the slab is opaque from every angle regardless of winding;
    # subdivision "none" so the renderer draws the actual extruded quads (a crisp
    # cover) rather than Catmull-Clark-rounding the slab into a pillow.
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    # Identity transform + world-space points (same contract as the membrane mesh:
    # under Fabric a non-identity xform double-transforms the world points).
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    if hide_membrane:
        UsdGeom.Imageable(sheet.mesh).MakeInvisible()

    return ShellMesh(prim_path, mesh, nx, ny, n_part, thickness, bias)


def sync_shell_fabric(view, fabric_points, shell: ShellMesh, idx: int = 0):
    """Blit the live deformed slab points into the shell's Fabric ``points`` (for
    ``use_fabric=True``). ``fabric_points`` comes from :func:`make_fabric_points`
    on the shell prim. Returns the (N,3) membrane positions."""
    import numpy as np
    import usdrt

    p = view_positions(view, idx)
    sp = shell_points(p, shell.nx, shell.ny, shell.thickness, shell.bias)
    fabric_points.Set(usdrt.Vt.Vec3fArray(np.ascontiguousarray(sp, dtype=np.float32)))
    return p
