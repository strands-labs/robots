"""The record session controller and its ``/api/record`` router.

One recording session at a time: a leader arm drives a follower, frames land in a LeRobot dataset,
and the fleet peers for both arms are parked for the duration and respawned on close.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from strands_robots.dashboard import access, record_crash, record_joints
from strands_robots.dashboard.dataset_check import (
    OUTSIDE_DATASET_HOME,
    _as_int,
    dataset_id_error,
    dataset_id_is_a_path,
    record_target_verdict,
)
from strands_robots.dashboard.record_worker import RecordWorker, hardware_backend, thumb_name
from strands_robots.utils import refusal_str, require_optional

logger = logging.getLogger(__name__)

#: robot_name of the arm driven as leader -> lerobot teleoperator type
LEADER_TYPES = {
    "so100": "so100_leader",
    "so101": "so101_leader",
    "koch": "koch_leader",
}

EMPTY_SESSION: dict[str, Any] = {
    "dataset": None,
    "task": "",
    "leader": None,
    "follower": None,
    "target_episodes": 10,
    "fps": 30,
    "phase": "idle",
    "episodes": [],
    "error": None,
}


def _under_dataset_home(dataset: str) -> bool:
    """Whether the directory this id resolves to sits under ``$HF_LEROBOT_HOME``.

    The id has already passed :func:`dataset_id_error`, so this is the second lock on the same
    door: whatever ``resolve_dataset_dir`` makes of the name, the result is checked after
    ``resolve()`` (symlinks followed, ``..`` folded) against the home.
    """
    from strands_robots.dataset_recorder import resolve_dataset_dir
    from strands_robots.dataset_source import _lerobot_home

    try:
        home = Path(_lerobot_home()).expanduser().resolve()
        target = Path(resolve_dataset_dir(dataset)).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return target == home or target.is_relative_to(home)


def _target_facts(dataset: str) -> dict[str, Any]:
    """What is on disk where this dataset would be written."""
    name = (dataset or "").strip()
    if not name:
        return {}
    try:
        from strands_robots.dataset_recorder import resolve_dataset_dir

        d = resolve_dataset_dir(name)
        if not d.exists():
            return {"exists": False}
        meta = (d / "meta" / "info.json").exists()
        episodes = None
        if meta:
            episodes = _as_int(json.loads((d / "meta" / "info.json").read_text(encoding="utf-8")).get("total_episodes"))
        return {
            "exists": True,
            "has_meta": meta,
            "episodes": episodes,
            "non_empty": any(d.iterdir()),
        }
    except Exception:  # noqa: BLE001 - a blind check must never block a recording
        return {}


def _hub_facts(repo_id: str | None) -> dict[str, Any]:
    name = (repo_id or "").strip().strip("/")
    if not name or "/" not in name:
        return {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        if not api.repo_exists(repo_id=name, repo_type="dataset"):
            return {"exists": False}
        episodes = None
        try:
            info = api.dataset_info(repo_id=name)
            card = getattr(info, "cardData", None) or {}
            raw = card.get("total_episodes") if isinstance(card, dict) else None
            episodes = _as_int(raw)
        except Exception:  # noqa: BLE001 - the count is a nicety, existence is the fact
            episodes = None
        return {"exists": True, "episodes": episodes}
    except Exception:  # noqa: BLE001
        return {}


def _default_recorder_factory(backend: Any) -> Callable[..., Any]:
    def make(*, repo_id: str, fps: int, task: str) -> Any:
        from strands_robots.dataset_recorder import DatasetRecorder

        extra = backend.recorder_kwargs() if hasattr(backend, "recorder_kwargs") else {}
        return DatasetRecorder.create(repo_id=repo_id, fps=fps, task=task, **extra)

    return make


def lerobot_refusal(purpose: str = "recording teleop datasets") -> HTTPException | None:
    """The refusal to hand back when lerobot is not installed, or None when it is.

    The recorder, the leader teleoperator and the dataset format are all lerobot's; without the
    ``lerobot`` extra a session cannot open, so the route says which extra instead of failing
    somewhere inside the first import.
    """
    try:
        require_optional("lerobot", extra="lerobot", purpose=purpose)
    except ImportError as exc:
        return HTTPException(
            424,
            {
                "error": "missing_optional",
                "extra": "lerobot",
                "install": "pip install 'strands-robots[lerobot]'",
                "detail": str(exc),
            },
        )
    return None


class RecordController:
    """One recording session at a time, with fleet peers parked around it.

    ``devices`` is the dashboard's :class:`~strands_robots.dashboard.device_manager.DeviceManager`;
    ``None`` builds a controller that answers the session view but refuses to open (no arms to park).
    """

    def __init__(
        self,
        devices: Any,
        *,
        backend_factory: Callable[..., Any] | None = None,
        recorder_factory_factory: Callable[[Any], Callable[..., Any]] | None = None,
        thumb_root: str | None = None,
        bridge: Any | None = None,
    ) -> None:
        self._devices = devices
        # Where the live per-camera frame evidence comes from.
        self._bridge = bridge
        self._backend_factory = backend_factory or hardware_backend
        self._recorder_factory_factory = recorder_factory_factory or _default_recorder_factory
        self._thumb_root = Path(thumb_root or (Path(tempfile.gettempdir()) / "strands-record-thumbs"))
        self._lock = threading.Lock()
        self._worker: RecordWorker | None = None
        self._crumb = record_crash.crumb_path()
        self._parked: list[dict[str, Any]] = []
        self._disk_cache: dict[str, Any] | None = None
        self._disk_seen: float | None = None

    DISK_POLL_S = 20.0

    def _disk_notice(self) -> dict[str, Any] | None:
        """The free-space warning for the volume datasets land on, or None."""
        now = time.time()
        if self._disk_seen is not None and now - self._disk_seen < self.DISK_POLL_S:
            return self._disk_cache
        try:
            from strands_robots.dashboard import disk_headroom
        except ImportError:  # the notice is a nicety; the devices lane ships the gauge
            self._disk_seen = now
            self._disk_cache = None
            return None
        got = disk_headroom.free_space()
        self._disk_cache = disk_headroom.headroom_verdict(
            free_mb=got.get("free_mb"),
            total_mb=got.get("total_mb"),
            where="the dataset home",
        )
        self._disk_seen = now
        return self._disk_cache

    # ------------------------------------------------------------- session

    def session(self) -> dict[str, Any]:
        """The wire-shaped session view: idle (with any crash and disk notices) or the live worker."""
        with self._lock:
            if self._worker is None:
                idle = dict(EMPTY_SESSION)
                crumb = record_crash.read_crumb(self._crumb)
                notice = record_crash.interrupted_notice(
                    crumb, same_process=bool(crumb) and cast("dict[str, Any]", crumb).get("pid") == os.getpid()
                )
                if notice:
                    idle["interrupted"] = notice
                disk = self._disk_notice()
                if disk:
                    idle["disk_notice"] = disk
                return idle
            live = self._worker.session()
            disk = self._disk_notice()
            if disk:
                live = {**live, "disk_notice": disk}
            return live

    @property
    def thumb_dir(self) -> Path:
        """Where episode thumbnails are written."""
        return self._thumb_root

    # ---------------------------------------------------------------- open

    def open(self, body: dict[str, Any]) -> dict[str, Any]:
        """Park a leader/follower pair out of the fleet and start the teleop record loop."""
        with self._lock:
            if self._worker is not None:
                raise HTTPException(409, "a recording session is already open - close it first")
            refusal = lerobot_refusal()
            if refusal is not None:
                raise refusal
            if self._devices is None:
                raise HTTPException(
                    503,
                    "no device manager is attached to this dashboard - recording parks real arms spawned "
                    "from the Devices screen, and there is none to park",
                )
            dataset = str(body.get("dataset", "")).strip()
            task = str(body.get("task", "")).strip()
            leader_id = str(body.get("leader", "")).strip()
            follower_id = str(body.get("follower", "")).strip()

            # Shape, then containment, then what is on disk - in that order, so a name that is a
            # path never reaches the existence check and the refusal never says whether it exists.
            if dataset_id_is_a_path(dataset) or (dataset and not _under_dataset_home(dataset)):
                raise HTTPException(400, OUTSIDE_DATASET_HOME)
            bad_id = dataset_id_error(dataset)
            if bad_id:
                raise HTTPException(422, bad_id)
            bad = record_target_verdict(dataset, **_target_facts(dataset))
            if bad:
                raise HTTPException(409 if dataset.strip() else 422, bad)

            leader = self._managed(leader_id, role="leader")
            follower = self._managed(follower_id, role="follower")

            # Joints BEFORE cameras: a missing camera view is a degraded dataset, positions that cannot be
            # read are an empty one.
            _now = time.time()
            for _role, _pid in (("leader", leader_id), ("follower", follower_id)):
                bad_joints = record_joints.refusal(
                    role=_role,
                    peer_id=_pid,
                    peer=self._peer_snapshot(_pid),
                    problem=self._joint_problem(_pid),
                    now=_now,
                )
                if bad_joints:
                    raise HTTPException(409, bad_joints)

            # A dataset is the expensive artifact here: an hour of hand-guiding an arm, discovered to be
            # useless at training time.
            from strands_robots.dashboard import camera_liveness

            if not body.get("ignore_dead_cameras"):
                dead = camera_liveness.dead_cameras(follower.cameras, self._camera_meta(follower_id), now=time.time())
                if dead:
                    raise HTTPException(409, camera_liveness.refusal(dead, peer_id=follower_id))

            # The rail above judges frame AGE, so it is blind to a camera that never published at all -
            # which is exactly a camera unplugged before the arm subscribed.
            if not body.get("ignore_missing_cameras"):
                missing = camera_liveness.missing_cameras(follower.cameras, self._present_camera_indices())
                if missing:
                    raise HTTPException(409, camera_liveness.missing_refusal(missing, peer_id=follower_id))

            # The third shape, and the one both rails above call healthy: an index is a POSITION in a list
            # that closes up when a device is removed, so pulling the camera at index 1 slides index 2
            # into its place.
            if not body.get("ignore_camera_identity"):
                drift = camera_liveness.identity_drift(follower.cameras, self._present_camera_roster())
                if drift:
                    raise HTTPException(409, camera_liveness.drift_refusal(drift, peer_id=follower_id))

            leader_type = str(body.get("leader_type") or LEADER_TYPES.get(leader.robot_name, "")).strip()
            if not leader_type:
                raise HTTPException(
                    422,
                    f"no teleoperator type known for leader robot "
                    f"'{leader.robot_name}' - pass leader_type explicitly "
                    f"(known: {sorted(LEADER_TYPES)})",
                )

            # Park the peers: remember their spawn configs, stop them, and
            # keep the watcher's hands off the freed ports.
            parked = [
                self._spawn_cfg(leader),
                self._spawn_cfg(follower),
            ]
            watcher = getattr(self._devices, "autospawn", None)
            if watcher is not None:
                watcher.suspended = True
            self._devices.despawn(leader_id)
            self._devices.despawn(follower_id)
            self._parked = parked

            try:
                backend = self._backend_factory(
                    follower_name=follower.robot_name,
                    follower_port=follower.port,
                    leader_type=leader_type,
                    leader_port=leader.port,
                    cameras=follower.cameras,
                )
                self._worker = RecordWorker(
                    dataset=dataset,
                    task=task,
                    leader=leader_id,
                    follower=follower_id,
                    target_episodes=int(body.get("target_episodes", 10) or 10),
                    fps=int(body.get("fps", 30) or 30),
                    backend=backend,
                    recorder_factory=self._recorder_factory_factory(backend),
                    thumb_dir=self._thumb_root,
                )
            except HTTPException as exc:
                lost = self._unpark_locked()
                if not lost:
                    raise
                raise HTTPException(exc.status_code, _with_lost_arms(exc.detail, lost)) from exc
            except ValueError as exc:
                lost = self._unpark_locked()
                raise HTTPException(422, _with_lost_arms(str(exc), lost)) from exc
            except Exception as exc:
                # a failed open must leave the fleet exactly as it found it - and admit it when it could not
                lost = self._unpark_locked()
                raise HTTPException(500, _with_lost_arms(f"could not open the arms: {exc}", lost)) from exc
            opened = self._worker.session()
            record_crash.write_crumb(opened, path=self._crumb)
            return opened

    # : How old an already-taken camera roster may be and still count as evidence.
    ROSTER_MAX_AGE_S = 300.0

    def _present_camera_roster(self) -> tuple[dict[str, Any], ...]:
        """The camera roster this machine ALREADY took, if it is still evidence. Deliberately does not
        trigger a scan.
        """
        devices = self._devices
        try:
            roster = getattr(devices, "_camera_names_cache", None)
            taken_at = float(getattr(devices, "_camera_names_cache_t", 0.0) or 0.0)
            if not roster or taken_at <= 0 or time.monotonic() - taken_at > self.ROSTER_MAX_AGE_S:
                return ()
            return tuple(dict(r) for r in roster if isinstance(r, Mapping))
        except Exception:  # noqa: BLE001 - evidence gathering must never break a session
            logger.debug("[record] could not read the camera roster")
            return ()

    def _present_camera_indices(self) -> tuple[int, ...]:
        """Just the indices of :meth:`_present_camera_roster`, for the absence rail."""
        return tuple(
            int(r["listing_index"])
            for r in self._present_camera_roster()
            if isinstance(r.get("listing_index"), int) and not isinstance(r.get("listing_index"), bool)
        )

    def _camera_meta(self, peer_id: str) -> dict[str, Any]:
        """What the fleet snapshot last saw from this peer's cameras."""
        bridge = self._bridge
        if bridge is None:
            return {}
        try:
            peers = bridge.snapshot().get("peers") or {}
            cams = (peers.get(peer_id) or {}).get("cameras") or {}
            return dict(cams) if isinstance(cams, dict) else {}
        except Exception:  # noqa: BLE001 - a probe must not break the session
            logger.debug("[record] could not read camera evidence for %s", peer_id)
            return {}

    def _peer_snapshot(self, peer_id: str) -> Any:
        """This peer's entry in the live mesh snapshot, or None when we cannot see it."""
        try:
            bridge = self._bridge
            if bridge is None:
                return None
            return (bridge.snapshot().get("peers") or {}).get(peer_id)
        except Exception:  # noqa: BLE001 - a probe must not break the session
            logger.debug("[record] could not read the mesh snapshot for %s", peer_id)
            return None

    def _joint_problem(self, peer_id: str) -> Any:
        """The classified reason this peer publishes no joints, if the dashboard has one."""
        try:
            fields = self._devices.annotations_by_peer().get(peer_id) or {}
            return fields.get("joint_problem")
        except Exception:  # noqa: BLE001
            logger.debug("[record] could not read the joint annotation for %s", peer_id)
            return None

    def _managed(self, peer_id: str, *, role: str) -> Any:
        if not peer_id:
            raise HTTPException(422, f"{role} peer_id is required")
        m = getattr(self._devices, "robots", {}).get(peer_id)
        if m is None:
            raise HTTPException(
                404,
                f"{role} '{peer_id}' is not a robot this dashboard manages - "
                f"recording needs both arms spawned from this machine",
            )
        if m.mode != "real" or not m.port:
            raise HTTPException(
                422,
                f"{role} '{peer_id}' is {m.mode} - recording teleop episodes needs a real arm with a serial port",
            )
        return m

    @staticmethod
    def _spawn_cfg(m: Any) -> dict[str, Any]:
        return {
            "robot_name": m.robot_name,
            "mode": m.mode,
            "peer_id": m.peer_id,
            "port": m.port,
            "cameras": m.cameras,
            "remember": False,
        }

    # --------------------------------------------------------------- steps

    def _require_worker(self) -> RecordWorker:
        if self._worker is None:
            raise HTTPException(409, "no recording session is open")
        return self._worker

    def start_episode(self) -> dict[str, Any]:
        """Begin recording frames into a new episode."""
        with self._lock:
            return self._require_worker().start_episode()

    def stop_episode(self) -> dict[str, Any]:
        """Stop the episode in flight and save it."""
        with self._lock:
            return self._require_worker().stop_episode()

    def redo_episode(self) -> dict[str, Any]:
        """Drop the episode in flight without saving it."""
        with self._lock:
            return self._require_worker().redo_episode()

    def discard(self, index: int) -> dict[str, Any]:
        """Mark a saved episode as excluded from training."""
        with self._lock:
            try:
                return self._require_worker().discard(index)
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0])) from exc

    # --------------------------------------------------------------- close

    def close(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Finalize (optionally upload) the dataset and bring the parked arms back to the fleet."""
        body = body or {}
        with self._lock:
            worker = self._require_worker()
            try:
                result = worker.close(
                    upload=bool(body.get("upload")),
                    repo_id=(str(body.get("repo_id") or "").strip() or None),
                )
            finally:
                # the fleet comes back whether or not finalize/upload worked -
                # a hub hiccup must not leave the desk armless
                self._worker = None
                record_crash.clear_crumb(self._crumb)
                self._unpark_locked()
            return result

    def _unpark_locked(self) -> list[str]:
        lost: list[str] = []
        for cfg in self._parked:
            peer = str(cfg.get("peer_id") or "an arm")
            try:
                res = self._devices.spawn(**cfg)
                if res.get("error"):
                    logger.warning(
                        "respawn of %s after record session failed: %s",
                        cfg.get("peer_id"),
                        res["error"],
                    )
                    lost.append(f"{peer} ({res['error']})")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "respawn of %s after record session failed: %r",
                    cfg.get("peer_id"),
                    exc,
                )
                lost.append(f"{peer} ({exc})")
        self._parked = []
        watcher = getattr(self._devices, "autospawn", None)
        if watcher is not None:
            watcher.suspended = False
        return lost


def _with_lost_arms(detail: Any, lost: Sequence[str]) -> Any:
    """Append the arms that did not come back to whatever the refusal was going to say."""
    if not lost:
        return detail
    note = (
        f"AND THE FLEET DID NOT FULLY RECOVER: {', '.join(lost)} could not be respawned. "
        "Bring it back from Devices (spawn remembered) before trying again."
    )
    if isinstance(detail, dict):
        merged = dict(detail)
        merged["arms_not_restored"] = list(lost)
        merged["hint"] = f"{detail['hint']} {note}" if isinstance(detail.get("hint"), str) else note
        return merged
    return f"{refusal_str(detail)} - {note}" if detail else note


def build_router(
    controller: RecordController | Callable[[Request], RecordController],
    on_activity: Callable[..., None] | Callable[[Request], Callable[..., None] | None] | None = None,
    *,
    late_bound: bool = False,
) -> APIRouter:
    """The HTTP surface under ``/api/record``; every route needs a session (401 otherwise).

    ``controller`` is either the controller itself or, with ``late_bound=True``, a resolver called with
    the request (``routes_record`` reads it off ``app.state`` so the router can exist before the app
    does). ``on_activity`` follows the same rule: a callable taking ``(source, action, ...)``, or with
    ``late_bound`` a resolver returning that callable (or None) for the request's app.
    """
    r = APIRouter(prefix="/api/record", tags=["record"], dependencies=[Depends(access.require_session)])

    def ctl(request: Request) -> RecordController:
        if late_bound:
            return cast("Callable[[Request], RecordController]", controller)(request)
        return cast("RecordController", controller)

    def activity(request: Request) -> Callable[..., None] | None:
        if on_activity is None:
            return None
        if late_bound:
            return cast("Callable[[Request], Callable[..., None] | None]", on_activity)(request)
        return cast("Callable[..., None]", on_activity)

    @r.get("/session")
    async def session(request: Request) -> dict[str, Any]:
        return ctl(request).session()

    @r.get("/upload-preflight")
    async def upload_preflight_route(request: Request) -> dict[str, Any]:
        from strands_robots.dashboard.checkpoints import hf_auth_state
        from strands_robots.dashboard.upload_preflight import destination, upload_preflight

        current = ctl(request).session() or {}
        dataset = current.get("dataset") or current.get("repo_id")

        def judge() -> dict[str, Any]:
            auth = hf_auth_state()
            user = auth.get("user") if isinstance(auth, dict) else None
            existing = None
            if isinstance(auth, dict) and auth.get("authenticated") is True:
                existing = _hub_facts(destination(dataset or "", user if isinstance(user, str) else None))
            return upload_preflight(dataset=dataset, auth=auth, existing=existing)

        return await asyncio.to_thread(judge)

    @r.post("/open")
    async def open_session(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        result = ctl(request).open(body)
        # Opening a session PARKS real arms into teleop - an audit-worthy moment: it is the record
        # screen taking the robots away from the fleet.
        note = activity(request)
        if note is not None:
            note(
                "record",
                "session_open",
                target=str(body.get("dataset") or body.get("repo_id") or "session"),
                detail=f"task={body.get('task', '')!r}",
                ok="error" not in result,
            )
        return result

    @r.post("/episode/start")
    async def episode_start(request: Request) -> dict[str, Any]:
        return ctl(request).start_episode()

    @r.post("/episode/stop")
    async def episode_stop(request: Request) -> dict[str, Any]:
        return ctl(request).stop_episode()

    @r.post("/episode/redo")
    async def episode_redo(request: Request) -> dict[str, Any]:
        return ctl(request).redo_episode()

    @r.post("/episode/discard")
    async def episode_discard(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        if "index" not in body:
            raise HTTPException(422, "index required")
        return ctl(request).discard(int(body["index"]))

    @r.post("/close")
    async def close_session(request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
        result = ctl(request).close(body)
        note = activity(request)
        if note is not None:
            episodes = result.get("episodes_kept", result.get("episodes"))
            note(
                "record",
                "session_close",
                target="session",
                detail=None if episodes is None else f"{episodes} episodes",
                ok="error" not in result,
            )
        return result

    @r.get("/thumb/{episode}/{camera}")
    async def thumb(request: Request, episode: int, camera: str) -> FileResponse:
        # camera comes from a URL path - keep it a bare name, no traversal (the writer in
        # record_worker.tick applies the same record_worker.thumb_name reduction)
        safe = thumb_name(camera)
        path = ctl(request).thumb_dir / f"{int(episode)}_{safe}.jpg"
        if not path.is_file():
            raise HTTPException(404, "no thumbnail for that episode/camera")
        return FileResponse(path, media_type="image/jpeg")

    return r
