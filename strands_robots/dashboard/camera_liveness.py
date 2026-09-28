"""Is a camera this recording is about to depend on actually producing frames?"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEAD_CAMERA_AGE_S = 120.0


def camera_age(meta: Any, now: float) -> float | None:
    """Seconds since this camera's last CAPTURE, or None when unknowable."""
    if not isinstance(meta, Mapping):
        return None
    t = meta.get("t")
    if not isinstance(t, (int, float)) or isinstance(t, bool) or t <= 0:
        return None
    age = now - float(t)
    if age < 0:
        return None
    return age


def dead_cameras(
    configured: Iterable[str] | Mapping[str, Any] | None,
    cam_meta: Mapping[str, Any] | None,
    *,
    now: float,
    max_age_s: float = DEAD_CAMERA_AGE_S,
) -> list[dict[str, Any]]:
    """The configured cameras we have POSITIVE evidence have stopped publishing."""
    names = list(configured.keys()) if isinstance(configured, Mapping) else list(configured or [])
    meta = cam_meta or {}
    out: list[dict[str, Any]] = []
    for name in names:
        age = camera_age(meta.get(name), now)
        if age is not None and age > max_age_s:
            out.append({"camera": str(name), "age_s": round(age, 1)})
    return out


def _ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def refusal(dead: list[dict[str, Any]], *, peer_id: str) -> str:
    """Why this session is refused, in the words an operator can act on."""
    which = ", ".join(f"{d['camera']} (last frame {_ago(float(d['age_s']))})" for d in dead)
    plural = "cameras" if len(dead) > 1 else "camera"
    return (
        f"{peer_id}: {len(dead)} configured {plural} stopped publishing - {which}. "
        "Recording now would produce episodes with a frozen or missing image stream, "
        "which you would only discover at training time. Fix or detach it in the robot's "
        "camera settings, or pass ignore_dead_cameras to record without it anyway."
    )


def missing_cameras(
    configured: Mapping[str, Any] | None,
    present_indices: Iterable[int] | None,
) -> list[dict[str, Any]]:
    """The configured cameras whose INDEX this machine no longer lists at all. The frame-age rail above
    cannot see the worst case it was built for.
    """
    if not isinstance(configured, Mapping):
        return []
    present = {int(i) for i in present_indices or () if isinstance(i, (int, float))}
    if not present:
        return []
    out: list[dict[str, Any]] = []
    for name, cfg in configured.items():
        index = cfg.get("index_or_path") if isinstance(cfg, Mapping) else cfg
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if index not in present:
            out.append({"camera": str(name), "index": index})
    return out


def missing_refusal(missing: list[dict[str, Any]], *, peer_id: str) -> str:
    """Why a session with an unlisted camera index is refused."""
    which = ", ".join(f"{m['camera']} (index {m['index']})" for m in missing)
    plural = "cameras are" if len(missing) > 1 else "camera is"
    return (
        f"{peer_id}: {len(missing)} configured {plural} not listed by this machine at all - {which}. "
        "Nothing is there to open, so the episodes would carry no image stream for it. Replug it "
        "(a direct port beats a hub chain) and RESCAN before recording: removing a camera renumbers "
        "the rest, so the same index may now be a different camera and the dataset would record the "
        "wrong view. Pass ignore_missing_cameras to record without it anyway."
    )


def _norm(name: Any) -> str:
    """Compare device names the way a human would read them, not byte for byte."""
    return " ".join(str(name or "").split()).casefold()


def identity_drift(
    configured: Mapping[str, Any] | None,
    roster: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Configured cameras whose index is now answered by a DIFFERENT device."""
    if not isinstance(configured, Mapping):
        return []
    by_index: dict[int, str] = {}
    for entry in roster or ():
        if not isinstance(entry, Mapping):
            continue
        index, name = entry.get("listing_index"), entry.get("name")
        if isinstance(index, int) and not isinstance(index, bool) and str(name or "").strip():
            by_index[int(index)] = str(name)
    if not by_index:
        return []

    out: list[dict[str, Any]] = []
    for cam, cfg in configured.items():
        if not isinstance(cfg, Mapping):
            continue
        remembered = cfg.get("device_name")
        index = cfg.get("index_or_path")
        if not str(remembered or "").strip() or isinstance(index, bool) or not isinstance(index, int):
            continue
        now_name = by_index.get(int(index))
        if now_name is None or _norm(now_name) == _norm(remembered):
            continue
        elsewhere = [i for i, n in by_index.items() if _norm(n) == _norm(remembered)]
        item: dict[str, Any] = {
            "camera": str(cam),
            "index": int(index),
            "remembered": str(remembered),
            "now": str(now_name),
        }
        if len(elsewhere) == 1:
            item["moved_to"] = elsewhere[0]
        elif len(elsewhere) > 1:
            # Two devices answer to that name; picking one would be a coin flip dressed as advice.
            item["ambiguous"] = True
        out.append(item)
    return out


def drift_refusal(drift: list[dict[str, Any]], *, peer_id: str) -> str:
    """Why a session whose camera index changed hands is refused, and how to fix it in one step."""
    parts = []
    for d in drift:
        line = f"{d['camera']} index {d['index']} was {d['remembered']}, now {d['now']}"
        if isinstance(d.get("moved_to"), int):
            line += f" ({d['remembered']} is at index {d['moved_to']} now)"
        elif d.get("ambiguous"):
            line += f" (more than one camera is called {d['remembered']}, so its new index is a guess)"
        parts.append(line)
    plural = "cameras" if len(drift) > 1 else "camera"
    return (
        f"{peer_id}: {len(drift)} configured {plural} changed hands - {'; '.join(parts)}. "
        "Removing a camera renumbers the rest, so this index still opens and still streams, with the "
        "wrong view: the episodes would look perfectly healthy and be unusable. Point the camera at "
        "its new index in the robot's camera settings, or pass ignore_camera_identity to record with "
        "the index as it stands."
    )


# : Keys the dashboard adds to a camera config for its OWN memory, which no camera driver
# declares. : ``device_name`` is the roster name an index carried when it was configured (see
# identity_drift).
ANNOTATION_KEYS: tuple[str, ...] = ("device_name",)


def stamp_device_names(
    cameras: Mapping[str, Any] | None,
    roster: Iterable[Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    """Remember WHICH device each numeric camera index was, at the moment it is configured."""
    if not isinstance(cameras, Mapping) or not cameras:
        return cameras
    by_index: dict[int, str] = {}
    for entry in roster or ():
        if not isinstance(entry, Mapping):
            continue
        index, name = entry.get("listing_index"), entry.get("name")
        if isinstance(index, int) and not isinstance(index, bool) and str(name or "").strip():
            by_index[int(index)] = str(name)
    if not by_index:
        return cameras

    out: dict[str, Any] = dict(cameras)
    changed = False
    for cam, cfg in cameras.items():
        if not isinstance(cfg, Mapping):
            continue
        if str(cfg.get("device_name") or "").strip():
            continue  # already remembered - see the rule above
        index = cfg.get("index_or_path")
        # A path-configured camera is not judged by index, so a name buys it nothing; a bool is
        # not an index (True == 1 would stamp whatever sits at index 1 onto it).
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        name = by_index.get(int(index))
        if not name:
            continue  # the roster does not list it: there is nothing to remember
        stamped = dict(cfg)
        stamped["device_name"] = name
        out[cam] = stamped
        changed = True
    return out if changed else cameras


def without_annotations(cameras: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The same camera config with the dashboard's own bookkeeping keys removed. MUST be used on every
    path that hands a camera config to a robot process.
    """
    if not isinstance(cameras, Mapping) or not cameras:
        return cameras
    if not any(isinstance(cfg, Mapping) and any(k in cfg for k in ANNOTATION_KEYS) for cfg in cameras.values()):
        return cameras
    out: dict[str, Any] = {}
    for cam, cfg in cameras.items():
        if isinstance(cfg, Mapping) and any(k in cfg for k in ANNOTATION_KEYS):
            out[cam] = {k: v for k, v in cfg.items() if k not in ANNOTATION_KEYS}
        else:
            out[cam] = cfg
    return out
