"""``/api/training``, ``/api/checkpoints``, ``/api/policies``, ``/api/deploy`` and ``/api/calibration``.

The train side of the operator workflow: pick a dataset and a trainer, grade
and submit a run, watch it, export it, find the checkpoint again, check that it
fits a robot, get a script that deploys it, and calibrate an arm.

Every path a client names (an output dir, a dataset root, a checkpoint path) is
contained to its home in :mod:`strands_robots.dashboard.training` before the
filesystem is consulted, and refused with 400 in the same words whether or not
anything exists there. Every route requires a session. The calibration wizard
is the one route here that reaches hardware: it disables torque on a real arm,
so it runs only with the operator's confirm or a standing motion grant, and a
refusal carries the consent card the browser knows how to draw.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

from strands_robots.dashboard import access, agent_motion, checkpoints, consent, deploy, policy_fit, settings, training
from strands_robots.dashboard import calibration as calib
from strands_robots.dashboard import calibration_run as cr
from strands_robots.dashboard.training import PathOutside

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["train"])

__all__ = ["router", "attach"]


def _refuse_path(exc: PathOutside) -> HTTPException:
    """The 400 for a path outside its home: names the home, never the path's existence."""
    return HTTPException(400, exc.refusal())


def _record_activity(request: Request, *args: Any, **kwargs: Any) -> None:
    """Note an operator action on the mesh bridge's activity log when a bridge is mounted."""
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None:
        return
    try:
        bridge.record_activity(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the ledger is a courtesy, the action already happened
        logger.debug("activity not recorded: %s", exc)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


@router.get("/training/trainers")
async def training_trainers(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """The trainers the form can submit to, the ones it cannot and why, and the spec fields."""
    # `trainers` keeps its shape (a list of names): an older cached bundle renders that list
    # directly, and changing it to objects would crash those tabs.
    return {
        "trainers": await asyncio.to_thread(training.list_trainers),
        "unsupported": await asyncio.to_thread(training.form_unsupported),
        "fields": list(training.SPEC_KEYS),
        "output_home": str(training.output_home()),
    }


@router.get("/training/datasets")
async def training_datasets(
    request: Request,
    q: str = "",
    hub: bool = True,
    limit: int = 12,
    _: dict = Depends(access.require_session),
) -> dict[str, Any]:
    """Datasets for the submit form's picker: local roots plus a Hub search."""
    from strands_robots.dashboard.dataset_check import mark_live_recording

    active, captured = None, None
    try:
        session = getattr(request.app.state, "record", None)
        live = session.session() if session is not None else {}
        active = live.get("dataset") or None
        episodes = live.get("episodes")
        captured = len(episodes) if isinstance(episodes, list) else None
    except Exception:  # noqa: BLE001 - the picker is worth more than this annotation
        active, captured = None, None

    if not hub:
        rows = await asyncio.to_thread(training.local_datasets, q)
        return {"datasets": mark_live_recording(rows, active, episodes_so_far=captured)}
    found = await asyncio.to_thread(training.search_datasets, q, checkpoints.clamp_limit(limit, 12, 50))
    try:
        found["hf_auth"] = await asyncio.to_thread(checkpoints.hf_auth_state)
    except Exception as exc:  # noqa: BLE001 - never let an auth probe break search
        found["hf_auth"] = {
            "authenticated": False,
            "user": None,
            "detail": f"auth state unavailable ({type(exc).__name__})",
        }
    return {**found, "datasets": mark_live_recording(found.get("datasets", []), active, episodes_so_far=captured)}


@router.get("/training/jobs")
async def training_jobs(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Runs submitted from this dashboard, and whether the ledger they come from is whole."""
    rows = await asyncio.to_thread(training.jobs)
    # `problem` is about the LEDGER, not the runs: an unreadable history and a dashboard
    # that has never trained anything both produce an empty list, and only one of them
    # means runs were forgotten.
    return {"jobs": rows, "problem": training.jobs_problem()}


def _graded(body: dict[str, Any]) -> dict[str, Any]:
    """The body, or the 422 that marks each field the form got wrong."""
    problems = training.spec_problems(body)
    if problems:
        raise HTTPException(
            422,
            {
                "message": "the training spec has " + ("a problem" if len(problems) == 1 else "problems"),
                "fields": problems,
            },
        )
    return body


@router.post("/training/validate")
async def training_validate(body: dict[str, Any], _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Grade the spec field by field, then dry-run it through the trainer. Launches nothing."""
    return await asyncio.to_thread(training.validate, _graded(body))


@router.get("/training/output-dir")
async def training_output_dir(path: str = "", _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """What is already at an output dir and whether training there needs a confirm."""
    if not path.strip():
        raise HTTPException(422, "path is required")
    try:
        target = training.contain_output_dir(path)
    except PathOutside as exc:
        raise _refuse_path(exc) from exc
    return await asyncio.to_thread(training.output_dir_verdict, str(target))


@router.post("/training/submit")
async def training_submit(
    body: dict[str, Any], request: Request, _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Launch a training job. The spec is graded here, then by the trainer's own validate()."""
    spec = dict(body)
    confirm_clear = spec.pop("confirm_clear", False)
    _graded(spec)
    spec["confirm_clear"] = confirm_clear
    result = await asyncio.to_thread(training.submit, spec)
    job = (result.get("data") or {}).get("job_id") if isinstance(result, dict) else None
    _record_activity(
        request,
        "training",
        "submit",
        target=str(job or body.get("provider", "?")),
        detail=f"{body.get('provider')} on {body.get('dataset_root') or body.get('dataset_repo_id') or '?'}",
        ok=bool(isinstance(result, dict) and result.get("status") == "success"),
    )
    return result


@router.get("/training/status")
async def training_status(provider: str, job_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """The trainer's verdict on one job: running, learning, or neither."""
    return await asyncio.to_thread(training.status, provider, job_id)


@router.post("/training/export")
async def training_export(body: dict[str, Any], _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Export the latest checkpoint as a loadable policy artifact, then check it on disk."""
    try:
        output_dir = str(training.contain_output_dir(str(body.get("output_dir") or "")))
        dataset_root = str(body.get("dataset_root") or "")
        if dataset_root:
            dataset_root = str(training.contain_dataset_root(dataset_root))
    except PathOutside as exc:
        raise _refuse_path(exc) from exc
    return await asyncio.to_thread(
        training.export,
        body.get("provider", "lerobot_local"),
        output_dir,
        dataset_root,
        body.get("dataset_repo_id"),
        body.get("base_model", ""),
    )


# ---------------------------------------------------------------------------
# checkpoints + policies
# ---------------------------------------------------------------------------


@router.get("/checkpoints/search")
async def checkpoints_search(q: str = "", limit: int = 15, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Type-ahead checkpoint search: trained here, the local HF cache, then the Hub."""
    return await asyncio.to_thread(checkpoints.search, q, checkpoints.clamp_limit(limit))


@router.get("/checkpoints/features")
async def checkpoint_features(repo_id: str = "", _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """The features a checkpoint declares, by repo id or by a contained local path."""
    try:
        repo_id = training.contain_checkpoint_ref(repo_id)
    except PathOutside as exc:
        raise _refuse_path(exc) from exc
    return await asyncio.to_thread(checkpoints.declared_features, repo_id)


@router.get("/checkpoints/families")
async def checkpoint_families(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """policy_type values the lerobot family accepts (the type dropdown)."""
    return {"families": await asyncio.to_thread(checkpoints.policy_families)}


@router.get("/policies")
async def policies(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Full provider catalog - the schema the run form is generated from."""
    catalog = await asyncio.to_thread(policy_fit.policy_catalog)
    if not catalog:
        raise HTTPException(500, "policy registry unavailable")
    return {"providers": catalog, "names": [p["name"] for p in catalog]}


@router.post("/policies/validate")
async def validate_policy(
    body: dict[str, Any], request: Request, _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Dry-run a provider config against a peer's joints and cameras without touching a robot."""
    provider = (body.get("policy_provider") or "").strip()
    if not provider:
        raise HTTPException(422, "policy_provider required")
    config = body.get("policy_config") or {}
    if not isinstance(config, dict):
        raise HTTPException(422, "policy_config must be an object")

    # Preflight's most useful check compares the model's declared image inputs against the
    # observation keys it will actually receive, so pass the target peer's real joints and
    # cameras when the mesh bridge knows them.
    peer_id = body.get("peer_id") or ""
    bridge = getattr(request.app.state, "bridge", None)
    peer = (getattr(bridge, "peers", None) or {}).get(peer_id) or {}
    observation_keys = set((peer.get("state") or {}).get("joints") or {})
    observation_keys |= set(peer.get("cameras") or {})

    def _check() -> dict[str, Any]:
        from strands_robots.policies import policy_provider_error, preflight_policy

        problem = policy_provider_error(provider, **config)
        if problem:
            return {"ok": False, "stage": "provider", "error": problem}
        try:
            from strands_robots.policies.factory import _check_trust_remote_code

            _check_trust_remote_code(provider)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 - the trust gate, verbatim
            return consent.attach_consent({"ok": False, "stage": "trust", "error": str(exc)}, exc, subject=provider)
        try:
            # Returns None and raises ValueError on a bad config; a provider without a
            # preflight hook is a silent pass.
            preflight_policy(provider, observation_keys, **config)
        except Exception as exc:  # noqa: BLE001 - surfacing is the point
            return consent.attach_consent(
                {"ok": False, "stage": "preflight", "error": f"{type(exc).__name__}: {exc}"}, exc, subject=provider
            )
        return {"ok": True, "stage": "preflight"}

    result = await asyncio.to_thread(_check)
    result["policy_provider"] = provider
    result["observation_keys"] = sorted(observation_keys)
    return result


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------


@router.post("/deploy/snippet")
async def deploy_snippet(
    body: dict[str, Any], request: Request, _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Render a spawn payload or a remembered device profile as a deployable script."""
    payload = body.get("payload")
    serial = body.get("serial")
    if serial and not payload:
        devices = getattr(request.app.state, "devices", None)
        profiles = getattr(devices, "profiles", None)
        if profiles is None:
            raise HTTPException(404, "no device manager is running here, so no profile is remembered")
        payload = profiles.get(str(serial))
        if payload is None:
            raise HTTPException(404, f"no profile remembered for {serial!r}")
    if not isinstance(payload, dict):
        raise HTTPException(422, "payload object or serial required")
    hub_host = body.get("hub_host")
    hub_note = None
    if hub_host is None:
        hub_host, hub_note = deploy.hub_host_from_reached(request.url.hostname)
    result = deploy.render_snippet(
        payload,
        hub_host=hub_host or None,
        hub_note=hub_note,
        mesh_env=os.environ,
        hub_port=settings.get("mesh", "port", deploy.DEFAULT_HUB_PORT),
    )
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


@router.get("/calibration")
async def calibration_list(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Every calibration on this machine, as the drawer's ``{status, text}`` plus rows."""
    return await asyncio.to_thread(calib.listing)


@router.get("/calibration/{name}")
async def calibration_view(
    name: str,
    device_type: str | None = None,
    device_model: str | None = None,
    _: dict = Depends(access.require_session),
) -> dict[str, Any]:
    """One calibration's per-motor detail."""
    found = await asyncio.to_thread(calib.candidates, name, device_type=device_type, device_model=device_model)
    if not found:
        raise HTTPException(
            404,
            {
                "error": f"no calibration named {name!r}",
                "hint": "GET /api/calibration lists every calibration on this machine",
            },
        )
    if len(found) > 1:
        raise HTTPException(
            409,
            {
                "error": f"{name!r} exists {len(found)} times - say which with ?device_type=&device_model=",
                "candidates": found,
            },
        )
    target = found[0]
    info = await asyncio.to_thread(
        calib.calibration_info, target["device_type"], target["device_model"], target["device_id"]
    )
    if info is None:
        return {"status": "error", "text": f"{target['path']} could not be read", **target}
    out: dict[str, Any] = {
        "status": "success",
        "text": f"**Calibration {target['device_id']}** ({target['device_type']}/{target['device_model']})",
    }
    out.update(calib.payload(info))
    return out


def _calibration_gate(request: Request, payload: dict[str, Any], port: str, device_id: str) -> None:
    """Refuse a wizard that would collide with a running arm or start without the operator's word.

    The wizard is the one route here that commands hardware: ``lerobot-calibrate``
    disables torque, and a limp arm drops. So it runs only with ``confirm: true``
    from the wizard's own confirm sheet, or under the standing motion grant an
    operator can give the dashboard. The refusal carries the consent card so the
    browser can ask instead of showing a wall.
    """
    devices = getattr(request.app.state, "devices", None)
    owner = devices.port_owner(port) if devices is not None and hasattr(devices, "port_owner") else None
    if owner:
        raise HTTPException(
            409,
            {
                "error": f"{port} is held by the running robot {owner!r} - two owners on one servo bus is "
                "the 'Port is in use!' collision, and the wizard would measure a bus that is mid-conversation",
                "remedy": f"despawn {owner} first (its profile is remembered, respawn after)",
            },
        )
    confirmed = payload.get(cr.CONFIRM_KEY) is True
    verdict = agent_motion.agent_motion_allowed("task", peer=None, target=device_id or port)
    if confirmed or verdict.get("granted"):
        return
    shown = device_id or port or "this arm"
    refusal = {
        "error": (
            f"refused: calibrating {shown} disables torque on a REAL arm over {port}, and this dashboard does "
            f"not start that without the operator. Nothing was sent. Confirm it from the wizard's sheet "
            f"(send confirm: true), or grant unattended motion once."
        ),
        "code": "calibration_confirm_required",
        "remedy": "POST again with confirm: true after reading the wizard's confirm sheet",
    }
    # The card's kind is the standing motion grant (consent kinds are closed); its message is
    # this refusal, so the operator reads about the arm on the desk, not about the agent.
    raise HTTPException(403, consent.attach_consent(refusal, {**verdict, "reason": refusal["error"]}, subject=shown))


@router.post("/calibration/run")
async def calibration_run_start(
    payload: dict[str, Any], request: Request, _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Start the lerobot-calibrate wizard on one arm, under a pty the dashboard walks step by step."""
    port = str(payload.get("port") or "").strip()
    device_id = str(payload.get("device_id") or "").strip()
    _calibration_gate(request, payload, port, device_id)
    try:
        run = cr.start(
            role=str(payload.get("role") or ""),
            model=str(payload.get("model") or ""),
            device_id=device_id,
            port=port,
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    except ImportError as e:
        raise HTTPException(501, {"error": str(e), "extra": "lerobot"}) from e
    _record_activity(
        request, "calibration", "start", target=device_id or port, detail=f"{payload.get('role')} on {port}"
    )
    return run.status()


def _run_or_404(sid: str) -> cr.CalibrationRun:
    run = cr.get(sid)
    if run is None:
        raise HTTPException(
            404,
            f"no calibration session {sid!r} - it may have been superseded; start a new one from the devices drawer",
        )
    return run


@router.get("/calibration/run/{sid}")
async def calibration_run_status(sid: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Where the wizard is right now."""
    return _run_or_404(sid).status()


@router.post("/calibration/run/{sid}/key")
async def calibration_run_key(
    sid: str, payload: dict[str, Any], _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Answer the wizard's current prompt: 'enter' continues, 'c' redoes an existing calibration."""
    run = _run_or_404(sid)
    try:
        run.press(str(payload.get("key") or ""))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(409, str(e)) from e
    return run.status()


@router.post("/calibration/run/{sid}/cancel")
async def calibration_run_cancel(sid: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Stop the wizard; the arm stays as it is (torque was already off)."""
    run = _run_or_404(sid)
    await asyncio.to_thread(run.cancel)
    return run.status()


# ---------------------------------------------------------------------------
# lifespan
# ---------------------------------------------------------------------------


async def _close_calibration_runs() -> None:
    """No wizard outlives the dashboard: a pty with nobody reading it is a hung arm."""
    for run in list(cr.runs.values()):
        try:
            await asyncio.to_thread(run.close)
        except Exception as exc:  # noqa: BLE001 - shutdown keeps going
            logger.debug("calibration run %s not closed: %s", run.id, exc)
    cr.runs.clear()


def attach(app: FastAPI) -> None:
    """Register this lane's lifespan work: close live calibration wizards on shutdown."""
    hooks = getattr(app.state, "shutdown_hooks", None)
    if hooks is None:
        hooks = []
        app.state.shutdown_hooks = hooks
    hooks.append(_close_calibration_runs)
