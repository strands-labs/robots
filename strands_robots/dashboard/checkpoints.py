"""Policy checkpoint discovery for the run form.

Three sources, merged and ranked: checkpoints THIS dashboard trained (found
through the training jobs ledger), the local Hugging Face cache, and a Hub
search of public LeRobot checkpoints ranked by downloads. A policy the
dashboard itself trained must be findable in the dashboard's own picker.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

from strands_robots.dashboard.ttl_cache import TTLCache

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 300.0
#: Bounded and self-pruning: a dict keyed by query grows one entry per keystroke.
_CACHE: TTLCache[list[dict[str, Any]]] = TTLCache(_CACHE_TTL_S)

#: lerobot policy family names that can appear in checkpoint tags/names.
#: FALLBACK ONLY: the live list is read from lerobot's own choice registry
#: (see _family_matcher), because a hand list keyed to an upstream registry
#: is only as true as the day it was written.
_FALLBACK_FAMILIES = (
    "smolvla",
    "act",
    "diffusion",
    "pi0_fast",
    "pi05",
    "pi0",
    "tdmpc",
    "vqbet",
    "groot",
    "molmoact2",
    "molmoact",
    "sac",
    "xvla",
    "wall_x",
    "eo1",
    "evo1",
)


def _build_family_matcher(names: tuple[str, ...]) -> tuple[re.Pattern[str], dict[str, str]]:
    """Compile a separator-tolerant matcher + squashed-name → canonical map.

    Longest-first so ``pi0_fast`` wins over ``pi0``; each ``_`` in a family
    name becomes ``-?`` because callers normalize ``_``→``-`` (word-boundary
    ``\\b`` misses ``smolvla_base`` otherwise) and Hub names also write
    ``pi0fast`` with no separator at all.
    """
    alts = sorted(names, key=len, reverse=True)
    pattern = "|".join(re.escape(n).replace("_", "-?") for n in alts)
    regex = re.compile(rf"\b({pattern})\b", re.IGNORECASE)
    canonical = {n.replace("_", "").lower(): n for n in names}
    return regex, canonical


_FAMILY_MATCHER: tuple[re.Pattern[str], dict[str, str]] | None = None


def _family_matcher() -> tuple[re.Pattern[str], dict[str, str]]:
    """The registry-derived matcher, built once; fallback list if lerobot is absent."""
    global _FAMILY_MATCHER
    if _FAMILY_MATCHER is None:
        names: tuple[str, ...] = ()
        try:
            import lerobot.policies  # noqa: F401 - importing registers the config subclasses
            from lerobot.configs.policies import PreTrainedConfig

            names = tuple(PreTrainedConfig.get_known_choices())
        except Exception:  # noqa: BLE001 - checkpoint browsing must survive a broken lerobot
            names = ()
        _FAMILY_MATCHER = _build_family_matcher(names or _FALLBACK_FAMILIES)
    return _FAMILY_MATCHER


def _guess_policy_type(repo_id: str, tags: list[str]) -> str | None:
    """Best-effort policy family from repo name + tags (form prefill only)."""
    regex, canonical = _family_matcher()
    for source in (*tags, repo_id):
        # underscores are word chars, so \bsmolvla\b misses 'smolvla_base' -
        # normalize separators before matching.
        m = regex.search((source or "").replace("_", "-"))
        if m:
            squashed = m.group(1).lower().replace("-", "")
            return canonical.get(squashed, squashed)
    return None


def _hf_cache_root() -> Path:
    """The Hub cache root; :func:`training.hf_cache_root` owns the rule, this module reads it."""
    from strands_robots.dashboard.training import hf_cache_root

    return hf_cache_root()


def local_checkpoints(query: str = "") -> list[dict[str, Any]]:
    """Already-downloaded LeRobot-shaped checkpoints in the HF cache."""
    hub = _hf_cache_root()
    if not hub.is_dir():
        return []
    q = query.lower()
    out: list[dict[str, Any]] = []
    for entry in hub.iterdir():
        if not entry.name.startswith("models--"):
            continue
        repo_id = entry.name[len("models--") :].replace("--", "/", 1)
        if q and q not in repo_id.lower():
            continue
        snapshots = entry / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            snap = next(iter(sorted(snapshots.iterdir())), None)
        except OSError:
            continue
        if snap is None:
            continue
        cfg = snap / "config.json"
        train_cfg = snap / "train_config.json"
        if not (cfg.exists() or train_cfg.exists()):
            continue
        policy_type = None
        try:
            if cfg.exists():
                import json

                data = json.loads(cfg.read_text(encoding="utf-8"))
                # lerobot policy configs carry "type"; transformers configs
                # carry "model_type" - only the former is runnable here.
                policy_type = data.get("type")
                if policy_type is None and "model_type" in data:
                    continue  # transformers model, not a policy checkpoint
        except Exception:  # noqa: BLE001 - unreadable config -> still list it
            pass
        out.append(
            {
                "repo_id": repo_id,
                "local": True,
                "downloads": None,
                "policy_type": policy_type or _guess_policy_type(repo_id, []),
                "tags": [],
            }
        )
    return out


def hub_search(query: str, limit: int = 12) -> tuple[list[dict[str, Any]], str | None]:
    """Type-ahead search of public LeRobot checkpoints on the Hub. Returns ``(rows, problem)`` -
    ``problem`` is None when the Hub answered, otherwise a short human sentence.
    """
    key = f"{query}:{limit}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached, None
    try:
        from huggingface_hub import HfApi

        # newer huggingface_hub removed direction=; sort="downloads" alone
        # returns descending. Fall back to unsorted on any TypeError so a
        # hub-client version bump can't kill search.
        api = HfApi()
        try:
            models = api.list_models(filter="lerobot", search=query or None, sort="downloads", limit=limit)
        except TypeError:
            models = api.list_models(filter="lerobot", search=query or None, limit=limit)
        rows = [
            {
                "repo_id": m.id,
                "local": False,
                "downloads": getattr(m, "downloads", None),
                "policy_type": _guess_policy_type(m.id, list(m.tags or [])),
                "tags": [t for t in (m.tags or []) if not t.startswith(("region:", "license:", "arxiv:"))][:6],
            }
            for m in models
        ]
    except Exception as exc:  # noqa: BLE001 - hub outage degrades to local-only
        logger.warning("hub checkpoint search failed: %r", exc)
        # do NOT cache a failure - the next keystroke should retry
        kind = type(exc).__name__
        return [], f"Hub search unavailable ({kind}) - showing local cache only"
    _CACHE.put(key, rows)
    return rows, None


_WHOAMI: dict[str, Any] = {"at": 0.0, "value": None, "token": None}
_WHOAMI_TTL_S = 600.0
# : A REJECTED token is cached far more briefly than a good one.
_WHOAMI_REJECTED_TTL_S = 20.0


def _token_fingerprint(token: str | None) -> str | None:
    """A stable, non-secret id for a token, for cache keying only. The verdict is about a SPECIFIC
    token, so the cache has to be able to notice a different one.
    """
    if not token:
        return None
    return hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:16]


def whoami_cache_verdict(
    entry: dict[str, Any],
    fingerprint: str | None,
    now: float,
    *,
    ttl_s: float = _WHOAMI_TTL_S,
    rejected_ttl_s: float = _WHOAMI_REJECTED_TTL_S,
) -> dict[str, Any] | None:
    """The cached answer that is still true, or None to go and ask the Hub."""
    value = entry.get("value")
    if not isinstance(value, dict):
        return None
    if entry.get("token") is None or entry.get("token") != fingerprint:
        return None
    age = now - float(entry.get("at") or 0.0)
    budget = ttl_s if value.get("authenticated") else rejected_ttl_s
    return value if 0.0 <= age < budget else None


def hf_auth_state() -> dict[str, Any]:
    """Whether this machine can reach gated/private HF repos, and as whom."""
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:  # noqa: BLE001
        token = None
    if not token:
        return {"authenticated": False, "user": None, "detail": "no HF token on this machine"}
    fingerprint = _token_fingerprint(token)
    now = time.time()
    cached = whoami_cache_verdict(_WHOAMI, fingerprint, now)
    if cached is not None:
        return cached
    try:
        from huggingface_hub import HfApi

        user = HfApi().whoami(token=token).get("name")
        value = {"authenticated": True, "user": user, "detail": None}
    except Exception as exc:  # noqa: BLE001
        value = {
            "authenticated": False,
            "user": None,
            "detail": f"HF token present but rejected ({type(exc).__name__})",
        }
    _WHOAMI.update(at=now, value=value, token=fingerprint)
    return value


#: Widest page the hub search is asked for, and the ceiling for any caller.
MAX_LIMIT = 40


def clamp_limit(limit: Any, default: int = 15, ceiling: int = MAX_LIMIT) -> int:
    """A limit is a promise to the caller: 1 means one row."""
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, ceiling))


def _artifact_dir(output_dir: Path) -> Path | None:
    """The loadable artifact inside one training run's output_dir, or None."""
    if (output_dir / "config.json").exists() or (output_dir / "train_config.json").exists():
        return output_dir
    ckpts = output_dir / "checkpoints"
    if not ckpts.is_dir():
        return None
    candidates = [ckpts / "last"]

    def _step(d: Path) -> tuple[int, str]:
        # LeRobot zero-pads step dirs so name-sort works, but a bare "900" vs
        # "1000" must still order numerically.
        try:
            return (int(d.name), d.name)
        except ValueError:
            return (-1, d.name)

    try:
        candidates += sorted(
            (d for d in ckpts.iterdir() if d.name != "last"),
            key=_step,
            reverse=True,
        )
    except OSError:
        return None
    for cand in candidates:
        if cand.name == "last":
            # lerobot's own pointer - but resolve it: a picker row must name
            # the concrete weights, not a path whose contents silently change
            # when training resumes.
            try:
                cand = cand.resolve(strict=True)
            except OSError:
                continue
        pm = cand / "pretrained_model"
        if (pm / "config.json").exists() or (pm / "train_config.json").exists():
            return pm
    return None


def trained_checkpoints(query: str = "") -> list[dict[str, Any]]:
    """Checkpoints THIS dashboard trained, discovered via the jobs ledger."""
    from strands_robots.dashboard import training

    q = query.lower()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in reversed(training.jobs()):  # newest training first
        raw = (job.get("output_dir") or "").strip()
        if not raw:
            continue
        try:
            artifact = _artifact_dir(Path(raw).expanduser())
        except OSError:
            continue
        if artifact is None:
            continue  # still training, failed, or cleaned up - not loadable, not listed
        path = str(artifact)
        if path in seen:
            continue
        label = f"{path} {job.get('base_model') or ''} {job.get('dataset') or ''}"
        if q and q not in label.lower():
            continue
        seen.add(path)
        policy_type = None
        try:
            import json as _json

            cfg = artifact / "config.json"
            if cfg.exists():
                policy_type = _json.loads(cfg.read_text(encoding="utf-8")).get("type")
        except Exception:  # noqa: BLE001 - unreadable config -> still list it
            pass
        out.append(
            {
                "repo_id": path,
                "local": True,
                "source": "trained",
                "downloads": None,
                "policy_type": policy_type or _guess_policy_type(str(job.get("base_model") or ""), []),
                "tags": [],
                "job_id": job.get("job_id"),
                "dataset": job.get("dataset"),
                "trained_at": job.get("submitted_at"),
            }
        )
    return out


def search(query: str = "", limit: int = 15) -> dict[str, Any]:
    """Merged checkpoint search: trained here, then local cache, then hub."""
    limit = clamp_limit(limit)
    trained = trained_checkpoints(query)
    local = local_checkpoints(query)
    local_ids = {r["repo_id"] for r in trained} | {r["repo_id"] for r in local}
    remote_rows, hub_problem = hub_search(query, limit=limit)
    remote = [r for r in remote_rows if r["repo_id"] not in local_ids]
    rows = trained + local + remote
    return {
        "query": query,
        "results": rows[:limit],
        "total_matched": len(rows),
        "hub_problem": hub_problem,
        "hf_auth": hf_auth_state(),
    }


def policy_families() -> list[str]:
    """``policy_type`` values lerobot_local accepts (for the type dropdown)."""
    try:
        from strands_robots.policies.lerobot_local.resolution import list_policy_types

        return list(list_policy_types())
    except Exception:  # noqa: BLE001 - torch-less install
        return ["act", "diffusion", "pi0", "pi0_fast", "smolvla", "tdmpc", "vqbet"]


def declared_features(repo_id: str) -> dict[str, Any]:
    """The input/output features a checkpoint declares, read from its config on disk.

    ``repo_id`` is either a local training output / exported artifact path or an
    ``org/name`` id whose snapshot is in the HF cache. ``{}`` when neither holds a
    readable config: no evidence is not a verdict.
    """
    import json

    raw = (repo_id or "").strip()
    # Trailing/leading slashes are noise in a REPO ID - but a leading slash is the whole meaning of
    # an absolute output_dir, so the two readings keep their own strings.
    name = raw.strip("/")
    if not name:
        return {}
    candidates: list[Path] = []

    # 1. a local training output / exported artifact path
    try:
        p = Path(raw).expanduser()
        if p.is_dir():
            art = _artifact_dir(p)
            if art is not None:
                candidates.append(art)
    except OSError:
        pass  # an unreadable output dir contributes no candidates; the ledger still lists the job

    # 2. the HF cache snapshot for an org/name repo id
    try:
        root = _hf_cache_root() / f"models--{name.replace('/', '--')}" / "snapshots"
        if root.is_dir():
            snaps = sorted(root.iterdir())
            if snaps:
                candidates.append(snaps[-1])
    except OSError:
        pass  # an unreadable Hub cache contributes no candidates

    for d in candidates:
        for fname in ("config.json", "train_config.json"):
            f = d / fname
            try:
                if not f.exists():
                    continue
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - unreadable config = no evidence
                continue
            if not isinstance(data, dict):
                continue
            # train_config.json nests the policy config under "policy".
            pol = data.get("policy")
            block = pol if isinstance(pol, dict) else data
            inp = block.get("input_features")
            out = block.get("output_features")
            if isinstance(inp, dict) or isinstance(out, dict):
                return {
                    "repo_id": raw,
                    "input_features": inp if isinstance(inp, dict) else {},
                    "output_features": out if isinstance(out, dict) else {},
                    "policy_type": block.get("type"),
                    "norm_tags": _declared_norm_tags(d),
                }
    return {}


def _declared_norm_tags(d: Path) -> list[str]:
    """The normalisation tags ``norm_stats.json`` declares in ``d``, or [] when unknowable."""
    import json

    f = d / "norm_stats.json"
    try:
        if not f.exists():
            return []
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - unreadable stats = no evidence, same as an absent config
        return []
    if not isinstance(data, dict):
        return []
    return sorted(k for k in data if isinstance(k, str))
