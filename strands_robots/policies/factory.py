"""Policy factory - create_policy() and runtime registration."""

import logging
import os
from collections.abc import Callable, Mapping

from strands_robots.policies.base import Policy
from strands_robots.registry import (
    import_policy_class,
    list_policy_aliases,
    list_policy_providers,
    resolve_policy,
)

# The one canonicalisation rule, shared rather than restated: a decision keyed
# on a provider name has to resolve the caller's spelling first, and a second
# copy of that rule here is a second thing to keep in step with policies.json.
from strands_robots.registry.policies import _canonical_provider_name

logger = logging.getLogger(__name__)

#
# Runtime registration (for user-defined providers not in JSON)
#

_runtime_registry: dict[str, Callable[[], type[Policy]]] = {}
_runtime_aliases: dict[str, str] = {}


def register_policy(
    name: str,
    loader: Callable[[], type[Policy]],
    aliases: list[str] | None = None,
):
    """Register a custom policy provider at runtime.

    Use this to add providers without editing policies.json.

    Example::

        from strands_robots.policies import register_policy

        register_policy("my_provider", lambda: MyPolicy, aliases=["my"])
        policy = create_policy("my_provider", ...)
    """
    _runtime_registry[name] = loader
    if aliases:
        for alias in aliases:
            _runtime_aliases[alias] = name


def list_providers() -> list[str]:
    """List all available policy provider names (JSON + runtime)."""
    names = list_policy_providers()
    names.extend(_runtime_registry.keys())
    names.extend(_runtime_aliases.keys())
    return sorted(set(names))


def list_aliases() -> dict[str, str]:
    """Return every provider alias and the canonical name it resolves to.

    :func:`create_policy` accepts a provider's declared aliases and
    shorthands as readily as its canonical name, but
    :func:`list_providers` reports the canonical names from the JSON
    registry. Together the two surfaces enumerate every spelling the
    registries hold::

        registered = set(list_providers()) | set(list_aliases())

    That is every *registered* spelling, not every spelling
    :func:`create_policy` resolves.
    :func:`~strands_robots.registry.policies.import_policy_class` falls back
    to auto-discovery, so a module under ``strands_robots.policies`` that
    exports a :class:`~strands_robots.policies.base.Policy` subclass resolves
    under its own module name with no registry entry. Two ship, and neither is
    a registry provider because each wraps a policy the caller already holds
    rather than building one from config:

    * ``composite``
      (:class:`~strands_robots.policies.composite.CompositePolicy`) builds
      through this factory -- ``create_policy("composite", lower=..., upper=...)``
      -- and is the one spelling ``registered`` above omits.
    * ``persistent``
      (:class:`~strands_robots.policies.persistent.PersistentPolicy`) resolves
      but cannot be built here: its first parameter is named ``provider``,
      which :func:`create_policy` has already bound, so it is constructed
      directly.

    Covers both registries, matching the union :func:`list_providers`
    reports: aliases declared in ``policies.json`` and aliases passed to
    :func:`register_policy` at runtime. A runtime alias shadows a JSON
    alias of the same name, which is the precedence
    :func:`create_policy` applies.

    Returns:
        Mapping of alias to the canonical provider name it resolves to.
    """
    return {**list_policy_aliases(), **_runtime_aliases}


class UntrustedRemoteCodeError(RuntimeError):
    """Raised when a HF model requires trust_remote_code but the user has not opted in."""


# Providers whose HuggingFace model loading path calls ``trust_remote_code=True``.
# Any provider that downloads and executes code from a model repository
# **must** be listed here so users are forced to explicitly opt in.
_HF_REMOTE_CODE_PROVIDERS: frozenset[str] = frozenset(
    {
        "lerobot_local",
        "kimodo",
    }
)


def _check_trust_remote_code(provider: str) -> None:
    """Enforce the trust-remote-code gate for HuggingFace-backed providers.

    Only providers listed in ``_HF_REMOTE_CODE_PROVIDERS`` are gated.
    These providers load models with ``trust_remote_code=True``, which
    allows **arbitrary code execution** from the model repository.

    Set the environment variable ``STRANDS_TRUST_REMOTE_CODE=1`` to opt in.
    """
    if provider not in _HF_REMOTE_CODE_PROVIDERS:
        return

    opted_in = os.environ.get("STRANDS_TRUST_REMOTE_CODE", "").strip()
    if opted_in in ("1", "true", "yes"):
        return

    raise UntrustedRemoteCodeError(
        f"Policy provider '{provider}' loads HuggingFace models with "
        f"trust_remote_code=True, which allows arbitrary code execution "
        f"from the model repository.\n\n"
        f"Only load models from organisations you trust.\n\n"
        f"To acknowledge this risk and proceed, set the environment variable:\n"
        f"    export STRANDS_TRUST_REMOTE_CODE=1\n"
    )


def _resolve_policy_class(provider: str, **kwargs) -> tuple[str, type[Policy], dict]:
    """Resolve ``provider`` to its policy class WITHOUT instantiating it.

    Imports the class and computes the effective constructor kwargs using the
    same three-stage lookup as :func:`create_policy` (runtime registry, smart
    string, then ``policies.json``), but never calls the constructor and never
    enforces the trust-remote-code gate. This lets callers inspect or run a
    class-level :meth:`Policy.preflight` check before paying the cost (and,
    for remote-code providers, the risk) of construction.

    Args:
        provider: Provider name, HF model ID, or server URL.
        **kwargs: Provider-specific parameters.

    Returns:
        ``(canonical_provider_name, PolicyClass, resolved_kwargs)``.

    Raises:
        ImportError / ValueError: Propagated from the underlying class import
            or smart-string resolution when the provider cannot be resolved.
    """
    # 1. Runtime registry (user-registered providers).
    resolved_name = _runtime_aliases.get(provider, provider)
    if resolved_name in _runtime_registry:
        return resolved_name, _runtime_registry[resolved_name](), dict(kwargs)

    # 2. Smart string (HF ID, URL, etc.).
    _needs_resolution = (
        "/" in provider
        or (":" in provider and not provider.replace("_", "").isalpha())
        or provider.startswith("ws://")
        or provider.startswith("grpc://")
        or provider.startswith("zmq://")
    )
    if _needs_resolution:
        try:
            resolved_provider, resolved_kwargs = resolve_policy(provider, **kwargs)
        except ImportError:
            resolved_provider = None
            resolved_kwargs = {}
        except Exception as e:
            logger.warning("Policy resolution failed for '%s': %s", provider, e)
            resolved_provider = None
            resolved_kwargs = {}
        if resolved_provider:
            return resolved_provider, import_policy_class(resolved_provider), dict(resolved_kwargs)

    # 3. Standard lookup from policies.json. The name returned is the canonical
    #    one, not the caller's spelling: create_policy keys the
    #    trust-remote-code gate on it and that gate membership-tests a set of
    #    canonical names, so returning a declared alias would skip the gate for
    #    every spelling but one. Stages 1 and 2 already canonicalise (the
    #    runtime alias map, and resolve_policy's shorthand stage); this is the
    #    third.
    return _canonical_provider_name(provider), import_policy_class(provider), dict(kwargs)


# ``policy_config`` (and the per-call ``policy_kwargs``) are opaque provider
# keyword bags: callers hand them to ``create_policy`` / ``get_actions``, which
# splat them with ``**``. A non-mapping value therefore fails inside CPython's
# call machinery with a bare ``TypeError`` naming this module's internals, which
# tells the caller nothing about which parameter to fix. Callers validate the
# value against this helper first and wrap the message in their own error
# envelope, mirroring ``VideoConfig.validation_error``.
_POLICY_MAPPING_HINTS: dict[str, str] = {
    "policy_config": (
        "provider kwargs forwarded to create_policy, e.g. policy_config={'host': '127.0.0.1', 'port': 5555}"
    ),
    "policy_kwargs": ("per-call kwargs forwarded to policy.get_actions, e.g. policy_kwargs={'target_pose': [...]}"),
}


def policy_mapping_error(value: object, param: str = "policy_config") -> str | None:
    """Describe why ``value`` cannot be used as a provider keyword mapping.

    ``policy_config`` / ``policy_kwargs`` are free-form dicts with no signature
    to bounce off, so a value of the wrong *shape* - a ``"host=1"`` string, a
    list of pairs, a JSON blob an agent forgot to parse - is only detected when
    CPython splats it, far from the call the caller made.

    Args:
        value: The caller-supplied value, or ``None`` (always accepted: the
            parameter is optional).
        param: Parameter name to quote in the message; also selects the
            example shown. Unknown names fall back to a generic hint.

    Returns:
        A single-sentence explanation naming the parameter, the type received
        and a correct example, or ``None`` when ``value`` is usable as ``**``
        keyword arguments.
    """
    if value is None or isinstance(value, Mapping):
        return None
    hint = _POLICY_MAPPING_HINTS.get(param, "keyword arguments")
    return f"{param} must be a dict of {hint}; got {type(value).__name__} ({value!r})."


def create_policy(provider: str, **kwargs) -> Policy:
    """Create a policy instance.

    Accepts either a provider name or a smart string:

    - Provider name: ``create_policy("groot", port=5555)``
    - ZMQ URL: ``create_policy("zmq://localhost:5555")``
    - Shorthand: ``create_policy("mock")``

    All provider definitions live in ``registry/policies.json``.

    Args:
        provider: Provider name, HF model ID, or server URL.
        **kwargs: Provider-specific parameters.

    Returns:
        Policy instance ready for get_actions().

    Raises:
        UntrustedRemoteCodeError: If the provider loads HF models with
            ``trust_remote_code=True`` and ``STRANDS_TRUST_REMOTE_CODE``
            is not set.
    """
    canonical, PolicyClass, resolved_kwargs = _resolve_policy_class(provider, **kwargs)
    _check_trust_remote_code(canonical)
    return PolicyClass(**resolved_kwargs)


def preflight_policy(provider: str, observation_keys: set[str], **kwargs) -> None:
    """Run a provider's class-level :meth:`Policy.preflight` check, if any.

    Resolves ``provider`` to its policy class WITHOUT instantiating it (so no
    model weights are downloaded) and invokes the class's ``preflight`` hook
    with the runtime ``observation_keys`` and the provider kwargs. Providers
    that do not override :meth:`Policy.preflight` are a no-op.

    This is the fail-fast seam used by ``SimEngine.run_policy`` /
    ``eval_policy`` to catch a misconfiguration (e.g. sim camera names that
    cannot be routed to the model's declared image inputs) BEFORE the
    expensive ``create_policy`` download, instead of crashing deep inside the
    first inference. Resolution failures are swallowed (the matching error is
    surfaced authoritatively by the subsequent ``create_policy``); only the
    provider's own ``preflight`` ``ValueError`` propagates.

    Args:
        provider: Provider name, HF model ID, or server URL (as passed to
            ``create_policy``).
        observation_keys: Keys the runtime observation will contain (joint
            names + camera names).
        **kwargs: Provider-specific parameters (the policy_config).

    Raises:
        ValueError: When the resolved provider's ``preflight`` rejects the
            configuration.
    """
    try:
        _canonical, PolicyClass, resolved_kwargs = _resolve_policy_class(provider, **kwargs)
    except Exception as e:
        # Resolution problems (unknown provider, missing optional dep) are not
        # this hook's concern - create_policy raises the authoritative error.
        logger.debug("preflight_policy: could not resolve '%s' (%s); skipping", provider, e)
        return

    hook = getattr(PolicyClass, "preflight", None)
    base_hook = getattr(Policy.preflight, "__func__", Policy.preflight)
    if hook is None or getattr(hook, "__func__", hook) is base_hook:
        # Provider did not override the default no-op preflight.
        return
    PolicyClass.preflight(set(observation_keys), **resolved_kwargs)


def policy_provider_error(provider: str, **kwargs) -> str | None:
    """Return why ``provider`` cannot be resolved to a policy class, or ``None``.

    Probes the SAME resolution path :func:`create_policy` uses, without
    instantiating anything, so every spelling that provider accepts -- a
    registered name, a HuggingFace model ID, a ``zmq://`` / ``ws://`` URL, a
    ``host:port`` pair -- resolves here too. Only a name no spelling can reach
    yields a reason.

    This is the agent-tool companion to :func:`preflight_policy`, which
    deliberately swallows resolution failures on the stated grounds that
    "create_policy raises the authoritative error". That premise holds for a
    library caller, which sees the raise. It does not hold for the simulation's
    agent-tool surfaces: a raise out of ``run_policy`` / ``eval_policy``
    escapes the ``status=error`` envelope those tools are documented to return,
    and ``start_policy`` builds the policy on a worker thread, so the raise is
    never surfaced at all and the caller is told the policy started. Returning
    the reason lets each surface report it on its own channel instead.

    The returned message names every registered provider, so a caller that
    guessed a name gets the available set back rather than a traceback.

    A non-string ``provider`` is refused here too: resolution indexes the
    registry with it, so it would otherwise arrive as a bare ``TypeError``
    naming neither the parameter nor the problem.

    Args:
        provider: Provider name, HF model ID, or server URL (as passed to
            ``create_policy``).
        **kwargs: Provider-specific parameters (the policy_config), forwarded
            so resolution sees exactly what ``create_policy`` will.

    Returns:
        The resolution failure message, or ``None`` when ``provider`` resolves.
    """
    if not isinstance(provider, str):
        # Resolution indexes the registry with this value, so a non-string
        # reaches it as a bare TypeError naming neither the parameter nor the
        # problem ("argument of type 'NoneType' is not iterable").
        return (
            f"policy_provider must be a string, got {type(provider).__name__}. "
            "Pass a provider name (list_providers() reports them), a HuggingFace "
            "model ID, or a server URL."
        )
    try:
        _resolve_policy_class(provider, **kwargs)
    except ValueError as e:
        # ValueError is the unresolvable-NAME verdict. A missing optional
        # dependency (ImportError) and the trust-remote-code gate are separate
        # concerns with their own reporting, and are deliberately not caught.
        return str(e)
    return None
