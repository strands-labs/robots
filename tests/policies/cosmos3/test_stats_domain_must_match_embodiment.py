"""Explicit de-normalization stats must declare the domain they describe.

:func:`~strands_robots.policies.cosmos3.sim_ik.decode_cosmos_chunk_to_targets`
inverts the model's ``[-1, 1]`` quantile normalization with per-domain
``q01``/``q99`` stats before solving IK. Only two domains ship bundled stats
(``droid_lerobot``, ``bridge_orig_lerobot``); ``umi`` and ``av`` do not, and
those are the two domains ``nvidia/Cosmos3-Edge`` documents for its
forward-dynamics and inverse-dynamics examples.

That combination used to route a caller into a silent rescale:
:func:`~strands_robots.policies.cosmos3.action_decode.load_action_stats` refused
an unbundled domain and advised passing stats explicitly, while
``denormalize_quantile`` validates only the action *width* - and ``umi``,
``droid_lerobot`` and ``bridge_orig_lerobot`` are all 10 columns. So the only
stats a caller could load were accepted for ``umi`` and rescaled every
commanded pose delta (measured up to 2.77x between the two bundled domains) with
nothing reported.

These tests pin that explicit stats now declare their domain and that a domain
which does not describe the embodiment is refused by name.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import sim_ik
from strands_robots.policies.cosmos3.action_decode import denormalize_quantile, load_action_stats
from strands_robots.policies.cosmos3.embodiments import get_embodiment

# Reuse the sibling module's complete mink-free stand-ins rather than rebuilding
# them: ``_FakeBridge`` models a perfect IK solver (and carries the ``model.nq``
# the re-anchoring path reads), ``_home_pose`` a reachable starting pose.
from .test_sim_ik_decode_orchestration import _FakeBridge, _home_pose

# Domains with bundled stats, and the two Cosmos3-Edge documents without them.
BUNDLED = ("droid_lerobot", "bridge_orig_lerobot")
UNBUNDLED = ("umi", "av")


def _chunk(width: int, t: int = 4) -> np.ndarray:
    """A small in-range normalized action chunk of the given width."""
    return np.full((t, width), 0.1, dtype=np.float32)


def _decode(**kwargs: Any) -> Any:
    """Funnel so deliberately off-contract arguments stay one documented shape."""
    return sim_ik.decode_cosmos_chunk_to_targets(**kwargs)


class TestThePremiseTheGuardRestsOn:
    """The widths really do coincide, so a width check cannot separate domains."""

    def test_the_two_bundled_domains_and_umi_share_an_action_width(self) -> None:
        widths = {d: load_action_stats(d)["q01"].shape[-1] for d in BUNDLED}
        assert set(widths.values()) == {10}, widths
        assert get_embodiment("umi").raw_action_dim == 10

    @pytest.mark.parametrize("domain", UNBUNDLED)
    def test_the_domains_cosmos3_edge_documents_ship_no_bundled_stats(self, domain: str) -> None:
        with pytest.raises(FileNotFoundError, match=domain):
            load_action_stats(domain)

    def test_another_domains_quantiles_rescale_the_same_action(self) -> None:
        """Why the domain matters physically, not just as bookkeeping."""
        action = np.full((1, 10), 0.5, dtype=np.float32)
        a, b = (load_action_stats(d) for d in BUNDLED)
        first = denormalize_quantile(action, a["q01"], a["q99"])[0, :3]
        second = denormalize_quantile(action, b["q01"], b["q99"])[0, :3]
        ratio = np.abs(second) / np.abs(first)
        assert ratio.max() > 2.0, (first, second, ratio)


class TestExplicitStatsMustDeclareTheirDomain:
    """``stats=`` without ``stats_domain=`` is refused, naming the domain to pass."""

    def test_stats_without_a_domain_is_refused(self) -> None:
        emb = get_embodiment("droid")
        stats = load_action_stats(emb.domain_name)
        with pytest.raises(ValueError, match="stats_domain") as excinfo:
            _decode(
                action_chunk=_chunk(emb.raw_action_dim),
                embodiment=emb,
                ik_bridge=_FakeBridge(home=_home_pose()),
                q_init=np.zeros(7),
                stats=stats,
            )
        message = str(excinfo.value)
        # Names the parameter, the domain to declare, and the consequence.
        assert f"stats_domain={emb.domain_name!r}" in message
        assert "silently rescales" in message

    def test_a_domain_that_does_not_describe_the_embodiment_is_refused(self) -> None:
        """The umi-with-droid-quantiles path: same width, different physical scale."""
        emb = get_embodiment("umi")
        wrong = load_action_stats("droid_lerobot")
        assert wrong["q01"].shape[-1] == emb.raw_action_dim  # width alone cannot catch it
        with pytest.raises(ValueError, match="does not") as excinfo:
            _decode(
                action_chunk=_chunk(emb.raw_action_dim),
                embodiment=emb,
                ik_bridge=_FakeBridge(home=_home_pose()),
                q_init=np.zeros(7),
                stats=wrong,
                stats_domain="droid_lerobot",
            )
        message = str(excinfo.value)
        assert "droid_lerobot" in message and repr(emb.domain_name) in message

    def test_a_matching_domain_is_honored(self) -> None:
        """The over-reach control: a declared, matching domain still decodes."""
        emb = get_embodiment("droid")
        stats = load_action_stats(emb.domain_name)
        out = _decode(
            action_chunk=_chunk(emb.raw_action_dim, t=4),
            embodiment=emb,
            ik_bridge=_FakeBridge(home=_home_pose()),
            q_init=np.zeros(7),
            stats=stats,
            stats_domain=emb.domain_name,
        )
        assert out["qpos"].shape == (4, 7)

    def test_an_unbundled_domain_decodes_once_its_own_stats_are_declared(self) -> None:
        """The remedy the refusal advises: Edge's ``umi`` domain becomes usable."""
        emb = get_embodiment("umi")
        d = emb.raw_action_dim
        own = {
            "q01": np.full(d, -0.05, dtype=np.float32),
            "q99": np.full(d, 0.05, dtype=np.float32),
        }
        out = _decode(
            action_chunk=_chunk(d, t=emb.action_chunk_size),
            embodiment=emb,
            ik_bridge=_FakeBridge(home=_home_pose()),
            q_init=np.zeros(7),
            stats=own,
            stats_domain=emb.domain_name,
        )
        assert out["qpos"].shape == (emb.action_chunk_size, 7)
        assert out["gripper"] is not None  # umi carries a trailing grasp column


class TestTheDefaultPathIsUnchanged:
    """Omitting ``stats`` still loads the embodiment's own bundled stats."""

    @pytest.mark.parametrize("name", ["droid", "bridge"])
    def test_a_bundled_embodiment_needs_no_stats_argument(self, name: str) -> None:
        emb = get_embodiment(name)
        out = _decode(
            action_chunk=_chunk(emb.raw_action_dim, t=3),
            embodiment=emb,
            ik_bridge=_FakeBridge(home=_home_pose()),
            q_init=np.zeros(7),
        )
        assert out["qpos"].shape == (3, 7)

    def test_an_unbundled_embodiment_still_reports_the_missing_domain(self) -> None:
        emb = get_embodiment("umi")
        with pytest.raises(FileNotFoundError) as excinfo:
            _decode(
                action_chunk=_chunk(emb.raw_action_dim),
                embodiment=emb,
                ik_bridge=_FakeBridge(home=_home_pose()),
                q_init=np.zeros(7),
            )
        message = str(excinfo.value)
        # The advice now names the safe way to supply them.
        assert "stats_domain='umi'" in message
        assert "not a substitute" in message
