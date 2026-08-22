"""The ABC owns the well-known goal vocabulary, and shipped providers grade it.

:meth:`Policy.get_actions` is where the issue #300 ``**kwargs`` goal vocabulary is
*defined* - it is the docstring a provider author reads to learn which keys a
caller may pass without coupling to a backend, and every other enumeration in the
tree (the package docstring, ``SimEngine.run_policy``, the mesh wire validator,
the mesh dispatcher's forward set) restates it downstream.

That makes the ABC the one list a new goal key can be forgotten on while every
consumer still works, because nothing executes a docstring. ``target_velocity``
was forgotten on it for exactly that reason: it is read by two independent
provider families, carried by ``run_policy`` / ``eval_policy`` /
``run_benchmark``, admitted by the mesh wire and forwarded by the mesh
dispatcher, and named by ``docs/policies/wbc.md`` as "one of the issue #300
well-known goal keys" - and absent from the ABC that the package docstring
explicitly points at for the list.

So these guards deliberately do NOT compare the ABC against another docstring.
They derive what the vocabulary must contain from **shipped provider code** - the
keys providers actually read out of ``**kwargs`` - and hold the ABC to it. A
docstring can drift from code; it cannot drift from code that is parsed.

The discriminator between a shared vocabulary key and a provider-private kwarg is
the one AGENTS.md convention 11 already codifies for value-domain guards: a second
independent caller is the evidence that a shared name is right, and one caller is
evidence of nothing. Applied here, a "provider family" is the provider package
(``policies/wbc/``, ``policies/motionbricks/``, ...), so ``wbc/policy.py`` and its
``wbc/gait.py`` variant count once between them. Measured on this tree the rule
separates the vocabulary from the private kwargs with no special cases at all:

    read by two or more families -> target_pose, target_joints,
                                    target_velocity, world_update
    read by exactly one family   -> target_orientation, target_heading,
                                    target_heading_angle, height,
                                    gait_frequency, style, mode,
                                    locomotion_style, speed_scale, replan,
                                    scene_model, planning_group, seed,
                                    text_prompt, guidance_scale, ... (26 keys)

which is why the assertion below can be a set *equality* rather than a
containment: the vocabulary is neither missing a key two families share nor
carrying one that no two families do.
"""

from __future__ import annotations

import ast
import collections
import pathlib
import re

from strands_robots.policies.base import Policy

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_POLICIES_ROOT = _REPO_ROOT / "strands_robots" / "policies"
_SIM_BASE = _REPO_ROOT / "strands_robots" / "simulation" / "base.py"

# A bullet in the ABC's well-known block: ``- ``name: type`` - prose``.
_WELL_KNOWN_BULLET = re.compile(r"^\s*-\s+``(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:", re.MULTILINE)

# The sentinels bounding that block inside the ``**kwargs:`` Args entry. The
# vocabulary is the bulleted list between "are **well-known**" and the
# "Providers MUST ignore unknown" paragraph that closes the entry.
_BLOCK_OPEN = "are **well-known**"
_BLOCK_CLOSE = "Providers MUST ignore unknown"


def _abc_well_known_keys() -> frozenset[str]:
    """The goal keys :meth:`Policy.get_actions` documents as well-known.

    Read from the live docstring rather than a copy, so this cannot pass by
    agreeing with a stale duplicate of the list it is checking.
    """
    doc = Policy.get_actions.__doc__ or ""
    start = doc.find(_BLOCK_OPEN)
    end = doc.find(_BLOCK_CLOSE)
    if start == -1 or end == -1 or end <= start:
        return frozenset()
    return frozenset(m.group("name") for m in _WELL_KNOWN_BULLET.finditer(doc[start:end]))


def _is_kwargs_name(node: ast.expr) -> bool:
    """True for ``kwargs`` and the locals providers derive from it.

    ``motionbricks`` reads its goal off a ``call_kwargs`` copy it forwards, so
    matching the bare name only would miss a real read - and reading a key off a
    derived mapping is the same exposure as reading it off ``kwargs`` directly.
    """
    return isinstance(node, ast.Name) and node.id.endswith("kwargs")


def _kwargs_key_read(node: ast.AST) -> str | None:
    """The string key ``node`` reads out of a kwargs mapping, if it does."""
    if (
        isinstance(node, ast.Subscript)
        and _is_kwargs_name(node.value)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get", "pop"}
        and _is_kwargs_name(node.func.value)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _goal_kwargs_by_provider_family() -> dict[str, frozenset[str]]:
    """Map each kwarg key providers read to the provider families that read it.

    A family is the provider package directory, so a provider's own variants
    (``wbc/policy.py`` and ``wbc/gait.py``) count once between them - otherwise
    every provider that ships two entry points would look like a consensus.
    """
    families: dict[str, set[str]] = collections.defaultdict(set)
    for py in sorted(_POLICIES_ROOT.rglob("*.py")):
        relative = py.relative_to(_POLICIES_ROOT).parts
        if len(relative) < 2:
            continue  # policies/*.py is shared machinery, not a provider family
        family = relative[0]
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            key = _kwargs_key_read(node)
            if key is not None:
                families[key].add(family)
    return {key: frozenset(value) for key, value in families.items()}


def _shared_goal_kwargs() -> frozenset[str]:
    """Kwarg keys two or more independent provider families read."""
    return frozenset(key for key, fams in _goal_kwargs_by_provider_family().items() if len(fams) >= 2)


def _run_policy_documented_goal_keys() -> frozenset[str]:
    """The goal keys ``SimEngine.run_policy`` names in its ``policy_kwargs`` entry.

    Parsed from source rather than imported so the guard does not depend on the
    simulation backend's optional dependencies being installed.
    """
    tree = ast.parse(_SIM_BASE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_policy":
            doc = ast.get_docstring(node) or ""
            start = doc.find("policy_kwargs:")
            if start == -1:
                return frozenset()
            # The entry ends at the next same-indent Args key (``n_episodes:``).
            end = doc.find("n_episodes:", start)
            block = doc[start:end] if end != -1 else doc[start:]
            return frozenset(re.findall(r"``(target_[a-z_]+|world_update)``", block))
    return frozenset()


def _restated_keys(text: str, marker: str) -> frozenset[str]:
    """The goal keys enumerated in the parenthesised run following ``marker``.

    Shared by the two graded restatements so there is exactly one parser to keep
    correct. Returns an empty set when the marker or the parentheses are absent;
    every caller asserts non-emptiness separately, because "the prose moved" and
    "the prose disagrees" are different failures and must not be reported as the
    same one.
    """
    start = text.find(marker)
    if start == -1:
        return frozenset()
    open_paren = text.find("(", start)
    close_paren = text.find(")", open_paren)
    if open_paren == -1 or close_paren == -1:
        return frozenset()
    return frozenset(re.findall(r"``(target_[a-z_]+|world_update)``", text[open_paren:close_paren]))


def test_the_abc_documents_a_non_empty_goal_vocabulary() -> None:
    """Non-vacuity: a docstring reflow that empties the parse must not read clean.

    Every assertion below compares against this set, so a parser that silently
    returns nothing would turn all of them green while the vocabulary went
    undefended - the failure mode a bounding-sentinel parse actually has.
    """
    keys = _abc_well_known_keys()
    assert keys, (
        "parsed no well-known goal keys out of Policy.get_actions.__doc__; the "
        f"block sentinels ({_BLOCK_OPEN!r} / {_BLOCK_CLOSE!r}) or the bullet "
        "format changed, and every parity guard in this module is now vacuous"
    )


def test_the_scan_finds_the_provider_reads_it_grades_against() -> None:
    """Non-vacuity for the other half: the code scan must actually find reads.

    If the AST scan returned nothing, the set-equality guard would reduce to
    "the ABC documents nothing" and pass on an empty vocabulary.
    """
    by_family = _goal_kwargs_by_provider_family()
    assert by_family, "the provider scan found no kwargs reads at all under policies/"
    assert _shared_goal_kwargs(), (
        "no kwarg key is read by two provider families; the family grouping or "
        f"the read patterns changed. scanned keys: {sorted(by_family)}"
    )


def test_the_abc_documents_every_goal_key_two_provider_families_share() -> None:
    """A key two provider families read is shared vocabulary and must be on the ABC.

    This is the direction that catches the real defect. ``target_velocity`` was
    read by ``wbc`` and ``motionbricks`` while the ABC listed three keys, so a
    provider author reading the contract could not discover the goal key two
    shipped families already honoured.
    """
    shared = _shared_goal_kwargs()
    documented = _abc_well_known_keys()
    by_family = _goal_kwargs_by_provider_family()
    missing = sorted(shared - documented)
    assert not missing, (
        "Policy.get_actions omits goal keys that two or more provider families "
        f"already read: {missing} "
        f"(read by { ({k: sorted(by_family[k]) for k in missing}) }). "
        f"It documents {sorted(documented)}. Add the key to the well-known block "
        "in strands_robots/policies/base.py - a caller cannot pass a goal key the "
        "contract does not name."
    )


def test_the_abc_documents_nothing_no_two_provider_families_share() -> None:
    """The vocabulary is extended by evidence, not widened by guesswork.

    The converse direction: a key on the ABC that no two families read is either
    dead vocabulary or a provider-private kwarg promoted by mistake. Keeping both
    directions means the ABC tracks shipped behaviour exactly rather than drifting
    in whichever direction is easier to add to.
    """
    shared = _shared_goal_kwargs()
    documented = _abc_well_known_keys()
    unshared = sorted(documented - shared)
    by_family = _goal_kwargs_by_provider_family()
    assert not unshared, (
        "Policy.get_actions documents well-known goal keys that no two provider "
        f"families read: {unshared} "
        f"(read by { ({k: sorted(by_family.get(k, frozenset())) for k in unshared}) }). "
        "Per AGENTS.md convention 11 a second independent caller is what makes a "
        "name shared; one caller is evidence of nothing, so such a key belongs in "
        "that provider's own documentation and not in the shared vocabulary."
    )


def test_a_single_family_kwarg_stays_out_of_the_shared_vocabulary() -> None:
    """Negative control naming the keys the fix must NOT have swept in.

    ``target_orientation`` and ``height`` are read by ``wbc`` (across its policy
    and its gait variant, one family) and ``target_heading`` by ``motionbricks``
    alone. They are ``target_``-shaped and sit in the same ``get_actions`` bodies
    as the real vocabulary, so a guard keyed on the name shape rather than on the
    second-caller rule would promote them. Pinning them here is what makes the
    equality above evidence that the allowlist was extended by one key rather
    than opened to everything that looks like a goal.
    """
    by_family = _goal_kwargs_by_provider_family()
    documented = _abc_well_known_keys()
    for private in ("target_orientation", "target_heading", "height"):
        assert private in by_family, (
            f"{private} is no longer read by any provider; this control now pins "
            "nothing and should be re-pointed at a current single-family kwarg"
        )
        assert len(by_family[private]) == 1, (
            f"{private} is now read by {sorted(by_family[private])} - two families "
            "share it, so it has become shared vocabulary and this control is stale"
        )
        assert private not in documented, (
            f"{private} is read by exactly one provider family "
            f"({sorted(by_family[private])}) but is documented as shared "
            "well-known vocabulary on Policy.get_actions"
        )


def test_the_abc_and_run_policy_document_the_same_goal_set() -> None:
    """The ABC and its main in-tree consumer must name one vocabulary.

    ``SimEngine.run_policy`` offers itself as "the local-sim analogue of the mesh
    ``tell()`` path", and the mesh wire validator and dispatcher forward set are
    already graded against *its* docstring. Pinning the two docstrings equal is
    what makes those downstream guards transitively anchored to the ABC that owns
    the vocabulary, instead of to a second copy of the list that can drift from
    it independently - which is the state this test was written to end.
    """
    abc_keys = _abc_well_known_keys()
    run_policy_keys = _run_policy_documented_goal_keys()
    assert run_policy_keys, (
        "parsed no goal keys out of SimEngine.run_policy's policy_kwargs entry; "
        "this guard is vacuous until its parse is repaired"
    )
    assert abc_keys == run_policy_keys, (
        "Policy.get_actions and SimEngine.run_policy document different #300 goal "
        f"vocabularies. ABC: {sorted(abc_keys)}; run_policy: {sorted(run_policy_keys)}. "
        f"Only on the ABC: {sorted(abc_keys - run_policy_keys)}; only on run_policy: "
        f"{sorted(run_policy_keys - abc_keys)}. The ABC is the definition - bring the "
        "consumer to it."
    )


def test_the_package_docstring_enumerates_the_same_vocabulary_as_the_abc() -> None:
    """``strands_robots.policies``'s own docstring copies the list and cites its source.

    It reads "the well-known ``**kwargs`` keys (...) documented on
    :meth:`Policy.get_actions`" - a convenience copy that names the ABC as its
    authority, so a reader who trusts the citation never opens the ABC and sees
    whatever the copy last said. It carried the same three-key set the ABC did.

    Only this one restatement is graded. Two other kinds deliberately are not, and
    both are correct at three keys: a *provider-scoped* list (``curobo/__init__``
    says cuRobo reads pose / joints / world_update, which is what cuRobo reads),
    and a *historical* one ("the ABC contract landed in #300 (well-known ... )",
    true of #300 as shipped). Holding either to the current vocabulary would
    demand a false statement, so the discriminator for grading is that the
    restatement claims to be the present, whole vocabulary.
    """
    import strands_robots.policies as policies_pkg

    enumerated = _restated_keys(policies_pkg.__doc__ or "", "well-known")
    assert enumerated, (
        "parsed no goal keys out of the strands_robots.policies docstring's "
        "well-known enumeration; the prose moved and this guard is vacuous "
        "until it is re-pointed"
    )
    documented = _abc_well_known_keys()
    assert enumerated == documented, (
        "the strands_robots.policies docstring enumerates a different goal "
        f"vocabulary than the Policy.get_actions ABC it cites. package: "
        f"{sorted(enumerated)}; ABC: {sorted(documented)}. Only in the package "
        f"docstring: {sorted(enumerated - documented)}; only on the ABC: "
        f"{sorted(documented - enumerated)}."
    )


def test_the_cosmos3_restatement_of_the_abc_list_stays_accurate() -> None:
    """``Cosmos3Policy.get_actions`` restates the ABC's list to disclaim all of it.

    It says "None of the well-known keys the ABC lists (...) is read by either
    backend" - an enumeration *and* a behavioural claim about the whole
    vocabulary. Both halves go stale when the vocabulary grows: the list silently
    falls behind, and the disclaimer starts covering fewer keys than it names
    while still reading as though it covered them all.

    So both halves are graded. The enumeration must equal the ABC's, and the
    behavioural claim must actually hold - checked against the same provider scan
    the vocabulary itself is derived from, not against the prose.
    """
    source = (_POLICIES_ROOT / "cosmos3" / "policy.py").read_text(encoding="utf-8")
    marker = "well-known keys the ABC lists"
    assert marker in source, (
        f"cosmos3/policy.py no longer contains {marker!r}; the disclaimer was "
        "reworded and this guard is vacuous until it is re-pointed"
    )
    enumerated = _restated_keys(source, marker)
    documented = _abc_well_known_keys()
    assert enumerated == documented, (
        "cosmos3/policy.py restates the ABC's well-known key list but names a "
        f"different set. cosmos3: {sorted(enumerated)}; ABC: {sorted(documented)}. "
        f"Only in cosmos3: {sorted(enumerated - documented)}; only on the ABC: "
        f"{sorted(documented - enumerated)}."
    )

    # The behavioural half of the same sentence.
    read_by_cosmos3 = {key for key, fams in _goal_kwargs_by_provider_family().items() if "cosmos3" in fams}
    contradicted = sorted(read_by_cosmos3 & documented)
    assert not contradicted, (
        "cosmos3/policy.py claims none of the well-known goal keys is read by "
        f"either backend, but it reads {contradicted}. Either the provider grew a "
        "goal-key reader or the disclaimer is now false."
    )
