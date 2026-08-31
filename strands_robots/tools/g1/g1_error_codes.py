"""Agent-facing lookup for the error codes the G1 locomotion / arm SDK returns.

The Unitree G1 SDK surfaces RPC-level and handler-level refusals as
integer return codes (``rc=3104`` for an RPC timeout, ``rc=7302`` for an
invalid FSM id, ``rc=7404`` for a write on an FSM the driver's gate
refuses). Every verb this package already ships that surfaces one of
those codes quotes the same
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES` entry verbatim so
the caller and the driver both read the SDK's refusal by the same
sentence. That table has lived in :mod:`._g1_common` since the package
was created but no ``@tool``-callable verb has ever exposed it: an agent
that reads the ``refusal_text`` a verb like
``g1_fsm_target_admits`` (removed lookup verb; clamps now live inline)
returns sees a single decoded line and cannot ask the package what the
other refusal codes are without importing the constant directly. This
module closes that gap so an agent planning a call sequence can list
the whole catalogue - or decode one code by number - through the same
tool surface every other verb here answers on.

Two things this module is deliberately *not*:

* An execution path. No SDK call runs here; the lookup answers by
  reading :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
  directly, and every field the verbs return is a snapshot of that
  dict. A verb like
  ``g1_fsm_target_admits`` (removed lookup verb; clamps now live inline)
  already surfaces the decoded text on the write-planning side; this
  module is the read-only catalogue side of the same conversation, so
  a caller who receives a ``refusal_code`` from any other verb can
  ask :func:`g1_decode_error_code` for the same text without having
  to reach the private constant.
* An SDK re-import. ``ERR_CODES`` is a Python literal snapshot of the
  return-code lexicon the SDK's :mod:`unitree_sdk2py.g1.loco` /
  :mod:`unitree_sdk2py.g1.arm` handlers observed against the real
  robot; the mapping lives in :mod:`._g1_common` (which never imports
  the SDK either) so ``import
  strands_robots.tools.g1.g1_error_codes`` pulls no
  ``unitree_sdk2py`` submodule - the import-hygiene contract every
  other file in this package carries, refs
  strands-labs/robots#358. An SDK release that widens or renames the
  code set is a
  :mod:`~strands_robots.tools.g1._g1_common`-side update; every verb
  quoting the text picks up the same change without a second
  copy-paste.

What this module does not decide.

* Whether a code is currently being returned by the driver. That is a
  live read on the driver's own status envelope
  (:meth:`~strands_robots.drivers.g1.G1Driver.get_status` names
  ``fsm_refusal`` / ``motion_switcher_open_error``); this catalogue
  is the *lexicon*, not the wire read. A caller who receives a
  refusal string on any verb can compare its ``code`` against this
  catalogue to see whether the SDK's handler has a known name for
  the number.
* Whether a code the SDK returns has a decoded name at all. The
  catalogue is a snapshot, not the SDK; a code the SDK returns that
  is missing from :data:`ERR_CODES` surfaces here as an
  ``unknown``-tagged refusal so a caller can distinguish a real
  gap (the SDK invented a code we have not observed) from a
  well-known refusal.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import ERR_CODES

#: The rc value the catalogue quotes on a lookup miss. Mirrors the same
#: contract as :func:`~strands_robots.tools.g1._g1_common.decode_code`:
#: a code outside the snapshot's keys renders as ``"unknown"`` so a
#: caller can distinguish a name the package has ("Invalid FSM id
#: (loco)") from a name the package does not (a code the SDK may have
#: added after the snapshot was taken). Named as a module constant so
#: a re-word lands in one place instead of drifting between the two
#: verbs' payloads.
_UNKNOWN_CODE_TEXT: str = "unknown"


def _describe(code: int) -> dict[str, Any]:
    """Build the per-code descriptor the verbs return.

    Kept here rather than inlined in :func:`g1_list_error_codes` so
    :func:`g1_decode_error_code`'s admitted-path payload names the same
    fields, and so a widen to the descriptor lands in one place. Every
    field is a snapshot read; no bus is touched.
    """
    return {
        "code": code,
        "text": ERR_CODES[code],
    }


@tool
def g1_list_error_codes() -> dict[str, Any]:
    """Return the SDK error codes the G1 locomotion / arm handlers surface.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    snapshot of :data:`~strands_robots.tools.g1._g1_common.ERR_CODES`
    read at call time. Useful when a caller receives a ``refusal_code``
    from any other verb in this package and wants to compare it
    against the catalogue of names the package quotes verbatim, or to
    plan a call sequence knowing which codes the driver's handlers
    surface on a refusal.

    Returns:
        A dict with ``status``, a ``count`` naming the number of
        catalogued codes, an ``error_codes`` list of descriptors (one
        per code, sorted ascending) carrying ``code`` and ``text``,
        and a bare ``codes`` list of just the integer codes for a
        caller who only needs the set. Every field is a snapshot of a
        module-level constant; no dynamic decode runs here. The
        catalogue mirrors what
        :func:`~strands_robots.tools.g1._g1_common.decode_code`
        would render for the same numbers, so a caller reading a
        refusal text from any other verb reads the same sentence
        here.
    """
    codes = sorted(ERR_CODES)
    return {
        "status": "success",
        "count": len(ERR_CODES),
        "error_codes": [_describe(code) for code in codes],
        "codes": codes,
    }


@tool
def g1_decode_error_code(code: int) -> dict[str, Any]:
    """Decode one SDK return code against the catalogued name.

    Read-only. Answers the same lookup
    :func:`~strands_robots.tools.g1._g1_common.decode_code` would
    compute internally, exposed through the ``@tool`` surface so a
    caller who received a ``refusal_code`` from any other verb can
    resolve it without importing the private constant. A code inside
    :data:`~strands_robots.tools.g1._g1_common.ERR_CODES` returns
    ``known=True`` with the decoded ``text``; a code outside returns
    ``known=False`` and the catalogue's ``unknown`` marker, so a
    caller can distinguish a name the package has from a name the
    package does not.

    Args:
        code: The integer rc the SDK returned. Must be an ``int``;
            ``bool`` is refused (``True`` is ``int(1)`` but a
            passed-through boolean is a caller mistake, not a valid
            decode query). A negative value is admitted decidably as
            ``unknown`` rather than refused: the catalogue carries no
            negative codes, but ``_g1_common.decode_code`` already
            renders any integer, so refusing one here would make this
            verb narrower than the renderer whose text it quotes. A
            transport-level -1 is the convention for an SDK call that
            raised instead of returning a rc; this verb reports it as
            ``known=False`` without inventing text for it.

    Returns:
        A dict with ``status`` (``"success"`` on any decidable answer,
        ``"error"`` on the type-mistake refusal), a ``query`` sub-dict
        carrying the supplied ``code``, a ``known`` boolean naming
        whether the code appears in the snapshot, and (when ``known``
        is ``True``) the decoded ``text`` from the catalogue. On an
        unknown code the dict carries the ``unknown`` marker under
        ``text`` so the returned envelope always names *something* -
        a caller composing an error message off this call does not
        have to branch on a missing key.
    """
    if isinstance(code, bool):
        return {
            "status": "error",
            "message": (f"code must be int, got bool ({code!r}). Refs strands-labs/robots#358."),
        }
    if not isinstance(code, int):
        return {
            "status": "error",
            "message": (f"code must be int, got {type(code).__name__} ({code!r}). Refs strands-labs/robots#358."),
        }

    known = code in ERR_CODES
    result: dict[str, Any] = {
        "status": "success",
        "query": {"code": code},
        "known": known,
    }
    result["text"] = ERR_CODES[code] if known else _UNKNOWN_CODE_TEXT
    return result
