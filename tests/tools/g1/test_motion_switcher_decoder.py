"""The motion-switcher decoder produces the FSM id the driver's gate reads.

The G1 driver's ``_check_motion_gates`` compares ``_fsm_id`` against
``HANDSHAKE_FSMS`` and ``WALK_FSMS`` to decide whether an arm-SDK or locomotion
write is admitted. On the shipped tree ``_fsm_id`` is ``None`` and no code
writes it, so every gate refuses. Issue #2765 names ``MotionSwitcherClient``
as the missing producer; :mod:`._motion_switcher` owns the read half of that
API, and this file grades the decoder against every return shape the SDK is
known to produce plus every shape it does not.

Every cell drives a fake ``MotionSwitcherClient``: opening a real one requires
the DDS bus and the SDK, both of which the module deliberately does not import
at load time. The fake reproduces the ``(status, result)`` return shape the
SDK's ``CheckMode`` docs describe, and the checks below drive it through the
values that separate one branch of :func:`decode_fsm_id` from another.

Refutation for the whole file: this decoder would be untestable off-hardware
if the SDK's ``CheckMode`` return-shape were undocumented or private. It is
neither -- the shape is a ``(status, result)`` pair whose ``result`` carries
``name`` (mode label) and ``form`` (FSM id), and the SDK's own example prints
that dict from the same call. Every fake here matches that dict exactly, and a
new SDK release that changed the shape would fire the shape-refusal cells,
naming the received type in the failure message.
"""

from __future__ import annotations

from typing import Any

import pytest

import strands_robots.tools.g1._motion_switcher as _motion_switcher
from strands_robots.tools.g1._motion_switcher import (
    FSMReading,
    decode_fsm_id,
    read_fsm_id,
)


class _FakeSwitcherClient:
    """Stands in for ``MotionSwitcherClient`` on the wire.

    ``CheckMode`` returns whatever was queued -- one queued value per call,
    consumed in order, so a cell driving two reads sees two distinct
    returns. A caller that reads past the queue would get an
    ``IndexError``; the cells below never do, and that failure mode is a
    louder signal than a silently repeated value.
    """

    def __init__(self, returns: list[Any]) -> None:
        self._returns = list(returns)
        self.check_mode_call_count = 0

    def CheckMode(self) -> Any:  # noqa: N802 - SDK spelling
        self.check_mode_call_count += 1
        return self._returns.pop(0)


class _RaisingSwitcherClient:
    """A client whose ``CheckMode`` raises -- transport-failure branch."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.check_mode_call_count = 0

    def CheckMode(self) -> Any:  # noqa: N802
        self.check_mode_call_count += 1
        raise self._exc


class TestDecodeFsmIdOnOkReadings:
    """The OK branch: ``status == 0`` and a well-shaped result."""

    def test_active_mode_yields_the_form_integer_as_fsm_id(self) -> None:
        """A well-shaped OK reading with an active mode reports its FSM id.

        The SDK's ``CheckMode`` on an active mode returns ``(0, {"name":
        "ai", "form": 500})``. The decoder reads ``form`` as the FSM id
        because that is the integer the driver's ``HANDSHAKE_FSMS`` /
        ``WALK_FSMS`` sets are stated in. The ``name`` string is
        preserved on the reading so a diagnostic message names the mode
        without reading the SDK a second time.
        """
        reading = decode_fsm_id((0, {"name": "ai", "form": 500}))
        assert reading == FSMReading(fsm_id=500, mode_name="ai", refusal=None)

    def test_no_mode_selected_yields_fsm_id_none_without_refusal(self) -> None:
        """``name == ""`` is the SDK's "no mode" reading, not a failure.

        The SDK returns ``(0, {"name": ""})`` when no motion mode is
        selected. That is the same information the driver's gate carries
        with ``_fsm_id = None`` today, and the decoder must not refuse
        this reading -- a refusal here would fabricate an error the SDK
        did not report and would keep the gate from ever transitioning
        out of "unknown" cleanly.
        """
        reading = decode_fsm_id((0, {"name": ""}))
        assert reading == FSMReading(fsm_id=None, mode_name="", refusal=None)

    @pytest.mark.parametrize("fsm_id", [500, 501, 801])
    def test_every_handshake_fsm_id_round_trips(self, fsm_id: int) -> None:
        """The three FSM ids ``HANDSHAKE_FSMS`` names all decode as-is.

        Every FSM id the arm-SDK gate names round-trips through the
        decoder unchanged. This is a completeness cell: a decoder that
        masked, clamped or remapped the value would drift out of
        agreement with the gate here rather than at a hardware session.
        """
        reading = decode_fsm_id((0, {"name": "ai", "form": fsm_id}))
        assert reading.fsm_id == fsm_id
        assert reading.refusal is None


class TestDecodeFsmIdRefusesNonOkStatus:
    """The RPC-failed branch: ``status != 0`` refuses without reading result."""

    def test_non_zero_status_refuses_and_names_the_code(self) -> None:
        """A non-zero status refuses with the code decoded via ``ERR_CODES``.

        ``ERR_CODES`` renders known codes; unknown codes surface as the
        integer plus ``unknown``. Either way the refusal names the code
        the caller received, which is the actionable value. The result
        dict is *not* read on this branch -- a failed RPC's result is
        not a truthful reading, and reading it would drift into
        reporting an FSM id the switcher did not confirm.
        """
        reading = decode_fsm_id((7301, {"name": "ai", "form": 500}))
        assert reading.fsm_id is None
        assert reading.mode_name == ""
        assert reading.refusal is not None
        assert "7301" in reading.refusal
        # The rendered text should carry the ``ERR_CODES`` translation.
        assert "LocoState not available" in reading.refusal

    def test_unknown_status_code_still_refuses_by_number(self) -> None:
        """A code not in ``ERR_CODES`` surfaces as its integer, not as OK.

        A firmware update that adds a new response code must not decode
        as success. The refusal path names the integer even when the
        translation is unknown, so an operator reading the log sees the
        value to file, not a silent OK.
        """
        reading = decode_fsm_id((9999, {"name": "ai", "form": 500}))
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "9999" in reading.refusal


class TestDecodeFsmIdRefusesMalformedShapes:
    """The shape-refusal branch: the SDK never returns these, and the decoder says so."""

    def test_return_that_is_not_a_tuple_refuses(self) -> None:
        """A non-tuple return is a shape the SDK does not produce.

        ``CheckMode`` returns a ``(status, result)`` pair. A dict, a
        list, or an integer at that position is a wire-shape mismatch
        the gate cannot silently open on; the refusal names the
        received type so a mis-mocked test or a broken SDK is diagnosed
        by its shape, not by a downstream ``TypeError``.
        """
        reading = decode_fsm_id({"name": "ai", "form": 500})
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "tuple" in reading.refusal
        assert "dict" in reading.refusal

    def test_tuple_of_wrong_length_refuses(self) -> None:
        """A tuple of length != 2 is a shape mismatch, not a valid reading."""
        reading = decode_fsm_id((0,))
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "length" in reading.refusal or "tuple" in reading.refusal

    def test_non_int_status_refuses(self) -> None:
        """A non-int status is a shape mismatch even when the result looks OK.

        The SDK types ``status`` as an integer response code. A string,
        ``None``, or a float here means the client is not a
        ``MotionSwitcherClient`` -- a fixture accident that would
        otherwise decode the result at position 1 as if it were a
        successful reading.
        """
        reading = decode_fsm_id(("OK", {"name": "ai", "form": 500}))
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "status" in reading.refusal
        assert "str" in reading.refusal

    def test_non_dict_result_refuses(self) -> None:
        """The result must be a dict; anything else is a shape mismatch."""
        reading = decode_fsm_id((0, "ai"))
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "dict" in reading.refusal

    def test_missing_name_key_refuses(self) -> None:
        """A result dict without ``name`` is a shape the SDK does not produce.

        Every ``CheckMode`` return carries ``name`` -- it is the only key
        the SDK populates on the "no mode" reading. A dict without it is
        a fixture error or an SDK version drift; either way the refusal
        must name the received keys so the drift is diagnosable.
        """
        reading = decode_fsm_id((0, {"form": 500}))
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "name" in reading.refusal

    def test_non_string_name_refuses(self) -> None:
        """``name`` must be a string; an integer or ``None`` refuses.

        A caller who wrote ``result["name"] = 500`` mistaking the mode
        label for the FSM id would silently decode the value as if it
        were a mode. Refusing on type keeps that error at the wire
        boundary rather than letting it become an admitted gate.
        """
        reading = decode_fsm_id((0, {"name": 500, "form": 500}))
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "name" in reading.refusal
        assert "int" in reading.refusal

    def test_active_mode_without_form_refuses(self) -> None:
        """A named mode without ``form`` cannot yield an FSM id.

        The SDK populates ``form`` when a mode is active. A dict with
        ``name != ""`` but no ``form`` is a shape the SDK does not
        produce on any recorded version, and defaulting to ``None`` here
        would confuse "no mode selected" with "mode selected, FSM id
        unavailable". The refusal names the mode and the missing key
        so the two are distinguishable at the log.
        """
        reading = decode_fsm_id((0, {"name": "ai"}))
        assert reading.fsm_id is None
        assert reading.mode_name == "ai"
        assert reading.refusal is not None
        assert "form" in reading.refusal
        assert "ai" in reading.refusal

    def test_non_int_form_refuses(self) -> None:
        """``form`` must be an ``int``; a string or float refuses."""
        reading = decode_fsm_id((0, {"name": "ai", "form": "500"}))
        assert reading.fsm_id is None
        assert reading.mode_name == "ai"
        assert reading.refusal is not None
        assert "form" in reading.refusal
        assert "str" in reading.refusal

    def test_boolean_form_refuses_even_though_bool_is_a_subclass_of_int(self) -> None:
        """``True`` must not decode as FSM id ``1``.

        ``bool`` is a subclass of ``int`` in Python, so a naive
        ``isinstance(form, int)`` accepts ``True`` and ``False`` and
        decodes them as ``1`` and ``0``. Neither is a value
        ``HANDSHAKE_FSMS`` or ``WALK_FSMS`` names, but the point of
        refusing here is to catch the class of mis-mock that produced a
        boolean in the first place. A silent decode to ``1`` would let a
        wrong test pass; the explicit refusal names the type.
        """
        reading = decode_fsm_id((0, {"name": "ai", "form": True}))
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "bool" in reading.refusal


class TestReadFsmIdCallsCheckMode:
    """``read_fsm_id`` calls the client once and decodes what it returns."""

    def test_healthy_client_returns_a_decoded_reading(self) -> None:
        """The one-call path: happy return decodes into an ``FSMReading``."""
        client = _FakeSwitcherClient([(0, {"name": "ai", "form": 501})])
        reading = read_fsm_id(client)
        assert reading == FSMReading(fsm_id=501, mode_name="ai", refusal=None)
        assert client.check_mode_call_count == 1

    def test_second_call_reads_the_second_return(self) -> None:
        """Two calls yield two decoded readings; the client is stateful.

        A driver that polls ``read_fsm_id`` from a control loop expects
        the value to reflect the most recent switcher state, not a
        cached first reading. This cell drives two returns and asserts
        the second one comes through, so a caching bug in
        :func:`read_fsm_id` would fire here rather than at a hardware
        session where the mode changed unnoticed.
        """
        client = _FakeSwitcherClient(
            [
                (0, {"name": "ai", "form": 500}),
                (0, {"name": ""}),  # released back to no-mode
            ]
        )
        first = read_fsm_id(client)
        second = read_fsm_id(client)
        assert first.fsm_id == 500
        assert second.fsm_id is None
        assert second.refusal is None
        assert client.check_mode_call_count == 2

    def test_client_without_check_mode_refuses(self) -> None:
        """A mis-passed object refuses at the read boundary.

        A caller who passed the wrong object (a subscriber set, say,
        confused for the switcher) would otherwise raise
        ``AttributeError`` deep inside :func:`read_fsm_id`. The refusal
        names the type received so the mis-plumb is diagnosed at the
        boundary.
        """
        reading = read_fsm_id(object())

        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "CheckMode" in reading.refusal
        assert "object" in reading.refusal

    def test_client_whose_check_mode_is_not_callable_refuses(self) -> None:
        """``CheckMode`` present but non-callable is refused, not called.

        A test double that assigned ``CheckMode = None`` -- the class of
        typo that survives a lint pass -- must refuse rather than let
        the ``None()`` call propagate a ``TypeError`` upward.
        """

        class Broken:
            CheckMode = None  # noqa: N815 - matching SDK spelling

        reading = read_fsm_id(Broken())
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "CheckMode" in reading.refusal

    def test_transport_exception_reports_the_class_and_message(self) -> None:
        """A raising ``CheckMode`` is caught and reported, not propagated.

        The SDK's RPC path can raise on a broken transport. A control
        loop that let that exception propagate would abort the step and
        leak thread state; the driver's gate wants a *refusal*, not an
        exception. This cell drives a raising client and asserts the
        message names the exception class so an operator reading the
        log sees the transport failure by its type.
        """
        client = _RaisingSwitcherClient(TimeoutError("switcher offline"))
        reading = read_fsm_id(client)
        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "TimeoutError" in reading.refusal
        assert "switcher offline" in reading.refusal


class TestModuleImportsWithoutTheSDK:
    """The whole module loads on hosts without ``unitree_sdk2py`` installed.

    The invariant :mod:`._dds_engine` and :mod:`._g1_common` already carry --
    "``unitree_sdk2py`` is not imported at module load" -- extends to this
    module too. Every SDK read goes through ``_load_motion_switcher_client``,
    which lazy-imports on first call. This class asserts the import itself
    is safe without the SDK by re-importing the module in isolation.
    """

    def test_module_can_be_imported_without_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing :mod:`._motion_switcher` never touches the SDK.

        If the module imported the SDK at load, this test would either
        fail on a host without ``unitree_sdk2py`` (CI, Thor) or would
        need to mock ``sys.modules`` to skip the import -- both of
        which are louder failure modes than the check here. The cell
        re-imports the module through :mod:`importlib` and asserts no
        ``unitree_sdk2py`` submodule was pulled in as a side effect.

        The registry entry is dropped through ``monkeypatch`` so it is put
        back afterwards. Removing it and leaving it removed does not undo an
        import, it orphans every reference already bound to this module -
        ``tests/drivers/test_motion_switcher_open_is_under_the_shared_dds_lock``
        binds it at module level and patches
        :func:`_load_motion_switcher_client` on it, and against an orphan that
        patch is invisible to the driver's own lazy import.
        """
        import importlib
        import sys

        # Snapshot the SDK modules present before the re-import.
        before = {k for k in sys.modules if k.startswith("unitree_sdk2py")}

        # Force a fresh import so the module's top-level runs again.
        monkeypatch.delitem(sys.modules, "strands_robots.tools.g1._motion_switcher", raising=False)
        importlib.import_module("strands_robots.tools.g1._motion_switcher")

        after = {k for k in sys.modules if k.startswith("unitree_sdk2py")}
        # The re-import must not have added any new ``unitree_sdk2py`` entries.
        # (If the SDK was already loaded by a sibling test, ``before`` will
        # have entries; the assertion is on the *delta*, not the set.)
        assert after == before, (
            "importing _motion_switcher pulled in "
            f"{sorted(after - before)}; the SDK must be lazy-loaded only "
            "inside _load_motion_switcher_client"
        )


class TestTheSdkSeamNamesTheModuleTheSdkShips:
    """The lazy-import seam resolves to a module ``unitree_sdk2py`` has.

    :func:`_load_motion_switcher_client` is the only function in the module
    that names an SDK path, and no other cell in this file calls it -- every
    other cell hands :func:`read_fsm_id` an already-open client or calls
    :func:`decode_fsm_id` directly. That is the right shape for grading the
    decode, and it means the *path* is graded by nothing: a seam naming a
    module the SDK does not ship stays green through this whole file and
    raises ``ModuleNotFoundError`` the first time a caller opens a real
    client.

    ``MotionSwitcherClient`` ships at
    ``unitree_sdk2py.comm.motion_switcher.motion_switcher_client``. It sits
    under ``comm/`` rather than ``g1/`` because the motion switcher is shared
    across platforms -- the SDK's ``example/g1``, ``example/h1``,
    ``example/h1_2``, ``example/go2``, ``example/b2`` and ``example/b2w``
    low-level examples all import it from that one place, and ``g1/`` holds
    only ``arm``, ``audio`` and ``loco``.

    Two layers, because the SDK is not installable in CI:

    1. :meth:`test_the_seam_resolves_against_a_stand_in_sdk_at_the_real_path`
       builds the package tree at the real path in ``sys.modules`` and asserts
       the seam finds the class there. No SDK needed, so this is the layer
       that grades the path on every install.
    2. :meth:`test_the_seam_resolves_against_the_real_sdk` calls the seam for
       real when ``unitree_sdk2py`` happens to be importable (a hardware host,
       never CI). It skips otherwise, so it adds a hardware check without
       making the file depend on one.
    """

    def test_the_constant_names_the_shared_comm_package(self) -> None:
        """The path constant names ``comm.motion_switcher``, not ``g1``.

        Stated as a literal so a re-guess at the package lands here with a
        message naming both spellings, rather than in a traceback on a robot.
        """
        assert _motion_switcher._SDK_MODULE == ("unitree_sdk2py.comm.motion_switcher.motion_switcher_client"), (
            "the motion switcher ships under comm/, shared across platforms; "
            f"the seam names {_motion_switcher._SDK_MODULE!r}"
        )
        assert "unitree_sdk2py.g1.motion_switcher" not in _motion_switcher._SDK_MODULE, (
            "unitree_sdk2py/g1/ holds only arm, audio and loco -- there is no g1.motion_switcher package to import"
        )

    def test_the_seam_resolves_against_a_stand_in_sdk_at_the_real_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The seam finds the class when the SDK tree is at the real path.

        The stand-in is registered at
        ``unitree_sdk2py.comm.motion_switcher.motion_switcher_client`` only.
        A seam looking anywhere else raises ``ModuleNotFoundError``, which is
        exactly the failure a real host produces -- so this cell reproduces the
        hardware failure with no hardware and no SDK.
        """
        import sys
        import types

        sentinel = type("MotionSwitcherClient", (), {})

        leaf = types.ModuleType("unitree_sdk2py.comm.motion_switcher.motion_switcher_client")
        leaf.MotionSwitcherClient = sentinel  # type: ignore[attr-defined]
        # Every level of the tree, with ``__path__`` so the intermediate names
        # are packages rather than plain modules.
        packages = {
            "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
            "unitree_sdk2py.comm": types.ModuleType("unitree_sdk2py.comm"),
            "unitree_sdk2py.comm.motion_switcher": types.ModuleType("unitree_sdk2py.comm.motion_switcher"),
        }
        for name, module in packages.items():
            module.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(
            sys.modules,
            "unitree_sdk2py.comm.motion_switcher.motion_switcher_client",
            leaf,
        )
        # ``import a.b.c`` reads the parent attribute, so wire each level.
        packages["unitree_sdk2py"].comm = packages["unitree_sdk2py.comm"]  # type: ignore[attr-defined]
        packages["unitree_sdk2py.comm"].motion_switcher = packages[  # type: ignore[attr-defined]
            "unitree_sdk2py.comm.motion_switcher"
        ]
        packages["unitree_sdk2py.comm.motion_switcher"].motion_switcher_client = leaf  # type: ignore[attr-defined]

        assert _motion_switcher._load_motion_switcher_client() is sentinel

    def test_the_seam_resolves_against_the_real_sdk(self) -> None:
        """On a host with the SDK, the seam returns a real class.

        Skipped where ``unitree_sdk2py`` is absent -- which is every CI run,
        so this cell is the hardware-side half and the stand-in cell above is
        the one that holds everywhere.
        """
        pytest.importorskip(
            "unitree_sdk2py.comm.motion_switcher.motion_switcher_client",
            reason="the Unitree SDK is not installed on this host",
        )
        client_class = _motion_switcher._load_motion_switcher_client()
        assert isinstance(client_class, type)
        assert client_class.__name__ == "MotionSwitcherClient"
        assert callable(getattr(client_class, "CheckMode", None)), (
            "the class the seam resolves must carry the CheckMode method decode_fsm_id decodes"
        )


class TestAValueWhoseLengthCannotBeReadIsRefusedNotRaised:
    """The shape refusal reports a 0-d array rather than raising from it.

    :func:`decode_fsm_id` refuses a return that is not a ``(status, result)``
    pair, and its message names the length it received. Reading that length
    with ``hasattr(value, "__len__")`` followed by ``len(value)`` is unsafe for
    a value class this library receives routinely: a 0-d numpy array
    (``np.array(0.5)``, the result of a reduction such as ``np.mean(...)``) and
    a 0-d torch tensor both *declare* ``__len__`` and then raise from it, so the
    probe passes and the ``len()`` call escapes with a bare ``len() of unsized
    object`` -- out of the one function whose whole purpose is to answer an
    unusable input with a message.

    :func:`strands_robots.utils.sequence_length` is the single owner of that
    rule and reports ``None`` for a 0-d array and for a plain scalar alike;
    ``tests/test_unsized_value_is_refused_not_raised.py`` pins every surface in
    the package through it. These cells pin the two ends of the behaviour here:
    the 0-d value is refused with a message, and a value that genuinely has a
    readable length still reports it.
    """

    def test_a_zero_dimensional_array_is_refused_with_a_message(self) -> None:
        """A 0-d numpy array reports as a refusal, not a ``TypeError``.

        The pre-fix spelling raised ``TypeError: len() of unsized object`` here
        -- from inside the refusal path, past the caller that was promised a
        :class:`FSMReading`.
        """
        numpy = pytest.importorskip("numpy")

        reading = decode_fsm_id(numpy.array(0.5))

        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "must be a (status, result) tuple" in reading.refusal
        assert "ndarray" in reading.refusal, (
            f"the refusal names the type it received so the caller can see what was handed in; got {reading.refusal!r}"
        )

    def test_a_zero_dimensional_tensor_is_refused_with_a_message(self) -> None:
        """A 0-d torch tensor takes the same branch, for the same reason."""
        torch = pytest.importorskip("torch")

        reading = decode_fsm_id(torch.tensor(0.5))

        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "must be a (status, result) tuple" in reading.refusal

    @pytest.mark.parametrize(
        ("value", "expected_length"),
        [
            ((0, {"name": ""}, "extra"), "3"),
            ((0,), "1"),
            ((), "0"),
        ],
        ids=["three-element-tuple", "one-element-tuple", "empty-tuple"],
    )
    def test_a_readable_length_is_still_reported(self, value: Any, expected_length: str) -> None:
        """A value with a readable length still names it in the refusal.

        The over-reach guard for the cells above: routing the probe through
        :func:`~strands_robots.utils.sequence_length` must not turn every
        length into ``?``.
        """
        reading = decode_fsm_id(value)

        assert reading.refusal is not None
        assert f"of length {expected_length}" in reading.refusal

    def test_a_value_with_no_length_at_all_still_reports_a_question_mark(self) -> None:
        """A plain scalar carries no length and says so.

        This one passed before the fix too -- a ``float`` has no ``__len__``, so
        the ``hasattr`` probe declined it correctly. It is here as the control
        that the ``?`` branch is still reachable.
        """
        reading = decode_fsm_id(0.5)

        assert reading.refusal is not None
        assert "got float of length ?" in reading.refusal


class TestTheAbsorbedSdkExceptionIsReported:
    """The one absorbed failure in the module is logged, not only returned.

    Every other failure :func:`decode_fsm_id` and :func:`read_fsm_id` produce
    is already a ``refusal`` string the caller reads, so nothing is lost. The
    ``except`` clause in :func:`read_fsm_id` is different: it converts an SDK
    exception into a refusal, and the traceback is not part of what the caller
    gets back. An FSM read that fails once a second inside a control loop is
    exactly the case an operator needs a log line for.

    ``_dds_engine`` reports its own absorbed exceptions at WARNING with a
    ``%s``-style format, and these cells hold this module to the same spelling
    so the two boundaries read alike.
    """

    def test_a_raising_check_mode_is_logged_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """The SDK exception's class and message reach the log."""

        class Boom:
            def CheckMode(self) -> Any:
                raise RuntimeError("dds participant is gone")

        with caplog.at_level("WARNING", logger=_motion_switcher.__name__):
            reading = read_fsm_id(Boom())

        assert reading.fsm_id is None
        assert reading.refusal is not None
        assert "RuntimeError" in reading.refusal
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, f"expected one WARNING, got {caplog.records!r}"
        message = warnings[0].getMessage()
        assert "RuntimeError" in message
        assert "dds participant is gone" in message

    def test_the_absorbed_exception_is_the_only_thing_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A refusal the caller already reads is not also logged.

        The over-reach guard: the module reports through its return value, so a
        shape refusal must stay silent. Logging every declined decode would make
        an ordinary "no mode selected" reading indistinguishable from a
        transport failure in an operator's log.
        """
        with caplog.at_level("DEBUG", logger=_motion_switcher.__name__):
            assert decode_fsm_id((0, {"name": ""})).refusal is None
            assert decode_fsm_id("not a tuple").refusal is not None
            assert decode_fsm_id((0, {"name": "ai"})).refusal is not None

        assert caplog.records == [], (
            "the module reports refusals through FSMReading.refusal; only the "
            f"absorbed SDK exception is logged, got {caplog.records!r}"
        )
