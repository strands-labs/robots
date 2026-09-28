"""The red button's label and the action a click takes come from one lockout state.

The button reads E-STOP or RESUME and a click posts `/api/safety/estop` or
`/api/safety/resume`. The two were decided in different places: the label was
written only by the click handler's own answer, while every other path that
painted the lockout line - page load, a telemetry frame, an error handler -
left the label as it was. So a page loaded under an e-stop engaged elsewhere
read "e-stop engaged" beside a button that read E-STOP, and pressing that
button thawed every frozen session: the operator's one reflex under an e-stop,
inverted. The error handlers compounded it by painting the line `locked` for
any failure - a 429 session cap, a network error - so the button could flip
into resume mode while the server's lockout was clear.

Now `lockoutLine()` is the one writer of the client's lockout state, it writes
the label, and the click handler reads the action from that same state. A
failed request is not an e-stop: it is shown as a message and the line is
re-read from `/api/safety`.

The UI is now the React SPA (`frontend/src`), where the same rule reads: the Sim
tab holds ONE `lockout` state, every write of it is a server answer (the
`/api/safety` read, the `/api/safety/<action>` reply, a telemetry frame that
carried one), both the button's label and the click's action are read from that
state, and a failed request becomes a message, never a lockout. These cells read
`SimTab.tsx` for that shape, so the rule holds without a browser in the suite.
"""

from __future__ import annotations

import pathlib
import re

FRONTEND_SRC = pathlib.Path(__file__).parent.parent / "strands_robots" / "dashboard" / "frontend" / "src"
SIM_TAB = FRONTEND_SRC / "components" / "SimTab.tsx"


def _function_body(source: str, header: str) -> str:
    """The text from ``header`` to the end of its braced block, braces balanced."""
    start = source.index(header)
    depth = 0
    for i in range(source.index("{", start), len(source)):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            return source[start : i + 1]
    raise AssertionError(f"unbalanced braces after {header!r}")


class TestTheEstopButtonReadsOneState:
    def test_one_lockout_state_and_every_write_is_a_server_answer(self) -> None:
        """The tab keeps one ``lockout`` state, and nothing writes it but what the server said."""
        source = SIM_TAB.read_text(encoding="utf-8")
        states = re.findall(r"useState<SimLockout \| null>", source)
        assert len(states) == 1, f"the lockout is held in {len(states)} states, so two can disagree"
        writes = [line.strip() for line in source.splitlines() if "setLockout(" in line]
        assert writes, "nothing writes the lockout, so the button reads a state that never changes"
        for write in writes:
            assert re.search(r"await (api|post)<\{ lockout: SimLockout \}>|onLockout\(|\(m\.lockout\)|\(l\)", write), (
                f"a lockout write that is not a server answer: {write}"
            )

    def test_the_label_and_the_click_read_the_same_state(self) -> None:
        """The label and the action both come from ``lockout.state``, so they cannot invert."""
        source = SIM_TAB.read_text(encoding="utf-8")
        assert re.search(r"lockout\?\.state === 'locked' \? 'RESUME' : 'E-STOP", source), (
            "the button's label is not read from the recorded lockout"
        )
        handler = _function_body(source, "const toggleEstop = async () =>")
        assert re.search(r"const action = lockout\?\.state === 'locked' \? 'resume' : 'estop'", handler), (
            "the click does not choose its action from the recorded lockout"
        )
        assert "className" not in handler, "the click reads a painted class, which the label does not follow"

    def test_no_handler_fabricates_a_lockout(self) -> None:
        """A failed request is reported as a message; only a server answer becomes the lockout."""
        source = SIM_TAB.read_text(encoding="utf-8")
        fabricated = [
            f"{n}: {line.strip()}"
            for n, line in enumerate(source.splitlines(), 1)
            if re.search(r"setLockout\(\s*\{\s*state\s*:", line)
        ]
        assert fabricated == [], f"a lockout the server never reported is written: {fabricated}"
        handler = _function_body(source, "const toggleEstop = async () =>")
        assert re.search(r"catch \(\w+\) \{ await failed\(", handler), "a failed e-stop request's reason is not shown"
        assert "setLockout" not in handler.split("catch")[1], "a failed request is painted as a lockout"
