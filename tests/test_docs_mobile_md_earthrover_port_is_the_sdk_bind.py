"""``docs/robots/mobile.md`` teaches the EarthRover SDK on the port the SDK binds.

The earth-rovers-sdk has one bind (``hypercorn main:app --bind 0.0.0.0:8000``)
and :data:`~strands_robots.drivers.earthrover.DEFAULT_SDK_URL` dials it. The
page's hardware fence and its ``port`` table spelled every address on ``:8001``,
so a reader who copied ``port="http://10.0.0.9:8001"`` for a stock SDK got the
same "unreachable" report the old default gave. Every port literal on the page
is graded against the driver's own default here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import urlsplit

import strands_robots
from strands_robots.drivers.earthrover import DEFAULT_SDK_URL

_PAGE = Path(strands_robots.__file__).resolve().parent.parent / "docs" / "robots" / "mobile.md"
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PORT_LITERAL = re.compile(r":(\d{2,5})(?=[/\s`\"',|)]|$)")


def _fence_earthrover_ports() -> list[int]:
    ports: list[int] = []
    for fence in _PYTHON_FENCE.findall(_PAGE.read_text(encoding="utf-8")):
        try:
            tree = ast.parse(fence)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Robot"):
                continue
            if not (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "earthrover"):
                continue
            for kw in node.keywords:
                if kw.arg == "port" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    ports.append(urlsplit(kw.value.value).port or -1)
    return ports


def test_earthrover_fence_port_is_the_sdk_bind() -> None:
    ports = _fence_earthrover_ports()
    assert ports, "mobile.md has no Robot('earthrover', ..., port=...) fence"
    assert set(ports) == {urlsplit(DEFAULT_SDK_URL).port}


def test_every_port_literal_on_the_page_is_the_sdk_bind() -> None:
    sdk_port = str(urlsplit(DEFAULT_SDK_URL).port)
    found = _PORT_LITERAL.findall(_PAGE.read_text(encoding="utf-8"))
    assert found, "mobile.md spells no host:port address"
    assert set(found) == {sdk_port}
