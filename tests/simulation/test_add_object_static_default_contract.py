"""``add_object``'s declared ``is_static`` default is the one a backend delivers.

The base contract declared ``is_static: bool = False``. The default backend does
not deliver that value, and cannot: ``shape="plane"`` is infinite and cannot
carry a dynamic mass, so MuJoCo distinguishes "the caller did not specify" from
"the caller said ``False``" and takes ``bool | None = None``. Measured against a
real compiled model, one ``add_object`` per case:

* ``add_object(shape="plane")`` returned ``status="success"`` with the body
  **static** - the opposite of the declared ``False``.
* ``add_object(shape="plane", is_static=False)`` - restating the value the base
  contract declared as the default - returned ``status="error"``.

So a caller reading the base signature and writing out its declared default got
a hard refusal for the one shape whose entire purpose is being static, and a
caller who omitted it got a value the signature said was impossible. Three other
surfaces already stated the real contract while the signature did not: MuJoCo's
own ``add_object`` records it in a comment ("Explicit is_static=False for a plane
is an error; None or True both resolve to True"), :meth:`SimEngine.describe`
already advertised ``is_static=None``, and the README's footgun list already says
"passing ``is_static=False`` is a hard error". Only the signatures disagreed.

``None`` is therefore the declared default, and the ABC states what it means: a
backend MAY derive the answer from ``shape``, and one with no shape-derived rule
resolves ``None`` to ``False``. Neither Newton nor Isaac special-cases any shape
(neither supports ``shape="plane"`` at all), so both resolve it to ``False``
before their first read of it and no behaviour changes on any backend - the point
is that the declared contract is now the delivered one, and expressible.

``TestRestatingTheDeclaredDefaultIsANoOp`` is the behaviour pin, read from
:func:`inspect.signature` rather than hardcoded, so it grades the contract and
not a literal. ``TestTheRefusalSurvives`` pins what must NOT change: a plane
handed an explicit ``is_static=False`` is still refused, with the remedy named -
silently overriding it would make the declared default deliverable the wrong
way, which is the obvious wrong fix. ``TestNoDeclaredDefaultDrifts`` keeps every
surface that publishes a default agreeing with the signature it describes.

These tests need neither ``newton``/``warp`` nor ``isaacsim``: both classes
import without their runtimes, so their signatures are readable everywhere. Only
the MuJoCo behaviour class compiles a model.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine


@pytest.fixture
def sim():
    """A live MuJoCo world; each test gets its own so names stay free."""
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    engine = Simulation(tool_name="devx_static_default", mesh=False)
    engine.create_world()
    try:
        yield engine
    finally:
        engine.cleanup(policy_stop_timeout=0.5)


def _declared_static_default() -> Any:
    """The value the base contract publishes as ``is_static``'s default."""
    return inspect.signature(SimEngine.add_object).parameters["is_static"].default


def _backend_classes() -> dict[str, type]:
    """Every concrete ``add_object`` implementation, runtime or not."""
    from strands_robots.simulation.isaac.simulation import IsaacSimulation
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return {"mujoco": MuJoCoSimEngine, "newton": NewtonSimEngine, "isaac": IsaacSimulation}


class TestTheDeclaredDefaultIsExpressibleEverywhere:
    """A caller must be able to hand any backend the base-declared default."""

    def test_the_base_declares_the_unspecified_sentinel(self):
        """``None`` is the only value that can mean "the caller did not say".

        A ``bool`` default cannot: ``False`` is also what a caller passes to ask
        for a dynamic body, so a backend reading it has no way to tell the two
        apart and cannot derive the answer from ``shape`` without overriding an
        explicit request.
        """
        assert _declared_static_default() is None

    @pytest.mark.parametrize("label", ["mujoco", "newton", "isaac"])
    def test_every_backend_accepts_the_declared_default(self, label):
        """Liskov, at the value level rather than the type level.

        A backend annotating ``is_static: bool`` refuses the base-declared
        default under a type checker even where it happens to behave correctly
        at runtime, so the contract would be undeclarable rather than merely
        undocumented.
        """
        cls = _backend_classes()[label]
        param = inspect.signature(cls.add_object).parameters["is_static"]
        assert param.default is _declared_static_default(), (
            f"{label}.add_object declares is_static={param.default!r} while the base contract "
            f"declares {_declared_static_default()!r}; a caller omitting it gets a different "
            f"value depending on which backend answers"
        )
        assert "None" in str(param.annotation), (
            f"{label}.add_object annotates is_static as {param.annotation!r}, which excludes the "
            f"base-declared default {_declared_static_default()!r}"
        )


class TestRestatingTheDeclaredDefaultIsANoOp:
    """The headline: writing out the declared default must not change anything.

    Pre-fix the base declared ``False`` and, for ``shape="plane"``, omitting the
    argument succeeded while passing that declared ``False`` was refused - the
    two spellings of "I did not choose" disagreed.
    """

    @pytest.mark.parametrize("shape", ["plane", "box", "sphere", "cylinder"])
    def test_omitting_and_restating_the_default_agree(self, sim, shape):
        declared = _declared_static_default()

        omitted = sim.add_object(name=f"{shape}_omitted", shape=shape, position=[0.0, 0.0, 0.3])
        restated = sim.add_object(name=f"{shape}_restated", shape=shape, position=[0.6, 0.0, 0.3], is_static=declared)

        assert omitted["status"] == restated["status"], (
            f"shape={shape!r}: omitting is_static gave {omitted['status']!r} but passing the "
            f"base-declared default is_static={declared!r} gave {restated['status']!r} "
            f"({restated['content'][0]['text']})"
        )
        assert omitted["status"] == "success", omitted
        assert sim._world is not None
        assert sim._world.objects[f"{shape}_omitted"].is_static == sim._world.objects[f"{shape}_restated"].is_static

    def test_an_unspecified_plane_is_static_and_an_unspecified_box_is_not(self, sim):
        """The shape-derived rule the sentinel exists to allow, still applied.

        This is the reason the default cannot be ``False``: the delivered value
        genuinely differs by shape when the caller does not choose.
        """
        assert sim.add_object(name="ground", shape="plane", position=[0.0, 0.0, 0.0])["status"] == "success"
        assert sim.add_object(name="crate", shape="box", position=[0.0, 0.0, 0.4])["status"] == "success"
        assert sim._world is not None
        assert sim._world.objects["ground"].is_static is True
        assert sim._world.objects["crate"].is_static is False

    def test_the_resolved_flag_is_a_bool_and_not_the_sentinel(self, sim):
        """Every later reader treats the flag as a bool; the sentinel is resolved.

        ``SimObject.is_static`` is annotated ``bool``, and ``list_objects`` and
        the rebuild both read it directly, so leaving ``None`` on the record
        would push the sentinel into surfaces that never asked about it.
        """
        assert sim.add_object(name="pebble", shape="sphere", position=[0.0, 0.0, 0.4])["status"] == "success"
        assert sim._world is not None
        assert isinstance(sim._world.objects["pebble"].is_static, bool)


class TestTheRefusalSurvives:
    """What must NOT change, and the obvious wrong fix it rules out.

    Making the declared default deliverable by having a plane silently accept
    ``is_static=False`` would satisfy the headline test and lose the refusal:
    the caller would be told a dynamic plane was built and get a static one.
    """

    def test_an_explicit_false_for_a_plane_is_still_refused(self, sim):
        result = sim.add_object(name="bad_ground", shape="plane", position=[0.0, 0.0, 0.0], is_static=False)
        assert result["status"] == "error", result
        assert "bad_ground" not in (sim._world.objects if sim._world else {})

    def test_the_refusal_names_the_value_that_works(self, sim):
        """Parsed back out and applied, so the remedy is pinned, not the wording."""
        result = sim.add_object(name="ground", shape="plane", position=[0.0, 0.0, 0.0], is_static=False)
        text = result["content"][0]["text"]
        assert "is_static=True" in text, text
        assert (
            sim.add_object(name="ground", shape="plane", position=[0.0, 0.0, 0.0], is_static=True)["status"]
            == "success"
        )

    def test_an_explicit_true_still_pins_a_non_plane_shape(self, sim):
        assert (
            sim.add_object(name="table", shape="box", position=[0.0, 0.0, 0.3], is_static=True)["status"] == "success"
        )
        assert sim._world is not None
        assert sim._world.objects["table"].is_static is True


class TestNoDeclaredDefaultDrifts:
    """Structural: a published default must be the signature's default.

    Two surfaces publish defaults for the same parameters - the signature, and
    the ``describe()`` string an agent reads to drive the API without one - and
    nothing compared them. ``describe()`` advertised ``is_static=None`` for the
    whole run in which the signature declared ``False``, so the disagreement was
    visible in the same file and unenforced.
    """

    @staticmethod
    def _advertised(cls: type) -> dict[str, str]:
        """``{method: signature string}`` from the class's own ``describe``."""
        source = inspect.getsourcefile(cls)
        assert source is not None
        tree = ast.parse(Path(source).read_text(encoding="utf-8"))
        out: dict[str, str] = {}
        for klass in ast.walk(tree):
            if not (isinstance(klass, ast.ClassDef) and klass.name == cls.__name__):
                continue
            for node in ast.walk(klass):
                if not (isinstance(node, ast.FunctionDef) and node.name == "describe"):
                    continue
                for sub in ast.walk(node):
                    # {"name": "(...) -> dict"} entries
                    if isinstance(sub, ast.Dict):
                        for key, value in zip(sub.keys, sub.values):
                            if (
                                isinstance(key, ast.Constant)
                                and isinstance(key.value, str)
                                and isinstance(value, ast.Constant)
                                and isinstance(value.value, str)
                                and value.value.lstrip().startswith("(")
                            ):
                                out.setdefault(key.value, value.value)
                    # base["methods"]["name"] = "(...) -> dict"
                    if (
                        isinstance(sub, ast.Assign)
                        and isinstance(sub.value, ast.Constant)
                        and isinstance(sub.value.value, str)
                        and sub.value.value.lstrip().startswith("(")
                    ):
                        for target in sub.targets:
                            if (
                                isinstance(target, ast.Subscript)
                                and isinstance(target.slice, ast.Constant)
                                and isinstance(target.slice.value, str)
                            ):
                                out.setdefault(target.slice.value, sub.value.value)
        return out

    @staticmethod
    def _parameters(signature: str) -> list[str]:
        """Top-level comma split of an advertised parameter list."""
        body = signature[signature.index("(") + 1 :]
        depth, current, parts = 0, "", []
        for char in body:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                if depth == 0:
                    break
                depth -= 1
            if char == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += char
        if current.strip():
            parts.append(current)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _name_and_default(parameter: str) -> tuple[str | None, str | None]:
        """``('name', 'default source')`` for one advertised parameter."""
        if parameter in ("*", "/") or parameter.startswith("*"):
            return None, None
        depth, split_at = 0, -1
        for index, char in enumerate(parameter):
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif (
                char == "="
                and depth == 0
                and (index == 0 or parameter[index - 1] not in "!<>=")
                and (index + 1 >= len(parameter) or parameter[index + 1] != "=")
            ):
                split_at = index
        if split_at < 0:
            return parameter.split(":")[0].strip(), None
        return parameter[:split_at].split(":")[0].strip(), parameter[split_at + 1 :].strip()

    @pytest.mark.parametrize("label", ["base", "mujoco", "newton", "isaac"])
    def test_describe_advertises_the_default_the_signature_declares(self, label):
        """Graded only where ``describe()`` states a literal.

        A parameter documented by its DOMAIN rather than its default
        (``state='open'|'close'``, ``position=[x,y,z]``) is not a default claim,
        and reading it as one would report the whole documented-domain idiom.
        """
        cls = {"base": SimEngine, **_backend_classes()}[label]
        offenders = []
        graded = 0
        for method_name, signature in sorted(self._advertised(cls).items()):
            method = getattr(cls, method_name, None)
            if not callable(method):
                continue
            try:
                real = inspect.signature(method)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            for parameter in self._parameters(signature):
                name, advertised = self._name_and_default(parameter)
                if name is None or advertised is None or name not in real.parameters:
                    continue
                try:
                    advertised_value = ast.literal_eval(advertised)
                except (ValueError, SyntaxError):
                    continue  # a domain or a shape, not a default
                graded += 1
                declared = real.parameters[name].default
                if declared is inspect.Parameter.empty or declared != advertised_value:
                    offenders.append(
                        f"{method_name}.{name}: describe() says {advertised!r}, signature says {declared!r}"
                    )
        assert graded >= 10, (
            f"premise: only {graded} literal defaults graded on {label}; the extractor stopped reaching describe()"
        )
        assert not offenders, f"{label}.describe() publishes a default the signature does not declare: " + "; ".join(
            offenders
        )

    @pytest.mark.parametrize("label", ["mujoco", "newton", "isaac"])
    def test_no_override_narrows_a_base_declared_default(self, label):
        """A shared parameter's default must not depend on which backend answers.

        Making a base-required parameter optional is a widening and stays
        allowed; changing a value the base already declared is not, because the
        base signature is the documented contract a caller reads.
        """
        cls = _backend_classes()[label]
        offenders = []
        graded = 0
        for method_name, base_method in vars(SimEngine).items():
            if not callable(base_method) or method_name.startswith("__"):
                continue
            override = getattr(cls, method_name, None)
            if override is None or override is base_method:
                continue
            try:
                base_signature = inspect.signature(base_method)
                override_signature = inspect.signature(override)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            for name, base_parameter in base_signature.parameters.items():
                if base_parameter.default is inspect.Parameter.empty:
                    continue  # required in the base; an override may widen it
                override_parameter = override_signature.parameters.get(name)
                if override_parameter is None or override_parameter.default is inspect.Parameter.empty:
                    continue
                graded += 1
                if override_parameter.default != base_parameter.default:
                    offenders.append(
                        f"{method_name}({name}=): base declares {base_parameter.default!r}, "
                        f"{label} declares {override_parameter.default!r}"
                    )
        assert graded >= 10, f"premise: only {graded} shared defaults graded on {label}"
        assert not offenders, "; ".join(offenders)
