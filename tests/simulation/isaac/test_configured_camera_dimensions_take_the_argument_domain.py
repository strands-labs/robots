# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A configured camera resolution takes the same domain as the render argument.

One resolution has two owners: the ``width`` / ``height`` arguments of
``add_camera`` and of the render family, and the
:attr:`~strands_robots.simulation.isaac.config.IsaacConfig.camera_width` /
``camera_height`` fields those arguments fall back to when omitted. Both call
sites grade their argument on the shared pixel floor
:func:`~strands_robots.utils.positive_count_error`, and ``_render_frame``'s own
comment says the two owners agree: *"Same shared pixel floor ``add_camera``
applies, so the config-time and call-time domains agree."* The field was graded
by a hand-rolled ``if self.camera_width < 1 or self.camera_height < 1`` instead,
so the verdict depended on which spelling the caller happened to use. Measured
over thirteen values, eight disagreed:

* ``True`` -- the exact value the shared domain's docstring says a bare
  ``value < 1`` test lets through as "a silent count of 1" -- plus ``640.0``,
  ``640.5``, ``nan`` and ``inf`` were **accepted and stored**, then spent as
  ``np.zeros((h, w, 3))``: ``render()`` raised ``TypeError: 'float' object
  cannot be interpreted as an integer`` (``an integer is required`` for the
  boolean) from a method whose documented contract is a ``{"status", "content"}``
  dict, with the ``np.zeros`` outside any try block. Every one of them is
  refused as an argument, on the same engine, in the same session.
* ``"640"`` and ``None`` raised ``TypeError: '<' not supported between
  instances of 'str' and 'int'`` **from the comparison itself**, so the
  configuration error was reported as an unrelated type error that names
  neither the field nor a usable resolution.
* ``False`` was refused, but the message read ``camera dimensions must be >= 1,
  got Falsex480``.

The fix points the existing gate at the shared owner the two call sites already
read, and grades each field separately so the message names the one to fix. It
is not a restatement of the rule: it removes one.

Nothing here needs Isaac Sim or a GPU. The refusals happen in
``__post_init__``, before any Kit import, and the frame the resolution sizes is
the documented ``headless`` blank frame, reached on the skeleton-via-``__new__``
engine the sibling ``test_render_refuses_a_camera_the_scene_does_not_carry.py``
uses.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation

#: Every spelling a caller can reach through either owner. The verdict must not
#: depend on which one carried it.
SHARED_DOMAIN_VALUES = (
    640,
    1,
    100000,
    0,
    -1,
    True,
    False,
    640.0,
    640.5,
    "640",
    float("nan"),
    float("inf"),
)

#: The one value whose verdict legitimately differs between the two owners, with
#: the reason. ``None`` as the ``width=`` argument is the spelling of "not
#: stated" - it is the parameter's own default, and means "take the configured
#: default" - while ``None`` set on the field is a stated value that is not a
#: pixel count. Membership decides that, not truthiness, so ``0`` stays in the
#: shared roster above and is refused on both sides.
OWNER_SPECIFIC_VALUES: dict[Any, str] = {
    None: (
        "as an argument None is the parameter default and means 'unstated, read the field'; "
        "as a field value it is a stated non-integer and is refused"
    ),
}

#: Values the field used to accept and now refuses, with the reason. The shared
#: domain admits only a true ``int`` because the value is spent directly as an
#: array dimension, and the ``width=`` argument has always refused these - so
#: the field agreeing with it is the point of the change rather than a side
#: effect of it.
NARROWED_VALUES: dict[Any, str] = {
    np.int64(640): (
        "the shared count domain admits only a true int - render(width=np.int64(640)) "
        "has always been refused, so the field now gives that same value one verdict"
    ),
}


def _engine(config: IsaacConfig) -> IsaacSimulation:
    """Skeleton engine carrying only what the headless render path reads."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = config
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._cameras = {}
    engine._prim_registry = []
    engine._cam_out_size = {}
    engine._camera_warmup_steps = 0
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()
    return engine


def _config_verdict(param: str, value: Any) -> str | None:
    """The refusal ``IsaacConfig`` gives for ``param=value``, or ``None``.

    Read through the public constructor, so the pin does not depend on where
    the rule is spelled.
    """
    try:
        IsaacConfig(**{param: value})
    except (ValueError, TypeError) as refused:
        return f"{type(refused).__name__}: {refused}"
    return None


def _argument_verdict(param: str, value: Any) -> str | None:
    """The refusal ``render`` gives for the same value passed as an argument."""
    result = _engine(IsaacConfig()).render(**{param: value})
    if result["status"] == "error":
        return next(block["text"] for block in result["content"] if "text" in block)
    return None


#: The field on the config, and the render argument it is the default for.
OWNER_PAIRS = (("camera_width", "width"), ("camera_height", "height"))


class TestOneDomainForOneResolution:
    """The field and the argument agree on every value a caller can spell."""

    @pytest.mark.parametrize(("field_name", "arg_name"), OWNER_PAIRS)
    @pytest.mark.parametrize("value", SHARED_DOMAIN_VALUES, ids=repr)
    def test_the_two_owners_reach_the_same_verdict(self, value: Any, field_name: str, arg_name: str) -> None:
        """Accept-or-refuse cannot depend on where the caller wrote the value.

        Pre-fix five of these were refused as an argument and stored from the
        field, and two more were reported as a ``TypeError`` from the
        comparison rather than as a configuration error.
        """
        from_field = _config_verdict(field_name, value)
        from_argument = _argument_verdict(arg_name, value)
        assert (from_field is None) == (from_argument is None), (
            f"{value!r} as {arg_name}= is {'refused' if from_argument else 'accepted'} but as "
            f"IsaacConfig.{field_name} is {'refused' if from_field else 'accepted'}: "
            f"field said {from_field!r}, argument said {from_argument!r}"
        )

    @pytest.mark.parametrize(("field_name", "arg_name"), OWNER_PAIRS)
    def test_a_refusal_is_a_value_error_naming_the_field(self, field_name: str, arg_name: str) -> None:
        """The field's refusal is reportable, and says which spelling to fix.

        ``"640"`` came back as ``TypeError: '<' not supported between instances
        of 'str' and 'int'`` pre-fix: the right verdict for the wrong reason,
        naming neither the field nor a usable resolution.
        """
        verdict = _config_verdict(field_name, "640")
        assert verdict is not None
        assert verdict.startswith("ValueError: ")
        assert field_name in verdict
        # The argument's message names the argument, not the field, so a caller
        # is not sent to the config for a value they wrote at the call site.
        argument = _argument_verdict(arg_name, "640")
        assert argument is not None
        assert field_name not in argument
        assert arg_name in argument


class TestAConfiguredResolutionCanAlwaysSizeAFrame:
    """The headline: a config that constructs cannot break the render contract."""

    def test_every_accepted_configuration_renders_a_frame(self) -> None:
        """No stored resolution raises out of ``render``'s result contract.

        ``render`` returns a ``{"status", "content"}`` dict for every failure it
        knows about, and sizes its headless blank frame from these fields with
        no coercion. Pre-fix five accepted values raised ``TypeError`` from
        ``np.zeros`` - outside any try block, so the envelope could not even
        report them.
        """
        rendered = 0
        for value in SHARED_DOMAIN_VALUES:
            if _config_verdict("camera_width", value) is not None:
                continue
            result = _engine(IsaacConfig(camera_width=value, render_mode="headless")).render()  # type: ignore[arg-type]
            assert result["status"] == "success", result
            rendered += 1
        assert rendered >= 3, f"only {rendered} configurations were accepted; the controls are not being exercised"

    @pytest.mark.parametrize("value", [True, 640.0, 640.5, float("nan"), float("inf")], ids=repr)
    def test_a_resolution_numpy_cannot_size_an_array_with_is_refused(self, value: Any) -> None:
        """The five values that reached ``np.zeros`` are refused on construction."""
        assert _config_verdict("camera_width", value) is not None

    def test_every_owner_specific_value_states_why(self) -> None:
        """A value exempt from the parity above needs a written reason.

        So a future exemption is a decision rather than a quietly widened list.
        """
        assert OWNER_SPECIFIC_VALUES
        for value, reason in OWNER_SPECIFIC_VALUES.items():
            assert reason.strip(), f"{value!r} is exempt from resolution parity with no reason"

    def test_every_narrowed_value_states_why(self) -> None:
        """A value this change stops accepting needs a written reason.

        So a future narrowing is a decision rather than a quietly extended list.
        """
        assert NARROWED_VALUES
        for value, reason in NARROWED_VALUES.items():
            assert reason.strip(), f"{value!r} is narrowed with no reason given"
            assert _config_verdict("camera_width", value) is not None
            assert _argument_verdict("width", value) is not None


class TestTheConfigDoesNotRestateTheRule:
    """Structural: one owner for the pixel floor, not a local copy of it."""

    def test_post_init_calls_the_shared_domain(self) -> None:
        """``__post_init__`` reaches the floor by calling it, not by re-deriving it.

        Graded on the call graph rather than the source text, so a comment that
        merely names the domain cannot satisfy it.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(IsaacConfig.__post_init__)))
        called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert "positive_count_error" in called, f"__post_init__ calls {sorted(called)}"


class TestTheResolutionsThatMustKeepWorking:
    """Controls: the fix refuses what cannot be honoured, and no more."""

    def test_the_default_configuration_renders_at_640x480(self) -> None:
        """The out-of-the-box resolution is unchanged."""
        config = IsaacConfig()
        assert (config.camera_width, config.camera_height) == (640, 480)
        result = _engine(config).render()
        assert result["status"] == "success"
        assert "640x480" in result["content"][0]["text"]

    @pytest.mark.parametrize(("width", "height"), [(1, 1), (320, 240), (1920, 1080), (100000, 1)])
    def test_a_positive_integer_pair_is_stored_and_spent_verbatim(self, width: int, height: int) -> None:
        """Any positive-integer resolution still sizes the frame it asked for."""
        result = _engine(IsaacConfig(camera_width=width, camera_height=height)).render()
        assert result["status"] == "success"
        assert f"{width}x{height}" in result["content"][0]["text"]

    def test_an_omitted_render_argument_still_reads_the_configured_default(self) -> None:
        """``None`` means "unstated" on the argument, and takes the field."""
        engine = _engine(IsaacConfig(camera_width=321, camera_height=123))
        assert "321x123" in engine.render(width=None, height=None)["content"][0]["text"]

    def test_a_stated_render_argument_outranks_the_configured_default(self) -> None:
        """An explicit argument still wins over the field."""
        engine = _engine(IsaacConfig(camera_width=321, camera_height=123))
        assert "64x48" in engine.render(width=64, height=48)["content"][0]["text"]

    def test_the_legacy_size_shortcut_still_reaches_the_config(self) -> None:
        """``default_width`` / ``default_height`` coerce, so they still resolve.

        The retired adapter's kwargs map onto these fields through ``int(...)``,
        which produces a true integer - the shortcut is unaffected by the
        stricter field domain.
        """
        sim = IsaacSimulation(default_width=800, default_height=600)
        assert (sim._config.camera_width, sim._config.camera_height) == (800, 600)

    def test_a_camera_size_shortcut_kwarg_still_reaches_the_config(self) -> None:
        """The canonical field names still work as ``IsaacSimulation`` kwargs."""
        sim = IsaacSimulation(camera_width=800, camera_height=600)
        assert (sim._config.camera_width, sim._config.camera_height) == (800, 600)
