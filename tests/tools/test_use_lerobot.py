"""Behavior tests for the ``use_lerobot`` universal LeRobot access tool.

``use_lerobot`` is to ``lerobot`` what ``use_aws`` is to boto3: a single
dispatcher that resolves any dotted path into the lerobot package and either
describes or calls it, with config choices discovered dynamically from
lerobot's own draccus ``ChoiceRegistry`` registries (never hardcoded).

These tests pin the contracts that make the tool trustworthy:

1. Every user-facing ``text`` field is plain ASCII (the project's no-emoji rule).
2. Import resolution distinguishes three failure modes precisely:
   genuinely-missing paths, paths that exist but need an optional dependency,
   and attribute-not-found on a resolved object.
3. The serializer is total -- it never raises on circular references, runaway
   nesting, bytes, numpy scalars/arrays, or objects with a hostile ``__repr__``.
4. Introspection never triggers descriptor/property side effects.
5. Image arrays become real Strands ``image`` content blocks with a sane codec.

All tests are hardware-free and do not require the optional ``lerobot[dataset]``
extra; the missing-dependency path is asserted precisely *because* it is absent.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.use_lerobot as M

# The tool is wrapped by the Strands @tool decorator; call the raw function.
_fn = getattr(M.use_lerobot, "__wrapped__", None) or M.use_lerobot

pytest.importorskip("lerobot", reason="use_lerobot requires the lerobot package")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _texts(result: dict[str, Any]) -> str:
    """Concatenate all content ``text`` fields from a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _assert_ascii(text: str) -> None:
    """Fail if any character is outside the ASCII range."""
    offenders = {hex(ord(c)) for c in text if ord(c) > 127}
    assert not offenders, f"non-ASCII characters in tool output: {offenders}"


def _images(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [item["image"] for item in result.get("content", []) if "image" in item]


# ----------------------------------------------------------------------------
# discovery + registries
# ----------------------------------------------------------------------------
def test_discovery_lists_packages_and_registries() -> None:
    """Discovery enumerates packages and at least the four config registries."""
    result = _fn(module="__discovery__", method="list_modules")
    assert result["status"] == "success"
    text = _texts(result)
    _assert_ascii(text)
    assert "LeRobot API Discovery" in text
    for kind in ("robots", "teleoperators", "cameras", "policies"):
        assert kind in text


def test_registry_listing_is_dynamic_not_hardcoded() -> None:
    """A registry listing reflects lerobot's own registered choices."""
    result = _fn(module="__registry__", method="robots")
    assert result["status"] == "success"
    text = _texts(result)
    _assert_ascii(text)
    assert "registry" in text
    # so100/so101 are stable, long-lived SO-arm choices.
    assert "so100_follower" in text or "so101_follower" in text


def test_unknown_registry_reports_valid_kinds() -> None:
    result = _fn(module="__registry__", method="totally_not_a_registry")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "Valid:" in text
    assert "robots" in text and "policies" in text


def test_empty_registry_method_defaults_to_robots() -> None:
    """An empty registry method falls back to the robots registry."""
    result = _fn(module="__registry__", method="")
    assert result["status"] == "success"
    assert "robots" in _texts(result)


# ----------------------------------------------------------------------------
# import resolution -- the three failure modes
# ----------------------------------------------------------------------------
def test_genuinely_missing_path_is_cannot_resolve() -> None:
    """A path with no lerobot module behind it -> 'Cannot resolve' + a tip."""
    result = _fn(module="does.not.exist", method="foo")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "Cannot resolve" in text
    assert "__discovery__" in text  # actionable tip
    # Must NOT misdirect the user to install an optional extra.
    assert "lerobot[dataset]" not in text


def test_fake_lerobot_submodule_is_cannot_resolve() -> None:
    result = _fn(module="robots.totally_fake_robot", method="x")
    assert result["status"] == "error"
    assert "Cannot resolve" in _texts(result)


def test_missing_optional_dependency_surfaces_real_error() -> None:
    """A real path needing an absent extra surfaces the dependency, not a
    misleading 'cannot resolve'. ``datasets`` is intentionally not installed."""
    if importlib.util.find_spec("datasets") is not None:
        pytest.skip("datasets extra is installed; cannot assert the missing-dep path")
    result = _fn(module="datasets.lerobot_dataset.LeRobotDataset", method="__describe__")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "exists but failed to import" in text
    assert "lerobot[dataset]" in text  # the actionable extra


def test_attribute_not_found_lists_available() -> None:
    """A bad method on a resolved module lists real alternatives."""
    result = _fn(module="policies.factory", method="definitely_not_here")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "not found" in text
    assert "Available:" in text


# ----------------------------------------------------------------------------
# calling + signatures
# ----------------------------------------------------------------------------
def test_call_with_params_succeeds() -> None:
    result = _fn(
        module="policies.factory",
        method="get_policy_class",
        parameters={"name": "act"},
    )
    assert result["status"] == "success"
    _assert_ascii(_texts(result))


def test_missing_required_arg_reports_signature() -> None:
    """A TypeError from a bad call surfaces the expected signature."""
    result = _fn(module="policies.factory", method="get_policy_class")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "TypeError" in text
    assert "name" in text  # the missing parameter is named


def test_read_constant_attribute() -> None:
    result = _fn(module="utils.constants", method="HF_LEROBOT_CALIBRATION")
    assert result["status"] == "success"
    _assert_ascii(_texts(result))


# ----------------------------------------------------------------------------
# introspection -- describe without side effects
# ----------------------------------------------------------------------------
def test_describe_separates_properties_from_methods() -> None:
    """``__describe__`` classifies properties, class methods, and instance
    methods distinctly, using static lookup (no descriptor side effects)."""
    import json

    result = _fn(module="cameras.opencv.OpenCVCamera", method="__describe__")
    assert result["status"] == "success"
    text = _texts(result)
    _assert_ascii(text)
    info = json.loads(text.split("\n", 1)[1])
    # is_connected is a property on the camera class, not a callable method.
    assert "is_connected" in info.get("properties", [])
    assert "is_connected" not in info.get("methods", [])
    # find_cameras is a classmethod; it should not be double-listed as a method.
    assert "find_cameras" in info.get("class_methods", [])
    assert "find_cameras" not in info.get("methods", [])


def test_describe_class_without_introspectable_init_omits_params() -> None:
    """A class whose ``__init__`` exposes no introspectable signature (as some
    C-extension-backed lerobot types do) is still described: the method list is
    populated and ``init_params`` is simply omitted rather than the whole
    describe call raising."""

    class NoInitSignature:
        # Accessing the class attribute yields the property descriptor itself,
        # which ``inspect.signature`` cannot introspect (raises TypeError) -- the
        # same failure mode as a builtin ``__init__`` slot wrapper.
        __init__ = property(lambda self: None)

    info = M._describe_object(NoInitSignature)
    assert info["name"] == "NoInitSignature"
    assert "methods" in info
    assert "init_params" not in info


# ----------------------------------------------------------------------------
# serializer -- totality under hostile input
# ----------------------------------------------------------------------------
def test_serializer_handles_circular_dict() -> None:
    d: dict[str, Any] = {}
    d["self"] = d
    out = M._serialize_result(d)
    assert "circular ref" in out


def test_serializer_handles_circular_list() -> None:
    lst: list[Any] = []
    lst.append(lst)
    assert "circular ref" in M._serialize_result(lst)


def test_serializer_handles_bytes_structurally() -> None:
    out = M._serialize_value(b"\x00\x01\x02hello")
    assert out["__bytes__"] is True
    assert out["length"] == 8
    assert out["preview_hex"].startswith("000102")


def test_serializer_handles_numpy_scalars() -> None:
    out = M._serialize_value({"i": np.int64(5), "f": np.float32(1.5)})
    assert out["i"] == 5
    assert out["f"] == pytest.approx(1.5)


def test_serializer_summarizes_large_arrays_structurally() -> None:
    out = M._serialize_value(np.zeros((2, 3, 64, 64)))
    assert isinstance(out, dict)
    assert out["__ndarray__"] is True
    assert out["shape"] == [2, 3, 64, 64]
    # A 4D tensor is not pixel-dumped as text.
    assert "values" not in out


def test_serializer_survives_hostile_repr() -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("boom")

    # Must not raise -- the tool stays alive even on pathological objects.
    out = M._serialize_result({"x": Hostile()})
    assert isinstance(out, str)


def test_serializer_falls_back_when_dataclass_expansion_fails() -> None:
    """A dataclass whose fields cannot be expanded (``asdict`` raises while
    deep-copying a hostile field) degrades to a repr string instead of
    propagating the error -- the serializer stays total on config-like objects."""
    import dataclasses

    class Uncopyable:
        def __deepcopy__(self, memo: dict) -> Uncopyable:
            raise RuntimeError("cannot copy")

    @dataclasses.dataclass
    class Config:
        field: object

    out = M._serialize_value(Config(field=Uncopyable()))
    # asdict blew up, so no structured __dataclass__ wrapper -- but no raise either.
    assert isinstance(out, str)
    assert "Config" in out


def test_serializer_depth_guard() -> None:
    """Deeply nested structures terminate rather than blowing the stack."""
    node: dict[str, Any] = {}
    cur = node
    for _ in range(200):
        nxt: dict[str, Any] = {}
        cur["next"] = nxt
        cur = nxt
    out = M._serialize_result(node)
    assert "max depth exceeded" in out


# ----------------------------------------------------------------------------
# image detection + encoding
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,expected",
    [
        ((100, 100, 3), True),  # RGB
        ((100, 100), True),  # grayscale
        ((100, 100, 4), True),  # RGBA
        ((1, 1, 3), False),  # too small
        ((10, 10, 2), False),  # invalid channel count
        ((2, 3, 64, 64), False),  # 4D tensor, not an image
    ],
)
def test_image_detection_heuristic(shape: tuple[int, ...], expected: bool) -> None:
    arr = np.zeros(shape, dtype=np.uint8)
    assert M._is_image_array(arr) is expected


def test_large_rgb_frame_encodes_as_jpeg() -> None:
    """Large opaque frames use JPEG (compact in-context)."""
    pytest.importorskip("cv2")
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    block = M._array_to_image_block(frame)
    assert block is not None
    assert block["image"]["format"] == "jpeg"
    assert block["image"]["source"]["bytes"]


def test_small_frame_and_alpha_use_png() -> None:
    """Small frames and anything with alpha stay PNG (lossless / alpha-safe)."""
    pytest.importorskip("cv2")
    small = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    rgba = (np.random.rand(480, 640, 4) * 255).astype(np.uint8)
    small_block = M._array_to_image_block(small)
    rgba_block = M._array_to_image_block(rgba)
    assert small_block is not None
    assert rgba_block is not None
    assert small_block["image"]["format"] == "png"
    assert rgba_block["image"]["format"] == "png"


def test_collect_images_finds_frames_in_dict() -> None:
    """Images nested one level inside a dict (camera_name -> frame) are found."""
    pytest.importorskip("cv2")
    frames = {
        "front": (np.random.rand(64, 64, 3) * 255).astype(np.uint8),
        "wrist": (np.random.rand(64, 64, 3) * 255).astype(np.uint8),
        "meta": "not an image",
    }
    blocks: list[dict[str, Any]] = []
    M._collect_images(frames, blocks)
    assert len(blocks) == 2


# ----------------------------------------------------------------------------
# deep-import cache
# ----------------------------------------------------------------------------
def test_deep_import_is_cached() -> None:
    """A second deep import of the same package is a no-op (cache hit)."""
    M._DEEP_IMPORTED.discard("lerobot.policies")
    M._deep_import("lerobot.policies")
    assert "lerobot.policies" in M._DEEP_IMPORTED
    # Idempotent: calling again must not raise and the marker persists.
    M._deep_import("lerobot.policies")
    assert "lerobot.policies" in M._DEEP_IMPORTED


# ----------------------------------------------------------------------------
# import resolution -- resolver-internal error classification
# ----------------------------------------------------------------------------
def test_empty_module_path_cannot_resolve() -> None:
    """A blank dotted path resolves to nothing rather than crashing."""
    with pytest.raises(M.LeRobotResolveError, match="Cannot resolve"):
        M._import_from_lerobot("")


def test_third_party_import_error_surfaces_real_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a lerobot submodule fails to import because a *third-party* package
    is missing, the resolver remembers and surfaces that real ImportError
    (carried on ``real_error``) instead of a misleading 'cannot resolve'."""
    real = importlib.import_module

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "lerobot.synthetic_thirdparty_gap":
            raise ImportError("No module named 'absent_extra'", name="absent_extra")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(M.LeRobotResolveError) as exc:
        M._import_from_lerobot("synthetic_thirdparty_gap")
    assert "failed to import" in str(exc.value)
    assert exc.value.real_error is not None


def test_non_import_error_during_import_surfaces_real_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-ImportError raised at module import time (e.g. a RuntimeError in a
    module's top-level code) is also captured as the real error, not masked."""
    real = importlib.import_module

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "lerobot.synthetic_runtime_boom":
            raise RuntimeError("import-time boom")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(M.LeRobotResolveError) as exc:
        M._import_from_lerobot("synthetic_runtime_boom")
    assert "failed to import" in str(exc.value)
    assert isinstance(exc.value.real_error, RuntimeError)


# ----------------------------------------------------------------------------
# introspection -- describe across object kinds
# ----------------------------------------------------------------------------
def test_describe_plain_function_lists_params_and_doc() -> None:
    """Describing a function reports its parameter defaults and docstring."""

    def sample(a: int, b: int = 3) -> int:
        """sample docstring"""
        return a + b

    info = M._describe_object(sample)
    assert info["type"] == "function"
    assert info["params"]["a"]["default"] == "REQUIRED"
    assert info["params"]["b"]["default"] == "3"
    assert info["doc"] == "sample docstring"


def test_describe_module_lists_public_names() -> None:
    """Describing a module enumerates its public names (and skips dunders)."""
    import json as json_mod

    info = M._describe_object(json_mod)
    assert "dumps" in info["public_names"]
    assert all(not n.startswith("_") for n in info["public_names"])


def test_describe_plain_value_uses_value_branch() -> None:
    """A non-class/callable/module value is described via its ``value`` field."""
    info = M._describe_object(42)
    assert info["type"] == "int"
    assert info["value"] == "42"


def test_describe_builtin_callable_skips_uninspectable_signature() -> None:
    """A C-extension callable (``min``) whose signature is not introspectable
    still yields a describe dict (with its docstring) -- no crash, no params
    map. This pins the (ValueError, TypeError) carve-out in the callable
    branch so signature-less builtins describe gracefully."""
    info = M._describe_object(min)
    assert info["type"] == "builtin_function_or_method"
    assert info["doc"]  # docstring still surfaced
    # min exposes no Python signature; the params map must be absent.
    assert "params" not in info


# ----------------------------------------------------------------------------
# image encoding -- normalization + codec selection
# ----------------------------------------------------------------------------
def test_float_frame_normalized_and_encoded() -> None:
    """Float images in [0,1] and in [0,255] both normalize to uint8 and encode."""
    pytest.importorskip("cv2")
    unit = np.random.rand(400, 400, 3).astype(np.float32)  # max <= 1.0
    scaled = (np.random.rand(400, 400, 3) * 200).astype(np.float64)  # max > 1.0
    assert M._array_to_image_block(unit) is not None
    assert M._array_to_image_block(scaled) is not None


def test_grayscale_frame_encodes() -> None:
    """A 2D grayscale frame encodes without a channel conversion."""
    pytest.importorskip("cv2")
    gray = (np.random.rand(400, 400) * 255).astype(np.uint8)
    block = M._array_to_image_block(gray)
    assert block is not None
    assert block["image"]["source"]["bytes"]


def test_integer_frame_clipped_not_scaled_and_round_trips() -> None:
    """Non-uint8 integer frames are clipped into [0, 255], not magnitude-scaled.

    Float frames are rescaled by their magnitude (a [0, 1] image becomes
    [0, 255]), but a plain integer buffer is assumed to already be in pixel
    units, so out-of-range values must saturate rather than wrap or rescale.
    The encoded block must round-trip back through cv2 to a valid uint8 image
    of the same spatial shape.
    """
    cv2 = pytest.importorskip("cv2")
    # int16 RGB frame with values deliberately outside [0, 255]. Per-pixel
    # values are channel-symmetric so the RGB->BGR swap does not reorder them.
    frame = np.full((8, 8, 3), 4095, dtype=np.int16)
    frame[0, 0] = -50  # must saturate to 0, not wrap
    frame[1, 1] = 200  # in range -> passes through unchanged
    block = M._array_to_image_block(frame)
    assert block is not None
    assert block["image"]["format"] == "png"  # small frame -> lossless PNG
    raw = block["image"]["source"]["bytes"]
    assert raw
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.dtype == np.uint8
    assert decoded.shape == (8, 8, 3)
    # Clip semantics (PNG is lossless): -50 saturates to 0, 4095 to 255, and the
    # in-range 200 survives -- a magnitude rescale would have crushed 200 instead.
    assert decoded.min() == 0
    assert decoded.max() == 255
    assert 200 in decoded


def test_collect_images_finds_frames_in_list() -> None:
    """Images one level inside a list are collected; non-images are ignored."""
    pytest.importorskip("cv2")
    items = [
        (np.random.rand(64, 64, 3) * 255).astype(np.uint8),
        "not an image",
        (np.random.rand(64, 64, 3) * 255).astype(np.uint8),
    ]
    blocks: list[dict[str, Any]] = []
    M._collect_images(items, blocks)
    assert len(blocks) == 2


# ----------------------------------------------------------------------------
# serializer -- remaining value kinds
# ----------------------------------------------------------------------------
def test_serializer_passes_through_scalars_and_none() -> None:
    assert M._serialize_value(None) is None
    assert M._serialize_value(True) is True
    assert M._serialize_value(7) == 7


def test_serializer_caps_long_string() -> None:
    """A string longer than the safety cap is truncated with a remainder note."""
    out = M._serialize_value("a" * (M._MAX_STR + 10))
    assert out.endswith("[+10 chars]")
    assert len(out) < M._MAX_STR + 100


def test_serializer_marks_image_arrays_not_pixel_dumped() -> None:
    """An image-shaped array is summarized as an image, never dumped as pixels."""
    out = M._serialize_value(np.zeros((64, 64, 3), dtype=np.uint8))
    assert isinstance(out, dict)
    assert out["__ndarray__"] is True
    assert out["is_image"] is True
    assert "values" not in out


def test_serializer_inlines_small_array_values() -> None:
    """A tiny (<=64 element) non-image array inlines its values."""
    out = M._serialize_value(np.array([1, 2, 3]))
    assert out["values"] == [1, 2, 3]


def test_serializer_truncates_large_list_and_dict() -> None:
    """Oversized lists and dicts are truncated with a remainder marker."""
    big_list = M._serialize_value(list(range(M._MAX_LIST_ITEMS + 5)))
    assert "more items" in big_list[-1]
    big_dict = M._serialize_value({str(i): i for i in range(M._MAX_DICT_ITEMS + 5)})
    assert "more keys" in big_dict["__truncated__"]


def test_serializer_expands_dataclass_fields() -> None:
    """A dataclass instance serializes to a tagged dict of its fields."""
    from dataclasses import dataclass

    @dataclass
    class Cfg:
        x: int = 1
        y: str = "z"

    out = M._serialize_value(Cfg())
    assert out["__dataclass__"] == "Cfg"
    assert out["x"] == 1 and out["y"] == "z"


# ----------------------------------------------------------------------------
# dispatch -- callable results, label logging, error funnels
# ----------------------------------------------------------------------------
def test_dispatch_non_callable_target_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolving to a non-callable attribute returns its serialized value."""
    monkeypatch.setattr(M, "_import_from_lerobot", lambda path: 12345)
    result = _fn(module="x.y", method="")
    assert result["status"] == "success"
    assert "12345" in _texts(result)


def test_dispatch_attaches_image_blocks_and_logs_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """A callable returning frames yields image content blocks; a label is
    accepted (and logged) without changing the success contract."""
    pytest.importorskip("cv2")

    def resolver(path: str) -> Any:
        def call() -> dict[str, Any]:
            return {"front": (np.random.rand(480, 640, 3) * 255).astype(np.uint8)}

        return call

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    result = _fn(module="x.y", method="", label="grab a frame")
    assert result["status"] == "success"
    assert len(_images(result)) == 1
    assert "image block(s) attached" in _texts(result)


def test_dispatch_generic_exception_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Type/Import error from the called target funnels to a clean error
    result naming the exception type, never escaping past the dispatcher."""

    def resolver(path: str) -> Any:
        def call() -> Any:
            raise ValueError("boom inside")

        return call

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    result = _fn(module="x.y", method="")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "ValueError" in text and "boom inside" in text


def test_dispatch_import_error_suggests_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ImportError raised while calling the target maps to an install hint."""

    def resolver(path: str) -> Any:
        def call() -> Any:
            raise ImportError("optional dep gone")

        return call

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    result = _fn(module="x.y", method="")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "Import error" in text
    assert "pip install lerobot" in text


def test_dispatch_missing_optional_dependency_suggests_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolve that fails because a path exists but an optional dependency is
    missing must return an actionable "install the extra" hint, not a dead-end.

    ``_import_from_lerobot`` classifies this case by carrying the underlying
    third-party ImportError on ``LeRobotResolveError.real_error``. The
    dispatcher branches on that marker: when present it tells the agent which
    ``lerobot[...]`` extra to install (distinct from the genuinely-missing-path
    branch, which points at ``__discovery__``). This pins the exact user-facing
    message so the two failure modes never collapse into one generic error.
    """

    def resolver(path: str) -> Any:
        raise M.LeRobotResolveError(
            "'datasets.lerobot_dataset' exists but failed to import: No module named 'datasets'",
            real_error=ImportError("No module named 'datasets'"),
        )

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    result = _fn(module="datasets.lerobot_dataset", method="LeRobotDataset")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    # Carries the real import failure and an actionable extra to install ...
    assert "exists but failed to import" in text
    assert "optional dependency is missing" in text
    assert "lerobot[dataset]" in text
    # ... and does NOT collapse into the genuinely-missing __discovery__ tip.
    assert "__discovery__" not in text


def test_dispatch_genuinely_missing_path_points_at_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolve that fails with no underlying dependency error (the path simply
    does not exist) must point the agent at ``__discovery__`` -- and must NOT
    emit the misleading "install an extra" hint reserved for the missing-dep
    branch. This is the sibling contract to the missing-dependency case.
    """

    def resolver(path: str) -> Any:
        raise M.LeRobotResolveError("Cannot resolve 'does.not.exist' in lerobot")

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    result = _fn(module="does.not.exist", method="")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "Cannot resolve" in text
    assert "__discovery__" in text
    assert "optional dependency is missing" not in text


# ----------------------------------------------------------------------------
# security guards: the dispatcher must refuse dangerous calls
# ----------------------------------------------------------------------------
# use_lerobot exposes the whole lerobot package to an LLM-driven agent. Two
# allowlists keep prompt-injected calls from spawning training subprocesses or
# pushing to the Hugging Face Hub: a blocked-module-prefix check and a
# blocked-method-name check. Both run only after the target resolves to a
# callable, so they are asserted with a resolver stub (no lerobot internals,
# no hardware). A regression that drops either guard would let a call under
# "lerobot.scripts" or a "push_to_hub" method through silently.
def test_blocked_module_prefix_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A callable under a restricted module namespace (lerobot.scripts spawns
    training subprocesses) is refused before it is ever called."""
    called = {"n": 0}

    def resolver(path: str) -> Any:
        def spawn(**kwargs: Any) -> None:
            called["n"] += 1  # must never run

        return spawn

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    # method="" so the resolved callable is the dispatch target directly.
    result = _fn(module="scripts.lerobot_train", method="")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "Blocked" in text
    assert "lerobot.scripts" in text
    assert called["n"] == 0, "blocked target must not be invoked"


def test_blocked_module_prefix_check_respects_explicit_lerobot_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefix check normalizes both bare ('scripts...') and fully-qualified
    ('lerobot.scripts...') module paths, so a caller cannot bypass it by
    spelling out the 'lerobot.' prefix themselves."""
    monkeypatch.setattr(M, "_import_from_lerobot", lambda path: lambda **kw: None)
    result = _fn(module="lerobot.common.datasets.push.foo", method="")
    assert result["status"] == "error"
    assert "Blocked" in _texts(result)


def test_blocked_method_name_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A method whose name has a dangerous side effect (push_to_hub uploads to
    the Hub) is refused even on an otherwise-safe module."""
    called = {"n": 0}

    def resolver(path: str) -> Any:
        class Dataset:
            def push_to_hub(self) -> None:
                called["n"] += 1  # must never run

        return Dataset()

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    result = _fn(module="datasets.lerobot_dataset.LeRobotDataset", method="push_to_hub")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "Blocked" in text
    assert "push_to_hub" in text
    assert called["n"] == 0, "blocked method must not be invoked"


def test_safe_method_on_unblocked_module_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guards are an allowlist, not a blanket denial: an ordinary method on
    an unrestricted module still dispatches and returns its result."""

    def resolver(path: str) -> Any:
        class Thing:
            def num_frames(self) -> int:
                return 42

        return Thing()

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    result = _fn(module="datasets.lerobot_dataset.LeRobotDataset", method="num_frames")
    assert result["status"] == "success"
    assert "42" in _texts(result)


# ----------------------------------------------------------------------------
# TypeError signature hint: bad kwargs get an actionable expected-signature
# ----------------------------------------------------------------------------
def test_type_error_reports_expected_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong-kwargs TypeError surfaces with the introspected parameter list so
    the agent can self-correct (the dispatcher re-introspects the target)."""

    def resolver(path: str) -> Any:
        def needs_args(repo_id: str, fps: int = 30) -> None:
            raise TypeError("needs_args() got an unexpected keyword argument 'bogus'")

        return needs_args

    monkeypatch.setattr(M, "_import_from_lerobot", resolver)
    result = _fn(module="datasets.create", method="", parameters={"bogus": 1})
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "TypeError" in text
    assert "Expected signature" in text
    assert "repo_id" in text and "fps" in text


def test_type_error_without_introspectable_signature_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the target raises TypeError but exposes no introspectable signature
    (a C builtin like range), the dispatcher still reports the error instead of
    crashing in the signature-formatting fallback."""
    monkeypatch.setattr(M, "_import_from_lerobot", lambda path: range)
    result = _fn(module="builtins.range", method="", parameters={"bogus": 1})
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "TypeError" in text


# ----------------------------------------------------------------------------
# Graceful degradation: optional deps absent or an image codec refuses a frame
#
# use_lerobot is the agent's primary, blind access path into lerobot. When an
# optional dependency (lerobot / numpy / cv2) is missing, or an image codec
# refuses a frame, the tool must degrade to a structured error or a clean
# structural result -- never raise past the dispatcher or emit a half-built
# content block. These pin that contract by simulating each dependency as
# absent via ``sys.modules[name] = None``, which makes the corresponding
# ``import`` raise ImportError exactly as it would on a host without the extra.
# ----------------------------------------------------------------------------


def test_discovery_reports_clean_error_when_lerobot_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery surfaces a structured 'not installed' error -- not a
    traceback -- when the lerobot package cannot be imported."""
    monkeypatch.setitem(sys.modules, "lerobot", None)
    result = _fn(module="__discovery__")
    assert result["status"] == "error"
    text = _texts(result)
    _assert_ascii(text)
    assert "lerobot not installed" in text


def test_is_image_array_false_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image detection returns False (not an exception) when numpy is absent."""
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert M._is_image_array(object()) is False


def test_serialize_value_survives_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The serializer still renders plain-Python structures when numpy is
    absent, falling through its numpy fast-path instead of raising."""
    payload = [1, "two", {"three": 3}]
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert M._serialize_value(payload) == [1, "two", {"three": 3}]


def test_array_to_image_block_none_without_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without cv2 there is no encoder, so the image block is dropped (None)
    rather than crashing result assembly."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert M._array_to_image_block(frame) is None


def test_array_to_image_block_falls_back_to_png_when_jpeg_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large opaque frame prefers JPEG, but when the JPEG encoder refuses the
    frame the encoder retries as PNG instead of yielding nothing."""
    import cv2

    real_imencode = cv2.imencode

    def fake_imencode(ext: str, img: Any, *args: Any) -> tuple[bool, Any]:
        if ext == ".jpg":
            return False, np.empty(0, dtype=np.uint8)
        return real_imencode(ext, img, *args)

    monkeypatch.setattr(cv2, "imencode", fake_imencode)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)  # > 320x240 -> JPEG preferred
    block = M._array_to_image_block(frame)
    assert block is not None
    assert block["image"]["format"] == "png"


def test_array_to_image_block_none_when_encoder_refuses_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every codec refuses the frame, the encoder yields None instead of a
    malformed content block."""
    import cv2

    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: (False, np.empty(0, dtype=np.uint8)))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)  # small -> PNG path
    assert M._array_to_image_block(frame) is None


def test_collect_images_stops_at_depth_limit() -> None:
    """Image collection does not recurse past two container levels, so a frame
    nested too deep is ignored rather than walked indefinitely."""
    blocks: list[dict[str, Any]] = []
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    M._collect_images(frame, blocks, _depth=3)
    assert blocks == []
