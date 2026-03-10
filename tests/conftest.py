#!/usr/bin/env python3
"""Test configuration — inject stubs for optional heavy dependencies.

When PIL (Pillow), requests, zmq, or msgpack are not installed (minimal CI),
we inject lightweight stubs into sys.modules so that:

1. `from PIL import Image` resolves to a functional stub with:
   - Image.fromarray(arr) → FakeImage with .size = (width, height)
   - Image.new(mode, size) → FakeImage with .size = size
   - Image.open(path) → FakeImage with .size and .save()
   - isinstance(img, Image.Image) → True for FakeImage instances
   - np.array(img) works correctly via __array__() protocol

2. `import requests` resolves to a MagicMock so that
   `patch("requests.get/post", ...)` works in test methods.

3. `import zmq` / `import msgpack` resolve to functional stubs so that
   MsgSerializer.to_bytes/from_bytes can be tested with real roundtrips.

If the real packages ARE installed, we skip injection entirely.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Dynamic collect_ignore — skip test files whose modules are not yet
# available (e.g. modules introduced in other PRs of a multi-PR stack).
#
# Uses AST parsing (no code execution) + importlib probing to detect
# missing imports at any depth. This prevents pytest collection errors
# (ModuleNotFoundError) from aborting the entire test run.
# ═══════════════════════════════════════════════════════════════════════


def _build_collect_ignore():
    """Return list of test filenames to skip due to missing imports.

    Parses each test_*.py file's AST to extract ALL import targets
    (including those inside classes and functions), then checks whether
    each module is importable and each name is accessible. Files with
    missing modules/names are added to collect_ignore so pytest skips
    them gracefully instead of erroring.

    This handles multi-PR stacks where test files reference modules or
    names that are introduced in other PRs not yet merged.
    """
    import ast
    import importlib
    import importlib.util
    import os

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    ignore = []
    _spec_cache = {}
    _import_cache = {}

    def _is_available(mod_name):
        """Check if a module can be found and imported.

        For strands_robots.* modules, we do a full import check because
        they may have transitive dependencies (psutil, serial, lerobot)
        that fail at import time even though the module spec exists.
        """
        if mod_name in _spec_cache:
            return _spec_cache[mod_name]
        try:
            spec_ok = importlib.util.find_spec(mod_name) is not None
            if not spec_ok:
                _spec_cache[mod_name] = False
                return False
            # For strands_robots.* modules, verify actual import works
            # (catches transitive dep failures like psutil, serial, lerobot)
            if mod_name.startswith("strands_robots."):
                importlib.import_module(mod_name)
            _spec_cache[mod_name] = True
            return True
        except (ModuleNotFoundError, ValueError, ImportError):
            _spec_cache[mod_name] = False
            return False

    def _can_import_name(mod_name, name):
        """Check if 'from mod_name import name' would succeed.

        Only called for strands_robots.* packages that exist but might
        be missing specific exports from other PRs.
        """
        cache_key = (mod_name, name)
        if cache_key in _import_cache:
            return _import_cache[cache_key]
        try:
            mod = importlib.import_module(mod_name)
            result = hasattr(mod, name)
        except (ImportError, ModuleNotFoundError):
            result = False
        _import_cache[cache_key] = result
        return result

    # Packages that exist but may be missing exports from other PRs.
    # For these, we verify each imported name actually exists.
    _CHECK_EXPORTS_FOR = {
        "strands_robots",
        "strands_robots.policies",
        "strands_robots.policies.groot",
        "strands_robots.training",
        "strands_robots.assets",
        "strands_robots.stereo",
        "strands_robots.tools",
        "strands_robots.simulation",
    }

    for fname in sorted(os.listdir(tests_dir)):
        if not fname.startswith("test_") or not fname.endswith(".py"):
            continue
        fpath = os.path.join(tests_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=fname)
        except SyntaxError:
            continue

        missing = False
        for node in ast.walk(tree):
            if missing:
                break
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_available(alias.name):
                        missing = True
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if not _is_available(node.module):
                        missing = True
                    elif node.module in _CHECK_EXPORTS_FOR and node.names:
                        for alias in node.names:
                            if not _can_import_name(node.module, alias.name):
                                missing = True
                                break
        if missing:
            ignore.append(fname)

    if ignore:
        print(f"conftest: skipping {len(ignore)} test files with unavailable imports")

    return ignore


collect_ignore = _build_collect_ignore()


# ═══════════════════════════════════════════════════════════════════════
# PIL / Pillow stub
# ═══════════════════════════════════════════════════════════════════════


def _install_pil_stub():
    """Install a minimal PIL stub if Pillow is not available."""
    try:
        from PIL import Image as _check  # noqa: F401

        return  # Real Pillow is installed — nothing to do
    except ImportError:
        pass

    class FakeImage:
        """Minimal stand-in for PIL.Image.Image.

        Stores underlying pixel data as a numpy array so that
        ``np.array(fake_img)`` works exactly like real PIL Images.
        This is critical because many code paths (go1._to_numpy_image,
        _basic_image_preprocess, isaac_gym_env, etc.) rely on
        ``np.array(pil_image)`` to extract pixel data.
        """

        def __init__(self, size=(0, 0), data=None):
            self._size = tuple(size)
            if data is not None:
                self._data = np.asarray(data, dtype=np.uint8)
            else:
                # Default: zeros with shape (H, W, 3) — RGB
                w, h = self._size
                self._data = np.zeros((max(h, 0), max(w, 0), 3), dtype=np.uint8)

        @property
        def size(self):
            return self._size

        def __array__(self, dtype=None, copy=None):
            """Support np.array(fake_image) — matches real PIL behavior.

            The ``copy`` keyword is required by NumPy >= 2.0 to avoid
            DeprecationWarning.
            """
            if dtype is not None:
                return self._data.astype(dtype)
            return self._data.copy()

        def convert(self, mode):
            return self

        def resize(self, size, *args, **kwargs):
            new_w, new_h = size
            resized_data = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            copy_h = min(self._data.shape[0], new_h)
            copy_w = min(self._data.shape[1], new_w)
            if copy_h > 0 and copy_w > 0 and self._data.size > 0:
                resized_data[:copy_h, :copy_w] = self._data[:copy_h, :copy_w]
            return FakeImage(size, data=resized_data)

        def save(self, fp, *args, **kwargs):
            """Write a minimal marker file so os.path.exists() works."""
            from pathlib import PurePath

            if isinstance(fp, (str, PurePath)):
                with open(fp, "wb") as f:
                    f.write(b"FAKE_PIL_IMAGE")

        def crop(self, box=None):
            if box:
                x0, y0, x1, y1 = box
                cropped = self._data[y0:y1, x0:x1].copy()
                return FakeImage((x1 - x0, y1 - y0), data=cropped)
            return self

        def copy(self):
            return FakeImage(self._size, data=self._data.copy())

        def __repr__(self):
            return f"FakeImage(size={self._size})"

    class FakeImageModule(ModuleType):
        """Stand-in for the PIL.Image module.

        In real Pillow:
          - `from PIL import Image`  → gives the Image MODULE
          - `Image.Image`            → the Image CLASS
          - `Image.fromarray(arr)`   → module-level function
          - `Image.new(mode, size)`  → module-level function
          - `Image.open(path)`       → module-level function
        """

        # The Image class (for isinstance checks)
        Image = FakeImage

        @staticmethod
        def fromarray(arr, mode=None):
            arr = np.asarray(arr)
            if arr.ndim >= 2:
                h, w = arr.shape[0], arr.shape[1]
                return FakeImage((w, h), data=arr)
            return FakeImage((0, 0))

        @staticmethod
        def new(mode, size, color=None):
            w, h = size
            channels = 4 if mode in ("RGBA", "PA") else 3
            if color is not None:
                if isinstance(color, (tuple, list)):
                    data = np.zeros((h, w, channels), dtype=np.uint8)
                    for c in range(min(len(color), channels)):
                        data[:, :, c] = color[c]
                elif isinstance(color, (int, float)):
                    data = np.full((h, w, channels), color, dtype=np.uint8)
                else:
                    data = np.zeros((h, w, channels), dtype=np.uint8)
            else:
                data = np.zeros((h, w, channels), dtype=np.uint8)
            return FakeImage(size, data=data)

        @staticmethod
        def open(fp, mode="r"):
            return FakeImage((100, 100))

    # Create the modules
    fake_image_mod = FakeImageModule("PIL.Image")
    fake_pil = ModuleType("PIL")
    fake_pil.Image = fake_image_mod  # `from PIL import Image` returns this

    # Register in sys.modules
    sys.modules["PIL"] = fake_pil
    sys.modules["PIL.Image"] = fake_image_mod


# ═══════════════════════════════════════════════════════════════════════
# requests stub
# ═══════════════════════════════════════════════════════════════════════


def _install_requests_stub():
    """Install a MagicMock for requests if not available.

    This allows `patch("requests.get", ...)` and `patch("requests.post", ...)`
    to work. The actual HTTP behavior is always mocked by the tests themselves.
    """
    try:
        import requests as _check  # noqa: F401

        return  # Real requests installed
    except ImportError:
        pass

    mock_requests = MagicMock()
    mock_requests.__name__ = "requests"
    mock_requests.__package__ = "requests"
    mock_requests.__spec__ = None
    sys.modules["requests"] = mock_requests


# ═══════════════════════════════════════════════════════════════════════
# zmq + msgpack stubs (for groot client roundtrip tests)
# ═══════════════════════════════════════════════════════════════════════


def _install_zmq_msgpack_stubs():
    """Install real-ish zmq/msgpack stubs for serialization roundtrip tests.

    The groot client's _ensure_deps() does `import zmq; import msgpack`
    and stores them in module globals. We inject minimal stubs that
    provide just enough for MsgSerializer tests to work.
    """
    # zmq stub
    try:
        import zmq as _check  # noqa: F401

        zmq_available = True
    except ImportError:
        zmq_available = False

    # msgpack stub
    try:
        import msgpack as _check2  # noqa: F401

        msgpack_available = True
    except ImportError:
        msgpack_available = False

    if zmq_available and msgpack_available:
        return  # Both available — nothing to do

    if not zmq_available:
        mock_zmq = MagicMock()
        mock_zmq.Context.return_value = MagicMock()
        mock_zmq.REQ = 3
        sys.modules["zmq"] = mock_zmq

    if not msgpack_available:
        # msgpack needs packb/unpackb that actually serialize for roundtrip.
        #
        # Key challenge: MsgSerializer.encode_custom_classes is passed as
        # `default` to packb. It converts np.ndarray → dict with bytes values.
        # In real msgpack, bytes are a native type. In our JSON-based stub,
        # we must recursively convert the entire output to JSON-safe types.
        import base64 as _b64
        import io as _io
        import json as _json

        def _make_json_safe(obj):
            """Recursively convert an object tree to JSON-safe types.

            Handles bytes (from np.save) and numpy scalars that the
            caller's `default` function may produce.
            """
            if isinstance(obj, dict):
                return {k: _make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_json_safe(v) for v in obj]
            if isinstance(obj, bytes):
                return {"__bytes_b64__": _b64.b64encode(obj).decode("ascii")}
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                buf = _io.BytesIO()
                np.save(buf, obj, allow_pickle=False)
                return {"__ndarray_class__": True, "as_npy": _b64.b64encode(buf.getvalue()).decode("ascii")}
            return obj

        class _MsgpackStub(ModuleType):
            """Minimal msgpack stub using JSON + base64 for roundtrip tests."""

            @staticmethod
            def packb(data, default=None):
                """Serialize data using JSON with custom encoder.

                Mimics msgpack's `default` callback: called for objects that
                can't be serialized natively. We first call `default` to get
                a serializable representation, then ensure the result is
                JSON-safe (converting bytes to base64 strings).
                """

                def _json_default(obj):
                    # First: try the caller's custom encoder
                    if default:
                        encoded = default(obj)
                        if encoded is not obj:
                            # default() transformed it — make it JSON-safe
                            return _make_json_safe(encoded)

                    # Fallback: handle common non-JSON types directly
                    return _make_json_safe(obj)

                return _json.dumps(data, default=_json_default).encode("utf-8")

            @staticmethod
            def unpackb(data, object_hook=None):
                """Deserialize data."""
                raw = _json.loads(data.decode("utf-8"))

                def _walk(node):
                    if isinstance(node, dict):
                        # First recurse into children
                        node = {k: _walk(v) for k, v in node.items()}
                        # Then decode this dict's custom markers
                        if "__ndarray_class__" in node:
                            npy_data = node["as_npy"]
                            if isinstance(npy_data, str):
                                npy_bytes = _b64.b64decode(npy_data)
                            elif isinstance(npy_data, dict) and "__bytes_b64__" in npy_data:
                                npy_bytes = _b64.b64decode(npy_data["__bytes_b64__"])
                            else:
                                npy_bytes = npy_data  # already bytes
                            return np.load(_io.BytesIO(npy_bytes), allow_pickle=False)
                        if "__bytes_b64__" in node:
                            return _b64.b64decode(node["__bytes_b64__"])
                        # Apply caller's object_hook for ModalityConfig etc.
                        if object_hook:
                            node = object_hook(node)
                        return node
                    elif isinstance(node, list):
                        return [_walk(v) for v in node]
                    return node

                return _walk(raw)

        stub = _MsgpackStub("msgpack")
        sys.modules["msgpack"] = stub


# ═══════════════════════════════════════════════════════════════════════
# Install all stubs at conftest import time (before tests collect)
# ═══════════════════════════════════════════════════════════════════════

_install_pil_stub()
_install_requests_stub()
_install_zmq_msgpack_stubs()


# ═══════════════════════════════════════════════════════════════════════
# cv2 __spec__ fix — ensure importlib.util.find_spec("cv2") works
# ═══════════════════════════════════════════════════════════════════════


def _fix_cv2_spec():
    """Ensure cv2 module in sys.modules has a valid __spec__.

    Some test files inject MagicMock as cv2 into sys.modules. When other
    code later calls importlib.util.find_spec('cv2'), it crashes with
    'ValueError: cv2.__spec__ is not set' because MagicMock.__spec__
    returns another MagicMock (truthy) but isn't a real ModuleSpec.

    This runs after all stubs are installed and patches any cv2 mock
    that lacks a proper __spec__.
    """
    import importlib.machinery
    import importlib.util

    cv2_mod = sys.modules.get("cv2")
    if cv2_mod is not None:
        spec = getattr(cv2_mod, "__spec__", None)
        # MagicMock attributes return another MagicMock — check for real spec
        if spec is None or isinstance(spec, MagicMock):
            real_spec = None
            # Try to find the real cv2 spec (if real cv2 is installed)
            try:
                # Temporarily remove the mock to find the real one
                saved = sys.modules.pop("cv2", None)
                real_spec = importlib.util.find_spec("cv2")
                if saved is not None:
                    sys.modules["cv2"] = saved
            except Exception:
                if saved is not None:
                    sys.modules["cv2"] = saved

            cv2_mod.__spec__ = real_spec or importlib.machinery.ModuleSpec("cv2", None)


_fix_cv2_spec()
