"""The in-process Cosmos 3 backend on an install without the native stack.

:mod:`strands_robots.policies.cosmos3.policy_diffusers` is documented as
optional and lazy: ``diffusers`` + ``torch`` + ``transformers`` is a heavy GPU
stack, so the ``cosmos3-service`` install (msgpack + websockets only) stays the
default and every native import sits inside the function that needs it. The
module makes **two opposite decisions** about that one missing import, three
lines apart, and which one applies is a property of the action mode:

* **Refuse.** :func:`~...policy_diffusers._install_hint` is the module's single
  actionable "the native stack is not importable" message - it names the extra
  to install *and* the service backend to fall back to. Three call sites raise
  it: ``_load_pipeline``, ``_import_condition_cls`` and ``_as_action_tensor``.
  ``forward_dynamics`` / ``inverse_dynamics`` reach the third, because building
  a ``CosmosActionCondition`` from ``raw_actions`` needs a real
  ``torch.Tensor`` and no substitute exists.
* **Degrade.** ``_to_numpy`` swallows the same ``ImportError`` and hands the
  value straight to NumPy. Without torch it cannot inspect the dtype, so it
  skips the half-precision up-cast rather than refusing - and that is what lets
  the ``policy``-mode rollout complete through the documented ``pipeline=`` /
  ``condition_cls=`` injection seams on an install that has no torch at all.

The happy-path suite in ``test_policy_diffusers.py`` drives two of the three
refusal sites (``test_missing_diffusers_raises_actionable_error`` and
``test_import_condition_cls_missing_diffusers_raises_install_hint``) and both
``_to_numpy`` up-casts with torch *present*. The third refusal site and the
degradation branch are the remainder, so nothing held either decision to its
documented direction: a regression that made ``_as_action_tensor`` degrade
would hand the pipeline a NumPy array where it expects a tensor, and one that
made ``_to_numpy`` refuse would take the whole no-GPU injection path with it.

No GPU, no model weights, no policy server, and - by construction - no torch on
the paths under test.
"""

import ast
import inspect
import pathlib
import sys
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import Cosmos3DiffusersBackend
from strands_robots.policies.cosmos3.embodiments import get_embodiment
from strands_robots.policies.cosmos3.policy_diffusers import _install_hint, _to_numpy

# The three functions that raise the shared install hint. Pinned as a set so a
# fourth site cannot ship without a decision about whether it is driven.
_REFUSAL_SITES = ("_load_pipeline", "_import_condition_cls", "_as_action_tensor")


class FakeCondition:
    """Stand-in for ``diffusers.CosmosActionCondition`` (records its kwargs)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class DetachChunk:
    """An action chunk exposing the ``.detach().cpu()`` protocol ``_to_numpy`` reads.

    A real ``Cosmos3OmniPipelineOutput.action`` is a ``list[torch.Tensor]``, and
    the only properties ``_to_numpy`` inspects are ``detach``, ``cpu`` and NumPy
    convertibility. Modelling exactly those lets the injected-pipeline path run
    on a torch-free install, which is the state under test: a real tensor cannot
    exist there, but the ``pipeline=`` seam is documented to accept any output
    satisfying the pipeline contract.
    """

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self) -> "DetachChunk":
        return self

    def cpu(self) -> np.ndarray:
        return self._arr

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._arr, dtype=dtype)


class FakePipeline:
    """Returns a canned output whose action chunk needs ``.detach()``."""

    def __init__(self, action: Any) -> None:
        self._action = action
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> types.SimpleNamespace:
        self.calls.append(kwargs)
        return types.SimpleNamespace(action=self._action, video="world", sound=None)


def _chunk() -> np.ndarray:
    """A distinctive ``[T, D]`` chunk, so a mangled value is visible."""
    return np.arange(4 * 10, dtype=np.float32).reshape(4, 10)


def _backend(pipeline: Any, **kwargs: Any) -> Cosmos3DiffusersBackend:
    """A backend built purely through the injection seams (no native import)."""
    return Cosmos3DiffusersBackend(
        embodiment=get_embodiment("droid"),
        pipeline=pipeline,
        condition_cls=FakeCondition,
        **kwargs,
    )


def _observation() -> dict[str, Any]:
    return {"prompt": "pick the cube", "observation/wrist_image_left": np.zeros((8, 8, 3), dtype=np.uint8)}


def _without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import torch`` raise, as it does on a ``cosmos3-service`` install.

    ``sys.modules["torch"] = None`` is the repository's established mechanism
    for this (see ``tests/policies/test_rng_parity.py``): the import machinery
    consults ``sys.modules`` first and raises ``ImportError`` on a ``None``
    entry. ``monkeypatch`` restores the real entry afterwards, so the numpy
    backed stand-in ``tests/conftest.py`` installs is untouched for every other
    test.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    assert sys.modules["torch"] is None, "the torch-free precondition was not established"


class TestTheNativeStackIsAbsent:
    """The refuse half: every site that needs torch reports the same remedy."""

    def test_as_action_tensor_refuses_with_the_shared_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_as_action_tensor`` is the third ``_install_hint`` site and the only
        one the happy-path suite reaches on its accepted values only. Coercing
        ``raw_actions`` needs a real ``torch.Tensor``, so without torch there is
        nothing to degrade to: it must raise the same actionable message its two
        siblings do, chained to the underlying ``ImportError``.
        """
        _without_torch(monkeypatch)
        with pytest.raises(ImportError) as excinfo:
            Cosmos3DiffusersBackend._as_action_tensor(_chunk())
        assert str(excinfo.value) == _install_hint()
        assert isinstance(excinfo.value.__cause__, ImportError), "the underlying import error must be chained"

    @pytest.mark.parametrize("site", _REFUSAL_SITES)
    def test_the_remedy_is_one_message_not_three_copies(self, monkeypatch: pytest.MonkeyPatch, site: str) -> None:
        """All three sites must report ``_install_hint()`` byte for byte. Each
        raises from its own ``except``, so a locally reworded copy would drift
        and a caller would get a different remedy depending on which import
        happened to be reached first.
        """
        _without_torch(monkeypatch)
        monkeypatch.setitem(sys.modules, "diffusers", None)
        backend = _backend(FakePipeline([DetachChunk(_chunk())]))

        calls = {
            "_load_pipeline": lambda: backend._load_pipeline(),
            "_import_condition_cls": lambda: backend._import_condition_cls(),
            "_as_action_tensor": lambda: Cosmos3DiffusersBackend._as_action_tensor(_chunk()),
        }
        with pytest.raises(ImportError) as excinfo:
            calls[site]()
        assert str(excinfo.value) == _install_hint()

    def test_the_remedy_names_the_extra_and_the_service_fallback(self) -> None:
        """The message is the only thing a caller on a service-only install has
        to act on, so both routes out must be in it: the extra that supplies the
        stack, and the backend that needs none of it.
        """
        hint = _install_hint()
        assert "strands-robots[cosmos3-diffusers]" in hint
        assert "backend='service'" in hint


class TestTheInjectionPathSurvivesAMissingStack:
    """The degrade half: ``policy`` mode completes with no torch at all."""

    def test_policy_mode_rollout_completes_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``policy`` mode builds its condition from an image alone, so the only
        torch reference on the whole path is ``_to_numpy``'s dtype inspection.
        Swallowing that ``ImportError`` is what makes the documented "no GPU, no
        weights" injection seam usable on a service-only install - so the chunk
        must arrive intact rather than the rollout refusing.
        """
        _without_torch(monkeypatch)
        expected = _chunk()
        backend = _backend(FakePipeline([DetachChunk(expected)]))

        action = backend.infer(_observation())["action"]

        np.testing.assert_array_equal(action, expected)
        assert action.dtype == np.float32
        assert sys.modules["torch"] is None, "the rollout must not have imported torch"

    def test_forward_dynamics_refuses_on_the_same_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The asymmetry, on one install and one backend: the mode that needs a
        tensor to build its condition refuses with the remedy, where ``policy``
        mode above completed. Reporting at the mode that cannot run is what
        keeps the one that can from being taken down with it.
        """
        _without_torch(monkeypatch)
        backend = _backend(FakePipeline([DetachChunk(_chunk())]), mode="forward_dynamics")
        with pytest.raises(ImportError) as excinfo:
            backend.infer(_observation(), raw_actions=_chunk())
        assert str(excinfo.value) == _install_hint()

    def test_the_same_rollout_is_unchanged_with_torch_present(self) -> None:
        """The over-reach control: the degradation must be invisible where torch
        *is* importable, so the two installs agree on the value the caller gets.
        """
        expected = _chunk()
        action = _backend(FakePipeline([DetachChunk(expected)])).infer(_observation())["action"]
        np.testing.assert_array_equal(action, expected)
        assert action.dtype == np.float32

    def test_to_numpy_reads_a_detachable_chunk_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_to_numpy`` in isolation: with torch absent it cannot ask whether
        the value is half precision, so it must fall through to ``np.asarray``
        rather than propagating. The two up-cast tests in the sibling module
        cover the torch-present direction only.
        """
        _without_torch(monkeypatch)
        expected = _chunk()
        arr = _to_numpy(DetachChunk(expected))
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, expected)


class TestTheRefusalSiteSetIsPinned:
    """A fourth site must not ship without a decision about its coverage."""

    @staticmethod
    def _sites_raising_the_hint(source: str) -> set[str]:
        """Functions whose body raises ``ImportError(_install_hint())``."""
        found: set[str] = set()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Raise) or inner.exc is None:
                    continue
                text = ast.unparse(inner.exc)
                if "_install_hint()" in text:
                    found.add(node.name)
        return found

    def test_exactly_these_three_sites_raise_the_hint(self) -> None:
        """Pinned as an exact set: a new lazy native import that reports the
        same remedy is a fourth cell of the matrix this module drives, and it
        should fail here until it is added rather than joining the untested half.
        """
        from strands_robots.policies.cosmos3 import policy_diffusers

        source = pathlib.Path(inspect.getfile(policy_diffusers)).read_text()
        assert self._sites_raising_the_hint(source) == set(_REFUSAL_SITES)

    def test_the_scanner_sees_a_planted_site(self) -> None:
        """Non-vacuity: an empty result would make the assertion above pass over
        a module it failed to read.
        """
        planted = "def _new_lazy_import():\n    raise ImportError(_install_hint()) from None\n"
        assert self._sites_raising_the_hint(planted) == {"_new_lazy_import"}
