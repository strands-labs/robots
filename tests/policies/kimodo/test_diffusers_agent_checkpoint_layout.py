# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``DiffusersKimodoAgent`` refusal when the target is not a diffusers pipeline.

Covers the load-time guard in
``strands_robots.policies.kimodo._diffusers_agent``. A checkpoint carrying no
``model_index.json`` is not a diffusers pipeline, so ``from_pretrained`` can
never load it -- ``nvidia/Kimodo-G1-RP-v1``, the provider's own default
``model_id``, publishes bare weights for NVIDIA's ``kimodo`` runtime and
declares ``library_name: kimodo`` on the Hub. Left unguarded, the caller gets a
404 for a file that will never exist, naming neither the cause nor a remedy.

The guard must fire for both shapes ``huggingface_hub`` raises and stay silent
for a transport failure, which is a genuinely different problem:

* ``EntryNotFoundError`` subclasses ``Exception`` alone; only
  ``RemoteEntryNotFoundError`` also inherits ``OSError``. An ``OSError``-only
  catch therefore leaves the refusal unreachable for the bare form, so both are
  driven here.
* A 401/503 must re-raise untouched -- reporting a network outage as a layout
  mismatch sends the caller after the wrong problem.

The pipeline is stubbed through the module's own import seam: the agent does
``from diffusers import DiffusionPipeline`` inside ``__init__``, so a stub
registered under that name is what it loads. No weights are fetched.
"""

from __future__ import annotations

import sys
import types

import pytest

from tests.mocks.torch_mock import real_torch_installed

pytest.importorskip("diffusers", reason="Kimodo loading needs the [kimodo] extra")

from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError  # noqa: E402

from strands_robots.policies.kimodo._diffusers_agent import DiffusersKimodoAgent  # noqa: E402
from strands_robots.policies.kimodo.config import KimodoConfig  # noqa: E402

# The agent reads ``torch.float16``/``bfloat16``, outside the test-suite torch mock.
pytestmark = pytest.mark.skipif(
    not real_torch_installed(),
    reason="needs real torch: the agent reads fp16/bf16 dtypes to build the loader kwargs",
)

_INDEX_URL = "https://huggingface.co/nvidia/Kimodo-G1-RP-v1/resolve/main/model_index.json"
_INDEX_404 = f"404 Client Error. Entry Not Found for url: {_INDEX_URL}."


def _bare_entry_not_found() -> BaseException:
    """The shape that subclasses ``Exception`` alone."""
    return EntryNotFoundError(_INDEX_404)


def _remote_entry_not_found() -> BaseException:
    """The shape the Hub actually raises, which also inherits ``OSError``.

    Built lazily: the constructor requires a live response object, whose
    transport type belongs to ``huggingface_hub`` rather than to this contract.
    """
    httpx = pytest.importorskip("httpx", reason="huggingface_hub carries its own transport")
    response = httpx.Response(404, request=httpx.Request("GET", _INDEX_URL))
    return RemoteEntryNotFoundError(_INDEX_404, response=response)


@pytest.fixture
def load_with_error(monkeypatch):
    """Build an agent whose ``from_pretrained`` raises *error*.

    Returns:
        A callable ``(error, **config_kwargs) -> DiffusersKimodoAgent``, which
        propagates whatever the loader raises.
    """

    def _load(error: BaseException, **config_kwargs):
        config_kwargs.setdefault("device", "cpu")

        class _StubDiffusionPipeline:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs: object) -> object:
                raise error

        stub_module = types.ModuleType("diffusers")
        stub_module.DiffusionPipeline = _StubDiffusionPipeline  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "diffusers", stub_module)
        return DiffusersKimodoAgent(KimodoConfig(**config_kwargs))

    return _load


# ----- The refusal (regression) ----- #


@pytest.mark.parametrize(
    "make_error",
    [
        pytest.param(_bare_entry_not_found, id="entry-not-found-not-an-oserror"),
        pytest.param(_remote_entry_not_found, id="remote-entry-not-found"),
    ],
)
def test_a_hub_repo_with_no_pipeline_index_is_refused_with_the_model_id_and_the_remedy(load_with_error, make_error):
    """Both hub shapes are refused, naming the model, the missing index and the fix.

    The bare ``EntryNotFoundError`` case is the one an ``OSError``-only catch
    would miss, leaving the actionable refusal unreachable.
    """
    error = make_error()
    with pytest.raises(RuntimeError) as excinfo:
        load_with_error(error)

    message = str(excinfo.value)
    assert "nvidia/Kimodo-G1-RP-v1" in message
    assert "model_index.json" in message
    assert "motion_agent=" in message
    # The 404 stays reachable as the chained cause rather than being swallowed.
    assert excinfo.value.__cause__ is error


def test_a_local_checkpoint_directory_with_no_pipeline_index_is_refused(load_with_error, tmp_path):
    """A bare local checkpoint is refused too, probed on disk rather than by message.

    diffusers raises a plain ``OSError`` for a local directory, carrying no
    structured entry name, so the index is checked on disk.
    """
    checkpoint = tmp_path / "Kimodo-G1-RP-v1"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("dummy: 1", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"")

    with pytest.raises(RuntimeError, match="is not a diffusers pipeline"):
        load_with_error(
            OSError(f"Error no file named model_index.json found in directory {checkpoint}."),
            model_id=str(checkpoint),
        )


# ----- What the refusal must NOT claim ----- #


def test_a_transport_failure_re_raises_untouched_instead_of_blaming_the_layout(load_with_error):
    """A 503 is a network problem, not a layout problem, so it surfaces as itself."""
    with pytest.raises(OSError) as excinfo:
        load_with_error(OSError("503 Server Error: Service Unavailable for url: https://huggingface.co/"))

    assert "503" in str(excinfo.value)
    assert "not a diffusers pipeline" not in str(excinfo.value)


def test_a_missing_pipeline_component_re_raises_untouched(load_with_error):
    """An absent component file is reported as itself, not as a missing index.

    The repo *is* a pipeline; a different entry is missing. Claiming otherwise
    would send the caller to the wrong remedy.
    """
    missing_component = EntryNotFoundError(
        "404 Client Error. Entry Not Found for url: "
        "https://huggingface.co/acme/pipeline/resolve/main/unet/diffusion_pytorch_model.safetensors."
    )

    with pytest.raises(EntryNotFoundError) as excinfo:
        load_with_error(missing_component, model_id="acme/pipeline")

    assert "diffusion_pytorch_model.safetensors" in str(excinfo.value)
    assert not isinstance(excinfo.value, RuntimeError)
