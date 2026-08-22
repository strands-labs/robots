"""Cosmos-Transfer-style backend: the pipeline seam, its refusals, its hand-off.

The backend is a vendor-neutral adapter: it binds any object satisfying the
``generate(video, prompt=..., seed=...)`` protocol - injected constructed, or
named by a dotted import path resolved lazily - and refuses cleanly (through
``validate``, never an ``AttributeError``) when none is bound. The tests
exercise the seam with a fake pipeline, the same dependency-injection shape
``Cosmos3Policy`` uses for its ``client=`` / ``diffusers_backend=`` kwargs, so
none of this needs a generation model installed.
"""

import ast
import inspect
import pathlib
import sys
import textwrap
import types

import numpy as np
import pytest

from strands_robots.transforms import TransformSpec, cosmos_transfer, derive_variant_seed, load_provenance
from strands_robots.transforms.cosmos_transfer import _PIPELINE_HELP, CosmosTransferTransform


class FakePipeline:
    """Minimal video2video pipeline: records calls, shifts pixels."""

    version = "fake-2.5"

    def __init__(self):
        self.calls = []

    def generate(self, video, prompt="", seed=None):
        """Record the call and return a deterministic pixel shift."""
        self.calls.append({"shape": video.shape, "prompt": prompt, "seed": seed})
        return np.clip(video.astype(np.int16) + 30, 0, 255).astype(np.uint8)


#: Module-level instance so the dotted-import-path seam has a real target.
FAKE_PIPELINE = FakePipeline()


def _frames(t: int = 3) -> np.ndarray:
    return np.full((t, 8, 8, 3), 100, dtype=np.uint8)


class TestPipelineSeam:
    def test_no_pipeline_is_a_validate_problem_not_a_crash(self, tmp_path):
        problems = CosmosTransferTransform().validate(TransformSpec())
        assert any("no video2video pipeline is bound" in p for p in problems)
        assert any("Cosmos-Transfer" in p for p in problems)

    def test_object_without_generate_is_refused(self):
        problems = CosmosTransferTransform(pipeline=object())._pipeline_problems()
        assert any("no callable generate()" in p for p in problems)

    def test_injected_pipeline_passes(self):
        assert CosmosTransferTransform(pipeline=FakePipeline())._pipeline_problems() == []

    def test_dotted_path_resolves_lazily(self):
        t = CosmosTransferTransform(pipeline="tests.transforms.test_cosmos_transfer:FAKE_PIPELINE")
        assert t._pipeline_problems() == []
        out = t.transform_frames("cam", _frames(), TransformSpec(prompt="night"), source_episode=0, variant=0)
        assert out.shape == _frames().shape

    def test_dotted_path_to_a_factory_is_constructed(self):
        t = CosmosTransferTransform(pipeline="tests.transforms.test_cosmos_transfer.FakePipeline")
        assert t._pipeline_problems() == []
        assert t.transform_version == "fake-2.5"

    def test_unresolvable_dotted_path_is_a_problem_not_a_raise(self):
        problems = CosmosTransferTransform(pipeline="no.such.module:thing")._pipeline_problems()
        assert any("did not resolve" in p for p in problems)

    def test_malformed_dotted_path_is_a_problem(self):
        problems = CosmosTransferTransform(pipeline="justoneword")._pipeline_problems()
        assert any("is not 'pkg.mod:attr'" in p for p in problems)

    def test_direct_frames_call_without_pipeline_raises_with_remedy(self):
        with pytest.raises(RuntimeError, match="no video2video pipeline"):
            CosmosTransferTransform().transform_frames("cam", _frames(), TransformSpec(), source_episode=0, variant=0)


class TestEveryProbePointReportsInsteadOfRaising:
    """A caller-supplied pipeline runs the caller's code at three probe points.

    ``validate`` is documented to *return* a list of problems, and
    :meth:`~strands_robots.transforms.base.DatasetTransform.transform` to
    return ``status="error"`` for anything but a backend shape / dtype bug - so
    a failure at any of the seam's three probe points (the module import and
    attribute lookup, the zero-arg construction, the ``generate`` read) is a
    problem, not an exception handed to the caller.

    Constructing a real generation pipeline loads weights and touches a device,
    so the classes are the deployment's: ``ImportError`` from an optional dep
    imported inside a factory body, ``RuntimeError`` with no driver, ``OSError``
    with the weights absent, ``ValueError`` on a malformed config. Only
    ``TypeError`` (a factory that needs arguments) was reported before.
    """

    # (probe-point label, pipeline= spelling built from a raising class)
    POINTS = ("import", "construct", "generate_read")
    CLASSES = (ImportError, ModuleNotFoundError, RuntimeError, OSError, FileNotFoundError, ValueError, KeyError)

    @staticmethod
    def _transform(point: str, exc: BaseException, monkeypatch) -> CosmosTransferTransform:
        """Build a transform whose seam raises ``exc`` at ``point``."""
        mod = types.ModuleType("fake_pipeline_mod")
        if point == "import":
            # A module-level ``__getattr__`` runs caller code on the lookup the
            # import handler wraps. Scoped to the target name: a module whose
            # every attribute raises also breaks the caller's own introspection
            # (``inspect.getmodule`` reads ``__file__``), which is not the
            # behaviour under test.
            def module_getattr(name: str):
                if name == "target":
                    raise exc
                raise AttributeError(name)

            mod.__getattr__ = module_getattr  # type: ignore[method-assign]
        elif point == "construct":

            def factory():
                raise exc

            mod.target = factory  # type: ignore[attr-defined]
        else:

            class LazyHandle:
                @property
                def generate(self):
                    raise exc

            mod.target = LazyHandle()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fake_pipeline_mod", mod)
        return CosmosTransferTransform(pipeline="fake_pipeline_mod:target")

    @pytest.mark.parametrize("point", POINTS)
    @pytest.mark.parametrize("exc_type", CLASSES, ids=lambda c: c.__name__)
    def test_a_raising_probe_point_is_a_problem(self, point, exc_type, monkeypatch):
        transform = self._transform(point, exc_type("deployment says no"), monkeypatch)
        problems = transform.validate(TransformSpec(source_root="/nope", output_root="/out"))
        assert any("deployment says no" in p for p in problems), problems
        assert any(_PIPELINE_HELP[:40] in p for p in problems), "the refusal must still carry the remedy"

    @pytest.mark.parametrize("point", POINTS)
    def test_the_documented_run_path_reports_error_rather_than_raising(self, point, monkeypatch, tmp_path):
        """The package docstring's own usage: validate, then transform."""
        source = tmp_path / "src"
        (source / "meta").mkdir(parents=True)
        (source / "meta" / "info.json").write_text('{"codebase_version": "v3.0"}')
        transform = self._transform(point, RuntimeError("Found no NVIDIA driver"), monkeypatch)
        spec = TransformSpec(source_root=source, output_root=tmp_path / "out")
        assert transform.validate(spec), "the seam failure must be reported by validate"
        result = transform.transform(spec)
        assert result.status == "error"
        assert "Found no NVIDIA driver" in result.message

    @pytest.mark.parametrize("point", POINTS)
    @pytest.mark.parametrize("exc_type", (KeyboardInterrupt, SystemExit, GeneratorExit), ids=lambda c: c.__name__)
    def test_an_operator_interrupt_is_not_downgraded_to_a_spec_problem(self, point, exc_type, monkeypatch):
        """Ctrl-C while a multi-gigabyte pipeline loads is not a bad spec.

        Fails if the handlers widen to ``BaseException``: the caller asked to
        stop, and reporting that as a validation problem would keep going.
        """
        transform = self._transform(point, exc_type(), monkeypatch)
        with pytest.raises(exc_type):
            transform.validate(TransformSpec(source_root="/nope", output_root="/out"))

    def test_a_construction_failure_is_not_reported_as_a_bad_signature(self, monkeypatch):
        """A factory that raised is constructible; only the deployment failed.

        Folding every construction failure into the zero-arg wording would name
        the wrong cause - the remedy for "no driver" is not a different factory
        signature.
        """
        transform = self._transform("construct", RuntimeError("Found no NVIDIA driver"), monkeypatch)
        problem = transform._pipeline_problems()[0]
        assert "raised RuntimeError while being constructed" in problem
        assert "not constructible zero-arg" not in problem

    def test_a_factory_needing_arguments_keeps_its_wording(self, monkeypatch):
        """The one class already reported keeps the message it already had."""
        mod = types.ModuleType("argful_mod")

        def needs_args(cfg):  # zero-arg construction cannot satisfy this
            return cfg

        mod.target = needs_args  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "argful_mod", mod)
        problems = CosmosTransferTransform(pipeline="argful_mod:target")._pipeline_problems()
        assert any("is not constructible zero-arg" in p for p in problems), problems

    def test_an_injected_object_with_a_raising_generate_read_is_a_problem(self):
        """The third probe point is also reached with no import path at all."""

        class LazyHandle:
            @property
            def generate(self):
                raise RuntimeError("CUDA context not initialised")

        problems = CosmosTransferTransform(pipeline=LazyHandle())._pipeline_problems()
        assert any("while its generate() surface was read" in p for p in problems), problems
        assert any("CUDA context not initialised" in p for p in problems), problems

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("tests.transforms.test_cosmos_transfer:FAKE_PIPELINE", None),
            ("tests.transforms.test_cosmos_transfer.FakePipeline", None),
            ("justoneword", "is not 'pkg.mod:attr'"),
            ("no.such.module:thing", "did not resolve"),
            ("tests.transforms.test_cosmos_transfer:_frames", "resolves to no generate() surface"),
        ],
    )
    def test_the_verdict_on_every_previously_accepted_spelling_is_unchanged(self, spelling, expected):
        """Widening the handlers changes no verdict that was already reached."""
        problems = CosmosTransferTransform(pipeline=spelling)._pipeline_problems()
        if expected is None:
            assert problems == []
        else:
            assert any(expected in p for p in problems), problems


class TestTheSeamHasOneGuardedProbeRule:
    """Structural: a probe point added later cannot re-derive an unguarded read.

    The behavioural tests above cover the three probe points the seam has
    today; these pin the rule itself, so a fourth one is held to it.
    """

    @staticmethod
    def _func(name: str) -> ast.FunctionDef:
        source = textwrap.dedent(inspect.getsource(getattr(cosmos_transfer, name, None) or CosmosTransferTransform))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} not found")

    def test_the_generate_read_has_a_single_owner(self):
        """No other function reads ``generate`` off the caller's object."""
        tree = ast.parse(pathlib.Path(cosmos_transfer.__file__).read_text(encoding="utf-8"))
        owners = set()
        for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for call in [n for n in ast.walk(func) if isinstance(n, ast.Call)]:
                if ast.unparse(call.func) != "getattr":
                    continue
                if any(isinstance(a, ast.Constant) and a.value == "generate" for a in call.args):
                    owners.add(func.name)
        assert owners == {"_generate_surface"}, f"the generate read must have one guarded owner, found {owners}"

    @pytest.mark.parametrize("func_name", ["_generate_surface", "_pipeline_problems"])
    def test_every_probe_handler_reports_rather_than_raises(self, func_name):
        """Each ``try`` in the resolution names ``Exception`` in some handler."""
        func = self._func(func_name)
        tries = [n for n in ast.walk(func) if isinstance(n, ast.Try)]
        assert tries, f"premise: {func_name} guards at least one probe"
        for node in tries:
            named = set()
            for handler in node.handlers:
                if handler.type is None:
                    named.add("bare")
                elif isinstance(handler.type, ast.Tuple):
                    named.update(ast.unparse(e) for e in handler.type.elts)
                else:
                    named.add(ast.unparse(handler.type))
            assert "Exception" in named, f"{func_name}:{node.lineno} reports only {sorted(named)}"

    def test_the_module_import_is_guarded(self):
        """``import_module`` runs a caller's module body; never outside a try."""
        func = self._func("_pipeline_problems")
        guarded = {id(n) for t in ast.walk(func) if isinstance(t, ast.Try) for n in ast.walk(t)}
        imports = [
            n for n in ast.walk(func) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("import_module")
        ]
        assert imports, "premise: the seam resolves a dotted path by importing it"
        for call in imports:
            assert id(call) in guarded, f"import_module at line {call.lineno} is not inside a try"


class TestHandOff:
    def test_prompt_and_derived_seed_reach_the_pipeline(self):
        pipeline = FakePipeline()
        t = CosmosTransferTransform(pipeline=pipeline)
        spec = TransformSpec(prompt="the same scene at night", seed=11)
        t.transform_frames("cam", _frames(), spec, source_episode=2, variant=1)
        assert pipeline.calls == [
            {
                "shape": (3, 8, 8, 3),
                "prompt": "the same scene at night",
                "seed": derive_variant_seed(11, 2, 1),
            }
        ]

    def test_no_seed_stays_none(self):
        pipeline = FakePipeline()
        CosmosTransferTransform(pipeline=pipeline).transform_frames(
            "cam", _frames(), TransformSpec(), source_episode=0, variant=0
        )
        assert pipeline.calls[0]["seed"] is None

    def test_transform_version_reads_the_pipeline_never_guesses(self):
        assert CosmosTransferTransform(pipeline=FakePipeline()).transform_version == "fake-2.5"
        assert CosmosTransferTransform().transform_version == "unversioned"


class TestEndToEnd:
    def test_pipeline_pixels_land_in_a_provenance_marked_dataset(self, record_source_dataset, tmp_path):
        """The generic orchestration carries the fake pipeline's output to disk."""
        pytest.importorskip("lerobot")
        source_root = record_source_dataset([100])
        output_root = str(tmp_path / "cosmos_out")
        pipeline = FakePipeline()
        result = CosmosTransferTransform(pipeline=pipeline).transform(
            TransformSpec(source_root=source_root, output_root=output_root, prompt="cluttered kitchen", seed=5)
        )
        assert result.status == "success", result.message
        assert result.episodes_written == 1
        assert pipeline.calls and pipeline.calls[0]["prompt"] == "cluttered kitchen"
        records = load_provenance(output_root)
        assert records[0]["transform"] == "cosmos_transfer"
        assert records[0]["transform_version"] == "fake-2.5"
        assert records[0]["prompt"] == "cluttered kitchen"

    def test_wrong_shape_return_is_refused_loudly(self, record_source_dataset, tmp_path):
        """A pipeline that changes the stream's schema is a bug, not a dataset."""
        pytest.importorskip("lerobot")

        class WrongShapePipeline:
            def generate(self, video, prompt="", seed=None):
                return video[:, ::2, ::2, :]  # downscales - schema breach

        source_root = record_source_dataset([100])
        spec = TransformSpec(source_root=source_root, output_root=str(tmp_path / "cosmos_bad"))
        with pytest.raises(ValueError, match="a transform changes pixels, never the stream's schema"):
            CosmosTransferTransform(pipeline=WrongShapePipeline()).transform(spec)


class TestCallShapedAdapterExample:
    """The documented ``__call__``-shaped adapter is a working remedy, not prose.

    Diffusers-family video pipelines (including the diffusers-hosted Cosmos
    Transfer checkpoints) expose ``__call__``, return an object carrying
    ``.frames``, seed via ``generator=`` and emit ``float`` frames in
    ``[0, 1]``. ``_PIPELINE_HELP`` documents a thin adapter closing those
    gaps; these tests pin that the guidance is present in every pipeline
    refusal and that an adapter written exactly as documented satisfies the
    seam, so the example cannot rot into the guesswork it exists to remove.
    """

    def test_pipeline_refusals_carry_the_adapter_guidance(self):
        problems = CosmosTransferTransform().validate(TransformSpec())
        assert any("__call__-shaped" in p for p in problems)
        assert any(".frames" in p for p in problems)

    def test_documented_adapter_bridges_a_call_shaped_pipeline(self):
        torch = pytest.importorskip("torch")

        class FakeDiffusersOutput:
            def __init__(self, frames):
                self.frames = frames

        class FakeDiffusersPipe:
            """``__call__``-shaped, ``.frames``-returning, ``generator=``-seeded."""

            def __init__(self):
                self.calls = []

            def __call__(self, *, video, prompt, output_type, generator):
                self.calls.append({"n_frames": len(video), "prompt": prompt, "generator": generator})
                assert output_type == "np"
                batch = np.stack([np.asarray(f, dtype=np.float64) / 255.0 for f in video])
                return FakeDiffusersOutput(frames=[batch])  # float in [0, 1], batched

        pipe = FakeDiffusersPipe()

        # The adapter exactly as _PIPELINE_HELP and the module docstring show it.
        class Adapter:
            def generate(self, video, prompt="", seed=None):
                g = None if seed is None else torch.Generator().manual_seed(seed)
                f = pipe(video=list(video), prompt=prompt, output_type="np", generator=g).frames[0]
                return np.clip(np.round(np.asarray(f) * 255.0), 0, 255).astype(np.uint8)

        transform = CosmosTransferTransform(pipeline=Adapter())
        assert transform._pipeline_problems() == []

        spec = TransformSpec(prompt="the same scene at night", seed=11)
        out = transform.transform_frames("cam", _frames(), spec, source_episode=2, variant=1)
        # The round-trip through the float [0, 1] convention lands back on the contract.
        assert out.shape == _frames().shape
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, _frames())
        # The derived seed reaches the pipeline as a generator, not a raw int.
        assert pipe.calls[0]["prompt"] == "the same scene at night"
        assert pipe.calls[0]["generator"].initial_seed() == derive_variant_seed(11, 2, 1)

        # An unseeded spec stays unseeded: no generator is fabricated for None.
        transform.transform_frames("cam", _frames(), TransformSpec(), source_episode=0, variant=0)
        assert pipe.calls[1]["generator"] is None
