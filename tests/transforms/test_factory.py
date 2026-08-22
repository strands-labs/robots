"""Tests for the dataset-transform factory: registry, aliases, discovery."""

import ast
import importlib
import inspect
import re
import textwrap

import pytest

from strands_robots.transforms import (
    DatasetTransform,
    create_transform,
    import_transform_class,
    list_transforms,
    register_transform,
)
from strands_robots.transforms.cosmos_transfer import CosmosTransferTransform
from strands_robots.transforms.factory import _runtime_aliases, _runtime_registry
from strands_robots.transforms.mock import MockTransform


class TestFactory:
    """The factory resolves the built-ins, runtime names and refusals."""

    def test_create_mock(self):
        """``mock`` resolves through the package's built-in registration."""
        t = create_transform("mock")
        assert isinstance(t, MockTransform)
        assert t.provider_name == "mock"

    def test_create_cosmos_transfer(self):
        """``cosmos_transfer`` constructs without any generation stack installed."""
        t = create_transform("cosmos_transfer")
        assert isinstance(t, CosmosTransferTransform)
        assert t.provider_name == "cosmos_transfer"

    def test_list_transforms_includes_builtins(self):
        registered = list_transforms()
        assert "mock" in registered
        assert "cosmos_transfer" in registered

    def test_kwargs_passthrough(self):
        """Constructor kwargs reach the transform class."""
        t = create_transform("mock", pixel_shift=10)
        assert isinstance(t, MockTransform)

    def test_runtime_register_and_alias(self):
        register_transform("custom_v2v", lambda: MockTransform, aliases=["cv"])
        assert isinstance(create_transform("custom_v2v"), MockTransform)
        assert isinstance(create_transform("cv"), MockTransform)
        assert "custom_v2v" in list_transforms()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="No transform registered"):
            create_transform("does_not_exist_xyz")

    def test_discovered_module_with_a_missing_dependency_surfaces_the_import_error(self, tmp_path, monkeypatch):
        """A provider module that EXISTS but whose backend dep is absent is not "unregistered".

        Collapsing that ``ModuleNotFoundError`` into the "No transform
        registered ... Available transforms" refusal sends the user to the
        wrong problem: the fix is installing the dependency, not spelling the
        provider differently. Only the absence of the provider module itself
        may fall through to the ValueError.
        """
        import strands_robots.transforms as transforms_pkg

        (tmp_path / "heavybackend.py").write_text(
            '"""Fake discovered transform whose backend SDK is not installed."""\n'
            "import a_backend_sdk_that_is_not_installed_xyz  # noqa: F401\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(transforms_pkg, "__path__", [*transforms_pkg.__path__, str(tmp_path)])
        importlib.invalidate_caches()
        with pytest.raises(ModuleNotFoundError, match="a_backend_sdk_that_is_not_installed_xyz"):
            import_transform_class("heavybackend")

    def test_auto_discovery_import(self, tmp_path, monkeypatch):
        """``import_transform_class`` finds the module-per-provider layout.

        Exercised with a provider that exists ONLY as a module: the shipped
        ``mock`` / ``cosmos_transfer`` are runtime-registered, so they are
        answered by the runtime rung and cannot cover this one.
        """
        import strands_robots.transforms as transforms_pkg

        (tmp_path / "discovered.py").write_text(
            '"""Fake module-only transform provider."""\n'
            "from strands_robots.transforms.base import DatasetTransform\n"
            "\n"
            "class DiscoveredTransform(DatasetTransform):\n"
            '    """Module-only provider, reachable by auto-discovery alone."""\n'
            "    @property\n"
            "    def provider_name(self):\n"
            '        return "discovered"\n'
            "    def validate(self, spec):\n"
            "        return self._spec_problems(spec)\n"
            "    def transform_frames(self, camera_key, frames, spec, *, source_episode, variant):\n"
            "        return frames\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(transforms_pkg, "__path__", [*transforms_pkg.__path__, str(tmp_path)])
        importlib.invalidate_caches()
        assert import_transform_class("discovered").__name__ == "DiscoveredTransform"

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DatasetTransform()  # type: ignore[abstract]


class TestBothResolversServeEveryListedTransform:
    """``import_transform_class`` resolves every name ``list_transforms`` advertises.

    The factory has two entry points onto one question - which
    :class:`DatasetTransform` subclass a provider name means.
    :func:`create_transform` answers it to build an instance;
    :func:`import_transform_class` is the public answer for a caller that wants
    the class without paying for construction. A name only one of them can
    serve makes the pair a coin flip on which door the caller used, and the
    refusal ``import_transform_class`` raises builds its available list from
    ``list_transforms()`` - so a name it cannot serve is advertised by the very
    message that rejects it.
    """

    @pytest.fixture(autouse=True)
    def _a_runtime_only_provider(self):
        """Put a runtime-ONLY name in the graded set, and take it back out.

        Registered here rather than relied upon from a sibling test: the three
        sweeps below grade whatever ``list_transforms()`` returns, and both
        shipped transforms happen to sit under ``strands_robots.transforms``,
        so without a name that ships no module they would pass on a resolver
        that never consults the runtime registry at all.
        """

        class SweepProbeTransform(MockTransform):
            pass

        register_transform("sweep_probe", lambda: SweepProbeTransform, aliases=["sp"])
        try:
            yield SweepProbeTransform
        finally:
            _runtime_registry.pop("sweep_probe", None)
            _runtime_aliases.pop("sp", None)

    def test_every_listed_transform_resolves(self):
        """No advertised provider is refused by the public resolver."""
        refused = {}
        for name in list_transforms():
            try:
                import_transform_class(name)
            except Exception as e:  # noqa: BLE001 - report every failure, not the first
                refused[name] = f"{type(e).__name__}: {e}"
        assert not refused, f"list_transforms() advertises providers import_transform_class refuses: {refused}"

    def test_the_refusal_advertises_only_names_it_can_serve(self):
        """A refusal must not enumerate the provider it just rejected."""
        with pytest.raises(ValueError) as exc:
            import_transform_class("no_such_transform_xyz")
        message = str(exc.value)
        match = re.search(r"Available transforms: (\[[^\]]*\])", message)
        assert match, f"premise: the refusal names an available-transforms list to grade: {message}"
        advertised = ast.literal_eval(match.group(1))
        assert advertised, "premise: the available-transforms list is non-empty"
        unservable = []
        for name in advertised:
            try:
                import_transform_class(name)
            except Exception:  # noqa: BLE001 - any failure means the list over-promises
                unservable.append(name)
        assert not unservable, f"the refusal offers providers it cannot resolve: {unservable}\n  message: {message}"

    def test_the_two_entry_points_agree_on_every_listed_name(self):
        """Resolving a class and building an instance answer the same question."""
        disagree = []
        for name in list_transforms():
            try:
                import_transform_class(name)
                imports = True
            except Exception:  # noqa: BLE001
                imports = False
            try:
                create_transform(name)
                creates = True
            except Exception:  # noqa: BLE001
                creates = False
            if imports is not creates:
                disagree.append((name, imports, creates))
        assert not disagree, f"(name, import_transform_class, create_transform) disagree: {disagree}"

    def test_the_builtin_transforms_are_reachable_through_the_public_resolver(self):
        """``mock`` and ``cosmos_transfer`` register at runtime, not in policies.json."""
        assert import_transform_class("mock") is MockTransform
        assert import_transform_class("cosmos_transfer") is CosmosTransferTransform

    def test_a_runtime_registered_transform_and_its_alias_resolve(self):
        """The documented ``register_transform`` route reaches the public resolver.

        This is the route ``register_transform``'s own example and the custom-
        backend docs prescribe, and it ships no module under
        ``strands_robots.transforms`` and no ``transform`` block - so the
        runtime rung is the only one that can answer for it.
        """

        class ResolverProbeTransform(MockTransform):
            pass

        register_transform("resolver_probe", lambda: ResolverProbeTransform, aliases=["rp"])
        try:
            assert import_transform_class("resolver_probe") is ResolverProbeTransform
            assert import_transform_class("rp") is ResolverProbeTransform
        finally:
            _runtime_registry.pop("resolver_probe", None)
            _runtime_aliases.pop("rp", None)

    def test_every_listed_transform_is_runtime_registered(self):
        """Non-vacuity: today the runtime rung is the only one with any entries.

        No provider in policies.json declares a ``transform`` block, so every
        name this factory can serve arrives through :func:`register_transform`.
        A resolver that skipped the runtime rung could therefore only ever
        answer by the auto-discovery accident of a provider's module happening
        to sit under ``strands_robots.transforms``.
        """
        from strands_robots.registry.policies import get_policy_provider

        listed = list_transforms()
        assert listed, "premise: the factory advertises at least one transform"
        json_declared = [name for name in listed if (get_policy_provider(name) or {}).get("transform")]
        assert not json_declared, f"expected no JSON-declared transforms yet, got {json_declared}"


class TestTheRuntimeRungKeepsItsPrecedence:
    """A runtime registration shadows a JSON ``transform`` block, as before.

    ``create_transform`` consulted the runtime registry ahead of the registry
    lookup, so a caller could already override a shipped provider's transform
    by re-registering the name. Sharing one resolver has to keep that ordering:
    demoting the runtime rung below the JSON rung would silently ignore such an
    override instead of honoring it.
    """

    def test_a_runtime_registration_overrides_a_json_transform_block(self, monkeypatch):
        """A re-registered name wins over a shipped ``transform`` block."""
        import strands_robots.transforms.factory as factory_mod

        providers = {
            "jsonprov": {
                "transform": {
                    "module": "strands_robots.transforms.mock",
                    "class": "MockTransform",
                }
            }
        }
        monkeypatch.setattr(factory_mod, "get_policy_provider", providers.get)
        monkeypatch.setattr(factory_mod, "list_policy_providers", lambda: list(providers))

        # The JSON rung answers on its own.
        assert import_transform_class("jsonprov") is MockTransform

        class ShadowingTransform(MockTransform):
            pass

        register_transform("jsonprov", lambda: ShadowingTransform)
        try:
            assert import_transform_class("jsonprov") is ShadowingTransform
            assert isinstance(create_transform("jsonprov"), ShadowingTransform)
        finally:
            _runtime_registry.pop("jsonprov", None)
        # The JSON rung answers again once the override is gone.
        assert import_transform_class("jsonprov") is MockTransform

    def test_an_unknown_provider_is_still_refused(self):
        """Neither entry point invents a transform for an unregistered name."""
        with pytest.raises(ValueError, match="No transform registered"):
            import_transform_class("definitely_not_a_transform_xyz")
        with pytest.raises(ValueError, match="No transform registered"):
            create_transform("definitely_not_a_transform_xyz")

    def test_constructor_kwargs_still_reach_the_transform(self):
        """Delegating to one resolver must not drop ``create_transform``'s kwargs."""
        transform = create_transform("mock", pixel_shift=7)
        assert isinstance(transform, MockTransform)
        assert transform._pixel_shift == 7

    def test_create_transform_adds_nothing_but_the_constructor_call(self):
        """One resolver owns the question - no second copy of the rung ladder.

        Both docstrings state that the two entry points answer for the same set
        of names, and a private copy of the runtime-registry lookup inside
        :func:`create_transform` is precisely how they came apart: the rung
        lived in one function and not the other, so the pair disagreed on every
        runtime-registered name while each looked correct on its own.
        """
        source = textwrap.dedent(inspect.getsource(create_transform))
        assert "import_transform_class(" in source, "create_transform must resolve through the shared resolver"
        assert "_runtime_registry" not in source, (
            "create_transform re-derives the runtime rung instead of delegating - "
            "two copies of one resolution order is how the entry points diverged"
        )
