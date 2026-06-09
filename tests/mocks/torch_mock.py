"""Comprehensive torch mock for CI environments without PyTorch.

Used by conftest.py to enable running all unit tests in CI without installing
PyTorch (~2GB). The mock provides numpy-backed replacements sufficient for
testing policy logic, observation mapping, and action conversion without
actual GPU inference.

Provides numpy-backed replacements for:
- torch.Tensor (MockTensor) - arithmetic, reshaping, device, slicing
- torch.nn.Parameter (MockParameter) - with requires_grad and device
- torch.device (MockDevice) - type string, equality, hashing
- Factory functions: tensor, zeros, ones, randint, rand, from_numpy, stack, cat
- Context managers: no_grad, inference_mode
- Submodules: torch.nn, torch.cuda, torch.backends, torch.amp

Usage:
    from tests.mocks.torch_mock import install_torch_mock
    install_torch_mock()  # no-op if real torch is available
"""

import logging
import sys
import types
from unittest.mock import MagicMock

import numpy as np

logger = logging.getLogger(__name__)


class MockTensor:
    """Minimal torch.Tensor replacement backed by numpy."""

    def __init__(self, data=None, dtype=None, device=None):
        if isinstance(data, MockTensor):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, (int, float)):
            self._data = np.array([data], dtype=np.float32)
        elif data is None:
            self._data = np.array([], dtype=np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)

    # Properties

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self):
        return MockDevice("cpu")

    # Shape / size helpers

    def dim(self):
        return self._data.ndim

    def size(self, dim=None):
        if dim is not None:
            return self._data.shape[dim]
        return self._data.shape

    def numel(self):
        return int(self._data.size)

    # Conversion

    def item(self):
        return float(self._data.flat[0])

    def numpy(self):
        return self._data.copy()

    def cpu(self):
        return self

    def detach(self):
        return self

    def clone(self):
        return MockTensor(self._data.copy())

    def float(self):
        return self

    def bool(self):
        return MockTensor(self._data.astype(np.bool_).astype(np.float32))

    def long(self):
        return MockTensor(self._data.astype(np.int64).astype(np.float32))

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    # Reshaping

    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self._data, axis=dim))

    def squeeze(self, dim=None):
        if dim is not None:
            return MockTensor(np.squeeze(self._data, axis=dim))
        return MockTensor(np.squeeze(self._data))

    def view(self, *shape):
        return MockTensor(self._data.reshape(shape))

    def reshape(self, *shape):
        return MockTensor(self._data.reshape(shape))

    def permute(self, *dims):
        return MockTensor(np.transpose(self._data, dims))

    # Reduction

    def max(self):
        return float(self._data.max()) if self._data.size > 0 else 0.0

    def min(self):
        return float(self._data.min()) if self._data.size > 0 else 0.0

    # Dunder methods

    def __len__(self):
        return self._data.shape[0] if self._data.ndim > 0 else 1

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return MockTensor(result)
        return MockTensor(np.array([result]))

    def __repr__(self):
        return f"MockTensor({self._data})"

    def __float__(self):
        return float(self._data.flat[0])

    def __eq__(self, other):
        if isinstance(other, MockTensor):
            return np.array_equal(self._data, other._data)
        return np.array_equal(self._data, other)

    def __abs__(self):
        return MockTensor(np.abs(self._data))

    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data - other._data)
        return MockTensor(self._data - other)

    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data + other._data)
        return MockTensor(self._data + other)

    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data / other._data)
        return MockTensor(self._data / other)

    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data * other._data)
        return MockTensor(self._data * other)


class MockParameter(MockTensor):
    """torch.nn.Parameter replacement."""

    def __init__(self, data=None, requires_grad=True):
        super().__init__(data)
        self.requires_grad = requires_grad

    @property
    def device(self):
        return MockDevice("cpu")


class MockDevice:
    """torch.device replacement."""

    def __init__(self, device_str="cpu"):
        if isinstance(device_str, MockDevice):
            device_str = device_str.type
        self.type = str(device_str).split(":")[0]

    def __repr__(self):
        return f"device(type='{self.type}')"

    def __str__(self):
        return self.type

    def __eq__(self, other):
        if isinstance(other, str):
            return self.type == other
        if isinstance(other, MockDevice):
            return self.type == other.type
        return False

    def __hash__(self):
        return hash(self.type)


class _NoGrad:
    """torch.no_grad / torch.inference_mode replacement."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, func):
        return func


# Factory functions


def _tensor(data, dtype=None, device=None):
    return MockTensor(data, dtype=dtype, device=device)


def _zeros(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return MockTensor(np.zeros(shape, dtype=np.float32))


def _ones(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return MockTensor(np.ones(shape, dtype=np.float32))


def _from_numpy(arr):
    return MockTensor(arr)


def _stack(tensors, dim=0):
    arrays = [t._data if isinstance(t, MockTensor) else np.array(t) for t in tensors]
    return MockTensor(np.stack(arrays, axis=dim))


def _cat(tensors, dim=0):
    arrays = [t._data if isinstance(t, MockTensor) else np.array(t) for t in tensors]
    return MockTensor(np.concatenate(arrays, axis=dim))


def _randint(low, high, size, dtype=None):
    return MockTensor(np.random.randint(low, high, size=size).astype(np.float32))


def _rand(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return MockTensor(np.random.rand(*shape).astype(np.float32))


def _randn(*shape, dtype=None, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return MockTensor(np.random.randn(*shape).astype(np.float32))


# Public API


def install_torch_mock():
    """Install a comprehensive torch mock into sys.modules.

    No-op if real torch is already importable.
    """
    try:
        import torch  # noqa: F401

        logger.info("Real torch is available (version=%s) - mock not installed", torch.__version__)
        return  # Real torch available - nothing to do
    except Exception as exc:  # noqa: BLE001 - diagnostics: any import failure means we mock
        # IMPORTANT: print to stderr (not just logging.info, which pytest captures
        # and hides) so CI logs ALWAYS show WHY the mock was installed. A silent
        # fallback here previously masked an env-resolution bug (a CUDA torch
        # wheel that failed to import) for hours of log-archaeology.
        _msg = (
            f"[torch_mock] real torch import FAILED ({type(exc).__name__}: {exc}); "
            "installing numpy mock. If this is unexpected, the torch wheel in this "
            "env is broken/unimportable (e.g. wrong CUDA build)."
        )
        # pytest captures stdout/stderr, so ALSO write to a sentinel file that
        # CI can cat unconditionally -- this is what makes the diagnosis a
        # one-line grep instead of log-archaeology.
        print(_msg, file=sys.stderr)
        try:
            import os as _os

            with open(
                _os.environ.get("TORCH_MOCK_SENTINEL", "/tmp/torch_mock_active.txt"),
                "w",
            ) as _fh:
                _fh.write(_msg + "\n")
        except OSError:
            # Sentinel-file write is best-effort diagnostics only (read-only or
            # full /tmp, restricted CI sandbox, etc.). The stderr message above
            # already conveys why the mock was installed, so a failed write must
            # never abort mock installation or fail the test run.
            pass

    logger.info("Installing torch mock (real torch not available)")

    # Root module
    torch_mock = types.ModuleType("torch")
    torch_mock.Tensor = MockTensor
    torch_mock.device = MockDevice
    torch_mock.float32 = np.float32
    torch_mock.float64 = np.float64
    torch_mock.int32 = np.int32
    torch_mock.int64 = np.int64
    torch_mock.long = np.int64
    torch_mock.bool = np.bool_

    torch_mock.tensor = _tensor
    torch_mock.zeros = _zeros
    torch_mock.ones = _ones
    torch_mock.from_numpy = _from_numpy
    torch_mock.stack = _stack
    torch_mock.cat = _cat
    torch_mock.as_tensor = _tensor
    torch_mock.randint = _randint
    torch_mock.rand = _rand
    torch_mock.randn = _randn

    torch_mock.no_grad = _NoGrad
    torch_mock.inference_mode = _NoGrad
    torch_mock.manual_seed = lambda seed: None

    # torch.nn
    nn_mock = types.ModuleType("torch.nn")
    nn_mock.Parameter = MockParameter
    nn_mock.Module = MagicMock
    torch_mock.nn = nn_mock

    nn_functional_mock = types.ModuleType("torch.nn.functional")
    torch_mock.nn.functional = nn_functional_mock

    # torch.cuda
    cuda_mock = types.ModuleType("torch.cuda")
    cuda_mock.is_available = lambda: False
    cuda_mock.device_count = lambda: 0
    cuda_mock.manual_seed_all = lambda seed: None
    torch_mock.cuda = cuda_mock

    # torch.backends
    backends_mock = types.ModuleType("torch.backends")
    mps_mock = types.ModuleType("torch.backends.mps")
    mps_mock.is_available = lambda: False
    backends_mock.mps = mps_mock
    cudnn_mock = types.ModuleType("torch.backends.cudnn")
    cudnn_mock.deterministic = False
    cudnn_mock.benchmark = True
    backends_mock.cudnn = cudnn_mock
    torch_mock.backends = backends_mock

    # torch.amp
    amp_mock = types.ModuleType("torch.amp")
    amp_mock.autocast = MagicMock
    torch_mock.amp = amp_mock

    # Register in sys.modules
    sys.modules["torch"] = torch_mock
    sys.modules["torch.nn"] = nn_mock
    sys.modules["torch.nn.functional"] = nn_functional_mock
    sys.modules["torch.cuda"] = cuda_mock
    sys.modules["torch.backends"] = backends_mock
    sys.modules["torch.backends.mps"] = mps_mock
    sys.modules["torch.backends.cudnn"] = cudnn_mock
    sys.modules["torch.amp"] = amp_mock

    # torchvision
    torchvision_mock = types.ModuleType("torchvision")
    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_mock.transforms = torchvision_transforms
    sys.modules["torchvision"] = torchvision_mock
    sys.modules["torchvision.transforms"] = torchvision_transforms
