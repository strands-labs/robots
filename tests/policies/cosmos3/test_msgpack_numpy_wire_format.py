"""Tests for the vendored NumPy-aware msgpack wire codec used by the Cosmos 3 client.

The Cosmos 3 RoboLab policy server speaks msgpack with NumPy arrays encoded via
the ``_msgpack_numpy`` codec. The client (``cosmos3.client``) packs observations
and unpacks action chunks with this codec, so its round-trip fidelity is part of
the on-the-wire contract: a regression here silently corrupts every action the
remote policy returns. These tests pin that contract end to end.
"""

import msgpack
import numpy as np
import pytest

from strands_robots.policies.cosmos3 import _msgpack_numpy as mnp


def test_ndarray_round_trip_preserves_dtype_shape_values():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = mnp.unpackb(mnp.packb(arr))
    assert isinstance(out, np.ndarray)
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    assert np.array_equal(out, arr)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8"])
def test_ndarray_round_trip_across_dtypes(dtype):
    arr = (np.arange(6) - 2).astype(dtype).reshape(2, 3)
    out = mnp.unpackb(mnp.packb(arr))
    assert out.dtype == np.dtype(dtype)
    assert np.array_equal(out, arr)


def test_high_dimensional_array_round_trip():
    arr = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    out = mnp.unpackb(mnp.packb(arr))
    assert out.shape == (4, 5, 6)
    assert np.array_equal(out, arr)


def test_zero_dimensional_array_round_trip():
    arr = np.array(7.0, dtype=np.float32)
    out = mnp.unpackb(mnp.packb(arr))
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert float(out) == 7.0


def test_non_contiguous_array_round_trip():
    # A transposed view is not C-contiguous; tobytes() must capture logical order.
    arr = np.arange(6, dtype=np.float32).reshape(2, 3).T
    out = mnp.unpackb(mnp.packb(arr))
    assert out.shape == (3, 2)
    assert np.array_equal(out, arr)


def test_numpy_generic_round_trip_preserves_scalar_type():
    # float16 / uint8 are not Python scalar subclasses, so they exercise the
    # __npgeneric__ encode/decode branch and must come back as numpy scalars.
    for value in (np.float16(1.5), np.uint8(200), np.int8(-5), np.float32(2.5)):
        out = mnp.unpackb(mnp.packb(value))
        assert isinstance(out, np.generic)
        assert out.dtype == value.dtype
        assert out == value


def test_nested_dict_with_arrays_round_trip():
    payload = {"image": np.ones((2, 2), dtype=np.uint8), "step": 5, "task": "pick"}
    out = mnp.unpackb(mnp.packb(payload))
    assert out["step"] == 5
    assert out["task"] == "pick"
    assert np.array_equal(out["image"], payload["image"])


def test_non_numpy_payload_passes_through_unchanged():
    payload = {"a": 1, "b": [1, 2, 3], "c": "hi"}
    assert mnp.unpackb(mnp.packb(payload)) == payload


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_complex_dtype_is_rejected(dtype):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mnp.packb(np.zeros(2, dtype=dtype))


def test_object_dtype_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mnp.packb(np.array([object()], dtype=object))


def test_void_structured_dtype_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mnp.packb(np.zeros(2, dtype=[("a", "f4")]))


def test_pack_array_returns_non_numpy_inputs_unchanged():
    assert mnp.pack_array(5) == 5
    assert mnp.pack_array("hello") == "hello"


def test_unpack_array_returns_plain_mapping_unchanged():
    assert mnp.unpack_array({"x": 1}) == {"x": 1}


def test_streaming_packer_and_unpacker_round_trip():
    packer = mnp.Packer()
    buffer = packer.pack(np.arange(3, dtype=np.int32)) + packer.pack({"x": np.ones(2, dtype=np.uint8)})

    unpacker = mnp.Unpacker()
    unpacker.feed(buffer)
    items = list(unpacker)

    assert len(items) == 2
    assert np.array_equal(items[0], np.array([0, 1, 2], dtype=np.int32))
    assert np.array_equal(items[1]["x"], np.array([1, 1], dtype=np.uint8))


def test_codec_is_interoperable_with_plain_msgpack_for_scalars():
    # numpy float64 serializes as a native msgpack float, so a vanilla decoder
    # still reads the value: the codec only adds support, never breaks the base
    # protocol for primitives the server may send without the ndarray envelope.
    blob = mnp.packb({"reward": np.float64(1.25)})
    assert msgpack.unpackb(blob, raw=False) == {"reward": 1.25}


def test_decoded_array_is_writable_and_owns_its_data():
    # A ``np.ndarray(buffer=...)`` view over the transient msgpack recv bytes is
    # read-only and non-owning, so normalizing a decoded observation in place or
    # handing a decoded action chunk to ``torch.from_numpy`` (zero-copy) crashes
    # or hits torch's "not writable -> undefined behavior" hazard. The decode
    # must yield a writable, owning array (parity with the VERA packer and the
    # inference protocol). Fails before the copy fix.
    action = np.arange(8 * 10, dtype=np.float32).reshape(8, 10)
    out = mnp.unpackb(mnp.packb({"action": action}))["action"]
    assert out.flags.writeable, "decoded array must be writable"
    assert out.flags.owndata, "decoded array must own its data (not alias recv bytes)"


def test_decoded_array_supports_in_place_mutation_without_aliasing():
    # In-place ops (e.g. image /= 255) must succeed and must not be a view over
    # the transient wire buffer, so mutating the decoded array cannot corrupt a
    # subsequent decode of the same wire payload.
    arr = np.ones((2, 3), dtype=np.float32)
    blob = mnp.packb({"image": arr})
    first = mnp.unpackb(blob)["image"]
    first += 5.0  # read-only view would raise here
    second = mnp.unpackb(blob)["image"]
    assert np.array_equal(second, arr), "decoded arrays must be independent copies"
