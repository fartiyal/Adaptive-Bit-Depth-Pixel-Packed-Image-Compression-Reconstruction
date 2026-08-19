"""
bitpack.py
----------
Vectorized (numpy-based) arbitrary bit-width packing / unpacking.

Given an array of integer values that are known to fit in `nbits` bits
each (0 <= value < 2**nbits), pack_bits() tightly packs them MSB-first
into a byte string with no wasted space between values (only the final
byte may be zero-padded). unpack_bits() reverses the process.

Supports any nbits in [1, 8], which covers the 1/2/4/6/8-bit modes
used by the CPX format.
"""

import numpy as np


def pack_bits(values: np.ndarray, nbits: int) -> bytes:
    """Pack an array of small integers into a tightly-packed byte string.

    Args:
        values: 1-D array of non-negative integers, each < 2**nbits.
        nbits: number of bits used to represent each value (1-8).

    Returns:
        Packed bytes (MSB-first, zero-padded at the very end if needed).
    """
    if nbits < 1 or nbits > 8:
        raise ValueError("nbits must be between 1 and 8")

    values = np.asarray(values, dtype=np.uint32).reshape(-1)
    if values.size == 0:
        return b""

    # Expand each value into its `nbits` binary digits, MSB first.
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint32)
    bits = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    flat_bits = bits.reshape(-1)

    pad = (-flat_bits.size) % 8
    if pad:
        flat_bits = np.concatenate([flat_bits, np.zeros(pad, dtype=np.uint8)])

    packed = np.packbits(flat_bits)
    return packed.tobytes()


def unpack_bits(data: bytes, nbits: int, count: int) -> np.ndarray:
    """Unpack `count` integers of width `nbits` from a packed byte string.

    Args:
        data: packed bytes as produced by pack_bits().
        nbits: bit width each value was packed with.
        count: number of values to extract.

    Returns:
        1-D uint32 numpy array of length `count`.
    """
    if nbits < 1 or nbits > 8:
        raise ValueError("nbits must be between 1 and 8")
    if count == 0:
        return np.zeros(0, dtype=np.uint32)

    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    needed = count * nbits
    if bits.size < needed:
        raise ValueError("Not enough packed data for requested count/nbits")
    bits = bits[:needed].reshape(count, nbits)

    weights = (1 << np.arange(nbits - 1, -1, -1)).astype(np.uint32)
    values = (bits.astype(np.uint32) * weights[None, :]).sum(axis=1)
    return values


def bits_required(max_value: int) -> int:
    """Minimum number of bits needed to represent integers in [0, max_value]."""
    if max_value <= 0:
        return 1
    return int(max_value).bit_length()
