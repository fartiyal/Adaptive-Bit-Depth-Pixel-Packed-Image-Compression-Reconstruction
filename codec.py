"""
codec.py
--------
CPX: a small adaptive image codec following the pipeline:

    input image -> image analysis -> {lossless | lossy} -> adaptive
    bit packing -> CPX container -> decoder -> reconstruction
    (exact for lossless, optional CNN enhancement for lossy)

Lossless mode:
    Per channel, the encoder finds min/max of that channel and computes
    the *minimum number of bits* needed to represent (max - min).
    Values are stored as (pixel - min) packed at that bit width, so the
    reconstruction is bit-exact.

Lossy mode:
    Per channel, pixel values are uniformly quantized to one of the
    allowed bit depths {1, 2, 4, 6, 8} (i.e. 2, 4, 16, 64, 256 levels)
    and the quantized levels are bit-packed. Reconstruction dequantizes
    back to 0-255; this introduces quantization error that a CNN
    enhancement stage (see enhance.py) can partially recover.

Container format (all integers little-endian, unsigned):
    magic        4 bytes   b"CPX1"
    mode         1 byte    0 = lossless, 1 = lossy
    channels     1 byte
    height       4 bytes   uint32
    width        4 bytes   uint32
    per-channel header, repeated `channels` times:
        bits     1 byte    bits used to pack this channel (1-8)
        min_val  1 byte    offset subtracted before packing (lossless only,
                            0 and unused in lossy mode)
    payload      concatenation of each channel's packed bit-stream
                 (length is derivable from height*width*bits, no
                 explicit length field needed)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from bitpack import pack_bits, unpack_bits, bits_required

MAGIC = b"CPX1"
MODE_LOSSLESS = 0
MODE_LOSSY = 1
ALLOWED_LOSSY_BITS = (1, 2, 4, 6, 8)


@dataclass
class ChannelStats:
    min_val: int
    max_val: int
    bits_needed: int


@dataclass
class ImageAnalysis:
    height: int
    width: int
    channels: int
    per_channel: list  # List[ChannelStats]
    recommended_mode: int
    recommended_bits: Optional[int]  # only meaningful for lossy


def analyze_image(img: np.ndarray) -> ImageAnalysis:
    """Inspect an image and recommend a compression mode.

    Performs the "unique-value analysis" step: for each channel, finds
    the value range and the minimum number of bits required to encode
    it losslessly. If every channel already fits in <= 4 bits losslessly,
    lossless mode is recommended (it is already very compact). Otherwise
    lossy mode is recommended at a bit depth chosen from the allowed set
    based on how much dynamic range the image actually uses.
    """
    img = _ensure_hwc(img)
    h, w, c = img.shape

    stats = []
    max_bits_needed = 0
    for ch in range(c):
        channel = img[:, :, ch]
        mn, mx = int(channel.min()), int(channel.max())
        nb = bits_required(mx - mn)
        stats.append(ChannelStats(min_val=mn, max_val=mx, bits_needed=nb))
        max_bits_needed = max(max_bits_needed, nb)

    if max_bits_needed <= 4:
        rec_mode = MODE_LOSSLESS
        rec_bits = None
    else:
        rec_mode = MODE_LOSSY
        # pick the smallest allowed bit depth that still covers most of
        # the observed dynamic range reasonably well
        rec_bits = min(b for b in ALLOWED_LOSSY_BITS if b >= min(max_bits_needed, 8))

    return ImageAnalysis(
        height=h, width=w, channels=c,
        per_channel=stats,
        recommended_mode=rec_mode,
        recommended_bits=rec_bits,
    )


def _ensure_hwc(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def encode(img: np.ndarray, mode: str = "auto", bits: Optional[int] = None) -> Tuple[bytes, ImageAnalysis]:
    """Encode an image into the CPX format.

    Args:
        img: HxW or HxWxC uint8 array.
        mode: "auto", "lossless", or "lossy".
        bits: for lossy mode, bit depth in {1,2,4,6,8}. Ignored for
              lossless mode. If mode="auto" and the analysis recommends
              lossy, `bits` overrides the recommendation if provided.

    Returns:
        (cpx_bytes, analysis) tuple.
    """
    img = _ensure_hwc(img)
    analysis = analyze_image(img)
    h, w, c = analysis.height, analysis.width, analysis.channels

    if mode == "auto":
        chosen_mode = analysis.recommended_mode
        chosen_bits = bits if bits is not None else analysis.recommended_bits
    elif mode == "lossless":
        chosen_mode = MODE_LOSSLESS
        chosen_bits = None
    elif mode == "lossy":
        chosen_mode = MODE_LOSSY
        chosen_bits = bits if bits is not None else 6
    else:
        raise ValueError("mode must be 'auto', 'lossless', or 'lossy'")

    if chosen_mode == MODE_LOSSY and chosen_bits not in ALLOWED_LOSSY_BITS:
        raise ValueError(f"lossy bits must be one of {ALLOWED_LOSSY_BITS}")

    header = bytearray()
    header += MAGIC
    header += bytes([chosen_mode])
    header += bytes([c])
    header += int(h).to_bytes(4, "little")
    header += int(w).to_bytes(4, "little")

    payload = bytearray()

    for ch in range(c):
        channel = img[:, :, ch]
        if chosen_mode == MODE_LOSSLESS:
            stat = analysis.per_channel[ch]
            ch_bits = max(1, stat.bits_needed)
            min_val = stat.min_val
            values = (channel.astype(np.int32) - min_val).reshape(-1)
            packed = pack_bits(values, ch_bits)
            header += bytes([ch_bits, min_val & 0xFF])
        else:
            ch_bits = chosen_bits
            levels = 2 ** ch_bits
            scale = 255.0 / (levels - 1) if levels > 1 else 255.0
            quant = np.round(channel.astype(np.float32) / scale).clip(0, levels - 1).astype(np.uint32)
            packed = pack_bits(quant.reshape(-1), ch_bits)
            header += bytes([ch_bits, 0])
        payload += packed

    return bytes(header) + bytes(payload), analysis


def decode(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode CPX bytes back into an image array.

    Returns:
        (image, mode) where image is HxWxC uint8 and mode is
        MODE_LOSSLESS or MODE_LOSSY.
    """
    if data[:4] != MAGIC:
        raise ValueError("Not a valid CPX file (bad magic)")

    mode = data[4]
    channels = data[5]
    height = int.from_bytes(data[6:10], "little")
    width = int.from_bytes(data[10:14], "little")

    offset = 14
    ch_headers = []
    for _ in range(channels):
        ch_bits = data[offset]
        min_val = data[offset + 1]
        ch_headers.append((ch_bits, min_val))
        offset += 2

    n_pixels = height * width
    out = np.zeros((height, width, channels), dtype=np.uint8)

    for ch, (ch_bits, min_val) in enumerate(ch_headers):
        n_bytes = -(-(n_pixels * ch_bits) // 8)  # ceil division
        chunk = data[offset:offset + n_bytes]
        offset += n_bytes
        values = unpack_bits(chunk, ch_bits, n_pixels)

        if mode == MODE_LOSSLESS:
            pixels = values.astype(np.int32) + min_val
        else:
            levels = 2 ** ch_bits
            scale = 255.0 / (levels - 1) if levels > 1 else 255.0
            pixels = np.round(values.astype(np.float32) * scale)

        out[:, :, ch] = np.clip(pixels, 0, 255).reshape(height, width).astype(np.uint8)

    return out, mode


def compressed_size_bits(analysis: ImageAnalysis, mode: int, bits: Optional[int]) -> int:
    """Compute the payload size (in bits) a given mode/bits choice would produce,
    without actually encoding -- useful for quickly comparing options."""
    n_pixels = analysis.height * analysis.width
    total = 0
    for ch in range(analysis.channels):
        if mode == MODE_LOSSLESS:
            ch_bits = max(1, analysis.per_channel[ch].bits_needed)
        else:
            ch_bits = bits
        total += n_pixels * ch_bits
    return total
