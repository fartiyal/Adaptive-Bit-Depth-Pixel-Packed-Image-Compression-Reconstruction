"""
metrics.py
----------
Evaluation utilities: compression ratio and reconstruction quality
(PSNR, SSIM).
"""

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


def compression_ratio(original_shape, original_dtype_bits: int, compressed_size_bytes: int) -> float:
    """Ratio of original size to compressed size (higher = better)."""
    n_elements = int(np.prod(original_shape))
    original_bits = n_elements * original_dtype_bits
    compressed_bits = compressed_size_bytes * 8
    return original_bits / compressed_bits


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB. Returns inf for identical images."""
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Structural similarity index, averaged over channels if multichannel."""
    if not _HAVE_SKIMAGE:
        raise ImportError("scikit-image is required for SSIM computation")
    original = np.squeeze(original)
    reconstructed = np.squeeze(reconstructed)
    multichannel = original.ndim == 3
    if multichannel:
        return _ssim(original, reconstructed, channel_axis=-1, data_range=255)
    return _ssim(original, reconstructed, data_range=255)


def summarize(original: np.ndarray, reconstructed: np.ndarray, compressed_size_bytes: int) -> dict:
    """Convenience wrapper returning all metrics in one dict."""
    ratio = compression_ratio(original.shape, 8, compressed_size_bytes)
    result = {
        "compression_ratio": ratio,
        "psnr_db": psnr(original, reconstructed),
        "compressed_bytes": compressed_size_bytes,
        "original_bytes": int(np.prod(original.shape)),
    }
    if _HAVE_SKIMAGE:
        result["ssim"] = ssim(original, reconstructed)
    return result
