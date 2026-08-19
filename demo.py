"""
demo.py
-------
End-to-end demonstration of the CPX pipeline:

    image -> analysis -> lossless / lossy(bits) -> CPX bytes -> decode
    -> (optional CNN enhancement for lossy) -> metrics

Run:
    python demo.py [path/to/image.png]

If no image path is given, a bundled scikit-image sample image is used.
"""

import sys
import time

import numpy as np

from codec import encode, decode, analyze_image, MODE_LOSSLESS, MODE_LOSSY, ALLOWED_LOSSY_BITS
from metrics import summarize
import enhance


def load_image(path: str = None) -> np.ndarray:
    if path:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    from skimage import data
    return data.astronaut()  # 512x512x3 uint8 sample image, ships with skimage


def run_case(img: np.ndarray, mode: str, bits=None, use_enhancement: bool = False) -> dict:
    t0 = time.perf_counter()
    cpx_bytes, analysis = encode(img, mode=mode, bits=bits)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    recon, decoded_mode = decode(cpx_bytes)
    t_decode = time.perf_counter() - t0

    label = "lossless" if decoded_mode == MODE_LOSSLESS else f"lossy-{bits}bit"

    if use_enhancement and decoded_mode == MODE_LOSSY:
        t0 = time.perf_counter()
        if enhance._HAVE_TORCH:
            model = enhance.ResidualEnhanceCNN(channels=img.shape[2])
            model = enhance.train_toy(model, img, recon, epochs=150)
            recon_enh = enhance.enhance(recon, model)
        else:
            recon_enh = enhance.enhance(recon)
        t_enhance = time.perf_counter() - t0
        label += " +CNN enhance" if enhance._HAVE_TORCH else " +fallback enhance"
    else:
        recon_enh = None
        t_enhance = 0.0

    metrics = summarize(img, recon, len(cpx_bytes))
    row = {
        "label": label,
        "encode_s": t_encode,
        "decode_s": t_decode,
        **metrics,
    }

    if recon_enh is not None:
        enh_metrics = summarize(img, recon_enh, len(cpx_bytes))
        row["psnr_enhanced_db"] = enh_metrics["psnr_db"]
        row["ssim_enhanced"] = enh_metrics.get("ssim")
        row["enhance_s"] = t_enhance

    return row


def print_table(rows):
    cols = ["label", "compression_ratio", "psnr_db", "ssim",
            "psnr_enhanced_db", "ssim_enhanced", "encode_s", "decode_s"]
    header = f"{'mode':<22}{'ratio':>8}{'PSNR(dB)':>10}{'SSIM':>8}{'PSNR+CNN':>10}{'SSIM+CNN':>10}{'enc(s)':>8}{'dec(s)':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<22}"
            f"{r['compression_ratio']:>8.2f}"
            f"{r['psnr_db']:>10.2f}"
            f"{r.get('ssim', float('nan')):>8.4f}"
            f"{r.get('psnr_enhanced_db', float('nan')):>10.2f}"
            f"{r.get('ssim_enhanced', float('nan')):>10.4f}"
            f"{r['encode_s']:>8.4f}"
            f"{r['decode_s']:>8.4f}"
        )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    img = load_image(path)
    print(f"Loaded image: shape={img.shape}, dtype={img.dtype}\n")

    analysis = analyze_image(img)
    print("Image analysis (per channel):")
    for i, s in enumerate(analysis.per_channel):
        print(f"  channel {i}: min={s.min_val} max={s.max_val} bits_needed={s.bits_needed}")
    print(f"Recommended mode: {'lossless' if analysis.recommended_mode == MODE_LOSSLESS else 'lossy'}"
          f"{f' @ {analysis.recommended_bits} bits' if analysis.recommended_bits else ''}\n")

    rows = [run_case(img, mode="lossless")]
    for bits in ALLOWED_LOSSY_BITS:
        rows.append(run_case(img, mode="lossy", bits=bits, use_enhancement=(bits <= 4)))

    print_table(rows)


if __name__ == "__main__":
    main()
