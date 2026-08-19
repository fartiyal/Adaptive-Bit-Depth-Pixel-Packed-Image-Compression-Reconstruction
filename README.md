# CPX — Adaptive Bit-Depth Image Codec
#Adaptive Lossless + Lossy Pixel-Packed Image Compression & Reconstruction
A small, self-contained implementation of the pipeline you sketched:

```
input image -> image analysis -> lossless / lossy -> adaptive bit packing
            -> CPX container -> decoder -> reconstruction (exact / CNN-enhanced)
            -> evaluation (compression ratio, PSNR/SSIM, runtime)
```

## Files

| File          | Purpose                                                          |
|---------------|-------------------------------------------------------------------|
| `bitpack.py`  | Vectorized arbitrary-bit-width (1–8 bit) packer/unpacker (numpy) |
| `codec.py`    | Image analysis + lossless/lossy encode/decode + CPX container    |
| `enhance.py`  | Optional CNN post-processing for lossy reconstructions           |
| `metrics.py`  | Compression ratio, PSNR, SSIM                                    |
| `demo.py`     | Runs every mode on a test image and prints a comparison table    |

## How each stage works

**Image analysis / unique-value analysis** (`analyze_image`): for each
channel, computes `min`, `max`, and the minimum number of bits needed
to represent `max - min` exactly. If every channel fits in ≤4 bits,
lossless is recommended (it's already compact); otherwise lossy is
recommended at the smallest allowed bit depth that covers the observed
range.

**Lossless mode**: stores `pixel - min` for each channel, packed at
exactly the bits required for that channel's range. Bit-exact
reconstruction, verified in testing (`np.array_equal(original, recon)
== True`). Real photographs that use the full 0–255 range won't
compress under this scheme (correctly — there's no redundancy to
exploit this way); low-dynamic-range images (masks, sprites, scans
with a limited value range) do compress, e.g. a 3-channel image with
values confined to a 15-level range packs into 4 bits/pixel instead of
8 (~2× ratio) with zero error.

**Lossy mode**: uniformly quantizes each channel to one of
`{1, 2, 4, 6, 8}` bits (2/4/16/64/256 levels) and packs the quantized
levels. Bit depth is either explicit (`bits=...`) or chosen by
analysis.

**Adaptive packing** (`bitpack.py`): a fully vectorized numpy bit
packer that handles any width from 1–8 bits with no python-level
per-pixel loop — expands each value to its binary digits, flattens,
and calls `np.packbits`.

**CPX container**: `magic | mode | channels | height | width |
(bits, min_val) per channel | packed payload`. Payload lengths are
derived from `height*width*bits`, so no explicit length fields are
needed.

**Decoder / reconstruction**: lossless reconstruction is exact
(add back the stored offset). Lossy reconstruction dequantizes to
0–255; `enhance.py` can then apply a small residual CNN
(`ResidualEnhanceCNN`, 3 conv layers, PyTorch) trained to correct the
quantization error, if PyTorch is installed. **The CNN ships
untrained** (`train_toy()` does a quick per-image overfit as a
demonstration of the integration point, not a real trained model) — if
you want this to generalize across images you'll need a real training
set of (original, quantized) pairs and a proper train/val split. If
PyTorch isn't installed, `enhance()` automatically falls back to a
lightweight non-CNN deblocking filter so the pipeline still runs
end-to-end.

**Evaluation** (`metrics.py`): compression ratio
(`original_bits / compressed_bits`), PSNR, and SSIM (via
scikit-image), plus wall-clock encode/decode timing in `demo.py`.

## Usage

```python
import numpy as np
from codec import encode, decode

img = np.array(...)  # HxWxC or HxW uint8

# Automatic mode selection
data, analysis = encode(img, mode="auto")
recon, mode = decode(data)

# Explicit lossless
data, _ = encode(img, mode="lossless")

# Explicit lossy at 4 bits/pixel
data, _ = encode(img, mode="lossy", bits=4)
recon, _ = decode(data)

# Optional CNN enhancement of a lossy reconstruction
import enhance
if enhance._HAVE_TORCH:
    model = enhance.ResidualEnhanceCNN(channels=img.shape[2])
    model = enhance.train_toy(model, img, recon, epochs=150)
    recon_enhanced = enhance.enhance(recon, model)
else:
    recon_enhanced = enhance.enhance(recon)  # fallback filter
```

Run the full comparison:

```bash
python demo.py                  # uses a bundled scikit-image sample image
python demo.py path/to/img.png  # or your own image
```

## Sample output (astronaut.png, 512×512×3, natural photo)

```
mode                     ratio  PSNR(dB)    SSIM  PSNR+CNN  SSIM+CNN  enc(s)  dec(s)
------------------------------------------------------------------------------------
lossless                  1.00       inf  1.0000       nan       nan  0.1024  0.0532
lossy-1bit +fallback enh   8.00     11.66  0.4697     12.01    0.4857  0.0032  0.0047
lossy-2bit +fallback enh   4.00     21.03  0.6760     21.73    0.7013  0.0114  0.0204
lossy-4bit +fallback enh   2.00     34.92  0.9112     35.59    0.9285  0.0112  0.0237
lossy-6bit                1.33     46.78  0.9910       nan       nan  0.0141  0.0307
lossy-8bit                1.00       inf  1.0000       nan       nan  0.0246  0.0340
```

(Lossless gives ratio 1.0 here because this is a full-dynamic-range
photo with no redundancy for this simple offset+minbits scheme to
exploit — try it on a low-dynamic-range image, e.g. a mask or sprite,
and you'll see real compression with `bit-exact reconstruction ==
True`.)

## Notes / limitations

- Lossless mode is a **minimum-bits-per-channel** scheme, not
  entropy coding — it won't beat PNG/WebP on natural photos, but it's
  simple, exact, and genuinely adaptive to each image's actual value
  range (useful for e.g. depth maps, label masks, sensor data with
  a known bounded range).
- Lossy quantization is **uniform** (not perceptually weighted); a
  natural extension would be per-channel min/max-aware quantization
  or a learned (vector) quantizer.
- The CNN enhancement stage is a real, working PyTorch module and
  training loop, but ships untrained for general use — see the
  docstring in `enhance.py` for what a production training setup
  would need.
