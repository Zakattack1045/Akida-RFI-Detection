"""
prepare_dataset.py
------------------
Converts the existing RFI segmentation dataset into uint8 PNG format for Akida.

Output layout (mirrors current structure):
    Data/Akida_Dataset/
    ├── Combined_Dataset/
    │   ├── spectrograms/   <- uint8 grayscale PNG, 64x64, values 0-255
    │   └── masks/          <- binary PNG, 64x64, values 0 or 1
    └── Combined_Test/
        ├── spectrograms/
        └── masks/

Run from the FYP/ root directory:
    python Akida/prepare_dataset.py
"""

import os
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

HEIGHT = 64
WIDTH  = 64

SPLITS = {
    "Combined_Dataset": "Data/OUTPUTS/Combined_Dataset",
    "Combined_Test":    "Data/OUTPUTS/Combined_Test",
}

OUTPUT_ROOT = "Data/Akida_Dataset"

# ── Helpers ───────────────────────────────────────────────────────────────────

def convert_spectrogram(src_path: str, dst_path: str) -> None:
    """Load a spectrogram PNG, convert to uint8 grayscale 64x64, save."""
    img = Image.open(src_path).convert("L")                              # -> (H, W) uint8
    img = img.resize((WIDTH, HEIGHT), resample=Image.BILINEAR)           # -> 64x64
    img.save(dst_path)


def convert_mask(src_path: str, dst_path: str) -> None:
    """Load a mask PNG, resize with nearest-neighbour, binarise to 0/255, save."""
    mask = Image.open(src_path).convert("L")                             # -> (H, W) uint8
    mask = mask.resize((WIDTH, HEIGHT), resample=Image.NEAREST)          # preserve edges
    arr  = np.array(mask, dtype=np.uint8)
    # Normalise to 0/1 first, then scale to 0/255 for visual compatibility
    if arr.max() > 1:
        arr = (arr > 127).astype(np.uint8)
    arr = arr * 255                                                        # 0/1 -> 0/255
    Image.fromarray(arr, mode="L").save(dst_path)


def process_split(split_name: str, src_root: str, dst_root: str) -> None:
    spec_src = os.path.join(src_root, "spectrograms")
    mask_src = os.path.join(src_root, "masks")
    spec_dst = os.path.join(dst_root, "spectrograms")
    mask_dst = os.path.join(dst_root, "masks")

    os.makedirs(spec_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    filenames = sorted(
        f for f in os.listdir(spec_src)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not filenames:
        print(f"  [WARN] No images found in {spec_src}")
        return

    mask_values_seen = set()
    skipped = 0

    for fname in filenames:
        out_spec = os.path.join(spec_dst, fname)
        out_mask = os.path.join(mask_dst, fname)

        # Skip if both outputs already exist (idempotent)
        if os.path.exists(out_spec) and os.path.exists(out_mask):
            skipped += 1
            continue

        convert_spectrogram(os.path.join(spec_src, fname), out_spec)

        mask_path = os.path.join(mask_src, fname)
        if os.path.exists(mask_path):
            convert_mask(mask_path, out_mask)
            # Sanity-check: sample unique values from saved mask
            arr = np.array(Image.open(out_mask))
            mask_values_seen.update(arr.flatten().tolist())
        else:
            print(f"  [WARN] No mask for {fname}, skipping mask")

    converted = len(filenames) - skipped
    print(f"  {split_name}: {converted} converted, {skipped} already existed")
    if mask_values_seen:
        print(f"    Mask pixel values: {sorted(mask_values_seen)}  (expected: [0, 255])")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Output size: {HEIGHT}x{WIDTH}")
    print(f"Output root: {OUTPUT_ROOT}\n")

    for split_name, src_root in SPLITS.items():
        if not os.path.isdir(src_root):
            print(f"[SKIP] Source not found: {src_root}")
            continue
        dst_root = os.path.join(OUTPUT_ROOT, split_name)
        print(f"Processing {split_name} ...")
        process_split(split_name, src_root, dst_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
