# ============================
# convert_onnx_to_akida.py
# PURPOSE: ONNX model -> QuantizeML calibration quantization -> Akida v2 model.
#
# Pipeline:
#   1. Load ONNX model (exported by train_akida_v2.py)
#   2. Load calibration samples (float32 [0,1], training set)
#   3. Quantize with QuantizeML (calibration-based PTQ — no QAT needed at 8-bit)
#   4. Convert to Akida model via cnn2snn.convert()
#   5. Evaluate on real test set
#   6. Save Akida model to .fbz
#
# Key difference from convert_to_akida.py:
#   - Uses QuantizeML instead of cnn2snn.quantize()
#   - Supports residual Add, DepthwiseConv2D, Conv2DTranspose (Akida v2 via ONNX)
#   - input_dtype='uint8' in QuantizationParams replaces input_scaling=(255, 0)
# ============================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import onnx
import tensorflow as tf
import cnn2snn
from PIL import Image
from quantizeml.models import quantize, QuantizationParams


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_SAVE_ROOT = "Models/V3"
ONNX_MODEL_PATH = os.path.join(MODEL_SAVE_ROOT,"Akida_ONNIX.onnx")
TRAIN_DIR       = "Data/Akida_Dataset/Combined_Dataset"
TEST_DIR        = "Data/Akida_Dataset/Combined_Test"
AKIDA_SAVE      = os.path.join(MODEL_SAVE_ROOT,"Akida_ONNIX.fbz")

INPUT_H = 64
INPUT_W = 64
OUT_H   = INPUT_H
OUT_W   = INPUT_W

# QuantizeML calibration settings (no QAT — PTQ at 8-bit is sufficient)
WEIGHT_BITS  = 8
ACTIV_BITS   = 8
N_CALIB      = 256   # calibration samples from training set

# Sample image saving
SAVE_SAMPLES = True
N_SAMPLES    = 8
SAMPLES_DIR  = os.path.join(MODEL_SAVE_ROOT,"Samples")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_test_sample(spec_path, mask_path):
    """
    Load one spectrogram/mask pair for Akida inference.
      spectrogram: (1, INPUT_H, INPUT_W, 1) uint8,  values 0-255
      mask:        (OUT_H, OUT_W)            int32,  values 0 or 1
    """
    spec = np.array(
        Image.open(spec_path).convert("L"),
        dtype=np.uint8
    )[None, ..., None]  # (1, 64, 64, 1)

    mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if mask_raw.max() > 1:
        mask_raw = (mask_raw > 127).astype(np.uint8)
    if mask_raw.shape != (OUT_H, OUT_W):
        mask_raw = np.array(
            Image.fromarray(mask_raw).resize((OUT_W, OUT_H), resample=Image.NEAREST),
            dtype=np.int32
        )
    else:
        mask_raw = mask_raw.astype(np.int32)
    return spec, mask_raw


def collect_test_paths(test_dir):
    spec_dir = os.path.join(test_dir, "spectrograms")
    mask_dir = os.path.join(test_dir, "masks")
    fnames = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))
    return [
        (os.path.join(spec_dir, f), os.path.join(mask_dir, f))
        for f in fnames
        if os.path.exists(os.path.join(mask_dir, f))
    ]


def load_calibration_data(train_dir, n_samples):
    """
    Load N training spectrograms as float32 [0,1] for QuantizeML calibration.
    QuantizeML uses these to determine per-layer activation quantisation ranges.
    """
    spec_dir = os.path.join(train_dir, "spectrograms")
    fnames = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))[:n_samples]
    specs = []
    for fname in fnames:
        spec = np.array(
            Image.open(os.path.join(spec_dir, fname)).convert("L"),
            dtype=np.float32
        )[..., None] / 255.0   # (64, 64, 1), values 0-1
        specs.append(spec)
    return np.array(specs, dtype=np.float32)  # (N, 64, 64, 1)


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_sample_images(specs_u8, masks, preds, save_dir, prefix="sample"):
    """
    Save one PNG per sample: [Input | Ground Truth | Prediction | Diff]
    Diff colour key: TP=green, FP=red, FN=yellow, TN=black.

    specs_u8: (N, 1, H, W, 1) uint8  — Akida input format
    masks:    (N, H, W)        int32
    preds:    (N, H, W)        int
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(specs_u8)):
        spec_img = specs_u8[i, 0, :, :, 0].astype(np.float32) / 255.0
        mask     = masks[i]
        pred     = preds[i]

        diff = np.zeros((*mask.shape, 3), dtype=np.float32)
        diff[(pred == 1) & (mask == 1)] = [0.0, 1.0, 0.0]  # TP green
        diff[(pred == 1) & (mask == 0)] = [1.0, 0.0, 0.0]  # FP red
        diff[(pred == 0) & (mask == 1)] = [1.0, 1.0, 0.0]  # FN yellow

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(spec_img, cmap="viridis"); axes[0].set_title("Input");         axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Prediction");   axes[2].axis("off")
        axes[3].imshow(diff); axes[3].set_title("Diff  (TP=G  FP=R  FN=Y)");         axes[3].axis("off")

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(akida_model, test_pairs):
    total_pixels = 0
    correct = 0
    tp = fp = tn = fn = 0

    sample_specs, sample_masks, sample_preds = [], [], []

    for spec_path, mask_path in test_pairs:
        x_u8, y_true = load_test_sample(spec_path, mask_path)

        y_akida = akida_model.predict(x_u8)           # (1, OUT_H, OUT_W, 2)
        pred    = np.argmax(y_akida, axis=-1)[0]       # (OUT_H, OUT_W)

        total_pixels += y_true.size
        correct      += (pred == y_true).sum()
        tp += ((pred == 1) & (y_true == 1)).sum()
        fp += ((pred == 1) & (y_true == 0)).sum()
        tn += ((pred == 0) & (y_true == 0)).sum()
        fn += ((pred == 0) & (y_true == 1)).sum()

        if SAVE_SAMPLES and len(sample_specs) < N_SAMPLES:
            sample_specs.append(x_u8)
            sample_masks.append(y_true)
            sample_preds.append(pred)

    accuracy    = correct / total_pixels
    precision   = tp / (tp + fp + 1e-8)
    recall      = tp / (tp + fn + 1e-8)
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    iou         = tp / (tp + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    metrics = {
        "accuracy":    accuracy,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "iou":         iou,
        "specificity": specificity,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    if SAVE_SAMPLES and sample_specs:
        print(f"\nSaving {len(sample_specs)} sample images to {SAMPLES_DIR} ...")
        save_sample_images(
            np.array(sample_specs),
            np.array(sample_masks),
            np.array(sample_preds),
            SAMPLES_DIR,
            prefix="akida_v2",
        )

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1) Load ONNX model
    print(f"Loading ONNX model from {ONNX_MODEL_PATH} ...")
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    # quantizeml 1.2.3 supports up to ir_version 10; tf2onnx >=1.16 exports ir_version 11.
    if onnx_model.ir_version > 10:
        print(f"  Capping ir_version {onnx_model.ir_version} -> 10 for quantizeml compatibility.")
        onnx_model.ir_version = 10
    print("ONNX model loaded.")

    # 2) Load calibration data
    print(f"\nLoading {N_CALIB} calibration samples from {TRAIN_DIR} ...")
    calib_samples = load_calibration_data(TRAIN_DIR, N_CALIB)
    print(f"Calibration data: shape={calib_samples.shape}, dtype={calib_samples.dtype}")

    # 3) QuantizeML calibration-based quantization
    # input_dtype='uint8': inserts an InputQuantizer so the final Akida model
    # accepts raw uint8 images (0-255) — replaces input_scaling=(255, 0) from old path.
    # Calibration samples remain float32 [0,1] to match the trained model's expected input.
    print(f"\nQuantizing ONNX model (weights={WEIGHT_BITS}-bit, activations={ACTIV_BITS}-bit) ...")
    qparams = QuantizationParams(
        weight_bits=WEIGHT_BITS,
        activation_bits=ACTIV_BITS,
        input_weight_bits=WEIGHT_BITS,
        input_dtype="uint8",
    )

    q_model = quantize(
        onnx_model,
        qparams=qparams,
        samples=calib_samples,
        num_samples=N_CALIB,
        batch_size=8,
    )
    print("Quantization complete.")

    # 4) Convert to Akida model
    # No input_scaling needed: the InputQuantizer from QuantizeML handles uint8->float32.
    print("\nConverting to Akida model ...")
    akida_model = cnn2snn.convert(q_model)
    print("Akida model created.")
    print(akida_model.summary())

    # 5) Evaluate on test set
    print(f"\nLoading test data from {TEST_DIR} ...")
    test_pairs = collect_test_paths(TEST_DIR)
    print(f"Found {len(test_pairs)} test samples.")

    print("Running inference ...")
    metrics = evaluate(akida_model, test_pairs)

    print("\n── Results ───────────────────────────────────")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  IoU (RFI):   {metrics['iou']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

    # 6) Save Akida model
    akida_model.save(AKIDA_SAVE)
    print(f"\nSaved Akida model to {AKIDA_SAVE}")


if __name__ == "__main__":
    main()
