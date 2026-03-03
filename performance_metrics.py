# ============================
# performance_metrics.py
# PURPOSE: Side-by-side performance evaluation of the CNN and Akida SNN models
#          on the held-out test set.
#
# Metrics computed per model (accumulated globally over all test pixels):
#   Accuracy    = (TP + TN) / total_pixels
#   Precision   = TP / (TP + FP + ε)
#   Recall      = TP / (TP + FN + ε)
#   F1 Score    = 2 · P · R / (P + R + ε)
#   IoU         = TP / (TP + FP + FN + ε)   [RFI class only]
#   Specificity = TN / (TN + FP + ε)
#
# Also reported:
#   Inference time — mean milliseconds per image (wall-clock, software simulation)
#
# Input formats:
#   CNN   — float32 [0, 1]   (H, W, 1)
#   Akida — uint8  [0, 255]  (H, W, 1)
# ============================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import numpy as np
import tensorflow as tf
from PIL import Image


# ── Config ─────────────────────────────────────────────────────────────────────

CNN_PATH = "Models/Akida_v1.keras"
SNN_PATH = "Models/Akida_v1.fbz"     # run convert_to_akida.py first to generate
TEST_DIR = "Data/Akida_Dataset/Combined_Test"
N_EVAL   = None   # None = use ALL test samples; set to an int to limit

EPS = 1e-8   # small constant to avoid division by zero (matches convert_to_akida.py)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_samples(test_dir, n=None):
    """
    Load test spectrograms as float32 [0, 1] and binary int32 masks.

    Returns
    -------
    x_float : ndarray  (N, H, W, 1)  float32
    y_masks : ndarray  (N, H, W)     int32   (values 0 or 1)
    """
    spec_dir = os.path.join(test_dir, "spectrograms")
    mask_dir = os.path.join(test_dir, "masks")
    fnames = sorted(
        f for f in os.listdir(spec_dir) if f.lower().endswith(".png")
    )
    if n is not None:
        fnames = fnames[:n]

    specs, masks = [], []
    for fname in fnames:
        spec = np.array(
            Image.open(os.path.join(spec_dir, fname)).convert("L"),
            dtype=np.float32,
        )[..., None] / 255.0   # (H, W, 1), values in [0, 1]

        mask_raw = np.array(
            Image.open(os.path.join(mask_dir, fname)).convert("L"),
            dtype=np.uint8,
        )
        if mask_raw.max() > 1:
            mask_raw = (mask_raw > 127).astype(np.int32)
        else:
            mask_raw = mask_raw.astype(np.int32)

        specs.append(spec)
        masks.append(mask_raw)

    return np.array(specs, dtype=np.float32), np.array(masks, dtype=np.int32)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(preds, masks):
    """
    Compute segmentation metrics from pixel-level predictions.

    Parameters
    ----------
    preds : ndarray  (N, H, W)  int   — predicted class (0 or 1)
    masks : ndarray  (N, H, W)  int   — ground-truth class (0 or 1)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, iou, specificity
    """
    preds = preds.astype(np.int32)
    masks = masks.astype(np.int32)

    tp = int(np.sum((preds == 1) & (masks == 1)))
    fp = int(np.sum((preds == 1) & (masks == 0)))
    tn = int(np.sum((preds == 0) & (masks == 0)))
    fn = int(np.sum((preds == 0) & (masks == 1)))

    total      = tp + fp + tn + fn
    accuracy   = (tp + tn) / (total + EPS)
    precision  = tp / (tp + fp + EPS)
    recall     = tp / (tp + fn + EPS)
    f1         = 2 * precision * recall / (precision + recall + EPS)
    iou        = tp / (tp + fp + fn + EPS)
    specificity = tn / (tn + fp + EPS)

    return {
        "accuracy":    accuracy,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "iou":         iou,
        "specificity": specificity,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ── CNN inference ──────────────────────────────────────────────────────────────

def run_cnn(model, x_float):
    """
    Run the CNN on all samples and measure wall-clock time.

    Returns
    -------
    preds         : ndarray  (N, H, W)  int
    ms_per_image  : float
    """
    t0     = time.perf_counter()
    logits = model.predict(x_float, verbose=0)   # (N, H, W, 2)
    elapsed = time.perf_counter() - t0

    preds = np.argmax(logits, axis=-1).astype(np.int32)   # (N, H, W)
    ms_per_image = elapsed / len(x_float) * 1000.0
    return preds, ms_per_image


# ── Akida SNN inference ────────────────────────────────────────────────────────

def run_akida(snn_path, x_uint8 : np.ndarray):
    """
    Load the Akida SNN model, run inference on all samples, measure time.

    Returns
    -------
    preds         : ndarray  (N, H, W)  int    — or None on error
    ms_per_image  : float                       — or None on error
    """
    try:
        import akida
    except ImportError:
        print("  [WARNING] akida package not importable — skipping SNN evaluation.")
        return None, None

    if not os.path.exists(snn_path):
        print(f"  [WARNING] {snn_path} not found — run convert_to_akida.py first.")
        return None, None

    print(f"  Loading SNN from {snn_path} ...")
    snn_model = akida.Model(snn_path)
    print(f"  Loaded: {len(snn_model.layers)} layers.")

    t0     = time.perf_counter()
    logits = snn_model.predict(x_uint8)   # (N, H, W, 2)
    elapsed = time.perf_counter() - t0

    preds = np.argmax(logits, axis=-1).astype(np.int32)   # (N, H, W)
    ms_per_image = elapsed / len(x_uint8) * 1000.0
    return preds, ms_per_image


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(cnn_m, snn_m, cnn_ms, snn_ms, n_samples):
    """
    Print a side-by-side formatted comparison of CNN vs Akida SNN metrics.

    Parameters
    ----------
    cnn_m   : dict   — output of compute_metrics() for CNN
    snn_m   : dict or None — output of compute_metrics() for SNN
    cnn_ms  : float  — CNN ms-per-image
    snn_ms  : float or None — SNN ms-per-image
    n_samples : int
    """
    W = 72

    def _val(m, key):
        """Format a metric value, or '  N/A  ' if metrics dict is None."""
        if m is None:
            return "     N/A"
        return f"{m[key]:>8.4f}"

    def _ms(v):
        return f"{v:>8.3f} ms" if v is not None else "     N/A   "

    def _delta(cnn_m, snn_m, key):
        if snn_m is None:
            return "N/A"
        d = cnn_m[key] - snn_m[key]
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    def _pct(cnn_m, snn_m, key):
        if snn_m is None or cnn_m[key] < EPS:
            return ""
        pct = (cnn_m[key] - snn_m[key]) / cnn_m[key] * 100
        direction = "degradation" if pct >= 0 else "improvement"
        return f"  ({abs(pct):.1f}% {direction} from quantisation)"

    print("\n" + "─" * W)
    print(f"{'Metric':<28} {'CNN (float32)':>16}  {'Akida SNN (int4)':>16}")
    print("─" * W)

    rows = [
        ("Accuracy",           "accuracy"),
        ("Precision",          "precision"),
        ("Recall",             "recall"),
        ("F1 Score",           "f1"),
        ("IoU (RFI class)",    "iou"),
        ("Specificity",        "specificity"),
    ]
    for label, key in rows:
        print(f"{label:<28} {_val(cnn_m, key):>16}  {_val(snn_m, key):>16}")

    print("─" * W)
    print(f"{'Inference time':<28} {_ms(cnn_ms):>16}  {_ms(snn_ms):>16}")

    # Raw pixel counts
    print(f"\n{'─ Confusion matrix totals (pixels) ':-<{W}}")
    for tag in ("tp", "fp", "tn", "fn"):
        c = f"{cnn_m[tag]:>16,}" if cnn_m else "             N/A"
        s = f"{snn_m[tag]:>16,}" if snn_m else "             N/A"
        print(f"  {tag.upper():<26} {c}  {s}")

    # Delta rows
    print(f"\n{'─ Δ CNN − SNN ':-<{W}}")
    for label, key in [("F1 Score", "f1"), ("IoU", "iou"),
                        ("Precision", "precision"), ("Recall", "recall")]:
        d = _delta(cnn_m, snn_m, key)
        note = _pct(cnn_m, snn_m, key)
        print(f"  Δ {label:<24} {d}{note}")

    print(f"\n  Evaluated on {n_samples} test samples.")
    print(
        "  Metrics are accumulated globally over all pixels (not per-sample averaged).\n"
        "  Inference times are wall-clock (software simulation — not hardware)."
    )
    print("─" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Load CNN model
    print(f"Loading CNN model from {CNN_PATH} ...")
    cnn_model = tf.keras.models.load_model(CNN_PATH, compile=False)
    print(f"Loaded: {cnn_model.name}  ({cnn_model.count_params():,} parameters)")

    # 2. Load test data
    n_label = str(N_EVAL) if N_EVAL is not None else "all"
    print(f"\nLoading test samples ({n_label}) from {TEST_DIR} ...")
    x_float, y_masks = load_samples(TEST_DIR, N_EVAL)
    print(f"Loaded {len(x_float)} samples, shape {x_float.shape}")

    # 3. Derive uint8 input for Akida (scale [0,1] → [0,255])
    x_uint8 = np.round(x_float * 255).astype(np.uint8)

    # 4. CNN inference + metrics
    print("\n─── CNN Evaluation " + "─" * 52)
    cnn_preds, cnn_ms = run_cnn(cnn_model, x_float)
    cnn_metrics = compute_metrics(cnn_preds, y_masks)
    print(
        f"  Done. F1={cnn_metrics['f1']:.4f}  IoU={cnn_metrics['iou']:.4f}"
        f"  ({cnn_ms:.3f} ms/image)"
    )

    # 5. Akida SNN inference + metrics
    print("\n─── Akida SNN Evaluation " + "─" * 47)
    snn_preds, snn_ms = run_akida(SNN_PATH, x_uint8)
    if snn_preds is not None:
        snn_metrics = compute_metrics(snn_preds, y_masks)
        print(
            f"  Done. F1={snn_metrics['f1']:.4f}  IoU={snn_metrics['iou']:.4f}"
            f"  ({snn_ms:.3f} ms/image)"
        )
    else:
        snn_metrics = None

    # 6. Side-by-side report
    print_report(cnn_metrics, snn_metrics, cnn_ms, snn_ms, len(x_float))


if __name__ == "__main__":
    main()
