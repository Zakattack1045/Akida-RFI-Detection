# ============================
# convert_rfi_to_akida.py
# PURPOSE: CNN -> (quantize) -> Akida model -> evaluate on real RFI test set.
# Adapted from convert_to_akida.py for the trained rfi_seg_akida model.
# ============================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import cnn2snn
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH   = "Akida/rfi_seg_akida.keras"
TEST_DIR     = "Data/Akida_Dataset/Combined_Test"
AKIDA_SAVE   = "Akida/rfi_seg_akida.fbz"   # optional: save converted Akida model

INPUT_H = 64
INPUT_W = 64
OUT_H   = INPUT_H   # no upsampling — network output matches input resolution
OUT_W   = INPUT_W


# ── Data loading ───────────────────────────────────────────────────────────────

def load_test_sample(spec_path, mask_path):
    """
    Load one spectrogram/mask pair for Akida inference.
      spectrogram: (1, INPUT_H, INPUT_W, 1) uint8,  values 0-255
      mask:        (OUT_H, OUT_W)            int32,  values 0 or 1
    """
    # Spectrogram: uint8 grayscale, shape (INPUT_H, INPUT_W, 1) with batch dim
    spec = np.array(
        Image.open(spec_path).convert("L"),
        dtype=np.uint8
    )[None, ..., None]  # (1, 64, 64, 1)

    # Mask: 0/255 -> threshold 0/1 -> resize to output resolution
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
    pairs = [
        (os.path.join(spec_dir, f), os.path.join(mask_dir, f))
        for f in fnames
        if os.path.exists(os.path.join(mask_dir, f))
    ]
    return pairs


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(akida_model, test_pairs):
    total_pixels = 0
    correct = 0
    tp = fp = tn = fn = 0

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

    accuracy    = correct / total_pixels
    precision   = tp / (tp + fp + 1e-8)
    recall      = tp / (tp + fn + 1e-8)
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    iou         = tp / (tp + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "accuracy":    accuracy,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "iou":         iou,
        "specificity": specificity,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1) Load trained CNN
    print(f"Loading ANN model from {MODEL_PATH} ...")
    ann_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Loaded:", ann_model.name)

    # 2) Check Akida compatibility
    print("\nChecking model compatibility ...")
    cnn2snn.check_model_compatibility(ann_model)
    print("Model is compatible with cnn2snn conversion.")

    # 3) Quantize (weight_quantization=8, activ_quantization=4 as in dummy script)
    print("\nQuantizing ...")
    q_model = cnn2snn.quantize(
        ann_model,
        weight_quantization=8,
        activ_quantization=4,
        fold_BN=True,
    )
    print("Quantized model created.")

    # 4) Convert to Akida model
    # Rescaling is NOT in the model (removed to fix QuantizedConv2DTranspose
    # FixedPoint constraint). input_scaling=(255, 0) tells Akida to divide
    # raw uint8 inputs by 255 internally, equivalent to normalising to [0, 1].
    print("\nConverting to Akida model ...")
    akida_model = cnn2snn.convert(q_model, input_scaling=(255, 0))
    print("Akida model created.")
    print(akida_model.summary())

    # 5) Evaluate on real test set
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

    # 6) Optionally save the Akida model
    akida_model.save(AKIDA_SAVE)
    print(f"\nSaved Akida model to {AKIDA_SAVE}")


if __name__ == "__main__":
    main()
