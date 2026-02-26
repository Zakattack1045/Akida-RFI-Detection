# ============================
# convert_rfi_to_akida.py
# PURPOSE: CNN -> (quantize + QAT) -> Akida model -> evaluate on real RFI test set.
# ============================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import cnn2snn
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH   = "Models/Akida_v1.keras"
TRAIN_DIR    = "Data/Akida_Dataset/Combined_Dataset"  # training data for QAT
TEST_DIR     = "Data/Akida_Dataset/Combined_Test"
AKIDA_SAVE   = "Models/Akida_v1.fbz"

INPUT_H = 64
INPUT_W = 64
OUT_H   = INPUT_H
OUT_W   = INPUT_W

# Quantization settings
WEIGHT_BITS = 4    # weight quantization bits
ACTIV_BITS  = 4    # activation quantization bits (4 was too aggressive — 8 recommended)

# Quantization-Aware Training (QAT)
# Fine-tunes the quantized model on training data so weights can adapt to the
# quantization grid. This is the primary fix for the high FP / low precision issue.
# Set QAT_EPOCHS = 0 to skip QAT (post-training quantization only, not recommended).
QAT_EPOCHS  = 10
QAT_LR      = 1e-4
QAT_BATCH   = 8

# Sample image saving
# Set SAVE_SAMPLES = True to save visualisation PNGs (input / mask / pred / diff)
# for the first N_SAMPLES test images after evaluation.
SAVE_SAMPLES = True
N_SAMPLES    = 8
SAMPLES_DIR  = "Samples/Akida"


# ── Loss (mirrors train_akida.py) ──────────────────────────────────────────────

_class_weights = tf.constant([0.2, 0.8], dtype=tf.float32)


def weighted_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    w = tf.gather(_class_weights, tf.cast(y_true, tf.int32))
    return tf.reduce_mean(ce * w)


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
    pairs = [
        (os.path.join(spec_dir, f), os.path.join(mask_dir, f))
        for f in fnames
        if os.path.exists(os.path.join(mask_dir, f))
    ]
    return pairs


def load_train_data_for_qat(train_dir):
    """
    Load all training spectrograms and masks as numpy arrays for QAT fine-tuning.
    Spectrograms are float32 normalised to [0, 1], matching train_akida.py.
    """
    spec_dir = os.path.join(train_dir, "spectrograms")
    mask_dir = os.path.join(train_dir, "masks")
    fnames = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))

    specs, masks = [], []
    for fname in fnames:
        spec = np.array(
            Image.open(os.path.join(spec_dir, fname)).convert("L"),
            dtype=np.float32
        )[..., None] / 255.0  # (64, 64, 1), values 0-1

        mask_raw = np.array(
            Image.open(os.path.join(mask_dir, fname)).convert("L"),
            dtype=np.uint8
        )
        if mask_raw.max() > 1:
            mask_raw = (mask_raw > 127).astype(np.int32)
        else:
            mask_raw = mask_raw.astype(np.int32)

        specs.append(spec)
        masks.append(mask_raw)

    return np.array(specs, dtype=np.float32), np.array(masks, dtype=np.int32)


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_sample_images(specs_u8, masks, preds, save_dir, prefix="sample"):
    """
    Save one PNG per sample showing:
      [Input spectrogram | Ground-truth mask | Prediction | Difference]
    Difference colour key: TP=green, FP=red, FN=yellow, TN=black.

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
        axes[0].imshow(spec_img, cmap="viridis")
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        axes[3].imshow(diff)
        axes[3].set_title("Diff  (TP=G  FP=R  FN=Y)")
        axes[3].axis("off")

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

    # Collect samples for optional visualisation
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
            sample_specs.append(x_u8)          # (1, H, W, 1) uint8
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
            prefix="akida",
        )

    return metrics


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

    # 3) Quantize
    print(f"\nQuantizing (weights={WEIGHT_BITS}-bit, activations={ACTIV_BITS}-bit) ...")
    q_model = cnn2snn.quantize(
        ann_model,
        weight_quantization=WEIGHT_BITS,
        activ_quantization=ACTIV_BITS,
        fold_BN=True,
    )
    print("Quantized model created.")

    # 4) QAT fine-tuning
    # Fine-tuning the quantized model lets weights adapt to the quantization grid,
    # dramatically reducing the accuracy drop versus post-training quantization alone.
    if QAT_EPOCHS > 0:
        print(f"\nLoading training data for QAT from {TRAIN_DIR} ...")
        x_train, y_train = load_train_data_for_qat(TRAIN_DIR)
        print(f"Loaded {len(x_train)} training samples.")

        q_model.compile(
            optimizer=tf.keras.optimizers.Adam(QAT_LR),
            loss=weighted_loss,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pix_acc")],
        )
        print(f"QAT fine-tuning for {QAT_EPOCHS} epochs at lr={QAT_LR} ...")
        q_model.fit(
            x_train, y_train,
            epochs=QAT_EPOCHS,
            batch_size=QAT_BATCH,
            shuffle=True,
        )
        print("QAT complete.")

    # 5) Convert to Akida model
    print("\nConverting to Akida model ...")
    akida_model = cnn2snn.convert(q_model, input_scaling=(255, 0))
    print("Akida model created.")
    print(akida_model.summary())

    # 6) Evaluate on real test set
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

    # 7) Save the Akida model
    akida_model.save(AKIDA_SAVE)
    print(f"\nSaved Akida model to {AKIDA_SAVE}")


if __name__ == "__main__":
    main()
