# ============================
# convert_tenn_st.py
# PURPOSE: Quantize the TENN ONNX model (from train_tenn_st.py), convert to an
#          Akida SNN, evaluate with FIFO streaming inference, and save the model.
#
# Pipeline:
#   1. Load ONNX model — tenns_modules.export_to_onnx already bufferizes temporal
#      convolutions during export, so input shape is (B, 1, H, W) per frame (4D).
#   2. Load calibration frames: (N, 1, H, W) float32 [0,1] — individual frames.
#   3. quantizeml.quantize() — quantizes weights/activations (bufferization already done).
#   4. cnn2snn.convert() — produce Akida SNN.
#   5. Evaluate on test sequences:
#        for each sequence: reset_states → feed T frames → argmax last output.
#   6. Save Akida model to .fbz.
# ============================

import os
import numpy as np
from PIL import Image

import cnn2snn
import onnx
from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams


# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_SAVE_ROOT  = "Models/V7"
ONNX_TRAIN_PATH  = os.path.join(MODEL_SAVE_ROOT, "Akida_tenn_CNN.onnx")
AKIDA_SAVE_PATH  = os.path.join(MODEL_SAVE_ROOT, "Akida_tenn_SNN.fbz")
DATASET_ROOT     = "Data/Akida_Full_Dataset"
TRAIN_DIR        = os.path.join(DATASET_ROOT,"Full_Data_Train")
TEST_DIR         = os.path.join(DATASET_ROOT,"Full_Data_Test")

INPUT_H = 64
INPUT_W = 64
OUT_H   = INPUT_H
OUT_W   = INPUT_W

SEQUENCE_LEN = 4    # must match train_tenn_st.py

N_CALIB = 256       # calibration frames (individual — ONNX is already bufferized, input is 4D)

SAVE_SAMPLES = True
N_SAMPLES    = 8
SAMPLES_DIR  = os.path.join(MODEL_SAVE_ROOT, "Samples")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_calibration_frames(train_dir, n_frames):
    """
    Load n_frames individual training spectrograms as (N, 1, H, W) float32 [0,1].

    tenns_modules.export_to_onnx already bufferizes temporal convolutions during
    export, so the ONNX model expects 4D per-frame input (B, 1, H, W), not 5D.
    Calibration therefore uses individual frames matching this shape.
    """
    spec_dir = os.path.join(train_dir, "spectrograms")
    fnames = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))[:n_frames]
    frames = []
    for fname in fnames:
        arr = np.array(
            Image.open(os.path.join(spec_dir, fname)).convert("L"),
            dtype=np.float32
        ) / 255.0   # (H, W)
        frames.append(arr[np.newaxis, np.newaxis])   # (1, 1, H, W)
    return np.concatenate(frames, axis=0).astype(np.float32)   # (N, 1, H, W)


def collect_test_sequences(test_dir, seq_len):
    """Returns list of sequences, each a list of (spec_path, mask_path) tuples."""
    spec_dir = os.path.join(test_dir, "spectrograms")
    mask_dir = os.path.join(test_dir, "masks")
    fnames = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))

    n_seqs = len(fnames) // seq_len
    fnames = fnames[:n_seqs * seq_len]

    sequences = []
    for i in range(0, len(fnames), seq_len):
        window = fnames[i:i + seq_len]
        seq = [
            (os.path.join(spec_dir, f), os.path.join(mask_dir, f))
            for f in window
            if os.path.exists(os.path.join(mask_dir, f))
        ]
        if len(seq) == seq_len:
            sequences.append(seq)
    return sequences


def load_frame_uint8(spec_path):
    """Load spectrogram as (1, H, W, 1) uint8 for Akida inference (NHWC format)."""
    arr = np.array(Image.open(spec_path).convert("L"), dtype=np.int8)
    return arr[np.newaxis, np.newaxis, ...]   # (1, H, W, 1)


def load_mask(mask_path):
    """Load mask as (H, W) int32, values 0 or 1."""
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
    return mask_raw


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_sample_images(frames_u8, masks, preds, save_dir, prefix="sample"):
    """
    frames_u8: (N, 1, 1, H, W) int8   — last frame fed to Akida (NCHW format, batch+channel dims)
    masks:     (N, H, W)        int32
    preds:     (N, H, W)        int
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(frames_u8)):
        spec_img = frames_u8[i, 0, 0].astype(np.float32) / 127.0  # (H, W), int8 → [0,1]
        mask     = masks[i]
        pred     = preds[i]

        diff = np.zeros((*mask.shape, 3), dtype=np.float32)
        diff[(pred == 1) & (mask == 1)] = [0.0, 1.0, 0.0]
        diff[(pred == 1) & (mask == 0)] = [1.0, 0.0, 0.0]
        diff[(pred == 0) & (mask == 1)] = [1.0, 1.0, 0.0]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(spec_img, cmap="viridis"); axes[0].set_title("Input (last frame)"); axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Prediction");   axes[2].axis("off")
        axes[3].imshow(diff); axes[3].set_title("Diff  (TP=G  FP=R  FN=Y)"); axes[3].axis("off")

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(akida_model, test_sequences):
    """
    Streaming evaluation: for each sequence reset Akida state, feed T frames
    one at a time, take the prediction from the last frame's output.

    Akida output shape: (1, H, W, 2) — NHWC, argmax on axis=-1 (class dim).
    """
    total_pixels = 0
    correct = 0
    tp = fp = tn = fn = 0
    eval_index = 0

    sample_frames, sample_masks, sample_preds = [], [], []

    for seq in test_sequences:
        # No state reset needed: T-1 warm-up frames flush the FIFO such that
        # the prediction frame sees exactly [curr_0 .. curr_T-1] as context.

        # Warm-up: feed frames 0 .. T-2
        for spec_path, _ in seq[:-1]:
            akida_model.predict(load_frame_uint8(spec_path))

        # Predict on last frame
        last_spec_path, last_mask_path = seq[-1]
        frame_u8 = load_frame_uint8(last_spec_path)
        y_true   = load_mask(last_mask_path)

        logits = akida_model.predict(frame_u8)      # (1, H, W, 2)
        pred   = np.argmax(logits, axis=-1)[0]      # (H, W)

        total_pixels += y_true.size
        correct      += (pred == y_true).sum()
        tp += ((pred == 1) & (y_true == 1)).sum()
        fp += ((pred == 1) & (y_true == 0)).sum()
        tn += ((pred == 0) & (y_true == 0)).sum()
        fn += ((pred == 0) & (y_true == 1)).sum()

        if SAVE_SAMPLES and len(sample_frames) < N_SAMPLES and eval_index % 5 == 0:
            sample_frames.append(frame_u8)
            sample_masks.append(y_true)
            sample_preds.append(pred)
        eval_index += 1

    accuracy    = correct / total_pixels
    precision   = tp / (tp + fp + 1e-8)
    recall      = tp / (tp + fn + 1e-8)
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    iou         = tp / (tp + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    metrics = {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "iou": iou, "specificity": specificity,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    if SAVE_SAMPLES and sample_frames:
        print(f"\nSaving {len(sample_frames)} sample images to {SAMPLES_DIR} ...")
        save_sample_images(
            np.array(sample_frames), np.array(sample_masks), np.array(sample_preds),
            SAMPLES_DIR, prefix="tenn_akida"
        )

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1) Load training ONNX model (bufferized: input shape (B, 1, H, W) per frame)
    print(f"Loading ONNX model from {ONNX_TRAIN_PATH} ...")
    onnx_model = onnx.load(ONNX_TRAIN_PATH)
    print("ONNX model loaded.")

    # 2) Load calibration frames: (N, 1, H, W) float32 [0,1]
    print(f"\nLoading {N_CALIB} calibration frames from {TRAIN_DIR} ...")
    calib_samples = load_calibration_frames(TRAIN_DIR, N_CALIB)
    print(f"Calibration data: shape={calib_samples.shape}, dtype={calib_samples.dtype}")

    # 3) Quantize (FIFO ops already in place from export_to_onnx)
    print("\nQuantizing ONNX model ...")
    qparams = QuantizationParams(per_tensor_activations=True, input_dtype="int8")
    q_model = quantize(
        onnx_model,
        qparams=qparams,
        epochs=1,
        batch_size=8,
        samples=calib_samples,
    )
    print("Quantization complete.")

    # 4) Convert to Akida model
    print("\nConverting to Akida model ...")
    akida_model = cnn2snn.convert(q_model)
    print("Akida model created.")
    print(akida_model.summary())

    # 5) Evaluate on test sequences
    print(f"\nLoading test sequences from {TEST_DIR} ...")
    test_sequences = collect_test_sequences(TEST_DIR, SEQUENCE_LEN)
    print(f"Found {len(test_sequences)} test sequences ({SEQUENCE_LEN} frames each).")

    print("Running streaming inference (warm-up T-1 frames → predict last frame) ...")
    metrics = evaluate(akida_model, test_sequences)

    print("\n── Results ───────────────────────────────────")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  IoU (RFI):   {metrics['iou']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

    # 6) Save Akida model
    akida_model.save(AKIDA_SAVE_PATH)
    print(f"\nSaved Akida model to {AKIDA_SAVE_PATH}")


if __name__ == "__main__":
    main()
