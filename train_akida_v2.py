# ============================
# train_akida_v2.py
# PURPOSE: Train an enhanced CNN for RFI segmentation, compatible with the
#          QuantizeML/ONNX -> Akida v2 conversion path.
#
# Architecture (mirrors Lava/test_network.py, adapted for Akida v2 via ONNX):
#   Stem:  Conv2D(stride=2) -> Conv2DTranspose(stride=2)  (replaces dilated conv)
#          DepthwiseConv2D -> pointwise 1x1 compress
#   IEMs:  Residual blocks with Add skip connections
#          Depthwise (3x3) replaces 1x5 / 5x1 asymmetric convs (not safe for Akida)
#   Head:  1x1 Conv2D, 2-class logits
#
# After training, exports the model to ONNX for use in convert_onnx_to_akida.py.
# ============================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from PIL import Image


# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_ROOT = "Data/Akida_Dataset"
TRAIN_SPLIT  = "Combined_Dataset"
TEST_SPLIT   = "Combined_Test"

INPUT_H = 64
INPUT_W = 64
OUT_H   = INPUT_H
OUT_W   = INPUT_W

BATCH_SIZE   = 8
EPOCHS       = 5
STEPS_EPOCH  = 244   # ceil(1950 / 8)
VAL_STEPS    = 13    # ceil(100 / 8)

MODEL_SAVE_ROOT = "Models/V3"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_ROOT,"Akida_ONNIX.keras")
ONNX_SAVE_PATH  = os.path.join(MODEL_SAVE_ROOT,"Akida_ONNIX.onnx")

# Sample image saving
SAVE_SAMPLES = True
N_SAMPLES    = 8
SAMPLES_DIR  = os.path.join(MODEL_SAVE_ROOT,"Samples")

LOG_DIR = os.path.join(MODEL_SAVE_ROOT,"logs")


# ── Dataset (identical to train_akida.py) ──────────────────────────────────────

def _load_split_generator(root, split_name, out_h, out_w):
    spec_dir = os.path.join(root, split_name, "spectrograms")
    mask_dir = os.path.join(root, split_name, "masks")
    filenames = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))

    for fname in filenames:
        spec = np.array(
            Image.open(os.path.join(spec_dir, fname)).convert("L"),
            dtype=np.float32
        )[..., None] / 255.0  # (64, 64, 1), values 0-1

        mask_raw = np.array(
            Image.open(os.path.join(mask_dir, fname)).convert("L"),
            dtype=np.uint8
        )
        if mask_raw.max() > 1:
            mask_raw = (mask_raw > 127).astype(np.uint8)
        if out_h != mask_raw.shape[0] or out_w != mask_raw.shape[1]:
            mask_raw = np.array(
                Image.fromarray(mask_raw).resize((out_w, out_h), resample=Image.NEAREST),
                dtype=np.int32
            )
        else:
            mask_raw = mask_raw.astype(np.int32)
        yield spec, mask_raw


def make_dataset(root, split_name, out_h, out_w, batch_size, shuffle=True):
    def gen():
        yield from _load_split_generator(root, split_name, out_h, out_w)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(INPUT_H, INPUT_W, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(out_h, out_w),        dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(256)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()


# ── Model Building Blocks ──────────────────────────────────────────────────────

def conv_bn_relu(x, filters, k, name, **kwargs):
    """Conv2D -> BatchNorm -> ReLU."""
    x = tf.keras.layers.Conv2D(
        filters, k, padding="same", use_bias=False, name=name + "_conv", **kwargs
    )(x)
    x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    return tf.keras.layers.ReLU(name=name + "_relu")(x)


def depthwise_bn_relu(x, name):
    """DepthwiseConv2D(3x3) -> BatchNorm -> ReLU. Channel count unchanged."""
    x = tf.keras.layers.DepthwiseConv2D(
        3, padding="same", use_bias=False, depth_multiplier=1, name=name + "_dw"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x)
    return tf.keras.layers.ReLU(name=name + "_relu")(x)


def residual_iem_block(x, hidden_ch, out_ch, name):
    """
    Residual IEM block (mirrors Lava test_network IEM, adapted for Akida v2):

      expand  (1x1 Conv)  -> BN -> ReLU
      local   (3x3 DW)    -> BN -> ReLU    [was 3x3 DW in Lava]
      spatial1 (3x3 DW)   -> BN -> ReLU    [was 1x5 horizontal in Lava — replaced]
      spatial2 (3x3 DW)   -> BN -> ReLU    [was 5x1 vertical in Lava — replaced]
      compress (1x1 Conv) -> BN             [NO activation before Add]
      Add([compressed, residual])
      ReLU                                  [<Add, ReLU> — supported QuantizeML pattern]

    Projection shortcut (1x1 Conv + BN) added when in/out channels differ.
    """
    residual = x

    # Main path
    x = conv_bn_relu(x, hidden_ch, 1, name=name + "_expand")
    x = depthwise_bn_relu(x, name=name + "_local")
    x = depthwise_bn_relu(x, name=name + "_spatial1")
    x = depthwise_bn_relu(x, name=name + "_spatial2")

    # Compress — no activation; ReLU comes after Add
    x = tf.keras.layers.Conv2D(
        out_ch, 1, padding="same", use_bias=False, name=name + "_compress_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=name + "_compress_bn")(x)

    # Projection shortcut if channel count differs
    if residual.shape[-1] != out_ch:
        residual = tf.keras.layers.Conv2D(
            out_ch, 1, padding="same", use_bias=False, name=name + "_proj_conv"
        )(residual)
        residual = tf.keras.layers.BatchNormalization(name=name + "_proj_bn")(residual)

    # <Add, ReLU> pattern
    x = tf.keras.layers.Add(name=name + "_add")([x, residual])
    return tf.keras.layers.ReLU(name=name + "_relu")(x)


# ── Full Model ─────────────────────────────────────────────────────────────────

def build_model(in_h=INPUT_H, in_w=INPUT_W):
    inp = tf.keras.Input(shape=(in_h, in_w, 1), dtype=tf.float32, name="spectrogram")

    # ── Stem ───────────────────────────────────────────────────────────────────
    # stem1: strided Conv2D downsamples 64->32 (provides wide context, replacing dilation)
    x = conv_bn_relu(inp, 32, 3, name="stem1", strides=2)   # 64x64 -> 32x32

    # stem2: Conv2DTranspose upsamples back 32->64 (the single transposed conv in the stem)
    x = tf.keras.layers.Conv2DTranspose(
        32, 3, strides=2, padding="same", use_bias=False, name="stem2_conv"
    )(x)                                                      # 32x32 -> 64x64
    x = tf.keras.layers.BatchNormalization(name="stem2_bn")(x)
    x = tf.keras.layers.ReLU(name="stem2_relu")(x)

    # stem3: depthwise (mirrors the depthwise in Lava stem)
    x = depthwise_bn_relu(x, name="stem3")                   # 64x64, channels unchanged

    # stem4: pointwise compress (mirrors the 1x1 in Lava stem)
    x = conv_bn_relu(x, 16, 1, name="stem4")                 # 64x64x16

    # ── Residual IEM Blocks ────────────────────────────────────────────────────
    x = residual_iem_block(x, hidden_ch=64,  out_ch=16, name="iem1")  # 64x64x16
    x = residual_iem_block(x, hidden_ch=128, out_ch=32, name="iem2")  # 64x64x32
    x = residual_iem_block(x, hidden_ch=64,  out_ch=16, name="iem3")  # 64x64x16
    x = residual_iem_block(x, hidden_ch=32,  out_ch=8,  name="iem4")  # 64x64x8

    # ── Head: 1x1 conv -> 2-class logits ──────────────────────────────────────
    logits = tf.keras.layers.Conv2D(2, 1, padding="same", name="logits")(x)

    return tf.keras.Model(inp, logits, name="rfi_seg_akida_v2")


# ── Loss ───────────────────────────────────────────────────────────────────────

_class_weights = tf.constant([0.2, 0.8], dtype=tf.float32)


def weighted_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    w = tf.gather(_class_weights, y_true)
    return tf.reduce_mean(ce * w)


# ── Metrics Callback ───────────────────────────────────────────────────────────

class SegMetricsCallback(tf.keras.callbacks.Callback):
    """Computes per-epoch TP/FP/TN/FN/Precision/Recall/F1 on validation set."""

    def __init__(self, val_dataset, val_steps, log_dir):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps   = val_steps
        os.makedirs(log_dir, exist_ok=True)
        self._writer = tf.summary.create_file_writer(
            os.path.join(log_dir, "val_custom")
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tp = fp = tn = fn = 0

        for x_batch, y_batch in self.val_dataset.take(self.val_steps):
            logits = self.model(x_batch, training=False)
            preds  = tf.argmax(logits, axis=-1)
            labels = tf.cast(y_batch, tf.int64)
            p = tf.reshape(preds,  [-1])
            l = tf.reshape(labels, [-1])
            tp += int(tf.reduce_sum(tf.cast((p == 1) & (l == 1), tf.int32)))
            fp += int(tf.reduce_sum(tf.cast((p == 1) & (l == 0), tf.int32)))
            tn += int(tf.reduce_sum(tf.cast((p == 0) & (l == 0), tf.int32)))
            fn += int(tf.reduce_sum(tf.cast((p == 0) & (l == 1), tf.int32)))

        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f1        = 2 * precision * recall / (precision + recall + 1e-7)

        with self._writer.as_default():
            tf.summary.scalar("precision", precision, step=epoch)
            tf.summary.scalar("recall",    recall,    step=epoch)
            tf.summary.scalar("f1",        f1,        step=epoch)
            tf.summary.scalar("tp",        tp,        step=epoch)
            tf.summary.scalar("fp",        fp,        step=epoch)
            tf.summary.scalar("tn",        tn,        step=epoch)
            tf.summary.scalar("fn",        fn,        step=epoch)
        self._writer.flush()

        print(f"  [Metrics] loss={logs.get('val_loss', float('nan')):.4f}  "
              f"acc={logs.get('val_pix_acc', float('nan')):.4f}  "
              f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  "
              f"TP={tp}  FP={fp}  FN={fn}")


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_sample_images(specs, masks, preds, save_dir, prefix="sample"):
    """
    Save one PNG per sample: [Input | Ground Truth | Prediction | Diff]
    Diff colour key: TP=green, FP=red, FN=yellow, TN=black.

    specs: (N, H, W, 1) float32 [0, 1]
    masks: (N, H, W)    int32
    preds: (N, H, W)    int
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(specs)):
        spec_img = specs[i, :, :, 0]
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Input: {INPUT_H}x{INPUT_W}  Output: {OUT_H}x{OUT_W}")
    print(f"Dataset: {DATASET_ROOT}")

    train_ds = make_dataset(DATASET_ROOT, TRAIN_SPLIT, OUT_H, OUT_W, BATCH_SIZE, shuffle=True)
    val_ds   = make_dataset(DATASET_ROOT, TEST_SPLIT,  OUT_H, OUT_W, BATCH_SIZE, shuffle=False)

    model = build_model(INPUT_H, INPUT_W)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=weighted_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pix_acc")],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR, histogram_freq=0, write_graph=False, update_freq="epoch"
        ),
        SegMetricsCallback(val_dataset=val_ds, val_steps=VAL_STEPS, log_dir=LOG_DIR),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_EPOCH,
        validation_steps=VAL_STEPS,
        callbacks=callbacks,
    )

    # Save Keras model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or ".", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nSaved Keras model to {MODEL_SAVE_PATH}")

    # Export to ONNX for QuantizeML conversion path
    print(f"\nExporting model to ONNX at {ONNX_SAVE_PATH} ...")
    import tf2onnx
    import onnx as onnx_lib
    spec = (tf.TensorSpec((None, INPUT_H, INPUT_W, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=ONNX_SAVE_PATH,
    )
    # Cap ir_version to 10 — quantizeml 1.2.3 rejects ir_version > 10.
    _om = onnx_lib.load(ONNX_SAVE_PATH)
    if _om.ir_version > 10:
        _om.ir_version = 10
        onnx_lib.save(_om, ONNX_SAVE_PATH)
    print(f"ONNX export complete: {ONNX_SAVE_PATH}")

    # Save sample visualisations
    if SAVE_SAMPLES:
        print(f"\nGenerating {N_SAMPLES} sample images ...")
        gen = _load_split_generator(DATASET_ROOT, TEST_SPLIT, OUT_H, OUT_W)
        sample_specs, sample_masks = [], []
        for spec_arr, mask in gen:
            sample_specs.append(spec_arr)
            sample_masks.append(mask)
            if len(sample_specs) >= N_SAMPLES:
                break

        x = np.array(sample_specs, dtype=np.float32)
        y = np.array(sample_masks, dtype=np.int32)
        logits = model.predict(x)
        preds  = np.argmax(logits, axis=-1)
        save_sample_images(x, y, preds, SAMPLES_DIR, prefix="cnn_v2")


if __name__ == "__main__":
    main()
