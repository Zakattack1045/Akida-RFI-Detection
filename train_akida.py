# ============================
# train_akida.py
# PURPOSE: Train a CNN compatible with cnn2snn->Akida conversion for RFI segmentation.
#
# Architecture ported from Lava/Scripts/network.py with Akida constraints applied:
#   - Stem:  Conv2DTranspose(stride=2) replaces dilated conv (dilation unsupported)
#            DepthwiseConv2D replaces groups= conv (Conv2D(groups=N) unsupported)
#   - IEMs:  Residual versions with Add skip connections (projection shortcut)
#            DepthwiseConv2D replaces groups= in local conv
#   - Head:  1x1 Conv2D, 2-class logits
#
# Akida constraints observed:
#   - NO Conv2D(dilation_rate > 1)
#   - NO Conv2D(groups > 1)   — use DepthwiseConv2D instead
#   - Conv2DTranspose: stride=2, square kernel (3x3 or 4x4), padding='same' only
#   - DepthwiseConv2D: square kernel, depth_multiplier=1
#   - Add: no built-in activation (use a separate ReLU layer after)
#   - Rescaling(1/255) baked into model for consistent ANN training and Akida inference
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

INPUT_H = 64    # Spectrogram input height (from prepare_dataset.py)
INPUT_W = 64    # Spectrogram input width

# No upsampling — network operates at input resolution throughout.
OUT_H = INPUT_H
OUT_W = INPUT_W

BATCH_SIZE   = 8
EPOCHS       = 30
STEPS_EPOCH  = 244   # ceil(1950 train samples / batch 8)
VAL_STEPS    = 13    # ceil(100 test samples / batch 8)

MODEL_SAVE_ROOT = "Models/V2/"
MODEL_SAVE_NAME = "Akida_v2.keras"

# Sample image saving
# Set SAVE_SAMPLES = True to save visualisation PNGs after training.
# Each PNG shows: [Input | Ground Truth | CNN Prediction | Difference]
# Difference colour key: TP=green, FP=red, FN=yellow, TN=black.
SAVE_SAMPLES = True
N_SAMPLES    = 8
SAMPLES_DIR  = os.path.join(MODEL_SAVE_ROOT,"Samples/CNN")
os.makedirs(SAMPLES_DIR,exist_ok=True)

# TensorBoard
LOG_DIR = os.path.join(MODEL_SAVE_ROOT, "logs")   # Models/V2/logs/


# ── Dataset ────────────────────────────────────────────────────────────────────

def _load_split_generator(root, split_name, out_h, out_w):
    """
    Yields (spectrogram, mask) pairs.
      spectrogram: (INPUT_H, INPUT_W, 1) float32, values 0-1
      mask:        (out_h, out_w) int32, values 0 or 1
    Masks are resized from INPUT_H x INPUT_W to out_h x out_w (nearest-neighbour)
    to match the network output resolution.
    """
    spec_dir = os.path.join(root, split_name, "spectrograms")
    mask_dir = os.path.join(root, split_name, "masks")

    filenames = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))

    for fname in filenames:
        # Spectrogram: grayscale uint8 -> float32 normalised to [0, 1]
        spec = np.array(
            Image.open(os.path.join(spec_dir, fname)).convert("L"),
            dtype=np.float32
        )[..., None] / 255.0  # (64, 64, 1), values 0-1

        # Mask: 0/255 PNG -> threshold to 0/1 -> resize to output resolution
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
    x = tf.keras.layers.ReLU(name=name + "_relu")(x)
    return x


def iem_block(x, hidden_ch, out_ch, name):
    """
    Sequential IEM block. All layers are plain Conv2D — the cnn2snn sequential
    Keras pathway supports only Conv2D (square kernels, no dilation) + BN + ReLU.
    DepthwiseConv2D, dilation, Conv2DTranspose, and Add all fail at conversion.

    expand  (Conv2D 1x1) -> BN -> ReLU
    local   (Conv2D 3x3) -> BN -> ReLU
    spatial1(Conv2D 3x3) -> BN -> ReLU
    spatial2(Conv2D 3x3) -> BN -> ReLU
    compress(Conv2D 1x1) -> BN -> ReLU
    """
    x = conv_bn_relu(x, hidden_ch, 1, name=name + "_expand")
    x = conv_bn_relu(x, hidden_ch, 3, name=name + "_local")
    x = conv_bn_relu(x, hidden_ch, 3, name=name + "_spatial1")
    x = conv_bn_relu(x, hidden_ch, 3, name=name + "_spatial2")
    x = conv_bn_relu(x, out_ch,    1, name=name + "_compress")
    return x


# ── Full Model ─────────────────────────────────────────────────────────────────

def build_model(in_h=INPUT_H, in_w=INPUT_W):
    inp = tf.keras.Input(shape=(in_h, in_w, 1), dtype=tf.float32, name="spectrogram")

    # NOTE: No Rescaling layer here — inputs are normalised to [0, 1] in the
    # dataset loader; input_scaling=(255, 0) is passed to cnn2snn.convert()
    # so Akida applies the division at inference time.

    # ── Stem ───────────────────────────────────────────────────────────────────
    x = conv_bn_relu(inp, 32, 3, name="stem1")
    x = conv_bn_relu(x, 32, 3, name="stem2")
    x = conv_bn_relu(x, 32, 3, name="stem3")
    x = conv_bn_relu(x, 8, 1, name="stem4")

    # ── IEM Blocks (channel config matches Lava network.py) ───────────────────
    x = iem_block(x, hidden_ch=32,  out_ch=16, name="iem1")
    x = iem_block(x, hidden_ch=64,  out_ch=32, name="iem2")
    x = iem_block(x, hidden_ch=128, out_ch=16, name="iem3")
    x = iem_block(x, hidden_ch=64,  out_ch=8,  name="iem4")

    # ── Head: 1x1 conv -> 2-class logits ──────────────────────────────────────
    logits = tf.keras.layers.Conv2D(2, 1, padding="same", name="logits")(x)

    return tf.keras.Model(inp, logits, name="rfi_seg_akida")


# ── Loss ───────────────────────────────────────────────────────────────────────

# Higher weight on RFI class (1) to counteract class imbalance
_class_weights = tf.constant([0.2, 0.8], dtype=tf.float32)


def weighted_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )  # (B, H, W)
    w = tf.gather(_class_weights, y_true)  # (B, H, W)
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
            logits = self.model(x_batch, training=False)   # (B, H, W, 2)
            preds  = tf.argmax(logits, axis=-1)            # (B, H, W) int64
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
    Save one PNG per sample showing:
      [Input spectrogram | Ground-truth mask | Prediction | Difference]
    Difference colour key: TP=green, FP=red, FN=yellow, TN=black.

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Input:  {INPUT_H}x{INPUT_W}  ->  Output: {OUT_H}x{OUT_W}")
    print(f"Dataset: {DATASET_ROOT}")

    train_ds = make_dataset(
        DATASET_ROOT, TRAIN_SPLIT, OUT_H, OUT_W, BATCH_SIZE, shuffle=True
    )
    val_ds = make_dataset(
        DATASET_ROOT, TEST_SPLIT, OUT_H, OUT_W, BATCH_SIZE, shuffle=False
    )

    model = build_model(INPUT_H, INPUT_W)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=weighted_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pix_acc")],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
        ),
        SegMetricsCallback(
            val_dataset=val_ds,
            val_steps=VAL_STEPS,
            log_dir=LOG_DIR,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_EPOCH,
        validation_steps=VAL_STEPS,
        callbacks=callbacks,
    )

    os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_ROOT, MODEL_SAVE_NAME)
    model.save(save_path)
    print(f"\nSaved trained model to {save_path}")

    if SAVE_SAMPLES:
        print(f"\nGenerating {N_SAMPLES} sample images ...")
        gen = _load_split_generator(DATASET_ROOT, TEST_SPLIT, OUT_H, OUT_W)
        sample_specs, sample_masks = [], []
        for spec, mask in gen:
            sample_specs.append(spec)
            sample_masks.append(mask)
            if len(sample_specs) >= N_SAMPLES:
                break

        x = np.array(sample_specs, dtype=np.float32)   # (N, H, W, 1)
        y = np.array(sample_masks, dtype=np.int32)      # (N, H, W)
        logits = model.predict(x)                       # (N, H, W, 2)
        preds  = np.argmax(logits, axis=-1)             # (N, H, W)

        save_sample_images(x, y, preds, SAMPLES_DIR, prefix="cnn")


if __name__ == "__main__":
    main()
