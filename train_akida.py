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
EPOCHS       = 3
STEPS_EPOCH  = 244   # ceil(1950 train samples / batch 8)
VAL_STEPS    = 13    # ceil(100 test samples / batch 8)

MODEL_SAVE_PATH = "Akida/rfi_seg_akida.keras"


# ── Dataset ────────────────────────────────────────────────────────────────────

def _load_split_generator(root, split_name, out_h, out_w):
    """
    Yields (spectrogram, mask) pairs.
      spectrogram: (INPUT_H, INPUT_W, 1) float32, values 0-255
      mask:        (out_h, out_w) int32, values 0 or 1
    Masks are resized from INPUT_H x INPUT_W to out_h x out_w (nearest-neighbour)
    to match the network output resolution.
    """
    spec_dir = os.path.join(root, split_name, "spectrograms")
    mask_dir = os.path.join(root, split_name, "masks")

    filenames = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))

    for fname in filenames:
        # Spectrogram: grayscale uint8 -> float32 normalised to [0, 1]
        # (Rescaling is NOT in the model; normalise here so training and
        #  Akida inference via input_scaling=(255,0) see the same range)
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
    # stem1: regular 3x3 conv.
    # dilation_rate>1, Conv2DTranspose, and Add are all rejected by cnn2snn's
    # Keras pathway. A plain Conv2D is the only compatible option here.
    x = conv_bn_relu(inp, 32, 3, name="stem1")

    # stem2: regular 3x3 conv
    x = conv_bn_relu(x, 32, 3, name="stem2")

    # stem3: regular 3x3 conv (depthwise also unsupported in conversion)
    x = conv_bn_relu(x, 32, 3, name="stem3")

    # stem4: pointwise 1x1 (channel reduction 32 -> 8)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Input:  {INPUT_H}x{INPUT_W}  ->  Output: {OUT_H}x{OUT_W} (after stem1 upsample)")
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

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_EPOCH,
        validation_steps=VAL_STEPS,
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\nSaved trained model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
