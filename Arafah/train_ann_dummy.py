# ============================
# train_ann_dummy.py
# PURPOSE: Train a CNN that is compatible with cnn2snn->Akida conversion.
# KEY IDEAS (Akida constraints you hit):
# - NO DepthwiseConv2D
# - NO residual/concat merges (no Add/Concatenate)
# - Inputs should match Akida inference: uint8-like range 0..255
#   so we put Rescaling(1/255) INSIDE the model.
# ============================
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf


# ---------- dummy data ----------
def make_blob_mask(h, w, num_blobs=3, r_min=10, r_max=35):
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(num_blobs):
        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)
        r = np.random.randint(r_min, r_max)
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        mask[blob] = 1
    return mask

def make_spectrogram_like_float(mask, noise_level=0.15):
    h, w = mask.shape
    img = np.random.normal(loc=0.4, scale=noise_level, size=(h, w)).astype(np.float32)
    img += mask.astype(np.float32) * 0.45
    for _ in range(np.random.randint(2, 6)):
        col = np.random.randint(0, w)
        width = np.random.randint(1, 5)
        img[:, max(0, col - width): min(w, col + width)] += np.random.uniform(0.05, 0.25)
    return np.clip(img, 0.0, 1.0)

def dummy_dataset(num_samples=200, h=256, w=256, batch_size=2):
    """
    We generate an image in [0,1], then convert it to uint8-like 0..255,
    but store as float32 tensor (Keras trains on float).
    This matches Akida inference expectation later.
    """
    def gen():
        for _ in range(num_samples):
            y = make_blob_mask(h, w, num_blobs=np.random.randint(1, 5)).astype(np.int32)  # (H,W) 0/1
            img01 = make_spectrogram_like_float(y.astype(np.uint8))                       # (H,W) float [0,1]
            x_u8 = (255.0 * img01).astype(np.uint8)                                       # (H,W) uint8 0..255
            x = x_u8[..., None].astype(np.float32)                                        # (H,W,1) float but 0..255
            yield x, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(h, w, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(h, w), dtype=tf.int32),
        )
    )
    return ds.shuffle(64).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()


# ---------- model (Akida-safe) ----------
def conv_bn_relu(x, filters, k, name):
    x = tf.keras.layers.Conv2D(filters, k, padding="same", use_bias=False, name=name+"_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=name+"_bn")(x)  # will be folded by cnn2snn.quantize(fold_BN=True)
    x = tf.keras.layers.ReLU(name=name+"_relu")(x)
    return x

def iem_block_akida(x, in_ch=16, hidden_ch=32, name="iem"):
    # Sequential "IEM-like" block (no depthwise, no skip merge)
    x = conv_bn_relu(x, hidden_ch, 1,     name=name+"_expand")
    x = conv_bn_relu(x, hidden_ch, 3,     name=name+"_local")
    x = conv_bn_relu(x, hidden_ch, (1,5), name=name+"_h")
    x = conv_bn_relu(x, hidden_ch, (5,1), name=name+"_v")
    x = conv_bn_relu(x, in_ch,     1,     name=name+"_compress")
    return x

def build_model(h=256, w=256, base_ch=16, hidden_ch=32, num_iem=4):
    inp = tf.keras.Input(shape=(h, w, 1), name="spectrogram", dtype=tf.float32)

    # IMPORTANT: bake input scaling into the model.
    # This makes training and Akida inference consistent:
    # - we feed 0..255 values
    # - model internally converts to 0..1
    x = tf.keras.layers.Rescaling(1/255.0, name="in_scale")(inp)

    # Stem (no dilation, no depthwise)
    x = conv_bn_relu(x, 32, 3, name="stem1")
    x = conv_bn_relu(x, 32, 3, name="stem2")
    x = conv_bn_relu(x, 32, 3, name="stem3")
    x = conv_bn_relu(x, base_ch, 1, name="stem4_pw")

    # IEM chain
    for i in range(num_iem):
        x = iem_block_akida(x, in_ch=base_ch, hidden_ch=hidden_ch, name=f"iem{i+1}")

    logits = tf.keras.layers.Conv2D(2, 1, padding="same", name="logits")(x)
    return tf.keras.Model(inp, logits, name="rfi_seg_ann")


# ---------- weighted loss to force learning class 1 ----------
class_weights = tf.constant([0.2, 0.8], dtype=tf.float32)  # [background, RFI]

def weighted_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)  # (B,H,W)
    w = tf.gather(class_weights, y_true)                                                     # (B,H,W)
    return tf.reduce_mean(ce * w)


def main():
    H = W = 256
    train_ds = dummy_dataset(num_samples=300, h=H, w=W, batch_size=2)
    val_ds   = dummy_dataset(num_samples=60,  h=H, w=W, batch_size=2)

    model = build_model(H, W, base_ch=16, hidden_ch=32, num_iem=4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=weighted_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pix_acc")]
    )

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=3, steps_per_epoch=100, validation_steps=20)

    model.save("rfi_seg_ann.keras")
    print("Saved trained CNN to rfi_seg_ann.keras")


if __name__ == "__main__":
    main()