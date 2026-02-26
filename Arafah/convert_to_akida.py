# ============================
# convert_to_akida.py
# PURPOSE: CNN -> (quantize) -> Akida model (SNN-ish runtime) -> simulate predict
# ============================
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import cnn2snn

H = W = 256


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

def make_spectrogram_like_uint8(h=256, w=256):
    y_true = make_blob_mask(h, w, num_blobs=np.random.randint(1, 5))
    img = np.random.normal(loc=0.4, scale=0.15, size=(h, w)).astype(np.float32)
    img += y_true.astype(np.float32) * 0.45
    img = np.clip(img, 0.0, 1.0)
    x_u8 = (255.0 * img).astype(np.uint8)  # this is what Akida expects
    return x_u8[None, ..., None], y_true[None, ...]


# 1) load trained CNN
ann_model = tf.keras.models.load_model("rfi_seg_ann.keras", compile=False)
print("Loaded ANN/CNN model:", ann_model.name)

# 2) quantize (activ bits must be 1,2,4)
q_model = cnn2snn.quantize(
    ann_model,
    weight_quantization=8,
    activ_quantization=4,
    fold_BN=True
)
print("Quantized model created.")

# 3) convert to Akida model
# NOTE: input scaling is already inside the model (Rescaling(1/255)), so keep input_scaling=None.
akida_model = cnn2snn.convert(q_model, input_scaling=None)
print("Converted to Akida model.")
print(akida_model.summary())

# 4) predict on spectrogram-like uint8 input
x_u8, y_true = make_spectrogram_like_uint8(H, W)
y_akida = akida_model.predict(x_u8)
akida_mask = np.argmax(y_akida, axis=-1)

print("Akida mask unique:", np.unique(akida_mask), "True unique:", np.unique(y_true))
print("Pixel agreement:", np.mean((akida_mask == y_true).astype(np.float32)))