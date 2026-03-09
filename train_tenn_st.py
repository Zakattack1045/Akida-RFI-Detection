# ============================
# train_tenn_st.py
# PURPOSE: Train a proper spatiotemporal TENN model for RFI segmentation using
#          Brainchip's tenns_modules SpatioTemporalBlock (PyTorch).
#
# Architecture: 3 SpatioTemporalBlocks (temporal conv + spatial conv, s_stride=1
#   to maintain 64x64 spatial resolution) + 1x1x1 Conv3d segmentation head.
#
# Key advantage over BufferTempConv training: ALL layers receive gradients because
#   temporal convolutions are standard differentiable PyTorch ops. At quantization
#   time, quantizeml.quantize() automatically converts temporal convolutions to
#   FIFO-based BufferTempConv layers ("bufferization").
#
# Pipeline: train → export_to_onnx → (convert_tenn_st.py: quantize+bufferize → evaluate)
# ============================

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tenns_modules import SpatioTemporalBlock, export_to_onnx


# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_ROOT = "Data/Akida_Dataset"
TRAIN_SPLIT  = "Combined_Dataset"
TEST_SPLIT   = "Combined_Test"

INPUT_H = 64
INPUT_W = 64

SEQUENCE_LEN = 4    # temporal window (FIFO size after bufferization)

BATCH_SIZE = 8
EPOCHS     = 3
LR         = 1e-3

MODEL_SAVE_ROOT = "Models/V4"
ONNX_SAVE_PATH  = os.path.join(MODEL_SAVE_ROOT, "Akida_tenn_train.onnx")

SAVE_SAMPLES = True
N_SAMPLES    = 8
SAMPLES_DIR  = os.path.join(MODEL_SAVE_ROOT, "Samples", "TENN_CNN")


# ── Dataset ────────────────────────────────────────────────────────────────────

class RFISequenceDataset(Dataset):
    """
    Non-overlapping windows of SEQUENCE_LEN consecutive spectrogram frames.
    Item:
      frames: (1, T, H, W)  float32 [0,1]  — channels-first, PyTorch convention
      mask:   (H, W)        int64           — last frame's binary mask
    """
    def __init__(self, root, split_name, seq_len):
        spec_dir = os.path.join(root, split_name, "spectrograms")
        mask_dir = os.path.join(root, split_name, "masks")
        fnames   = sorted(f for f in os.listdir(spec_dir) if f.lower().endswith(".png"))

        n_seqs = len(fnames) // seq_len
        fnames = fnames[:n_seqs * seq_len]

        self.sequences = []
        for i in range(0, len(fnames), seq_len):
            window = fnames[i:i + seq_len]
            spec_paths = [os.path.join(spec_dir, f) for f in window]
            mask_path  = os.path.join(mask_dir, window[-1])
            if os.path.exists(mask_path):
                self.sequences.append((spec_paths, mask_path))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        spec_paths, mask_path = self.sequences[idx]

        # Load T frames → (1, T, H, W) float32 [0,1]
        frames = []
        for p in spec_paths:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            frames.append(arr)
        frames = np.stack(frames, axis=0)       # (T, H, W)
        frames = frames[np.newaxis, ...]        # (1, T, H, W)
        frames = torch.from_numpy(frames)

        # Load last frame's mask → (H, W) int64
        mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if mask_raw.max() > 1:
            mask_raw = (mask_raw > 127).astype(np.uint8)
        mask = torch.from_numpy(mask_raw.astype(np.int64))

        return frames, mask


# ── Model ──────────────────────────────────────────────────────────────────────

class TennRFISeg(nn.Module):
    """
    Spatiotemporal segmentation model using Brainchip SpatioTemporalBlock.

    Input:  (B, 1, T, H, W) — 1-channel spectrogram, T consecutive frames
    Output: (B, 2, T, H, W) — 2-class logits per frame per pixel

    SpatioTemporalBlock performs:
      - Causal temporal convolution (kernel_size=T along time axis)
      - Spatial convolution (3x3 depthwise or full)
    with s_stride=1 to maintain 64x64 spatial resolution throughout.

    At quantization time, quantizeml.quantize() converts the causal temporal
    convolutions to FIFO-based BufferTempConv layers (bufferization), enabling
    streaming inference one frame at a time.
    """

    def __init__(self, T=SEQUENCE_LEN):
        super().__init__()
        self.st1 = SpatioTemporalBlock(
            in_channels=1, med_channels=8, out_channels=8,
            t_kernel_size=T, s_kernel_size=3, s_stride=1, bias=False
        )
        self.st2 = SpatioTemporalBlock(
            in_channels=8, med_channels=16, out_channels=16,
            t_kernel_size=T, s_kernel_size=3, s_stride=1, bias=False
        )
        self.st3 = SpatioTemporalBlock(
            in_channels=16, med_channels=8, out_channels=8,
            t_kernel_size=T, s_kernel_size=3, s_stride=1, bias=False
        )
        # 1×1×1 Conv3d — acts as a per-pixel, per-timestep linear classifier
        self.head = nn.Conv3d(8, 2, kernel_size=1, bias=True)

    def forward(self, input):
        # input: (B, 1, T, H, W)
        # Arg named 'input' to match tenns_modules.export_to_onnx dynamic_shapes key.
        x = self.st1(input)   # (B,  8, T, H, W)
        x = self.st2(x)       # (B, 16, T, H, W)
        x = self.st3(x)       # (B,  8, T, H, W)
        return self.head(x)   # (B,  2, T, H, W)


# ── Loss ───────────────────────────────────────────────────────────────────────

# Higher weight on RFI class to counteract class imbalance
_class_weights = torch.tensor([0.2, 0.8], dtype=torch.float32)


def weighted_loss(logits_last, masks):
    """
    logits_last: (B, 2, H, W) — last timestep's logits
    masks:       (B, H, W)    — int64 ground truth
    """
    return F.cross_entropy(logits_last, masks, weight=_class_weights.to(logits_last.device))


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(model, loader, device):
    """Run validation, return per-pixel TP/FP/TN/FN and scalar metrics."""
    model.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for frames, masks in loader:
            frames = frames.to(device)
            masks  = masks.to(device)

            logits = model(frames)                       # (B, 2, T, H, W)
            logits_last = logits[:, :, -1, :, :]        # (B, 2, H, W)

            total_loss += weighted_loss(logits_last, masks).item()
            n_batches  += 1

            preds  = logits_last.argmax(dim=1)           # (B, H, W)
            tp += ((preds == 1) & (masks == 1)).sum().item()
            fp += ((preds == 1) & (masks == 0)).sum().item()
            tn += ((preds == 0) & (masks == 0)).sum().item()
            fn += ((preds == 0) & (masks == 1)).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)

    return {
        "loss": total_loss / max(n_batches, 1),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_sample_images(frames_seq, masks, preds, save_dir, prefix="sample"):
    """
    frames_seq: list of (1, T, H, W) float32 tensors
    masks:      list of (H, W) int64 tensors
    preds:      list of (H, W) numpy int arrays
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(frames_seq)):
        spec_img = frames_seq[i][0, -1].numpy()  # last frame, (H, W)
        mask     = masks[i].numpy()
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Input: {INPUT_H}x{INPUT_W}  Sequence length: {SEQUENCE_LEN}")

    train_ds = RFISequenceDataset(DATASET_ROOT, TRAIN_SPLIT, SEQUENCE_LEN)
    val_ds   = RFISequenceDataset(DATASET_ROOT, TEST_SPLIT,  SEQUENCE_LEN)
    print(f"Train sequences: {len(train_ds)}  Val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = TennRFISeg(T=SEQUENCE_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for frames, masks in train_loader:
            frames = frames.to(device)
            masks  = masks.to(device)

            logits = model(frames)                      # (B, 2, T, H, W)
            logits_last = logits[:, :, -1, :, :]       # (B, 2, H, W)
            loss = weighted_loss(logits_last, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──────────────────────────────────────────────────────────
        m = compute_metrics(model, val_loader, device)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={np.mean(train_losses):.4f}  "
              f"val_loss={m['loss']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)
    model.cpu().eval()
    print(f"\nExporting model to ONNX at {ONNX_SAVE_PATH} ...")
    export_to_onnx(model, (1, 1, SEQUENCE_LEN, INPUT_H, INPUT_W), out_path=ONNX_SAVE_PATH)
    print(f"ONNX export complete: {ONNX_SAVE_PATH}")

    # ── Sample visualisations ─────────────────────────────────────────────────
    if SAVE_SAMPLES:
        print(f"\nGenerating {N_SAMPLES} sample images ...")
        model.eval()
        sample_frames, sample_masks, sample_preds = [], [], []

        with torch.no_grad():
            for frames, mask in val_ds:
                logits = model(frames.unsqueeze(0))         # (1, 2, T, H, W)
                logits_last = logits[0, :, -1, :, :]       # (2, H, W)
                pred = logits_last.argmax(dim=0).numpy()    # (H, W)

                if mask.sum() == 0:   # skip empty-mask (no RFI) frames
                    continue
                sample_frames.append(frames)
                sample_masks.append(mask)
                sample_preds.append(pred)

                if len(sample_frames) >= N_SAMPLES:
                    break

        save_sample_images(sample_frames, sample_masks, sample_preds,
                           SAMPLES_DIR, prefix="tenn_cnn")


if __name__ == "__main__":
    main()
