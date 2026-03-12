# ============================
# energy_estimate.py
# PURPOSE: Theoretical energy estimation for CNN and Akida SNN models.
#
# CNN energy:
#   Theoretical: MACs = H_out × W_out × C_out × kH × kW × C_in (all neurons active)
#   Active only: count(input_activation > 0) × kH × kW × C_out (sparsity-aware)
#
# SNN energy (two models shown side by side):
#   Rate-coded:    activation value = spike count → SynOps = sum(values) × fan_out
#   Latency-coded: each neuron fires at most once (time-to-first-spike) → SynOps = count(>0) × fan_out
#   The correct model depends on Akida's internal spike encoding; both are plausible.
#   The Akida 2.x Python API does not expose per-layer spike counts in software
#   simulation (model.statistics only populates when mapped to physical hardware).
#
# Energy constants (configurable below):
#   E_MAC_FP32 = 4.6  pJ  (Horowitz, "1.1 computing's energy problem", ISSCC 2014)
#   E_MAC_INT8 = 0.47 pJ  (Horowitz ISSCC 2014)
#   E_SYN      = 0.9  pJ  (Merolla et al., "A million spiking-neuron integrated
#                           circuit", Science 2014)
# ============================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from PIL import Image


# ── Config ─────────────────────────────────────────────────────────────────────

CNN_PATH  = "Models/Akida_v1.keras"
SNN_PATH  = "Models/Akida_v1.fbz"     # run convert_to_akida.py first to generate
TEST_DIR  = "Data/Akida_Dataset/Combined_Test"
N_EVAL    = 10   # test samples used for empirical counts (more = more stable)

# Must match the ACTIV_BITS value used in convert_to_akida.py.
# Spike counts are estimated by scaling float activations to [0, 2^ACTIV_BITS - 1].
ACTIV_BITS = 4

# Energy per operation (Joules) — adjust to explore sensitivity.
E_MAC_FP32 = 4.60e-12   # FP32 multiply-accumulate (Horowitz 2014)
E_MAC_INT8 = 0.47e-12   # INT8 multiply-accumulate (Horowitz 2014)
E_SYN      = 3e-12    # (Akida)
# E_SYN      = 26e-12   # Neuromorphic synaptic operation (IBM True North) (Merolla et al. 2014)
# E_SYN      = 23.6e-12   # (Lava)

# Memory access energy (Joules/byte) — Horowitz 2014 ISSCC.
# DRAM:  640 pJ per 32-bit word → 640/4 = 160 pJ/byte  (off-chip, worst case)
# SRAM:    5 pJ per 32-bit word →   5/4 = 1.25 pJ/byte (on-chip, best case)
E_DRAM_PER_BYTE = 160e-12   # J/byte
E_SRAM_PER_BYTE = 1.25e-12  # J/byte


# ── Data loading ───────────────────────────────────────────────────────────────

def load_samples(test_dir, n):
    """
    Load up to N test spectrograms as float32 [0, 1] and binary int32 masks.
    Returns (x_float, y_masks): shapes (N, H, W, 1) and (N, H, W).
    """
    spec_dir = os.path.join(test_dir, "spectrograms")
    mask_dir = os.path.join(test_dir, "masks")
    fnames = sorted(
        f for f in os.listdir(spec_dir) if f.lower().endswith(".png")
    )[:n]

    specs, masks = [], []
    for fname in fnames:
        spec = np.array(
            Image.open(os.path.join(spec_dir, fname)).convert("L"),
            dtype=np.float32,
        )[..., None] / 255.0  # (H, W, 1), values 0-1

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_conv_layers(model):
    """Return all Conv2D layers in the model, in forward order."""
    return [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]


def build_extractor(model, conv_layers):
    """
    Build a single multi-output Keras model that captures the input tensor
    feeding into each Conv2D layer.  One predict() call retrieves all tensors.
    """
    return tf.keras.Model(
        inputs=model.input,
        outputs=[l.input for l in conv_layers],
    )


def run_extractor(extractor, x):
    """
    Run the extractor and always return a list (one array per Conv2D layer),
    even if the model happens to have only one Conv2D.
    """
    result = extractor.predict(x, verbose=0)
    if isinstance(result, np.ndarray):
        result = [result]   # single-output models return a bare ndarray
    return result


def layer_fan_out(layer):
    """kH × kW × C_out for a Conv2D — the number of downstream synapses per spike."""
    kH, kW = layer.kernel_size
    return kH * kW * layer.filters


# ── CNN analysis ───────────────────────────────────────────────────────────────


def count_cnn_macs(model, conv_layers):
    """
    Analytical MAC count per Conv2D layer.
    MACs = H_out × W_out × C_out × kH × kW × C_in
    No inference required — derived entirely from model architecture.
    Returns {layer_name: int}.
    """
    macs = {}
    for layer in conv_layers:
        _, H_out, W_out, C_out = layer.output_shape
        _, H_in,  W_in,  C_in  = layer.input_shape
        kH, kW = layer.kernel_size
        macs[layer.name] = int(H_out * W_out * C_out * kH * kW * C_in)
    return macs


def count_cnn_active_ops(conv_layers, all_inputs, n_samples):
    """
    Sparsity-aware CNN operation count.
    Each non-zero input activation fires exactly once ("count once" model).
    active_ops_L = mean_samples( count(activation > 0) × kH × kW × C_out )
    Returns {layer_name: float}.
    """
    result = {}
    for layer, act in zip(conv_layers, all_inputs):
        # act: (N, H, W, C_in)
        fo = layer_fan_out(layer)
        active_per_sample = (act > 0).reshape(n_samples, -1).sum(axis=1) * fo
        result[layer.name] = float(active_per_sample.mean())
    return result


def compute_cnn_memory_bounds(conv_layers, bits_weights, bits_acts):
    """
    Lower and upper bounds on CNN memory access energy per inference.

    Lower bound (ideal on-chip SRAM, perfect reuse):
        Each weight and activation loaded exactly once.
        Data moved per layer = weight_bytes + input_act_bytes + output_act_bytes
        Energy = total_bytes × E_SRAM_PER_BYTE
        Achievable on a custom ASIC/FPGA with sufficient on-chip SRAM.

    Upper bound (no caching, all-DRAM streaming):
        Each MAC re-fetches its weight and input activation from DRAM.
        No spatial reuse of weights across output positions.
        Data moved per layer = MACs × (bytes_per_weight + bytes_per_act)
                               + output_act_bytes (written once)
        Energy = total_bytes × E_DRAM_PER_BYTE
        Worst case for CPU/GPU without explicit data-reuse management.

    Sub-byte precisions (4 bits = 0.5 bytes) use fractional bytes to represent
    the amortised cost of packed element pairs (Akida packs two 4-bit values
    per byte access).

    Returns (e_lower_J, e_upper_J).
    """
    bpw = bits_weights / 8.0
    bpa = bits_acts    / 8.0
    e_lower = 0.0
    e_upper = 0.0

    for layer in conv_layers:
        _, H_out, W_out, C_out = layer.output_shape
        _, H_in,  W_in,  C_in  = layer.input_shape
        kH, kW = layer.kernel_size

        macs_layer       = H_out * W_out * C_out * kH * kW * C_in
        weight_bytes     = kH * kW * C_in * C_out     * bpw
        input_act_bytes  = H_in  * W_in  * C_in       * bpa
        output_act_bytes = H_out * W_out * C_out       * bpa

        # Lower: load every element once from on-chip SRAM
        e_lower += (weight_bytes + input_act_bytes + output_act_bytes) * E_SRAM_PER_BYTE

        # Upper: every MAC re-reads weight + input from DRAM; output written once
        e_upper += (macs_layer * bpw
                 +  macs_layer * bpa
                 +  output_act_bytes) * E_DRAM_PER_BYTE

    return e_lower, e_upper


# ── SNN analysis ───────────────────────────────────────────────────────────────

def count_snn_synops_rate(conv_layers, all_inputs, n_samples, activ_bits):
    """
    Rate-coded SNN SynOps estimate.

    Assumes each unit of activation = one binary spike (rate coding).  A neuron
    with quantized value 11/15 fires 11 times → 11 SynOps × fan_out.

    Float activations are scaled to [0, 2^activ_bits − 1]:
      - Input layer (pixel values in [0, 1]): scale directly.
      - Hidden layers (ReLU >= 0): clip at 99th-pct then scale.

    SynOps_L = mean_samples( sum(spike_values) × fan_out )
    Returns {layer_name: float}.
    """
    max_spikes = (2 ** activ_bits) - 1
    result = {}

    for i, (layer, act) in enumerate(zip(conv_layers, all_inputs)):
        fo = layer_fan_out(layer)

        if i == 0:
            spikes = np.round(act * max_spikes)
        else:
            clip_val = float(np.percentile(act, 99))
            if clip_val < 1e-9:
                spikes = np.zeros_like(act)
            else:
                spikes = np.round(
                    np.clip(act, 0.0, clip_val) / clip_val * max_spikes
                )

        synops_per_sample = spikes.reshape(n_samples, -1).sum(axis=1) * fo
        result[layer.name] = float(synops_per_sample.mean())

    return result


def count_snn_synops_latency(conv_layers, all_inputs, n_samples):
    """
    Latency-coded (time-to-first-spike) SNN SynOps estimate.

    Each neuron fires at most once per inference; the timing of that spike
    encodes the activation magnitude.  A non-zero activation = one synaptic
    event per downstream connection, regardless of the activation value.

    SynOps_L = mean_samples( count(activation > 0) × fan_out )
    Returns {layer_name: float}.
    """
    result = {}
    for layer, act in zip(conv_layers, all_inputs):
        fo = layer_fan_out(layer)
        synops_per_sample = (act > 0).reshape(n_samples, -1).sum(axis=1) * fo
        result[layer.name] = float(synops_per_sample.mean())
    return result


# ── Akida statistics probe ─────────────────────────────────────────────────────

def probe_akida_statistics(snn_path, x_float_one):
    """
    Load the Akida SNN model, run one inference, and print .statistics.
    In software-only mode (no Akida hardware) fps/energy/powers will be None
    or zero — they are printed here so that running on actual hardware would
    show real values automatically.
    """
    try:
        import akida
    except ImportError:
        print("  akida package not importable — skipping SNN statistics probe.")
        return

    if not os.path.exists(snn_path):
        print(f"  {snn_path} not found — skipping (run convert_to_akida.py first).")
        return

    print(f"  Loading SNN model from {snn_path} ...")
    snn_model = akida.Model(snn_path)
    print(f"  Loaded: {len(snn_model.layers)} layers.")

    # Show layer variables (useful for checking future akida versions)
    for layer in snn_model.layers:
        vnames = layer.get_variable_names()
        if vnames:
            print(f"    {layer.name}: available variables = {vnames}")

    # Run one sample (uint8 for Akida)
    x_u8 = np.round(x_float_one * 255).astype(np.uint8)  # (1, H, W, 1)
    snn_model.predict(x_u8)

    stats = snn_model.statistics
    for field in ("fps", "energy", "inference_clk", "powers"):
        val = getattr(stats, field, "(attribute not present)")
        print(f"  statistics.{field:<16}: {val}")

    print(
        "  (All None/zero in software mode — populate automatically with hardware.)"
    )


# ── Report ─────────────────────────────────────────────────────────────────────

def _fmt(n):
    return f"{int(n):>15,}"


def print_report(conv_layers, macs, synops_rate, synops_latency, activ_bits, n_eval):
    max_spikes = (2 ** activ_bits) - 1
    W = 72

    print("\n" + "─" * W)
    print(f"{'Layer':<28} {'MACs (theory)':>15}  {'SNN rate':>12}  {'SNN latency':>12}")
    print("─" * W)

    total_macs         = 0
    total_synops_rate  = 0.0
    total_synops_lat   = 0.0

    for layer in conv_layers:
        n  = layer.name
        m  = macs.get(n, 0)
        sr = synops_rate.get(n, 0.0)
        sl = synops_latency.get(n, 0.0)
        total_macs        += m
        total_synops_rate += sr
        total_synops_lat  += sl
        print(f"{n:<28} {_fmt(m)}  {int(sr):>12,}  {int(sl):>12,}")

    print("─" * W)
    print(
        f"{'TOTAL':<28} {_fmt(total_macs)}  "
        f"{int(total_synops_rate):>12,}  {int(total_synops_lat):>12,}"
    )

    max_possible = sum(macs.get(l.name, 0) * max_spikes for l in conv_layers)
    if max_possible > 0:
        pct_rate = 100.0 * total_synops_rate / max_possible
        pct_lat  = 100.0 * total_synops_lat  / max_possible
        print(f"\n  SNN spike activity — rate model    (fraction of max): {pct_rate:.2f}%")
        print(f"  SNN spike activity — latency model (fraction of max): {pct_lat:.2f}%")

    print(
        f"\n  Rate model:    activations scaled to [0, {max_spikes}] (ACTIV_BITS={activ_bits}).\n"
        f"  Latency model: counts any non-zero activation as one spike (time-to-first-spike).\n"
        f"  ACTIV_BITS must match convert_to_akida.py for the rate estimate to be valid.\n"
        f"  Evaluated on {n_eval} test samples."
    )
    print("─" * W)


def print_summary(conv_layers, macs, synops_rate, synops_latency, activ_bits, n_eval):
    """
    Unified energy summary printed last.

    CNN: FP32 and INT8 compute ops energy alongside SRAM (lower) and DRAM (upper)
         memory access bounds, giving total lower and upper bounds per inference.
    SNN: Rate-coded and latency-coded synaptic operation energy alongside neuron
         update energy (E_update = 2 × E_syn, 1 timestep assumed), giving totals.
    """
    W = 96

    total_macs        = sum(macs.values())
    total_synops_rate = sum(synops_rate.values())
    total_synops_lat  = sum(synops_latency.values())
    max_spikes        = (2 ** activ_bits) - 1

    def _uj(e):
        return f"{e * 1e6:.2f} µJ"

    # ── CNN ──────────────────────────────────────────────────────────────────────
    e_fp32 = total_macs * E_MAC_FP32
    e_int8 = total_macs * E_MAC_INT8
    mem_lo_fp32, mem_hi_fp32 = compute_cnn_memory_bounds(conv_layers, 32, 32)
    mem_lo_int8, mem_hi_int8 = compute_cnn_memory_bounds(conv_layers,  8,  8)

    # ── SNN ──────────────────────────────────────────────────────────────────────
    e_syn_rate = total_synops_rate * E_SYN
    e_upd_rate = e_syn_rate * 2            # E_update = 2 × E_syn, 1 timestep
    e_syn_lat  = total_synops_lat  * E_SYN
    e_upd_lat  = e_syn_lat  * 2

    print("\n" + "═" * W)
    print("ENERGY SUMMARY — per inference (64×64 spectrogram)")
    print("═" * W)

    # CNN table
    print(f"\n  CNN  [total MACs: {total_macs:,}]")
    print(
        f"  {'Precision':<8}  {'Compute ops':>13}  {'Mem lower (SRAM)':>18}"
        f"  {'Mem upper (DRAM)':>18}  {'Total lower':>13}  {'Total upper':>13}"
    )
    print("  " + "─" * (W - 2))
    for label, e_comp, mem_lo, mem_hi in (
        ("FP32", e_fp32, mem_lo_fp32, mem_hi_fp32),
        ("INT8", e_int8, mem_lo_int8, mem_hi_int8),
    ):
        print(
            f"  {label:<8}  {_uj(e_comp):>13}  {_uj(mem_lo):>18}"
            f"  {_uj(mem_hi):>18}  {_uj(e_comp + mem_lo):>13}  {_uj(e_comp + mem_hi):>13}"
        )

    # SNN table
    print(
        f"\n  SNN  [E_syn = {E_SYN*1e12:.2f} pJ  |  "
        f"neuron update = 2 × E_syn  |  1 timestep/inference]"
    )
    print(
        f"  {'Encoding':<16}  {'SynOps energy':>15}  {'Neuron updates':>16}  {'Total':>13}"
    )
    print("  " + "─" * (W - 2))
    for label, e_syn, e_upd in (
        ("Rate-coded",    e_syn_rate, e_upd_rate),
        ("Latency-coded", e_syn_lat,  e_upd_lat),
    ):
        print(
            f"  {label:<16}  {_uj(e_syn):>15}  {_uj(e_upd):>16}  {_uj(e_syn + e_upd):>13}"
        )

    print(
        f"\n  Memory bounds (Horowitz 2014 ISSCC):\n"
        f"  • SRAM lower: 1.25 pJ/byte (5 pJ / 32-bit word) "
        f"— ideal on-chip scratchpad, each weight/activation loaded exactly once\n"
        f"  • DRAM upper: 160 pJ/byte (640 pJ / 32-bit word) "
        f"— no caching, every MAC re-fetches weight and input activation from off-chip DRAM"
    )
    print("═" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Load CNN
    print(f"Loading CNN model from {CNN_PATH} ...")
    cnn_model = tf.keras.models.load_model(CNN_PATH, compile=False)
    print(f"Loaded: {cnn_model.name}  ({cnn_model.count_params():,} parameters)")

    # 2. Probe Akida SNN statistics (None in software mode; real values with hardware)
    print(f"\n─── Akida SNN Statistics Probe {'─' * 42}")
    x_probe, _ = load_samples(TEST_DIR, 1)
    probe_akida_statistics(SNN_PATH, x_probe)

    # 3. Load test samples for empirical activation counting
    print(f"\nLoading {N_EVAL} test samples for empirical analysis ...")
    x_float, _ = load_samples(TEST_DIR, N_EVAL)
    print(f"Loaded {len(x_float)} samples, shape {x_float.shape}")

    # 4. Analytical MAC count (architecture only, no inference needed)
    conv_layers = get_conv_layers(cnn_model)
    print(f"\nFound {len(conv_layers)} Conv2D layers.  Counting MACs analytically ...")
    macs = count_cnn_macs(cnn_model, conv_layers)

    # 5. Build activation extractor and run once for both CNN and SNN analyses
    print("Extracting per-layer activations (single forward pass) ...")
    extractor = build_extractor(cnn_model, conv_layers)
    all_inputs = run_extractor(extractor, x_float)

    # 6. SNN SynOps — both models
    print(f"Estimating SNN SynOps — rate-coded model (ACTIV_BITS={ACTIV_BITS}) ...")
    synops_rate = count_snn_synops_rate(conv_layers, all_inputs, len(x_float), ACTIV_BITS)

    print("Estimating SNN SynOps — latency-coded model ...")
    synops_latency = count_snn_synops_latency(conv_layers, all_inputs, len(x_float))

    # 7. Print full comparison report
    print_report(conv_layers, macs, synops_rate, synops_latency,
                 ACTIV_BITS, len(x_float))

    # 8. Unified energy summary (CNN compute + memory bounds; SNN synops + neuron updates)
    print_summary(conv_layers, macs, synops_rate, synops_latency,
                  ACTIV_BITS, len(x_float))


if __name__ == "__main__":
    main()
