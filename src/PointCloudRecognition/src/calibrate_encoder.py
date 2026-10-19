# calibrate_encoder.py
"""
Calibration utilities for SDRPlaceEncoder.

This module derives encoder parameters from a dataset of point clouds:
global statistics, ranges, resolutions, collision checks, and JSON
persistence. It is kept separate from place_encoder.py so that the core
encoder remains dataset-agnostic.

Pipeline:
    1. collect_global_stats()   - run PlaceDescriptor on all clouds
    2. get_encoder_ranges()     - derive (lo, hi) per feature, clippedy
    3. compute_resolutions()    - derive resolution per feature
    4. build_encoder_from_stats() - instantiate SDRPlaceEncoder
    5. save_encoder_config()    - persist to JSON for reuse
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np

root_project = Path(__file__).resolve().parent.parent # py/htm/examples/PointCloudRecognition
sys.path.append(str(root_project))

from src.place_encoder import PointCloud, PlaceDescriptor, SDRPlaceEncoder


# ============================================================
# Feature schema
# ============================================================
# Feature names, in the same order as PlaceDescriptor.to_vector().
FEATURE_NAMES: List[str] = (
    [f"eigval_{i+1}" for i in range(3)] +       # 0..2
    [f"hist_z_{i}"   for i in range(10)] +      # 3..12
    [f"hist_r_{i}"   for i in range(10)] +      # 13..22
    ["height", "density", "volume",             # 23..25
     "mean_radius", "std_radius",               # 26..27
     "mean_height", "std_height"]               # 28..29  <-- NOVO
)
# Physical lower bounds per feature. All our features are non-negative.
FEATURE_LOWER_BOUND = np.array(
    [0.0] * 3 +        # eigval_1..3
    [0.0] * 10 +       # hist_z_0..9
    [0.0] * 10 +       # hist_r_0..9
    [0.0,              # height
     0.0,              # density
     0.0,              # volume
     0.0,              # mean_radius
     0.0,              # std_radius
     -np.inf,          # mean_height  <-- pode ser negativo!
     -np.inf],         # std_height   <-- std é >= 0, mas deixamos -inf por segurança
    dtype=np.float64,
)

# Physical upper bounds per feature. Only the normalized histograms have a
# hard ceiling at 1.0; the rest are unbounded in principle.
FEATURE_UPPER_BOUND = np.array(
    [np.inf] * 3 +
    [1.0] * 10 +
    [1.0] * 10 +
    [np.inf, np.inf, np.inf, np.inf, np.inf,
     np.inf,           # mean_height
     np.inf],          # std_height
    dtype=np.float64,
)


# ============================================================
# 1. Global statistics with outlier filtering
# ============================================================
def collect_global_stats(
    npy_paths: Sequence[Path],
    n_bins_z: int = 10,
    n_bins_r: int = 10,
    reject_outliers: bool = True,
    max_mean_radius: float = 15.0,
    max_volume: float = 500.0,
    max_eigval_1: float = 100.0,
    min_points: int = 100,
) -> Tuple[dict, np.ndarray, List[Tuple[str, str]]]:
    """
    Run PlaceDescriptor on every point cloud and collect per-feature statistics.

    Outliers can be filtered by threshold on high-level descriptors. Set
    `reject_outliers=False` to disable.

    Returns
    -------
    stats : dict
        Per-feature min, max, p05, p95, median and number of samples used.
    V : np.ndarray
        (M, D) matrix of descriptor vectors (M = kept clouds, D = 28).
    rejected : list of (filename, reason)
        Clouds that were dropped, with the reason.
    """
    vectors: List[np.ndarray] = []
    rejected: List[Tuple[str, str]] = []

    for path in npy_paths:
        try:
            pc = PointCloud.from_npy(path)
        except Exception as e:
            rejected.append((path.name, f"load error: {e}"))
            continue

        try:
            desc = PlaceDescriptor.from_pointcloud(pc, n_bins_z, n_bins_r)
        except ValueError as e:
            rejected.append((path.name, f"descriptor error: {e}"))
            continue

        if reject_outliers:
            reason = None
            if pc.n_points < min_points:
                reason = f"too few points ({pc.n_points})"
            elif desc.mean_radius > max_mean_radius:
                reason = f"mean_radius={desc.mean_radius:.2f} > {max_mean_radius}"
            elif desc.volume > max_volume:
                reason = f"volume={desc.volume:.2f} > {max_volume}"
            elif desc.eigvals[0] > max_eigval_1:
                reason = f"eigval_1={desc.eigvals[0]:.2f} > {max_eigval_1}"
            if reason is not None:
                rejected.append((path.name, reason))
                continue

        vectors.append(desc.to_vector())

    if not vectors:
        raise ValueError(
            "All point clouds were rejected or failed. "
            f"Checked {len(npy_paths)} files, {len(rejected)} rejected."
        )

    V = np.array(vectors)  # (M (Accepted Clouds), D (descriptor Features))
    stats = {
        "min":        V.min(axis=0).tolist(),
        "max":        V.max(axis=0).tolist(),
        "p05":        np.percentile(V, 5,  axis=0).tolist(),
        "p95":        np.percentile(V, 95, axis=0).tolist(),
        "median":     np.median(V, axis=0).tolist(),
        "n_samples":  len(vectors),
        "n_rejected": len(rejected),
    }
    return stats, V, rejected


# ============================================================
# 2. Ranges and resolutions (with physical clipping)
# ============================================================
def get_encoder_ranges(
    stats: dict,
    margin: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn global statistics into encoder (lo, hi) ranges per feature.

    Steps:
        1. lo, hi = p05, p95
        2. expand by margin * span on both sides
        3. clip lo at FEATURE_LOWER_BOUND (avoid impossible negatives)
        4. clip hi at FEATURE_UPPER_BOUND (e.g. histograms <= 1)
    """
    lo = np.array(stats["p05"], dtype=np.float64)
    hi = np.array(stats["p95"], dtype=np.float64)
    span = hi - lo
    lo = lo - margin * span
    hi = hi + margin * span

    # Physical clipping
    lo = np.maximum(lo, FEATURE_LOWER_BOUND)
    hi = np.minimum(hi, FEATURE_UPPER_BOUND)

    # Safety: if hi <= lo (degenerate feature), expand by 1.0
    degenerate = hi <= lo
    if np.any(degenerate):
        hi[degenerate] = lo[degenerate] + 1.0

    return lo, hi


def compute_resolutions(
    lo: np.ndarray,
    hi: np.ndarray,
    num_buckets: int = 50,
) -> np.ndarray:
    """Compute a resolution per feature for a target number of buckets."""
    span = hi - lo
    return np.maximum(span / num_buckets, 1e-6)


# ============================================================
# 3. Collision check
# ============================================================
def check_rdse_params(
    size: int,
    active_bits: int,
    num_buckets: int,
    feature_index: int = -1,
    feature_name: str = "",
    max_ratio: float = 2.0,
) -> None:
    """
    Estimate RDSE hash-collision risk before instantiating it.

    The RDSE implementation tolerates ratios up to ~2.5 in practice.
    We use 2.0 as a safety margin.
    """
    ratio = (active_bits * num_buckets) / max(size, 1)
    if ratio > max_ratio:
        suggested_size = int(np.ceil(active_bits * num_buckets * 1.5))
        suggested_buckets = max(1, int(size / (1.5 * active_bits)))
        tag = f"Feature {feature_index} ({feature_name})" if feature_name else "RDSE"
        raise ValueError(
            f"{tag}: RDSE collision risk too high.\n"
            f"  active_bits={active_bits}, num_buckets={num_buckets}, "
            f"size={size} -> ratio={ratio:.2f} (must be < {max_ratio}).\n"
            f"  Try: size >= {suggested_size}, "
            f"or num_buckets <= {suggested_buckets}, "
            f"or reduce active_bits."
        )


# ============================================================
# 4. Encoder construction
# ============================================================
def build_encoder_from_stats(
    stats: dict,
    num_buckets: int = 50,
    feature_sizes: int | Sequence[int] = 2000,
    active_bits: int | Sequence[int] = 21,
    seed: int = 42,
) -> SDRPlaceEncoder:
    """
    Build a fully calibrated SDRPlaceEncoder from global statistics.

    feature_sizes and active_bits can be:
        - an int → same value for every feature
        - a Sequence → per-feature values

    Raises
    ------
    ValueError
        If RDSE parameters would cause excessive hash collisions.
    """
    lo, hi = get_encoder_ranges(stats)
    resolutions = compute_resolutions(lo, hi, num_buckets)

    n_feat = len(lo)

    # --- Normalize feature_sizes ---
    if isinstance(feature_sizes, int):
        sizes = [feature_sizes] * n_feat
    else:
        sizes = list(feature_sizes)
        if len(sizes) != n_feat:
            raise ValueError(
                f"feature_sizes has {len(sizes)} entries, "
                f"expected {n_feat}."
            )

    # --- Normalize active_bits ---
    if isinstance(active_bits, int):
        active_list = [active_bits] * n_feat
    else:
        active_list = list(active_bits)
        if len(active_list) != n_feat:
            raise ValueError(
                f"active_bits has {len(active_list)} entries, "
                f"expected {n_feat}."
            )

    # ---- Diagnostic table ----
    print(f"\n{'#':>3} | {'feature':<14} | {'buckets':>8} | {'size':>6} "
          f"| {'w':>3} | {'ratio':>7} | {'spars':>6} | status")
    print("-" * 78)
    for i in range(n_feat):
        res = resolutions[i]
        sz = sizes[i]
        w = active_list[i]
        nb = max(1, int((hi[i] - lo[i]) / max(res, 1e-12)))
        ratio = (w * nb) / max(sz, 1)
        spars = w / max(sz, 1)
        flags = []
        if ratio > 2.0:
            flags.append("RATIO HIGH")
        if spars > 0.3:
            flags.append("SPARSITY HIGH")
        if w < 20:
            flags.append("W<20")
        status = "OK" if not flags else " ".join(flags)
        print(f"{i:>3} | {FEATURE_NAMES[i]:<14} | {nb:>8d} | {sz:>6d} "
              f"| {w:>3d} | {ratio:>7.2f} | {spars:>6.3f} | {status}")
    print()

    # ---- Pre-flight collision check ----
    for i in range(n_feat):
        nb = max(1, int((hi[i] - lo[i]) / max(resolutions[i], 1e-12)))
        check_rdse_params(
            size=sizes[i],
            active_bits=active_list[i],
            num_buckets=nb,
            feature_index=i,
            feature_name=FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else "",
            max_ratio=2.0,
        )

    return SDRPlaceEncoder(
        feature_ranges=list(zip(lo.tolist(), hi.tolist())),
        feature_resolutions=resolutions.tolist(),
        feature_sizes=sizes,
        active_bits=active_list,
        seed=seed,
    )

# ============================================================
# 5. Persistence
# ============================================================
def save_encoder_config(
    path: Path,
    lo: np.ndarray,
    hi: np.ndarray,
    resolutions: np.ndarray,
    feature_sizes: Sequence[int],
    active_bits: int | Sequence[int],
    seed: int,
) -> None:
    """Persist encoder parameters to a JSON file."""
    # Normalize active_bits to a list for JSON
    if isinstance(active_bits, int):
        active_bits_list = [active_bits] * len(feature_sizes)
    else:
        active_bits_list = list(active_bits)

    config = {
        "feature_ranges":      list(zip(np.asarray(lo).tolist(),
                                        np.asarray(hi).tolist())),
        "feature_resolutions": np.asarray(resolutions).tolist(),
        "feature_sizes":       list(feature_sizes),
        "active_bits":         active_bits_list,
        "seed":                seed,
        "feature_names":       FEATURE_NAMES,
    }
    Path(path).write_text(json.dumps(config, indent=2))


def load_encoder_from_config(path: Path) -> SDRPlaceEncoder:
    """Rebuild an SDRPlaceEncoder from a saved JSON config."""
    config = json.loads(Path(path).read_text())
    return SDRPlaceEncoder(
        feature_ranges=config["feature_ranges"],
        feature_resolutions=config["feature_resolutions"],
        feature_sizes=config["feature_sizes"],
        active_bits=config["active_bits"],   # pode ser int ou list
        seed=config["seed"],
    )

# ============================================================
# 6. Summary table
# ============================================================
def print_encoder_summary(
    encoder: SDRPlaceEncoder,
    stats: dict,
    num_buckets: int = 50,
) -> None:
    """Print a per-feature summary of the encoder configuration."""
    lo, hi = get_encoder_ranges(stats)
    resolutions = compute_resolutions(lo, hi, num_buckets)

    print()
    print(f"{'#':>3} | {'feature':<14} | {'lo':>10} | {'hi':>10} "
          f"| {'resolution':>11} | {'size':>6} | {'w':>3} | {'buckets':>8}")
    print("-" * 88)
    for i, name in enumerate(FEATURE_NAMES):
        size = encoder.feature_sizes[i]
        w = encoder.active_bits_list[i]
        nb = max(1, int((hi[i] - lo[i]) / max(resolutions[i], 1e-12)))
        print(f"{i:>3} | {name:<14} | {lo[i]:>10.4f} | {hi[i]:>10.4f} "
              f"| {resolutions[i]:>11.6f} | {size:>6} | "
              f"{w:>3} | {nb:>8d}")

    total_size = encoder.total_size
    total_active = sum(encoder.active_bits_list)
    print("-" * 88)
    print(f"Total SDR size  : {total_size} bits")
    print(f"Total active    : {total_active} bits")
    print(f"Sparsity        : {total_active / total_size:.4f} "
          f"({100 * total_active / total_size:.2f} %)")
    print(f"Num features    : {len(FEATURE_NAMES)}")


# ============================================================
# 7. Calibration script
# ============================================================
if __name__ == "__main__":
    # ---- Config ----
    DATA_DIR = Path("/home/fabio/Documents/SPOT_Data/extracted_spot_ros2_data")
    CONFIG_PATH = Path("encoder_config.json")

    NUM_BUCKETS   = 50
    SEED          = 42

    # ------------------------------------------------------------
    # Feature sizes (bits per feature)
    # ------------------------------------------------------------
    # Reasoning: features with high mutual correlation (see
    # diagnose_feature_redundancy.py) receive fewer bits. Features
    # that are independent and informative receive more.
    #
    # Correlation blocks found in the dataset:
    #   - eigval_1, eigval_2, volume, mean_radius  (all scale-related)
    #   - eigval_3, mean_height, std_height        (vertical-related)
    # hist_z is independent (but low variance) — keep moderate size.
    # hist_r is the most discriminative block — give it full size.
    # ------------------------------------------------------------
    # FEATURE_SIZES = [
    #     # eigvals (3): λ1, λ2 redundant with each other; λ3 is unique
    #     840,  840,  2520,
    #     # hist_z (10): low variance but independent → moderate
    #     840, 840, 840, 840, 840,
    #     840, 840, 840, 840, 840,
    #     # hist_r (10): most discriminative → full
    #     2520, 2520, 2520, 2520, 2520,
    #     2520, 2520, 2520, 2520, 2520,
    #     # scalars (7)
    #     2520,   # height       (independent, informative)
    #     1680,   # density      (independent, moderate)
    #     840,    # volume       (redundant with eigval_1/2)
    #     1344,   # mean_radius  (redundant with volume)
    #     2520,   # std_radius   (more informative than mean_radius)
    #     2520,   # mean_height  (new, informative)
    #     2520,   # std_height   (new, informative)
    # ]

    FEATURE_SIZES = [
            2000, 2000, 600,
            2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
            600, 600, 600, 600, 600, 600, 600, 600, 600, 600,
            2000, 2000, 600, 2000, 600, 2000, 2000,
        ]
    # ------------------------------------------------------------
    # Active bits per feature
    # ------------------------------------------------------------
    # Intuition: larger SDRs can afford more active bits.
    # Rule of thumb: w ≈ 1.5% of size, with a floor of 21.
    # We set w proportional to sqrt(size) or to size / 100, capped.
    # For this experiment: give more active bits to larger features.
    ACTIVE_BITS = [
            # eigvals (3)
            21, 21, 21,             # size 840, 840, 2520
            # hist_z (10)
            21, 21, 21, 21, 21,
            21, 21, 21, 21, 21,     # size 1680
            # hist_r (10)
            21, 21, 21, 21, 21,
            21, 21, 21, 21, 21,     # size 2520
            # scalars (7)
            21,   # height       (2520)
            21,   # density      (1680)
            21,   # volume       (840)
            21,   # mean_radius  (1344)
            21,   # std_radius   (2520)
            21,   # mean_height  (2520)
            21,   # std_height   (2520)
        ]
    
    # Sanity checks
    n_expected = len(FEATURE_NAMES)
    if len(FEATURE_SIZES) != n_expected:
        raise SystemExit(
            f"FEATURE_SIZES has {len(FEATURE_SIZES)} entries, "
            f"expected {n_expected}."
        )
    if len(ACTIVE_BITS) != n_expected:
        raise SystemExit(
            f"ACTIVE_BITS has {len(ACTIVE_BITS)} entries, "
            f"expected {n_expected}."
        )
    if any(w >= s for w, s in zip(ACTIVE_BITS, FEATURE_SIZES)):
        raise SystemExit("Some features have active_bits >= size.")

    total_bits = sum(FEATURE_SIZES)
    total_active = sum(ACTIVE_BITS)
    print(f"Feature config: {len(FEATURE_SIZES)} features, "
          f"{total_bits} bits, {total_active} active "
          f"({100 * total_active / total_bits:.2f}% sparsity)")

    # Outlier thresholds
    REJECT_OUTLIERS  = True
    MAX_MEAN_RADIUS  = 50.0
    MAX_VOLUME       = 5000.0
    MAX_EIGVAL_1     = 1000.0
    MIN_POINTS       = 100

    # ---- 1. Discover dataset ----
    npy_paths = sorted(DATA_DIR.glob("*.npy"))
    print(f"\nFound {len(npy_paths)} point clouds in {DATA_DIR}")
    if not npy_paths:
        raise SystemExit("No .npy files found. Check DATA_DIR.")

    # ---- 2. Collect statistics ----
    print("\nCollecting global statistics...")
    stats, V, rejected = collect_global_stats(
        npy_paths,
        reject_outliers=REJECT_OUTLIERS,
        max_mean_radius=MAX_MEAN_RADIUS,
        max_volume=MAX_VOLUME,
        max_eigval_1=MAX_EIGVAL_1,
        min_points=MIN_POINTS,
    )
    print(f"  descriptor matrix: {V.shape} "
          f"({V.shape[0]} clouds x {V.shape[1]} features)")

    # ---- 3. Report rejected clouds ----
    if rejected:
        print(f"\n  Rejected {len(rejected)} cloud(s):")
        for name, reason in rejected[:20]:
            print(f"    - {name}: {reason}")
        if len(rejected) > 20:
            print(f"    ... and {len(rejected) - 20} more")
    else:
        print("  No clouds rejected.")

    # ---- 4. Build encoder ----
    print("\nBuilding encoder (this also runs collision checks)...")
    encoder = build_encoder_from_stats(
        stats,
        num_buckets=NUM_BUCKETS,
        feature_sizes=FEATURE_SIZES,
        active_bits=ACTIVE_BITS,
        seed=SEED,
    )

    # ---- 5. Summary ----
    print_encoder_summary(encoder, stats, num_buckets=NUM_BUCKETS)

    # ---- 6. Save config ----
    lo, hi = get_encoder_ranges(stats)
    resolutions = compute_resolutions(lo, hi, num_buckets=NUM_BUCKETS)
    save_encoder_config(
        path=CONFIG_PATH,
        lo=lo, hi=hi,
        resolutions=resolutions,
        feature_sizes=FEATURE_SIZES,
        active_bits=ACTIVE_BITS,
        seed=SEED,
    )
    print(f"\nConfig saved to: {CONFIG_PATH.resolve()}")

    # ---- 7. Smoke test ----
    sample_path = npy_paths[0]
    pc = PointCloud.from_npy(sample_path)
    desc = PlaceDescriptor.from_pointcloud(pc)
    sdr = encoder.encode(desc.to_vector())

    print(f"\nSmoke test on {sample_path.name}:")
    print(f"  SDR shape     : {sdr.shape}")
    print(f"  Active bits   : {int(sdr.sum())}")
    print(f"  Sparsity      : {sdr.mean():.4f} "
          f"({100 * sdr.mean():.2f} %)")

