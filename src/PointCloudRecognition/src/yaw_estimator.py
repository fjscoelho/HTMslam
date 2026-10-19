# yaw_estimator.py
"""
Yaw estimation between two point clouds captured from the same place.

Given two point clouds A and B that are known (or believed) to be from the
same physical location, estimate the relative yaw angle of B with respect
to A.

Method
------
1. Compute an angular density histogram for each cloud: for each point,
   compute theta = atan2(y - cy, x - cx) relative to the centroid, then
   bin the angles into N bins over (-pi, pi].

2. Compute the circular cross-correlation between the two histograms via
   FFT. The location of the peak gives the relative yaw.

3. Confidence: peak-to-sidelobe ratio. A tall, sharp peak means a
   confident estimate; a flat, ambiguous correlation means low confidence.

Optionally, refine with a multi-start 2D ICP restricted to yaw rotation.

Notes
-----
- The two clouds must be from the SAME place (same room, same area). If
  they are from different places, the yaw estimate is meaningless.
- Assumes the vertical axis is Z and the yaw is rotation around Z.
- The estimator is invariant to translation and works on XY projection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ============================================================
# Result container
# ============================================================
@dataclass
class YawEstimate:
    """Result of a yaw estimation."""
    yaw_rad: float              # estimated relative yaw in radians, in (-pi, pi]
    yaw_deg: float              # same in degrees, for convenience
    confidence: float           # peak-to-sidelobe ratio (higher = better)
    ambiguity: float            # 2nd-peak / 1st-peak ratio (lower = better)
    method: str                 # "histogram" | "histogram+icp"

    def __repr__(self):
        return (f"YawEstimate(yaw={self.yaw_deg:+.2f}°, "
                f"conf={self.confidence:.2f}, "
                f"ambiguity={self.ambiguity:.2f}, "
                f"method={self.method})")


# ============================================================
# 1. Angular histogram
# ============================================================
def angular_histogram(pc, n_bins: int = 360,
                     use_centroid: bool = True) -> np.ndarray:
    """
    Compute a circular angular density histogram for a point cloud.

    Parameters
    ----------
    pc : PointCloud or (N,3) array
        The point cloud. If an object with `.points`, that's used; otherwise
        the array is used directly.
    n_bins : int
        Number of angular bins. 360 gives 1° resolution.
    use_centroid : bool
        If True, angles are computed relative to the XY centroid.
        If False, angles are computed relative to the sensor origin.

    Returns
    -------
    hist : np.ndarray, shape (n_bins,)
        Normalized angular histogram (sums to 1).
    """
    points = pc.points if hasattr(pc, "points") else np.asarray(pc)
    xy = points[:, :2]

    if use_centroid:
        anchor = xy.mean(axis=0)
    else:
        anchor = np.array([0.0, 0.0])

    rel = xy - anchor
    theta = np.arctan2(rel[:, 1], rel[:, 0])   # in (-pi, pi]

    hist, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
    # Normalize to density so that total point count does not matter
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    else:
        hist = hist.astype(np.float64)
    return hist


# ============================================================
# 2. Cross-correlation of angular histograms
# ============================================================
def _circular_cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the circular cross-correlation between two 1D histograms via FFT.

    Returns an array `corr` of length len(a), where corr[k] is the dot
    product between `a` and `b` shifted by k bins. Positive k means `b`
    is shifted counterclockwise relative to `a`, matching the sign of
    the yaw returned by estimate_yaw().
    """
    n = len(a)
    if len(b) != n:
        raise ValueError("Histograms must have the same length.")
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    # Using conj(fa) * fb gives corr[k] = sum a[n] * b[n+k] in the FFT
    # indexing convention, which matches "shift of b" with the same sign
    # as the yaw angle.
    corr = np.fft.irfft(np.conj(fa) * fb, n=n)
    return corr


def estimate_yaw_from_histograms(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    refine_peak: bool = True,
) -> Tuple[float, float, float]:
    """
    Estimate the relative yaw between two angular histograms.

    Parameters
    ----------
    hist_a, hist_b : np.ndarray
        Angular histograms of the same length.
    refine_peak : bool
        If True, refine the discrete peak with parabolic interpolation on
        the neighboring bins, giving sub-bin resolution.

    Returns
    -------
    yaw_rad : float
        Estimated relative yaw in radians, in (-pi, pi].
    confidence : float
        Peak-to-sidelobe ratio (higher is better).
    ambiguity : float
        Second-peak / first-peak ratio (lower is better).
    """
    n = len(hist_a)
    corr = _circular_cross_correlation(hist_a, hist_b)

    # --- Find the peak ---
    peak_idx = int(np.argmax(corr))
    peak_val = float(corr[peak_idx])

    # --- Sub-bin refinement via parabolic fit on neighbors ---
    if refine_peak and n >= 3:
        left  = corr[(peak_idx - 1) % n]
        right = corr[(peak_idx + 1) % n]
        denom = 2.0 * peak_val - left - right
        if abs(denom) > 1e-12:
            delta = 0.5 * (right - left) / denom
            # Clip to avoid runaway
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
        peak_pos = (peak_idx + delta) % n
    else:
        peak_pos = float(peak_idx)

    # --- Convert bin position to angle ---
    # corr[k] corresponds to shifting b by k bins
    # Each bin is 2pi / n radians
    yaw_rad = 2.0 * np.pi * peak_pos / n
    # Wrap to (-pi, pi]
    yaw_rad = ((yaw_rad + np.pi) % (2 * np.pi)) - np.pi

    # --- Confidence via peak-to-sidelobe ratio ---
    # Zero out the peak (and a small neighborhood) to define the "sidelobes"
    n_exclude = max(2, n // 20)   # exclude 5% of bins around the peak
    mask = np.ones(n, dtype=bool)
    for d in range(-n_exclude, n_exclude + 1):
        mask[(peak_idx + d) % n] = False
    sidelobes = corr[mask]
    sidelobe_mean = float(sidelobes.mean())
    sidelobe_std  = float(sidelobes.std())
    confidence = (peak_val - sidelobe_mean) / max(sidelobe_std, 1e-12)

    # --- Ambiguity: second highest peak outside the primary neighborhood ---
    corr_copy = corr.copy()
    for d in range(-n_exclude, n_exclude + 1):
        corr_copy[(peak_idx + d) % n] = 0.0
    second_peak_val = float(corr_copy.max())
    ambiguity = second_peak_val / max(peak_val, 1e-12)

    return yaw_rad, confidence, ambiguity


# ============================================================
# 3. Optional ICP refinement (multi-start 2D)
# ============================================================
def _rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def _icp_2d_yaw(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    initial_yaw: float,
    max_iter: int = 30,
    tol: float = 1e-5,
    max_correspondence_dist: float = 1.0,
) -> Tuple[float, float]:
    """
    Run a simple point-to-point ICP restricted to yaw rotation (Z).

    Parameters
    ----------
    src_xy : (M, 2)
        Source points, XY only, centered.
    dst_xy : (N, 2)
        Target points, XY only, centered.
    initial_yaw : float
        Initial rotation applied to src before matching.
    max_iter : int
    tol : float
        Convergence tolerance on the yaw update.
    max_correspondence_dist : float
        Reject correspondences beyond this distance.

    Returns
    -------
    yaw : float
        Refined yaw.
    rmse : float
        Final RMSE of the inlier correspondences.
    """
    from scipy.spatial import cKDTree

    yaw = initial_yaw
    tree = cKDTree(dst_xy)
    prev_rmse = np.inf

    for _ in range(max_iter):
        # Rotate source by current yaw
        R = _rotation_matrix_2d(yaw)
        src_rot = src_xy @ R.T

        # Find nearest neighbor in target
        dists, idxs = tree.query(src_rot, k=1)
        mask = dists < max_correspondence_dist
        if mask.sum() < 10:
            # Too few correspondences; break
            break

        src_matched = src_rot[mask]
        dst_matched = dst_xy[idxs[mask]]

        # Compute optimal yaw delta from Procrustes
        # We want the rotation that aligns src_matched to dst_matched
        # Optimal yaw update via 2D Procrustes
        H = src_matched.T @ dst_matched
        # Correct sign: rotate src by +delta to align it with dst
        delta_yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        yaw = yaw + delta_yaw

        # RMSE
        R = _rotation_matrix_2d(yaw)
        src_rot = src_xy @ R.T
        dists, _ = tree.query(src_rot, k=1)
        rmse = float(np.sqrt(np.mean(dists[dists < max_correspondence_dist] ** 2)))

        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    return yaw, prev_rmse


def refine_with_icp(
    pc_a,
    pc_b,
    initial_yaw: float,
    n_starts: int = 6,
    span_deg: float = 30.0,
    max_correspondence_dist: float = 1.0,
) -> Tuple[float, float]:
    """
    Refine the yaw estimate with multi-start 2D ICP.

    Parameters
    ----------
    pc_a, pc_b : PointCloud or (N,3) arrays
    initial_yaw : float
        Initial estimate from the histogram method.
    n_starts : int
        Number of ICP starts, evenly spaced within +-span_deg of initial_yaw.
    span_deg : float
        Angular span around the initial yaw for the multi-start.
    max_correspondence_dist : float
        ICP correspondence threshold in meters.

    Returns
    -------
    yaw : float
        Refined yaw.
    rmse : float
        Final RMSE (lower = better).
    """
    pts_a = pc_a.points if hasattr(pc_a, "points") else np.asarray(pc_a)
    pts_b = pc_b.points if hasattr(pc_b, "points") else np.asarray(pc_b)

    # Center both clouds on their respective centroids for ICP
    a_xy = pts_a[:, :2] - pts_a[:, :2].mean(axis=0)
    b_xy = pts_b[:, :2] - pts_b[:, :2].mean(axis=0)

    # Multi-start
    span_rad = np.deg2rad(span_deg)
    starts = np.linspace(initial_yaw - span_rad,
                         initial_yaw + span_rad,
                         n_starts)

    best_yaw, best_rmse = initial_yaw, np.inf
    for s in starts:
        yaw, rmse = _icp_2d_yaw(
            a_xy, b_xy, s,
            max_correspondence_dist=max_correspondence_dist,
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_yaw = yaw

    # Normalize to (-pi, pi]
    best_yaw = ((best_yaw + np.pi) % (2 * np.pi)) - np.pi
    return best_yaw, best_rmse


# ============================================================
# 4. High-level API
# ============================================================
def estimate_yaw(
    pc_a,
    pc_b,
    n_bins: int = 360,
    use_centroid: bool = True,
    refine_icp: bool = False,
    icp_n_starts: int = 6,
    icp_span_deg: float = 30.0,
    icp_max_dist: float = 1.0,
    method: str = "stratified",     # "simple" or "stratified"
    n_rings: int = 3,
) -> YawEstimate:
    """
    Estimate the relative yaw of pc_b with respect to pc_a.

    Parameters
    ----------
    method : {"simple", "stratified"}
        - "simple": single angular histogram (original behavior).
        - "stratified": angular histogram split into radial rings;
          disambiguates symmetric scenes at low cost.
    n_rings : int
        Number of radial rings when method="stratified". 3 is a good
        default. Use 2 for speed, 4-5 for more discrimination.
    """
    if method == "stratified":
        hist_a = angular_histogram_stratified(
            pc_a, n_bins=n_bins, n_rings=n_rings,
            use_centroid=use_centroid,
        )
        hist_b = angular_histogram_stratified(
            pc_b, n_bins=n_bins, n_rings=n_rings,
            use_centroid=use_centroid,
        )
        yaw_rad, confidence, ambiguity = \
            estimate_yaw_from_stratified_histograms(hist_a, hist_b, refine_peak=False)
        method_str = "stratified"
    else:
        hist_a = angular_histogram(pc_a, n_bins=n_bins,
                                   use_centroid=use_centroid)
        hist_b = angular_histogram(pc_b, n_bins=n_bins,
                                   use_centroid=use_centroid)
        yaw_rad, confidence, ambiguity = \
            estimate_yaw_from_histograms(hist_a, hist_b)
        method_str = "histogram"

    if refine_icp:
        yaw_rad, _ = refine_with_icp(
            pc_a, pc_b, initial_yaw=yaw_rad,
            n_starts=icp_n_starts, span_deg=icp_span_deg,
            max_correspondence_dist=icp_max_dist,
        )
        method_str = method_str + "+icp"

    return YawEstimate(
        yaw_rad=yaw_rad,
        yaw_deg=float(np.rad2deg(yaw_rad)),
        confidence=confidence,
        ambiguity=ambiguity,
        method=method_str,
    )


# ============================================================
# 5. Utility for visual debugging
# ============================================================
def plot_angular_histograms(
    pc_a,
    pc_b,
    n_bins: int = 360,
    use_centroid: bool = True,
    ax=None,
):
    """Plot the two angular histograms overlaid for visual inspection."""
    import matplotlib.pyplot as plt

    ha = angular_histogram(pc_a, n_bins=n_bins, use_centroid=use_centroid)
    hb = angular_histogram(pc_b, n_bins=n_bins, use_centroid=use_centroid)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    theta = np.linspace(-180, 180, n_bins)
    ax.plot(theta, ha, label="A", alpha=0.7)
    ax.plot(theta, hb, label="B", alpha=0.7)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("density")
    ax.set_title("Angular histograms")
    ax.legend()
    return ax

def angular_histogram_stratified(
    pc,
    n_bins: int = 360,
    n_rings: int = 3,
    use_centroid: bool = True,
    ring_boundaries: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute an angular histogram stratified by radial distance.

    Divides points into `n_rings` concentric rings around the anchor (XY
    centroid or sensor origin), then computes a separate angular histogram
    for each ring. Returns a 2D array of shape (n_rings, n_bins).

    This preserves information about the joint (theta, radius) distribution
    and helps disambiguate scenes with bilateral symmetry.

    Parameters
    ----------
    pc : PointCloud or (N,3) array
    n_bins : int
        Number of angular bins (per ring).
    n_rings : int
        Number of radial rings. 2-4 is usually enough.
    use_centroid : bool
        Anchor for angles and radii.
    ring_boundaries : optional array of length (n_rings+1)
        Custom ring boundaries in the same units as `r`. If None, uses
        quantile-based boundaries so that each ring has roughly the same
        number of points.

    Returns
    -------
    hist : np.ndarray, shape (n_rings, n_bins)
        Each row sums to the fraction of points in that ring; all entries
        sum to 1.
    """
    points = pc.points if hasattr(pc, "points") else np.asarray(pc)
    xy = points[:, :2]

    if use_centroid:
        anchor = xy.mean(axis=0)
    else:
        anchor = np.array([0.0, 0.0])

    rel = xy - anchor
    theta = np.arctan2(rel[:, 1], rel[:, 0])       # (-pi, pi]
    r = np.linalg.norm(rel, axis=1)                # radial distance

    # --- Determine ring boundaries ---
    if ring_boundaries is None:
        # Quantile-based: each ring has ~ equal number of points
        qs = np.linspace(0, 100, n_rings + 1)
        ring_boundaries = np.percentile(r, qs)
    ring_boundaries = np.asarray(ring_boundaries)
    if len(ring_boundaries) != n_rings + 1:
        raise ValueError("ring_boundaries must have n_rings+1 entries")

    # --- Compute per-ring angular histograms ---
    hist = np.zeros((n_rings, n_bins), dtype=np.float64)
    total = len(r)
    if total == 0:
        return hist

    for k in range(n_rings):
        lo, hi = ring_boundaries[k], ring_boundaries[k + 1]
        if k == n_rings - 1:
            # Include the last boundary to avoid excluding r == max
            mask = (r >= lo) & (r <= hi)
        else:
            mask = (r >= lo) & (r < hi)
        if mask.sum() == 0:
            continue
        h, _ = np.histogram(theta[mask], bins=n_bins,
                            range=(-np.pi, np.pi))
        hist[k] = h.astype(np.float64) / total

    return hist

def estimate_yaw_from_stratified_histograms(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    ring_weights: Optional[np.ndarray] = None,
    refine_peak: bool = True,
) -> Tuple[float, float, float]:
    """
    Estimate yaw from stratified angular histograms.

    Combines the correlations of all rings into a single score per
    candidate yaw, then picks the peak.

    Parameters
    ----------
    hist_a, hist_b : np.ndarray, shape (n_rings, n_bins)
    ring_weights : optional array of length n_rings
        Weight per ring when combining correlations. If None, weights are
        chosen automatically: outer rings get more weight (more spatial
        diversity, less prone to being dominated by near-field clutter).
    refine_peak : bool
        Sub-bin refinement of the peak.

    Returns
    -------
    yaw_rad, confidence, ambiguity : same as estimate_yaw_from_histograms
    """
    n_rings, n_bins = hist_a.shape
    if hist_b.shape != hist_a.shape:
        raise ValueError("Histograms must have the same shape")

    # --- Default weights: emphasize outer rings ---
    if ring_weights is None:
        ring_weights = np.linspace(1.0, 2.0, n_rings)
        ring_weights = ring_weights / ring_weights.sum()

    # --- Sum weighted correlations ---
    combined = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_rings):
        c = _circular_cross_correlation(hist_a[k], hist_b[k])
        combined += ring_weights[k] * c

    # --- Find peak ---
    peak_idx = int(np.argmax(combined))
    peak_val = float(combined[peak_idx])

    if refine_peak and n_bins >= 3:
        left  = combined[(peak_idx - 1) % n_bins]
        right = combined[(peak_idx + 1) % n_bins]
        denom = 2.0 * peak_val - left - right
        delta = 0.5 * (right - left) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        peak_pos = (peak_idx + delta) % n_bins
    else:
        peak_pos = float(peak_idx)

    yaw_rad = 2.0 * np.pi * peak_pos / n_bins
    yaw_rad = ((yaw_rad + np.pi) % (2 * np.pi)) - np.pi

    # --- Confidence (peak-to-sidelobe) ---
    n_exclude = max(2, n_bins // 20)
    mask = np.ones(n_bins, dtype=bool)
    for d in range(-n_exclude, n_exclude + 1):
        mask[(peak_idx + d) % n_bins] = False
    sidelobes = combined[mask]
    sidelobe_mean = float(sidelobes.mean())
    sidelobe_std  = float(sidelobes.std())
    confidence = (peak_val - sidelobe_mean) / max(sidelobe_std, 1e-12)

    # --- Ambiguity (second peak) ---
    combined_copy = combined.copy()
    for d in range(-n_exclude, n_exclude + 1):
        combined_copy[(peak_idx + d) % n_bins] = 0.0
    second_val = float(combined_copy.max())
    ambiguity = second_val / max(peak_val, 1e-12)

    return yaw_rad, confidence, ambiguity


# ============================================================
# 5b. Odometry disambiguation
# ============================================================
def _wrap_pi(angle_rad: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return ((angle_rad + np.pi) % (2 * np.pi)) - np.pi


def resolve_yaw_ambiguity_with_odometry(
    yaw_rad: float,
    yaw_a_odom_rad: float,
    yaw_b_odom_rad: float,
    period_rad: float = np.pi,
) -> Tuple[float, float]:
    """
    Disambiguate a yaw estimate using an odometric prediction.

    An angular histogram (like PCA) is ambiguous: for a scene with
    bilateral symmetry the histogram and its 180-degree rotation give
    equally strong correlation peaks. Odometry drifts, but between two
    passes through the same place the drift is usually far smaller than
    the ambiguity period, so it is enough to pick the candidate closest
    to the odometric prediction.

    Parameters
    ----------
    yaw_rad : float
        Raw estimate from the histogram method, in (-pi, pi].
    yaw_a_odom_rad, yaw_b_odom_rad : float
        Odometric yaw of the robot at the two captures. The odometric
        prediction is ``wrap_pi(yaw_b - yaw_a)``.
    period_rad : float
        Ambiguity period. ``pi`` for the 180-degree flip (default).

    Returns
    -------
    yaw_rad : float
        Disambiguated yaw, in (-pi, pi].
    yaw_rel_odom_rad : float
        Odometric relative-yaw prediction, for reference.
    """
    yaw_rel_odom = _wrap_pi(yaw_b_odom_rad - yaw_a_odom_rad)

    best = _wrap_pi(yaw_rad)
    best_dist = abs(_wrap_pi(best - yaw_rel_odom))

    # Number of alternative candidates that fit in a full turn.
    n_alt = max(int(round(2 * np.pi / period_rad)) - 1, 1)
    for k in range(1, n_alt + 1):
        cand = _wrap_pi(yaw_rad + k * period_rad)
        dist = abs(_wrap_pi(cand - yaw_rel_odom))
        if dist < best_dist:
            best, best_dist = cand, dist

    return best, yaw_rel_odom


def estimate_yaw_from_histograms_with_odometry(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    yaw_a_odom_rad: float,
    yaw_b_odom_rad: float,
    refine_peak: bool = True,
) -> Tuple[float, float, float]:
    """
    Simple-histogram yaw estimate with the 180-degree ambiguity resolved
    by odometry.

    Returns
    -------
    yaw_rad, confidence, ambiguity : same as estimate_yaw_from_histograms,
        but ``yaw_rad`` is the candidate closest to the odometric
        prediction.
    """
    yaw_rad, confidence, ambiguity = estimate_yaw_from_histograms(
        hist_a, hist_b, refine_peak=refine_peak)
    yaw_rad, _ = resolve_yaw_ambiguity_with_odometry(
        yaw_rad, yaw_a_odom_rad, yaw_b_odom_rad)
    return yaw_rad, confidence, ambiguity


def estimate_yaw_from_stratified_histograms_with_odometry(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    yaw_a_odom_rad: float,
    yaw_b_odom_rad: float,
    ring_weights: Optional[np.ndarray] = None,
    refine_peak: bool = True,
) -> Tuple[float, float, float]:
    """
    Stratified-histogram yaw estimate with the 180-degree ambiguity
    resolved by odometry.

    Returns
    -------
    yaw_rad, confidence, ambiguity : same as
        estimate_yaw_from_stratified_histograms, but ``yaw_rad`` is the
        candidate closest to the odometric prediction.
    """
    yaw_rad, confidence, ambiguity = estimate_yaw_from_stratified_histograms(
        hist_a, hist_b, ring_weights=ring_weights, refine_peak=refine_peak)
    yaw_rad, _ = resolve_yaw_ambiguity_with_odometry(
        yaw_rad, yaw_a_odom_rad, yaw_b_odom_rad)
    return yaw_rad, confidence, ambiguity


# ============================================================
# 6. Rich visualization
# ============================================================
def plot_yaw_diagnostics(
    pc_a,
    pc_b,
    n_bins: int = 360,
    method: str = "stratified",
    n_rings: int = 10,
    use_centroid: bool = True,
    estimated_yaw_deg: Optional[float] = None,
    gt_yaw_deg: Optional[float] = None,
    title: str = "",
):
    """
    Plot the histograms and correlation used to compute the yaw estimate.

    Layout:
        Row 1: aggregated angular histogram of A vs B (overlaid)
        Row 2: circular cross-correlation with peak marked
        Row 3 (if stratified): per-ring histograms (A in blue, B in orange)
                               for a subset of rings (max 4 shown)

    Parameters
    ----------
    estimated_yaw_deg : optional
        If given, drawn as a marker on the correlation plot.
    gt_yaw_deg : optional
        Ground-truth yaw (drawn as a vertical line on the correlation plot
        for comparison).
    title : str
        Overall title.
    """
    import matplotlib.pyplot as plt

    # ---- Compute histograms ----
    if method == "stratified":
        hist_a = angular_histogram_stratified(
            pc_a, n_bins=n_bins, n_rings=n_rings, use_centroid=use_centroid
        )
        hist_b = angular_histogram_stratified(
            pc_b, n_bins=n_bins, n_rings=n_rings, use_centroid=use_centroid
        )
        # Aggregate across rings for display
        agg_a = hist_a.sum(axis=0)
        agg_b = hist_b.sum(axis=0)
        # Combined correlation
        ring_weights = np.linspace(1.0, 2.0, n_rings)
        ring_weights = ring_weights / ring_weights.sum()
        combined = np.zeros(n_bins)
        for k in range(n_rings):
            combined += ring_weights[k] * _circular_cross_correlation(
                hist_a[k], hist_b[k]
            )
    else:
        agg_a = angular_histogram(pc_a, n_bins=n_bins,
                                  use_centroid=use_centroid)
        agg_b = angular_histogram(pc_b, n_bins=n_bins,
                                  use_centroid=use_centroid)
        combined = _circular_cross_correlation(agg_a, agg_b)
        hist_a = None
        hist_b = None

    # ---- Determine how many rows we need ----
    n_rows = 2 if method == "simple" else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.2 * n_rows))

    theta_deg = np.linspace(-180, 180, n_bins, endpoint=False)

    # ---- Row 1: aggregated angular histograms ----
    ax = axes[0]
    ax.plot(theta_deg, agg_a, label="A (stored)", color="tab:blue",
            linewidth=1.2)
    ax.plot(theta_deg, agg_b, label="B (query)", color="tab:orange",
            linewidth=1.2, alpha=0.8)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("density")
    ax.set_title("Aggregated angular histograms")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # ---- Row 2: correlation ----
    ax = axes[1]

    # Build the angle axis matching the FFT bin indexing:
    # bin k corresponds to angle 2*pi*k/n_bins, wrapped to (-180, 180].
    bin_angles_rad = 2 * np.pi * np.arange(n_bins) / n_bins
    bin_angles_rad = ((bin_angles_rad + np.pi) % (2 * np.pi)) - np.pi
    bin_angles_deg = np.rad2deg(bin_angles_rad)

    # For display, sort by angle so the x-axis increases monotonically:
    order = np.argsort(bin_angles_deg)
    x_plot = bin_angles_deg[order]
    y_plot = combined[order]

    ax.plot(x_plot, y_plot, color="tab:green", linewidth=1.2)

    # Peak position — use the same angle convention as the computation:
    peak_idx = int(np.argmax(combined))
    peak_angle_deg = bin_angles_deg[peak_idx]
    ax.axvline(peak_angle_deg, color="red", linestyle="--",
            linewidth=1.5, label=f"peak @ {peak_angle_deg:+.1f}°")

    # Estimate (already in degrees, wrapped)
    if estimated_yaw_deg is not None:
        est = ((estimated_yaw_deg + 180) % 360) - 180
        ax.axvline(est, color="purple", linestyle=":",
                linewidth=1.5,
                label=f"estimate = {estimated_yaw_deg:+.2f}°")

    # Ground truth
    if gt_yaw_deg is not None:
        gt = ((gt_yaw_deg + 180) % 360) - 180
        ax.axvline(gt, color="black", linestyle="-.",
                linewidth=1.5,
                label=f"ground truth = {gt_yaw_deg:+.2f}°")

    ax.set_xlabel("shift (deg)")
    ax.set_ylabel("correlation")
    ax.set_title("Circular cross-correlation")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # ---- Row 3 (stratified only): per-ring histograms ----
    if method == "stratified" and hist_a is not None:
        ax = axes[2]
        # Show at most 4 rings, evenly spaced
        n_show = min(4, n_rings)
        ring_indices = np.linspace(0, n_rings - 1, n_show).astype(int)
        colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, n_show))
        colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, n_show))
        for i, k in enumerate(ring_indices):
            ax.plot(theta_deg, hist_a[k], color=colors_a[i],
                    linewidth=1.0,
                    label=f"A ring {k}" if i == 0 else None,
                    alpha=0.9)
            ax.plot(theta_deg, hist_b[k], color=colors_b[i],
                    linewidth=1.0, linestyle="--",
                    label=f"B ring {k}" if i == 0 else None,
                    alpha=0.9)
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("density")
        ax.set_title(f"Per-ring histograms (showing "
                     f"{n_show} of {n_rings} rings)")
        ax.legend(loc="upper right", ncol=2)
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig