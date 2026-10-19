# yaw_estimator_pca.py
"""
Yaw estimation between two point clouds via 2D PCA.

Method
------
1. Project both clouds onto XY (ignore Z).
2. Center each cloud on its XY centroid.
3. Compute the 2x2 covariance matrix.
4. Extract the principal axis angle via the closed-form formula:
       phi = 0.5 * atan2(2*sxy, sxx - syy)
5. Relative yaw = phi_B - phi_A, wrapped to (-pi, pi].

Notes
-----
- The PCA axis is a LINE, not a direction. The estimate is therefore
  ambiguous modulo 180 degrees. This is intrinsic to the method.
- PCA is sensitive to outliers. Use `max_radius` to clip points that are
  too far from the centroid.
- Useful when the two clouds are truly from the same place and the
  dominant structure (e.g., a wall, a corridor) is well-defined.
- Use `estimate_yaw_pca_with_odometry` to resolve the 180-degree
  ambiguity using an approximate heading from odometry.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ============================================================
# Result container
# ============================================================
@dataclass
class YawEstimatePCA:
    """Result of a PCA-based yaw estimation."""
    yaw_rad: float              # relative yaw (rad), wrapped to (-pi, pi]
    yaw_deg: float
    phi_a_rad: float            # principal axis angle of cloud A
    phi_b_rad: float            # principal axis angle of cloud B
    lambda_1: float             # largest eigenvalue (along principal axis)
    lambda_2: float             # smallest eigenvalue (perpendicular)
    anisotropy: float           # lambda_1 / lambda_2 (higher = more "line-like")
    method: str = "pca2d"

    def __repr__(self):
        return (f"YawEstimatePCA(yaw={self.yaw_deg:+.2f}°, "
                f"anisotropy={self.anisotropy:.2f}, "
                f"phi_a={np.rad2deg(self.phi_a_rad):+.2f}°, "
                f"phi_b={np.rad2deg(self.phi_b_rad):+.2f}°)")


# ============================================================
# Angle helpers
# ============================================================
def _wrap_pi(angle_rad: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return ((angle_rad + np.pi) % (2 * np.pi)) - np.pi


def _wrap_pi_over_2(angle_rad: float) -> float:
    """Wrap an angle to (-pi/2, pi/2]. Useful for axis angles."""
    return ((angle_rad + np.pi / 2) % np.pi) - np.pi / 2


# ============================================================
# Core PCA helpers
# ============================================================
def _project_and_filter(
    pc,
    max_radius: Optional[float] = None,
    use_centroid: bool = True,
) -> np.ndarray:
    """
    Project a point cloud onto XY and (optionally) filter far points.

    Parameters
    ----------
    pc : PointCloud or (N,3) array
    max_radius : float or None
        If given, discard points whose XY distance from the anchor
        exceeds this value. Reduces the influence of outliers.
    use_centroid : bool
        If True, subtract the XY centroid (recommended).
        If False, keep the original XY (useful if the clouds are
        already aligned at the sensor origin).

    Returns
    -------
    xy : (M, 2) array of centered (or not) XY points
    """
    points = pc.points if hasattr(pc, "points") else np.asarray(pc)
    xy = points[:, :2].astype(np.float64)

    if use_centroid:
        xy = xy - xy.mean(axis=0)

    if max_radius is not None:
        r = np.linalg.norm(xy, axis=1)
        mask = r <= max_radius
        xy = xy[mask]

    return xy


def _pca_2d(xy: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the 2D PCA of a set of XY points (assumed already centered).

    Returns
    -------
    phi : float
        Principal axis angle (rad), in (-pi/2, pi/2].
    eigvals : (2,) array
        [lambda_1, lambda_2], descending.
    eigvecs : (2, 2) array
        Columns are the eigenvectors corresponding to eigvals.
    """
    n = len(xy)
    if n < 3:
        raise ValueError("Need at least 3 points for PCA")

    cov = (xy.T @ xy) / n
    sxx = cov[0, 0]
    syy = cov[1, 1]
    sxy = cov[0, 1]

    phi = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
    phi = _wrap_pi_over_2(phi)

    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = np.sqrt(max(trace * trace / 4.0 - det, 0.0))
    lam1 = trace / 2.0 + disc
    lam2 = trace / 2.0 - disc

    v1 = np.array([np.cos(phi), np.sin(phi)])
    v2 = np.array([-np.sin(phi), np.cos(phi)])
    eigvecs = np.column_stack([v1, v2])
    eigvals = np.array([lam1, lam2])

    return phi, eigvals, eigvecs


# ============================================================
# High-level API: plain PCA
# ============================================================
def estimate_yaw_pca(
    pc_a,
    pc_b,
    max_radius: Optional[float] = None,
    use_centroid: bool = True,
    resolve_180_with_z: bool = False,
) -> YawEstimatePCA:
    """
    Estimate the relative yaw of pc_b with respect to pc_a via 2D PCA.

    Parameters
    ----------
    pc_a, pc_b : PointCloud or (N,3) arrays
    max_radius : float or None
        If given, discard points whose XY distance from the centroid
        exceeds this value. Helps reduce outlier influence.
    use_centroid : bool
        If True (recommended), center each cloud on its XY centroid
        before computing PCA. If False, use the original XY coordinates
        (useful if the clouds already share the sensor origin and you
        want to preserve translation information).
    resolve_180_with_z : bool
        If True, attempt to disambiguate the 180-degree ambiguity by
        comparing the vertical (Z) asymmetry of the two clouds.

    Returns
    -------
    YawEstimatePCA
    """
    xy_a = _project_and_filter(pc_a, max_radius=max_radius,
                               use_centroid=use_centroid)
    xy_b = _project_and_filter(pc_b, max_radius=max_radius,
                               use_centroid=use_centroid)

    phi_a, eigvals_a, _ = _pca_2d(xy_a)
    phi_b, eigvals_b, _ = _pca_2d(xy_b)

    aniso_a = eigvals_a[0] / max(eigvals_a[1], 1e-12)
    aniso_b = eigvals_b[0] / max(eigvals_b[1], 1e-12)
    anisotropy = min(aniso_a, aniso_b)

    yaw_rad = _wrap_pi(phi_b - phi_a)

    if resolve_180_with_z:
        sign = _z_asymmetry_sign(pc_a, pc_b, phi_a, phi_b)
        if sign < 0:
            yaw_rad = _wrap_pi(yaw_rad + np.pi)

    return YawEstimatePCA(
        yaw_rad=yaw_rad,
        yaw_deg=float(np.rad2deg(yaw_rad)),
        phi_a_rad=float(phi_a),
        phi_b_rad=float(phi_b),
        lambda_1=float(eigvals_a[0]),
        lambda_2=float(eigvals_a[1]),
        anisotropy=float(anisotropy),
    )


def _z_asymmetry_sign(
    pc_a,
    pc_b,
    phi_a: float,
    phi_b: float,
) -> float:
    """
    Heuristic to disambiguate the 180-degree ambiguity using Z asymmetry.
    Returns +1 if the current yaw sign is correct, -1 if a flip is needed.
    """
    pts_a = pc_a.points if hasattr(pc_a, "points") else np.asarray(pc_a)
    pts_b = pc_b.points if hasattr(pc_b, "points") else np.asarray(pc_b)

    perp_a = np.array([-np.sin(phi_a), np.cos(phi_a)])
    xy_a = pts_a[:, :2] - pts_a[:, :2].mean(axis=0)
    z_a = pts_a[:, 2] - pts_a[:, 2].mean()
    proj_a = xy_a @ perp_a
    sign_a = np.sign(np.dot(proj_a, z_a))

    perp_b = np.array([-np.sin(phi_b), np.cos(phi_b)])
    xy_b = pts_b[:, :2] - pts_b[:, :2].mean(axis=0)
    z_b = pts_b[:, 2] - pts_b[:, 2].mean()
    proj_b = xy_b @ perp_b
    sign_b = np.sign(np.dot(proj_b, z_b))

    return +1.0 if sign_a * sign_b >= 0 else -1.0


# ============================================================
# High-level API: PCA + odometry disambiguation
# ============================================================
def estimate_yaw_pca_with_odometry(
    pc_a,
    pc_b,
    yaw_a_odom_rad: float,
    yaw_b_odom_rad: float,
    max_radius: Optional[float] = None,
    use_centroid: bool = True,
) -> YawEstimatePCA:
    """
    Estimate relative yaw via 2D PCA, disambiguating the 180-degree
    ambiguity using odometry.

    Parameters
    ----------
    pc_a, pc_b : PointCloud or (N,3) arrays
        Stored and query clouds.
    yaw_a_odom_rad : float
        Odometric yaw of the robot when pc_a was captured.
    yaw_b_odom_rad : float
        Odometric yaw of the robot when pc_b was captured.
    max_radius, use_centroid : passed to the PCA core.

    Returns
    -------
    YawEstimatePCA
        Same as estimate_yaw_pca, but the 180-degree ambiguity has been
        resolved using odometry. `method` is "pca2d+odom".

    Notes
    -----
    Odometry drifts, but the drift between two passes through the same
    place is typically much smaller than 180 degrees. So even a rough
    heading estimate is enough to pick between phi and phi+180.
    """
    pca_res = estimate_yaw_pca(
        pc_a, pc_b,
        max_radius=max_radius,
        use_centroid=use_centroid,
        resolve_180_with_z=False,   # we use odometry instead
    )

    # Odometric prediction of the relative yaw
    yaw_rel_odom = _wrap_pi(yaw_b_odom_rad - yaw_a_odom_rad)

    # Two candidates: phi and phi + 180
    yaw_c1 = _wrap_pi(pca_res.yaw_rad)
    yaw_c2 = _wrap_pi(pca_res.yaw_rad + np.pi)

    # Angular distance to the odometric prediction
    dist_1 = abs(_wrap_pi(yaw_c1 - yaw_rel_odom))
    dist_2 = abs(_wrap_pi(yaw_c2 - yaw_rel_odom))

    chosen_yaw = yaw_c1 if dist_1 <= dist_2 else yaw_c2

    return YawEstimatePCA(
        yaw_rad=chosen_yaw,
        yaw_deg=float(np.rad2deg(chosen_yaw)),
        phi_a_rad=pca_res.phi_a_rad,
        phi_b_rad=pca_res.phi_b_rad,
        lambda_1=pca_res.lambda_1,
        lambda_2=pca_res.lambda_2,
        anisotropy=pca_res.anisotropy,
        method="pca2d+odom",
    )