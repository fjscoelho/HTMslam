# place_encoder.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from htm.bindings.encoders import RDSE, RDSE_Parameters


# ============================================================
# PointCloud
# ============================================================
@dataclass
class PointCloud:
    """
    Encapsulates a (N, 3) point cloud with basic geometric operations.
    Assumes Z is the vertical axis (yaw = rotation around Z).
    Points are expected to be in the SENSOR frame: the origin (0,0,0)
    coincides with the sensor's optical/acoustic center.
    """
    points: np.ndarray  # (N, 3)

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError(f"points must be (N,3), got {self.points.shape}")
        if len(self.points) == 0:
            raise ValueError("points is empty")
        self.points = self.points[:, :3].astype(np.float64)

    # ---------- Properties ----------
    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def centroid(self) -> np.ndarray:
        """3D centroid (geometric, not used as anchor)."""
        return self.points.mean(axis=0)

    @property
    def centroid_xy(self) -> np.ndarray:
        """Centroid projected onto XY plane."""
        return self.centroid[:2]

    @property
    def xy(self) -> np.ndarray:
        """XY coordinates in the SENSOR frame."""
        return self.points[:, :2]

    @property
    def z(self) -> np.ndarray:
        """Z coordinate in the SENSOR frame."""
        return self.points[:, 2]

    @property
    def r_xy(self) -> np.ndarray:
        """Radial distance in XY relative to the SENSOR origin."""
        return np.linalg.norm(self.points[:, :2], axis=1)

    # ---------- Constructors ----------
    @classmethod
    def from_npy(cls, path: str | Path) -> "PointCloud":
        """Load from a .npy file."""
        pc = np.load(str(path))
        return cls(points=pc)

    # ---------- Operations ----------
    def rotated_yaw(self, angle_rad: float) -> "PointCloud":
        """Return a new PointCloud rotated around the SENSOR origin (Z axis)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])
        return PointCloud(points=self.points @ R.T)

    # ---------- Visualization ----------
    def plot_top_view(self, ax=None, show_centroid: bool = False):
        """Plot top view (XY) with centroid highlighted."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.points[:, 0], self.points[:, 1],
                   s=1, alpha=0.4, label="points")

        if show_centroid:
            cx, cy = self.centroid_xy
            ax.scatter([cx], [cy], s=75, c="red", marker="X",
                       edgecolors="black", linewidths=1.0, zorder=5,
                       label=f"centroid ({cx:.2f}, {cy:.2f})")

        # Mark the sensor origin
        ax.scatter([0], [0], s=75, c="lime", marker="o",
                   edgecolors="black", linewidths=1.0, zorder=6,
                   label="sensor origin")

        ax.set_aspect("equal")
        ax.set_title(f"Top view (N={self.n_points})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=8)
        return ax


# ============================================================
# PlaceDescriptor
# ============================================================
@dataclass
class PlaceDescriptor:
    """
    Yaw-invariant descriptors anchored at the SENSOR ORIGIN (0, 0, 0).

    Rationale
    ---------
    The centroid of a point cloud shifts with which points were observed,
    so two captures from the same physical pose can have different centroids.
    The sensor origin, in contrast, is fixed: captures from the same pose
    produce identical anchors, which stabilizes the features.

    All features below are invariant to yaw rotation around Z, while still
    preserving information about the robot's vertical pose (mean_height)
    and sensor-axis alignment.
    """
    eigvals: np.ndarray      # (3,) eigenvalues of 3D covariance (descending)
    hist_z: np.ndarray       # (n_bins_z,) normalized z histogram (sensor frame)
    hist_r: np.ndarray       # (n_bins_r,) normalized radial histogram (sensor frame)
    height: float            # z_max - z_min (sensor frame)
    density: float           # log1p(N)
    volume: float            # volume of the covariance ellipsoid
    mean_radius: float       # mean radial distance to sensor
    std_radius: float        # std of radial distance to sensor
    mean_height: float       # mean z in sensor frame
    std_height: float        # std of z in sensor frame

    # ---------- Main constructor ----------
    @classmethod
    def from_pointcloud(
        cls,
        pc: PointCloud,
        n_bins_z: int = 10,
        n_bins_r: int = 10,
        z_range: Optional[Tuple[float, float]] = None,
        r_range: Optional[Tuple[float, float]] = None,
    ) -> "PlaceDescriptor":
        """
        Compute yaw-invariant descriptors anchored at the sensor origin.

        Parameters
        ----------
        pc : PointCloud
            Points in the sensor frame (origin = sensor).
        n_bins_z, n_bins_r : int
            Number of histogram bins.
        z_range, r_range : optional (min, max)
            Fixed ranges for histograms (recommended for cross-cloud
            comparison). If None, uses the per-cloud min/max.
        """
        N = pc.n_points
        if N < 100:
            raise ValueError(f"Too few points: {N}")

        pts = pc.points              # (N, 3) in sensor frame
        z = pts[:, 2]
        xy = pts[:, :2]
        r = np.linalg.norm(xy, axis=1)

        # --- 3D covariance eigenvalues (rotation-invariant) ---
        cov = np.cov(pts.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]  # λ1 ≥ λ2 ≥ λ3
        eigvals = np.maximum(eigvals, 0.0)                # guard: non-negative

        # --- Histograms (sensor-anchored) ---
        z_range = cls._fix_range(z_range, z,
                                 default_min=z.min(), default_max=z.max())
        r_range = cls._fix_range(r_range, r,
                                 default_min=0.0, default_max=r.max())

        hist_z, _ = np.histogram(z, bins=n_bins_z, range=z_range)
        hist_r, _ = np.histogram(r, bins=n_bins_r, range=r_range)
        hist_z = hist_z.astype(np.float64) / N
        hist_r = hist_r.astype(np.float64) / N

        # --- Scalars (all in sensor frame) ---
        bbox = pts.max(axis=0) - pts.min(axis=0)
        height = float(bbox[2])
        density = float(np.log1p(N))
        volume_elipsoide = float((4/3) * np.pi * np.sqrt(np.prod(eigvals) + 1e-12))
        mean_radius = float(r.mean())
        std_radius = float(r.std())
        mean_height = float(z.mean())
        std_height = float(z.std())

        return cls(
            eigvals=eigvals,
            hist_z=hist_z,
            hist_r=hist_r,
            height=height,
            density=density,
            volume=volume_elipsoide,
            mean_radius=mean_radius,
            std_radius=std_radius,
            mean_height=mean_height,
            std_height=std_height,
        )

    # ---------- Helpers ----------
    @staticmethod
    def _fix_range(rng, values, default_min, default_max):
        """Normalize ranges and avoid degenerate cases."""
        if rng is None:
            rng = (default_min, default_max)
        if rng[1] - rng[0] < 1e-9:
            rng = (rng[0] - 0.5, rng[1] + 0.5)
        return rng

    # ---------- Public API ----------
    def to_vector(self) -> np.ndarray:
        """Concatenate all features into a 1D vector (order matters)."""
        return np.concatenate([
            self.eigvals,                                # 3
            self.hist_z,                                 # n_bins_z
            self.hist_r,                                 # n_bins_r
            [self.height,                                # 1
             self.density,                               # 1
             self.volume,                                # 1
             self.mean_radius,                           # 1
             self.std_radius,                            # 1
             self.mean_height,                           # 1
             self.std_height],                           # 1
        ])

    def distance_to(self, other: "PlaceDescriptor") -> float:
        """Normalized Euclidean distance between two descriptors."""
        v1, v2 = self.to_vector(), other.to_vector()
        return float(np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) + 1e-9))

    def correlation_with(self, other: "PlaceDescriptor") -> float:
        """Pearson correlation between the descriptor vectors."""
        v1, v2 = self.to_vector(), other.to_vector()
        return float(np.corrcoef(v1, v2)[0, 1])

    # ---------- Visualization ----------
    def plot_histograms(self, axs=None):
        """Plot z and radial histograms."""
        if axs is None:
            _, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        axs[0].bar(range(len(self.hist_z)), self.hist_z)
        axs[0].set_title("hist_z (sensor frame, normalized)")
        axs[0].set_xlabel("bin"); axs[0].set_ylabel("density")
        axs[1].bar(range(len(self.hist_r)), self.hist_r)
        axs[1].set_title("hist_r (sensor-anchored)")
        axs[1].set_xlabel("bin"); axs[1].set_ylabel("density")
        return axs


# ============================================================
# SDRPlaceEncoder  (unchanged)
# ============================================================
class SDRPlaceEncoder:
    """
    Encodes a PlaceDescriptor into a unified SDR.
    Each feature has its own RDSE encoder; outputs are concatenated.
    """

    def __init__(self, feature_ranges, feature_resolutions,
                 feature_sizes=None, active_bits=21, seed=42):
        """
        Parameters
        ----------
        feature_ranges : Sequence[(lo, hi)]
        feature_resolutions : Sequence[float]
        feature_sizes : int or Sequence[int], optional
            If int, same size for every feature. If Sequence, per-feature.
            Defaults to 400 bits per feature.
        active_bits : int or Sequence[int]
            If int, same active bits per feature. If Sequence, per-feature.
            Defaults to 21.
        seed : int
            RNG seed for RDSE reproducibility.
        """
        n_feat = len(feature_ranges)

        # --- Normalize feature_sizes to a list ---
        if feature_sizes is None:
            sizes = [400] * n_feat
        elif isinstance(feature_sizes, int):
            sizes = [feature_sizes] * n_feat
        else:
            sizes = list(feature_sizes)
            if len(sizes) != n_feat:
                raise ValueError(
                    f"feature_sizes has {len(sizes)} entries, "
                    f"expected {n_feat}."
                )

        # --- Normalize active_bits to a list ---
        if isinstance(active_bits, int):
            active_bits_list = [active_bits] * n_feat
        else:
            active_bits_list = list(active_bits)
            if len(active_bits_list) != n_feat:
                raise ValueError(
                    f"active_bits has {len(active_bits_list)} entries, "
                    f"expected {n_feat}."
                )

        # --- Save configuration ---
        self.encoders = []
        self.feature_ranges = list(feature_ranges)
        self.feature_resolutions = list(feature_resolutions)
        self.feature_sizes = sizes
        self.active_bits_list = active_bits_list
        self.seed = seed

        # --- Instantiate one RDSE per feature ---
        for i in range(n_feat):
            lo, hi = feature_ranges[i]
            resolution = feature_resolutions[i]
            size = sizes[i]
            w = active_bits_list[i]

            # Sanity check: buckets vs. size (empirical rule from RDSE)
            n_buckets = (hi - lo) / max(resolution, 1e-12)
            if n_buckets > size / 2:
                raise ValueError(
                    f"Feature {i}: too many buckets ({n_buckets:.0f}) "
                    f"for size {size}. Increase resolution or size."
                )

            # Sanity check: active bits must be < size
            if w >= size:
                raise ValueError(
                    f"Feature {i}: active_bits ({w}) >= size ({size})."
                )

            params = RDSE_Parameters()
            params.size       = size
            params.activeBits = w
            params.resolution = resolution
            params.seed       = seed
            self.encoders.append(RDSE(params))

        self.total_size = sum(sizes)

    def encode(self, desc_vector: np.ndarray) -> np.ndarray:
        """
        desc_vector: (D,) from PlaceDescriptor.to_vector()
        Returns: (total_size,) uint8 array of 0/1
        """
        parts = []
        for i, enc in enumerate(self.encoders):
            out = enc.encode(desc_vector[i])
            arr = out.dense.flatten() if hasattr(out, "dense") else np.asarray(out).flatten()
            parts.append(arr.astype(np.uint8))
        return np.concatenate(parts)


# ============================================================
# Utility: combined visualization
# ============================================================
def plot_pointcloud_and_descriptors(pc: PointCloud,
                                    n_bins_z: int = 10,
                                    n_bins_r: int = 10,
                                    z_range=None, r_range=None):
    """Plot top view + descriptor histograms."""
    desc = PlaceDescriptor.from_pointcloud(
        pc, n_bins_z=n_bins_z, n_bins_r=n_bins_r,
        z_range=z_range, r_range=r_range,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    pc.plot_top_view(ax=axes[0], show_centroid=True)
    desc.plot_histograms(axs=(axes[1], axes[2]))
    plt.tight_layout()
    plt.show()
    return desc