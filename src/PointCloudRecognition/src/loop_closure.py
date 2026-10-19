# loop_closure.py
"""
Block 5 of the PointCloudRecognition pipeline.

This module merges the HTM (block 4) with the loop-closure logic (block 5):

    1. Point cloud -> encoder -> SDR                       (blocks 2-3)
    2. SDR -> PlaceDatabase (place match + HTM context)     (block 4)
    3. If the place was already visited (loop closure):
         - signal the loop closure;
         - estimate the relative yaw (PCA+odometry or histogram);
         - add the new yaw to the corresponding experience (angular template);
         - relate the new yaw to the HTM temporal context;
    4. During localization, the temporal context indicates which stored
       angle is the correct one.

The HTM lives inside :class:`~src.place_database.PlaceDatabase`; this module
drives it and exposes the block-5 decisions (loop-closure events and a
read-only localization query).

Typical use
-----------
    db = PlaceDatabase(sdr_size=encoder.total_size, enable_context=True)
    lc = LoopClosureManager(db)

    for wp, path in enumerate(paths):
        pc  = PointCloud.from_npy(path)
        sdr = encoder.encode(PlaceDescriptor.from_pointcloud(pc).to_vector())
        ev  = lc.observe(sdr, pc=pc, wp_index=wp, odom_yaw_rad=odom[wp])

    # later, during localization (no learning):
    result = lc.localize(sdr, pc=pc, odom_yaw_rad=odom_now)
    print(result["angle_deg"])
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # normal case: project root on sys.path
    from src.place_database import PlaceDatabase
    from src.yaw_estimator import estimate_yaw
    from src.yaw_estimator_pca import estimate_yaw_pca_with_odometry
except ImportError:  # fallback when executing from inside src/
    from place_database import PlaceDatabase
    from yaw_estimator import estimate_yaw
    from yaw_estimator_pca import estimate_yaw_pca_with_odometry


__all__ = ["LoopClosureEvent", "LoopClosureManager"]


# ============================================================
# Event container
# ============================================================
@dataclass
class LoopClosureEvent:
    """One observation processed by the pipeline (blocks 4 + 5)."""

    iteration: int
    wp_new: int
    place_id: int
    matched: bool
    is_loop_closure: bool
    overlap_ratio: float
    wp_stored: Optional[int]
    wp_delta: Optional[int]
    anomaly: float
    n_winner_cells: int
    yaw_deg: float
    yaw_method: str
    yaw_confidence: float
    yaw_ambiguity: float
    template_idx: int
    template_created: bool
    context_template_idx: int
    context_similarity: float
    context_agrees: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Flat, CSV-friendly representation."""
        d = self.__dict__.copy()
        return d


# ============================================================
# Loop-closure manager
# ============================================================
class LoopClosureManager:
    """
    Drives the place database + HTM and implements the block-5 decisions.

    Parameters
    ----------
    db : PlaceDatabase
        Database to use. Context (HTM) should be enabled for the
        context-based angle retrieval to work.
    temporal_filter : int
        A match is only a loop closure if the new waypoint is more than this
        many waypoints away from the one that created the place (filters out
        trivially re-observed neighbours).
    yaw_method : {"stratified", "pca_odom"}
        - ``"stratified"``: yaw-invariant radial/angular histogram (no odometry).
        - ``"pca_odom"``: PCA-2D disambiguated with odometry (falls back to
          the histogram when odometry is unavailable).
    hist_n_bins, hist_n_rings : int
        Parameters of the stratified histogram.
    refine_icp : bool
        Refine the histogram estimate with yaw-only ICP.
    context_agreement_min_similarity : float
        Similarity above which the context-selected angle is considered a
        confident match (used to set ``context_agrees``).
    allow_context_angle_recovery : bool
        Optional heuristic (default False). When True, the temporal context
        may recover an existing angular template when the estimated yaw falls
        outside every known template, instead of creating a new one. Enable it
        only when the yaw estimate is unreliable: because the encoder is
        yaw-invariant, the context is a weak cue for the yaw itself and may
        merge genuinely different headings. The context's main role for the
        angle is the read-only :meth:`localize` selection.
    """

    def __init__(
        self,
        db: PlaceDatabase,
        temporal_filter: int = 5,
        yaw_method: str = "stratified",
        hist_n_bins: int = 360,
        hist_n_rings: int = 10,
        refine_icp: bool = False,
        context_agreement_min_similarity: float = 0.0,
        allow_context_angle_recovery: bool = False,
    ):
        self.db = db
        self.temporal_filter = int(temporal_filter)
        self.yaw_method = yaw_method
        self.hist_n_bins = int(hist_n_bins)
        self.hist_n_rings = int(hist_n_rings)
        self.refine_icp = bool(refine_icp)
        self.context_agreement_min_similarity = float(context_agreement_min_similarity)

        # Block 5: mapping is yaw-driven; the context is used to *select* the
        # angle during localization. Optionally let the context override yaw.
        self.db.allow_context_angle_recovery = bool(allow_context_angle_recovery)

        self.iteration = 0
        self.n_loop_closures = 0
        self.events: List[LoopClosureEvent] = []

        # Bookkeeping that the SDR database does not hold: the waypoint index
        # that created each place and the reference cloud / odom yaw to use
        # when estimating a relative yaw.
        self.place_first_wp: Dict[int, int] = {}
        self.place_ref_cloud: Dict[int, object] = {}
        self.place_ref_yaw: Dict[int, Optional[float]] = {}

    # ---------- Yaw estimation ----------
    def _estimate_yaw(
        self,
        ref_pc,
        query_pc,
        place_id: int,
        odom_yaw_rad: Optional[float],
    ) -> Tuple[float, str, float, float]:
        """
        Return ``(yaw_rad, method, confidence, ambiguity)`` for the relative
        yaw of `query_pc` w.r.t. the stored reference cloud.
        """
        nan = float("nan")

        if self.yaw_method == "pca_odom" and odom_yaw_rad is not None:
            ref_yaw = self.place_ref_yaw.get(place_id)
            if ref_yaw is not None:
                res = estimate_yaw_pca_with_odometry(
                    ref_pc, query_pc,
                    yaw_a_odom_rad=ref_yaw,
                    yaw_b_odom_rad=odom_yaw_rad,
                )
                return res.yaw_rad, res.method, nan, nan

        # Histogram fallback (also used as plain default).
        res = estimate_yaw(
            ref_pc, query_pc,
            n_bins=self.hist_n_bins,
            method="stratified",
            n_rings=self.hist_n_rings,
            refine_icp=self.refine_icp,
        )
        return res.yaw_rad, res.method, res.confidence, res.ambiguity

    # ---------- Main entry point (learning / mapping) ----------
    def observe(
        self,
        sdr: np.ndarray,
        pc=None,
        wp_index: Optional[int] = None,
        odom_yaw_rad: Optional[float] = None,
        label: str = "",
        learn: bool = True,
        context_reset: bool = False,
    ) -> LoopClosureEvent:
        """
        Process one point cloud: match, (maybe) close a loop, estimate yaw and
        update the HTM context + angular templates.

        Parameters
        ----------
        sdr : np.ndarray
            Encoder SDR of the observation.
        pc : PointCloud, optional
            The cloud itself (needed to estimate the relative yaw).
        wp_index : int, optional
            Chronological index of the observation (used for the temporal gap).
        odom_yaw_rad : float, optional
            Odometric heading at this observation (for ``yaw_method="pca_odom"``).
        label : str
            Label stored with a newly created place.
        learn : bool
            Passed to the HTM (``True`` while mapping).
        context_reset : bool
            Reset the HTM sequence at this step (new episode).
        """
        self.iteration += 1
        nan = float("nan")

        # ---- 1. Preliminary match (to know the candidate place) ----
        place_idx, ratio, _overlap = self.db.best_match(sdr)
        matched_candidate = (
            place_idx >= 0 and ratio >= self.db.match_threshold
        )
        cand_place_id = (
            self.db.places[place_idx].place_id if matched_candidate else None
        )

        # ---- 2. Temporal gap -> is this a loop closure? ----
        stored_wp: Optional[int] = (
            self.place_first_wp.get(cand_place_id)
            if cand_place_id is not None else None
        )
        wp_delta: Optional[int] = None
        if wp_index is not None and stored_wp is not None:
            wp_delta = abs(wp_index - stored_wp)
        is_loop_closure = bool(
            matched_candidate
            and (wp_delta is None or wp_delta > self.temporal_filter)
        )

        # ---- 3. Relative yaw (only for genuine revisits) ----
        yaw_rad, yaw_method, yaw_conf, yaw_ambig = 0.0, "none", nan, nan
        if is_loop_closure and pc is not None and cand_place_id is not None:
            ref_pc = self.place_ref_cloud.get(cand_place_id)
            if ref_pc is not None:
                yaw_rad, yaw_method, yaw_conf, yaw_ambig = self._estimate_yaw(
                    ref_pc, pc, cand_place_id, odom_yaw_rad
                )

        # ---- 4. Commit: match/create, update templates, advance the HTM ----
        place_id, matched, template_idx = self.db.match_or_create(
            sdr,
            yaw_rad=yaw_rad,
            label=label,
            learn_context=learn,
            context_reset=context_reset,
        )

        context = self.db.last_context
        context_cells = context.winner_cells if context is not None else None
        anomaly = float(context.anomaly) if context is not None else nan
        n_winner = int(context_cells.size) if context_cells is not None else 0

        # ---- 5. Context -> angle retrieval (block 5 core) ----
        ctx_tidx, ctx_sim = self.db.resolve_angle_by_context(place_id, context_cells)
        context_agrees = bool(
            template_idx >= 0
            and ctx_tidx >= 0
            and ctx_tidx == template_idx
            and ctx_sim >= self.context_agreement_min_similarity
        )

        # ---- 6. Register the reference cloud/waypoint for new places ----
        if not matched:
            self.place_first_wp[place_id] = (
                wp_index if wp_index is not None else self.iteration
            )
            self.place_ref_cloud[place_id] = pc
            self.place_ref_yaw[place_id] = odom_yaw_rad

        if is_loop_closure:
            self.n_loop_closures += 1

        event = LoopClosureEvent(
            iteration=self.iteration,
            wp_new=wp_index if wp_index is not None else self.iteration,
            place_id=place_id,
            matched=bool(matched),
            is_loop_closure=is_loop_closure,
            overlap_ratio=float(ratio),
            wp_stored=stored_wp,
            wp_delta=wp_delta,
            anomaly=anomaly,
            n_winner_cells=n_winner,
            yaw_deg=float(np.rad2deg(yaw_rad)),
            yaw_method=yaw_method,
            yaw_confidence=float(yaw_conf),
            yaw_ambiguity=float(yaw_ambig),
            template_idx=int(template_idx),
            template_created=bool(self.db.last_template_created),
            context_template_idx=int(ctx_tidx),
            context_similarity=float(ctx_sim),
            context_agrees=context_agrees,
        )
        self.events.append(event)
        return event

    # ---------- Localization (inference, no learning) ----------
    def localize(
        self,
        sdr: np.ndarray,
        pc=None,
        odom_yaw_rad: Optional[float] = None,
        learn: bool = False,
        context_reset: bool = False,
        min_context_similarity: float = 0.0,
    ) -> dict:
        """
        Query the map without modifying it.

        Returns a dict with:
            matched, place_id, overlap_ratio,
            context_template_idx, context_similarity, angle_deg (context-selected),
            yaw_deg, yaw_method   (yaw estimate vs. the reference cloud, if any),
            anomaly
        """
        nan = float("nan")

        place_idx, ratio, _ = self.db.best_match(sdr)
        matched = place_idx >= 0 and ratio >= self.db.match_threshold
        place_id = self.db.places[place_idx].place_id if matched else -1

        context = self.db.query_context(sdr, learn=learn, reset=context_reset)
        context_cells = context.winner_cells if context is not None else None
        anomaly = float(context.anomaly) if context is not None else nan

        angle_rad, angle_deg, ctx_tidx, ctx_sim = None, None, -1, 0.0
        if matched and context_cells is not None:
            angle_rad, ctx_tidx, ctx_sim = self.db.get_angle_for_context(
                place_id, context_cells, min_similarity=min_context_similarity
            )
            if angle_rad is not None:
                angle_deg = float(np.rad2deg(angle_rad))

        yaw_rad, yaw_method, yaw_conf, yaw_ambig = nan, "none", nan, nan
        if matched and pc is not None:
            ref_pc = self.place_ref_cloud.get(place_id)
            if ref_pc is not None:
                yaw_rad, yaw_method, yaw_conf, yaw_ambig = self._estimate_yaw(
                    ref_pc, pc, place_id, odom_yaw_rad
                )

        return {
            "matched":        bool(matched),
            "place_id":       int(place_id),
            "overlap_ratio":  float(ratio),
            "anomaly":        anomaly,
            "context_template_idx": int(ctx_tidx),
            "context_similarity":   float(ctx_sim),
            "angle_deg":      angle_deg,
            "yaw_deg":        float(np.rad2deg(yaw_rad)) if not np.isnan(yaw_rad) else None,
            "yaw_method":     yaw_method,
            "yaw_confidence": yaw_conf,
            "yaw_ambiguity":  yaw_ambig,
        }

    # ---------- Reporting / persistence ----------
    def summary(self) -> str:
        matched = [e for e in self.events if e.matched]
        ctx_matched = [e for e in matched if e.context_template_idx >= 0]
        agree = sum(1 for e in ctx_matched if e.context_agrees)
        lines = [
            f"LoopClosureManager: {self.iteration} observations, "
            f"{self.n_loop_closures} loop closures, "
            f"{len(matched)} matches, {len(self.db.places)} places",
            f"  context resolved an angle in {len(ctx_matched)}/{len(matched)} "
            f"matched events; agreed with the yaw template in {agree}",
        ]
        return "\n".join(lines)

    def save_events_csv(self, path: str | Path) -> None:
        if not self.events:
            return
        fieldnames = list(self.events[0].to_dict().keys())
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.events:
                writer.writerow(e.to_dict())
