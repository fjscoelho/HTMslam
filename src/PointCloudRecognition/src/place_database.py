# place_database.py
"""
Place database for the no-SP pipeline (blocks 2-4, feeding block 5).

Stores one entry per visited place. Each entry contains:
    - The SDR produced by the encoder (for place matching).
    - A list of angular templates (yaw clusters) seen at this place.
    - Visit counts and timestamps.
    - (optional) the HTM temporal context associated with each yaw template.

Place matching is done by SDR overlap: a new cloud is assigned to the
existing place with the highest overlap, if that overlap exceeds a
threshold. Otherwise a new place is created.

Angular templates are stored as clusters. When a new cloud is matched to
an existing place, its estimated yaw is compared against existing
templates: if it is close to one (within `yaw_tolerance_deg`), that
template's count is incremented; otherwise a new template is added.

HTM temporal context (block 4)
------------------------------
When ``enable_context`` is set, the database owns a
:class:`~src.place_context.PlaceContextMemory` (an HTM TemporalMemory whose
mini-column space equals the encoder SDR size). Every observation is fed to
the TM and its **winner cells** -- which encode the temporal context, i.e.
the sequence of places that led here -- are:

    * recorded in ``context_log`` (one entry per iteration), and
    * merged into the matched :class:`AngularTemplate` (relating a yaw to the
      context in which it was seen).

Block 5 can then ask the database which angular template best matches the
*current context* (:meth:`PlaceDatabase.resolve_angle_by_context`): the
context indicates the correct angle when yaw alone is ambiguous.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # normal case: scripts put the project root on sys.path and import `src.*`
    from src.place_context import PlaceContextMemory, ContextState
except ImportError:  # fallback when executing from inside src/
    from place_context import PlaceContextMemory, ContextState


# ============================================================
# Data classes
# ============================================================
@dataclass
class AngularTemplate:
    """A yaw cluster seen at a given place."""
    mean_yaw_rad: float        # representative yaw (circular mean)
    visit_count: int = 1
    last_seen_ts: float = field(default_factory=time.time)
    # Optional: store the angular histogram for later refinement
    angular_histogram: Optional[np.ndarray] = None
    # HTM temporal context associated with this yaw: union of the winner-cell
    # sets observed whenever this template was matched (see place_context.py).
    context_cells: Optional[np.ndarray] = None
    context_visits: int = 0    # how many observations contributed a context

    @property
    def n_context_cells(self) -> int:
        return 0 if self.context_cells is None else int(self.context_cells.size)

    def to_json(self) -> dict:
        d = {
            "mean_yaw_rad":     self.mean_yaw_rad,
            "visit_count":      self.visit_count,
            "last_seen_ts":     self.last_seen_ts,
            "context_visits":   self.context_visits,
        }
        if self.angular_histogram is not None:
            d["angular_histogram"] = self.angular_histogram.tolist()
        if self.context_cells is not None:
            d["context_cells"] = self.context_cells.tolist()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "AngularTemplate":
        hist = d.get("angular_histogram")
        ctx = d.get("context_cells")
        return cls(
            mean_yaw_rad=d["mean_yaw_rad"],
            visit_count=d["visit_count"],
            last_seen_ts=d["last_seen_ts"],
            angular_histogram=np.array(hist) if hist is not None else None,
            context_cells=(np.asarray(ctx, dtype=np.uint32)
                           if ctx is not None else None),
            context_visits=int(d.get("context_visits", 0)),
        )


@dataclass
class PlaceEntry:
    """A single place in the database."""
    place_id: int
    sdr: np.ndarray                          # (total_size,) 0/1
    angular_templates: List[AngularTemplate] = field(default_factory=list)
    visit_count: int = 1
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    label: str = ""                          # optional human-readable label

    @property
    def n_templates(self) -> int:
        return len(self.angular_templates)

    @property
    def context_cells(self) -> Optional[np.ndarray]:
        """
        Union of the winner-cell contexts of all yaw templates at this place
        (``None`` if no template carries a context yet).
        """
        cells = [t.context_cells for t in self.angular_templates
                 if t.context_cells is not None and t.context_cells.size > 0]
        if not cells:
            return None
        return np.unique(np.concatenate(cells))

    def to_json(self) -> dict:
        return {
            "place_id":          self.place_id,
            "visit_count":       self.visit_count,
            "first_seen_ts":     self.first_seen_ts,
            "last_seen_ts":      self.last_seen_ts,
            "label":             self.label,
            "angular_templates": [t.to_json() for t in self.angular_templates],
            # sdr is stored separately in a .npy file
        }


# ============================================================
# Helper: circular angle utilities
# ============================================================
def circular_mean(angles_rad: List[float]) -> float:
    """Circular mean of a list of angles (radians), robust to wraparound."""
    s = np.mean(np.sin(angles_rad))
    c = np.mean(np.cos(angles_rad))
    return float(np.arctan2(s, c))


def angular_distance(a_rad: float, b_rad: float) -> float:
    """Shortest angular distance between two angles (radians)."""
    d = abs(a_rad - b_rad) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))


# ============================================================
# PlaceDatabase
# ============================================================
class PlaceDatabase:
    """
    Manages a set of visited places, plus an optional HTM temporal context.

    Public API
    ----------
    match_or_create(sdr, yaw_rad, angular_hist) -> (place_id, matched, template_idx)
        Match a new observation to an existing place (or create a new one).
        If matched to a place, updates its angular templates.
        If ``enable_context`` is set, also advances the HTM and stores the
        winner cells (temporal context) on the matched template.
        Returns:
            place_id     : int
            matched      : bool  (True if matched, False if new place)
            template_idx : int   (-1 if new place; otherwise the template matched)

    best_match(sdr) -> (place_idx, overlap_ratio, overlap_count)
    query_context(sdr, learn) -> ContextState
        Run the HTM on an SDR without touching the place store (localization).
    resolve_angle_by_context(place_id, context_cells) -> (template_idx, similarity)
        Pick the angular template whose stored context best matches the query
        context. This is the block-5 "the context indicates the angle" step.

    get_sdr(place_id) -> np.ndarray
    get_templates(place_id) -> List[AngularTemplate]
    save(path_prefix) / load(path_prefix)
    """

    def __init__(
        self,
        sdr_size: int,
        match_threshold: float = 0.5,
        yaw_tolerance_deg: float = 15.0,
        enable_context: bool = True,
        cells_per_column: int = 8,
        context_seed: int = 42,
        context_reuse_min_similarity: float = 0.3,
        allow_context_angle_recovery: bool = False,
        context_params: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        sdr_size : int
            Length of the encoder SDR (e.g., 41800). Also used as the number
            of mini-columns of the HTM temporal memory.
        match_threshold : float
            Minimum overlap ratio (overlap / active_bits) to consider two
            SDRs as the same place. Typical: 0.5 (50%).
        yaw_tolerance_deg : float
            Maximum angular distance (degrees) to consider two yaws as the
            same template. Typical: 15 degrees.
        enable_context : bool
            If True (default), build an HTM TemporalMemory and record winner
            cells as temporal context on every observation.
        cells_per_column : int
            Cells per mini-column of the HTM.
        context_seed : int
            RNG seed of the HTM (reproducibility).
        context_reuse_min_similarity : float
            When a yaw lands outside every existing template's tolerance, the
            context may still associate it with an existing template if the
            context Jaccard similarity is at least this value. Only used when
            ``allow_context_angle_recovery`` is True.
        allow_context_angle_recovery : bool
            If True, the temporal context may recover an existing template when
            the estimated yaw falls outside every template (block-5 behaviour:
            "the context indicates the correct angle"). Disabled by default so
            that mapping stays yaw-driven; the block-5 driver enables it.
        context_params : dict, optional
            Extra hyper-parameters forwarded to :class:`PlaceContextMemory`
            (e.g. ``{"activation_threshold": 12, "min_threshold": 8}``).
        """
        self.sdr_size = int(sdr_size)
        self.match_threshold = float(match_threshold)
        self.yaw_tolerance_rad = float(np.deg2rad(yaw_tolerance_deg))

        self.places: List[PlaceEntry] = []
        self._next_place_id: int = 0

        # Stats (useful for diagnostics)
        self.n_queries: int = 0
        self.n_matches: int = 0
        self.n_creations: int = 0

        # Info about the most recent match_or_create() call (block 5 helpers).
        self.last_match_ratio: float = 0.0
        self.last_template_created: bool = False

        # ---- HTM temporal context (block 4) ----
        self.enable_context: bool = bool(enable_context)
        self.context_reuse_min_similarity: float = float(context_reuse_min_similarity)
        self.allow_context_angle_recovery: bool = bool(allow_context_angle_recovery)
        self.context_memory: Optional[PlaceContextMemory] = None
        self.context_log: List[ContextState] = []   # one entry per iteration
        self.last_context: Optional[ContextState] = None

        if self.enable_context:
            params = dict(context_params or {})
            self.context_memory = PlaceContextMemory(
                sdr_size=self.sdr_size,
                cells_per_column=cells_per_column,
                seed=context_seed,
                **params,
            )

    # ---------- Core API ----------
    def _best_place_index(
        self, sdr: np.ndarray, n_active: Optional[int] = None
    ) -> Tuple[int, float, int]:
        """Index/ratio/count of the place whose stored SDR best overlaps `sdr`."""
        if n_active is None:
            n_active = int(sdr.sum())
        best_idx = -1
        best_overlap = 0
        for idx, place in enumerate(self.places):
            overlap = int(np.logical_and(sdr, place.sdr).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        ratio = best_overlap / n_active if n_active > 0 else 0.0
        return best_idx, ratio, best_overlap

    def best_match(self, sdr: np.ndarray) -> Tuple[int, float, int]:
        """
        Read-only lookup of the best matching place.

        Returns
        -------
        place_idx : int      index into ``self.places`` (-1 if empty)
        ratio : float        overlap / active_bits
        overlap : int        number of shared active bits
        """
        sdr = np.asarray(sdr)
        n_active = int(sdr.sum())
        if n_active == 0:
            return -1, 0.0, 0
        return self._best_place_index(sdr, n_active)

    def _select_template(
        self,
        place: PlaceEntry,
        yaw_rad: float,
        context_cells: Optional[np.ndarray],
    ) -> Tuple[int, bool]:
        """
        Choose which angular template a new observation belongs to.

        Returns ``(template_idx, is_new)``. ``is_new=True`` means a new
        template should be appended (``template_idx`` is then meaningless).

        Rules
        -----
        1. Candidates within ``yaw_tolerance_rad`` are preferred; among them
           the one whose stored context best matches wins (tie-break: nearest
           yaw). If no context is available, the nearest yaw wins (legacy).
        2. If no template is within yaw tolerance *and*
           ``allow_context_angle_recovery`` is set, the context may still
           recover an existing template (Jaccard >=
           ``context_reuse_min_similarity``). This is how the context recovers
           the correct angle when yaw is ambiguous (block 5).
        3. Otherwise a new template is created.

        Note: rule 2 is opt-in because during *mapping* the yaw estimate is the
        primary evidence; enabling it lets the context override yaw when the
        yaw points outside every known template.
        """
        templates = place.angular_templates
        if not templates:
            return -1, True

        dists = [angular_distance(yaw_rad, t.mean_yaw_rad) for t in templates]
        within = [i for i, d in enumerate(dists) if d <= self.yaw_tolerance_rad]

        have_ctx = (context_cells is not None and context_cells.size > 0)

        if within:
            if have_ctx:
                best_i, best_sim = -1, 0.0
                for i in within:
                    sim = ContextState.jaccard(
                        templates[i].context_cells, context_cells
                    )
                    if sim > best_sim:
                        best_sim, best_i = sim, i
                if best_i >= 0:
                    return best_i, False
            # No usable context: fall back to nearest yaw (legacy behaviour).
            return min(within, key=lambda i: dists[i]), False

        # No yaw candidate: let the context try to recover the angle (opt-in).
        if have_ctx and self.allow_context_angle_recovery:
            best_i, best_sim = -1, 0.0
            for i, t in enumerate(templates):
                sim = ContextState.jaccard(t.context_cells, context_cells)
                if sim > best_sim:
                    best_sim, best_i = sim, i
            if best_i >= 0 and best_sim >= self.context_reuse_min_similarity:
                return best_i, False

        return -1, True

    @staticmethod
    def _context_copy(cells: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if cells is None or cells.size == 0:
            return None
        return np.asarray(cells, dtype=np.uint32).copy()

    @staticmethod
    def _merge_context(
        existing: Optional[np.ndarray], new: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Union of two winner-cell context sets."""
        if existing is None or existing.size == 0:
            return PlaceDatabase._context_copy(new)
        if new is None or new.size == 0:
            return existing
        return np.union1d(existing, new).astype(np.uint32, copy=False)

    def match_or_create(
        self,
        sdr: np.ndarray,
        yaw_rad: float,
        angular_hist: Optional[np.ndarray] = None,
        label: str = "",
        learn_context: bool = True,
        context_reset: bool = False,
    ) -> Tuple[int, bool, int]:
        """
        Match an observation against the database.

        If ``enable_context`` is set, the SDR is first presented to the HTM
        (block 4) and the resulting winner cells are used both to select the
        angular template and to update the stored context.

        Parameters
        ----------
        sdr : np.ndarray
            Encoder SDR of the observation.
        yaw_rad : float
            Estimated relative yaw (radians). Pass 0.0 if unknown.
        angular_hist : np.ndarray, optional
            Angular histogram of the cloud (stored for later refinement).
        label : str
            Optional human-readable label for a newly created place.
        learn_context : bool
            Enable HTM learning for this step (``False`` for pure inference).
        context_reset : bool
            Reset the HTM sequence state before this step (new episode).

        Returns
        -------
        place_id : int
        matched : bool
        template_idx : int  (-1 if new place, else index of the matched template)
        """
        self.n_queries += 1
        sdr = np.asarray(sdr)
        sdr_active = int(sdr.sum())
        if sdr_active == 0:
            raise ValueError("Empty SDR (no active bits)")

        # ---- 0. HTM temporal context (block 4) ----
        context: Optional[ContextState] = None
        ctx_cells: Optional[np.ndarray] = None
        if self.context_memory is not None:
            context = self.context_memory.compute(
                sdr, learn=learn_context, reset=context_reset
            )
            ctx_cells = context.winner_cells
        self.last_context = context

        # ---- 1. Find best-matching place by SDR overlap ----
        best_place_idx, best_overlap_ratio, _ = self._best_place_index(
            sdr, sdr_active
        )

        # ---- 2. Decide match vs. create ----
        matched = best_place_idx >= 0 and best_overlap_ratio >= self.match_threshold
        self.last_match_ratio = best_overlap_ratio

        if not matched:
            # Create new place
            place_id = self._next_place_id
            self._next_place_id += 1

            new_place = PlaceEntry(
                place_id=place_id,
                sdr=sdr.copy(),
                angular_templates=[
                    AngularTemplate(
                        mean_yaw_rad=yaw_rad,
                        visit_count=1,
                        angular_histogram=angular_hist,
                        context_cells=self._context_copy(ctx_cells),
                        context_visits=1 if ctx_cells is not None and ctx_cells.size else 0,
                    )
                ],
                visit_count=1,
                label=label,
            )
            self.places.append(new_place)
            self.n_creations += 1
            self.last_template_created = True
            self._log_context(context, place_id, matched=False)
            return place_id, False, -1

        # ---- 3. Matched: update place and angular templates ----
        place = self.places[best_place_idx]
        place.visit_count += 1
        place.last_seen_ts = time.time()

        t_idx, is_new = self._select_template(place, yaw_rad, ctx_cells)
        self.last_template_created = bool(is_new)

        if is_new:
            # Add a new template to this place, tagged with the current context
            place.angular_templates.append(
                AngularTemplate(
                    mean_yaw_rad=yaw_rad,
                    visit_count=1,
                    angular_histogram=angular_hist,
                    context_cells=self._context_copy(ctx_cells),
                    context_visits=1 if ctx_cells is not None and ctx_cells.size else 0,
                )
            )
            template_idx = len(place.angular_templates) - 1
        else:
            # Update the selected template (circular mean, histogram, context)
            t = place.angular_templates[t_idx]
            w_old = t.visit_count
            w_new = 1
            s = (w_old * np.sin(t.mean_yaw_rad) + w_new * np.sin(yaw_rad)) / (w_old + w_new)
            c = (w_old * np.cos(t.mean_yaw_rad) + w_new * np.cos(yaw_rad)) / (w_old + w_new)
            t.mean_yaw_rad = float(np.arctan2(s, c))
            t.visit_count += 1
            t.last_seen_ts = time.time()
            if angular_hist is not None and t.angular_histogram is not None:
                # Running average of histograms
                t.angular_histogram = (
                    (w_old * t.angular_histogram + w_new * angular_hist)
                    / (w_old + w_new)
                )
            # Relate this yaw to the observed temporal context.
            if ctx_cells is not None and ctx_cells.size:
                t.context_cells = self._merge_context(t.context_cells, ctx_cells)
                t.context_visits += 1
            template_idx = t_idx

        self.n_matches += 1
        self._log_context(context, place.place_id, matched=True)
        return place.place_id, True, template_idx

    def _log_context(
        self,
        context: Optional[ContextState],
        place_id: int,
        matched: bool,
    ) -> None:
        """Record the winner cells of one iteration (block 4 output log)."""
        if context is None:
            return
        context.place_id = place_id
        context.matched = matched
        self.context_log.append(context)

    # ---------- HTM context: inference helpers (block 5) ----------
    def query_context(
        self,
        sdr: np.ndarray,
        learn: bool = False,
        reset: bool = False,
    ) -> Optional[ContextState]:
        """
        Run the HTM on an SDR *without* touching the place store.

        Used during localization to obtain the current temporal context
        (winner cells) before deciding anything.
        """
        if self.context_memory is None:
            return None
        return self.context_memory.compute(sdr, learn=learn, reset=reset)

    def resolve_angle_by_context(
        self,
        place_id: int,
        context_cells: Optional[np.ndarray],
        min_similarity: float = 0.0,
    ) -> Tuple[int, float]:
        """
        Pick the angular template whose stored context best matches the query
        context. This is the block-5 rule "the context indicates the correct
        angle".

        Returns
        -------
        template_idx : int   index into the place's templates (-1 if none)
        similarity : float   Jaccard similarity of the best match
        """
        place = self.get_place(place_id)
        if place is None or context_cells is None or context_cells.size == 0:
            return -1, 0.0

        best_idx, best_sim = -1, 0.0
        for i, t in enumerate(place.angular_templates):
            sim = ContextState.jaccard(t.context_cells, context_cells)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        if best_sim < min_similarity:
            return -1, best_sim
        return best_idx, best_sim

    def get_angle_for_context(
        self,
        place_id: int,
        context_cells: Optional[np.ndarray],
        min_similarity: float = 0.0,
    ) -> Tuple[Optional[float], int, float]:
        """
        Convenience wrapper: return ``(mean_yaw_rad, template_idx, similarity)``
        for the template selected by the context (``None`` if unresolved).
        """
        idx, sim = self.resolve_angle_by_context(
            place_id, context_cells, min_similarity=min_similarity
        )
        place = self.get_place(place_id)
        if idx < 0 or place is None:
            return None, idx, sim
        return place.angular_templates[idx].mean_yaw_rad, idx, sim

    # ---------- Accessors ----------
    def get_place(self, place_id: int) -> Optional[PlaceEntry]:
        for p in self.places:
            if p.place_id == place_id:
                return p
        return None

    def get_sdr(self, place_id: int) -> Optional[np.ndarray]:
        p = self.get_place(place_id)
        return p.sdr if p is not None else None

    def get_templates(self, place_id: int) -> List[AngularTemplate]:
        p = self.get_place(place_id)
        return p.angular_templates if p is not None else []

    def summary(self) -> str:
        lines = [
            f"PlaceDatabase: {len(self.places)} places, "
            f"{self.n_queries} queries, "
            f"{self.n_matches} matches, "
            f"{self.n_creations} creations",
        ]
        if self.context_memory is not None:
            lines.append(
                f"  HTM context: {self.context_memory.n_columns} columns x "
                f"{self.context_memory.cells_per_column} cells/col, "
                f"iterations={self.context_memory.iteration}, "
                f"logged={len(self.context_log)}"
            )
        for p in self.places:
            yaws_deg = [f"{np.rad2deg(t.mean_yaw_rad):+.0f}°"
                        for t in p.angular_templates]
            n_ctx = sum(1 for t in p.angular_templates if t.context_cells is not None)
            lines.append(
                f"  place {p.place_id:>3}: "
                f"visits={p.visit_count:>3}, "
                f"templates={p.n_templates:>2} "
                f"[{', '.join(yaws_deg)}]"
                + (f", ctx_templates={n_ctx}" if self.context_memory is not None else "")
                + (f"  # {p.label}" if p.label else "")
            )
        return "\n".join(lines)

    # ---------- Persistence ----------
    def save(self, prefix: str | Path) -> None:
        """
        Save the database to disk.

        Files written:
            <prefix>_meta.json          - metadata + angular templates
            <prefix>_sdrs.npy           - (N_places, sdr_size) array of SDRs
            <prefix>_context_log.json   - winner cells per iteration (if enabled)
            <prefix>_tm.bin             - HTM snapshot (if enabled)
            <prefix>_context_meta.json  - HTM hyper-parameters (if enabled)
        """
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "sdr_size":         self.sdr_size,
            "match_threshold":  self.match_threshold,
            "yaw_tolerance_rad": self.yaw_tolerance_rad,
            "next_place_id":    self._next_place_id,
            "n_queries":        self.n_queries,
            "n_matches":        self.n_matches,
            "n_creations":      self.n_creations,
            "enable_context":   self.enable_context,
            "context_reuse_min_similarity": self.context_reuse_min_similarity,
            "allow_context_angle_recovery": self.allow_context_angle_recovery,
            "n_context_log":    len(self.context_log),
            "places":           [p.to_json() for p in self.places],
        }
        prefix.with_name(prefix.name + "_meta.json").write_text(
            json.dumps(meta, indent=2)
        )

        # Stack SDRs into one array
        if self.places:
            sdrs = np.stack([p.sdr for p in self.places], axis=0)
        else:
            sdrs = np.zeros((0, self.sdr_size), dtype=np.uint8)
        np.save(prefix.with_name(prefix.name + "_sdrs.npy"), sdrs)

        # ---- HTM temporal context ----
        if self.context_memory is not None:
            self.context_memory.save(prefix)

        if self.context_log:
            prefix.with_name(prefix.name + "_context_log.json").write_text(
                json.dumps([c.to_json() for c in self.context_log], indent=2)
            )

    @classmethod
    def load(
        cls,
        prefix: str | Path,
        enable_context: Optional[bool] = None,
    ) -> "PlaceDatabase":
        """
        Reload a database previously written by :meth:`save`.

        Parameters
        ----------
        enable_context : bool, optional
            If ``None`` (default), the HTM context is restored iff its files
            are present next to the database files.
        """
        prefix = Path(prefix)
        meta = json.loads(
            prefix.with_name(prefix.name + "_meta.json").read_text()
        )
        sdrs = np.load(prefix.with_name(prefix.name + "_sdrs.npy"))

        has_context_files = (
            prefix.with_name(prefix.name + "_context_meta.json").exists()
            and prefix.with_name(prefix.name + "_tm.bin").exists()
        )
        if enable_context is None:
            enable_context = has_context_files

        db = cls(
            sdr_size=meta["sdr_size"],
            match_threshold=meta["match_threshold"],
            yaw_tolerance_deg=float(np.rad2deg(meta["yaw_tolerance_rad"])),
            enable_context=enable_context,
            context_reuse_min_similarity=meta.get("context_reuse_min_similarity", 0.3),
            allow_context_angle_recovery=meta.get("allow_context_angle_recovery", False),
        )
        db._next_place_id = meta["next_place_id"]
        db.n_queries      = meta["n_queries"]
        db.n_matches      = meta["n_matches"]
        db.n_creations    = meta["n_creations"]
        db.enable_context = bool(meta.get("enable_context", enable_context))

        # Restore HTM state if available.
        if enable_context and has_context_files:
            db.context_memory = PlaceContextMemory.load(prefix)

        # Restore the per-iteration context log if available.
        log_path = prefix.with_name(prefix.name + "_context_log.json")
        if log_path.exists():
            raw = json.loads(log_path.read_text())
            db.context_log = [ContextState.from_json(d) for d in raw]

        for i, pj in enumerate(meta["places"]):
            place = PlaceEntry(
                place_id=pj["place_id"],
                sdr=sdrs[i],
                angular_templates=[
                    AngularTemplate.from_json(tj) for tj in pj["angular_templates"]
                ],
                visit_count=pj["visit_count"],
                first_seen_ts=pj["first_seen_ts"],
                last_seen_ts=pj["last_seen_ts"],
                label=pj["label"],
            )
            db.places.append(place)
        return db

    # ---------- Convenience: the per-iteration context log ----------
    def get_context_log_as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ``(place_ids, anomalies)`` aligned with ``context_log``, useful
        for quick diagnostics/plots.
        """
        place_ids = np.array([c.place_id for c in self.context_log], dtype=int)
        anomalies = np.array([c.anomaly for c in self.context_log], dtype=float)
        return place_ids, anomalies

    def winner_cells_of(self, iteration: int) -> Optional[np.ndarray]:
        """Winner cells recorded at a given iteration (``None`` if missing)."""
        if 0 <= iteration < len(self.context_log):
            return self.context_log[iteration].winner_cells
        return None