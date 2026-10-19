# place_context.py
"""
Temporal-context memory for the place-recognition pipeline (block 4).

This module wraps an HTM **TemporalMemory (TM)** whose mini-column space is
sized to the encoder SDR: one mini-column per SDR bit. Every point cloud is
presented to the TM as the set of active mini-columns (i.e. the SDR produced
by :class:`~src.place_encoder.SDRPlaceEncoder`).

Why the SDR is used directly as TM input
----------------------------------------
The encoder already produces a sparse, distributed representation (a few
hundred active bits out of tens of thousands). Feeding it straight to the TM
means we do **not** need a SpatialPooler: the "input side" of the HTM is just
the configured SDR size.

What the TM gives us (block 4 outputs)
--------------------------------------
* ``winner_cells``   – the cells that won the competition for this step.
  This set is the **temporal context**: it depends not only on the current
  place but on the sequence of places that led here. Two visits to the same
  place reached through different trajectories produce different winner-cell
  sets, which is exactly what block 5 needs to disambiguate angles.
* ``predictive_cells`` – cells predicting the next input (short-horizon
  expectation).
* ``anomaly``        – how unexpected the current input was (a loop closure
  after a long detour typically shows up as a high anomaly the first time).

The winner-cell set of every step is meant to be **persisted per iteration**
so the rest of the pipeline can reason about temporal context.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM


__all__ = ["ContextState", "PlaceContextMemory"]


# ============================================================
# Per-iteration context snapshot
# ============================================================
@dataclass
class ContextState:
    """
    Result of one TM step: the (winner) cells that encode the temporal
    context of a single observation.

    Only ``winner_cells`` is persisted to JSON (the active/predictive sets are
    debug aids and would bloat the file).
    """

    iteration: int
    anomaly: float
    winner_cells: np.ndarray                       # (W,) uint32 sparse cell ids
    active_cells: np.ndarray                       # (A,) uint32 sparse cell ids
    predictive_cells: np.ndarray                   # (P,) uint32 sparse cell ids
    place_id: int = -1                             # filled in by PlaceDatabase
    matched: bool = False                          # filled in by PlaceDatabase
    timestamp: float = field(default_factory=time.time)

    # ---------- Convenience ----------
    @property
    def n_winner(self) -> int:
        return int(self.winner_cells.size)

    @property
    def n_active(self) -> int:
        return int(self.active_cells.size)

    @property
    def n_predictive(self) -> int:
        return int(self.predictive_cells.size)

    # ---------- (De)serialization ----------
    def to_json(self) -> dict:
        """Compact, JSON-serializable view (winner cells only)."""
        return {
            "iteration":    int(self.iteration),
            "anomaly":      float(self.anomaly),
            "place_id":     int(self.place_id),
            "matched":      bool(self.matched),
            "timestamp":    float(self.timestamp),
            "winner_cells": self.winner_cells.tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "ContextState":
        return cls(
            iteration=int(d["iteration"]),
            anomaly=float(d.get("anomaly", 0.0)),
            winner_cells=np.asarray(d.get("winner_cells", []), dtype=np.uint32),
            active_cells=np.empty(0, dtype=np.uint32),
            predictive_cells=np.empty(0, dtype=np.uint32),
            place_id=int(d.get("place_id", -1)),
            matched=bool(d.get("matched", False)),
            timestamp=float(d.get("timestamp", 0.0)),
        )

    # ---------- Context comparison ----------
    @staticmethod
    def overlap(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> int:
        """Number of cell ids shared by two context sets."""
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0
        return int(np.intersect1d(a, b, assume_unique=False).size)

    @staticmethod
    def jaccard(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Jaccard similarity (0..1) between two context sets."""
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        inter = int(np.intersect1d(a, b, assume_unique=False).size)
        union = int(a.size) + int(b.size) - inter
        return float(inter / union) if union > 0 else 0.0


# ============================================================
# HTM temporal-context memory
# ============================================================
class PlaceContextMemory:
    """
    Thin, persistence-friendly wrapper around ``htm.algorithms.TemporalMemory``.

    Parameters
    ----------
    sdr_size : int
        Number of mini-columns. Set this to the encoder SDR size
        (``encoder.total_size``) so the HTM "input side" matches the SDR.
    cells_per_column : int
        Cells per mini-column. Small values (4-16) are plenty here because
        the input is already high-dimensional and sparse.
    **tm_params
        Optional overrides for the remaining TM hyper-parameters
        (activationThreshold, minThreshold, maxNewSynapseCount, ...).

    Notes
    -----
    The TM is fed in chronological order. Call :meth:`reset` when a new
    sequence/episode starts (e.g. a new run through the environment).
    """

    #: TM hyper-parameters that are exposed for tuning / persistence.
    _TM_PARAM_KEYS = (
        "activation_threshold",
        "initial_permanence",
        "connected_permanence",
        "min_threshold",
        "max_new_synapse_count",
        "permanence_increment",
        "permanence_decrement",
        "predicted_segment_decrement",
        "max_segments_per_cell",
        "max_synapses_per_segment",
    )

    #: Mapping from our snake_case names to the TM constructor kwargs.
    _TM_KWARG = {
        "activation_threshold":       "activationThreshold",
        "initial_permanence":         "initialPermanence",
        "connected_permanence":       "connectedPermanence",
        "min_threshold":              "minThreshold",
        "max_new_synapse_count":      "maxNewSynapseCount",
        "permanence_increment":       "permanenceIncrement",
        "permanence_decrement":       "permanenceDecrement",
        "predicted_segment_decrement": "predictedSegmentDecrement",
        "max_segments_per_cell":      "maxSegmentsPerCell",
        "max_synapses_per_segment":   "maxSynapsesPerSegment",
    }

    def __init__(
        self,
        sdr_size: int,
        cells_per_column: int = 8,
        activation_threshold: int = 13,
        initial_permanence: float = 0.21,
        connected_permanence: float = 0.5,
        min_threshold: int = 10,
        max_new_synapse_count: int = 20,
        permanence_increment: float = 0.1,
        permanence_decrement: float = 0.1,
        predicted_segment_decrement: float = 0.0,
        max_segments_per_cell: int = 255,
        max_synapses_per_segment: int = 255,
        seed: int = 42,
    ):
        self.sdr_size = int(sdr_size)
        if self.sdr_size <= 0:
            raise ValueError("sdr_size must be positive")
        self.cells_per_column = int(cells_per_column)
        self.seed = int(seed)

        # Store the tunable TM hyper-parameters for persistence.
        self.params = {
            "activation_threshold":        int(activation_threshold),
            "initial_permanence":          float(initial_permanence),
            "connected_permanence":        float(connected_permanence),
            "min_threshold":               int(min_threshold),
            "max_new_synapse_count":       int(max_new_synapse_count),
            "permanence_increment":        float(permanence_increment),
            "permanence_decrement":        float(permanence_decrement),
            "predicted_segment_decrement": float(predicted_segment_decrement),
            "max_segments_per_cell":       int(max_segments_per_cell),
            "max_synapses_per_segment":    int(max_synapses_per_segment),
        }

        self.tm = self._build_tm()
        self.iteration = 0

        # Reusable input SDR (avoids re-allocating tens of thousands of bits
        # on every step).
        self._input_sdr = SDR(self.sdr_size)

    # ---------- Construction helpers ----------
    def _build_tm(self) -> TM:
        kwargs = {self._TM_KWARG[k]: self.params[k] for k in self._TM_PARAM_KEYS}
        return TM(
            columnDimensions=(self.sdr_size,),
            cellsPerColumn=self.cells_per_column,
            seed=self.seed,
            **kwargs,
        )

    # ---------- Properties ----------
    @property
    def n_columns(self) -> int:
        return int(self.tm.numberOfColumns())

    @property
    def n_cells(self) -> int:
        return int(self.tm.numberOfCells())

    def __repr__(self) -> str:
        return (f"PlaceContextMemory(sdr_size={self.sdr_size}, "
                f"cells_per_column={self.cells_per_column}, "
                f"n_cells={self.n_cells}, iteration={self.iteration})")

    # ---------- Core API ----------
    def compute(
        self,
        sdr: np.ndarray,
        learn: bool = True,
        reset: bool = False,
    ) -> ContextState:
        """
        Advance the TM by one step.

        Parameters
        ----------
        sdr : np.ndarray
            Encoder SDR (0/1). Its length must equal ``self.sdr_size``.
        learn : bool
            Enable learning (``True`` while building the map, ``False`` during
            pure localization).
        reset : bool
            Reset the TM sequence state before this step (use at the start of
            a new trajectory/episode).

        Returns
        -------
        ContextState
            Winner/active/predictive cells and the anomaly for this step.
        """
        sdr = np.asarray(sdr).ravel()
        if sdr.size != self.sdr_size:
            raise ValueError(
                f"SDR size mismatch: got {sdr.size}, expected {self.sdr_size}"
            )

        active_idx = np.flatnonzero(sdr).astype(np.uint32)

        if reset:
            self.tm.reset()

        # Present the SDR as active mini-columns. The sparse setter keeps this
        # cheap even for tens of thousands of columns.
        self._input_sdr.sparse = active_idx
        self.tm.compute(self._input_sdr, learn=learn)
        # compute() already called activateDendrites() *before* activateCells;
        # calling it again makes getPredictiveCells() reflect the NEXT step.
        self.tm.activateDendrites(learn)

        winner = self.tm.getWinnerCells().sparse.astype(np.uint32, copy=True)
        active = self.tm.getActiveCells().sparse.astype(np.uint32, copy=True)
        predicted = self.tm.getPredictiveCells().sparse.astype(np.uint32, copy=True)

        state = ContextState(
            iteration=self.iteration,
            anomaly=float(self.tm.anomaly),
            winner_cells=winner,
            active_cells=active,
            predictive_cells=predicted,
        )
        self.iteration += 1
        return state

    def reset(self) -> None:
        """Reset the TM sequence state (start of a new episode)."""
        self.tm.reset()

    # ---------- Persistence ----------
    def save(self, prefix: str | Path) -> None:
        """
        Persist the TM and its metadata.

        Writes ``<prefix>_tm.bin`` (binary TM snapshot) and
        ``<prefix>_context_meta.json`` (hyper-parameters + iteration counter).
        """
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        self.tm.saveToFile(str(prefix.with_name(prefix.name + "_tm.bin")), "BINARY")

        meta = {
            "sdr_size":        self.sdr_size,
            "cells_per_column": self.cells_per_column,
            "seed":            self.seed,
            "iteration":       self.iteration,
            "n_columns":       self.n_columns,
            "n_cells":         self.n_cells,
            "params":          self.params,
        }
        prefix.with_name(prefix.name + "_context_meta.json").write_text(
            json.dumps(meta, indent=2)
        )

    @classmethod
    def load(cls, prefix: str | Path) -> "PlaceContextMemory":
        """Rebuild a memory previously written by :meth:`save`."""
        prefix = Path(prefix)
        meta_path = prefix.with_name(prefix.name + "_context_meta.json")
        tm_path = prefix.with_name(prefix.name + "_tm.bin")

        if not meta_path.exists() or not tm_path.exists():
            raise FileNotFoundError(
                f"Missing context files for prefix '{prefix}' "
                f"({meta_path.name}, {tm_path.name})"
            )

        meta = json.loads(meta_path.read_text())
        obj = cls(
            sdr_size=meta["sdr_size"],
            cells_per_column=meta.get("cells_per_column", 8),
            seed=meta.get("seed", 42),
            **meta.get("params", {}),
        )

        # Replace the freshly-built TM with the serialized one.
        obj.tm = TM()
        obj.tm.loadFromFile(str(tm_path), "BINARY")
        obj.iteration = int(meta.get("iteration", 0))
        return obj

    # ---------- Context comparison helpers ----------
    @staticmethod
    def overlap(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> int:
        """Number of shared cell ids between two context sets."""
        return ContextState.overlap(a, b)

    @staticmethod
    def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Jaccard similarity (0..1) between two context sets."""
        return ContextState.jaccard(a, b)
