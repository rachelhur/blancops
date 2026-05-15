from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Optional
import json
import pickle

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from blancops.configs.enums import LookupKeys
from blancops.configs.constants import FILTER2IDX
from blancops.math import units

import logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LookupTables:
    """Universal container for telescope/survey metadata.
 
    Shape contract: `target_fidfilt_counts` and `fidfilt_exptime` are
    both `(len(fields), len(FILTER2IDX))`, indexed by `field_id` along
    axis 0 and `filter_idx` along axis 1. The `fields` index must be
    `0..N-1` contiguous so array index and `field_id` coincide;
    `__post_init__` enforces this.
    """
    fields: pd.DataFrame                # index=field_id; required cols: name, ra, dec
    target_fidfilt_counts: np.ndarray   # (nfields, nfilters) int
    fidfilt_exptime: np.ndarray         # (nfields, nfilters) float
    dir: Path
    night2fid_visit_hist: Optional[dict] = None
    night2fidfilt_visit_hist: Optional[dict] = None
 
    # Derived marginals — populated in __post_init__, never set by callers.
    target_fid_counts: np.ndarray = field(init=False, default=None, repr=False)
    target_filt_counts: np.ndarray = field(init=False, default=None, repr=False)

    
    # ------------------------------------------------------------------
    # Validation + derived attributes
    # ------------------------------------------------------------------
 
    def __post_init__(self):
        # Coerce arrays to canonical dtype/layout
        object.__setattr__(
            self, "target_fidfilt_counts",
            np.ascontiguousarray(self.target_fidfilt_counts, dtype=np.int32),
        )
        object.__setattr__(
            self, "fidfilt_exptime",
            np.ascontiguousarray(self.fidfilt_exptime, dtype=np.float32),
        )
 
        # Validate fields index is 0..N-1 contiguous (required so that
        # `field_id` doubles as an array index in env code).
        expected_idx = pd.RangeIndex(len(self.fields))
        if not self.fields.index.equals(expected_idx):
            raise ValueError(
                "`fields` index must be 0..N-1 contiguous; got "
                f"min={self.fields.index.min()}, "
                f"max={self.fields.index.max()}, "
                f"len={len(self.fields)}, "
                f"unique={self.fields.index.nunique()}"
            )
 
        # Validate matrix shapes
        nfields, nfilters = self.target_fidfilt_counts.shape
        if nfields != len(self.fields):
            raise ValueError(
                f"target_fidfilt_counts has {nfields} rows but `fields` has "
                f"{len(self.fields)}"
            )
        if nfilters != len(FILTER2IDX):
            raise ValueError(
                f"target_fidfilt_counts has {nfilters} filter columns but "
                f"FILTER2IDX defines {len(FILTER2IDX)}"
            )
        if self.fidfilt_exptime.shape != self.target_fidfilt_counts.shape:
            raise ValueError(
                f"fidfilt_exptime shape {self.fidfilt_exptime.shape} does not "
                f"match target_fidfilt_counts shape "
                f"{self.target_fidfilt_counts.shape}"
            )
 
        # Validate per-field columns
        required_cols = {"object", "ra", "dec"}
        missing = required_cols - set(self.fields.columns)
        if missing:
            raise ValueError(f"`fields` is missing required columns: {missing}")
 
        # Compute marginals
        object.__setattr__(
            self, "target_fid_counts", self.target_fidfilt_counts.sum(axis=1)
        )
        object.__setattr__(
            self, "target_filt_counts", self.target_fidfilt_counts.sum(axis=0)
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
 
    @classmethod
    def load_from_dir(
        cls,
        data_dir: Path,
        is_historic: bool = False,
        overrides: Optional[Dict[LookupKeys, str]] = None,
    ) -> "LookupTables":
        """Load lookups from a directory.
 
        Loads:
          - `fields` → `fields` DataFrame (index restored from
            `field_id` column if present)
          - `TARGET_FIDFILT_COUNTS` → 2D int array
          - `FIDFILT_EXPTIME` → 2D float array
          - `NIGHT2FID_VISIT_HIST`, `NIGHT2FIDFILT_VISIT_HIST` (if
            `is_historic`)
        """
        overrides = overrides or {}
        data_dir = Path(data_dir).resolve()
 
        def get_path(key):
            return data_dir / overrides.get(key, key.value)
 
        # Fields DataFrame (canonical per-field data)
        fields = pd.read_json(get_path(LookupKeys.FIELDS))
        if "field_id" in fields.columns:
            fields = fields.set_index("field_id")
        fields = fields.sort_index()
        fields.index.name = "field_id"
 
        # Per-(field, filter) matrices
        with open(get_path(LookupKeys.TARGET_FIDFILT_COUNTS), "rb") as f:
            target_fidfilt_counts = pickle.load(f)
        with open(get_path(LookupKeys.FIDFILT_EXPTIME), "rb") as f:
            fidfilt_exptime = pickle.load(f)
 
        # Historic visit history
        night2fid_visit_hist = None
        night2fidfilt_visit_hist = None
        if is_historic:
            with open(get_path(LookupKeys.NIGHT2FID_VISIT_HIST), "rb") as f:
                night2fid_visit_hist = pickle.load(f)
            with open(get_path(LookupKeys.NIGHT2FIDFILT_VISIT_HIST), "rb") as f:
                night2fidfilt_visit_hist = pickle.load(f)
 
        return cls(
            fields=fields,
            target_fidfilt_counts=target_fidfilt_counts,
            fidfilt_exptime=fidfilt_exptime,
            dir=data_dir,
            night2fid_visit_hist=night2fid_visit_hist,
            night2fidfilt_visit_hist=night2fidfilt_visit_hist,
        )
 
    def write_to_disk(self, outdir: Optional[Path] = None) -> None:
        """Persist non-derived state. Marginals are recomputed on load."""
        outdir = Path(outdir if outdir is not None else self.dir)
        outdir.mkdir(parents=True, exist_ok=True)
 
        # Round-trip the index by resetting it as a column before save.
        self.fields.reset_index().to_json(outdir / LookupKeys.FIELDS.value)
 
        with open(outdir / LookupKeys.TARGET_FIDFILT_COUNTS.value, "wb") as f:
            pickle.dump(self.target_fidfilt_counts, f)
        with open(outdir / LookupKeys.FIDFILT_EXPTIME.value, "wb") as f:
            pickle.dump(self.fidfilt_exptime, f)
 
        if self.night2fid_visit_hist is not None:
            with open(outdir / LookupKeys.NIGHT2FID_VISIT_HIST.value, "wb") as f:
                pickle.dump(self.night2fid_visit_hist, f)
        if self.night2fidfilt_visit_hist is not None:
            with open(outdir / LookupKeys.NIGHT2FIDFILT_VISIT_HIST.value, "wb") as f:
                pickle.dump(self.night2fidfilt_visit_hist, f)
 
    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
 
    def merge(
        self,
        new_lookups: "LookupTables",
        new_dir: Optional[Path] = None,
    ) -> "LookupTables":
        """Append `new_lookups` to this one, returning a new LookupTables.
 
        Field IDs in `new_lookups` are reindexed to start at
        `self.fields.index.max() + 1`, so the combined index stays
        contiguous from 0. Visit history dicts are taken from `self`;
        new entries (e.g. ToOs) are assumed not to have history.
        """
        offset = (self.fields.index.max() + 1) if len(self.fields) else 0
 
        new_fields = new_lookups.fields.copy()
        new_fields.index = new_fields.index + offset
        new_fields.index.name = self.fields.index.name
 
        merged_fields = pd.concat([self.fields, new_fields])
        merged_fidfilt_counts = np.vstack([
            self.target_fidfilt_counts,
            new_lookups.target_fidfilt_counts,
        ])
        merged_fidfilt_exptime = np.vstack([
            self.fidfilt_exptime,
            new_lookups.fidfilt_exptime,
        ])
 
        return LookupTables(
            fields=merged_fields,
            target_fidfilt_counts=merged_fidfilt_counts,
            fidfilt_exptime=merged_fidfilt_exptime,
            dir=new_dir if new_dir is not None else self.dir,
            night2fid_visit_hist=self.night2fid_visit_hist,
            night2fidfilt_visit_hist=self.night2fidfilt_visit_hist,
        )
 
    # ------------------------------------------------------------------
    # Construction from raw fields file
    # ------------------------------------------------------------------
 
    @staticmethod
    def generate_lookups_from_fields(
        fields_path: Path,
        outdir: Optional[Path] = None,
        write_to_disk: bool = False,
    ) -> "LookupTables":
        """Build a LookupTables from a JSON fields file.
 
        Required input columns: `ra`, `dec`, `filter`, `count`,
        `exptime`. A name column is recognized as one of `object`,
        `field_name`, or `fieldname`; if none is present, names are
        auto-generated from `(ra, dec)` groups.
 
        Inputs are interpreted as one row per (field, filter); rows
        with the same `(name, filter)` are deduplicated keeping the
        first.
        """
        if write_to_disk and outdir is None:
            raise ValueError("Must specify `outdir` if `write_to_disk` is True")
 
        fields_path = Path(fields_path)
        df = pd.read_json(fields_path)
        df.columns = df.columns.str.lower()
 
        required = {"ra", "dec", "filter", "count", "exptime"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
 
        # Heuristic: if any value exceeds 2π, the file is in degrees;
        # convert to radians.
        if (df["ra"] > 2 * np.pi).any() or (df["dec"] > 2 * np.pi).any():
            df.loc[:, ["ra", "dec"]] = df.loc[:, ["ra", "dec"]] * units.deg
 
        # Resolve name column from common aliases
        if "field_name" in df.columns:
            df = df.rename(columns={"field_name": "object"})
        elif "fieldname" in df.columns:
            df = df.rename(columns={"fieldname": "object"})
        else:
            df["object"] = (
                "field_"
                + df.groupby(["ra", "dec"], sort=False).ngroup().astype(str)
            )
 
        # Drop within-(name, filter) duplicates
        df = df.drop_duplicates(subset=["object", "filter"])
 
        # Assign field_id (factorize on name) unless an existing field_id
        # already gives a 1-1 mapping with name
        has_consistent_fid = (
            "field_id" in df.columns
            and df.groupby("object")["field_id"].nunique().max() == 1
            and df.groupby("field_id")["object"].nunique().max() == 1
        )
        if not has_consistent_fid:
            df["field_id"] = pd.factorize(df["object"])[0]
 
        # Filter idx
        df["filter_idx"] = df["filter"].map(FILTER2IDX).fillna(-1).astype(int)
        if (df["filter_idx"] == -1).any():
            bad = df.loc[df["filter_idx"] == -1, "filter"].unique()
            raise ValueError(f"Unknown filter(s): {list(bad)}")
 
        # Validate per-field columns are constant within each field_id —
        # surfaces malformed inputs that previously got silently
        # collapsed by drop_duplicates.
        per_field_cols = ["object", "ra", "dec"]
        for col in per_field_cols:
            counts = df.groupby("field_id")[col].nunique()
            if (counts > 1).any():
                bad = list(counts[counts > 1].index)
                raise ValueError(
                    f"Column {col!r} varies within a field_id; cannot "
                    f"deduplicate. Affected field_ids: {bad}"
                )
 
        # Per-field DataFrame
        fields = (
            df.drop_duplicates(subset=["field_id"])[["field_id", *per_field_cols]]
              .set_index("field_id")
              .sort_index()
        )
        fields.index.name = "field_id"
 
        # Per-(field, filter) matrices.
        # `aggfunc='first'` for exptime — values are unique after the
        # earlier dedup, but using 'first' rather than 'sum' surfaces
        # any leftover collision (instead of doubling the exposure).
        nfilters = len(FILTER2IDX)
        target_fidfilt_counts = (
            df.pivot_table(
                index="field_id", columns="filter_idx",
                values="count", aggfunc="sum",
            )
            .reindex(index=fields.index, columns=range(nfilters), fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        fidfilt_exptime = (
            df.pivot_table(
                index="field_id", columns="filter_idx",
                values="exptime", aggfunc="first",
            )
            .reindex(index=fields.index, columns=range(nfilters), fill_value=0)
            .to_numpy(dtype=np.float32)
        )
 
        resolved_dir = (
            Path(outdir).resolve() if outdir is not None
            else fields_path.parent.resolve()
        )
 
        lookups = LookupTables(
            fields=fields,
            target_fidfilt_counts=target_fidfilt_counts,
            fidfilt_exptime=fidfilt_exptime,
            dir=resolved_dir,
        )
 
        if write_to_disk:
            lookups.write_to_disk(Path(outdir))
 
        return lookups
