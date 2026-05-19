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
 
    **Shape contract: `*_fidfilt_*` are of shape `(len(fields), len(FILTER2IDX))`,
        indexed by `field_id` along axis 0 and `filter_idx` along axis 1.
        The `fields` index must be `0..N-1` contiguous so array index and `field_id` coincide;
        `__post_init__` enforces this.
        #TODO do I want to change from contiguous to saving an additional fid->idx mapping?

    Visit-history dicts (set when ``is_historic=True``) are snapshots
    taken at the START of each observing night, derived from the FULL
    survey history regardless of which subset a downstream training run
    selects. Two parallel families exist:

    - ``night2{fid,fidfilt}_visit_hist``: integer running counts.
    - ``night2{fid,fidfilt}_last_visit``: number of observing time seconds that have passed since its last visit; 
        NaN means "no recorded visit so far".

        (if `is_historic` AND present on disk — older lookup dirs may predate these fields, 
            in which case they're loaded as None and downstream code falls back to the
            "no recorded visit" sentinel)
    - `total_ot_sec`
        
    Both are 1-D over fields or 2-D over (field, filter) respectively.
    Quality gating (``teff > 0.3``) must match between the two so the
    "what counted as a visit" semantics agree.
    """
    # Required lookup tables
    fields: pd.DataFrame                # index=field_id; required cols: name, ra, dec
    target_fidfilt_counts: np.ndarray   # (nfields, nfilters) int
    fidfilt_exptime: np.ndarray         # (nfields, nfilters) float
    dir: Path
    
    # Required lookup tables if is_historic is true
    night2fid_visit_hist: Optional[dict] = None
    night2fidfilt_visit_hist: Optional[dict] = None
    night2fid_last_visit_ts: Optional[dict] = None
    night2fidfilt_last_visit_ts: Optional[dict] = None
    night2fid_last_visit_ot: Optional[dict] = None
    night2fidfilt_last_visit_ot: Optional[dict] = None
    night2ot_clock_seconds: Optional[dict] = None
    total_ot_sec: Optional[float] = None
 
    # Derived marginals — populated in __post_init__, never set by callers.
    target_fid_counts: np.ndarray = field(init=False, default=None, repr=False)
    target_filt_counts: np.ndarray = field(init=False, default=None, repr=False)
    night2idx: Optional[dict] = None
    total_nights: Optional[int] = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
 
    @classmethod
    def load_from_dir(
        cls,
        data_dir: Path,
        include_historic: bool = False,
        overrides: Optional[Dict[LookupKeys, str]] = None,
    ) -> "LookupTables":
        """Load lookups from a directory.
        """
        overrides = overrides or {}
        data_dir = Path(data_dir).resolve()
        cls_kwargs = {}
 
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
 
        # Visit history
        night2fid_visit_hist = None
        night2fidfilt_visit_hist = None
        night2fid_last_visit_ts = None
        night2fidfilt_last_visit_ts = None
        night2fid_last_visit_ot = None
        night2fidfilt_last_visit_ot = None
        night2ot_clock_seconds = None
        night2idx = None
        total_ot_sec = 0
        total_nights = 0
        
        if include_historic:
            with open(get_path(LookupKeys.NIGHT2FID_VISIT_HIST), "rb") as f:
                night2fid_visit_hist = pickle.load(f)
            with open(get_path(LookupKeys.NIGHT2FIDFILT_VISIT_HIST), "rb") as f:
                night2fidfilt_visit_hist = pickle.load(f)
            with open(get_path(LookupKeys.NIGHT2OT_CLOCK_SECONDS), "rb") as f:
                night2ot_clock_seconds = pickle.load(f)
            with open(get_path(LookupKeys.TOTAL_OT_SECONDS), "r") as f:
                total_ot_sec = np.float64(f.read())
            # with open(get_path(LookupKeys.NIGHT2IDX), "rb") as f:
            #     night2idx = pickle.load(f)
            # with open(get_path(LookupKeys.TOTAL_NIGHTS), "r") as f:
            #     total_nights = int(f.read())
            
            # Last-visit timestamps
            fid_lv_path = get_path(LookupKeys.NIGHT2FID_LAST_VISIT_TS)
            ff_lv_path = get_path(LookupKeys.NIGHT2FIDFILT_LAST_VISIT_TS)
            if fid_lv_path.exists():
                with open(fid_lv_path, "rb") as f:
                    night2fid_last_visit_ts = pickle.load(f)
            else:
                logger.warning(
                    f"{fid_lv_path.name} not found in {data_dir}; "
                    f"t_since_last_visit will start from sentinel for every "
                    f"field. Rebuild lookups to enable staleness seeding."
                )
            if ff_lv_path.exists():
                with open(ff_lv_path, "rb") as f:
                    night2fidfilt_last_visit_ts = pickle.load(f)
            else:
                logger.warning(
                    f"{ff_lv_path.name} not found in {data_dir}; "
                    f"per-filter t_since_last_visit will start from sentinel."
                )
                
            # Last-visit dicts in ot
            fid_lv_path = get_path(LookupKeys.NIGHT2FID_LAST_VISIT_OT)
            ff_lv_path = get_path(LookupKeys.NIGHT2FIDFILT_LAST_VISIT_OT)
            if fid_lv_path.exists():
                with open(fid_lv_path, "rb") as f:
                    night2fid_last_visit_ot = pickle.load(f)
            else:
                logger.warning(
                    f"{fid_lv_path.name} not found in {data_dir}; "
                    f"t_since_last_visit will start from sentinel for every "
                    f"field. Rebuild lookups to enable staleness seeding."
                )
            if ff_lv_path.exists():
                with open(ff_lv_path, "rb") as f:
                    night2fidfilt_last_visit_ot = pickle.load(f)
            else:
                logger.warning(
                    f"{ff_lv_path.name} not found in {data_dir}; "
                    f"per-filter t_since_last_visit will start from sentinel."
                )
            
 
        return cls(
            fields=fields,
            target_fidfilt_counts=target_fidfilt_counts,
            fidfilt_exptime=fidfilt_exptime,
            dir=data_dir,
            night2fid_visit_hist=night2fid_visit_hist,
            night2fidfilt_visit_hist=night2fidfilt_visit_hist,
            night2fid_last_visit_ts=night2fid_last_visit_ts,
            night2fidfilt_last_visit_ts=night2fidfilt_last_visit_ts,
            night2fid_last_visit_ot=night2fid_last_visit_ot,
            night2fidfilt_last_visit_ot=night2fidfilt_last_visit_ot,
            night2ot_clock_seconds=night2ot_clock_seconds,
            total_ot_sec=total_ot_sec,
        )
 
    def write_to_disk(self, outdir: Optional[Path] = None) -> None:
        """Persist non-derived state. Marginals are recomputed on load."""
        outdir = Path(outdir if outdir is not None else self.dir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Round-trip the index by resetting it as a column before save.
        self.fields.reset_index().to_json(outdir / LookupKeys.FIELDS.value, orient='records')
 
        # TARGET COUNTS
        with open(outdir / LookupKeys.TARGET_FIDFILT_COUNTS.value, "wb") as f:
            pickle.dump(self.target_fidfilt_counts, f)
        # EXPOSURE TIME PER (FIELD, FILTER) PAIR
        with open(outdir / LookupKeys.FIDFILT_EXPTIME.value, "wb") as f:
            pickle.dump(self.fidfilt_exptime, f)
        
        # VISIT HISTORY
        if self.night2fid_visit_hist is not None:
            with open(outdir / LookupKeys.NIGHT2FID_VISIT_HIST.value, "wb") as f:
                pickle.dump(self.night2fid_visit_hist, f)
        if self.night2fidfilt_visit_hist is not None:
            with open(outdir / LookupKeys.NIGHT2FIDFILT_VISIT_HIST.value, "wb") as f:
                pickle.dump(self.night2fidfilt_visit_hist, f)

        # LAST VISIT TIMESTAMP
        if self.night2fid_last_visit_ts is not None:
            with open(outdir / LookupKeys.NIGHT2FID_LAST_VISIT_TS.value, "wb") as f:
                pickle.dump(self.night2fid_last_visit_ts, f)
        if self.night2fidfilt_last_visit_ts is not None:
            with open(outdir / LookupKeys.NIGHT2FIDFILT_LAST_VISIT_TS.value, "wb") as f:
                pickle.dump(self.night2fidfilt_last_visit_ts, f)
        
        # LAST VISIT TIME IN UNITS OBSERVING TIME
        if self.night2fid_last_visit_ot is not None:
            with open(outdir / LookupKeys.NIGHT2FID_LAST_VISIT_OT.value, "wb") as f:
                pickle.dump(self.night2fid_last_visit_ot, f)
        if self.night2fidfilt_last_visit_ts is not None:
            with open(outdir / LookupKeys.NIGHT2FIDFILT_LAST_VISIT_OT.value, "wb") as f:
                pickle.dump(self.night2fidfilt_last_visit_ot, f)
        
        # TOTAL OBSERVABLE SECONDS IN SURVEY
        with open(outdir / LookupKeys.NIGHT2OT_CLOCK_SECONDS.value, "wb") as f:
            pickle.dump(self.night2ot_clock_seconds, f)
        if self.total_ot_sec is not None:
            with open(outdir / LookupKeys.TOTAL_OT_SECONDS.value, "w") as f:
                f.write(f"{self.total_ot_sec}")
                
        # NIGHT INDICES
        # with open(outdir / LookupKeys.NIGHT2IDX.value, "wb") as f:
        #     pickle.dump(self.night2idx, f)
        # with open(outdir / LookupKeys.TOTAL_NIGHTS.value, "w") as f:
        #     f.write(f"{self.total_nights}")
                
    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
 
    def merge(
        self,
        new_lookups: "LookupTables",
        new_dir: Optional[Path] = None,
    ) -> "LookupTables":
        """Append new field targets to existing lookup table, returning a new LookupTables.
 
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
            night2fid_last_visit_ts=self.night2fid_last_visit_ts,
            night2fidfilt_last_visit_ts=self.night2fidfilt_last_visit_ts,
            night2fid_last_visit_ot=self.night2fid_last_visit_ot,
            night2fidfilt_last_visit_ot=self.night2fidfilt_last_visit_ot,
            total_ot_sec=self.total_ot_sec
        )
 
    # ------------------------------------------------------------------
    # Construction from raw fields file
    # ------------------------------------------------------------------
 
    @staticmethod
    def build_lookups_from_fields(
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
            .to_numpy(dtype=np.int64)
        )
        fidfilt_exptime = (
            df.pivot_table(
                index="field_id", columns="filter_idx",
                values="exptime", aggfunc="first",
            )
            .reindex(index=fields.index, columns=range(nfilters), fill_value=0)
            .to_numpy(dtype=np.int64)
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
        raise NotImplementedError
        return lookups


    # ------------------------------------------------------------------
    # Validation + derived attributes
    # ------------------------------------------------------------------
 
    def __post_init__(self):
        # Coerce arrays to canonical dtype/layout
        object.__setattr__(
            self, "target_fidfilt_counts",
            np.ascontiguousarray(self.target_fidfilt_counts, dtype=np.int64),
        )
        object.__setattr__(
            self, "fidfilt_exptime",
            np.ascontiguousarray(self.fidfilt_exptime, dtype=np.float64),
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

        # Validate visit-history shapes when present. Either both families
        # are populated together (built from the same loop) or both are
        # absent (non-historic lookups).
        self._validate_history_shapes(nfields, nfilters)
 
        # Compute marginals
        object.__setattr__(
            self, "target_fid_counts", self.target_fidfilt_counts.sum(axis=1)
        )
        object.__setattr__(
            self, "target_filt_counts", self.target_fidfilt_counts.sum(axis=0)
        )
        object.__setattr__(
            self, "night2idx", {night: i for i, night in enumerate(self.night2fidfilt_visit_hist.keys())}
        )
        object.__setattr__(
            self, "total_nights", len(self.night2idx)
        )
        
    def _validate_history_shapes(self, nfields, nfilters):
        """Per-night snapshot dicts must have matching keys and the
        expected per-field / per-(field, filter) shapes. Treats the
        last-visit dicts as optional even when visit-hist is present,
        for backward-compat with lookup directories written before the
        last-visit fields existed.
        """
        if self.night2fid_visit_hist is not None:
            for night, arr in self.night2fid_visit_hist.items():
                if arr.shape != (nfields,):
                    raise ValueError(
                        f"night2fid_visit_hist[{night!r}] has shape "
                        f"{arr.shape}; expected ({nfields},)"
                    )
        if self.night2fidfilt_visit_hist is not None:
            for night, arr in self.night2fidfilt_visit_hist.items():
                if arr.shape != (nfields, nfilters):
                    raise ValueError(
                        f"night2fidfilt_visit_hist[{night!r}] has shape "
                        f"{arr.shape}; expected ({nfields}, {nfilters})"
                    )
        if self.night2fid_last_visit_ts is not None:
            for night, arr in self.night2fid_last_visit_ts.items():
                if arr.shape != (nfields,):
                    raise ValueError(
                        f"night2fid_last_visit[{night!r}] has shape "
                        f"{arr.shape}; expected ({nfields},)"
                    )
        if self.night2fidfilt_last_visit_ts is not None:
            for night, arr in self.night2fidfilt_last_visit_ts.items():
                if arr.shape != (nfields, nfilters):
                    raise ValueError(
                        f"night2fidfilt_last_visit_ts[{night!r}] has shape "
                        f"{arr.shape}; expected ({nfields}, {nfilters})"
                    )
        # Key-set agreement between parallel dicts: if both members of a
        # pair are present, they should snapshot the same nights.
        def _check_keys(a, a_name, b, b_name):
            if a is not None and b is not None:
                if set(a.keys()) != set(b.keys()):
                    raise ValueError(
                        f"{a_name} and {b_name} have mismatched night keys: "
                        f"symmetric_diff="
                        f"{set(a.keys()).symmetric_difference(set(b.keys()))}"
                    )
        _check_keys(
            self.night2fid_visit_hist, "night2fid_visit_hist",
            self.night2fid_last_visit_ts, "night2fid_last_visit_ts",
        )
        _check_keys(
            self.night2fidfilt_visit_hist, "night2fidfilt_visit_hist",
            self.night2fidfilt_last_visit_ts, "night2fidfilt_last_visit_ts",
        )
