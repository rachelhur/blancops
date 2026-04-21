from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import json
import pickle

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np
from blancops.configs.enums import LookupKeys

@dataclass(frozen=True)
class LookupTables:
    """The universal container for telescope/survey metadata."""
    fid2name: dict
    fid2radec: np.array
    fid2filters: np.array
    target_fidfilt_counts: np.array
    target_filt_counts: np.array
    target_fid_counts: np.array
    # Add optional training-specific fields
    night2fid_visit_hist: Optional[dict] = None
    night2fidfilt_visit_hist: Optional[dict] = None

    @classmethod
    def load_from_dir(cls, data_dir: Path, is_historic: bool = True, is_training: bool = True, overrides: Dict[LookupKeys, str] = None):
        """Loads lookups from disk. Overrides allow custom filenames (e.g., for ToO)."""
        overrides = overrides or {}

        def get_path(key: LookupKeys) -> Path:
            filename = overrides.get(key, key.value)
            return data_dir / filename

        # MAPPINGS
        with open(get_path(LookupKeys.FID2NAME)) as f:
            f2name = {int(k): v for k, v in json.load(f).items()}
        with open(get_path(LookupKeys.FID2RADEC)) as f:
            f2radec = {int(k): v for k, v in json.load(f).items()}
        with open(get_path(LookupKeys.FID2FILTERS), "rb") as f:
            f2filts = pickle.load(f)
        
        # TARGETS
        with open(get_path(LookupKeys.TARGET_FIDFILT_COUNTS), "rb") as f:
            ff2max = pickle.load(f)
        with open(get_path(LookupKeys.TARGET_FILT_COUNTS), "rb") as f:
            target_filt_counts = pickle.load(f)
        
        visit_key = LookupKeys.TARGET_FID2VISITS_TRAIN if is_training else LookupKeys.TARGET_FID2VISITS_EVAL
        with open(get_path(visit_key)) as f:
            f2max = {int(k): v for k, v in json.load(f).items()}
        
        if not is_historic:
            return cls(fid2name=f2name, fid2radec=f2radec, 
                       fid2filters=f2filts, target_fid_counts=f2max, target_fidfilt_counts=ff2max, target_filt_counts=target_filt_counts)

        # SURVEY HISTORY
        with open(get_path(LookupKeys.NIGHT2FID_VISIT_HIST), "rb") as f:
            n2field = pickle.load(f)
        with open(get_path(LookupKeys.NIGHT2FIDFILT_VISIT_HIST), "rb") as f:
            n2fidfilt = pickle.load(f)

        return cls(
            fid2name=f2name, fid2radec=f2radec, fid2filters=f2filts, target_fid_counts=f2max, target_fidfilt_counts=ff2max, target_filt_counts=target_filt_counts,
            night2fid_visit_hist=n2field, night2fidfilt_visit_hist=n2fidfilt
        )

    def merge(self, new_fields: "LookupTables") -> "LookupTables":
        """
        Appends a new LookupTables object to the end of this one.
        Calculates the required index offset and applies it to all new dictionaries/arrays.
        """
        offset = max(self.fid2name.keys()) + 1 if self.fid2name else 0

        merged_f2name = self.fid2name.copy()
        merged_f2radec = self.fid2radec.copy()
        merged_f2max = self.fid2maxvisits.copy()

        for k, v in new_fields.fid2name.items():
            merged_f2name[k + offset] = v
        for k, v in new_fields.fid2radec.items():
            merged_f2radec[k + offset] = v
        for k, v in new_fields.fid2maxvisits.items():
            merged_f2max[k + offset] = v

        merged_ff2max = np.vstack((self.fieldfilter2maxvisits, new_fields.fieldfilter2maxvisits))

        return LookupTables(
            fid2name=merged_f2name,
            fid2radec=merged_f2radec,
            fid2maxvisits=merged_f2max,
            fieldfilter2maxvisits=merged_ff2max,
            # Pass training stats through completely unmodified 
            # (ToOs don't have historical survey progress)
            night2fieldvisithistory=self.night2fieldvisithistory,
            night2filtervisithistory=self.night2filtervisithistory,
            target_filter_counts=self.target_filter_counts
        )
