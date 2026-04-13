from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import pickle

import numpy as np
from blancops.configs.constants import TRAIN_DATA_DIR, LOOKUPS

@dataclass
class LookupTables:
    field2name: dict
    field2radec: np.array
    field2maxvisits: np.array
    night2fieldvisithistory: np.array
    night2filtervisithistory: np.array
    fieldfilter2maxvisits: np.array
    target_filter_counts: np.array

def load_lookup_tables(data_dir: Path = TRAIN_DATA_DIR) -> LookupTables:
    """Load all lookup tables from disk. `overrides` lets callers substitute paths."""

    with open(data_dir / LOOKUPS["FIELD2NAME"]) as f:
        field2name = json.load(f)
    with open(data_dir / LOOKUPS["FIELD2RADEC"]) as f:
        field2radec = {int(k): v for k, v in json.load(f).items()}
    with open(data_dir / LOOKUPS["FIELD2MAXVISITS_TRAIN"]) as f:
        field2maxvisits = {int(k): v for k, v in json.load(f).items()}
    with open(data_dir / LOOKUPS["NIGHT2FIELDVISITS"], "rb") as f:
        night2fieldvisits = pickle.load(f)
    with open(data_dir / LOOKUPS["NIGHT2FILTERVISITS"], "rb") as f:
        night2filtervisithistory = pickle.load(f)
    with open(data_dir / LOOKUPS["FIELDFILTER2MAXVISITS"], "rb") as f:
        fieldfilter2maxvisits = pickle.load(f)
    with open(data_dir / LOOKUPS["FILTER_TARGET_COUNTS"], "rb") as f:
        target_filter_counts = pickle.load(f)

    return LookupTables(
        field2name=field2name,
        field2radec=field2radec,
        field2maxvisits=field2maxvisits,
        night2fieldvisithistory=night2fieldvisits,
        night2filtervisithistory=night2filtervisithistory,
        fieldfilter2maxvisits=fieldfilter2maxvisits,
        target_filter_counts=target_filter_counts
    )