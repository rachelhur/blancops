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
from blancops.data.constants import FILTER2IDX
from blancops.math import units

import logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LookupTables:
    """The universal container for telescope/survey metadata."""
    fid2name: dict
    fid2radec: np.array
    fid2filters: np.array
    target_fidfilt_counts: np.array
    target_filt_counts: np.array
    target_fid_counts: np.array
    dir: Path
    # Add optional training-specific fields
    night2fid_visit_hist: Optional[dict] = None
    night2fidfilt_visit_hist: Optional[dict] = None
    # Add required deployment-specific fields
    fields_table: Optional[pd.DataFrame] = None
    

    @classmethod
    def load_from_dir(cls, data_dir: Path, is_historic: bool = False, is_training: bool = False, overrides: Dict[LookupKeys, str] = None):
        """Loads lookups from disk. Overrides allow custom filenames. `is_historic` is for historic data and `is_training` is for training data."""
        overrides = overrides or {}

        def get_path(key: LookupKeys) -> Path:
            filename = overrides.get(key, key.value)
            return data_dir / filename
            
        # MAPPINGS
        with open(get_path(LookupKeys.FID2NAME), "r") as f:
            f2name = {int(k): v for k, v in json.load(f).items()}
            
        with open(get_path(LookupKeys.FID2RADEC), "rb") as f:
            f2radec = pickle.load(f)
            
        with open(get_path(LookupKeys.FID2FILTERS), "rb") as f:
            f2filts = pickle.load(f)
        
        # TARGETS
        with open(get_path(LookupKeys.TARGET_FIDFILT_COUNTS), "rb") as f:
            ff2max = pickle.load(f)
            
        with open(get_path(LookupKeys.TARGET_FILT_COUNTS), "rb") as f:
            target_filt_counts = pickle.load(f)
        
        # VISIT COUNTS (Handling JSON vs Pickle appropriately)
        visit_key = LookupKeys.TARGET_FID2VISITS_TRAIN if is_training else LookupKeys.TARGET_FID2VISITS_EVAL
        visit_path = get_path(visit_key)
        
        if visit_path.suffix == '.json':
            with open(visit_path, "r") as f:
                f2max = json.load(f)
        else:
            with open(visit_path, "rb") as f:
                f2max = pickle.load(f)
        
        if not is_training:
            # FIELDS TABLE
            with open(get_path(LookupKeys.FIELDS_TABLE), "r") as f:
                fields_table = pd.read_json(f)
            if not is_historic:
                return cls(
                    fid2name=f2name, fid2radec=f2radec, fid2filters=f2filts, 
                    target_fid_counts=f2max, target_fidfilt_counts=ff2max, 
                    target_filt_counts=target_filt_counts, fields_table=fields_table, 
                    dir=data_dir
                )

        # SURVEY HISTORY
        with open(get_path(LookupKeys.NIGHT2FID_VISIT_HIST), "rb") as f:
            n2field = pickle.load(f)
            
        with open(get_path(LookupKeys.NIGHT2FIDFILT_VISIT_HIST), "rb") as f:
            n2fidfilt = pickle.load(f)
                
        return cls(
            fields_table=fields_table, dir=data_dir,
            fid2name=f2name, fid2radec=f2radec, fid2filters=f2filts, 
            target_fid_counts=f2max, target_fidfilt_counts=ff2max, target_filt_counts=target_filt_counts,
            night2fid_visit_hist=n2field, night2fidfilt_visit_hist=n2fidfilt
        )   
        
    def write_to_disk(self, outdir: Path):
        # FID2NAME
        with open(outdir / LookupKeys.FID2NAME.value, "w") as f:
            json.dump(self.fid2name, f)
    
        # FID2RADEC
        with open(outdir / LookupKeys.FID2RADEC.value, "wb") as f:
            pickle.dump(self.fid2radec, f)
    
        # FID2FILTERS
        with open(outdir / LookupKeys.FID2FILTERS.value, "wb") as f:
            pickle.dump(self.fid2filters, f)
    
        # TARGET_FIDFILT_COUNTS
        with open(outdir / LookupKeys.TARGET_FIDFILT_COUNTS.value, "wb") as f:
            pickle.dump(self.target_fidfilt_counts, f)
    
        # TARGET_FILT_COUNTS
        with open(outdir / LookupKeys.TARGET_FILT_COUNTS.value, "wb") as f:
            pickle.dump(self.target_filt_counts, f)
    
        # TARGET_FID_COUNTS
        with open(outdir / LookupKeys.TARGET_FID_COUNTS.value, "wb") as f:
            pickle.dump(self.target_fid_counts, f)
        
        # FIELDS LIST
        self.fields_table.to_json(outdir / LookupKeys.FIELDS_TABLE.value)

    def merge(self, new_lookups: "LookupTables", new_dir: Path = None) -> "LookupTables":
        """
        Appends entries from the new LookupTables object to the end of this one.
        Calculates the required index offset and applies it to all new dictionaries/arrays.
        Updates the directory if new_dir is provided.
        """
        offset = max(self.fid2name.keys()) + 1 if self.fid2name else 0

        merged_f2name = self.fid2name.copy()
        merged_f2radec = self.fid2radec.copy()
        merged_f2max = self.fid2maxvisits.copy()

        for k, v in new_lookups.fid2name.items():
            merged_f2name[k + offset] = v
        for k, v in new_lookups.fid2radec.items():
            merged_f2radec[k + offset] = v
        for k, v in new_lookups.fid2maxvisits.items():
            merged_f2max[k + offset] = v

        merged_ff2max = np.vstack((self.fieldfilter2maxvisits, new_lookups.fieldfilter2maxvisits))
        if new_dir:
            self.dir = new_dir

        return LookupTables(
            fid2name=merged_f2name,
            fid2radec=merged_f2radec,
            target_fid_counts=merged_f2max,
            target_fidfilt_counts=merged_ff2max,
            fields_table=self.fields_table.append(new_lookups.fields_table, ignore_index=True),
            # Pass training stats through completely unmodified 
            # (ToOs don't have historical survey progress)
            night2fieldvisithistory=self.night2fieldvisithistory,
            night2filtervisithistory=self.night2filtervisithistory,
            target_filter_counts=self.target_filter_counts,
            dir=self.dir
        )
        
    @staticmethod
    def generate_lookups_from_fields(fields_path, outdir: Path = None, write_to_disk: bool = False):
        """
        Generates a LookupTables object from a .json fields file.
        Fields file must have the following columns:
            ra, dec, filter, count, exptime, propid
        
        """
        assert (not write_to_disk and outdir is None) or (write_to_disk and outdir is not None), "Must specify `data_dir` if `write_to_disk` is True"
        outdir = Path(outdir).resolve()
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(fields_path) as f:
            fields_df = pd.read_json(f)
            fields_df.columns = fields_df.columns.str.lower()

            # CONVERT RADIANS TO DEGREES - NEED BETTER CHECK!
            if any(fields_df['ra'] > 2 * np.pi) or any(fields_df['dec'] > 2 * np.pi):
                fields_df.loc[:, ['ra', 'dec']] *= units.deg
                
        df = fields_df.copy()
        
        
        assert set(['ra', 'dec', 'filter', 'count', 'exptime', 'propid']).issubset(df.columns), f"Missing columns: {set(['ra', 'dec', 'filter', 'count', 'exptime', 'propid', 'comment']) - set(df.columns)}"
            
        # ASSIGN "OBJECT" - USE `field_name` OR `FIELDNAME`, OTHERWISE ASSIGN `FIELD_{FIELD_ID}`
        if 'object' in df.columns:
            pass
        elif 'field_name' in df.columns:
            df.rename(columns={'field_name': 'object'}, inplace=True)
        elif 'fieldname' in df.columns:
            df.rename(columns={'fieldname': 'object'}, inplace=True)
        else:
            # Create a new `object` column
            df['object'] = 'field_' + df.groupby(['ra', 'dec'], sort=False).ngroup().astype(str)

        # DROP DUPLICATES
        df = df.drop_duplicates(subset=['object', 'filter'])
            
        # ASSIGN FIELD IDS - REWRITE IF MAPPPING NOT UNIQUE
        if 'field_id' in df.columns:
            is_unique_mapping = df['field_id'].nunique() == len(df['field_id'])
        if 'field_id' not in df.columns or not is_unique_mapping:
            df['field_id'] = pd.factorize(df['object'])[0]

        # ASSIGN FILTER IDX
        df['filter_idx'] = df['filter'].map(FILTER2IDX).fillna(-1)
        assert all(df['filter_idx'] != -1)
        
        # CONSTRUCT LOOKUP TABLES
        fid2name = dict(zip(df.field_id.to_list(), df.object.values))
        fid2radec = pd.Series(list(zip(df['ra'], df['dec'])), index=df['field_id']).values
        fid2filter = pd.Series(df['filter'], index=df['field_id']).values
        pivot_df = df.pivot_table(index='field_id', columns='filter_idx', values='count', aggfunc='sum').fillna(0)
        full_columns = range(len(FILTER2IDX))
        pivot_df = pivot_df.reindex(columns=full_columns, fill_value=0)
        target_fidfilt_counts = pivot_df.to_numpy()
        target_filt_counts = df.groupby('filter_idx')['count'].sum()
        target_fid_counts = df.groupby('field_id')['count'].sum()

        lookups = LookupTables(
            dir=outdir, fields_table=fields_df,
            fid2name=fid2name, fid2radec=fid2radec, fid2filters=fid2filter, 
            target_fid_counts=target_fid_counts, target_fidfilt_counts=target_fidfilt_counts, target_filt_counts=target_filt_counts,
            )

        if write_to_disk:
            lookups.write_to_disk(outdir)
            
        return lookups