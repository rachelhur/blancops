

import json
import os
from pathlib import Path
import pickle
from turtle import pd

import numpy as np

from blancops.configs.enums import LookupKeys
from blancops.data.constants import FILTER2IDX
from blancops.data.features.normalizations import StateNormalizer, build_normalizer_kwargs, load_normalization_stats
from blancops.data.lookup import LookupTables
from blancops.environment.online_env import OnlineBlancoEnv
from blancops.math import units

def build_env(cfg, trained_model_dir, lookups, sun_el_lim=None, airmass_lim=None, t_start=None):
    z_score_stats, rel_norm_stats = load_normalization_stats(trained_model_dir)
    norm_kwargs = build_normalizer_kwargs(cfg.data.norm)
    global_normalizer = StateNormalizer(
        state_feature_names=cfg.data.global_features, 
        **norm_kwargs
    )
    bin_normalizer = StateNormalizer(
        state_feature_names=cfg.data.bin_features, 
        **norm_kwargs
    )
    env = OnlineBlancoEnv(
        cfg=cfg, 
        lookups=lookups,
        global_normalizer=global_normalizer,
        bin_normalizer=bin_normalizer,
        z_score_stats=z_score_stats, 
        rel_norm_stats=rel_norm_stats, 
    )
    return env

def generate_lookups_from_fields(df, data_dir: Path = None, write_to_disk: bool = False):
    assert (not write_to_disk and data_dir is None) or (write_to_disk and data_dir is not None), "Must specify `data_dir` if `write_to_disk` is True"
    data_dir = Path(data_dir).resolve()
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    df.columns = df.columns.str.lower()
    # CONVERT RADIANS TO DEGREES - NEED BETTER CHECK!
    if any(df['ra'] > 2 * np.pi) or any(df['dec'] > 2 * np.pi):
        df.loc[:, ['ra', 'dec']] *= units.deg
        
    assert ['ra', 'dec', 'filter', 'count', 'exptime', 'propid', 'comment'].issubset(df.columns), f"Missing columns: {set(['ra', 'dec', 'filter', 'count', 'exptime', 'propid', 'comment']) - set(df.columns)}"
        
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
    
    fid2name = dict(zip(df.field_id.values, df.object.values))
    fid2radec = pd.Series(list(zip(df['ra'], df['dec'])), index=df['field_id']).values
    fid2filter = pd.Series(df['filter'], index=df['field_id']).values
    pivot_df = df.pivot_table(index='field_id', columns='filter_idx', values='count', aggfunc='sum').fillna(0)
    full_columns = range(len(FILTER2IDX)) # Assuming mapping is 0-indexed
    pivot_df = pivot_df.reindex(columns=full_columns, fill_value=0)
    target_fidfilt_counts = pivot_df.to_numpy()
    target_filt_counts = df.groupby('filter_idx')['count'].sum()
    target_fid_counts = df.groupby('field_id')['count'].sum()

    lookups = LookupTables(fid2name=fid2name, fid2radec=fid2radec, fid2filters=fid2filter, 
                           target_fid_counts=target_fid_counts, target_fidfilt_counts=target_fidfilt_counts, target_filt_counts=target_filt_counts)

    if write_to_disk:
        lookups.write_to_disk(data_dir)
        
    return lookups