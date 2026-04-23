import os
from pathlib import Path
from turtle import pd

import numpy as np

from blancops.data.constants import FILTER2IDX
from blancops.data.features.normalizations import StateNormalizer, build_normalizer_kwargs, load_normalization_stats
from blancops.data.lookup import LookupTables
from blancops.environment.online_env import OnlineBlancoEnv
from blancops.math import units

def build_env(cfg, trained_model_dir, lookups, sun_el_lim=None, airmass_lim=None, t_start=None, chunk_size: int = 10):
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
        chunk_size=chunk_size,
        sun_el_lim=sun_el_lim,
        airmass_lim=airmass_lim,
        t_start=t_start
    )
    return env

