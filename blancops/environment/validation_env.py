from collections import defaultdict
from pathlib import Path
from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np
import pandas as pd

import torch

from blancops.configs.constants import CYCLICAL_FEATURE_NAMES
from blancops.data.features.normalizations import build_normalizer_kwargs
from blancops.data.lookup import load_lookup_tables
from blancops.data.constants import *
from blancops.math import units
from blancops.ephemerides import ephemerides
from blancops.data.features.normalizations import setup_feature_names
from blancops.data.features.glob_features import calc_moon_phase, calc_sun_and_moon_positions, calc_twilight, calc_urgency
from blancops.data.constants import *

from blancops.environment.base import BaseBlancoEnv
import pickle
import json

import logging
logger = logging.getLogger(__name__)

class ValidationBlancoEnv(BaseBlancoEnv):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, cfg, lookups, global_normalizer, bin_normalizer, max_nights=None, exp_time=90., global_pd_nightgroup=None, zenith_bin_states=None, fwhm_night_interps=None, z_score_stats=None, rel_norm_stats=None,
                 t_survey_arr=None, survey_nights_total=None):
        """
        Args
        ----
            dataset: An object (assumed to be OfflineDECamDataset instance) containing
                     static environment parameters and observation data.
        """
        assert cfg is not None, "Either cfg or test_dataset must be passed"
        self.cfg = cfg
        # Assign static attributes
        self.exp_time = exp_time
        self.lookups = lookups
        self.include_bin_features = cfg.data.bin_state_dim > 0
        self.action_space = cfg.data.action_space
        nside = cfg.data.nside
        self.hpGrid = ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in self.action_space))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._has_historical_features = any(sub in main_str for main_str in cfg.data.bin_features 
                                           for sub in ['num_unvisited_fields', 'num_incomplete_fields', 'min_tiling'])
        self.do_filt = 'filter' in self.action_space
        self._fwhm_night_interps = fwhm_night_interps
        self._t_survey_arr = t_survey_arr
        self.nfields = len(lookups.field2maxvisits)
        self._fids = np.array(list(lookups.field2maxvisits.keys())).astype(np.int32)
        assert np.array_equal(self._fids, np.arange(len(self._fids))), "Field IDs must be perfectly sequential and start at 0."
        self._ra_arr = np.array([lookups.field2radec[fid][0] for fid in self._fids])
        self._dec_arr = np.array([lookups.field2radec[fid][1] for fid in self._fids])
        self._max_s_visits_arr = np.array([lookups.field2maxvisits[fid] for fid in self._fids], dtype=np.int32)
        
        self._rel_norm_stats = rel_norm_stats
        self._z_score_stats = z_score_stats
        logger.debug(f"Loaded rel_norm_stats: {self._rel_norm_stats}")
        logger.debug(f"Loaded z_score_stats: {self._z_score_stats}")
        self.global_normalizer = global_normalizer
        self.bin_normalizer = bin_normalizer
        norm_kwargs = build_normalizer_kwargs(cfg.data.norm)
        self.cyclical_feature_names = norm_kwargs.get('cyclical_feature_names', [])
        self.do_cyclical_norm = len(self.cyclical_feature_names) > 0

        # Get filter lookup tables
        if self.do_filt:
            self.nfilters = NUM_FILTERS
            self.idx2filter = {v: k for k, v in FILTER2IDX.items()}
            if self.do_filt: 
                self._max_s_filter_visits_arr = np.array([lookups.fieldfilter2maxvisits[fid] for fid in self._fids], dtype=np.int32)

        # Bin-space dependent function to get fields in bin
        if not self.hpGrid.is_azel:
            # Get bin membership of all fields in survey
            self._bins_membership_arr = self.hpGrid.ang2idx(lon=self._ra_arr, lat=self._dec_arr) # Bin membership of each field ordered by field idx
            self._in_s_plan = self._max_s_visits_arr > 0 # should be all True - refactor code to make sure field_id array is dense and get rid of this condition - #TODO
            self._nfields_s = np.bincount(self._bins_membership_arr, weights=self._in_s_plan, minlength=self.nbins) # number of fields per bin
            self._active_bins_s = self._nfields_s > 0

        self.base_global_feature_names = cfg.data.global_features
        self.base_bin_feature_names = cfg.data.bin_features.copy()
        self.global_feature_names, self.bin_feature_names =\
            setup_feature_names(base_global_feature_names=cfg.data.global_features,
                                base_bin_feature_names=cfg.data.bin_features,
                                cyclical_feature_names=CYCLICAL_FEATURE_NAMES,
                                do_cyclical_norm=norm_kwargs['do_cyclical_norm'],
                                )
        if cfg.model.network == 'mlp':
            self.state_feature_names = self.global_feature_names + self.bin_feature_names
        else:
            self.state_feature_names = self.global_feature_names
        
        self.global_pd_nightgroup = global_pd_nightgroup
        self.zenith_bin_states = zenith_bin_states
        self.survey_nights_total = survey_nights_total

        self.max_nights = max_nights
        if max_nights is None:
            self.max_nights = self.global_pd_nightgroup.ngroups

        self.state_dim = cfg.data.state_dim
        self.bin_state_dim = cfg.data.bin_state_dim

        self._setup_action_and_obs_spaces()
        super().__init__()

    def _init_to_first_state(self):
        """
        Initializes the internal state variables for the start of a new episode.
        """
        self._action_mask = np.ones(self.nbins, dtype=bool)
        self._night_idx = -1
        self._is_new_night = True
        self._start_new_night()
        self._update_action_masks()
        # self._update_action_masks(timestamp=self._ts, field2maxvisits=self.field2maxvisits, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
        #                                           hpGrid=self.hpGrid, visited=self._s_visits_cur)
    
    def _start_new_night(self):
        self._night_idx += 1
        if self._night_idx >= self.max_nights:
            return

        # global features
        global_first_row = self.global_pd_nightgroup.head(1).iloc[self._night_idx]
        night = global_first_row['night']
        self._ts = global_first_row['timestamp']
        self._sunset_ts = calc_twilight(self._ts+10, 'set') # add 10 seconds just in case timestamp is exactly at twilight
        self._sunrise_ts = calc_twilight(self._ts+10, 'rise')
        self._night_end_ts = self.global_pd_nightgroup.tail(1).iloc[self._night_idx]['timestamp']
        self._night_start_ts = global_first_row['timestamp']
        self._field_id = global_first_row['field_id']
        self._bin_num = global_first_row['bin']
        self._survey_night_idx = global_first_row['night_idx']
        
        self._global_state = [global_first_row[feat_name] for feat_name in self.global_feature_names]

        # Get field visit counts at start of night
        self._s_visits_cur = self.lookups.night2fieldvisithistory[night][self._fids].copy().astype(np.int32)
        self._n_visits_cur = np.zeros(self.nfields, dtype=np.int32)
        
        # Get field filter visit counts at start of night
        if self.do_filt:
            self._s_filter_visits_cur = self.lookups.night2filtervisithistory[night].copy()
            self._n_filter_visits_cur = np.zeros((self.nfields, self.nfilters), dtype=np.int32)
            if 'raw_survey_progress_g' in list(global_first_row.keys()):
                self._global_urgency_arr = np.array([global_first_row[f'urgency_{filt_name}'] for filt_name in FILTER2IDX.keys()], dtype=np.float32)
                self._raw_survey_progress_arr = np.array([global_first_row[f"raw_survey_progress_{filt}"] for filt in FILTER2IDX.keys()], dtype=np.float32)
            
                                         
        if self.include_bin_features:
            global_night_df = self.global_pd_nightgroup.get_group(night).copy()
            zenith_bin_state_tonight = self.zenith_bin_states[self._night_idx] # shape (nbins, nfeats)
            self._bin_state = zenith_bin_state_tonight

            nonzenith_night_mask = global_night_df['object'] != 'zenith'
            night_fids = global_night_df['field_id'][nonzenith_night_mask].to_numpy().astype(np.int32)
            self._max_n_visits_arr = np.bincount(self._fids[night_fids], minlength=self.nfields)
            self._in_n_plan = self._max_n_visits_arr > 0

            if self.do_filt:
                if 'filt_idx' not in global_night_df.columns:
                    global_night_df['filt_idx'] = global_night_df['filter'].map(FILTER2IDX) #.fillna()
                n_filts = global_night_df['filt_idx'][nonzenith_night_mask].to_numpy(dtype=np.int32)
                self._max_n_filter_visits_arr = np.zeros((self.nfields, self.nfilters), dtype=np.int32)
                np.add.at(self._max_n_filter_visits_arr, (night_fids, n_filts), 1)
        else:
            self._bin_state = np.array([])
        self._update_action_masks()
