from collections import defaultdict
from pathlib import Path
from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np
import pandas as pd

import torch

from blancops.data_processing.data_processing import expand_feature_names_for_cyclic_norm
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.math import units
from blancops.ephemerides import ephemerides
from blancops.data_processing.offline_dataset import setup_feature_names
from blancops.data_processing.features import calculate_urgency, get_delta_az_el, get_moon_phase, get_relative_survey_progress_features, get_sun_and_moon_positions, normalize_timestamp, get_nautical_twilight, normalize_noncyclic_features, get_relative_feature
from blancops.data_processing.constants import *
from blancops.math import geometry

from blancops.environment.blanco_base import BaseBlancoEnv
from astropy.time import Time
from datetime import datetime, timezone, timedelta
import pickle
import json
from einops import rearrange

from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(__name__)

class OfflineBlancoTestingEnv(BaseBlancoEnv):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, gcfg, cfg, max_nights=None, exp_time=90., global_pd_nightgroup=None, zenith_bin_states=None, fwhm_night_interps=None, z_score_stats=None, rel_norm_stats=None,
                 t_survey_arr=None, survey_nights_total=None):
        """
        Args
        ----
            dataset: An object (assumed to be OfflineDECamDataset instance) containing
                     static environment parameters and observation data.
        """
        assert cfg is not None, "Either cfg or test_dataset must be passed"
        
        # Assign static attributes
        self.exp_time = exp_time

        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_sin_norm = cfg['data']['do_sin_norm']
        self.do_log_norm = cfg['data']['do_log_norm']
        self.do_fractional_norm = cfg['data']['do_fractional_norm']
        self.do_z_score_norm = cfg['data']['do_z_score_norm']
        self.do_local_mean_z_score = cfg['data']['do_local_mean_z_score']
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES'] if self.do_cyclical_norm else []
        self.sin_norm_feature_names = gcfg['features']['SIN_NORM_FEATURE_NAMES'] if self.do_sin_norm else []
        self.log_norm_feature_names = gcfg['features']['LOG_NORM_FEATURE_NAMES'] if self.do_log_norm else []
        self.fractional_norm_feature_names = gcfg['features']['FRACTIONAL_FEATURE_NAMES'] if self.do_fractional_norm else []
        self.z_score_feature_names = gcfg['features']['Z_SCORE_NORM_FEATURE_NAMES'] if self.do_z_score_norm else []
        self.local_mean_z_score_feature_names = gcfg['features']['LOCAL_MEAN_Z_SCORE_FEATURE_NAMES'] if self.do_local_mean_z_score else []

        self.include_bin_features = len(cfg['data']['bin_features']) > 0
        self.action_space = cfg['data']['action_space']
        nside = cfg['data']['nside']
        self.hpGrid = ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in self.action_space))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._action_architecture = cfg['model']['action_architecture']
        self._has_historical_features = any(sub in main_str for main_str in cfg['data']['bin_features'] 
                                           for sub in ['num_unvisited_fields', 'num_incomplete_fields', 'min_tiling'])
        self.do_filt = 'filter' in self.action_space
        self._fwhm_night_interps = fwhm_night_interps
        self._rel_norm_stats = rel_norm_stats if rel_norm_stats is not None else torch.load(Path(cfg['metadata']['outdir']) / "rel_norm_stats.pt")
        self._z_score_stats = z_score_stats if z_score_stats is not None else torch.load(Path(cfg['metadata']['outdir']) / "z_score_stats.pt")
        logger.debug(f"Loaded rel_norm_stats: {self._rel_norm_stats}")
        logger.debug(f"Loaded z_score_stats: {self._z_score_stats}")
        self._t_survey_arr = t_survey_arr
        # Get field lookup tables
        with open(gcfg['paths']['TRAIN_DIR'] + '/' + gcfg['files']['FIELD2RADEC'], 'r') as f:
            self.field2radec = json.load(f)
            self.field2radec = {int(k): v for k, v in self.field2radec.items()}
        with open(gcfg['paths']['TRAIN_DIR'] + '/' + gcfg['files']['FIELD2MAXVISITS_EVAL'], 'r') as f:
            self.field2maxvisits = json.load(f)
            self.field2maxvisits = {int(fid): int(count) for fid, count in self.field2maxvisits.items()}
        with open(gcfg['paths']['TRAIN_DIR'] + gcfg['files']['NIGHT2FIELDVISITS'], 'rb') as f:
            self.night2fieldvisithistory = pickle.load(f)

        self.nfields = len(self.field2maxvisits)
        self._fids = np.array(list(self.field2maxvisits.keys())).astype(np.int32)
        assert np.array_equal(self._fids, np.arange(len(self._fids))), "Field IDs must be perfectly sequential and start at 0."
        self._ra_arr = np.array([self.field2radec[fid][0] for fid in self._fids])
        self._dec_arr = np.array([self.field2radec[fid][1] for fid in self._fids])
        self._max_s_visits_arr = np.array([self.field2maxvisits[fid] for fid in self._fids], dtype=np.int32)

        # Get filter lookup tables
        if self.do_filt:
            with open(gcfg['paths']['TRAIN_DIR'] + '/' + gcfg['files']['FIELD2FILTERS'], 'rb') as f:
                self.field2filters = pickle.load(f)
                self.field2filters = {int(k): v for k, v in self.field2filters.items()}
            with open(gcfg['paths']['TRAIN_DIR'] + gcfg['files']['NIGHT2FILTERVISITS'], 'rb') as f:
                self.night2filtvisithistory = pickle.load(f)
            with open(gcfg['paths']['TRAIN_DIR'] + gcfg['files']['FIELDFILTER2MAXVISITS'], 'rb') as f:
                self.fieldfilter2maxvisits = pickle.load(f)
            with open(gcfg['paths']['TRAIN_DIR'] + gcfg['files']['FILTER_TARGET_COUNTS'], 'rb') as f:
                self._filter_target_counts = pickle.load(f)

            self.nfilters = NUM_FILTERS
            self.idx2filter = {v: k for k, v in FILTER2IDX.items()}
            if self.do_filt: 
                self._max_s_filter_visits_arr = np.array([self.fieldfilter2maxvisits[fid] for fid in self._fids], dtype=np.int32)

        # Bin-space dependent function to get fields in bin
        if not self.hpGrid.is_azel:
            # Get bin membership of all fields in survey
            self._bins_membership_arr = self.hpGrid.ang2idx(lon=self._ra_arr, lat=self._dec_arr) # Bin membership of each field ordered by field idx
            self._in_s_plan = self._max_s_visits_arr > 0 # should be all True - refactor code to make sure field_id array is dense and get rid of this condition - #TODO
            self._nfields_s = np.bincount(self._bins_membership_arr, weights=self._in_s_plan, minlength=self.nbins) # number of fields per bin
            self._active_bins_s = self._nfields_s > 0

        self.base_global_feature_names = cfg['data']['global_features'].copy()
        self.base_bin_feature_names = cfg['data']['bin_features'].copy()
        self.global_feature_names, self.bin_feature_names =\
            setup_feature_names(base_global_feature_names=cfg['data']['global_features'],
                                base_bin_feature_names=cfg['data']['bin_features'],
                                cyclical_feature_names=self.cyclical_feature_names,
                                do_cyclical_norm=self.do_cyclical_norm,
                                )
        
        if self._action_architecture is None:
            self.state_feature_names = self.global_feature_names + self.bin_feature_names
        elif self._action_architecture in ACTION_ARCHITECTURES:
            self.state_feature_names = self.global_feature_names
        
        self.global_pd_nightgroup = global_pd_nightgroup
        self.zenith_bin_states = zenith_bin_states
        self.survey_nights_total = survey_nights_total

        self.max_nights = max_nights
        if max_nights is None:
            self.max_nights = self.global_pd_nightgroup.ngroups

        self.state_dim = cfg['data']['state_dim']
        self.bin_state_dim = cfg['data']['bin_state_dim']

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
        self._sunset_ts = get_nautical_twilight(self._ts+10, 'set') # add 10 seconds just in case timestamp is exactly at twilight
        self._sunrise_ts = get_nautical_twilight(self._ts+10, 'rise')
        self._night_end_ts = self.global_pd_nightgroup.tail(1).iloc[self._night_idx]['timestamp']
        self._night_start_ts = global_first_row['timestamp']
        self._field_id = global_first_row['field_id']
        self._bin_num = global_first_row['bin']
        self._survey_night_idx = global_first_row['night_idx']
        
        self._global_state = [global_first_row[feat_name] for feat_name in self.global_feature_names]

        # Get field visit counts at start of night
        self._s_visits_cur = self.night2fieldvisithistory[night][self._fids].copy().astype(np.int32)
        self._n_visits_cur = np.zeros(self.nfields, dtype=np.int32)
        
        # Get field filter visit counts at start of night
        if self.do_filt:
            self._s_filter_visits_cur = self.night2filtvisithistory[night].copy()
            self._n_filter_visits_cur = np.zeros((self.nfields, self.nfilters), dtype=np.int32)
            if 'raw_survey_progress_g' in global_first_row.columns:
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
