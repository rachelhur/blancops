from collections import defaultdict
from pathlib import Path
import ephem
from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np
import pandas as pd
import math

import torch

from blancops.math import units
from blancops.ephemerides import ephemerides
from blancops.data.offline_dataset import setup_feature_names

from blancops.features.global_features import calc_moon_phase, calc_sun_and_moon_positions, calc_twilight, calc_urgency
from blancops.data.constants import *

from blancops.environment.base import BaseBlancoEnv
from datetime import datetime, timezone, timedelta

import logging
logger = logging.getLogger(__name__)

class OnlineBlancoEnv(BaseBlancoEnv):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, gcfg, cfg, observing_night_strs, data_dir, max_nights=0, horizon='-12', airmass_limit=1.4, z_score_stats=None, rel_norm_stats=None,
                 s_visits_cur=None, s_filter_visits_cur=None, field_priorities_arr = None, night1_ts_start=None):
        """
        """
        # Assign static attributes
        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_sin_norm = cfg['data']['do_sin_norm']
        self.do_log_norm = cfg['data']['do_log_norm']
        self.do_fractional_norm = cfg['data']['do_fractional_norm']
        self.do_z_score_norm = cfg['data']['do_z_score_norm']
        self.do_local_mean_z_score = cfg['data']['do_local_mean_z_score']
        self._init_s_visits = s_visits_cur.copy() if s_visits_cur is not None else None
        self._init_s_filter_visits = s_filter_visits_cur.copy() if s_filter_visits_cur is not None else None
        
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES'] if self.do_cyclical_norm else []
        self.sin_norm_feature_names = gcfg['features']['SIN_NORM_FEATURE_NAMES'] if self.do_sin_norm else []
        self.log_norm_feature_names = gcfg['features']['LOG_NORM_FEATURE_NAMES'] if self.do_log_norm else []
        self.fractional_norm_feature_names = gcfg['features']['FRACTIONAL_FEATURE_NAMES'] if self.do_fractional_norm else []
        self.z_score_feature_names = gcfg['features']['Z_SCORE_NORM_FEATURE_NAMES'] if self.do_z_score_norm else []
        self.local_mean_z_score_feature_names = gcfg['features']['LOCAL_MEAN_Z_SCORE_FEATURE_NAMES'] if self.do_local_mean_z_score else []
        
        self._z_score_stats = z_score_stats if z_score_stats is not None else torch.load(Path(cfg['metadata']['outdir']) / "z_score_stats.pt")
        self._rel_norm_stats = rel_norm_stats if rel_norm_stats is not None else torch.load(Path(cfg['metadata']['outdir']) / "rel_norm_stats.pt")
        
        self.include_bin_features = len(cfg['data']['bin_features']) > 0
        self.action_space = cfg['data']['action_space']
        nside = cfg['data']['nside']
        self.hpGrid = ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in self.action_space))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._action_architecture = cfg['model']['action_architecture']
        self._has_historical_features = any(sub in main_str for main_str in cfg['data']['bin_features'] 
                                           for sub in ['num_unvisited_fields', 'num_incomplete_fields', 'min_tiling'])
        self.horizon = horizon
        self.max_nights = max(len(observing_night_strs), max_nights) - 1
        self.night1_ts_start = night1_ts_start
        self._field_priorities_arr = field_priorities_arr
        # self._z_score_stats = torch.load(Path(cfg['metadata']['outdir']) / "z_score_stats.pt")
        # self._rel_norm_stats = torch.load(Path(cfg['metadata']['outdir']) / "rel_norm_stats.pt")
        
        self._night_info = []
        for obs_n_str in observing_night_strs:
            str_split = obs_n_str.split('-', maxsplit=3)
            night_str = '-'.join(str_split[:3])
            night_portion = str_split[-1]
            night_dt = datetime.strptime(night_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            midnight_dt = night_dt + (timedelta(days=1) - pd.Timedelta(nanoseconds=1))
            self._night_info.append((midnight_dt, night_portion))

        self._airmass_limit = airmass_limit
        self.do_filt = 'filter' in self.action_space
        self.nfilters = len(FILTER2IDX)

        self.field_lookup = pd.read_json(Path(data_dir) / "field_lookup.json" )
        # self.field2radec = pd.read_json(Path(data_dir) / "field2radec.json")
        from blancops.data.manager import load_field2radec_as_numpy
        self.field2radec = load_field2radec_as_numpy(Path(data_dir) / "field2radec.json")
                
        self._fids = np.unique(self.field_lookup['field_id'].to_numpy())
        self.nfields = len(self._fids)
        assert np.array_equal(self._fids, np.arange(len(self._fids))), "Field IDs must be perfectly sequential and start at 0."
        
        self._ra_arr = self.field2radec[:, 0]
        self._dec_arr = self.field2radec[:, 1]
        self._max_s_visits_arr = np.bincount(self.field_lookup['field_id'].values, weights=self.field_lookup['n_visits'].values).astype(int)

        # Get filter lookup tables
        if self.do_filt:
            self.fieldfilter2maxvisits = np.zeros((self.nfields, NUM_FILTERS))

            # Extract the full length-40 arrays from the dataframe
            all_fids = self.field_lookup['field_id'].values
            all_filters = self.field_lookup['filter_idx'].values
            all_visits = self.field_lookup['n_visits'].values

            # Vectorized (fielt, filter): max visits
            np.add.at(
                self.fieldfilter2maxvisits, 
                (all_fids, all_filters), 
                all_visits
            )
            # self._filter_idx_arr = self.field_lookup['filter_idx'].unique()
            # self.fieldfilter2maxvisits = np.zeros((self.nfields, NUM_FILTERS)) # shape = (nfields, nfilters)
            # for filt_idx in self._filter_idx_arr:
            #     f_mask = (self.field_lookup['filter_idx'] == filt_idx).values
            #     f_nvisits = (self.field_lookup['n_visits'][f_mask]).to_numpy()
            #     print('fids = ', self._fids)
            #     print('f_nvisits = ', f_nvisits)
            #     print('fieldfilter2maxvisits = ', self.fieldfilter2maxvisits)
            #     np.add.at(self.fieldfilter2maxvisits, (self._fids, filt_idx), f_nvisits)
            self.nfilters = len(FILTER2IDX)
            self.idx2filter = {v: k for k, v in FILTER2IDX.items()}
            self._max_s_filter_visits_arr = np.array([self.fieldfilter2maxvisits[fid] for fid in self._fids], dtype=np.int32)
            if self._init_s_filter_visits is not None:
                self._s_filter_visits_cur = self._init_s_filter_visits.copy()
            else:
                self._s_filter_visits_cur = np.zeros((len(self._fids), self.nfilters), dtype=np.int32)
            self.field2maxvisits = None
        else:
            self.fieldfilter2maxvisits = None
            self.field2maxvisits = self._max_s_visits_arr

        # Get static bin memberships for radec
        if not self.hpGrid.is_azel:
            # Get bin membership of all fields in survey
            self._bins_membership_arr = self.hpGrid.ang2idx(lon=self._ra_arr, lat=self._dec_arr) # Bin membership of each field ordered by field idx
            self._in_s_plan = self._max_s_visits_arr > 0
            self._nfields_s = np.bincount(self._bins_membership_arr, weights=self._in_s_plan, minlength=self.nbins) # number of fields per bin
            self._active_bins_s = self._nfields_s > 0
        else:
            self._bins_membership_arr = None

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
        else:
            raise NotImplementedError
        
        self.state_dim = cfg['data']['state_dim']
        self.bin_state_dim = cfg['data']['bin_state_dim']

        self._global_state = np.zeros(self.state_dim, dtype=np.float32)
        self._bin_state = np.zeros(self.bin_state_dim, dtype=np.float32)
        self._setup_action_and_obs_spaces()

        super().__init__()

    def _init_to_first_state(self):
        """
        Initializes the internal state variables for the start of a new episode.
        """
        self._action_mask = np.ones(self.nbins, dtype=bool)
        self._night_idx = -1
        if getattr(self, '_init_s_visits', None) is not None:
            self._s_visits_cur = self._init_s_visits.copy()
        else:
            self._s_visits_cur = np.zeros(self.nfields, dtype=np.int32)
            
        if getattr(self, 'do_filt', False):
            if getattr(self, '_init_s_filter_visits', None) is not None:
                self._s_filter_visits_cur = self._init_s_filter_visits.copy()
            else:
                self._s_filter_visits_cur = np.zeros((self.nfields, self.nfilters), dtype=np.int32)
        self._is_new_night = True
        self._start_new_night()
    
    def _start_new_night(self):
        self._night_idx += 1
        if self._night_idx > self.max_nights:
            logger.info('Reached maximum allowed nights.')
            return
        
        # global features
        night_dt, night_portion = self._night_info[self._night_idx]
        # night_ts = night_dt.timestamp()
        self._sunset_ts = math.ceil(calc_twilight(night_dt.timestamp(), 'set', self.horizon))
        self._sunrise_ts = math.ceil(calc_twilight(night_dt.timestamp(), 'rise', self.horizon))
        self._field_id = ZENITH_FIELD_ID
        self._bin_num = ZENITH_BIN_NUM
        self._filter_idx = ZENITH_FILTER_IDX
        self._night_end_ts = self._sunrise_ts
        self._night_start_ts = self._sunset_ts
        if night_portion != 'full':
            half_night_duration = self._get_half_night_duration(self._sunset_ts, self._sunrise_ts)
            if night_portion == 'half1':
                self._night_end_ts -= half_night_duration
            elif night_portion == 'half2':
                self._night_start_ts += half_night_duration
            else:
                raise ValueError("Environment arg `observing_night_strs` must be of the form `YY-MM-dd-<night_portion> where night_portion in {'full', 'half1', 'half2'}")
        if self.night1_ts_start:
            self._night_start_ts = self.night1_ts_start
        
        self._ts = self._night_start_ts
        
        self._max_n_filter_visits_arr = np.zeros((self.nfields, self.nfilters), dtype=np.int32)
        self._global_state = self._calculate_global_features(field_id=self._field_id, filter_idx=ZENITH_FILTER_IDX, timestamp=self._ts, sunset_ts=self._sunset_ts, sunrise_ts=self._sunrise_ts,
                                                             ra_arr=self._ra_arr, dec_arr=self._dec_arr)

        # Get field visit counts at start of night
        if self.do_filt:
            self._n_filter_visits_cur = np.zeros((self.nfields, self.nfilters), dtype=np.int32)

        self._n_visits_cur = np.zeros(self.nfields, dtype=np.int32)
        if self.include_bin_features:
            self._max_n_visits_arr = np.zeros_like(self._n_visits_cur)
            self._in_n_plan = self._max_n_visits_arr > 0
            self._bin_state = self._calculate_bin_features(timestamp=self._ts)
            #self._max_n_visits_arr = np.bincount(self._fids[night_fids], minlength=self.nfields)

            if self.do_filt:
                self._max_n_filter_visits_arr = np.zeros((self.nfields, self.nfilters), dtype=np.int32)
                # np.add.at(self._max_n_filter_visits_arr, (0, 0), 1)

            if self._action_architecture in ACTION_ARCHITECTURES:
                A, B = self.nbins, self.bin_state_dim
                self._bin_state = np.array(self._bin_state).reshape((A, B))
        else:
            self._bin_state = np.array([])
        self._update_action_masks()

        # self._update_action_masks(self._ts, field2maxvisits=self.field2maxvisits, fieldfilter2maxvisits=self.fieldfilter2maxvisits, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
                                                #   hpGrid=self.hpGrid, field_visits_arr=self._s_visits_cur, field_filter_visits_arr=self._s_filter_visits_cur)

    def _fast_forward(self, timestamp, ras, decs, visited, max_visits):
        incomplete_mask = visited < max_visits
        incomplete_ras = ras[incomplete_mask]
        incomplete_decs = decs[incomplete_mask]
        
        # If all fields complete, survey is terminated
        if len(incomplete_ras) == 0:
            return self._night_end_ts
        test_timestamp = timestamp
        step_size = 60*1 # inspect visibility every 5 mins

        while test_timestamp < self._night_end_ts:
            test_timestamp += step_size
            _, fields_el = ephemerides.equatorial_to_topographic(ra=incomplete_ras, dec=incomplete_decs, time=test_timestamp)
            fields_el = np.atleast_1d(fields_el)
            cos_zenith = np.cos(90 * units.deg - fields_el[fields_el > 0])
            airmass = 1 / np.clip(cos_zenith, a_min=1e-5, a_max=None)
            if np.any(airmass < self._airmass_limit):
                return test_timestamp
        # If fields never above horizon, return sunrise time
        return min(test_timestamp, self._night_end_ts)
