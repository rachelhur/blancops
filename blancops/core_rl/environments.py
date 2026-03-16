from collections import defaultdict
from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np
import pandas as pd
import math
from blancops.data_processing.data_processing import get_nautical_twilight

from blancops.data_processing.data_processing import normalize_noncyclic_features, normalize_timestamp
from blancops.math import units
from blancops.ephemerides import ephemerides
from blancops.math.geometry import angular_separation
from blancops.data_processing.offline_dataset import setup_feature_names
from blancops.data_processing.data_processing import *
from blancops.math import geometry

from astropy.time import Time
from datetime import datetime, timezone
import pickle
import json

import logging
logger = logging.getLogger(__name__)

class BaseTelescope(gym.Env):
    """
    Base class providing for a Gymnasium environment simulating sequential observation scheduling.

    This class provides core Gym Env structure (reset, step) and common logic.

    Subclasses must define the following methods:
        - _init_to_first_state(): Initializes the environment state for a new episode
        - _update_action_mask(): Updates action masks
        - _update_obs(action): Updates the internal state based on teh action taken
        - _get_state(): Converts internal state into the formal observation
        - _get_info(): Computes auxiliary information dictionary
        - _get_termination_status(): Checks if episode has terminated
    """
    def __init__(self):
        '''
        Subclasses must instantiate the following attributes here *before* calling self._init_to_first_state()):
            - reward_func (Callable): Function to calculate rewards for transitions
            - observation_space (gym.spaces)
            - action_space (gym.spaces)
            - obs_dim (int): Size of observation input vector
            - normalize_obs (bool): Whether observations should be normalized
            - norm (nd.array): Normalization factors for observations. If normalize_obs is False, it is simply an array of ones.
        Subclasses must call self._init_to_first_state() after assigning attributes
        '''

        super().__init__()

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args
        ----
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns
        -------
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # initialize into a non-state.
        # this allows first field choice to be learned
        self._init_to_first_state()
        obs = self._get_state()
        info = self._get_info()
        return obs, info
    
    def step(self, action: int):
        """Execute one timestep within the environment.

        Args
        ----
            action (int): The field ID to observe next.

        Returns
        -------
            tuple: (next_obs, reward, terminated, truncated, info)
                - next_obs (np.ndarray): The observation after the action.
                - reward (float): The reward obtained from the action.
                - terminated (bool): Whether the episode has ended (e.g., reached observation limit).
                - truncated (bool): Whether the episode was truncated (always False here).
                - info (dict): Auxiliary diagnostic information.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        # last_coord = self._coord
        last_field_id = np.int32(self._field_id)
        self._update_obs(action)
        
        # ------------------- Calculate reward ------------------- #
        reward = 0
        reward += self._get_rewards(last_field_id, self._field_id)
        # ------------------------------------------ #

        # end condition
        truncated = False
        terminated = self._get_termination_status()
        # get obs and info
        next_obs = self._get_state()
        info = self._get_info()

        return next_obs, reward, terminated, truncated, info
    
    def _get_rewards(self, last_field, next_field):
        '''
        Calculates the reward for a single state transition.

        Uses self._reward_func() if available, otherwise returns 1.

        Args
        ----
            last_field (int): Field ID before taking the action.
            next_field (int): Field ID after taking the action.

        Returns
        -------
            float: The calculated reward value.
        '''
        if getattr(self, "reward_func", None) is None:
            return 1
        return self._reward_func(last_field, next_field)    

class OfflineDECamTestingEnv(BaseTelescope):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, gcfg, cfg, max_nights=None, exp_time=90., slew_time=30., global_pd_nightgroup=None, bin_pd_nightgroup=None):
        """
        Args
        ----
            dataset: An object (assumed to be OfflineDECamDataset instance) containing
                     static environment parameters and observation data.
        """
        assert cfg is not None, "Either cfg or test_dataset must be passed"
        
        # Assign static attributes
        self.exp_time = exp_time
        self.slew_time = slew_time
        self.time_between_obs = exp_time + slew_time
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES']
        self.max_norm_feature_names = gcfg['features']['MAX_NORM_FEATURE_NAMES']
        self.ang_distance_feature_names = gcfg['features']['ANG_DISTANCE_NORM_FEATURE_NAMES']
        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_max_norm = cfg['data']['do_max_norm']
        self.do_inverse_norm = cfg['data']['do_inverse_norm']
        self.do_ang_distance_norm = cfg['data']['do_ang_distance_norm']
        self.include_bin_features = len(cfg['data']['bin_features']) > 0
        self.bin_space = cfg['data']['bin_space']
        nside = cfg['data']['nside']
        self.hpGrid = None if cfg['data']['bin_method'] != 'healpix' else ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in self.bin_space))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._grid_network = cfg['model']['grid_network']
        self._has_historical_features = any(sub in main_str for main_str in cfg['data']['bin_features'] 
                                           for sub in ['num_unvisited_fields', 'num_incomplete_fields', 'min_tiling'])

        with open(gcfg['paths']['TRAIN_DIR'] + '/' + gcfg['files']['FIELD2RADEC'], 'r') as f:
            field2radec = json.load(f)
            self.field2radec = {int(k): v for k, v in field2radec.items()}
        with open(gcfg['paths']['TRAIN_DIR'] + '/' + gcfg['files']['FIELD2MAXVISITS_EVAL'], 'r') as f:
            field2maxvisits = json.load(f)
            self.field2maxvisits = {int(fid): int(count) for fid, count in field2maxvisits.items()}
        with open(gcfg['paths']['TRAIN_DIR'] + '/' + gcfg['files']['FIELD2FILTERS'], 'rb') as f:
            field2filters = pickle.load(f)
            self.field2radec = {int(k): v for k, v in field2radec.items()}
        # Field to index mapping for sparse field ids; unused fields maps to -1
        self.nfields = len(self.field2maxvisits)
        self._fids = np.array(list(self.field2maxvisits.keys())).astype(np.int32)
        self._ra_arr = np.zeros(self.nfields)
        self._dec_arr = np.zeros(self.nfields)
        self._max_s_visits_arr = np.zeros(self.nfields, dtype=np.int32)
        for idx, fid in enumerate(self._fids):
            self._ra_arr[idx], self._dec_arr[idx] = self.field2radec[fid]
            self._max_s_visits_arr[idx] = self.field2maxvisits[fid]

        max_fid = self._fids[-1]
        fid2idx = np.full(max_fid + 1, -1, dtype=np.int32)
        for idx, fid in enumerate(self._fids):
            fid2idx[fid] = idx
 
        with open(gcfg['paths']['TRAIN_DIR'] + gcfg['files']['NIGHT2FIELDVISITS'], 'rb') as f:
            self.night2visithistory = pickle.load(f)

        # Bin-space dependent function to get fields in bin
        if not self.hpGrid.is_azel:
            # Get bin membership of all fields in survey
            self._bins_membership_arr = self.hpGrid.ang2idx(lon=self._ra_arr, lat=self._dec_arr) # Bin membership of each field ordered by field idx
            self._in_s_plan = self._max_s_visits_arr > 0 # should be all True - refactor code to make sure field_id array is dense and get rid of this condition - #TODO
            self._nfields_s = np.bincount(self._bins_membership_arr, weights=self._in_s_plan, minlength=self.nbins) # number of fields per bin
            self._active_bins_s = self._nfields_s > 0

        self.base_global_feature_names = cfg['data']['global_features'].copy()
        self.base_bin_feature_names = cfg['data']['bin_features'].copy()
        self.global_feature_names, self.bin_feature_names, self.prenorm_expanded_bin_feature_names =\
            setup_feature_names(base_global_feature_names=cfg['data']['global_features'],
                                base_bin_feature_names=cfg['data']['bin_features'],
                                cyclical_feature_names=self.cyclical_feature_names,
                                nbins=self.nbins,
                                do_cyclical_norm=self.do_cyclical_norm,
                                grid_network=self._grid_network
                                )
        
        if self._grid_network is None:
            self.state_feature_names = self.global_feature_names + self.bin_feature_names
        elif self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
            self.state_feature_names = self.global_feature_names
        
        self.global_pd_nightgroup = global_pd_nightgroup
        self.bin_pd_nightgroup = bin_pd_nightgroup

        self.max_nights = max_nights
        if max_nights is None:
            self.max_nights = self.global_pd_nightgroup.ngroups

        self.state_dim = cfg['data']['state_dim']
        self.bin_state_dim = cfg['data']['bin_state_dim']

        if self.include_bin_features:
            bin_state_shape = (self.nbins, self.bin_state_dim, )
        else:
            bin_state_shape = (0,)

        self.observation_space = gym.spaces.Dict(
            {
                "global_state": gym.spaces.Box(-2, 2, shape=(self.state_dim,), dtype=np.float32),
                "bin_state": gym.spaces.Box(-2, 2, shape=bin_state_shape, dtype=np.float32),
            }
        )

        # Define action space 
        self.action_space = gym.spaces.Dict(
            {
                "bin": gym.spaces.Discrete(self.nbins),
                "field_id": gym.spaces.Discrete(len(self.field2radec)),
                "filter": gym.spaces.Box(0., 1., dtype=np.float32)
            }
        )       
        # self.action_space = gym.spaces.Box(low=np.array([0, 0, 0.]), high=np.array([self.nbins, len(self.field2radec), 1.]), dtype=np.int32)

        self._global_state = np.zeros(self.state_dim, dtype=np.float32)
        self._bin_state = np.zeros(self.bin_state_dim, dtype=np.float32)
        super().__init__()
    
    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args
        ----
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns
        -------
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self._init_to_first_state()
        state = self._get_state()
        info = self._get_info()
        return state, info

    def step(self, actions: dict):
        """Execute one timestep within the environment.

        Args
        ----
            action (int): The field ID to observe next.

        Returns
        -------
            tuple: (next_obs, reward, terminated, truncated, info)
                - next_obs (np.ndarray): The observation after the action.
                - reward (float): The reward obtained from the action.
                - terminated (bool): Whether the episode has ended (e.g., reached observation limit).
                - truncated (bool): Whether the episode was truncated (always False here).
                - info (dict): Auxiliary diagnostic information.
        """
        assert self.action_space.contains(actions), f"Invalid action {actions}"
        last_field_id = np.int32(self._field_id)

        # ------------------- Advance state ------------------- #
        self._update_state(actions)
        
        # ------------------- Calculate reward ------------------- #

        reward = 0
        reward += self._get_rewards(last_field_id, self._field_id)

        # -------------------- Start new night if last transition -----------------------#

        is_new_night = self._timestamp >= np.min([self._sunrise_time, self._night_final_timestamp])
        self._is_new_night = is_new_night
        
        if is_new_night:
            self._start_new_night()
        
        # -------------------- Terminate condition -----------------------#
        truncated = False
        terminated = self._get_termination_status()

        # get obs and info
        next_state = self._get_state()
        info = self._get_info()

        return next_state, reward, terminated, truncated, info

    # ------------------------------------------------------------ #
    # -------------Convenience functions-------------------------- #
    # ------------------------------------------------------------ #

    def _init_to_first_state(self):
        """
        Initializes the internal state variables for the start of a new episode.
        """
        self._action_mask = np.ones(self.nbins, dtype=bool)
        self._night_idx = -1
        self._is_new_night = True
        self._start_new_night()
        self._update_action_masks(timestamp=self._timestamp, field2nvisits=self.field2maxvisits, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
                                                  hpGrid=self.hpGrid, visited=self._s_visits_cur)
    
    def _start_new_night(self):
        self._night_idx += 1
        if self._night_idx >= self.max_nights:
            return

        # global features
        global_first_row = self.global_pd_nightgroup.head(1).iloc[self._night_idx]
        logger.debug(f"environment start new night filter wave {global_first_row['filter_wave']}")
        night = global_first_row['night']
        self._timestamp = global_first_row['timestamp']
        self._sunset_time = get_nautical_twilight(self._timestamp+10, 'set') # add 10 seconds just in case timestamp is exactly at twilight
        self._sunrise_time = get_nautical_twilight(self._timestamp+10, 'rise')
        self._night_final_timestamp = self.global_pd_nightgroup.tail(1).iloc[self._night_idx]['timestamp']
        self._night_first_timestamp = global_first_row['timestamp']
        self._field_id = global_first_row['field_id']
        if self._field_id != -1:
            logger.debug('FIRST FIELD IS NOT -1!!!')
        self._bin_num = global_first_row['bin']
        self._global_state = [global_first_row[feat_name] for feat_name in self.global_feature_names]

        # Get field visit counts at start of night
        self._s_visits_cur = self.night2visithistory[night][self._fids].copy().astype(np.int32)
        self._n_visits_cur = np.zeros(self.nfields, dtype=np.int32)

        if self.include_bin_features:
            # bin_feature_names = expand_feature_names_for_cyclic_norm(self.base_global_feature_names.copy(), self.cyclical_feature_names)
            global_night_df = self.global_pd_nightgroup.get_group(night)
            first_row_in_night_bin = self.bin_pd_nightgroup.head(1).iloc[self._night_idx]
            self._bin_state = np.array([first_row_in_night_bin[feat_name] for feat_name in self.bin_feature_names])
            # self._bin_state = np.array([first_row_in_night_bin[feat_name] for feat_name in self.bin_feature_names]).T
            night_fids = global_night_df['field_id'][global_night_df['object'] != 'zenith'].to_numpy().astype(np.int32)
            self._max_n_visits_arr = np.bincount(self._fids[night_fids], minlength=self.nfields)
            self._in_n_plan = self._max_n_visits_arr > 0
            if self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
                A, B = self.nbins, self.bin_state_dim
                self._bin_state = np.array(self._bin_state).reshape((A, B))
        else:
            self._bin_state = np.array([])
        self._update_action_masks(self._timestamp, field2nvisits=self.field2maxvisits, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
                                                  hpGrid=self.hpGrid, visited=self._s_visits_cur)

    def _update_state(self, action):
        """
        Updates the internal state variables based on the action taken.

        Args
        ----
            action (int): The chosen field ID to observe next.
        """
        self._timestamp += self.time_between_obs
        bin_num, field_id, filter_wave = int(action['bin']), int(action['field_id']), float(action['filter'])

        self._bin_num = bin_num
        self._field_id = field_id
        self._n_visits_cur[field_id] += 1
        self._s_visits_cur[field_id] += 1

        self._global_state = self._calculate_global_features(field_id=self._field_id, filter_wave=filter_wave, timestamp=self._timestamp,
                                                          sunset_time=self._sunset_time, sunrise_time=self._sunrise_time
                                                          )
        self._bin_state = self._calculate_bin_features(timestamp=self._timestamp) if self.include_bin_features else np.array([])


        self._update_action_masks(timestamp=self._timestamp, field2nvisits=self.field2maxvisits, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
                                                  hpGrid=self.hpGrid, visited=self._s_visits_cur)

    def _calculate_global_features(self, field_id, filter_wave, timestamp, sunset_time, sunrise_time):
        new_features = {}
        astro_time = Time(timestamp, format='unix', scale='utc')
        lst = astro_time.sidereal_time('apparent', longitude="-70:48:23.49")  # Blanco longitude
        new_features['lst'] = lst.radian
        new_features['ra'], new_features['dec'] = self.field2radec[field_id]
        new_features['az'], new_features['el'] = ephemerides.equatorial_to_topographic(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        new_features['ha'] = ephemerides.equatorial_to_hour_angle(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        
        cos_zenith = np.cos(90 * units.deg - new_features['el'])
        new_features['airmass'] = 1.0 / cos_zenith #if cos_zenith > 0 else 99.0

        new_features['sun_ra'], new_features['sun_dec'] = ephemerides.get_source_ra_dec(source='sun', time=timestamp)
        new_features['sun_az'], new_features['sun_el'] = ephemerides.equatorial_to_topographic(ra=new_features['sun_ra'], dec=new_features['sun_dec'], time=timestamp)
        new_features['moon_ra'], new_features['moon_dec'] = ephemerides.get_source_ra_dec(source='moon', time=timestamp)
        new_features['moon_az'], new_features['moon_el'] = ephemerides.equatorial_to_topographic(ra=new_features['moon_ra'], dec=new_features['moon_dec'], time=timestamp)

        if sunrise_time == sunset_time:
            new_features['time_fraction_since_start'] = 0
        else:
            new_features['time_fraction_since_start'] = (timestamp - sunset_time) / (sunrise_time - sunset_time)

        new_features['filter_wave'] = filter_wave

        for feat_name in self.base_global_feature_names:
            if any(string in feat_name and 'bin' not in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                new_features.update({f'{feat_name}_cos': np.cos(new_features[feat_name])})
                new_features.update({f'{feat_name}_sin': np.sin(new_features[feat_name])})

        global_state_features = [new_features.get(feat, np.nan) for feat in self.global_feature_names]
        nan_feats = np.isnan(global_state_features)
        if any(nan_feats):
            nan_idxs = np.where(nan_feats == True)[0]
            for idx in nan_idxs:
                raise ValueError(f"Calculated nan value for global feature {self.global_feature_names[idx]}")


        return global_state_features
    
    def _calculate_bin_features(self, timestamp):
        features = {}
        current_ra, current_dec = self.field2radec[self._field_id]
        if self.hpGrid.is_azel:
            lons, lats = ephemerides.topographic_to_equatorial(az=self.hpGrid.lon, el=self.hpGrid.lat, time=timestamp)
            features['az'], features['el'] = self.hpGrid.lon, self.hpGrid.lat
            features['ra'], features['dec'] = lons, lats
            current_lon, current_lat = ephemerides.equatorial_to_topographic(ra=current_ra, dec=current_dec, time=timestamp)
        else:
            lons, lats = ephemerides.equatorial_to_topographic(ra=self.hpGrid.lon, dec=self.hpGrid.lat, time=timestamp)
            features['ra'], features['dec'] = self.hpGrid.lon, self.hpGrid.lat
            features['az'], features['el'] = lons, lats
            current_lon, current_lat = current_ra, current_dec
            
        features['angular_distance_to_pointing'] = self.hpGrid.get_angular_separations(lon=current_lon, lat=current_lat)
        features['ha'] = self.hpGrid.get_hour_angle(time=timestamp)
        features['airmass'] = self.hpGrid.get_airmass(timestamp)
        features['moon_distance'] = self.hpGrid.get_source_angular_separations('moon', time=timestamp)
        
        if self._has_historical_features:
            if not self.hpGrid.is_azel:
                max_s_visits_arr = np.maximum(self._max_n_visits_arr, self._max_s_visits_arr)
                nfields_n = np.bincount(self._bins_membership_arr, weights=self._in_n_plan, minlength=self.nbins)
                active_bins_n = nfields_n > 0

                # Get number of unvisited fields in each bin - bins below horizon have 0 fields unvisited
                s_unvisited = np.bincount(self._bins_membership_arr, weights=(self._s_visits_cur == 0) & self._in_s_plan, minlength=self.nbins)
                n_unvisited = np.bincount(self._bins_membership_arr, weights=(self._n_visits_cur == 0) & self._in_n_plan, minlength=self.nbins)

                # Get number of incomplete fields in each bin
                s_incomplete_mask = (self._s_visits_cur < max_s_visits_arr) & self._in_s_plan
                n_incomplete_mask = (self._n_visits_cur < self._max_n_visits_arr) & self._in_n_plan
                s_incomplete = np.bincount(self._bins_membership_arr, weights=s_incomplete_mask, minlength=self.nbins)
                n_incomplete = np.bincount(self._bins_membership_arr, weights=n_incomplete_mask, minlength=self.nbins)
        
                # Create a zero-filled array for the results
                for key in ['survey_num_unvisited_fields', 'night_num_unvisited_fields', 
                            'survey_num_incomplete_fields', 'night_num_incomplete_fields']:
                    features[key] = np.zeros(self.nbins, dtype=np.float32) # bins with no viable fields get sentinel value 0
                
                # Do division in-place (bypasses runtimewarning error )
                np.divide(s_unvisited, self._nfields_s, out=features['survey_num_unvisited_fields'], where=self._active_bins_s)
                np.divide(n_unvisited, nfields_n, out=features['night_num_unvisited_fields'], where=active_bins_n)
                np.divide(s_incomplete, self._nfields_s, out=features['survey_num_incomplete_fields'], where=self._active_bins_s)
                np.divide(n_incomplete, nfields_n, out=features['night_num_incomplete_fields'], where=active_bins_n)
        
                # Min tiling
                s_tiling_all = np.full_like(self._s_visits_cur, 2.0, dtype=np.float32)
                n_tiling_all = np.full_like(self._n_visits_cur, 2.0, dtype=np.float32)
                # current_num_visits_field / max_num_visits_field only where max_num_visits_field > 0 ie, in the plan
                np.divide(self._s_visits_cur, max_s_visits_arr, out=s_tiling_all, where=self._in_s_plan)
                np.divide(self._n_visits_cur, self._max_n_visits_arr, out=n_tiling_all, where=self._in_n_plan)
                
                s_mins = np.full(self.nbins, 2.0, dtype=np.float32)
                n_mins = np.full(self.nbins, 2.0, dtype=np.float32)
                np.minimum.at(s_mins, self._bins_membership_arr, s_tiling_all)
                np.minimum.at(n_mins, self._bins_membership_arr, n_tiling_all)
                
                # Reset bins with no fields back to -0.1
                s_mins[s_mins > 1.0] = -1.0
                n_mins[n_mins > 1.0] = -1.0
                features['survey_min_tiling'] = s_mins
                features['night_min_tiling'] = n_mins
            else:
                # Reset at each timestep since fields' bin memberships change over time
                az, el = ephemerides.equatorial_to_topographic(ra=self._ra_arr, dec=self._dec_arr, time=timestamp)
                bins = self.hpGrid.ang2idx(lon=az, lat=el) # Bin membership of each field
                bins = np.array([b if b is not None else -1 for b in bins], dtype=np.int32)
                valid_mask = (el > 0) & (bins != -1)
                valid_bins = bins[valid_mask].astype(np.int32)

                in_s_plan = self._max_s_visits_arr[valid_mask] > 0
                in_n_plan = self._max_n_visits_arr[valid_mask] > 0

                bin_count_s = np.bincount(valid_bins, weights=in_s_plan, minlength=self.nbins)
                bin_count_n = np.bincount(valid_bins, weights=in_n_plan, minlength=self.nbins)
                
                active_bins_s = bin_count_s > 0
                active_bins_n = bin_count_n > 0

                # Get field counts and max field counts over 
                v_survey_counts = self._s_visits_cur[valid_mask].astype(np.int32)
                v_night_counts = self._n_visits_cur[valid_mask].astype(np.int32)
                v_max_visits_survey = self._max_s_visits_arr[valid_mask]
                v_max_visits_night = self._max_n_visits_arr[valid_mask]
                
                # Re-create the plan masks
                in_s_plan = v_max_visits_survey > 0
                in_n_plan = v_max_visits_night > 0

                for key_n, key_s, mask_n, mask_s in [
                    # Must be unvisited AND in the respective plan
                    ('night_num_unvisited_fields', 'survey_num_unvisited_fields', 
                    (v_night_counts == 0) & in_n_plan, 
                    (v_survey_counts == 0) & in_s_plan),
                    
                    # Must be incomplete AND in the respective plan
                    ('night_num_incomplete_fields', 'survey_num_incomplete_fields', 
                    (v_night_counts < v_max_visits_night) & in_n_plan, 
                    (v_survey_counts < v_max_visits_survey) & in_s_plan)
                    ]:
                    res_n, res_s = np.zeros(self.nbins, dtype=np.float32), np.zeros(self.nbins, dtype=np.float32)
                    
                    # Use the correct denominators and active masks!
                    np.divide(np.bincount(valid_bins, weights=mask_n, minlength=self.nbins), bin_count_n, out=res_n, where=active_bins_n)
                    np.divide(np.bincount(valid_bins, weights=mask_s, minlength=self.nbins), bin_count_s, out=res_s, where=active_bins_s)
                    
                    res_n[~active_bins_n] = 0.
                    res_s[~active_bins_s] = 0.
                    
                    features[key_n] = res_n
                    features[key_s] = res_s

                # Min Tiling 
                s_tiling_all = np.full_like(v_survey_counts, 2.0, dtype=np.float32)
                n_tiling_all = np.full_like(v_night_counts, 2.0, dtype=np.float32)
                
                np.divide(v_survey_counts, v_max_visits_survey, out=s_tiling_all, where=in_s_plan)
                np.divide(v_night_counts, v_max_visits_night, out=n_tiling_all, where=in_n_plan)
                    
                s_mins, n_mins = np.full(self.nbins, 2.0, dtype=np.float32), np.full(self.nbins, 2.0, dtype=np.float32)
                np.minimum.at(s_mins, valid_bins, s_tiling_all)
                np.minimum.at(n_mins, valid_bins, n_tiling_all)
                    
                s_mins[~active_bins_s] = -0.1
                n_mins[~active_bins_n] = -0.1
                    
                features['survey_min_tiling'] = s_mins
                features['night_min_tiling'] = n_mins

        # Normalize periodic features here and add as df cols
        if self.do_cyclical_norm:
            for feat_name in self.base_bin_feature_names:
                if any(string in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                    if feat_name in features.keys():
                        features[f'{feat_name}_cos'] = np.cos(features[feat_name])
                        features[f'{feat_name}_sin'] = np.sin(features[feat_name])
                    else:
                        raise ValueError(f"{feat_name} was not calculated in _calculate_bin_features. Is this feature implemented?")
        
        expanded_feat_names = expand_feature_names_for_cyclic_norm(self.base_bin_feature_names, cyclical_feature_names=self.cyclical_feature_names)
        bin_state = np.vstack([features.get(feat_name, np.nan) for feat_name in expanded_feat_names]).T
        return bin_state

    def _get_state(self):
        global_state, bin_state = self._global_state, self._bin_state
        global_state_copy = global_state.copy()
        bin_state_copy = bin_state.copy()
        # state_normed = self._do_noncyclic_normalizations(state=state_copy)
        global_state_normed = normalize_noncyclic_features(
                            state=np.array(global_state_copy),
                            state_feature_names=self.state_feature_names,
                            max_norm_feature_names=self.max_norm_feature_names,
                            ang_distance_norm_feature_names=self.ang_distance_feature_names,
                            do_inverse_norm=self.do_inverse_norm,
                            do_max_norm=self.do_max_norm,
                            do_ang_distance_norm=self.do_ang_distance_norm,
                            bin_space=self.bin_space,
                            fix_nans=True
                        )
        if self.include_bin_features:
            bin_state_normed = normalize_noncyclic_features(
                state=np.array(bin_state_copy)[np.newaxis, ...], # add axis for function
                state_feature_names=self.bin_feature_names,
                max_norm_feature_names=self.max_norm_feature_names,
                ang_distance_norm_feature_names=self.ang_distance_feature_names,
                do_inverse_norm=self.do_inverse_norm,
                do_max_norm=self.do_max_norm,
                do_ang_distance_norm=self.do_ang_distance_norm,
                bin_space=self.bin_space,
                fix_nans=True
            )
            bin_state_normed = bin_state_normed[0] # remove axis
        else:
            bin_state_normed = np.array([])
        self._global_state = global_state_normed.astype(np.float32)
        self._bin_state = bin_state_normed.astype(np.float32)

        # logger.debug(f"Global state max, min {self._global_state.max()}, {self._global_state.min()}")
        # logger.debug(f"State above max: {np.where(self._global_state > 2)}")            
        # logger.debug(f"State below min: {np.where(self._global_state <-2)}")
        # if self.include_bin_features:
        #     logger.debug(f"Bins state max, min {self._bin_state.max()}, {self._bin_state.min()}")
        #     logger.debug(f"State above max: {np.where(self._bin_state > 2)}")            
        #     logger.debug(f"State below min: {np.where(self._bin_state <-2)}")
        return {"global_state": self._global_state, "bin_state": self._bin_state}

    def _get_info(self):
        """
        Compute auxiliary information for debugging and constrained action spaces.

        Returns
        -------
            dict: A dictionary containing the current action mask.
        """
        return {'action_mask': self._action_mask.copy(), 
                'visited': self._n_visits_cur.copy(),
                'timestamp': self._timestamp,
                'is_new_night': bool(self._is_new_night),
                'night_idx': int(self._night_idx),
                'bin': int(self._bin_num),
                'field_id': int(self._field_id),
                'valid_fields_per_bin': self._valid_fields_per_bin 
        }
    def _get_termination_status(self):
        """
        Checks if the episode has reached its termination condition.

        Termination occurs when the total number of observations for the night
        (based on the dataset) has been met, or, when all fields have been completely
        visited.

        Returns
        -------
            bool: True if the episode is terminated, False otherwise.
        """
        all_nights_completed = self._night_idx >= self.max_nights
        all_fields_visited = all(np.array([self._s_visits_cur[fid] >= self.field2maxvisits[fid] for fid in self._fids]))
        terminated = all_nights_completed or all_fields_visited
        return terminated
    
    def _update_fields_in_bin(self, field_ids, mask_invalid_fields):
        fields_in_bin = field_ids[mask_invalid_fields]
        self._fields_in_bin = fields_in_bin

    def _update_action_masks(self, timestamp, field2nvisits, field_ids, ras, decs, hpGrid, visited):
        # Mask fields which are completed 
        mask_completed_fields = np.array([visited[fid] < field2nvisits[fid] for fid in field_ids], dtype=bool) #TODO can probably track visits without repeating this operation
        fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=ras, dec=decs, time=timestamp)
        # Mask fields below horizon
        mask_fields_below_horizon = fields_el > 0
        sel_valid_fields = mask_completed_fields & mask_fields_below_horizon
        self._sel_valid_fields = sel_valid_fields
        # Get bins which are below horizon, masking completed bins
        valid_fids = field_ids[sel_valid_fields]
        if hpGrid.is_azel:
            valid_field_bins = hpGrid.ang2idx(lon=fields_az[sel_valid_fields], lat=fields_el[sel_valid_fields])
        else:
            valid_field_bins = hpGrid.ang2idx(lon=ras[sel_valid_fields], lat=decs[sel_valid_fields])
        self._valid_fields_per_bin = defaultdict(list)
        action_mask = np.zeros(shape=self.nbins, dtype=bool)
        for fid, bin_idx in zip(valid_fids, valid_field_bins):
            if bin_idx is not None:
                b = int(bin_idx)
                action_mask[b] = True
                self._valid_fields_per_bin[b].append(fid)
        if 'filter' in self.bin_space:
            action_mask = np.repeat(action_mask[:, np.newaxis], NUM_FILTERS, axis=1).flatten() #TODO 2. in todoist
        self._action_mask = action_mask
        logger.debug(f'environment action_mask action_mask.shape: {action_mask.shape}')
        return action_mask
    
    def _get_valid_fields_per_bin(self, ):
        pass

    def _get_filter_mask(self, ):
        pass
    def _get_slew_time(self, slew_time=None):
        if slew_time is not None:
            return slew_time
        else:
            raise NotImplementedError

    def _get_exposure_time(self, exp_time=None):
        if exp_time is not None:
            return exp_time
        else:
            raise NotImplementedError

class OnlineDECamEnv(BaseTelescope):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, gcfg, cfg, night_str, lookup_path=None, lookup_dict=None, horizon='-12', max_nights=None, airmass_limit=1.4):
        """
        """
        # Assign static attributes
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES']
        self.max_norm_feature_names = gcfg['features']['MAX_NORM_FEATURE_NAMES']
        self.ang_distance_feature_names = gcfg['features']['ANG_DISTANCE_NORM_FEATURE_NAMES']
        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_max_norm = cfg['data']['do_max_norm']
        self.do_inverse_norm = cfg['data']['do_inverse_norm']
        self.do_ang_distance_norm = cfg['data']['do_ang_distance_norm']
        self.include_bin_features = len(cfg['data']['bin_features']) > 0
        self.bin_space = cfg['data']['bin_space']
        nside = cfg['data']['nside']
        self.hpGrid = None if cfg['data']['bin_method'] != 'healpix' else ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in self.bin_space))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._grid_network = cfg['model']['grid_network']
        self._has_historical_features = any(sub in main_str for main_str in cfg['data']['bin_features'] 
                                           for sub in ['num_unvisited_fields', 'num_incomplete_fields', 'min_tiling'])
        self.horizon = horizon
        self.max_nights = max_nights if max_nights is not None else 1
        night_dt = datetime.strptime(night_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self._first_night = night_dt + (timedelta(days=1) - pd.Timedelta(nanoseconds=1))
        self._airmass_limit = airmass_limit
        
        if lookup_dict:
            self.field_lookup = lookup_dict
        elif lookup_path:
            with open(lookup_path, 'r') as f:
                self.field_lookup = json.load(f)
        else: raise AssertionError("Must pass either lookup_path or lookup dict")
                
        self.nfields = len(self.field_lookup['ra'])
        self._fids = np.array(list(self.field_lookup['ra'].keys())).astype(np.int32)
        self._ra_arr = np.array(list(self.field_lookup['ra'].values()))
        self._dec_arr = np.array(list(self.field_lookup['dec'].values()))
        self._max_s_visits_arr =  np.array(list(self.field_lookup['n_visits'].values()))
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
        self.global_feature_names, self.bin_feature_names, self.prenorm_expanded_bin_feature_names =\
            setup_feature_names(base_global_feature_names=cfg['data']['global_features'],
                                base_bin_feature_names=cfg['data']['bin_features'],
                                cyclical_feature_names=self.cyclical_feature_names,
                                nbins=self.nbins,
                                do_cyclical_norm=self.do_cyclical_norm,
                                grid_network=self._grid_network
                                )
        if self._grid_network is None:
            self.state_feature_names = self.global_feature_names + self.bin_feature_names
        elif self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
            self.state_feature_names = self.global_feature_names
        
        self.state_dim = cfg['data']['state_dim']
        self.bin_state_dim = cfg['data']['bin_state_dim']

        if self.include_bin_features:
            bin_state_shape = (self.nbins, self.bin_state_dim,)
        else:
            bin_state_shape = (0,)

        self.observation_space = gym.spaces.Dict(
            {
                "global_state": gym.spaces.Box(-2, 2, shape=(self.state_dim,), dtype=np.float32),
                "bin_state": gym.spaces.Box(-2, 2, shape=bin_state_shape, dtype=np.float32),
            }
        )

        # Define action space 
        self.action_space = gym.spaces.Dict(
            {
                "bin": gym.spaces.Discrete(self.nbins + 2, start=-2), # -2 == wait, -1 == zenith
                "field_id": gym.spaces.Discrete(len(self.field_lookup['field_id']) + 2, start=-2), # -2 == wait, -1 == zenith
                "filter": gym.spaces.Box(0., 1., shape=(1,), dtype=np.float32) # 0 == no filter (ie, zenith - perhaps replace zenith state filter with what filter is currently set up)
            }
        )
        self._global_state = np.zeros(self.state_dim, dtype=np.float32)
        self._bin_state = np.zeros(self.bin_state_dim, dtype=np.float32)
        super().__init__()
    
    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args
        ----
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns
        -------
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self._init_to_first_state()
        state = self._get_state()
        info = self._get_info()
        return state, info

    def step(self, actions: dict):
        """Execute one timestep within the environment.

        Args
        ----
            action (int): The field ID to observe next.

        Returns
        -------
            tuple: (next_obs, reward, terminated, truncated, info)
                - next_obs (np.ndarray): The observation after the action.
                - reward (float): The reward obtained from the action.
                - terminated (bool): Whether the episode has ended (e.g., reached observation limit).
                - truncated (bool): Whether the episode was truncated (always False here).
                - info (dict): Auxiliary diagnostic information.
        """
        assert self.action_space.contains(actions), f"Invalid action {actions}"
        
        last_field_id = np.int32(self._field_id)

        # ------------------- Advance state ------------------- #
        self._update_state(actions)
        
        # ------------------- Calculate reward ------------------- #

        reward = 0
        reward += self._get_rewards(last_field_id, self._field_id)

        # -------------------- Start new night if is last transition -----------------------#

        is_new_night = self._timestamp >= self._sunrise_time
        self._is_new_night = is_new_night
        
        if is_new_night:
            self._start_new_night()
        
        # -------------------- Terminate condition -----------------------#
        truncated = False
        terminated = self._get_termination_status()

        # get obs and info
        next_state = self._get_state()
        info = self._get_info()

        return next_state, reward, terminated, truncated, info

    # ------------------------------------------------------------ #
    # -------------Convenience functions-------------------------- #
    # ------------------------------------------------------------ #

    def _init_to_first_state(self):
        """
        Initializes the internal state variables for the start of a new episode.
        """
        self._action_mask = np.ones(self.nbins, dtype=bool)
        self._s_visits_cur = np.zeros(self.nfields, dtype=np.int32)
        self._night_idx = -1
        self._is_new_night = True
        self._night_dt = self._first_night
        self._start_new_night()
    
    def _start_new_night(self):
        self._night_idx += 1

        # global features
        night_ts = self._night_dt.timestamp()
        self._sunset_time = math.ceil(get_nautical_twilight(night_ts, 'set', self.horizon))
        self._sunrise_time = math.ceil(get_nautical_twilight(night_ts, 'rise', self.horizon))
        logger.debug(f'HOURS IN NIGHT FROM SUNSET to SUNRISE= : {(self._sunrise_time - self._sunset_time)/3600:.2f}')
        self._field_id = -1
        self._bin_num = -1
        self._timestamp = self._sunset_time

        self._global_state = self._calculate_global_features(filter_wave=0., timestamp=self._timestamp, sunset_time=self._sunset_time, sunrise_time=self._sunrise_time)

        # Get field visit counts at start of night
        self._n_visits_cur = np.zeros(self.nfields, dtype=np.int32)

        if self.include_bin_features:
            self._bin_state = self._calculate_bin_features(timestamp=self._timestamp)
            self._max_n_visits_arr = np.zeros_like(self._n_visits_cur)
            # self._max_n_visits_arr = np.bincount(self._fids[night_fids], minlength=self.nfields)
            # self._in_n_plan = self._max_n_visits_arr > 0
            if self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
                A, B = self.nbins, self.bin_state_dim
                self._bin_state = np.array(self._bin_state).reshape((A, B))
        else:
            self._bin_state = np.array([])
        self._update_action_masks(self._timestamp, field_lookup=self.field_lookup, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
                                                  hpGrid=self.hpGrid, visited=self._s_visits_cur)

    def _update_state(self, action):
        """
        Updates the internal state variables based on the action taken.

        Args
        ----
            action (int): The chosen field ID to observe next.
        """
        bin_num, field_id, filter_wave = int(action['bin']), int(action['field_id']), float(action['filter'])
        
        if bin_num == -2:
            self._timestamp = self._fast_forward_to_timestamp(
                timestamp=self._timestamp,
                ras=self._ra_arr,
                decs=self._dec_arr,
                visited=self._s_visits_cur,
                max_visits=self._max_s_visits_arr
            )
        else:
            last_field_id = self._field_id
            exptime = self._get_exposure_time(field_id=str(field_id))
            slew_time = self._get_slew_time(last_field_id, field_id)
            self._timestamp += exptime + slew_time
            self._n_visits_cur[field_id] += 1
            self._s_visits_cur[field_id] += 1
            self._bin_num = bin_num
            self._field_id = field_id

        self._global_state = self._calculate_global_features(filter_wave=filter_wave, timestamp=self._timestamp,
                                                        sunset_time=self._sunset_time, sunrise_time=self._sunrise_time
                                                        )
        self._bin_state = self._calculate_bin_features(timestamp=self._timestamp) if self.include_bin_features else np.array([])
        self._update_action_masks(timestamp=self._timestamp, field_lookup=self.field_lookup, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
                                                  hpGrid=self.hpGrid, visited=self._s_visits_cur)

    def _calculate_global_features(self, filter_wave, timestamp, sunset_time, sunrise_time):
        new_features = {}
        astro_time = Time(timestamp, format='unix', scale='utc')
        lst = astro_time.sidereal_time('apparent', longitude="-70:48:23.49")  # Blanco longitude
        new_features['lst'] = lst.radian
        if self._field_id == -1:
            blanco = ephemerides.blanco_observer(time=timestamp)
            new_features['ra'], new_features['dec'] = new_features['lst'], blanco.lon
        else:
            ra = self.field_lookup['ra'][str(self._field_id)]
            dec = self.field_lookup['dec'][str(self._field_id)]
            new_features['ra'], new_features['dec'] = ra, dec
        new_features['az'], new_features['el'] = ephemerides.equatorial_to_topographic(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        new_features['ha'] = ephemerides.equatorial_to_hour_angle(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        
        cos_zenith = np.cos(np.pi / 2 - new_features['el'])
        new_features['airmass'] = 1.0 / cos_zenith #if cos_zenith > 0 else 99.0

        new_features['sun_ra'], new_features['sun_dec'] = ephemerides.get_source_ra_dec(source='sun', time=timestamp)
        new_features['sun_az'], new_features['sun_el'] = ephemerides.equatorial_to_topographic(ra=new_features['sun_ra'], dec=new_features['sun_dec'], time=timestamp)
        new_features['moon_ra'], new_features['moon_dec'] = ephemerides.get_source_ra_dec(source='moon', time=timestamp)
        new_features['moon_az'], new_features['moon_el'] = ephemerides.equatorial_to_topographic(ra=new_features['moon_ra'], dec=new_features['moon_dec'], time=timestamp)

        if sunrise_time == sunset_time:
            raise AssertionError("Sunrise and sunset time is equal. Check night_str argument - it should be a time between sunset and sunrise")
        else:
            new_features['time_fraction_since_start'] = normalize_timestamp(timestamp, sunset_timestamp=sunset_time, sunrise_timestamp=sunrise_time)

        new_features['filter_wave'] = filter_wave

        for feat_name in self.base_global_feature_names:
            if any(string in feat_name and 'bin' not in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                new_features.update({f'{feat_name}_cos': np.cos(new_features[feat_name])})
                new_features.update({f'{feat_name}_sin': np.sin(new_features[feat_name])})

        global_state_features = [new_features.get(feat, np.nan) for feat in self.global_feature_names]
        nan_feats = np.isnan(global_state_features)
        if any(nan_feats):
            nan_idxs = np.where(nan_feats == True)[0]
            for idx in nan_idxs:
                raise ValueError(f"Calculated nan value for global feature {self.global_feature_names[idx]}")
        return global_state_features
    
    def _calculate_bin_features(self, timestamp):
        features = {}
        if self._field_id == -1:
            blanco = ephemerides.blanco_observer(time=timestamp)
            features['ra'], features['dec'] = blanco.lat, blanco.lon
        else:
            features['ra'], features['dec'] = self._ra_arr[self._field_id], self._dec_arr[self._field_id]
        current_ra, current_dec = features['ra'], features['dec']
        
        if self.hpGrid.is_azel:
            lons, lats = ephemerides.topographic_to_equatorial(az=self.hpGrid.lon, el=self.hpGrid.lat, time=timestamp)
            features['az'], features['el'] = self.hpGrid.lon, self.hpGrid.lat
            features['ra'], features['dec'] = lons, lats
            current_lon, current_lat = ephemerides.equatorial_to_topographic(ra=current_ra, dec=current_dec, time=timestamp)
        else:
            lons, lats = ephemerides.equatorial_to_topographic(ra=self.hpGrid.lon, dec=self.hpGrid.lat, time=timestamp)
            features['ra'], features['dec'] = self.hpGrid.lon, self.hpGrid.lat
            features['az'], features['el'] = lons, lats
            current_lon, current_lat = current_ra, current_dec
            
        features['angular_distance_to_pointing'] = self.hpGrid.get_angular_separations(lon=current_lon, lat=current_lat)
        features['ha'] = self.hpGrid.get_hour_angle(time=timestamp)
        features['airmass'] = self.hpGrid.get_airmass(timestamp)
        features['moon_distance'] = self.hpGrid.get_source_angular_separations('moon', time=timestamp)
        
        if self._has_historical_features:
            if not self.hpGrid.is_azel:
                max_s_visits_arr = np.maximum(self._max_n_visits_arr, self._max_s_visits_arr)
                nfields_n = np.bincount(self._bins_membership_arr, weights=self._in_n_plan, minlength=self.nbins)
                active_bins_n = nfields_n > 0

                # Get number of unvisited fields in each bin - bins below horizon have 0 fields unvisited
                s_unvisited = np.bincount(self._bins_membership_arr, weights=(self._s_visits_cur == 0) & self._in_s_plan, minlength=self.nbins)
                n_unvisited = np.bincount(self._bins_membership_arr, weights=(self._n_visits_cur == 0) & self._in_n_plan, minlength=self.nbins)

                # Get number of incomplete fields in each bin
                s_incomplete_mask = (self._s_visits_cur < max_s_visits_arr) & self._in_s_plan
                n_incomplete_mask = (self._n_visits_cur < self._max_n_visits_arr) & self._in_n_plan
                s_incomplete = np.bincount(self._bins_membership_arr, weights=s_incomplete_mask, minlength=self.nbins)
                n_incomplete = np.bincount(self._bins_membership_arr, weights=n_incomplete_mask, minlength=self.nbins)
        
                # Create a zero-filled array for the results
                for key in ['survey_num_unvisited_fields', 'night_num_unvisited_fields', 
                            'survey_num_incomplete_fields', 'night_num_incomplete_fields']:
                    features[key] = np.zeros(self.nbins, dtype=np.float32) # bins with no viable fields get sentinel value 0
                
                # Do division in-place (bypasses runtimewarning error )
                np.divide(s_unvisited, self._nfields_s, out=features['survey_num_unvisited_fields'], where=self._active_bins_s)
                np.divide(n_unvisited, nfields_n, out=features['night_num_unvisited_fields'], where=active_bins_n)
                np.divide(s_incomplete, self._nfields_s, out=features['survey_num_incomplete_fields'], where=self._active_bins_s)
                np.divide(n_incomplete, nfields_n, out=features['night_num_incomplete_fields'], where=active_bins_n)
        
                # Min tiling
                s_tiling_all = np.full_like(self._s_visits_cur, 2.0, dtype=np.float32)
                n_tiling_all = np.full_like(self._n_visits_cur, 2.0, dtype=np.float32)
                # current_num_visits_field / max_num_visits_field only where max_num_visits_field > 0 ie, in the plan
                np.divide(self._s_visits_cur, max_s_visits_arr, out=s_tiling_all, where=self._in_s_plan)
                np.divide(self._n_visits_cur, self._max_n_visits_arr, out=n_tiling_all, where=self._in_n_plan)
                
                s_mins = np.full(self.nbins, 2.0, dtype=np.float32)
                n_mins = np.full(self.nbins, 2.0, dtype=np.float32)
                np.minimum.at(s_mins, self._bins_membership_arr, s_tiling_all)
                np.minimum.at(n_mins, self._bins_membership_arr, n_tiling_all)
                
                # Reset bins with no fields back to -0.1
                s_mins[s_mins > 1.0] = -1.0
                n_mins[n_mins > 1.0] = -1.0
                features['survey_min_tiling'] = s_mins
                features['night_min_tiling'] = n_mins
            else:
                # Reset at each timestep since fields' bin memberships change over time
                az, el = ephemerides.equatorial_to_topographic(ra=self._ra_arr, dec=self._dec_arr, time=timestamp)
                bins = self.hpGrid.ang2idx(lon=az, lat=el) # Bin membership of each field
                bins = np.array([b if b is not None else -1 for b in bins], dtype=np.int32)
                valid_mask = (el > 0) & (bins != -1)
                valid_bins = bins[valid_mask].astype(np.int32)

                in_s_plan = self._max_s_visits_arr[valid_mask] > 0
                # in_n_plan = self._max_n_visits_arr[valid_mask] > 0

                bin_count_s = np.bincount(valid_bins, weights=in_s_plan, minlength=self.nbins)
                # bin_count_n = np.bincount(valid_bins, weights=in_n_plan, minlength=self.nbins)
                
                active_bins_s = bin_count_s > 0
                # active_bins_n = bin_count_n > 0

                # Get field counts and max field counts over 
                v_survey_counts = self._s_visits_cur[valid_mask].astype(np.int32)
                # v_night_counts = self._n_visits_cur[valid_mask].astype(np.int32)
                v_max_visits_survey = self._max_s_visits_arr[valid_mask]
                # v_max_visits_night = self._max_n_visits_arr[valid_mask]
                
                # Re-create the plan masks
                in_s_plan = v_max_visits_survey > 0
                # in_n_plan = v_max_visits_night > 0

                for key_n, key_s, mask_n, mask_s in [
                    # Must be unvisited AND in the respective plan
                    ('night_num_unvisited_fields', 'survey_num_unvisited_fields', 
                    # (v_night_counts == 0) & in_n_plan, 
                    (v_survey_counts == 0) & in_s_plan),
                    
                    # Must be incomplete AND in the respective plan
                    ('night_num_incomplete_fields', 'survey_num_incomplete_fields', 
                    # (v_night_counts < v_max_visits_night) & in_n_plan, 
                    (v_survey_counts < v_max_visits_survey) & in_s_plan)
                    ]:
                    res_n, res_s = np.zeros(self.nbins, dtype=np.float32), np.zeros(self.nbins, dtype=np.float32)
                    
                    # Use the correct denominators and active masks!
                    # np.divide(np.bincount(valid_bins, weights=mask_n, minlength=self.nbins), bin_count_n, out=res_n, where=active_bins_n)
                    np.divide(np.bincount(valid_bins, weights=mask_s, minlength=self.nbins), bin_count_s, out=res_s, where=active_bins_s)
                    
                    res_n[~active_bins_n] = 0.
                    res_s[~active_bins_s] = 0.
                    
                    features[key_n] = res_n
                    features[key_s] = res_s

                # Min Tiling 
                s_tiling_all = np.full_like(v_survey_counts, 2.0, dtype=np.float32)
                # n_tiling_all = np.full_like(v_night_counts, 2.0, dtype=np.float32)
                
                np.divide(v_survey_counts, v_max_visits_survey, out=s_tiling_all, where=in_s_plan)
                # np.divide(v_night_counts, v_max_visits_night, out=n_tiling_all, where=in_n_plan)
                    
                s_mins, n_mins = np.full(self.nbins, 2.0, dtype=np.float32), np.full(self.nbins, 2.0, dtype=np.float32)
                np.minimum.at(s_mins, valid_bins, s_tiling_all)
                # np.minimum.at(n_mins, valid_bins, n_tiling_all)
                    
                s_mins[~active_bins_s] = -0.1
                # n_mins[~active_bins_n] = -0.1
                    
                features['survey_min_tiling'] = s_mins
                # features['night_min_tiling'] = n_mins

        # Normalize periodic features here and add as df cols
        if self.do_cyclical_norm:
            for feat_name in self.base_bin_feature_names:
                if any(string in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                    if feat_name in features.keys():
                        features[f'{feat_name}_cos'] = np.cos(features[feat_name])
                        features[f'{feat_name}_sin'] = np.sin(features[feat_name])
                    else:
                        raise ValueError(f"{feat_name} was not calculated in _calculate_bin_features. Is this feature implemented?")
        
        expanded_feat_names = expand_feature_names_for_cyclic_norm(self.base_bin_feature_names, cyclical_feature_names=self.cyclical_feature_names)
        bin_state = np.vstack([features.get(feat_name, np.nan) for feat_name in expanded_feat_names]).T
        return bin_state

    def _get_state(self):
        global_state, bin_state = self._global_state, self._bin_state
        global_state_copy = global_state.copy()
        bin_state_copy = bin_state.copy()
        # state_normed = self._do_noncyclic_normalizations(state=state_copy)
        global_state_normed = normalize_noncyclic_features(
                            state=np.array(global_state_copy),
                            state_feature_names=self.state_feature_names,
                            max_norm_feature_names=self.max_norm_feature_names,
                            ang_distance_norm_feature_names=self.ang_distance_feature_names,
                            do_inverse_norm=self.do_inverse_norm,
                            do_max_norm=self.do_max_norm,
                            do_ang_distance_norm=self.do_ang_distance_norm,
                            bin_space=self.bin_space,
                            fix_nans=True
                        )
        if self.include_bin_features:
            bin_state_normed = normalize_noncyclic_features(
                state=np.array(bin_state_copy)[np.newaxis, ...], # add axis for function
                state_feature_names=self.bin_feature_names,
                max_norm_feature_names=self.max_norm_feature_names,
                ang_distance_norm_feature_names=self.ang_distance_feature_names,
                do_inverse_norm=self.do_inverse_norm,
                do_max_norm=self.do_max_norm,
                do_ang_distance_norm=self.do_ang_distance_norm,
                bin_space=self.bin_space,
                fix_nans=True
            )
            bin_state_normed = bin_state_normed[0] # remove axis
        else:
            bin_state_normed = np.array([])
        self._global_state = global_state_normed.astype(np.float32)
        self._bin_state = bin_state_normed.astype(np.float32)

        # logger.debug(f"Global state max, min {self._global_state.max()}, {self._global_state.min()}")
        # logger.debug(f"State above max: {np.where(self._global_state > 2)}")            
        # logger.debug(f"State below min: {np.where(self._global_state <-2)}")
        # if self.include_bin_features:
        #     logger.debug(f"Bins state max, min {self._bin_state.max()}, {self._bin_state.min()}")
        #     logger.debug(f"State above max: {np.where(self._bin_state > 2)}")            
        #     logger.debug(f"State below min: {np.where(self._bin_state <-2)}")
        return {"global_state": self._global_state, "bin_state": self._bin_state}

    def _get_info(self):
        """
        Compute auxiliary information for debugging and constrained action spaces.

        Returns
        -------
            dict: A dictionary containing the current action mask.
        """
        return {'action_mask': self._action_mask.copy(), 
                'visited': self._s_visits_cur.copy(),
                'timestamp': self._timestamp,
                'is_new_night': bool(self._is_new_night),
                'night_idx': int(self._night_idx),
                'bin': int(self._bin_num),
                'field_id': int(self._field_id),
                'valid_fields_per_bin': self._valid_fields_per_bin,
        }
    
    def _get_termination_status(self):
        """
        Checks if the episode has reached its termination condition.

        Termination occurs when the total number of observations for the night
        (based on the dataset) has been met, or, when all fields have been completely
        visited.

        Returns
        -------
            bool: True if the episode is terminated, False otherwise.
        """
        all_nights_completed = self._night_idx >= self.max_nights
        all_fields_visited = all(np.array([self._s_visits_cur[fid] >= self.field_lookup['n_visits'][str(fid)] for fid in self._fids]))
        terminated = all_nights_completed or all_fields_visited
        return terminated
    
    def _update_action_masks(self, timestamp, field_lookup, field_ids, ras, decs, hpGrid, visited):
        # Mask fields which are completed 
        mask_completed_fields = np.array([visited[fid] < field_lookup['n_visits'][str(fid)] for fid in field_ids], dtype=bool) #TODO can probably track visits without repeating this operation
        fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=ras, dec=decs, time=timestamp)
        # Mask fields below airmass limit
        mask_fields_below_horizon = fields_el > 0
        airmass = np.zeros_like(fields_el)
        airmass[mask_fields_below_horizon] = 1 / np.cos(90 * units.deg - fields_el[mask_fields_below_horizon])
        airmass[~mask_fields_below_horizon] = 10
        mask_airmass_lim = airmass < self._airmass_limit
        sel_valid_fields = mask_completed_fields & mask_airmass_lim
        self._sel_valid_fields = sel_valid_fields
        # Get bins which are below airmass limit, masking completed bins
        valid_fids = field_ids[sel_valid_fields]
        if hpGrid.is_azel:
            valid_field_bins = hpGrid.ang2idx(lon=fields_az[sel_valid_fields], lat=fields_el[sel_valid_fields])
        else:
            valid_field_bins = hpGrid.ang2idx(lon=ras[sel_valid_fields], lat=decs[sel_valid_fields])
        self._valid_fields_per_bin = defaultdict(list)
        action_mask = np.zeros(shape=self.nbins, dtype=bool)
        for fid, bin_idx in zip(valid_fids, valid_field_bins):
            if bin_idx is not None:
                b = int(bin_idx)
                action_mask[b] = True
                self._valid_fields_per_bin[b].append(fid)
        if 'filter' in self.bin_space:
            action_mask = np.repeat(action_mask[:, np.newaxis], NUM_FILTERS, axis=1).flatten() #TODO 2. in todoist
        self._action_mask = action_mask
        logger.debug(f'environment action_mask action_mask.shape: {action_mask.shape}')
        return action_mask
    
    def _fast_forward_to_timestamp(self, timestamp, ras, decs, visited, max_visits):
        incomplete_mask = visited < max_visits
        incomplete_ras = ras[incomplete_mask]
        incomplete_decs = decs[incomplete_mask]
        
        # If all fields complete, survey is terminated
        if len(incomplete_ras) == 0:
            return timestamp
        test_timestamp = timestamp
        step_size = 60*5 # inspect visibility every 5 mins

        while test_timestamp < self._sunrise_time:
            test_timestamp += step_size
            _, fields_el = ephemerides.equatorial_to_topographic(ra=incomplete_ras, dec=incomplete_decs, time=test_timestamp)
            airmass = 1 / np.cos(90 * units.deg - fields_el[fields_el > 0])
            if np.any(airmass < 1.2):
                return test_timestamp
        # If fields never above horizon, return sunrise time
        return self._sunrise_time
    
    def _get_filter_mask(self, ):
        pass

    def _get_slew_time(self, last_fid, current_fid):
        if last_fid == -1:
            blanco = ephemerides.blanco_observer(time=self._timestamp)
            last_pos = np.array(blanco.radec_of('0',  '90'))
        else:
            last_pos = self.field_lookup['ra'][str(last_fid)], self.field_lookup['dec'][str(last_fid)]
        current_pos = self.field_lookup['ra'][str(current_fid)], self.field_lookup['dec'][str(current_fid)]
        distance = geometry.angular_separation(last_pos, current_pos)
        slew_time = geometry.blanco_slew_time(distance)
        return slew_time

    def _get_exposure_time(self, field_id):
        if int(field_id) < 0:
            return 0.0
        return self.field_lookup['exptime'][str(field_id)]
