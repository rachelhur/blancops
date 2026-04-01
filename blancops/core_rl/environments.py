from collections import defaultdict
from pathlib import Path
from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np
import pandas as pd
import math

from blancops.data_processing.data_processing import expand_feature_names_for_cyclic_norm
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.math import units
from blancops.ephemerides import ephemerides
from blancops.data_processing.offline_dataset import setup_feature_names
from blancops.data_processing.features import normalize_timestamp, get_nautical_twilight, normalize_noncyclic_features
from blancops.data_processing.constants import *
from blancops.math import geometry

from astropy.time import Time
from datetime import datetime, timezone, timedelta
import pickle
import json
from einops import rearrange


from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(__name__)

class BaseTelescopeEnv(gym.Env, ABC):
    def __init__(self):
        super().__init__()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._init_to_first_state()
        state = self._get_state()
        info = self._get_info()
        # if not self.observation_space.contains(state):
        #     for (s_name, s), feat_names in zip(state.items(), [self.global_feature_names, self.bin_feature_names]):
        #         logger.warning(s_name)
        #         if s.max() > 2 or s.min() < -2:
        #             for i, col in enumerate(state['bin_state'].T):
        #                 logger.warning(f"Feature `{feat_names[i]}` is out of bounds vals {col[col < -2]}")
        #                 logger.warning(f"Feature `{feat_names[i]}` is out of bounds vals {col[col > 2]}")
        #                 # logger.warning(f"Feature `{feat_names[i]}` array: {col.max(), col.min(), col}")
        #     raise ValueError
        return state, info
    
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def _init_to_first_state(self):
        """Initializes the environment state for a new episode."""
        pass

    @abstractmethod
    def _update_action_masks(self): # Note: ensure naming consistency (action_mask vs action_masks)
        """Updates action masks."""
        pass

    @abstractmethod
    def _update_state(self, action): 
        # Note: Your online env uses _update_state instead of _update_obs. 
        # ABCs will force you to unify this naming convention.
        """Updates the internal state based on the action taken."""
        pass

    @abstractmethod
    def _get_state(self):
        """Converts internal state into the formal observation."""
        pass

    @abstractmethod
    def _get_info(self):
        """Computes auxiliary information dictionary."""
        pass

    @abstractmethod
    def _get_termination_status(self):
        """Checks if episode has terminated."""
        pass
    
    @abstractmethod
    def _get_exposure_time(self):
        """Get exposure time"""
        pass

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

class BaseBlancoEnv(BaseTelescopeEnv, ABC):
    """
    Intermediate base class containing shared DECam-specific state and logic.
    """
    def __init__(self):
        super().__init__()

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

        is_new_night = self._ts >= np.min([self._sunrise_ts, self._night_end_ts])
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
    
    def _get_slew_time(self, last_fid, current_fid):
        if last_fid == ZENITH_FIELD_ID:
            blanco = ephemerides.blanco_observer(time=self._ts)
            last_pos = np.array(blanco.radec_of('0',  '90'))
        else:
            last_pos = self._ra_arr[last_fid], self._dec_arr[last_fid]
        current_pos = self._ra_arr[current_fid], self._dec_arr[current_fid]
        distance = geometry.angular_separation(last_pos, current_pos)
        slew_time = geometry.blanco_slew_time(distance)
        return slew_time

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
        if self.do_filt:
            all_fields_visited = np.all(self._s_filter_visits_cur >= self._max_s_filter_visits_arr)
        else:
            all_fields_visited = np.all(self._s_visits_cur >= self._max_s_visits_arr)
        # all_fields_visited = all(np.array([self._s_visits_cur[fid] >= self.field2maxvisits[fid] for fid in self._fids]))
        
        terminated = all_nights_completed or all_fields_visited
        if terminated:
            logger.info(f"Did not visit all fields" if not all_fields_visited else "Visited all fields! :)")
        return terminated

    def _update_state(self, action):
        """
        Updates the internal state variables based on the action taken.

        Args
        ----
            action (int): The chosen field ID to observe next.
        """
        bin_num, field_id, filter_idx = int(action['bin']), int(action['field_id']), int(action['filter_idx'])

        # --- OnlineEnv Logic --- #
        if bin_num == WAIT_SIGNAL:
            print('ENVIRONMENT RECEIVED A WAIT SIGNAL')
            self._ts = self._fast_forward(
                timestamp=self._ts,
                ras=self._ra_arr,
                decs=self._dec_arr,
                visited=self._s_visits_cur,
                max_visits=self._max_s_visits_arr
            )
            # Stay in same field and filter after waiting
            field_id = self._field_id
            filter_idx = self._filter_idx
        # --- OnlineEnv Logic --- #
        else:
            last_field_id = self._field_id
            exptime = self._get_exposure_time(field_id=str(field_id))
            slew_time = self._get_slew_time(last_field_id, field_id)
            self._ts += exptime + slew_time

            self._n_visits_cur[field_id] += 1
            self._s_visits_cur[field_id] += 1
            if self.do_filt:
                self._s_filter_visits_cur[field_id, filter_idx] += 1
                self._n_filter_visits_cur[field_id, filter_idx] += 1  
             
        self._bin_num = bin_num
        self._field_id = field_id
        self._filter_idx = filter_idx
        
        self._global_state = self._calculate_global_features(field_id=field_id, filter_idx=filter_idx, timestamp=self._ts,
                                                          sunset_ts=self._sunset_ts, sunrise_ts=self._sunrise_ts,
                                                          ra_arr=self._ra_arr, dec_arr=self._dec_arr,
                                                          )
        self._bin_state = self._calculate_bin_features(timestamp=self._ts) if self.include_bin_features else np.array([])
        # self._update_action_masks(timestamp=self._ts, field2maxvisits=self.field2maxvisits, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
                                                #   hpGrid=self.hpGrid, visited=self._s_visits_cur)
        self._update_action_masks()

    def _get_state(self):
        global_state, bin_state = self._global_state, self._bin_state
        global_state_normed = normalize_noncyclic_features(
                            state=np.array(global_state),
                            state_feature_names=self.state_feature_names,
                            max_norm_feature_names=self.max_norm_feature_names,
                            ang_distance_norm_feature_names=self.ang_distance_feature_names,
                            do_inverse_norm=self.do_inverse_norm,
                            do_max_norm=self.do_max_norm,
                            do_ang_distance_norm=self.do_ang_distance_norm,
                            fix_nans=True,
                            do_debug=False
                        )
        if self.include_bin_features:
            bin_state_normed = normalize_noncyclic_features(
                state=np.array(bin_state), # add axis for function
                state_feature_names=self.bin_feature_names,
                max_norm_feature_names=self.max_norm_feature_names,
                ang_distance_norm_feature_names=self.ang_distance_feature_names,
                do_inverse_norm=self.do_inverse_norm,
                do_max_norm=self.do_max_norm,
                do_ang_distance_norm=self.do_ang_distance_norm,
                fix_nans=True,
                do_debug=True
            )
            bin_state_normed = bin_state_normed
        else:
            bin_state_normed = np.array([])
        self._global_state = global_state_normed.astype(np.float32)
        self._bin_state = bin_state_normed.astype(np.float32)
        # for feat_name, row in zip(self.global_feature_names, self._global_state):
        #     print(feat_name, row.max(), row.min())
        # for feat_name, row in zip(self.bin_feature_names, self._bin_state.T):
        #     print(feat_name, row.max(), row.min())

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
        info_dict = {'action_mask': self._action_mask.copy(), 
                's_visited': self._s_visits_cur.copy(),
                'n_visited': self._n_visits_cur.copy(),
                'valid_fields_per_bin': self._valid_fields_per_bin,
                'timestamp': self._ts,
                'is_new_night': bool(self._is_new_night),
                'night_idx': int(self._night_idx),
                'bin': int(self._bin_num),
                'field_id': int(self._field_id),
        }
        if getattr(self, 'do_filt', True):
            info_dict['s_filter_visits'] = self._s_filter_visits_cur.copy()
            info_dict['max_s_filter_visits'] = self._max_s_filter_visits_arr.copy()
        return info_dict

    def _calculate_global_features(self, field_id, filter_idx, timestamp, sunset_ts, sunrise_ts, ra_arr, dec_arr):
        new_features = {}
        astro_time = Time(timestamp, format='unix', scale='utc')
        lst = astro_time.sidereal_time('apparent', longitude=BLANCO_LON)
        new_features['lst'] = lst.radian
        
        # --- OnlineEnv Logic --- #
        if field_id == ZENITH_FIELD_ID:
            blanco = ephemerides.blanco_observer(time=timestamp)
            ra, dec = new_features['lst'], blanco.lon
            new_features['ra'], new_features['dec'] = ra, dec 
            new_features['filter_wave'] = 0.
        else: # if field_id is real field or is wait signal
            ra = ra_arr[field_id]
            dec = dec_arr[field_id]
            new_features['ra'], new_features['dec'] = ra, dec
            new_features['filter_wave'] = 0 if (self._bin_num == WAIT_SIGNAL) or (not self.do_filt) else IDX2WAVE[filter_idx] / FILTERWAVENORM
        # --- OnlineEnv Logic --- #
        
        # new_features['ra'], new_features['dec'] = self.field2radec[field_id]
        new_features['az'], new_features['el'] = ephemerides.equatorial_to_topographic(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        new_features['ha'] = ephemerides.equatorial_to_hour_angle(ra=new_features['ra'], dec=new_features['dec'], time=timestamp)
        
        # precision issue where el can be slightly negative just before sunrise/sunset, causing issues with airmass calculation - cap at pi/2
        new_features['el'] = max(new_features['el'], 0)
        new_features['el'] = min(new_features['el'], np.pi / 2)

        cos_zenith = np.cos(np.pi / 2 - new_features['el'])
        new_features['airmass'] = 1.0 / cos_zenith #if cos_zenith > 0 else 99.0

        new_features['sun_ra'], new_features['sun_dec'] = ephemerides.get_source_ra_dec(source='sun', time=timestamp)
        new_features['sun_az'], new_features['sun_el'] = ephemerides.equatorial_to_topographic(ra=new_features['sun_ra'], dec=new_features['sun_dec'], time=timestamp)
        new_features['moon_ra'], new_features['moon_dec'] = ephemerides.get_source_ra_dec(source='moon', time=timestamp)
        new_features['moon_az'], new_features['moon_el'] = ephemerides.equatorial_to_topographic(ra=new_features['moon_ra'], dec=new_features['moon_dec'], time=timestamp)
        for filt in FILTER2WAVE.keys():
            if filt != ZENITH_FILTER:
                new_features[f'sky_brightness_{filt}'] = estimate_sky_brightness(time=timestamp, ra=ra, dec=dec, band=filt)
        if sunrise_ts == sunset_ts:
            raise AssertionError("Sunrise and sunset time is equal. Check night_str argument - it should be a time between sunset and sunrise")
            # new_features['time_fraction_since_start'] = 0
        else:
            new_features['time_fraction_since_start'] = normalize_timestamp(timestamp, sunset_timestamp=sunset_ts, sunrise_timestamp=sunrise_ts)


        for feat_name in self.base_global_feature_names:
            if any(string in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
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

        # --- OnlineEnv Logic --- #
        if self._bin_num == WAIT_SIGNAL:
            blanco = ephemerides.blanco_observer(time=timestamp)
            pointing_radec = np.array(blanco.radec_of('0',  '90'))
        else:
            pointing_radec = np.array([self._ra_arr[self._field_id], self._dec_arr[self._field_id]])
        # --- OnlineEnv Logic --- #

        if self.hpGrid.is_azel:
            lons, lats = ephemerides.topographic_to_equatorial(az=self.hpGrid.lon, el=self.hpGrid.lat, time=timestamp)
            features['az'], features['el'] = self.hpGrid.lon, self.hpGrid.lat
            features['ra'], features['dec'] = lons, lats
            current_lon, current_lat = ephemerides.equatorial_to_topographic(ra=pointing_radec[0], dec=pointing_radec[1], time=timestamp)
        else:
            lons, lats = ephemerides.equatorial_to_topographic(ra=self.hpGrid.lon, dec=self.hpGrid.lat, time=timestamp)
            features['ra'], features['dec'] = self.hpGrid.lon, self.hpGrid.lat
            features['az'], features['el'] = lons, lats
            current_lon, current_lat = pointing_radec[0], pointing_radec[1]
        
        # One-shot calculations
        features['angular_distance_to_pointing'] = self.hpGrid.get_angular_separations(lon=current_lon, lat=current_lat)
        features['ha'] = self.hpGrid.get_hour_angle(time=timestamp)
        features['airmass'] = self.hpGrid.get_airmass(timestamp)
        features['moon_distance'] = self.hpGrid.get_source_angular_separations('moon', time=timestamp)
        
        if self._has_historical_features:
            sentinel_val = AZEL_BIN_FEAT_SENTINEL if self.hpGrid.is_azel else RADEC_BIN_FEAT_SENTINEL

            # Setup active masks depending on coordinate space
            if not self.hpGrid.is_azel:
                bins_mem = self._bins_membership_arr
                v_mask = slice(None) # select everything - assume all input fields are in survey plan
                act_s = self._active_bins_s
                act_n = (np.bincount(bins_mem, weights=self._in_n_plan, minlength=self.nbins) > 0) 
                if self.do_filt:
                    act_s_filter = np.zeros((self.nbins, self.nfilters), dtype=bool)
                    act_n_filter = np.zeros((self.nbins, self.nfilters), dtype=bool)
                    for f in range(self.nfilters):
                        act_s_filter[:, f] = np.bincount(bins_mem, weights=(self._max_s_filter_visits_arr[:, f] > 0), minlength=self.nbins) > 0
                        act_n_filter[:, f] = np.bincount(bins_mem, weights=(self._max_n_filter_visits_arr[:, f] > 0), minlength=self.nbins) > 0
            else:
                az, el = ephemerides.equatorial_to_topographic(ra=self._ra_arr, dec=self._dec_arr, time=timestamp)
                bins = self.hpGrid.ang2idx(lon=az, lat=el)
                bins = np.array([b if b is not None else ZENITH_BIN_NUM for b in bins], dtype=np.int32)
                
                v_mask = (el > 0) & (bins != ZENITH_BIN_NUM)
                bins_mem = bins[v_mask].astype(np.int32) # bins_mem acts as valid_bins

                in_s_plan = self._max_s_visits_arr[v_mask] > 0
                in_n_plan = self._max_n_visits_arr[v_mask] > 0
                
                act_s = np.bincount(bins_mem, weights=in_s_plan, minlength=self.nbins) > 0
                act_n = np.bincount(bins_mem, weights=in_n_plan, minlength=self.nbins) > 0

                if self.do_filt:
                    act_s_filter = np.zeros((self.nbins, self.nfilters), dtype=bool)
                    act_n_filter = np.zeros((self.nbins, self.nfilters), dtype=bool)
                    for f in range(self.nfilters):
                        act_s_filter[:, f] = np.bincount(bins_mem, weights=(self._max_s_filter_visits_arr[v_mask, f] > 0), minlength=self.nbins) > 0
                        act_n_filter[:, f] = np.bincount(bins_mem, weights=(self._max_n_filter_visits_arr[v_mask, f] > 0), minlength=self.nbins) > 0
            
            # Field counts
            v_s_vis, v_n_vis = self._s_visits_cur[v_mask], self._n_visits_cur[v_mask]
            v_max_s, v_max_n = self._max_s_visits_arr[v_mask], self._max_n_visits_arr[v_mask]
            in_s_plan, in_n_plan = v_max_s > 0, v_max_n > 0
            max_s_vis_adj = np.maximum(v_max_n, v_max_s)

            # True denominators
            bc_n = np.bincount(bins_mem, weights=in_n_plan, minlength=self.nbins)
            bc_s = np.bincount(bins_mem, weights=in_s_plan, minlength=self.nbins)

            def assign_state(m_n, m_s, count_n, count_s, act_n_msk, act_s_msk, key_n, key_s):
                res_n, res_s = np.zeros(self.nbins, dtype=np.float32), np.zeros(self.nbins, dtype=np.float32)
                
                num_n = np.bincount(bins_mem, weights=m_n, minlength=self.nbins)
                num_s = np.bincount(bins_mem, weights=m_s, minlength=self.nbins)
                
                np.divide(num_n, count_n, out=res_n, where=act_n_msk)
                np.divide(num_s, count_s, out=res_s, where=act_s_msk)
                
                res_n[~act_n_msk] = 0.0 # Using 0.0 as your default sentinel
                res_s[~act_s_msk] = 0.0
                features[key_n] = res_n
                features[key_s] = res_s

            # Execute 1D
            assign_state((v_n_vis == 0) & in_n_plan, (v_s_vis == 0) & in_s_plan, bc_n, bc_s, act_n, act_s, 'night_num_unvisited_fields', 'survey_num_unvisited_fields')
            assign_state((v_n_vis < v_max_n) & in_n_plan, (v_s_vis < max_s_vis_adj) & in_s_plan, bc_n, bc_s, act_n, act_s, 'night_num_incomplete_fields', 'survey_num_incomplete_fields')

            s_til, n_til = np.full_like(v_s_vis, 2.0, dtype=np.float32), np.full_like(v_n_vis, 2.0, dtype=np.float32)
            np.divide(v_s_vis, max_s_vis_adj, out=s_til, where=in_s_plan)
            np.divide(v_n_vis, v_max_n, out=n_til, where=in_n_plan)
            s_mins, n_mins = np.full(self.nbins, 2.0, dtype=np.float32), np.full(self.nbins, 2.0, dtype=np.float32)
            np.minimum.at(s_mins, bins_mem, s_til); np.minimum.at(n_mins, bins_mem, n_til)
            
            s_mins[~act_s | (s_mins > 1.0)] = sentinel_val
            n_mins[~act_n | (n_mins > 1.0)] = sentinel_val
            
            features['survey_min_tiling'] = s_mins
            features['night_min_tiling'] = n_mins

            # Filter counts
            if self.do_filt:
                v_s_f_vis, v_n_f_vis = self._s_filter_visits_cur[v_mask], self._n_filter_visits_cur[v_mask]
                v_max_s_f, v_max_n_f = self._max_s_filter_visits_arr[v_mask], self._max_n_filter_visits_arr[v_mask]
                in_s_f_plan, in_n_f_plan = v_max_s_f > 0, v_max_n_f > 0
                max_s_f_vis_adj = np.maximum(v_max_n_f, v_max_s_f)
                
                s_f_mins, n_f_mins = np.full((self.nbins, self.nfilters), 2.0, dtype=np.float32), np.full((self.nbins, self.nfilters), 2.0, dtype=np.float32)
                s_f_til, n_f_til = np.full_like(v_s_f_vis, 2.0, dtype=np.float32), np.full_like(v_n_f_vis, 2.0, dtype=np.float32)
                np.divide(v_s_f_vis, max_s_f_vis_adj, out=s_f_til, where=in_s_f_plan)
                np.divide(v_n_f_vis, v_max_n_f, out=n_f_til, where=in_n_f_plan)

                for f, filt_name in self.idx2filter.items():
                    bc_n_f = np.bincount(bins_mem, weights=in_n_f_plan[:, f], minlength=self.nbins)
                    bc_s_f = np.bincount(bins_mem, weights=in_s_f_plan[:, f], minlength=self.nbins)

                    assign_state((v_n_f_vis[:, f] == 0) & in_n_f_plan[:, f], (v_s_f_vis[:, f] == 0) & in_s_f_plan[:, f], 
                                 bc_n_f, bc_s_f, act_n_filter[:, f], act_s_filter[:, f], f'night_num_unvisited_fields_{filt_name}', f'survey_num_unvisited_fields_{filt_name}')
                    assign_state((v_n_f_vis[:, f] < v_max_n_f[:, f]) & in_n_f_plan[:, f], (v_s_f_vis[:, f] < max_s_f_vis_adj[:, f]) & in_s_f_plan[:, f],
                                 bc_n_f, bc_s_f, act_n_filter[:, f], act_s_filter[:, f], f'night_num_incomplete_fields_{filt_name}', f'survey_num_incomplete_fields_{filt_name}')
                    
                    np.minimum.at(s_f_mins[:, f], bins_mem, s_f_til[:, f])
                    np.minimum.at(n_f_mins[:, f], bins_mem, n_f_til[:, f])
                    s_f_mins[~act_s_filter[:, f] | (s_f_mins[:, f] > 1.0), f] = sentinel_val
                    n_f_mins[~act_n_filter[:, f] | (n_f_mins[:, f] > 1.0), f] = sentinel_val
                    
                    features[f'survey_min_tiling_{filt_name}'] = s_f_mins[:, f]
                    features[f'night_min_tiling_{filt_name}'] = n_f_mins[:, f]
        
            # FEATURE VALIDATION CHECK
            self._validate_bin_features(features, sentinel_val)

        # Normalize periodic features here and add as df cols
        if self.do_cyclical_norm:
            for feat_name in self.base_bin_feature_names:
                if any(string in feat_name and 'frac' not in feat_name for string in self.cyclical_feature_names):
                    if feat_name in features.keys():
                        features[f'{feat_name}_cos'] = np.cos(features[feat_name])
                        features[f'{feat_name}_sin'] = np.sin(features[feat_name])
                    else:
                        raise ValueError(f"{feat_name} was not calculated in _calculate_bin_features. Is this feature implemented?")
        
        bin_states = np.array([features.get(key, np.nan) for key in self.bin_feature_names])
        bin_states = rearrange(bin_states, 'nfeats nbins -> nbins nfeats')
        # bin_state = np.vstack([features.get(feat_name, np.full(self.nbins, np.nan, dtype=np.float32)) for feat_name in self.bin_feature_names]).T
        # assert (bin_state != np.nan).all()
        # assert (bin_state != np.inf).all()

        return bin_states

    def _validate_bin_features(self, features, sentinel_value):
        if self.do_filt and self._has_historical_features:
            for f, filt_name in self.idx2filter.items():
                
                # Fetch features for the current filter
                unvisited = features.get(f'survey_num_unvisited_fields_{filt_name}')
                incomplete = features.get(f'survey_num_incomplete_fields_{filt_name}')
                min_tiling = features.get(f'survey_min_tiling_{filt_name}')

                if unvisited is not None and incomplete is not None and min_tiling is not None:
                    
                    # 1. Bounds Check
                    if np.any((unvisited < 0.0) | (unvisited > 1.0)):
                        bad_bins = np.where((unvisited < 0.0) | (unvisited > 1.0))[0]
                        raise RuntimeError(f"FATAL: 'survey_num_unvisited_fields_{filt_name}' out of bounds in bins {bad_bins}. Max: {np.max(unvisited)}, Min: {np.min(unvisited)}")
                    
                    if np.any((incomplete < 0.0) | (incomplete > 1.0)):
                        bad_bins = np.where((incomplete < 0.0) | (incomplete > 1.0))[0]
                        raise RuntimeError(f"FATAL: 'survey_num_incomplete_fields_{filt_name}' out of bounds in bins {bad_bins}. Max: {np.max(incomplete)}, Min: {np.min(incomplete)}")

                    # 2. Subset Rule: Unvisited MUST be <= Incomplete
                    subset_violation = unvisited > (incomplete + 1e-5)
                    if np.any(subset_violation):
                        bad_bins = np.where(subset_violation)[0]
                        raise RuntimeError(
                            f"FATAL LOGIC LEAK: In filter '{filt_name}', unvisited fraction strictly exceeds incomplete fraction in bins {bad_bins}.\n"
                            f"Unvisited vals: {unvisited[bad_bins]}\n"
                            f"Incomplete vals: {incomplete[bad_bins]}"
                        )

                    # 3. Tiling Floor: If unvisited > 0, min_tiling MUST be 0.0 
                    # Note: We ignore inactive bins where min_tiling is explicitly set to -1.0
                    active_bins = min_tiling != sentinel_value
                    has_unvisited_fields = unvisited > 1e-5
                    
                    # Intersect: Active bins that have unvisited fields
                    tiling_check_mask = active_bins & has_unvisited_fields
                    
                    if np.any(min_tiling[tiling_check_mask] > 1e-5):
                        bad_bins = np.where(tiling_check_mask & (min_tiling > 1e-5))[0]
                        raise RuntimeError(
                            f"FATAL LOGIC LEAK: In filter '{filt_name}', bins {bad_bins} have unvisited fields, "
                            f"but 'survey_min_tiling' is > 0.0 ({min_tiling[bad_bins]}). "
                            f"Min tiling MUST be 0 if unvisited fields exist."
                        )

    def _setup_action_and_obs_spaces(self):
        if self.include_bin_features:
            bin_state_shape = (self.nbins, self.bin_state_dim, )
        else:
            bin_state_shape = (0,)
    
        # Define observation space 
        self.observation_space = gym.spaces.Dict(
            {
                "global_state": gym.spaces.Box(-2, 2, shape=(self.state_dim,), dtype=np.float32),
                "bin_state": gym.spaces.Box(-2, 2, shape=bin_state_shape, dtype=np.float32),
            }
        )

        # Define action space
        smallest_sentinel = min([WAIT_SIGNAL, ZENITH_BIN_NUM])
        self.action_space = gym.spaces.Dict(
            {
                "bin": gym.spaces.Discrete(self.nbins - smallest_sentinel, start=min([WAIT_SIGNAL, ZENITH_BIN_NUM])),
                "field_id": gym.spaces.Discrete(len(self._fids) - smallest_sentinel, start=min([WAIT_SIGNAL, ZENITH_FIELD_ID])),
                "filter_idx": gym.spaces.Discrete(NUM_FILTERS - smallest_sentinel, start=min([WAIT_SIGNAL, ZENITH_FIELD_ID]))
            }
        )       

    def _get_half_night_duration(self, sunset_ts, sunrise_ts):
        return (sunrise_ts - sunset_ts) // 2

    def _update_action_masks(self):
            fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=self._ra_arr, dec=self._dec_arr, time=self._ts)
            mask_fields_below_horizon = fields_el > 0

            airmass_limit = getattr(self, '_airmass_limit', None)
            if airmass_limit is not None:
                airmass = np.zeros_like(fields_el)
                airmass[mask_fields_below_horizon] = 1 / np.cos(90 * units.deg - fields_el[mask_fields_below_horizon])
                airmass[~mask_fields_below_horizon] = 10  # Sentinel high airmass for below horizon
                mask_visibility = airmass < airmass_limit
            else:
                mask_visibility = mask_fields_below_horizon # Fallback for offline

            if self.do_filt:
                sel_valid_ff = self._s_filter_visits_cur < self.fieldfilter2maxvisits
                sel_valid_ff &= mask_visibility[:, np.newaxis] 
                sel_valid_fields = sel_valid_ff.any(axis=1)
            else:
                if isinstance(self.field2maxvisits, dict):
                    mask_completed_fields = np.array([self._s_visits_cur[fid] < self.field2maxvisits[fid] for fid in self._fids], dtype=bool)
                else:
                    mask_completed_fields = self._s_visits_cur < self.field2maxvisits
                sel_valid_fields = mask_completed_fields & mask_visibility

            if self.hpGrid.is_azel:
                valid_field_bins = self.hpGrid.ang2idx(lon=fields_az[sel_valid_fields], lat=fields_el[sel_valid_fields])
            else:
                valid_field_bins = self.hpGrid.ang2idx(lon=self._ra_arr[sel_valid_fields], lat=self._dec_arr[sel_valid_fields])
            
            valid_bin_mask = np.array([b is not None for b in valid_field_bins], dtype=bool)
            clean_bins = np.array(valid_field_bins)[valid_bin_mask].astype(int)

            # 5. Mask Construction
            if self.do_filt:
                action_mask = np.zeros(shape=(self.nbins, NUM_FILTERS), dtype=bool)
                clean_ff = sel_valid_ff[sel_valid_fields][valid_bin_mask]
                np.logical_or.at(action_mask, clean_bins, clean_ff)
                action_mask = action_mask.flatten()
            else:
                action_mask = np.zeros(shape=self.nbins, dtype=bool)
                action_mask[clean_bins] = True

            # 6. Track Valid Fields Mapping
            valid_fids = self._fids[sel_valid_fields]
            clean_fids = valid_fids[valid_bin_mask] 
            self._valid_fields_per_bin = defaultdict(list)
            for b, fid in zip(clean_bins, clean_fids):
                self._valid_fields_per_bin[b].append(fid)

            self._action_mask = action_mask
            return action_mask

    def _get_exposure_time(self, field_id=None):
        if int(field_id) < 0:
            return 0.0
        elif (field_id is None) or getattr(self, 'field_lookup', None) is None:
            return 90.0
        return self.field_lookup['exptime'].values[int(field_id)]

class OfflineBlancoTestingEnv(BaseBlancoEnv):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, gcfg, cfg, max_nights=None, exp_time=90., global_pd_nightgroup=None, zenith_bin_states=None):
        """
        Args
        ----
            dataset: An object (assumed to be OfflineDECamDataset instance) containing
                     static environment parameters and observation data.
        """
        assert cfg is not None, "Either cfg or test_dataset must be passed"
        
        # Assign static attributes
        self.exp_time = exp_time
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES']
        self.max_norm_feature_names = gcfg['features']['MAX_NORM_FEATURE_NAMES']
        self.ang_distance_feature_names = gcfg['features']['ANG_DISTANCE_NORM_FEATURE_NAMES']
        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_max_norm = cfg['data']['do_max_norm']
        self.do_inverse_norm = cfg['data']['do_inverse_norm']
        self.do_ang_distance_norm = cfg['data']['do_ang_distance_norm']
        self.include_bin_features = len(cfg['data']['bin_features']) > 0
        self.action_space = cfg['data']['action_space']
        nside = cfg['data']['nside']
        self.hpGrid = ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in self.action_space))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._grid_network = cfg['model']['grid_network']
        self._has_historical_features = any(sub in main_str for main_str in cfg['data']['bin_features'] 
                                           for sub in ['num_unvisited_fields', 'num_incomplete_fields', 'min_tiling'])
        self.do_filt = 'filter' in self.action_space
        
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
        
        if self._grid_network is None:
            self.state_feature_names = self.global_feature_names + self.bin_feature_names
        elif self._grid_network in GRID_NETWORKS:
            self.state_feature_names = self.global_feature_names
        
        self.global_pd_nightgroup = global_pd_nightgroup
        self.zenith_bin_states = zenith_bin_states

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
        self._global_state = [global_first_row[feat_name] for feat_name in self.global_feature_names]

        # Get field visit counts at start of night
        self._s_visits_cur = self.night2fieldvisithistory[night][self._fids].copy().astype(np.int32)
        self._n_visits_cur = np.zeros(self.nfields, dtype=np.int32)
        
        # Get field filter visit counts at start of night
        if self.do_filt:
            self._s_filter_visits_cur = self.night2filtvisithistory[night].copy()
            self._n_filter_visits_cur = np.zeros((self.nfields, self.nfilters), dtype=np.int32)

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
        # self._update_action_masks(self._ts, field2maxvisits=self.field2maxvisits, field_ids=self._fids, ras=self._ra_arr, decs=self._dec_arr, 
        #                                           hpGrid=self.hpGrid, visited=self._s_visits_cur)
    # def _update_action_masks(self, timestamp, field2maxvisits, field_ids, ras, decs, hpGrid, visited):
    #     # Mask fields which are completed 
    #     mask_completed_fields = np.array([visited[fid] < field2maxvisits[fid] for fid in field_ids], dtype=bool) #TODO can probably track visits without repeating this operation
    #     fields_az, fields_el = ephemerides.equatorial_to_topographic(ra=ras, dec=decs, time=timestamp)
    #     # Mask fields below horizon
    #     mask_fields_below_horizon = fields_el > 0
    #     sel_valid_fields = mask_completed_fields & mask_fields_below_horizon
    #     # Get bins which are below horizon, masking completed bins
    #     valid_fids = field_ids[sel_valid_fields]
    #     if hpGrid.is_azel:
    #         valid_field_bins = hpGrid.ang2idx(lon=fields_az[sel_valid_fields], lat=fields_el[sel_valid_fields])
    #     else:
    #         valid_field_bins = hpGrid.ang2idx(lon=ras[sel_valid_fields], lat=decs[sel_valid_fields])
    #     self._valid_fields_per_bin = defaultdict(list)
    #     action_mask = np.zeros(shape=self.nbins, dtype=bool)
    #     for fid, bin_idx in zip(valid_fids, valid_field_bins):
    #         if bin_idx is not None:
    #             b = int(bin_idx)
    #             action_mask[b] = True
    #             self._valid_fields_per_bin[b].append(fid)

    #     if 'filter' in self.action_space:
    #         action_mask = np.repeat(action_mask[:, np.newaxis], NUM_FILTERS, axis=1).flatten() #TODO 2. in todoist
    #     self._action_mask = action_mask
    #     return action_mask
    
class OnlineBlancoEnv(BaseBlancoEnv):
    """
    A concrete Gymnasium environment implementation compatible with OfflineDataset.
    """
    def __init__(self, gcfg, cfg, observing_night_strs, data_dir, field2radec, max_nights=0, horizon='-12', airmass_limit=1.4):
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
        self.action_space = cfg['data']['action_space']
        nside = cfg['data']['nside']
        self.hpGrid = None if cfg['data']['bin_method'] != 'healpix' else ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in self.action_space))
        self.nbins = len(self.hpGrid.idx_lookup)
        self._grid_network = cfg['model']['grid_network']
        self._has_historical_features = any(sub in main_str for main_str in cfg['data']['bin_features'] 
                                           for sub in ['num_unvisited_fields', 'num_incomplete_fields', 'min_tiling'])
        self.horizon = horizon
        self.max_nights = max(len(observing_night_strs), max_nights) - 1
        
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
        self.field2radec = field2radec
                
        self._fids = np.unique(self.field_lookup['field_id'].to_numpy())
        self.nfields = len(self._fids)
        assert np.array_equal(self._fids, np.arange(len(self._fids))), "Field IDs must be perfectly sequential and start at 0."
        
        self._ra_arr = self.field2radec[:, 0]
        self._dec_arr = self.field2radec[:, 1]
        self._max_s_visits_arr = np.bincount(self.field_lookup['field_id'], weights=self.field_lookup['n_visits']).astype(int)

        # Get filter lookup tables
        if self.do_filt:
            self._filter_idx_arr = self.field_lookup['filter_idx'].unique()
            self.fieldfilter2maxvisits = np.zeros((self.nfields, NUM_FILTERS)) # shape = (nfields, nfilters)
            for filt_idx in self._filter_idx_arr:
                np.add.at(self.fieldfilter2maxvisits, (self._fids, filt_idx), self.field_lookup['n_visits'][self.field_lookup['filter_idx'] == filt_idx].to_numpy())

            self.nfilters = len(FILTER2IDX)
            self.idx2filter = {v: k for k, v in FILTER2IDX.items()}
            self._max_s_filter_visits_arr = np.array([self.fieldfilter2maxvisits[fid] for fid in self._fids], dtype=np.int32)
            self._s_filter_visits_cur = np.zeros((len(self._fids), self.nfilters))
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
        if self._grid_network is None:
            self.state_feature_names = self.global_feature_names + self.bin_feature_names
        elif self._grid_network in GRID_NETWORKS:
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
        self._s_visits_cur = np.zeros(self.nfields, dtype=np.int32)
        self._is_new_night = True
        self._start_new_night()
    
    def _start_new_night(self):
        self._night_idx += 1
        if self._night_idx >= self.max_nights:
            return
        
        # global features
        night_dt, night_portion = self._night_info[self._night_idx]
        # night_ts = night_dt.timestamp()
        self._sunset_ts = math.ceil(get_nautical_twilight(night_dt, 'set', self.horizon))
        self._sunrise_ts = math.ceil(get_nautical_twilight(night_dt, 'rise', self.horizon))
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
            self._bin_state = self._calculate_bin_features(timestamp=self._ts)
            #self._max_n_visits_arr = np.bincount(self._fids[night_fids], minlength=self.nfields)
            self._in_n_plan = self._max_n_visits_arr > 0

            if self.do_filt:
                self._max_n_filter_visits_arr = np.zeros((self.nfields, self.nfilters), dtype=np.int32)
                # np.add.at(self._max_n_filter_visits_arr, (0, 0), 1)

            if self._grid_network in GRID_NETWORKS:
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
        step_size = 60*5 # inspect visibility every 5 mins

        while test_timestamp < self._night_end_ts:
            test_timestamp += step_size
            _, fields_el = ephemerides.equatorial_to_topographic(ra=incomplete_ras, dec=incomplete_decs, time=test_timestamp)
            fields_el = np.atleast_1d(fields_el)
            airmass = 1 / np.cos(90 * units.deg - fields_el[fields_el > 0])
            if np.any(airmass < self._airmass_limit):
                print(f"TIMESTAMP FAST FORWARDING {self._night_end_ts - test_timestamp}")
                return test_timestamp
        # If fields never above horizon, return sunrise time
        return min(test_timestamp, self._night_end_ts)
