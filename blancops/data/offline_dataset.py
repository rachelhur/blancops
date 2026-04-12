import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from collections import defaultdict
import gc

from tqdm import tqdm

from blancops.data.preprocessing import drop_rows_in_DECam_data
from blancops.ephemerides import ephemerides
import pandas as pd
import json
from torch.utils.data import random_split, RandomSampler
import pickle
import gc

from blancops.features.bin_features import calculate_bin_features
from blancops.features.features import *
from blancops.features.global_features import *
from blancops.data.constants import *
from blancops.data.data_processing import *

# Get the logger associated with this module's name (e.g., 'my_module')
import logging

from blancops.features.global_features import calculate_global_features
from blancops.math import geometry
logger = logging.getLogger(__name__)

def reward_func_v0():
    raise NotImplementedError

    
class OfflineDataset(torch.utils.data.Dataset):
    def __init__(self, df=None, cfg=None, gcfg=None,
                 specific_years=None, specific_months=None, specific_days=None, specific_filters=None,
                 field2maxvisits_path=None, field2radec_path=None, field2name_path=None,
                 night2filtervisithistory_path=None, fieldfilter2maxvisits=None,
                 z_score_stats=None, rel_norm_stats=None
                 ): 
        assert cfg is not None and gcfg is not None, "Must pass both cfg and gcfg"

        # ASSIGN STATIC ATTS
        do_cyclical_norm = cfg['data'].get('do_cyclical_norm', True)
        self.do_sin_norm = cfg['data'].get('do_sin_norm', True)
        self.do_log_norm = cfg['data'].get('do_log_norm', True)
        self.do_fractional_norm = cfg['data'].get('do_fractional_norm', True)
        self.do_z_score_norm = cfg['data'].get('do_z_score_norm', True)
        self.objects_to_remove = ["guide", "DES vvds","J0'","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec", "(outlier)"]
        self.reward_choice = cfg['data']['reward_choice']
        self._calculate_action_mask = cfg['model']['algorithm'] != 'BC' # should be False if using bc (to minimize data processing time), otherwise True
        self._action_architecture = cfg['model']['action_architecture']

        # GET NORMALIZATION FEATURE NAMES FROM CONFIG
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES'] if do_cyclical_norm else []
        self.sin_norm_feature_names = gcfg['features']['SIN_NORM_FEATURE_NAMES'] if self.do_sin_norm else []
        self.log_norm_feature_names = gcfg['features']['LOG_NORM_FEATURE_NAMES'] if self.do_log_norm else []
        self.fractional_norm_feature_names = gcfg['features']['FRACTIONAL_FEATURE_NAMES'] if self.do_fractional_norm else []
        self.z_score_feature_names = gcfg['features']['Z_SCORE_NORM_FEATURE_NAMES'] if self.do_z_score_norm else []
        self.local_mean_z_score_feature_names = gcfg['features']['LOCAL_MEAN_Z_SCORE_FEATURE_NAMES']

        # Get other configurations
        action_space = cfg['data']['action_space']
        nside = cfg['data']['nside']
        # logger.info(f'Including the following bin features: {cfg["data"].get("bin_features")}')
        # logger.info(f'Including the following global features: {cfg["data"]["global_features"]}')
        include_bin_features = len(cfg['data']['bin_features']) > 0
        self.include_bin_features = include_bin_features

        # SETUP HPGRID AND ACTION SPACE PARAMS
        self.hpGrid = ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in action_space))
        self.num_filters = NUM_FILTERS if 'filter' in action_space else 1

        # Set number of actions based on binning method
        self.nbins = len(self.hpGrid.lon)
        if action_space == 'filter':
            self.num_actions = self.num_filters
        elif action_space == 'radec' or action_space == 'azel':
            self.num_actions = self.nbins
        elif 'filter' in action_space and ('radec' in action_space or 'azel' in action_space):
            self.num_actions = self.nbins * self.num_filters
        else:
            raise NotImplementedError
        
        # GET ORDERED, FEATURE NAMES (accounting for cyclic norms as well)
        self.base_global_feature_names = cfg['data']['global_features'].copy()
        self.base_bin_feature_names = cfg['data']['bin_features'].copy()
        self.global_feature_names, self.bin_feature_names = setup_feature_names(base_global_feature_names=self.base_global_feature_names,
                                                                                base_bin_feature_names=self.base_bin_feature_names,
                                                                                cyclical_feature_names=self.cyclical_feature_names,
                                                                                do_cyclical_norm=do_cyclical_norm,
                                                                                )
        self.do_local_mean_z_score = any('rel_' in name for name in self.bin_feature_names)
        
        logger.info(f"Final features names {self.global_feature_names, self.bin_feature_names}")

        # LOAD LOOKUP TABLES
        if field2maxvisits_path is None:
            field2maxvisits_path = gcfg['paths']['TRAIN_DIR'] + gcfg['files']['FIELD2MAXVISITS_TRAIN']
        if field2radec_path is None:
            field2radec_path = gcfg['paths']['TRAIN_DIR'] + gcfg['files']['FIELD2RADEC']
        if field2name_path is None:
            field2name_path = gcfg['paths']['TRAIN_DIR'] + gcfg['files']['FIELD2NAME']
        if night2filtervisithistory_path is None:
            night2filtervisithistory_path = gcfg['paths']['TRAIN_DIR'] + gcfg['files']['NIGHT2FILTERVISITS']
        if fieldfilter2maxvisits is None:
            fieldfilter2maxvisits = gcfg['paths']['TRAIN_DIR'] + gcfg['files']['FIELDFILTER2MAXVISITS']

        with open(field2name_path, 'r') as f:
            field2name = json.load(f)
        with open(gcfg['paths']['TRAIN_DIR'] + gcfg['files']['NIGHT2FIELDVISITS'], 'rb') as f:
            night2fieldvisits = pickle.load(f)
        with open(field2radec_path, 'r') as f:
            field2radec = json.load(f)
            field2radec = {int(k): v for k, v in field2radec.items()}
        with open(field2maxvisits_path, 'r') as f:
            field2maxvisits = json.load(f)
            field2maxvisits = {int(k): v for k, v in field2maxvisits.items()}
        with open(night2filtervisithistory_path, 'rb') as f:
            night2filtervisithistory = pickle.load(f)
        with open(fieldfilter2maxvisits, 'rb') as f:
            fieldfilter2maxvisits = pickle.load(f)

        # PROCESS RAW DATA FRAME INTO GLOBAL FEATURES
        self._df = drop_rows_in_DECam_data(
            df,
            specific_years=cfg['data']['specific_years'] if specific_years is None else specific_years, 
            specific_months=cfg['data']['specific_months'] if specific_months is None else specific_months, 
            specific_days=cfg['data']['specific_days'] if specific_days is None else specific_days,
            specific_filters=cfg['data']['specific_filters'] if specific_filters is None else specific_filters,
            objects_to_remove=self.objects_to_remove
            )
        self._df = calculate_global_features(
            df=self._df, 
            field2name=field2name, 
            hpGrid=self.hpGrid, 
            base_global_feature_names=self.base_global_feature_names,
            cyclical_feature_names=self.cyclical_feature_names, 
            do_cyclical_norm=do_cyclical_norm
        )
        # PROCESS RAW DATA FRAME INTO BIN FEATURES
        if include_bin_features:
            bin_states = calculate_bin_features(
                pt_df=self._df,
                hpGrid=self.hpGrid, 
                base_bin_feature_names=self.base_bin_feature_names, 
                bin_feature_names=self.bin_feature_names, 
                cyclical_feature_names=self.cyclical_feature_names, 
                do_cyclical_norm=do_cyclical_norm,
                do_local_mean_z_score=self.do_local_mean_z_score,
                field2radec=field2radec,
                night2fieldvisits=night2fieldvisits,
                fieldfilter2maxvisits=fieldfilter2maxvisits,
                night2filtervisithistory=night2filtervisithistory,
                field2maxvisits=field2maxvisits,
                action_space=action_space
            )
        else:
            bin_states = None

        # Save night dates, total number of nights in dataset, and number of obs per night
        self.unique_nights = self._df['night'].unique()
        self.n_nights = self._df.groupby('night').ngroups

        # CONSTRUCT TRANSITIONS
        states, bin_states, self.actions, self.rewards, self.dones, self.action_masks, self.num_transitions \
            = self._construct_transitions(
            df=self._df, 
            bin_states=bin_states,  
            include_bin_features=include_bin_features, 
            action_space=action_space,
            )
        self.slew_distances = self._construct_slew_distances(self._df)
        
        logger.info(f"States shape: {states.shape}, Actions shape: {self.actions.shape}, Rewards shape: {self.rewards.shape}, Dones shape: {self.dones.shape}, Action masks shape: {self.action_masks.shape}")
        logger.info(f"Bin states shape: {bin_states.shape if bin_states is not None else None}")
        
        self._prenorm_bin_states = bin_states
        
        # SPLIT INTO TRAIN AND VAL        
        val_split = cfg['data'].get('val_split', 0.1)
        random_seed = cfg['data'].get('random_seed', 42)
        self.train_transition_idxs, self.val_transition_idxs = self._determine_split(val_split, random_seed)
        
        # get train indices
        train_c_idxs = self.curr_compact_idxs[self.train_transition_idxs]
        train_n_idxs = self.next_compact_idxs[self.train_transition_idxs]
        train_state_idxs = np.unique(np.concatenate([train_c_idxs, train_n_idxs]))

        # NORMALIZE STATES
        state_feature_names = self.global_feature_names # if no bin features, state feature names are just global feature names
        logger.debug('Normalizing global features...')
        self.states, self.global_zscore_stats, self.global_rel_stats = normalize_noncyclic_features(
            state=states,
            state_feature_names=state_feature_names,
            sin_norm_feature_names=self.sin_norm_feature_names,
            log_norm_feature_names=self.log_norm_feature_names,
            fractional_norm_feature_names=self.fractional_norm_feature_names,
            local_mean_z_score_feature_names=self.local_mean_z_score_feature_names,
            z_score_feature_names=self.z_score_feature_names,
            do_sin_norm=self.do_sin_norm,
            do_log_norm=self.do_log_norm,
            do_fractional_norm=self.do_fractional_norm,
            do_local_mean_z_score=self.do_local_mean_z_score,
            do_z_score_norm=self.do_z_score_norm,
            fix_nans=True,
            z_stats=z_score_stats['global_features'] if z_score_stats is not None else None,
            rel_stats=rel_norm_stats['global_features'] if rel_norm_stats is not None else None,
            train_state_idxs=train_state_idxs
        )
        if include_bin_features and bin_states is not None:
            logger.debug('Normalizing bin features...')
            self.bin_states, self.bin_zscore_stats, self.bin_rel_stats = normalize_noncyclic_features(
                state=torch.tensor(bin_states).detach().clone(),
                state_feature_names=self.bin_feature_names,
                sin_norm_feature_names=self.sin_norm_feature_names,
                log_norm_feature_names=self.log_norm_feature_names,
                fractional_norm_feature_names=self.fractional_norm_feature_names,
                local_mean_z_score_feature_names=self.local_mean_z_score_feature_names,
                z_score_feature_names=self.z_score_feature_names,
                do_sin_norm=self.do_sin_norm,
                do_log_norm=self.do_log_norm,
                do_fractional_norm=self.do_fractional_norm,
                do_local_mean_z_score=self.do_local_mean_z_score,
                do_z_score_norm=self.do_z_score_norm,
                z_stats=z_score_stats['bin_features'] if z_score_stats is not None else None,
                rel_stats=rel_norm_stats['bin_features'] if rel_norm_stats is not None else None,
                fix_nans=True,
                train_state_idxs=train_state_idxs
            )
        else:
            self.bin_states = None
        if self.global_zscore_stats:
            self._save_norm_stats(cfg['metadata']['outdir'])
        # If using flat MLP
        if self._action_architecture is None:
            if include_bin_features and self.bin_states is not None:
                # Convert to tensors before manipulating
                if not isinstance(self.states, torch.Tensor):
                    self.states = torch.as_tensor(self.states, dtype=torch.float32)
                if not isinstance(self.bin_states, torch.Tensor):
                    self.bin_states = torch.as_tensor(self.bin_states, dtype=torch.float32)
                    
                # Flatten the 3D bin states [Batch, Bins, Features] -> [Batch, Bins * Features]
                bs_flat = self.bin_states.reshape(self.bin_states.shape[0], -1)
                
                self.states = torch.cat([self.states, bs_flat], dim=1)
                
                self.bin_states = None 
                
            self.bin_state_dim = 0
            self.state_dim = self.states.shape[-1]
            
        elif self._action_architecture in ACTION_ARCHITECTURES:
            self.state_dim = self.states.shape[-1]
            self.bin_state_dim = self.bin_states.shape[-1] if include_bin_features else 0
        else:
            raise NotImplementedError
        
        assert self.states.shape[0] == self.action_masks.shape[0], "States and masks must be 1:1"
        assert self.actions.shape[0] == self.rewards.shape[0] == self.dones.shape[0] == self.num_transitions, \
                f"Transition arrays shape mismatch: num_transitions {self.num_transitions}, bin_actions {self.actions.shape[0]}, rewards {self.rewards.shape[0]}, dones {self.dones.shape[0]}"
        if include_bin_features and self._action_architecture in ACTION_ARCHITECTURES:
            assert self.states.shape[0] == self.bin_states.shape[0],\
            f"State arrays shape mismatch: global state shape {self.states.shape[0]}, bin state shape {self.bin_states.shape[0]}"
            
            # self._save_to_cache()

    def _construct_transitions(self, df, bin_states, include_bin_features, action_space):
        """Constructs transition matrix from dataframe"""
        state_idxs, current_state_idxs, next_state_idxs, df_idx_to_compact = self._get_state_indices(df)
        self.df_idx_to_compact = df_idx_to_compact
        self.curr_compact_idxs = np.array([df_idx_to_compact[i] for i in current_state_idxs])
        self.next_compact_idxs = np.array([df_idx_to_compact[i] for i in next_state_idxs])
        # save for diagnostics
        self.state_idxs, self.current_state_idxs, self.next_state_idxs = state_idxs, current_state_idxs, next_state_idxs
        states, bin_states = self._construct_states(df=df, bin_states=bin_states, include_bin_features=include_bin_features, state_idxs=state_idxs)
        num_transitions = len(next_state_idxs)

        actions = self._construct_actions(df, action_space=action_space, next_state_idxs=next_state_idxs)
        rewards = self._construct_rewards(df, next_state_idxs=next_state_idxs, reward_choice=self.reward_choice)
        # dones = np.zeros(num_transitions, dtype=bool) # False unless last observation of the night
        # dones[-1] = True
        dones = self._construct_dones(num_transitions=num_transitions, next_state_idxs=next_state_idxs, current_state_idxs=current_state_idxs)
        # action_masks = self._construct_action_masks(state_df=df, action_space=action_space, num_transitions=num_transitions, state_idxs=state_idxs)
        action_masks = self._construct_action_masks(state_df=df, action_space=action_space, num_states=len(state_idxs), state_idxs=state_idxs)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.bool)
        action_masks = torch.as_tensor(action_masks, dtype=torch.bool)

        if include_bin_features:
            bin_states = torch.as_tensor(bin_states, dtype=torch.float32)
        else:
            bin_states = None
            
        return states, bin_states, actions, rewards, dones, action_masks, num_transitions

    def _construct_dones(self, num_transitions, next_state_idxs, current_state_idxs):
        """Constructs dones array, defined as end of night"""
        dones = np.zeros(num_transitions, dtype=bool)
        # A transition is the last of the night if its 'next' state isn't a 'current' state for a subsequent transition
        for i in range(num_transitions):
            if next_state_idxs[i] not in current_state_idxs:
                dones[i] = True
        dones[-1] = True # Failsafe
        return dones

    def _construct_states(self, df, bin_states, include_bin_features, state_idxs):
        global_states = self._construct_global_features(df=df, state_idxs=state_idxs)
        if include_bin_features:
            bin_states = self._construct_bin_states(bin_states=bin_states, state_idxs=state_idxs)
        else:
            bin_states = None
        return global_states, bin_states
    
    def _get_state_indices(self, df, max_time_diff_min=5):
        """Constructs state indexes for valid 'current_states'. Valid current states are observations which have occurred less than 5min from the previous state."""
        time_diffs = df['timestamp'].diff().values
        keep = time_diffs < max_time_diff_min * 60 + 90
        next_state_idxs = np.where(keep)[0]
        current_state_idxs = next_state_idxs - 1
        state_idxs = np.unique(np.concat([current_state_idxs, next_state_idxs])) # dataframe indices to extract
        df_idx_to_compact = {df_idx: compact_idx for compact_idx, df_idx in enumerate(state_idxs)}
        state_idxs = state_idxs
        logger.info(f'Removing {np.sum(~keep)} transitions with large time diffs > {max_time_diff_min} min. Total transitions: {len(keep)}')
        return state_idxs, current_state_idxs, next_state_idxs, df_idx_to_compact

    def _construct_global_features(self, df, state_idxs):
        """
        Constructs state and next_states for all transitions.
        Inserts a "null"/"zenith" observation before the first observation each night.
        The null observation state is defined as being an array of zeros
        """
        # global features already in DECam data
        missing_cols = set(self.global_feature_names) - set(df.columns) == 0
        assert missing_cols == 0, f'Features {missing_cols} do not exist in dataframe. Must be implemented in method self._process_dataframe()'

        state_df = df.iloc[state_idxs]
        global_features = state_df[self.global_feature_names].to_numpy()
        return global_features
        
    def _construct_bin_states(self, bin_states, state_idxs=None):
        """Returns bin states at state_idxs"""
        return bin_states[state_idxs]
    
    def _construct_actions(self, df, action_space, next_state_idxs):
        """Constructs action array from dataframe"""
        assert action_space in ['radec', 'azel', 'radec_filter', 'azel_filter', 'filter']

        next_state_df = df.iloc[next_state_idxs]
        if self.hpGrid.is_azel:
            lonlat = next_state_df[['az', 'el']].values
        else:
            lonlat = next_state_df[['ra', 'dec']].values
        bin_indices = self.hpGrid.ang2idx(lon=lonlat[:, 0], lat=lonlat[:, 1])

        if 'filter' not in action_space:
            actions = bin_indices
        elif ('radec' not in action_space) and ('azel' not in action_space):
            filter_series = next_state_df['filter']
            filter_indices = filter_series.map(FILTER2IDX).values.astype(np.int32)
            return filter_indices
        elif ('filter' in action_space) and any(coord_sys in action_space for coord_sys in ['radec', 'azel']):
            assert ZENITH_FILTER not in next_state_df['filter'].values, \
                f"Invalid data: Found '{ZENITH_FILTER}' in next_state_df. Zenith states must be dropped prior to action mapping."
            filter_series = next_state_df['filter']
            filter_indices = filter_series.map(FILTER2IDX).values
            assert not np.isnan(filter_indices).any(), \
                "nan filter values found in next_state_df. There should not be any zenith states in next_state_df"
            filter_indices = filter_indices.astype(np.int32)
            actions = (bin_indices * NUM_FILTERS) + filter_indices
        return actions

    def _construct_rewards(self, df, next_state_idxs, reward_choice='teff_rate'):
        assert reward_choice in ['teff_rate', 'expert_actions'], 'reward_choice must be teff_rate or expert_actions'
        """Constructs rewards for all transitions. Reward is defined as teff, normalized to [0, 1]."""
        if reward_choice == 'teff_rate':
            rewards = calc_inst_teff_rate(df=df, next_state_idxs=next_state_idxs)
        elif reward_choice == 'expert_actions':
            next_state_df = df.iloc[next_state_idxs]
            rewards = np.ones(len(next_state_df), dtype=np.float32)
        return rewards
    
    def _construct_slew_distances(self, df):
        curr_bids = df.iloc[self.current_state_idxs]['bin'].values.copy()
        next_bids = df.iloc[self.next_state_idxs]['bin'].values.copy()
        z_mask = curr_bids == -1
        if self.hpGrid.is_azel:
            curr_bids[z_mask] = self.hpGrid.ang2idx(lon=0, lat=np.pi/2) # map zenith bins to zenith bin        
        else:
            z_idxs = np.where(z_mask)[0]
            z_df_idxs = self.current_state_idxs[z_idxs]
            z_timestamps = df.iloc[z_df_idxs]['timestamp'].values
            for i, t in zip(z_idxs, z_timestamps):
                z_ra, z_dec = ephemerides.topographic_to_equatorial(lon=0, lat=np.pi/2, time=t)
                curr_bids[i] = self.hpGrid.ang2idx(lon=z_ra, lat=z_dec)
                
        curr_coords = np.array((self.hpGrid.lon[curr_bids], self.hpGrid.lat[curr_bids]))
        next_coords = np.array((self.hpGrid.lon[next_bids], self.hpGrid.lat[next_bids]))
        slew_dists = geometry.angular_separation(curr_coords, next_coords)
        slew_dists = torch.as_tensor(slew_dists, dtype=torch.float32)
        return slew_dists

    def _construct_action_masks(self, state_df, action_space, num_states, state_idxs):
        """
        Constructs action masks only with the condition that bins beyond horizon are masked
        """
        state_df = state_df.iloc[state_idxs]
        # given timestamp, determine bins which are outside of observable range
        els = np.empty((num_states, self.nbins), dtype=np.float32)
        
        if action_space == 'filter':
            return np.ones((num_states, self.num_filters), dtype=np.bool_)
        
        if self._calculate_action_mask:
            logger.info("Calculating action masks based on horizon. This may take a few minutes...")
            if not self.hpGrid.is_azel:
                lon, lat = self.hpGrid.lon, self.hpGrid.lat
                for i, time in tqdm(enumerate(state_df['timestamp'].values), total=len(state_df['timestamp'].values), desc="Calculating action mask"):
                    _, els[i] = ephemerides.equatorial_to_topographic(ra=lon, dec=lat, time=time)
                self._els = els
                action_mask = els > 0
            else:
                els = np.tile(self.hpGrid.lat[:, np.newaxis], reps=len(state_df['timestamp'].values)).T
                action_mask = els > 0
            if 'filter' in action_space:
                action_mask = np.repeat(action_mask, self.num_filters, axis=1)
        else:
            action_mask = np.ones((num_states, self.num_actions), dtype=np.bool_)
        return action_mask
        
    def _save_norm_stats(self, save_dir):
        """Persists the Z-score means and stds to disk for deployment/inference."""
        z_path = Path(save_dir) / "z_score_stats.pt"
        z_stats_dict = {
            'global_features': self.global_zscore_stats,
            'bin_features': self.bin_zscore_stats
        }
        torch.save(z_stats_dict, z_path)

        rel_path = Path(save_dir) / "rel_norm_stats.pt"
        rel_stats_dict = {
            'global_features': self.global_rel_stats,
            'bin_features': self.bin_rel_stats
        }
        torch.save(rel_stats_dict, rel_path)
        logger.debug(f"z-score stats: {z_stats_dict}")
        logger.debug(f"relative norm stats: {rel_stats_dict}")

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        """
        Returns a single transition tuple (state, action, reward, next state, done, action mask, next action mask, bin state, next bin state)
        """
        # Get compact indices for arrays constructed per state (ie, states, action_masks)
            # as opposed to arrays constructed per transition (ie, action, rewards, dones)
        c_idx = self.curr_compact_idxs[idx] 
        n_idx = self.next_compact_idxs[idx]

        # If done, these will get masked out during forward pass anyways (1-dones mask). Need dummy values here.
        is_done = self.dones[idx].item() # transition-level data takes idx

        if not self.include_bin_features:
            transition = (
                self.states[c_idx],
                self.actions[idx],
                self.rewards[idx],
                self.states[n_idx] if not is_done else torch.zeros_like(self.states[0]),
                self.dones[idx],
                self.action_masks[c_idx],
                self.action_masks[n_idx] if not is_done else torch.zeros_like(self.action_masks[0]),
                torch.as_tensor(0), # placeholder for bin state since not used in this case
                torch.as_tensor(0),
                self.slew_distances[idx]
            )
        elif self._action_architecture in ACTION_ARCHITECTURES:
            transition = (
                self.states[c_idx],
                self.actions[idx],
                self.rewards[idx],
                self.states[n_idx] if not is_done else torch.zeros_like(self.states[0]),
                self.dones[idx],
                self.action_masks[c_idx],
                self.action_masks[n_idx] if not is_done else torch.zeros_like(self.action_masks[0]),
                self.bin_states[c_idx], # shape (nstates, nbins, nfeatures)
                self.bin_states[n_idx] if not is_done else torch.zeros_like(self.bin_states[0]),
                self.slew_distances[idx]
            )
        return transition
    
    def _determine_split(self, val_split, random_seed, method='by_night'):
        """Determines train/val transition indices."""
        np.random.seed(random_seed)
        
        if method=='by_night':
            num_val_nights = max(1, int(self.n_nights * val_split))
            val_nights = np.random.choice(self.unique_nights, size=num_val_nights, replace=False)
            
            transition_nights = self._df.iloc[self.next_state_idxs - 1]['night']
            val_mask = np.isin(transition_nights, val_nights)
            
            train_indices = np.where(~val_mask)[0]
            val_indices = np.where(val_mask)[0]
        elif method=='by_transition':
            # Total number of valid transitions in the dataset
            num_transitions = len(self.next_state_idxs) 
            
            # Create a randomly shuffled array of all transition indices [0, 1, ..., N-1]
            shuffled_indices = np.random.permutation(num_transitions)
            
            # Determine where to slice the array based on val_split
            val_size = max(1, int(num_transitions * val_split))
            
            # Slice into validation and training sets
            val_indices = shuffled_indices[:val_size]
            train_indices = shuffled_indices[val_size:]
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        return train_indices, val_indices
    
    def get_dataloader(self, batch_size, num_workers, pin_memory, random_seed, drop_last=True):
        """Constructs pytorch dataloaders"""
        generator = torch.Generator().manual_seed(random_seed)
        
        train_dataset = Subset(self, self.train_transition_idxs.tolist())
        val_dataset = Subset(self, self.val_transition_idxs.tolist())
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(train_dataset, replacement=True, num_samples=10**10, generator=generator),
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, 
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    def old_get_dataloader(self, batch_size, num_workers, pin_memory, random_seed, drop_last=True, val_split=.1, return_train_and_val=True):
        generator = torch.Generator().manual_seed(random_seed)
    
        # Split dataset
        train_size = int(len(self) * (1 - val_split))
        val_size = len(self) - train_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size], generator=generator)
        
        # Train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size,
            sampler=RandomSampler(train_dataset, replacement=True, num_samples=10**10),
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader