import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from collections import defaultdict
import gc

from tqdm import tqdm

from blancops.ephemerides import ephemerides
import pandas as pd
import json
from torch.utils.data import random_split, RandomSampler
import pickle
import gc

from blancops.data_processing.features import *
from blancops.data_processing.constants import *
from blancops.data_processing.data_processing import *

# Get the logger associated with this module's name (e.g., 'my_module')
import logging
logger = logging.getLogger(__name__)

def reward_func_v0():
    raise NotImplementedError

class OfflineDataset(torch.utils.data.Dataset):
    def __init__(self, df=None, cfg=None, gcfg=None,
                 specific_years=None, specific_months=None, specific_days=None, specific_filters=None,
                 field2maxvisits_path=None, field2radec_path=None, field2name_path=None,
                 night2filtervisithistory_path=None, fieldfilter2maxvisits=None,
                 ): 
        assert cfg is not None and gcfg is not None, "Must pass both cfg and gcfg"

        # Assign static attributes
        self.do_cyclical_norm = cfg['data']['do_cyclical_norm']
        self.do_max_norm = cfg['data']['do_max_norm']
        self.do_inverse_norm = cfg['data']['do_inverse_norm']
        self.do_ang_distance_norm = cfg['data']['do_ang_distance_norm']
        self.objects_to_remove = ["guide", "DES vvds","J0'","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec", "(outlier)"]
        self.reward_choice = cfg['data']['reward_choice']
        self._calculate_action_mask = cfg['model']['algorithm'] != 'BC' # should be False if using bc (to minimize data processing time), otherwise True
        self._grid_network = cfg['model']['grid_network']
        self._cache_path = Path(cfg['metadata']['parent_results_dir']).resolve() / Path(cfg['metadata']['exp_name']) / "dataset_cache.pt"

        if os.path.exists(self._cache_path):
            self._load_from_cache()
        else:
            # Get global feature names
            self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES'] if self.do_cyclical_norm else []
            self.max_norm_feature_names = gcfg['features']['MAX_NORM_FEATURE_NAMES'] if self.do_max_norm else []
            self.ang_distance_feature_names = gcfg['features']['ANG_DISTANCE_NORM_FEATURE_NAMES'] if self.do_ang_distance_norm else []

            # Get other configurations
            action_space = cfg['data']['action_space']
            nside = cfg['data']['nside']
            logger.info(f'Including the following bin features: {cfg["data"].get("bin_features")}')
            logger.info(f'Including the following global features: {cfg["data"]["global_features"]}')
            include_bin_features = len(cfg['data']['bin_features']) > 0

            # Load lookup tables
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
            # Add this block to load the pickle file into the dictionary variable
            with open(fieldfilter2maxvisits, 'rb') as f:
                fieldfilter2maxvisits = pickle.load(f)

            self.hpGrid = ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in action_space))
            self.num_filters = NUM_FILTERS if 'filter' in action_space else 1

            # Set number of actions based on binning method
            self.nbins = len(self.hpGrid.lon)
            self.num_actions = self.nbins * self.num_filters

            # Save list of all feature names
            self.base_global_feature_names = cfg['data']['global_features'].copy()
            self.base_bin_feature_names = cfg['data']['bin_features'].copy()
            self.global_feature_names, self.bin_feature_names = setup_feature_names(base_global_feature_names=self.base_global_feature_names,
                                                                                    base_bin_feature_names=self.base_bin_feature_names,
                                                                                    cyclical_feature_names=self.cyclical_feature_names,
                                                                                    do_cyclical_norm=self.do_cyclical_norm,
                                                                                    )

            # Process dataframe to add columns for global features
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
                do_cyclical_norm=self.do_cyclical_norm
            )
            if len(self.bin_feature_names) > 0:
                bin_states = calculate_bin_features(
                    pt_df=self._df,
                    hpGrid=self.hpGrid, 
                    base_bin_feature_names=self.base_bin_feature_names, 
                    bin_feature_names=self.bin_feature_names, 
                    cyclical_feature_names=self.cyclical_feature_names, 
                    do_cyclical_norm=self.do_cyclical_norm, 
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

            # Construct Transitions
            states, bin_states, self.actions, self.rewards, self.dones, self.action_masks, self.num_transitions \
                = self._construct_transitions(
                df=self._df, 
                bin_states=bin_states,  
                include_bin_features=include_bin_features, 
                action_space=action_space,
                )
            
            self._prenorm_bin_states = bin_states
            
            logger.info(f"States shape: {states.shape}, Actions shape: {self.actions.shape}, Rewards shape: {self.rewards.shape}, Dones shape: {self.dones.shape}, Action masks shape: {self.action_masks.shape}")
            logger.info(f"Bin states shape: {bin_states.shape if bin_states is not None else None}")

            state_feature_names = self.global_feature_names

            self.states = normalize_noncyclic_features(
                state=states,
                state_feature_names=state_feature_names,
                max_norm_feature_names=self.max_norm_feature_names,
                ang_distance_norm_feature_names=self.ang_distance_feature_names,
                do_inverse_norm=self.do_inverse_norm,
                do_max_norm=self.do_max_norm,
                do_ang_distance_norm=self.do_ang_distance_norm,
                fix_nans=True
            )

            if include_bin_features and bin_states is not None:
                self.bin_states = normalize_noncyclic_features(
                    state=torch.tensor(bin_states).detach().clone(), # NOTE: use local `bin_states`, not `self.bin_states`
                    state_feature_names=self.bin_feature_names,
                    max_norm_feature_names=self.max_norm_feature_names,
                    ang_distance_norm_feature_names=self.ang_distance_feature_names,
                    do_inverse_norm=self.do_inverse_norm,
                    do_max_norm=self.do_max_norm,
                    do_ang_distance_norm=self.do_ang_distance_norm,
                    fix_nans=True,
                )
            else:
                self.bin_states = None
                self.next_bin_states = None

            # If using flat MLP
            if self._grid_network is None:
                if include_bin_features and self.bin_states is not None:
                    # Convert to tensors before manipulating
                    if not isinstance(self.states, torch.Tensor):
                        self.states = torch.as_tensor(self.states, dtype=torch.float32)
                    if not isinstance(self.bin_states, torch.Tensor):
                        self.bin_states = torch.as_tensor(self.bin_states, dtype=torch.float32)
                        
                    # Flatten the 3D bin states [Batch, Bins, Features] -> [Batch, Bins * Features]
                    bs_flat = self.bin_states.reshape(self.bin_states.shape[0], -1)
                    
                    # Concatenate flat bin features directly to global features
                    self.states = torch.cat([self.states, bs_flat], dim=1)
                    
                    # Cleanup to save VRAM and bypass grid_network checks
                    self.bin_states = None 
                    
                self.bin_state_dim = 0
                self.state_dim = self.states.shape[-1]
                
            elif self._grid_network in GRID_NETWORKS:
                self.state_dim = self.states.shape[-1]
                self.bin_state_dim = self.bin_states.shape[-1] if include_bin_features else 0
            else:
                raise NotImplementedError
            
            assert self.states.shape[0] == self.action_masks.shape[0], "States and masks must be 1:1"
            assert self.actions.shape[0] == self.rewards.shape[0] == self.dones.shape[0] == self.num_transitions, \
                    f"Transition arrays shape mismatch: num_transitions {self.num_transitions}, bin_actions {self.actions.shape[0]}, rewards {self.rewards.shape[0]}, dones {self.dones.shape[0]}"
            if include_bin_features and self._grid_network in GRID_NETWORKS:
                assert self.states.shape[0] == self.bin_states.shape[0],\
                f"State arrays shape mismatch: global state shape {self.states.shape[0]}, bin state shape {self.bin_states.shape[0]}"
                
            self._save_to_cache()

    def _construct_transitions(self, df, bin_states, include_bin_features, action_space):
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
    
    def _get_state_indices(self, df, max_time_diff_min=10):
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
        Inserts a "null" observation before the first observation each night.
        The null observation state is defined as being an array of zeros
        """
        # global features already in DECam data
        missing_cols = set(self.global_feature_names) - set(df.columns) == 0
        assert missing_cols == 0, f'Features {missing_cols} do not exist in dataframe. Must be implemented in method self._process_dataframe()'

        state_df = df.iloc[state_idxs]
        global_features = state_df[self.global_feature_names].to_numpy()
        return global_features
        
    def _construct_bin_states(self, bin_states, state_idxs=None):
        # Get bin_features and next_bin_features
        return bin_states[state_idxs]
    
    def _construct_actions(self, df, action_space, next_state_idxs):
        assert action_space in ['radec', 'azel', 'radec_filter', 'azel_filter'], 'action_space must be radec or azel'

        next_state_df = df.iloc[next_state_idxs]
        if self.hpGrid.is_azel:
            lonlat = next_state_df[['az', 'el']].values
        else:
            lonlat = next_state_df[['ra', 'dec']].values
        bin_indices = self.hpGrid.ang2idx(lon=lonlat[:, 0], lat=lonlat[:, 1])

        if 'filter' in action_space:
            assert ZENITH_FILTER not in next_state_df['filter'].values, \
                f"Invalid data: Found '{ZENITH_FILTER}' in next_state_df. Zenith states must be dropped prior to action mapping."
            filter_series = next_state_df['filter']
            filter_indices = filter_series.map(FILTER2IDX).values.astype(np.int32)
            assert not np.isnan(filter_indices).any(), \
                "Invalid data: Found unmapped filters. Ensure FILTER2IDX covers all strings in the dataset."
            actions = (bin_indices * NUM_FILTERS) + filter_indices
            return actions

        return bin_indices

    def _construct_rewards(self, df, next_state_idxs, reward_choice='teff_rate'):
        assert reward_choice in ['teff_rate', 'expert_actions'], 'reward_choice must be teff_rate or expert_actions'
        """Constructs rewards for all transitions. Reward is defined as teff, normalized to [0, 1]."""
        if reward_choice == 'teff_rate':
            rewards = get_inst_teff_rate(df=df, next_state_idxs=next_state_idxs)
        elif reward_choice == 'expert_actions':
            next_state_df = df.iloc[next_state_idxs]
            rewards = np.ones(len(next_state_df), dtype=np.float32)
        return rewards

    def _construct_action_masks(self, state_df, action_space, num_states, state_idxs):
        """
        Constructs action masks only with the condition that bins outside of horizon are masked
        """
        state_df = state_df.iloc[state_idxs]
        # given timestamp, determine bins which are outside of observable range
        els = np.empty((num_states, self.num_actions), dtype=np.float32)
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
            action_mask = np.ones((num_states, self.num_actions))
        return action_mask
        
    def _save_to_cache(self):
        """Packs the tensors into a dictionary and saves to disk."""
        logger.info(f"Saving processed dataset to {self.cache_path}")
        cache_dict = {
            'states': self.states,
            'bin_states': self.bin_states,
            'actions': self.actions,
            'action_masks': self.action_masks,
            'rewards': self.rewards,
            'dones': self.dones
        }
        # torch.save is highly optimized for writing dense tensors to disk
        torch.save(cache_dict, self.cache_path)

    def _load_from_cache(self):
        """Loads the pre-compiled tensors directly into memory."""
        logger.info(f"Loading processed dataset from {self.cache_path}")
        cache_dict = torch.load(self.cache_path, weights_only=True) # weights_only=True is safer and faster
        
        self.states = cache_dict['states']
        self.bin_states = cache_dict['bin_states']
        self.actions = cache_dict['actions']
        self.action_masks = cache_dict['action_masks']
        self.rewards = cache_dict['rewards']
        self.dones = cache_dict['dones']
        
    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        """
        Returns
        -------
        transition (tuple): (global_state, bin action, reward, next_state, done, action_mask, next_action_mask, bin_state, next_bin_state)
        """
        # Get compact indices for arrays constructed per state (ie, states, action_masks)
            # as opposed to arrays constructed per transition (ie, action, rewards, dones)
        c_idx = self.curr_compact_idxs[idx] 
        n_idx = self.next_compact_idxs[idx]

        # If done, these will get masked out during forward pass anyways (1-dones mask). Need dummy values here.
        is_done = self.dones[idx] # transition-level data takes idx

        if self._grid_network is None:
            transition = (
                self.states[c_idx],
                self.actions[idx],
                self.rewards[idx],
                self.states[n_idx] if not is_done else torch.zeros_like(self.states[0]),
                self.dones[idx],
                self.action_masks[c_idx],
                self.action_masks[n_idx] if not is_done else torch.zeros_like(self.action_masks[0]),
                torch.as_tensor(0), # placeholder for bin state since not used in this case
                torch.as_tensor(0)
            )
        elif self._grid_network in GRID_NETWORKS:
            transition = (
                self.states[c_idx],
                self.actions[idx],
                self.rewards[idx],
                self.states[n_idx] if not is_done else torch.zeros_like(self.states[0]),
                self.dones[idx],
                self.action_masks[c_idx],
                self.action_masks[n_idx] if not is_done else torch.zeros_like(self.action_masks[0]),
                self.bin_states[c_idx], # shape (nstates, nbins, nfeatures)
                self.bin_states[n_idx] if not is_done else torch.zeros_like(self.bin_states[0])
            )
        return transition
    
    def get_dataloader(self, batch_size, num_workers, pin_memory, random_seed, drop_last=True, val_split=.1, return_train_and_val=True):
        generator = torch.Generator().manual_seed(random_seed)
        np.random.seed(random_seed) # Ensure consistent night selection
        
        # Randomly sample whole nights for the validation set
        num_val_nights = max(1, int(self.n_nights * val_split))
        val_nights = np.random.choice(self.unique_nights, size=num_val_nights, replace=False)
        logger.info(f'Choosing {num_val_nights} nights for validation out of {self.n_nights} nights. Specifically, {np.sort(val_nights)}')
        
        transition_nights = self._df.iloc[self.next_state_idxs - 1]['night']
            
        val_mask = np.isin(transition_nights, val_nights)
        
        train_indices = np.where(~val_mask)[0].tolist()
        val_indices = np.where(val_mask)[0].tolist()
        
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(train_dataset, replacement=True, num_samples=10**10, generator=generator),
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        if return_train_and_val:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False, 
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            return train_loader, val_loader
            
        return train_loader

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
        if return_train_and_val:
            val_loader = DataLoader(
                val_dataset,
                batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            return train_loader, val_loader
        return train_loader