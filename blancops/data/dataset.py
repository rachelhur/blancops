import numpy as np
import pandas as pd
import json
import pickle
import gc
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset, RandomSampler

from blancops.ephemerides import ephemerides
from blancops.math import geometry

from blancops.data.preprocessing import drop_rows_in_DECam_data
from blancops.data.constants import *
from blancops.configs.constants import CYCLICAL_FEATURE_NAMES

from blancops.data.features.glob_features import GlobalFeatureEngineer, calc_inst_teff_rate
from blancops.data.features.bin_features import BinFeatureEngineer
from blancops.data.features.normalizations import StateNormalizer, build_normalizer_kwargs, setup_feature_names

logger = logging.getLogger(__name__)

class OfflineDataset(torch.utils.data.Dataset):
    def __init__(
        self, df=None, cfg=None, lookups=None,
        global_normalizer=None, bin_normalizer=None,
        years=None, months=None, days=None, filters=None,
        z_score_stats=None, rel_norm_stats=None, mode='train'
    ): 
        # Setup configurations, normalization method, and lookups
        norm_kwargs = build_normalizer_kwargs(cfg.data.norm)
        self._setup_configuration(cfg, norm_kwargs)
        self.lookups = lookups

        # 2. Raw Data Filtering
        df = drop_rows_in_DECam_data(
            df,
            specific_years=cfg.data.years if years is None else years, 
            specific_months=cfg.data.months if months is None else months, 
            specific_days=cfg.data.days if days is None else days,
            specific_filters=cfg.data.filters if filters is None else filters,
        )
        
        # 3. Feature Engineering Pipeline
        self._engineer_features(df, cfg, norm_kwargs)

        # 4. RL Transition Construction
        self._build_transitions(cfg.data.action_space)
        
        # 5. Train/Val Split
        self._split_data(cfg.data.train_val_split, cfg.train.seed)

        # 6. Normalization Pipeline
        self._normalize_states(
            mode, cfg, norm_kwargs, global_normalizer, bin_normalizer, 
            z_score_stats, rel_norm_stats
        )

        # 7. Final Formatting & Validation
        self._format_tensors_for_network(cfg.model.network)
        self._validate_dataset()

    def _setup_configuration(self, cfg, norm_kwargs):
        """Initializes constants, spaces, and feature names."""
        self.reward_choice = cfg.reward.reward_choice
        self._calculate_action_mask = cfg.model.algorithm != 'bc'
        self.include_bin_features = len(cfg.data.bin_features) > 0
        
        action_space = cfg.data.action_space
        self.hpGrid = ephemerides.HealpixGrid(nside=cfg.data.nside, is_azel=('azel' in action_space))
        self.num_filters = NUM_FILTERS if 'filter' in action_space else 1
        self.nbins = len(self.hpGrid.lon)

        if action_space == 'filter':
            self.num_actions = self.num_filters
        elif action_space in ['radec', 'azel']:
            self.num_actions = self.nbins
        elif 'filter' in action_space and any(c in action_space for c in ['radec', 'azel']):
            self.num_actions = self.nbins * self.num_filters
        else:
            raise NotImplementedError(f"Action space {action_space} not supported.")

        self.global_feature_names, self.bin_feature_names = setup_feature_names(
            base_global_feature_names=cfg.data.global_features.copy(),
            base_bin_feature_names=cfg.data.bin_features.copy(),
            cyclical_feature_names=CYCLICAL_FEATURE_NAMES,
            do_cyclical_norm=norm_kwargs.get('do_cyclical_norm', False),
        )
        self.do_local_mean_z_score = any('rel_' in name for name in self.bin_feature_names)

    def _engineer_features(self, df, cfg, norm_kwargs):
        """Executes the pandas/numpy feature engineering pipelines."""
        glob_feature_eng = GlobalFeatureEngineer(
            fid2name=self.lookups.fid2name, 
            hpGrid=self.hpGrid, 
            base_features=cfg.data.global_features,
            cyclical_features=CYCLICAL_FEATURE_NAMES, 
            do_cyclical_norm=norm_kwargs.get('do_cyclical_norm', True)
        )
        self._df = glob_feature_eng.transform(df)
        
        if self.include_bin_features:
            bin_feature_eng = BinFeatureEngineer(
                hpGrid=self.hpGrid, 
                base_features=cfg.data.bin_features, 
                cyclical_features=CYCLICAL_FEATURE_NAMES, 
                action_space=cfg.data.action_space,
                lookups=self.lookups,  # <-- REFACTORED: Single source of truth
                do_cyclical_norm=norm_kwargs.get('do_cyclical_norm', True),
                do_local_mean_z_score=self.do_local_mean_z_score
            )
            self._prenorm_bin_states = bin_feature_eng.transform(self._df, requested_features=self.bin_feature_names)
        else:
            self._prenorm_bin_states = None

        self.unique_nights = self._df['night'].unique()
        self.n_nights = self._df.groupby('night').ngroups

    def _build_transitions(self, action_space):
        """Builds the strict RL transitions (S, A, R, S', Dones, Masks)."""
        (self.states, self._prenorm_bin_states, self.actions, self.rewards, 
         self.dones, self.action_masks, self.num_transitions) = self._construct_transitions(
            df=self._df, 
            bin_states=self._prenorm_bin_states,  
            include_bin_features=self.include_bin_features, 
            action_space=action_space,
        )
        self.slew_distances = self._construct_slew_distances(self._df)

    def _split_data(self, train_val_split, seed):
        """Calculates indices for the train/val dataloaders."""
        val_split = 1 - train_val_split
        self.train_transition_idxs, self.val_transition_idxs = self._determine_split(val_split, seed)
        
        train_c_idxs = self.curr_compact_idxs[self.train_transition_idxs]
        train_n_idxs = self.next_compact_idxs[self.train_transition_idxs]
        self.train_state_idxs = np.unique(np.concatenate([train_c_idxs, train_n_idxs]))

    def _normalize_states(self, mode, cfg, norm_kwargs, global_normalizer, bin_normalizer, z_stats, rel_stats):
        """Executes the StateNormalizer class logic."""
        # 1. Initialize normalizers if not provided
        if global_normalizer is None:
            global_normalizer = StateNormalizer(state_feature_names=cfg.data.global_features, **norm_kwargs)
        if self.include_bin_features and bin_normalizer is None:
            bin_normalizer = StateNormalizer(state_feature_names=cfg.data.bin_features, **norm_kwargs)

        # 2. Normalize Global Features
        if mode == 'train':
            self.states, self.global_zscore_stats, self.global_rel_stats = global_normalizer.fit_transform(
                state=self.states, train_state_idxs=self.train_state_idxs
            )
        else:
            self.states = global_normalizer.transform(
                state=self.states, 
                z_stats_dict=z_stats.get('global_features', {}), 
                rel_stats_dict=rel_stats.get('global_features', {})
            )
            self.global_zscore_stats, self.global_rel_stats = None, None

        # 3. Normalize Bin Features
        if self.include_bin_features and self._prenorm_bin_states is not None:
            bin_tensor = torch.tensor(self._prenorm_bin_states).detach().clone()
            if mode == 'train':
                self.bin_states, self.bin_zscore_stats, self.bin_rel_stats = bin_normalizer.fit_transform(
                    state=bin_tensor, train_state_idxs=self.train_state_idxs
                )
            else:
                self.bin_states = bin_normalizer.transform(
                    state=bin_tensor, 
                    z_stats_dict=z_stats.get('bin_features', {}), 
                    rel_stats_dict=rel_stats.get('bin_features', {})
                )
                self.bin_zscore_stats, self.bin_rel_stats = None, None
        else:
            self.bin_states = None

        # 4. Save stats if training
        if mode == 'train' and (self.global_zscore_stats or self.global_rel_stats):
            self._save_norm_stats(Path(cfg.experiments_directory) / cfg.experiment_name)

    def _format_tensors_for_network(self, network_type):
        """Handles final reshaping (e.g., flattening bins for basic MLPs)."""
        if network_type == 'mlp':
            if self.include_bin_features and self.bin_states is not None:
                if not isinstance(self.states, torch.Tensor):
                    self.states = torch.as_tensor(self.states, dtype=torch.float32)
                if not isinstance(self.bin_states, torch.Tensor):
                    self.bin_states = torch.as_tensor(self.bin_states, dtype=torch.float32)
                    
                bs_flat = self.bin_states.reshape(self.bin_states.shape[0], -1)
                self.states = torch.cat([self.states, bs_flat], dim=1)
                self.bin_states = None 
                
            self.bin_state_dim = 0
            self.state_dim = self.states.shape[-1]
        else:
            self.state_dim = self.states.shape[-1]
            self.bin_state_dim = self.bin_states.shape[-1] if self.include_bin_features else 0

        self.dataset_dims = {
            'state_dim': self.state_dim,
            'bin_state_dim': self.bin_state_dim,
            'num_bins': self.nbins,
            'num_filters': self.num_filters,
            'num_actions': self.num_actions
        }
        self.dataset_feature_names = {
            'global_features': self.global_feature_names,
            'bin_features': self.bin_feature_names
        }

    def _validate_dataset(self):
        """Ensures tensor shapes match expectations before training."""
        assert self.states.shape[0] == self.action_masks.shape[0], "States and masks must be 1:1"
        assert self.actions.shape[0] == self.rewards.shape[0] == self.dones.shape[0] == self.num_transitions, \
                f"Transition mismatch: actions {self.actions.shape[0]}, rewards {self.rewards.shape[0]}, dones {self.dones.shape[0]}"
        if self.include_bin_features and self.bin_states is not None:
            assert self.states.shape[0] == self.bin_states.shape[0],\
                f"State mismatch: global {self.states.shape[0]}, bin {self.bin_states.shape[0]}"

    def _construct_transitions(self, df, bin_states, include_bin_features, action_space):
        """Constructs transition matrix from dataframe"""
        state_idxs, current_state_idxs, next_state_idxs, df_idx_to_compact = self._get_state_indices(df)
        self.df_idx_to_compact = df_idx_to_compact
        self.curr_compact_idxs = np.array([df_idx_to_compact[i] for i in current_state_idxs])
        self.next_compact_idxs = np.array([df_idx_to_compact[i] for i in next_state_idxs])
        self.state_idxs, self.current_state_idxs, self.next_state_idxs = state_idxs, current_state_idxs, next_state_idxs
        
        states, bin_states = self._construct_states(df=df, bin_states=bin_states, include_bin_features=include_bin_features, state_idxs=state_idxs)
        num_transitions = len(next_state_idxs)

        actions = self._construct_actions(df, action_space=action_space, next_state_idxs=next_state_idxs)
        rewards = self._construct_rewards(df, next_state_idxs=next_state_idxs, reward_choice=self.reward_choice)
        dones = self._construct_dones(num_transitions=num_transitions, next_state_idxs=next_state_idxs, current_state_idxs=current_state_idxs)
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
        time_diffs = df['timestamp'].diff().values
        keep = time_diffs < max_time_diff_min * 60 + 90
        next_state_idxs = np.where(keep)[0]
        current_state_idxs = next_state_idxs - 1
        state_idxs = np.unique(np.concatenate([current_state_idxs, next_state_idxs]))
        df_idx_to_compact = {df_idx: compact_idx for compact_idx, df_idx in enumerate(state_idxs)}
        logger.info(f'Removing {np.sum(~keep)} transitions with large time diffs > {max_time_diff_min} min. Total transitions: {len(keep)}')
        return state_idxs, current_state_idxs, next_state_idxs, df_idx_to_compact

    def _construct_global_features(self, df, state_idxs):
        missing_cols = set(self.global_feature_names) - set(df.columns)
        assert len(missing_cols) == 0, f'Features {missing_cols} do not exist in dataframe.'
        state_df = df.iloc[state_idxs]
        return state_df[self.global_feature_names].to_numpy()
        
    def _construct_bin_states(self, bin_states, state_idxs=None):
        return bin_states[state_idxs]
    
    def _construct_actions(self, df, action_space, next_state_idxs):
        assert action_space in ['radec', 'azel', 'radec_filter', 'azel_filter', 'filter']
        next_state_df = df.iloc[next_state_idxs]
        
        if self.hpGrid.is_azel:
            lonlat = next_state_df[['az', 'el']].values
        else:
            lonlat = next_state_df[['ra', 'dec']].values
            
        bin_indices = self.hpGrid.ang2idx(lon=lonlat[:, 0], lat=lonlat[:, 1])

        if 'filter' not in action_space:
            return bin_indices
        elif ('radec' not in action_space) and ('azel' not in action_space):
            filter_series = next_state_df['filter']
            return filter_series.map(FILTER2IDX).values.astype(np.int32)
        else:
            assert ZENITH_FILTER not in next_state_df['filter'].values, \
                f"Invalid data: Found '{ZENITH_FILTER}' in next_state_df."
            filter_series = next_state_df['filter']
            filter_indices = filter_series.map(FILTER2IDX).values.astype(np.int32)
            return (bin_indices * NUM_FILTERS) + filter_indices

    def _construct_rewards(self, df, next_state_idxs, reward_choice='teff_rate'):
        assert reward_choice in ['teff_rate', 'expert_actions']
        if reward_choice == 'teff_rate':
            return calc_inst_teff_rate(df=df, next_state_idxs=next_state_idxs)
        elif reward_choice == 'expert_actions':
            next_state_df = df.iloc[next_state_idxs]
            return np.ones(len(next_state_df), dtype=np.float32)
    
    def _construct_slew_distances(self, df):
        curr_bids = df.iloc[self.current_state_idxs]['bin'].values.copy()
        next_bids = df.iloc[self.next_state_idxs]['bin'].values.copy()
        z_mask = curr_bids == -1
        
        if self.hpGrid.is_azel:
            curr_bids[z_mask] = self.hpGrid.ang2idx(lon=0, lat=np.pi/2) 
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
        return torch.as_tensor(slew_dists, dtype=torch.float32)

    def _construct_action_masks(self, state_df, action_space, num_states, state_idxs):
        state_df = state_df.iloc[state_idxs]
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
        all_stats = {
            "z_score": {'global_features': self.global_zscore_stats, 'bin_features': self.bin_zscore_stats},
            "rel_norm": {'global_features': self.global_rel_stats, 'bin_features': self.bin_rel_stats}
        }
        save_path = Path(save_dir) / "normalization_stats.json"
        with open(save_path, 'w') as f:
            json.dump(all_stats, f, indent=4)
        logger.info(f"Normalization stats successfully saved to {save_path}")

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        c_idx = self.curr_compact_idxs[idx] 
        n_idx = self.next_compact_idxs[idx]
        is_done = self.dones[idx].item()

        transition = (
            self.states[c_idx],
            self.actions[idx],
            self.rewards[idx],
            self.states[n_idx] if not is_done else torch.zeros_like(self.states[0]),
            self.dones[idx],
            self.action_masks[c_idx],
            self.action_masks[n_idx] if not is_done else torch.zeros_like(self.action_masks[0]),
            self.bin_states[c_idx] if self.include_bin_features else torch.as_tensor(0),
            self.bin_states[n_idx] if (self.include_bin_features and not is_done) else (torch.zeros_like(self.bin_states[0]) if self.include_bin_features else torch.as_tensor(0)),
            self.slew_distances[idx]
        )
        return transition
    
    def _determine_split(self, val_split, random_seed, method='by_night'):
        np.random.seed(random_seed)
        
        if method == 'by_night':
            num_val_nights = max(1, int(self.n_nights * val_split))
            val_nights = np.random.choice(self.unique_nights, size=num_val_nights, replace=False)
            logger.info(f"VAL NIGHTS ({len(val_nights)}): {val_nights}")
            
            transition_nights = self._df.iloc[self.next_state_idxs - 1]['night']
            val_mask = np.isin(transition_nights, val_nights)
            
            train_indices = np.where(~val_mask)[0]
            val_indices = np.where(val_mask)[0]

            self.val_nights = val_nights.tolist()
            self.train_nights = set(self.unique_nights) - set(val_nights)
            
        elif method == 'by_transition':
            num_transitions = len(self.next_state_idxs) 
            shuffled_indices = np.random.permutation(num_transitions)
            val_size = max(1, int(num_transitions * val_split))
            val_indices = shuffled_indices[:val_size]
            train_indices = shuffled_indices[val_size:]
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        return train_indices, val_indices
    
    def get_dataloader(self, batch_size, num_workers, pin_memory, random_seed, drop_last=True):
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset = Subset(self, self.train_transition_idxs.tolist())
        val_dataset = Subset(self, self.val_transition_idxs.tolist())
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=RandomSampler(train_dataset, replacement=True, num_samples=10**10, generator=generator),
            drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            drop_last=False, num_workers=num_workers, pin_memory=pin_memory,
        )
        return train_loader, val_loader

def _validate_field_lookup_df(field_lookup_df):
    required_columns = ['field_id', 'exptime', 'ra', 'dec', 'n_visits', 'filter'] # 'dithers','object', 'priorities'
    missing_cols = [col for col in required_columns if col not in field_lookup_df.columns]
    
    if missing_cols:
        error_msg = f"field_lookup is missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def save_field_lookup(field_lookup_df, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    _validate_field_lookup_df(field_lookup_df)
    
    outpath = outdir / "field_lookup.json"
    field_lookup_df.to_json(outpath, indent=2, orient='columns')
    logger.info(f"Successfully saved field lookup to {outpath}")

def load_field_lookup_df(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot load field lookup. File not found: {filepath}")

    field_lookup_df = pd.read_json(filepath)
    _validate_field_lookup_df(field_lookup_df)
    return field_lookup_df