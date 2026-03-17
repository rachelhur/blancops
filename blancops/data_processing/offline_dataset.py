import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from collections import defaultdict
import gc

from tqdm import tqdm

from blancops.ephemerides import ephemerides
from blancops.data_processing.features import *
import pandas as pd
import json
from torch.utils.data import random_split, RandomSampler
import pickle
import gc

# Get the logger associated with this module's name (e.g., 'my_module')
import logging
logger = logging.getLogger(__name__)


# def get_lst(timestamp):
#     t = astropy.time.Time(timestamp, format='unix', scale='utc')
#     lst = t.sidereal_time('apparent', longitude=-1.2358069931456779) # blanco dec
#     return lst.to(astropy.units.rad).value

def reward_func_v0():
    raise NotImplementedError

class OfflineDataset(torch.utils.data.Dataset):
    def __init__(self, df=None, cfg=None, gcfg=None,
                 specific_years=None, specific_months=None, specific_days=None, specific_filters=None,
                 field2maxvisits_path=None, field2radec_path=None, field2name_path=None,
                 night2filtervisithistory_path=None, fieldfilter2maxvisits=None,
                 save_df=False
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

        # Get global feature names
        self.cyclical_feature_names = gcfg['features']['CYCLICAL_FEATURE_NAMES'] if self.do_cyclical_norm else []
        self.max_norm_feature_names = gcfg['features']['MAX_NORM_FEATURE_NAMES'] if self.do_max_norm else []
        self.ang_distance_feature_names = gcfg['features']['ANG_DISTANCE_NORM_FEATURE_NAMES'] if self.do_ang_distance_norm else []

        # Get other configurations
        bin_space = cfg['data']['bin_space']
        binning_method = cfg['data']['bin_method']
        nside = cfg['data']['nside']
        remove_large_time_diffs = cfg['data']['remove_large_time_diffs']
        logger.info(f'Including the following bin features: {cfg["data"]["bin_features"]}')
        logger.info(f'Including the following global features: {cfg["data"]["global_features"]}')
        include_bin_features = len(cfg['data']['bin_features']) > 0
        num_bins_1d = cfg['data']['num_bins_1d']

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

        # Initialize healpix grid if binning_method is healpix
        self.hpGrid = None if binning_method != 'healpix' else ephemerides.HealpixGrid(nside=nside, is_azel=('azel' in bin_space))
        self.num_filters = NUM_FILTERS if 'filter' in bin_space else 1

        # Set number of actions based on binning method
        if binning_method == 'uniform':
            self.num_bins = int(num_bins_1d**2)
        elif binning_method == 'healpix':
            self.num_bins = len(self.hpGrid.lon)
        self.num_actions = self.num_bins * self.num_filters

        # Save list of all feature names
        self.base_global_feature_names = cfg['data']['global_features'].copy()
        self.base_bin_feature_names = cfg['data']['bin_features'].copy()
        self.global_feature_names, self.bin_feature_names, self.prenorm_expanded_bin_feature_names =\
            setup_feature_names(base_global_feature_names=self.base_global_feature_names,
                                base_bin_feature_names=self.base_bin_feature_names,
                                cyclical_feature_names=self.cyclical_feature_names,
                                nbins=self.num_bins,
                                do_cyclical_norm=self.do_cyclical_norm,
                                grid_network=self._grid_network
                                )

        # Process dataframe to add columns for global features
        df = drop_rows_in_DECam_data(
            df,
            specific_years=cfg['data']['specific_years'] if specific_years is None else specific_years, 
            specific_months=cfg['data']['specific_months'] if specific_months is None else specific_months, 
            specific_days=cfg['data']['specific_days'] if specific_days is None else specific_days,
            specific_filters=cfg['data']['specific_filters'] if specific_filters is None else specific_filters,
            objects_to_remove=self.objects_to_remove
            )

        df = calculate_and_add_global_features(
            df=df, 
            field2name=field2name, 
            hpGrid=self.hpGrid, 
            base_global_feature_names=self.base_global_feature_names,
            cyclical_feature_names=self.cyclical_feature_names, 
            do_cyclical_norm=self.do_cyclical_norm
            )
        
        bin_df = calculate_and_add_bin_features(
            pt_df=df,
            datetimes=df['datetime'],
            hpGrid=self.hpGrid, 
            base_bin_feature_names=self.base_bin_feature_names, 
            prenorm_bin_feature_names=self.prenorm_expanded_bin_feature_names, 
            bin_feature_names=self.bin_feature_names, 
            cyclical_feature_names=self.cyclical_feature_names, 
            do_cyclical_norm=self.do_cyclical_norm, 
            field2radec=field2radec,
            night2fieldvisits=night2fieldvisits,
            fieldfilter2maxvisits=fieldfilter2maxvisits,
            night2filtervisithistory=night2filtervisithistory,
            field2maxvisits=field2maxvisits,
            bin_space=bin_space
        )

        if save_df:
            self._df = df # Save for diagnostics
            self._bin_df = bin_df # Save for diagnostics
        del df, bin_df
        gc.collect()
                    
        # Save night dates, total number of nights in dataset, and number of obs per night
        groups_by_night = df.groupby('night')
        self.unique_nights = df['night'].unique()
        self.n_nights = groups_by_night.ngroups
        self.n_obs_per_night = groups_by_night.size() # nights have different numbers of observations

        # Construct Transitions
        states, next_states, bin_states, next_bin_states, self.bin_actions, self.rewards, self.dones, self.action_masks, self.num_transitions \
            = self._construct_transitions(
            df=df, 
            bin_df=bin_df,  
            include_bin_features=include_bin_features, 
            num_bins_1d=num_bins_1d, 
            binning_method=binning_method, 
            bin_space=bin_space,
            remove_large_time_diffs=remove_large_time_diffs
            )

        
        logger.info(f"States shape: {states.shape}, Actions shape: {self.bin_actions.shape}, Rewards shape: {self.rewards.shape}, Next states shape: {next_states.shape}, Dones shape: {self.dones.shape}, Action masks shape: {action_masks.shape}")
        logger.info(f"Bin states shape: {bin_states.shape if bin_states is not None else None}, Next bin states shape: {next_bin_states.shape if next_bin_states is not None else None}")

        # Set dimension of observation
        self.state_dim = self.states.shape[-1]
        if self._grid_network is None:
            state_feature_names = self.global_feature_names + self.bin_feature_names
        elif self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
            self.bin_state_dim = self.bin_states.shape[-1]
            state_feature_names = self.global_feature_names

        # Normalize states and next_states

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
        self.next_states = normalize_noncyclic_features(
            state=next_states,
            state_feature_names=state_feature_names,
            max_norm_feature_names=self.max_norm_feature_names,
            ang_distance_norm_feature_names=self.ang_distance_feature_names,
            do_inverse_norm=self.do_inverse_norm,
            do_max_norm=self.do_max_norm,
            do_ang_distance_norm=self.do_ang_distance_norm,
            fix_nans=True
        )

        if self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
            self.bin_states = normalize_noncyclic_features(
                state=bin_states,
                state_feature_names=self.bin_feature_names,
                max_norm_feature_names=self.max_norm_feature_names,
                ang_distance_norm_feature_names=self.ang_distance_feature_names,
                do_inverse_norm=self.do_inverse_norm,
                do_max_norm=self.do_max_norm,
                do_ang_distance_norm=self.do_ang_distance_norm,
                fix_nans=True
            )
            self.next_bin_states = normalize_noncyclic_features(
                state=next_bin_states,
                state_feature_names=self.bin_feature_names,
                max_norm_feature_names=self.max_norm_feature_names,
                ang_distance_norm_feature_names=self.ang_distance_feature_names,
                do_inverse_norm=self.do_inverse_norm,
                do_max_norm=self.do_max_norm,
                do_ang_distance_norm=self.do_ang_distance_norm,
                fix_nans=True
            )
        if include_bin_features:
            assert self.states.shape[0] == self.bin_actions.shape[0] == self.rewards.shape[0] == self.next_states.shape[0] == self.dones.shape[0] == self.action_masks.shape[0] == self.bin_states.shape[0] == self.next_bin_states.shape[0], f"Shape mismatch: states {self.states.shape}, actions {self.bin_actions.shape}, rewards {self.rewards.shape}, next_states {self.next_states.shape}, dones {self.dones.shape}, action_masks {self.action_masks.shape}, bin_states {self.bin_states.shape}, next_bin_states {self.next_bin_states.shape}"
        else:
            assert self.states.shape[0] == self.bin_actions.shape[0] == self.rewards.shape[0] == self.next_states.shape[0] == self.dones.shape[0] == self.action_masks.shape[0], f"Shape mismatch: states {self.states.shape}, actions {self.bin_actions.shape}, rewards {self.rewards.shape}, next_states {self.next_states.shape}, dones {self.dones.shape}, action_masks {self.action_masks.shape}"
        # self._do_noncyclic_normalizations()

    def _construct_transitions(self, df, bin_df, include_bin_features, num_bins_1d, binning_method, bin_space, remove_large_time_diffs):
        if remove_large_time_diffs:
            next_state_idxs = self._get_next_state_indices(df)
        else:
            next_state_idxs = None
        states, next_states, bin_features, next_bin_features = self._construct_states(df=df, bin_df=bin_df, include_bin_features=include_bin_features, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        num_transitions = states.shape[0]
        actions = self._construct_actions(df, next_states=next_states, bin_space=bin_space, binning_method=binning_method, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs, num_bins_1d=num_bins_1d)
        rewards = self._construct_rewards(df, next_state_idxs=next_state_idxs, remove_large_time_diffs=remove_large_time_diffs, reward_choice=self.reward_choice)
        dones = np.zeros(num_transitions, dtype=bool) # False unless last observation of the night
        dones[-1] = True
        # dones = df.groupby('night').apply(lambda x: [False]*(len(x)-1) + [True]).explode().values.astype(bool)
        # dones = df.groupby('night').apply(lambda x: [False]*(len(x)-1) + [True]).explode().values
        action_masks = self._construct_action_masks(df=df, bin_space=bin_space, num_transitions=num_transitions, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        
        states = torch.as_tensor(states, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        bin_actions = torch.as_tensor(bin_actions, dtype=torch.int32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.bool)
        action_masks = torch.as_tensor(action_masks, dtype=torch.bool)

        if include_bin_features:
            bin_features = torch.as_tensor(bin_features, dtype=torch.float32)
            next_bin_features = torch.as_tensor(next_bin_features, dtype=torch.float32)
        else:
            bin_features = None
            next_bin_features = None
            
        return states, next_states, bin_features, next_bin_features, actions, rewards, dones, action_masks, num_transitions

    def _construct_states(self, df, bin_df, include_bin_features, remove_large_time_diffs, next_state_idxs):
        if remove_large_time_diffs:
            global_features, next_global_features = self._construct_global_features(df=df, remove_large_time_diffs=True, next_state_idxs=next_state_idxs)
            if include_bin_features:
                bin_states, next_bin_states = self._construct_bin_features(bin_df=bin_df, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
                if self._grid_network is None:
                    self.bin_states = np.array([])
                    self.next_bin_states = np.array([])
                    states = np.concatenate((global_features, bin_states), axis=1)
                    next_states = np.concatenate((next_global_features, next_bin_states), axis=1)
                    return states, next_states, bin_states, next_bin_states
                elif self._grid_network  in ['single_bin_scorer', 'multi_dim_scorer']:
                    self.bin_states = bin_states
                    self.next_bin_states = next_bin_states
                    return global_features, next_global_features, bin_states, next_bin_states
                else:
                    raise NotImplementedError(f"Grid network type {self._grid_network} not implemented for state construction.")
            return global_features, next_global_features, None, None
        else:
            global_features, next_global_features = self._construct_global_features(df=df, remove_large_time_diffs=remove_large_time_diffs)
            if include_bin_features:
                bin_states, next_bin_states = self._construct_bin_features(bin_df=bin_df, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=None)
                self.bin_states = bin_states
                self.next_bin_states = next_bin_states
                states = np.concatenate((global_features, bin_states), axis=1)
                next_states = np.concatenate((next_global_features, next_bin_states), axis=1)
                return states, next_states
            return global_features, next_global_features
    
    def _get_next_state_indices(self, df, max_time_diff_min=10):
        time_diffs = df['timestamp'].diff().values
        keep = time_diffs < max_time_diff_min * 60 + 90
        next_state_idxs = np.where(keep)[0]
        self.next_state_idxs = next_state_idxs  # Save for diagnostics
        logger.debug(f'Removing {np.sum(~keep)} transitions with large time diffs > {max_time_diff_min} min. Total transitions: {len(keep)}')
        return next_state_idxs

    def _construct_global_features(self, df, remove_large_time_diffs, next_state_idxs=None):
        """
        Constructs state and next_states for all transitions.
        Inserts a "null" observation before the first observation each night.
        The null observation state is defined as being an array of zeros
        """
        # global features already in DECam data
        missing_cols = set(self.global_feature_names) - set(df.columns) == 0
        assert missing_cols == 0, f'Features {missing_cols} do not exist in dataframe. Must be implemented in method self._process_dataframe()'
        if remove_large_time_diffs:
            next_state_df = df.iloc[next_state_idxs]
            current_state_df = df.iloc[next_state_idxs - 1]
            global_features = current_state_df[self.global_feature_names].to_numpy()
            next_global_features = next_state_df[self.global_feature_names].to_numpy()
        else:
            global_features = df.groupby('night')[self.global_feature_names].apply(lambda group: group.iloc[:-1, :]).to_numpy()
            next_global_features = df.groupby('night')[self.global_feature_names].apply(lambda group: group.iloc[1:, :]).to_numpy()
        return global_features, next_global_features
        
    def _construct_bin_features(self, bin_df, remove_large_time_diffs, next_state_idxs):
        # Get bin_features and next_bin_features
        if remove_large_time_diffs:
            next_bindf = bin_df.iloc[next_state_idxs][self.bin_feature_names]
            bindf = bin_df.iloc[next_state_idxs - 1][self.bin_feature_names]
            if self._grid_network is None:
                bin_states = bindf.to_numpy()
                next_bin_states = next_bindf.to_numpy()
            elif self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
                nrows, ncols = bindf.shape
                num_feats_per_bin = int(ncols / self.num_bins)
                bin_states = bindf.to_numpy().reshape((nrows, self.num_bins, num_feats_per_bin))
                next_bin_states = next_bindf.to_numpy().reshape((nrows, self.num_bins, num_feats_per_bin))
            else:
                raise NotImplementedError(f"Grid network type {self._grid_network} not implemented for bin feature construction.")
        else:
            if self._grid_network is None:
                bin_states = bin_df.groupby('night')[self.bin_feature_names].apply(lambda group: group.iloc[:-1, :]).to_numpy()
                next_bin_states = bin_df.groupby('night')[self.bin_feature_names].apply(lambda group: group.iloc[1:, :]).to_numpy()
            elif self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
                bindf = bin_df.groupby('night')[self.bin_feature_names].apply(lambda group: group.iloc[:-1, :])
                next_bindf = bin_df.groupby('night')[self.bin_feature_names].apply(lambda group: group.iloc[1:, :])
                A, B = bindf.shape
                num_feats_per_bin = int(B / self.num_bins)
                bin_states = bindf.to_numpy().reshape((A, self.num_bins, num_feats_per_bin))
                next_bin_states = next_bindf.to_numpy().reshape((A, self.num_bins, num_feats_per_bin))
            else:
                raise NotImplementedError(f"Grid network type {self._grid_network} not implemented for bin feature construction.")
        return bin_states, next_bin_states
    
    def _construct_actions(self, df, next_states, bin_space, binning_method, remove_large_time_diffs, next_state_idxs, num_bins_1d=None):
        assert bin_space in ['radec', 'azel', 'radec_filter', 'azel_filter'], 'bin_space must be radec or azel'
        assert binning_method in ['uniform', 'healpix'], 'bining_method must be uniform or healpix'

        if binning_method == 'healpix':
            if remove_large_time_diffs:
                next_state_df = df.iloc[next_state_idxs]
                if self.hpGrid.is_azel:
                    lonlat = next_state_df[['az', 'el']].values
                else:
                    lonlat = next_state_df[['ra', 'dec']].values
                bin_indices = self.hpGrid.ang2idx(lon=lonlat[:, 0], lat=lonlat[:, 1])
                indices = bin_indices

                if 'filter' in bin_space:
                    filter_indices = next_state_df['filter'].map(FILTER2IDX).fillna(0).values.astype(np.int32)
                    indices = (bin_indices * NUM_FILTERS) + filter_indices
            else:
                if self.hpGrid.is_azel:
                    lonlat_no_zen = df.groupby('night').tail(-1)[['az', 'el']].values
                else:
                    # lon, lat = df.ra.values, df.dec.values
                    lonlat_no_zen = df.groupby('night').tail(-1)[['ra', 'dec']].values
                indices = self.hpGrid.ang2idx(lon=lonlat_no_zen[:, 0], lat=lonlat_no_zen[:, 1])
            return indices
        
        elif binning_method == 'uniform' and bin_space == 'azel':
            raise NotImplementedError
            az_edges = np.linspace(0, 360, num_bins_1d + 1, dtype=np.float32)
            # az_centers = az_edges[:-1] + (az_edges[1] - az_edges[0])/2
            el_edges = np.linspace(0, 90, num_bins_1d + 1, dtype=np.float32)
            # az_centers = el_edges[:-1] + (el_edges[1] - el_edges[0])/2

            i_x = np.digitize(next_states[:, az_idx], az_edges).astype(np.int32) - 1
            i_y = np.digitize(next_states[:, el_idx], el_edges).astype(np.int32) - 1
            bin_ids = i_x + i_y * (num_bins_1d)
            self.az_edges = az_edges
            self.el_edges = el_edges
            
            id2azel = defaultdict(list)
            for az, el, bin_id in zip(next_states[:, az_idx], next_states[:, el_idx], bin_ids):
                id2azel[bin_id].append((az, el))            
            self.id2azel = dict(sorted(id2azel.items()))

            id2radec = defaultdict(list)
            for ra, dec, bin_id in zip(df['ra'].values, df['dec'].values, bin_ids):
                id2radec[bin_id].append((ra, dec))
            self.id2radec = dict(sorted(id2radec.items()))

            return bin_ids
        else:
            raise NotImplementedError

    def _construct_rewards(self, df, remove_large_time_diffs, next_state_idxs, reward_choice='teff_rate'):
        assert reward_choice in ['teff_rate', 'expert_actions'], 'reward_choice must be teff_rate or expert_actions'
        """Constructs rewards for all transitions. Reward is defined as teff, normalized to [0, 1]."""
        if reward_choice == 'teff_rate':
            rewards = get_inst_teff_rate(df=df, remove_large_time_diffs=remove_large_time_diffs, next_state_idxs=next_state_idxs)
        elif reward_choice == 'expert_actions':
            if remove_large_time_diffs:
                next_state_df = df.iloc[next_state_idxs]
                rewards = np.ones(len(next_state_df), dtype=np.float32)
        return rewards

    def _construct_action_masks(self, df, bin_space, num_transitions, remove_large_time_diffs, next_state_idxs):
        """
        Constructs action masks only with the condition that bins outside of horizon are masked
        """
        if remove_large_time_diffs:
            df = df.iloc[next_state_idxs-1]
        # given timestamp, determine bins which are outside of observable range
        els = np.empty((num_transitions, self.num_actions))
        if self._calculate_action_mask:
            logger.info("Calculating action masks based on horizon. This may take a few minutes...")
            if not self.hpGrid.is_azel:
                lon, lat = self.hpGrid.lon, self.hpGrid.lat
                for i, time in tqdm(enumerate(df['timestamp'].values), total=len(df['timestamp'].values), desc="Calculating action mask"):
                    _, els[i] = ephemerides.equatorial_to_topographic(ra=lon, dec=lat, time=time)
                self._els = els
                action_mask = els > 0
            else:
                els = np.tile(self.hpGrid.lat[:, np.newaxis], reps=len(df['timestamp'].values)).T
                action_mask = els > 0
            if 'filter' in bin_space:
                action_mask = np.repeat(action_mask, NUM_FILTERS, axis=1)
        else:
            action_mask = np.ones((num_transitions, self.num_actions))
        return action_mask
        
    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        if self._grid_network is None:
            transition = (
                self.states[idx],
                self.bin_actions[idx],
                self.rewards[idx],
                self.next_states[idx],
                self.dones[idx],
                self.action_masks[idx],
                torch.as_tensor(0), # placeholder for bin state since not used in this case
                torch.as_tensor(0)
            )
        elif self._grid_network in ['single_bin_scorer', 'multi_dim_scorer']:
            transition = (
                self.states[idx],
                self.bin_actions[idx],
                self.rewards[idx],
                self.next_states[idx],
                self.dones[idx],
                self.action_masks[idx],
                self.bin_states[idx], # shape (ntransitions, nbins, nfeatures)
                self.next_bin_states[idx]
            )
        return transition
    
    def get_dataloader(self, batch_size, num_workers, pin_memory, random_seed, drop_last=True, val_split=.1, return_train_and_val=True):
        generator = torch.Generator().manual_seed(random_seed)
        np.random.seed(random_seed) # Ensure consistent night selection
        
        # 1. Randomly sample whole nights for the validation set
        num_val_nights = max(1, int(self.n_nights * val_split))
        val_nights = np.random.choice(self.unique_nights, size=num_val_nights, replace=False)
        logger.info(f'Choosing {num_val_nights} nights for validation out of {self.n_nights} nights. Specifically, {val_nights}')
        
        # 2. Track which night each transition belongs to
        # We must handle the two different ways states are constructed
        if hasattr(self, 'next_state_idxs') and self.next_state_idxs is not None:
            # When remove_large_time_diffs is True
            transition_nights = self._df.iloc[self.next_state_idxs - 1]['night']
        else:
            # When remove_large_time_diffs is False
            transition_nights = self._df.groupby('night')['night'].apply(lambda x: x.iloc[:-1])
            
        # 3. Create boolean mask mapping transitions to the selected val nights
        # print(transition_nights, val_nights)
        val_mask = np.isin(transition_nights, val_nights)
        
        # 4. Extract the exact indices for train and val subsets
        train_indices = np.where(~val_mask)[0].tolist()
        val_indices = np.where(val_mask)[0].tolist()
        
        # 5. Create Subsets
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

class ToyDatasetv0:
    """
    A dataset wrapper converting a nightly telescope schedule into a structured transition dataset.
    First version of dataset class for telescope data. Designed for behavior cloning where the original
     schedule is mimicked exactly (no generalization).
    
    Attributes
    ----------
        obs (ndarray): The full set of observations; has shape (obs_dim, n_nights, n_transitions) and dtype float32
            to be input into the q-network.
        actions (ndarray): The array of field ids of shape (n_nights, n_transitions) which indicate the
            next field id to be observed
        next_obs (ndarray): The next observations given obs and action; shape (obs_dim, n_nights, n_transitions)
        rewards (ndarray): The reward of doing next_obs given obs and action; shape (n_nights, n_transitions)
        dones (ndarray): Boolean array indicating whether or not this action terminates the episode; 
            shape (n_nights, n_transitions)
        action_masks (ndarray): Boolean array indicating valid possible actions

        device (str): The device to place tensors on ('cpu' or 'cuda')
        normalize_obs (bool): Whether to normalize observations.
        norm (ndarray): Normalization constants applied to observations
        obs_dim (int): Size of observations for a single transition
        num_actions (int): Size of action space
        reward_func (Callable | None): User-provided reward_function

        _radec (ndarray): RA/Dec positions for each scheduled field
        _obs_indices (ndarray): A time index for each observation made in a single night (starts at 0)
        _schedule_field_id (ndarray): The actual, time-ordered schedule; has shape (n_nights, n_obs_in_night) where
            n_obs_in_night = n_transitions + 1
        _unique_field_ids (ndarray): An array of the unique field ids in increasing order
        _n_obs_per_night (int): Number of observations in each night
        _n_transitions (int): Number of transitions in dataset
        _id2pos (dict): A dictionary of field_id as keys and values as coordinates

        _

    """
    def __init__(self, schedule, id2pos, reward_func=None, normalize_obs=True, device='cpu'):
        """
        Args
        ---- 
            schedule (pd.Dataframe): Pandas dataframe with columns 'field_id' and 'next_field_id'
            id2pos (dict): Mapping from field ID to its (RA, Dec) coordinates
            reward_func (Callable | None): Optional custom reward ufnction. Should have signature: ``reward_func(obs, action, next_obs) -> float or array``
            normalize_obs (bool): Whether or not to normalize observations
            device (str): Device where tensors will be moved ('cuda' or 'CPU')
        """
        self._schedule_field_ids = schedule.field_id.values.astype(np.float32)
        if len(self._schedule_field_ids.shape) == 1:
            self._schedule_field_ids = self._schedule_field_ids[np.newaxis, :]
        assert len(self._schedule_field_ids.shape) == 2

        # vars used internally to help calculate transition variables
        self._radec = np.array([id2pos[field_id] for field_id in schedule.field_id.values], dtype=np.float32)
        self._unique_field_ids, counts = np.unique(self._schedule_field_ids, return_counts=True)
        self._max_visits = np.int32(np.max(counts))
        self._nfields = np.int32(len(self._unique_field_ids))
        self._n_obs_per_night = np.int32(self._schedule_field_ids.shape[-1])
        self._n_transitions = np.int32(len(self._radec))
        self._id2pos = id2pos
        self.device = device
        self.normalize_obs = normalize_obs
        self.norm = 1
        if self.normalize_obs:
            self.norm = np.array([self._n_obs_per_night, self._nfields])[:, np.newaxis]

        # 
        self.reward_func = reward_func

        self._obs_indices = schedule.index.to_numpy()[np.newaxis, np.newaxis, :-1]
        self._field_ids = schedule.field_id.values[:-1][np.newaxis, np.newaxis, :]
        self._next_obs_indices = schedule.index.to_numpy()[np.newaxis, np.newaxis, 1:]
        self._next_field_ids = schedule.field_id.values[1:][np.newaxis, np.newaxis, :]

        self.obs = np.concatenate((self._field_ids, self._obs_indices), axis=0)
        self.next_obs = np.concatenate((self._next_field_ids, self._next_obs_indices), axis=0)

        self.obs_dim = self.obs.shape[0]
        self._n_nights = np.int32(self.obs.shape[1])
        self.num_actions = self._nfields 

        self.actions = schedule.field_id.values[1:][np.newaxis, :]
        self.rewards = self._get_rewards()
        self.dones = np.zeros_like(self.obs[0], dtype=np.bool_)
        self.dones[:, -1] = True
        self.action_masks = self._get_action_masks()

    def _get_rewards(self):
        """Compute rewards for each transition
        
        Returns
        -------
            Array of shape (n_nights, n_obs_per_night - 1) containing rewards. If ```reward_func``` is None, returns an array of ones
        """
        if self.reward_func is None:
            return np.ones_like(self.obs[0])
        return self._reward_func(self.obs, self.actions, self.next_obs)
    
    def _get_action_masks(self):
        """Computes action masks based on per-field visit limits.

        Tracks cumulative number of visits per field and disallows actions
        (fields) that exceed the maximum visit count observed in the schedule.

        Returns
        --------
            Boolean array of shape (n_nights, n_obs_per_night, n_fields) where ``True`` indicates the action is allowed.
        """
        nvisits_base = np.zeros(shape=(self._nfields), dtype=np.int32)
        full_nvisits = np.zeros(shape=(self._n_nights, self._n_obs_per_night, self._nfields), dtype=np.int32)

        for i, night_ids in enumerate(self._schedule_field_ids):
            for j, field_id in enumerate(night_ids):
                nvisits_base[np.int32(field_id)] += 1
                full_nvisits[i, j] = nvisits_base.copy()
        action_masks = full_nvisits != self._max_visits
        return action_masks

    def __len__(self):
        """Number of nights available in the dataset.

        Returns:
            int: Number of nights.
        """
        return self._n_nights

    def sample(self, batch_size):
        """Randomly samples a batch of transitions. Sampling is performed by uniformly selecting a random night and a random observation step
            within that night
        """
        #TODO should flatten transition dataset first, then sample uniformly
        night_indices = np.random.choice(self._n_nights, batch_size, replace=True)
        obs_indices = np.random.choice(self._n_obs_per_night - 1, batch_size)
        return (
            np.array(self.obs[:, night_indices, obs_indices]/self.norm, dtype=np.float32).T, # needs to be float for network
            np.array(self.actions[night_indices, obs_indices], dtype=np.int32),
            np.array(self.rewards[night_indices, obs_indices], dtype=np.float32),
            np.array(self.next_obs[:, night_indices, obs_indices]/self.norm, dtype=np.float32).T,
            np.array(self.dones[night_indices, obs_indices], dtype=np.bool_),
            np.array(self.action_masks[night_indices, obs_indices], dtype=bool),
        )
    