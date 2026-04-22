import random

import numpy as np
import torch

from blancops.data.constants import NO_FILTER_SIGNAL, WAIT_SIGNAL
from blancops.ephemerides import ephemerides
from blancops.math.interpolate import interpolate_on_sphere

import logging
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, algorithm, cfg, lookups, field_choice_method='interp'):
        self.algorithm = algorithm
        self.lookups = lookups
        self.cfg = cfg
        self.device = algorithm.device
        self.field_choice_method = field_choice_method
        
    def _choose_bin_and_filter(self, x_glob, x_bin, action_mask, epsilon):
        """Selects an action using the underlying algorithm.

        Args:
            x_glob (array-like):
                Pointing and global state features (normalized if applicable).
            x_bin (array-like):
                Per-bin features (normalized if applicable).
            action_mask (array-like | None):
                Boolean mask indicating which actions are legal.
            epsilon (float | None):
                Epsilon for epsilon-greedy exploration. If None, selects greedily.

        Returns:
            int: Selected action index.
        """
        action = self.algorithm.select_action(x_glob=x_glob, x_bin=x_bin, action_mask=action_mask, epsilon=epsilon)
        if 'filter' in self.cfg.data.action_space:
            bin_idx = int(action // self.algorithm.policy.num_filters)
            filter_idx = int(action % self.algorithm.policy.num_filters)
        else:
            bin_idx = action
            filter_idx = NO_FILTER_SIGNAL
        return bin_idx, filter_idx

    def _determine_valid_fields(self, bin_idx, filter_idx, info):
        # Unpack info and get valid fields in bin
        valid_fields_per_bin = info.get('valid_fields_per_bin', {})
        valid_fields_in_bin = np.array(valid_fields_per_bin.get(int(bin_idx), []))
        assert len(valid_fields_in_bin) != 0, f"No valid fields are in bin {bin_idx}. Check environment's output mask."
        
        s_visited = info.get('s_visited', None)
        s_filter_visits = info.get('s_filter_visits', None)
        max_s_filter_visits = info.get('max_s_filter_visits', None)

        # Filter out completed fields in (bin, filter)
        if (s_filter_visits is not None) and (max_s_filter_visits is not None) and (filter_idx >= 0):
            field_ids_in_bin = [fid for fid in valid_fields_in_bin if s_filter_visits[fid, filter_idx] < max_s_filter_visits[fid, filter_idx]]
        else:
            field_ids_in_bin = [fid for fid in valid_fields_in_bin if s_visited[fid] < self.lookups.target_fid_counts[fid]]
        
        assert len(field_ids_in_bin) != 0, "No valid fields are in bin...check environment's output mask."
        logger.debug(f'Chosen bin contains {len(field_ids_in_bin)} incomplete fields out of {len(valid_fields_in_bin)} fields total')
        return field_ids_in_bin
        
    def choose_bin_filter_field(self, obs, info, hpGrid, epsilon=None): 
        """
        Choose field in bin based on interpolated Q-values
        """
        # Unpack obs
        x_glob = obs['global_state']
        x_bin = obs['bin_state']
        
        # Choose action in action space
        bin_idx, filter_idx = self._choose_bin_and_filter(x_glob, x_bin, info.get('action_mask', None), epsilon)

        # Get valid fields in bin
        valid_field_ids = self._determine_valid_fields(bin_idx, filter_idx, info)

        if self.field_choice_method == 'interp':
            with torch.no_grad():
                # Ensure tensors have the batch dimension expected by ScoreMLP
                glob_tensor = torch.as_tensor(x_glob, device=self.device, dtype=torch.float32).unsqueeze(0)
                bin_tensor = torch.as_tensor(x_bin, device=self.device, dtype=torch.float32).unsqueeze(0)
                
                # Get raw joint scores from MLP: shape (1, n_bins * n_filters)
                raw_scores = self.algorithm.policy.core_net(glob_tensor, bin_tensor)
                
                n_bins = bin_tensor.shape[1]
                n_filters = raw_scores.shape[-1] // n_bins
                
                # Reshape to (n_bins, n_filters) and slice the specific filter
                q_map = raw_scores.view(n_bins, n_filters)[:, filter_idx].cpu().numpy()

            lon_data = hpGrid.lon 
            lat_data = hpGrid.lat

            # CHECK
            # target_coords = np.array([fid2radec[fid] for fid in field_ids_in_bin])
            target_coords = np.array([self.lookups.fid2radec[fid] for fid in valid_field_ids])
            
            if hpGrid.is_azel:
                # Project RA/Dec to local Az/El frame using the current timestamp
                timestamp = info.get('timestamp')
                target_lons, target_lats = ephemerides.equatorial_to_topographic(
                    ra=target_coords[:, 0], 
                    dec=target_coords[:, 1], 
                    time=timestamp
                )
            else:
                target_lons = target_coords[:, 0]
                target_lats = target_coords[:, 1]

            q_interpolated = interpolate_on_sphere(
                az=target_lons,
                el=target_lats,  # Target coordinates
                az_data=lon_data,
                el_data=lat_data,        # Bin centers (grid)
                values=q_map                      # Filter-specific Q-values
            )
            
            best_idx = np.argmax(q_interpolated)

            field_id = valid_field_ids[best_idx]

        elif self.field_choice_method == 'random':
            field_id = random.choice(valid_field_ids)
        
        return bin_idx, filter_idx, field_id