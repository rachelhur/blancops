import json
from pathlib import Path

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def expand_feature_names_for_cyclic_norm(feature_names, cyclical_feature_names):
    feature_names_out = []
    for feat_name in feature_names:
        is_rel_feat = feat_name.startswith('rel_')
        is_delta_feat = feat_name.startswith('delta_')
        never_cyclic_feat = is_rel_feat & is_delta_feat
        is_cyclic = any((feat_name == cyc_feat) or feat_name.endswith(f"_{cyc_feat}") for cyc_feat in cyclical_feature_names)
        
        if is_cyclic and never_cyclic_feat:
            feature_names_out.extend([f"{feat_name}_cos", f"{feat_name}_sin"])
        else:
            feature_names_out.append(feat_name)
    return feature_names_out

def setup_feature_names(base_global_feature_names, base_bin_feature_names, cyclical_feature_names, do_cyclical_norm):
    # Replace cyclical features with their cyclical transforms/normalizations if on  
    if do_cyclical_norm:
        global_feature_names = expand_feature_names_for_cyclic_norm(base_global_feature_names.copy(), cyclical_feature_names)
        bin_feature_names = expand_feature_names_for_cyclic_norm(base_bin_feature_names.copy(), cyclical_feature_names)
    else:
        global_feature_names = base_global_feature_names
        bin_feature_names = base_bin_feature_names
    return global_feature_names, bin_feature_names

def normalize_timestamp(timestamp, sunset_timestamp, sunrise_timestamp):
    return (timestamp - sunset_timestamp) / (sunrise_timestamp - sunset_timestamp)

def sin_normalize(state, state_feature_names, sin_norm_feature_names, is_torch, rel_mask):
    sin_norm_mask = np.array([any(norm_feat in feat for norm_feat in sin_norm_feature_names) for feat in state_feature_names], dtype=bool) & ~rel_mask
    if is_torch:
        sin_func = torch.sin
        sin_norm_mask = torch.tensor(sin_norm_mask, dtype=torch.bool, device=state.device)
    else:
        sin_func = np.sin
    state[..., sin_norm_mask] = sin_func(state[..., sin_norm_mask])
    assert (state[..., sin_norm_mask] <= 1).all()
    assert (state[..., sin_norm_mask] >= -1).all()
    
def load_normalization_stats(load_dir):
    """Loads stats from JSON and unpacks them for two-pass normalization."""
    load_path = Path(load_dir) / "normalization_stats.json"
    
    with open(load_path, "r") as f:
        all_stats = json.load(f)
        
    # Safely unpack the Z-score dictionaries
    z_score_stats = all_stats.get("z_score", {})
    
    # Safely unpack the Relative dictionaries
    rel_norm_stats = all_stats.get("rel_norm", {})
    
    return z_score_stats, rel_norm_stats

def build_stats_tensor(loaded_stats_dict, expected_features, mask, is_torch=True, device='cpu'):
    """Rebuilds the mean/std tensors dynamically based on current feature order."""
    active_features = [f for f, m in zip(expected_features, mask) if m]
    
    means = []
    stds = []
    
    for feat in active_features:
        if feat not in loaded_stats_dict:
            raise KeyError(f"CRITICAL: Model expects normalization stats for '{feat}', but it is missing from the saved JSON!")
            
        means.append(loaded_stats_dict[feat]['mean'])
        stds.append(loaded_stats_dict[feat]['std'])
        
    if is_torch:
        return {
            'mean': torch.tensor(means, dtype=torch.float32, device=device),
            'std': torch.tensor(stds, dtype=torch.float32, device=device)
        }
    else:
        return {
            'mean': np.array(means, dtype=np.float32),
            'std': np.array(stds, dtype=np.float32)
        }

def normalize_noncyclic_features(state, 
                                state_feature_names,
                                sin_norm_feature_names,
                                log_norm_feature_names,
                                fractional_norm_feature_names,
                                z_score_feature_names,
                                local_mean_z_score_feature_names,
                                do_sin_norm, do_log_norm, do_fractional_norm, do_z_score_norm, do_local_mean_z_score,
                                fix_nans=True,
                                train_state_idxs=None,
                                z_stats=None,
                                rel_stats=None,
                                do_debug=True):
    is_torch = torch.is_tensor(state)
    
    if do_debug:
        if is_torch:
            assert not torch.isnan(state).any(), "NaNs detected in input state"
        else:
            assert not np.isnan(state).any(), "NaNs detected in input state"

    rel_mask = np.array(['rel_' in feat_name for feat_name in state_feature_names], dtype=bool)
    
    rel_norm_mask = np.array([any(norm_feat == feat for norm_feat in local_mean_z_score_feature_names) for feat in state_feature_names], dtype=bool)
    sin_norm_mask = np.array([any(norm_feat in feat for norm_feat in sin_norm_feature_names) for feat in state_feature_names], dtype=bool) & ~rel_mask
    log_norm_mask = np.array([any(norm_feat == feat for norm_feat in log_norm_feature_names) for feat in state_feature_names], dtype=bool) & ~rel_mask
    fractional_mask = np.array([any(norm_feat == feat for norm_feat in fractional_norm_feature_names) for feat in state_feature_names], dtype=bool) & ~rel_mask
    z_score_mask = np.array([
        any(feat == norm_feat or feat.endswith(f"_{norm_feat}") for norm_feat in z_score_feature_names) 
        for feat in state_feature_names
        ], dtype=bool) & ~rel_mask
    
    for m_name, m in zip(['z_score', 'fractional_norm', 'log_norm', 'sin_norm', 'local_mean_z_score'],
                         [z_score_mask, fractional_mask, log_norm_mask, sin_norm_mask, rel_norm_mask]
                         ):
        logger.debug(f"{m_name} will be applied to features: {[feat for feat, mask in zip(state_feature_names, m) if mask]}")
    
    sin_func = np.sin
    log_func = np.log
    if is_torch:
        sin_norm_mask = torch.tensor(sin_norm_mask, dtype=torch.bool, device=state.device)
        log_norm_mask = torch.tensor(log_norm_mask, dtype=torch.bool, device=state.device)
        fractional_mask = torch.tensor(fractional_mask, dtype=torch.bool, device=state.device)
        rel_norm_mask = torch.tensor(rel_norm_mask, dtype=torch.bool, device=state.device)
        z_score_mask = torch.tensor(z_score_mask, dtype=torch.bool, device=state.device)
        sin_func = torch.sin
        log_func = torch.log
    
    if do_sin_norm and (sin_norm_mask.sum() > 0):
        state[..., sin_norm_mask] = sin_func(state[..., sin_norm_mask])
        assert (state[..., sin_norm_mask] <= 1).all()
        assert (state[..., sin_norm_mask] >= -1).all()
    if do_log_norm and (log_norm_mask.sum()) > 0:
        state[..., log_norm_mask] = log_func(state[..., log_norm_mask] + 1e-9)
    if do_fractional_norm and fractional_mask.sum() > 0:
        state[..., fractional_mask] = 2 * (state[..., fractional_mask] - .5)
    
    z_stats_out = {}
    if do_z_score_norm and (z_score_mask.sum() > 0):
        if z_stats is not None: # INFERENCE MODE
            active_z_features = [f for f, m in zip(state_feature_names, z_score_mask) if m]
            means, stds = [], []
            
            for feat in active_z_features:
                if feat not in z_stats:
                    raise KeyError(f"CRITICAL: Missing Z-score stats for '{feat}' in loaded JSON!")
                means.append(z_stats[feat]['mean'])
                stds.append(z_stats[feat]['std'])
                
            # Define mean/std natively so UnboundLocalError is mathematically impossible
            if is_torch:
                mean = torch.tensor(means, dtype=torch.float32, device=state.device)
                std = torch.tensor(stds, dtype=torch.float32, device=state.device)
            else:
                mean = np.array(means, dtype=np.float32)
                std = np.array(stds, dtype=np.float32)
                
        else: # TRAIN MODE
            if train_state_idxs is not None:
                train_z_states = state[train_state_idxs][..., z_score_mask]
            else:
                raise ValueError("Must pass train_state_idxs during normalization in training mode to perform z-score normalization")
            train_data_flat = train_z_states.reshape(-1, train_z_states.shape[-1])

            if is_torch:
                mean = torch.nanmean(train_data_flat, dim=0)
                var = torch.nanmean((train_data_flat - mean)**2, dim=0)
                std = torch.clamp(torch.sqrt(var), min=1e-6)
            else:
                mean = np.nanmean(train_data_flat, axis=0)
                std = np.clip(np.nanstd(train_data_flat, axis=0), a_min=1e-6, a_max=None)
                    
            active_z_features = [f for f, m in zip(state_feature_names, z_score_mask) if m]
            z_stats_out = {
                feat: {'mean': float(m), 'std': float(s)} 
                for feat, m, s in zip(active_z_features, mean, std)
            }
            
        state[..., z_score_mask] = (state[..., z_score_mask] - mean) / std
        
    rel_stats_out = {}
    if do_local_mean_z_score and (rel_norm_mask.sum() > 0):
        if rel_stats is not None: # INFERENCE MODE
            active_rel_features = [f for f, m in zip(state_feature_names, rel_norm_mask) if m]
            stds = []
            
            for feat in active_rel_features:
                if feat not in rel_stats:
                    raise KeyError(f"CRITICAL: Missing relative stats for '{feat}' in loaded JSON!")
                stds.append(rel_stats[feat]['std'])
                
            if is_torch:
                std = torch.tensor(stds, dtype=torch.float32, device=state.device)
            else:
                std = np.array(stds, dtype=np.float32)
                
        else: # TRAIN MODE
            if train_state_idxs is None:
                raise ValueError("Must pass train_state_idxs to compute global std for rel normalization")
            train_rel_states = state[train_state_idxs][..., rel_norm_mask]
            train_rel_flat = train_rel_states.reshape(-1, train_rel_states.shape[-1])
            
            if is_torch:
                mean = torch.nanmean(train_rel_flat, dim=0)
                var = torch.nanmean((train_rel_flat - mean)**2, dim=0)
                std = torch.clamp(torch.sqrt(var), min=1e-6)
            else:
                mean = np.nanmean(train_rel_flat, axis=0)
                std = np.clip(np.nanstd(train_rel_flat, axis=0), a_min=1e-6, a_max=None)
                
            active_rel_features = [f for f, m in zip(state_feature_names, rel_norm_mask) if m]
            rel_stats_out = {
                feat: {'mean': float(m), 'std': float(s)} 
                for feat, m, s in zip(active_rel_features, mean, std)
            }
            
        state[..., rel_norm_mask] = state[..., rel_norm_mask] / std
        
    if fix_nans:
        if is_torch:
            state[torch.isnan(state)] = 1.2
        else:
            state[np.isnan(state)] = 1.2
            
    return state, z_stats_out, rel_stats_out