import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize_timestamp(timestamp, sunset_timestamp, sunrise_timestamp):
    return (timestamp - sunset_timestamp) / (sunrise_timestamp - sunset_timestamp)

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
        if z_stats is not None: # inference mode
            mean = z_stats['mean'].detach().numpy()
            std = z_stats['std'].detach().numpy()
        else: # train mode
            if train_state_idxs is not None:
                train_z_states = state[train_state_idxs][..., z_score_mask]
            else:
                raise ValueError("Must pass train_state_idxs during normalization in training mode to perform z-score normalization")
            train_data_flat = train_z_states.reshape(-1, train_z_states.shape[-1])

            if is_torch:
                # torch.nanmean is available in PyTorch 1.11+
                mean = torch.nanmean(train_data_flat, dim=0)
                var = torch.nanmean((train_data_flat - mean)**2, dim=0)
                std = torch.clamp(torch.sqrt(var), min=1e-6)
            else:
                mean = np.nanmean(train_data_flat, axis=0)
                std = np.clip(np.nanstd(train_data_flat, axis=0), a_min=1e-6, a_max=None)
                    
            z_stats_out = {'mean': mean, 'std': std}
        state[..., z_score_mask] = (state[..., z_score_mask] - mean) / std
        
    rel_stats_out = {}
    if do_local_mean_z_score and (rel_norm_mask.sum() > 0):
        if rel_stats is not None:
            std = rel_stats['std'].detach().numpy()
        else:
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
                
            rel_stats_out = {'std': std}
            
        state[..., rel_norm_mask] = state[..., rel_norm_mask] / std
        
    if fix_nans:
        if is_torch:
            state[torch.isnan(state)] = 1.2
        else:
            state[np.isnan(state)] = 1.2
    return state, z_stats_out, rel_stats_out