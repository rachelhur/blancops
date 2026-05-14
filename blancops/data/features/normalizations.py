import torch
import numpy as np
import logging

from blancops.configs.schema import NormalizationConfig

logger = logging.getLogger(__name__)

def build_normalizer(state_feature_names, cfg):
    norm_kwargs = build_normalizer_kwargs(cfg.data.norm)
    return StateNormalizer(state_feature_names=state_feature_names, **norm_kwargs)

def build_normalizer_kwargs(norm_config: NormalizationConfig) -> dict:
    """Translates the Pydantic schema into the exact kwargs expected by StateNormalizer."""
    kwargs = {
        'cyclical_feature_names': [],
        'sin_norm_feature_names': [],
        'log_norm_feature_names': [],
        'fractional_norm_feature_names': [],
        'z_score_feature_names': [],
        'local_mean_z_score_feature_names': [],
    }
    
    name_map = {
        'cyclical': 'cyclical_feature_names',
        'sin': 'sin_norm_feature_names',
        'log': 'log_norm_feature_names',
        'fractional': 'fractional_norm_feature_names',
        'z_score': 'z_score_feature_names',
        'local_mean_z': 'local_mean_z_score_feature_names'
    }
    
    for feature, requested_norms in norm_config.feature_norm_mappings.items():
        for norm in requested_norms:
            target_list = name_map[norm]
            kwargs[target_list].append(feature)
    
    kwargs['do_cyclical_norm'] = len(kwargs['cyclical_feature_names']) > 0
    kwargs['do_sin_norm'] = len(kwargs['sin_norm_feature_names']) > 0
    kwargs['do_log_norm'] = len(kwargs['log_norm_feature_names']) > 0
    kwargs['do_fractional_norm'] = len(kwargs['fractional_norm_feature_names']) > 0
    kwargs['do_z_score_norm'] = len(kwargs['z_score_feature_names']) > 0
    kwargs['do_local_mean_z_score'] = len(kwargs['local_mean_z_score_feature_names']) > 0
    
    # Pass through the fix_nans flag
    kwargs['fix_nans'] = norm_config.fix_nans
    
    return kwargs

def expand_feature_names_for_cyclic_norm(feature_names, cyclical_feature_names):
    feature_names_out = []
    for feat_name in feature_names:
        is_rel_feat = feat_name.startswith('rel_')
        is_delta_feat = feat_name.startswith('delta_')
        never_cyclic_feat = is_rel_feat or is_delta_feat
        is_cyclic = any((feat_name == cyc_feat) or feat_name.endswith(f"_{cyc_feat}") for cyc_feat in cyclical_feature_names)
        
        if is_cyclic and not never_cyclic_feat:
            logger.info(f"Expanding {feat_name} to {feat_name}_cos and {feat_name}_sin")
            feature_names_out.extend([f"{feat_name}_cos", f"{feat_name}_sin"])
        else:
            feature_names_out.append(feat_name)
    return feature_names_out

def setup_feature_names(base_global_feature_names, base_bin_feature_names, cyclical_feature_names, do_cyclical_norm):
    if do_cyclical_norm:
        global_feature_names = expand_feature_names_for_cyclic_norm(base_global_feature_names.copy(), cyclical_feature_names)
        bin_feature_names = expand_feature_names_for_cyclic_norm(base_bin_feature_names.copy(), cyclical_feature_names)
    else:
        global_feature_names = base_global_feature_names.copy()
        bin_feature_names = base_bin_feature_names.copy()
    return global_feature_names, bin_feature_names

def apply_cyclical_features(features, base_names, cyclical_names):
    for name in base_names:
        if name.startswith(("rel_", "delta_")):
            continue

        if name not in features:
            continue

        if any(name == cyc or name.endswith(f"_{cyc}") for cyc in cyclical_names):
            features[f"{name}_cos"] = np.cos(features[name])
            features[f"{name}_sin"] = np.sin(features[name])


class StateNormalizer:
    def __init__(
        self, 
        state_feature_names,
        sin_norm_feature_names,
        log_norm_feature_names,
        fractional_norm_feature_names,
        z_score_feature_names,
        local_mean_z_score_feature_names,
        do_sin_norm=True, 
        do_log_norm=True, 
        do_fractional_norm=True, 
        do_z_score_norm=True, 
        do_local_mean_z_score=True,
        fix_nans=True,
        do_cyclical_norm=None,
        cyclical_feature_names=None,
        sentinel_value=-1
    ):
        self.feature_names = state_feature_names
        
        # Config Flags
        self.do_sin = do_sin_norm
        self.do_log = do_log_norm
        self.do_frac = do_fractional_norm
        self.do_z = do_z_score_norm
        self.do_rel = do_local_mean_z_score
        self.do_cyclical_norm = do_cyclical_norm
        self.cyclical_feature_names = cyclical_feature_names
        self.sentinel_value = sentinel_value
        self.fix_nans = fix_nans

        # Pre-compute masks ONCE during initialization
        self._build_masks(
            sin_norm_feature_names, 
            log_norm_feature_names, 
            fractional_norm_feature_names, 
            z_score_feature_names, 
            local_mean_z_score_feature_names
        )

    def _build_masks(self, sin_feats, log_feats, frac_feats, z_feats, rel_feats):
        """Build boolean masks for each normalization type."""
        names = self.feature_names
        
        # Base exclusion masks
        rel_exclusion = np.array(['rel_' in f for f in names], dtype=bool)
        
        self.masks = {
            'sin': np.array([any(nf == f for nf in sin_feats) for f in names]) & ~rel_exclusion,
            'log': np.array([any(nf == f for nf in log_feats) for f in names]) & ~rel_exclusion,
            'frac': np.array([any(nf == f for nf in frac_feats) for f in names]) & ~rel_exclusion,
            'rel': np.array([any(nf == f for nf in rel_feats) for f in names]),
            'z': np.array([any(f == nf or f.endswith(f"_{nf}") for nf in z_feats) for f in names]) & ~rel_exclusion,
        }
        
        # Cache active feature names for dictionary building later
        self.active_features = {
            name: [f for f, m in zip(names, mask) if m] 
            for name, mask in self.masks.items()
        }

    def _get_backend(self, state):
        """Returns the appropriate math module and converts masks to the correct device."""
        is_torch = torch.is_tensor(state)
        math_backend = torch if is_torch else np
        
        # Convert pre-computed numpy masks to torch bool tensors if necessary
        active_masks = {}
        for key, mask in self.masks.items():
            if is_torch:
                active_masks[key] = torch.tensor(mask, dtype=torch.bool, device=state.device)
            else:
                active_masks[key] = mask
                
        return is_torch, math_backend, active_masks

    def fit_transform(self, state, train_state_idxs):
        """TRAINING MODE: Calculates stats from training indices, applies them, and returns the stats dictionaries."""
        if train_state_idxs is None:
            raise ValueError("train_state_idxs must be provided in fit_transform mode.")
            
        is_torch, backend, m = self._get_backend(state)
        self._apply_stateless_norms(state, backend, m)

        z_stats_out, rel_stats_out = {}, {}

        # 1. Z-Score (Global Mean/Std)
        if self.do_z and m['z'].sum() > 0:
            logger.info(f"[Normalizer] Performing Z-Score Normalization for {self.active_features['z']}")
            train_data = state[train_state_idxs][..., m['z']]
            train_flat = train_data.reshape(-1, train_data.shape[-1])
            
            mean = backend.nanmean(train_flat, dim=0) if is_torch else np.nanmean(train_flat, axis=0)
            std = self._calc_std(train_flat, mean, backend, is_torch)
            
            state[..., m['z']] = (state[..., m['z']] - mean) / std
            z_stats_out = self._build_stats_dict(self.active_features['z'], mean, std)

        # 2. Relative Local Mean Z-Score (Global Std only)
        if self.do_rel and m['rel'].sum() > 0:
            logger.info(f"[Normalizer] Performing Relative Local Mean Z-Score Normalization for {self.active_features['rel']}")
            train_data = state[train_state_idxs][..., m['rel']]
            train_flat = train_data.reshape(-1, train_data.shape[-1])
            
            mean = backend.nanmean(train_flat, dim=0) if is_torch else np.nanmean(train_flat, axis=0)
            std = self._calc_std(train_flat, mean, backend, is_torch)
            
            state[..., m['rel']] = state[..., m['rel']] / std
            rel_stats_out = self._build_stats_dict(self.active_features['rel'], mean, std)

        if self.fix_nans:
            nan_mask = backend.isnan(state)
            state[nan_mask] = self.sentinel_value
            assert state.isnan().sum() == 0, f"State contains nans"

        return state, z_stats_out, rel_stats_out, nan_mask

    def transform(self, state, z_stats_dict, rel_stats_dict):
        """INFERENCE MODE: Applies previously calculated stats dictionaries to the state."""
        is_torch, backend, m = self._get_backend(state)
        self._apply_stateless_norms(state, backend, m)

        # 1. Apply Z-Score
        if self.do_z and m['z'].sum() > 0:
            mean, std = self._extract_stats_arrays(z_stats_dict, self.active_features['z'], backend, state)
            state[..., m['z']] = (state[..., m['z']] - mean) / std

        # 2. Apply Relative Norm
        if self.do_rel and m['rel'].sum() > 0:
            _, std = self._extract_stats_arrays(rel_stats_dict, self.active_features['rel'], backend, state)
            state[..., m['rel']] = state[..., m['rel']] / std

        if self.fix_nans:
            state[backend.isnan(state)] = self.sentinel_value

        return state

    def _apply_stateless_norms(self, state, backend, m):
        if self.do_sin and m['sin'].sum() > 0:
            state[..., m['sin']] = backend.sin(state[..., m['sin']])
            # state[..., m['sin']][state[backend.isnan(state[..., m['sin']])]] 
        if self.do_log and m['log'].sum() > 0:
            state[..., m['log']] = backend.log(state[..., m['log']] + 1e-9)
        if self.do_frac and m['frac'].sum() > 0:
            state[..., m['frac']] = 2 * (state[..., m['frac']] - 0.5)

    def _calc_std(self, flat_data, mean, backend, is_torch):
        """Population std, NaN-aware, min-clipped to avoid zero-division. Unified across backends."""
        # var = backend.nanmean((flat_data - mean) ** 2, dim=0) if is_torch \
        #     else np.nanmean((flat_data - mean) ** 2, axis=0)
        # std = backend.sqrt(var)
        # return torch.clamp(std, min=1e-6) if is_torch else np.maximum(std, 1e-6)
        if is_torch:
            var = torch.nanmean((flat_data - mean)**2, dim=0)
            return torch.clamp(torch.sqrt(var), min=1e-6)
        else:
            return np.clip(np.nanstd(flat_data, axis=0), a_min=1e-6, a_max=None)
            
    def _build_stats_dict(self, active_features, mean_arr, std_arr):
        """Converts internal tensors/arrays to standard Python floats for JSON serialization."""
        return {
            feat: {'mean': float(m), 'std': float(s)} 
            for feat, m, s in zip(active_features, mean_arr, std_arr)
        }

    def _extract_stats_arrays(self, stats_dict, active_features, backend, state):
        """Pulls stats from JSON-loaded dicts and formats them for math operations."""
        means, stds = [], []
        for feat in active_features:
            if feat not in stats_dict:
                raise KeyError(f"CRITICAL: Model expects normalization stats for '{feat}', but it is missing!")
            means.append(stats_dict[feat]['mean'])
            stds.append(stats_dict[feat]['std'])

        if backend == torch:
            return (
                torch.tensor(means, dtype=torch.float32, device=state.device),
                torch.tensor(stds, dtype=torch.float32, device=state.device)
            )
        return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)
    

def normalize_timestamp(timestamp, sunset_timestamp, sunrise_timestamp):
    return (timestamp - sunset_timestamp) / (sunrise_timestamp - sunset_timestamp)