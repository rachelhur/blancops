import numpy as np
from einops import rearrange
from tqdm import tqdm
from blancops.ephemerides import ephemerides
from blancops.data.constants import *

import logging
logger = logging.getLogger(__name__)

class BinFeatureEngineer:
    def __init__(
        self, 
        hpGrid, 
        base_features, 
        cyclical_features, 
        action_space,
        lookups,  # <-- REFACTORED: Single source of truth
        do_cyclical_norm=True, 
        do_local_mean_z_score=True
    ):
        self.hpGrid = hpGrid
        self.base_features = base_features
        self.cyclical_features = cyclical_features
        self.action_space = action_space
        self.lookups = lookups
        self.do_cyclical_norm = do_cyclical_norm
        self.do_local_mean_z_score = do_local_mean_z_score

    def transform(self, pt_df, requested_features) -> np.ndarray:
        """Executes the bin feature pipeline and returns a 3D tensor."""
        timestamps = pt_df['timestamp'].values
        assert all(np.diff(timestamps) > 0), "Timestamps must be strictly increasing."
        
        n_timestamps = len(timestamps)
        n_bins = len(self.hpGrid.idx_lookup)
        
        features = self._pre_allocate_memory(n_timestamps, n_bins)
        
        self._calculate_ephemeris_features(features, timestamps, pt_df)
        
        if self._needs_history_features():
            history_feats = self._calculate_history(pt_df, requested_features)
            features.update(history_feats)
            
        if self.do_cyclical_norm:
            self._apply_cyclical_norms(features)
            
        if self.do_local_mean_z_score:
            self._apply_relative_norms(features)
            
        return self._stack_and_rearrange(features, requested_features)


    def _pre_allocate_memory(self, n_timestamps, n_bins) -> dict:
        """Determines which features are requested and allocates np.empty arrays."""
        features = {}
        bf = self.base_features
        shape = (n_timestamps, n_bins)

        # Boolean Flags
        do_pointing_distance = "pointing_distance" in bf
        do_rel_ha = "rel_ha" in bf
        do_ha = "ha" in bf or do_rel_ha
        do_airmass = "airmass" in bf
        do_ra, do_dec = "ra" in bf, "dec" in bf
        do_az, do_el = "az" in bf, "el" in bf
        do_rel_moon_distance = "rel_moon_distance" in bf
        do_moon_dist = "moon_distance" in bf or do_rel_moon_distance
        do_delta_az, do_delta_el = "delta_az" in bf, "delta_el" in bf
        do_coords = do_ra or do_dec or do_az or do_el or self.do_local_mean_z_score or do_delta_az or do_delta_el or do_ha

        # Memory Allocation
        if do_ha: features['ha'] = np.empty(shape, dtype=np.float32)
        if do_rel_ha: features['rel_ha'] = np.empty(shape, dtype=np.float32)
        if do_airmass: features['airmass'] = np.empty(shape, dtype=np.float32)
        if do_moon_dist: features['moon_distance'] = np.empty(shape, dtype=np.float32)
        if do_rel_moon_distance: features['rel_moon_distance'] = np.empty(shape, dtype=np.float32)

        if do_ra or do_dec or (do_coords and not self.hpGrid.is_azel):
            features['ra'] = np.empty(shape, dtype=np.float32)
            features['dec'] = np.empty(shape, dtype=np.float32)

        if do_coords:
            features['az'] = np.empty(shape, dtype=np.float32)
            features['el'] = np.empty(shape, dtype=np.float32)

        if do_delta_az: features['delta_az'] = np.empty(shape, dtype=np.float32)
        if do_delta_el: features['delta_el'] = np.empty(shape, dtype=np.float32)

        if do_pointing_distance or do_delta_az or do_delta_el:
            features['pointing_distance'] = np.empty(shape, dtype=np.float32)

        return features

    def _calculate_ephemeris_features(self, features, timestamps, pt_df):
        """The core timestamp loop for coordinates and ephemeris."""
        lon, lat = self.hpGrid.lon, self.hpGrid.lat

        # Set up target coordinates for distance calculations
        if 'pointing_distance' in features or 'delta_az' in features:
            if self.hpGrid.is_azel:
                target_lons, target_lats = pt_df['az'].values, pt_df['el'].values
            else:
                target_lons, target_lats = pt_df['ra'].values, pt_df['dec'].values

        # Broadcast static coordinates instantly across all timestamps
        do_coords = any(k in features for k in ['az', 'el', 'ra', 'dec'])
        if do_coords or self.do_local_mean_z_score:
            if self.hpGrid.is_azel:
                if 'az' in features: features['az'][:] = lon
                if 'el' in features or self.do_local_mean_z_score: features['el'][:] = lat
            else:
                if 'ra' in features or self.do_local_mean_z_score: features['ra'][:] = lon
                if 'dec' in features or self.do_local_mean_z_score: features['dec'][:] = lat

        # Timestamp dependent ephemeris
        for i, time in tqdm(enumerate(timestamps), total=len(timestamps), desc='Calculating bin ephemeris'):
            if 'ha' in features: 
                features['ha'][i] = self.hpGrid.get_hour_angle(time=time)
            if 'airmass' in features: 
                features['airmass'][i] = self.hpGrid.get_airmass(time)
            if 'moon_distance' in features: 
                features['moon_distance'][i] = self.hpGrid.get_source_angular_separations('moon', time=time)
            if 'pointing_distance' in features: 
                features['pointing_distance'][i] = self.hpGrid.get_angular_separations(lon=target_lons[i], lat=target_lats[i])
            if 'delta_az' in features or 'delta_el' in features:
                features['delta_az'][i], features['delta_el'][i] = get_delta_az_el(lon, lat, target_lons[i], target_lats[i])
                
            # Coordinate Transformations
            if do_coords or self.do_local_mean_z_score:
                if self.hpGrid.is_azel:
                    if 'ra' in features or 'dec' in features: 
                        features['ra'][i], features['dec'][i] = ephemerides.topographic_to_equatorial(az=lon, el=lat, time=time)
                else:
                    if 'az' in features or 'el' in features or self.do_local_mean_z_score:
                        features['az'][i], features['el'][i] = ephemerides.equatorial_to_topographic(ra=lon, dec=lat, time=time)

    def _needs_history_features(self) -> bool:
        history_keywords = ["num_unvisited_fields", "num_incomplete_fields", "min_tiling"]
        return any(
            hk in base_feat 
            for base_feat in self.base_features 
            for hk in history_keywords
        )

    def _calculate_history(self, pt_df, requested_features) -> dict:
        """Acts as a bridge to the pure history function."""
        logger.info("Calculating history-based features...")
        return calculate_history_dependent_bin_features(
            pt_df=pt_df, 
            hpGrid=self.hpGrid, 
            lookups=self.lookups,  # <-- REFACTORED
            action_space=self.action_space,
            requested_features=requested_features
        )

    def _apply_cyclical_norms(self, features):
        """Applies cos/sin expansions to cyclical features."""
        calc_feature_names = list(features.keys())
        for cyclical_feat in self.cyclical_features:
            for feat_name in calc_feature_names:
                is_exact = (feat_name == cyclical_feat)
                is_suffix = feat_name.endswith(f"_{cyclical_feat}")
                is_rel = feat_name.startswith("rel_")
                
                if (is_exact or is_suffix) and not is_rel:
                    features[f"{feat_name}_cos"] = np.cos(features[feat_name])
                    features[f"{feat_name}_sin"] = np.sin(features[feat_name])

    def _apply_relative_norms(self, features):
        """Subtracts the local valid mean from targeted features."""
        el_mask = features.get('el', np.ones_like(next(iter(features.values())))) > 0
        
        if 'moon_distance' in features and 'rel_moon_distance' in self.base_features:
            features['rel_moon_distance'] = self._get_relative_feature(features['moon_distance'], el_mask)
            
        if 'ha' in features and 'rel_ha' in self.base_features:
            features['rel_ha'] = self._get_relative_feature(features['ha'], el_mask)
            
        if self._needs_history_features():
            for filt in FILTER2IDX.keys():
                for s_feat_name in ['survey_num_unvisited_fields', 'survey_num_incomplete_fields', 'survey_min_tiling']:
                    raw_key = f"{s_feat_name}_{filt}"
                    if raw_key in features:
                        valid_cols = np.where(el_mask, features[raw_key], np.nan)
                        features[f"rel_{raw_key}"] = self._get_relative_feature(valid_cols, el_mask)

    def _stack_and_rearrange(self, features, requested_features) -> np.ndarray:
        """Pops requested arrays, validates them, and reshapes via einops."""
        final_arrays = []
        for key in requested_features:
            if key in features:
                arr = features.pop(key)
                assert not np.isnan(arr).any(), f"NaN values found in calculated feature {key}"
                final_arrays.append(arr)
            else:
                raise ValueError(f"Requested feature '{key}' was not calculated by the pipeline.")
                
        assert len(final_arrays) == len(requested_features)
        
        bin_states = np.array(final_arrays)
        return rearrange(bin_states, 'nfeats nrows nbins -> nrows nbins nfeats')

    @staticmethod
    def _get_relative_feature(feat_arr, el_mask):
        valid_cols = np.where(el_mask, feat_arr, np.nan)
        return feat_arr - np.nanmean(valid_cols, axis=-1, keepdims=True)
    
def calc_relative_survey_progress_features(feature_dict, el_mask):
    for filt in FILTER2IDX.keys():
        for s_feat_name in ['survey_num_unvisited_fields', 'survey_num_incomplete_fields', 'survey_min_tiling']:
            raw_key = f"{s_feat_name}_{filt}"
            if raw_key in feature_dict:
                valid_cols = np.where(el_mask, feature_dict[raw_key], np.nan)
                feature_dict[f"rel_{raw_key}"] = get_relative_feature(valid_cols, el_mask)
    return feature_dict

def get_relative_feature(feat_arr, el_mask):
    valid_cols = np.where(el_mask, feat_arr, np.nan)
    return feat_arr - np.nanmean(valid_cols, axis=-1, keepdims=True)

def get_delta_az_el(bin_azs, bin_els, target_az, target_el):
    azs = (bin_azs - target_az + np.pi) % (2 * np.pi) - np.pi
    els = bin_els - target_el
    return azs, els

def calculate_history_dependent_bin_features(pt_df, hpGrid, lookups, action_space, requested_features):
    # <-- REFACTORED: Now accepts `lookups` directly
    n_bins = len(hpGrid.idx_lookup)
    arr_shape = (len(pt_df), n_bins)
    
    # REFACTORED: field_ids is just the indices of the numpy array
    field_ids = np.arange(len(lookups.target_fid_counts))
    nfields, nfilters = len(field_ids), len(FILTER2IDX)
    idx2filter = {v: k for k, v in FILTER2IDX.items()}
    sentinel_val = AZEL_BIN_FEAT_SENTINEL if hpGrid.is_azel else RADEC_BIN_FEAT_SENTINEL

    do_filt = 'filter' in action_space
    is_azel = hpGrid.is_azel
    
    # State Assignment Helper
    def assign_state(mask_n, mask_s, count_n, count_s, act_n_msk, act_s_msk, key_n, key_s):
        res_n, res_s = np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
        np.divide(np.bincount(v_bins, weights=mask_n, minlength=n_bins), count_n, out=res_n, where=act_n_msk)
        np.divide(np.bincount(v_bins, weights=mask_s, minlength=n_bins), count_s, out=res_s, where=act_s_msk)
        res_n[~act_n_msk] = sentinel_val; res_s[~act_s_msk] = sentinel_val
        if key_n in historic_features: historic_features[key_n][global_idx] = res_n
        if key_s in historic_features: historic_features[key_s][global_idx] = res_s

    # ---------------------------------------------------------
    # 1. STRICT MEMORY ALLOCATION
    # ---------------------------------------------------------
    base_keys = ['night_num_unvisited_fields', 'night_num_incomplete_fields', 'night_min_tiling',
                 'survey_num_unvisited_fields', 'survey_num_incomplete_fields', 'survey_min_tiling']
    
    # Get required features
    matched_keys = [
        req_key for req_key in requested_features 
        if any(base_key in req_key for base_key in base_keys)
    ]

    logger.debug(f"Requested history-based features: {matched_keys}")
    
    historic_features = {}
    for key in matched_keys:
        historic_features[key] = np.full(arr_shape, sentinel_val, dtype=np.float32)
    
    # ---------------------------------------------------------
    # 2. SURVEY-WIDE SETUP
    # ---------------------------------------------------------
    
    # REFACTORED: Direct numpy slicing instead of slow list comprehensions
    ra_arr = lookups.fid2radec[:, 0]
    dec_arr = lookups.fid2radec[:, 1]
    
    # REFACTORED: Direct numpy array assignment
    if do_filt:
        max_s_f_vis_all = lookups.target_fidfilt_counts.astype(np.int32)
    else:
        max_s_vis_all = lookups.target_fid_counts.astype(np.int32)

    pt_df['filt_idx'] = pt_df['filter'].map(FILTER2IDX).fillna(ZENITH_FILTER_IDX).astype(np.int32)
    if pt_df['filt_idx'].isna().any():
        bad_filters = pt_df.loc[pt_df['filt_idx'].isna(), 'filter'].unique()
        logger.warning(f"Found {pt_df['filt_idx'].isna().sum()} NaNs in 'filt_idx'.")
        logger.warning(f"Unmapped filter strings causing NaNs: {bad_filters}")
        logger.warning(f"Current FILTER2IDX keys: {list(FILTER2IDX.keys())}")
    
    # STATIC RADEC CACHE (Computed once if static coords)
    if not is_azel:
        bins_raw = hpGrid.ang2idx(lon=ra_arr, lat=dec_arr)
        bins_static = np.array([b if b is not None else ZENITH_BIN_NUM for b in bins_raw], dtype=np.int32)
        v_mask_static = bins_static != ZENITH_BIN_NUM
        v_bins_static = bins_static[v_mask_static]

        if do_filt:
            bc_s_f_static = np.zeros((n_bins, nfilters), dtype=np.float64)
            for f in range(nfilters):
                bc_s_f_static[:, f] = np.bincount(v_bins_static, weights=(max_s_f_vis_all[v_mask_static, f] > 0), minlength=n_bins)
            act_s_f_static = bc_s_f_static > 0
        else:
            bc_s_static = np.bincount(v_bins_static, weights=(max_s_vis_all[v_mask_static] > 0), minlength=n_bins)
            act_s_static = bc_s_static > 0

    # ---------------------------------------------------------
    # 3. NIGHT LOOP
    # ---------------------------------------------------------
    cache_time = -1e9
    v_bins, v_mask = None, None
    bc_s, bc_n, act_s, act_n = None, None, None, None
    bc_s_f, bc_n_f, act_s_f, act_n_f = None, None, None, None
    global_idx = 0

    for night, group in tqdm(pt_df.groupby('night'), desc=f'Calculating {"AzEl" if is_azel else "RaDec"} History'):
        # A. Initialize Night Counters
        
        # REFACTORED: Accessing history from the lookups object
        if do_filt:
            cur_s_f_vis = lookups.night2fidfilt_visit_hist[night].copy()
            cur_n_f_vis = np.zeros((nfields, nfilters), dtype=np.int32)
        else:
            cur_s_vis = lookups.night2fid_visit_hist[night].copy().astype(np.int32)
            cur_n_vis = np.zeros(nfields, dtype=np.int32)
            
        step_fids = group['field_id'].to_numpy(dtype=np.int32)
        step_filts = group['filt_idx'].to_numpy(dtype=np.int32)
        step_times = group['timestamp'].to_numpy(dtype=np.int32)

        # B. Safely Calculate Target Max Arrays (Filtering out -1)
        valid_night = group['object'] != 'zenith'
        n_fids_raw = group['field_id'][valid_night].to_numpy(dtype=np.int32)
        valid_fids = n_fids_raw != ZENITH_FIELD_ID
        map_n_fids = field_ids[n_fids_raw[valid_fids]]
        valid_map = map_n_fids != ZENITH_FIELD_ID
        final_fids = map_n_fids[valid_map]
        
        if do_filt:
            n_filts_raw = group['filt_idx'][valid_night].to_numpy(dtype=np.int32)
            aligned_filts = n_filts_raw[valid_fids][valid_map]
            valid_filt = aligned_filts != ZENITH_FILTER_IDX
            
            max_n_f_vis = np.zeros((nfields, nfilters), dtype=np.int32)
            np.add.at(max_n_f_vis, (final_fids[valid_filt], aligned_filts[valid_filt]), 1)
            max_s_f_vis = np.maximum(max_n_f_vis, max_s_f_vis_all)
        else:
            max_n_vis = np.bincount(final_fids, minlength=nfields)
            max_s_vis = np.maximum(max_n_vis, max_s_vis_all)

        # C. If RaDec, inject static variables for the night
        if not is_azel:
            v_bins, v_mask = v_bins_static, v_mask_static
            
            if do_filt:
                bc_s_f, act_s_f = bc_s_f_static, act_s_f_static
                bc_n_f = np.zeros((n_bins, nfilters), dtype=np.float64)
                for f in range(nfilters):
                    bc_n_f[:, f] = np.bincount(v_bins, weights=(max_n_f_vis[v_mask, f] > 0), minlength=n_bins)
                act_n_f = bc_n_f > 0
            else:
                bc_s, act_s = bc_s_static, act_s_static
                bc_n = np.bincount(v_bins, weights=(max_n_vis[v_mask] > 0), minlength=n_bins)
                act_n = bc_n > 0
                
        # ---------------------------------------------------------
        # 4. TIMESTEP LOOP
        # ---------------------------------------------------------
        for timestamp, obs_fid, obs_filt in zip(step_times, step_fids, step_filts):
            
            # I. Update Tracking Counters
            if obs_fid != ZENITH_FIELD_ID:
                if do_filt:
                    cur_s_f_vis[obs_fid, obs_filt] += 1
                    cur_n_f_vis[obs_fid, obs_filt] += 1
                else:
                    cur_s_vis[obs_fid] += 1
                    cur_n_vis[obs_fid] += 1
 
            # II. If AzEl, do 5-minute dynamic cache updates
            if is_azel and abs(timestamp - cache_time) > 300:
                az, el = ephemerides.equatorial_to_topographic(ra_arr, dec_arr, time=timestamp)
                bins = np.array([b if b is not None else ZENITH_BIN_NUM for b in hpGrid.ang2idx(lon=az, lat=el)], dtype=np.int32)
                v_mask = (el > 0) & (bins != ZENITH_BIN_NUM)
                v_bins = bins[v_mask]
                
                if do_filt:
                    bc_s_f, bc_n_f = np.zeros((n_bins, nfilters)), np.zeros((n_bins, nfilters))
                    for f in range(nfilters):
                        bc_s_f[:, f] = np.bincount(v_bins, weights=(max_s_f_vis[v_mask, f] > 0), minlength=n_bins)
                        bc_n_f[:, f] = np.bincount(v_bins, weights=(max_n_f_vis[v_mask, f] > 0), minlength=n_bins)
                    act_s_f, act_n_f = bc_s_f > 0, bc_n_f > 0
                else:    
                    bc_s = np.bincount(v_bins, weights=(max_s_vis[v_mask] > 0), minlength=n_bins)
                    bc_n = np.bincount(v_bins, weights=(max_n_vis[v_mask] > 0), minlength=n_bins)
                    act_s, act_n = bc_s > 0, bc_n > 0
                    
                cache_time = timestamp
            
            # Execute 2D States (If Filter Space Active)
            if do_filt:
                v_s_f_vis, v_n_f_vis = cur_s_f_vis[v_mask], cur_n_f_vis[v_mask]
                v_max_s_f, v_max_n_f = max_s_f_vis[v_mask], max_n_f_vis[v_mask]
                in_s_f_plan, in_n_f_plan = v_max_s_f > 0, v_max_n_f > 0
                
                # Initialize with np.inf to prevent masking highly over-visited fields
                s_f_mins = np.full((n_bins, nfilters), np.inf, dtype=np.float32)
                n_f_mins = np.full((n_bins, nfilters), np.inf, dtype=np.float32)
                s_f_til = np.full_like(v_s_f_vis, np.inf, dtype=np.float32)
                n_f_til = np.full_like(v_n_f_vis, np.inf, dtype=np.float32)
                
                np.divide(v_s_f_vis, v_max_s_f, out=s_f_til, where=in_s_f_plan)
                np.divide(v_n_f_vis, v_max_n_f, out=n_f_til, where=in_n_f_plan)

                for f, filt_name in idx2filter.items():
                    assign_state((v_n_f_vis[:, f] == 0) & in_n_f_plan[:, f], (v_s_f_vis[:, f] == 0) & in_s_f_plan[:, f], 
                                 bc_n_f[:, f], bc_s_f[:, f], act_n_f[:, f], act_s_f[:, f], f'night_num_unvisited_fields_{filt_name}', f'survey_num_unvisited_fields_{filt_name}')
                    assign_state((v_n_f_vis[:, f] < v_max_n_f[:, f]) & in_n_f_plan[:, f], (v_s_f_vis[:, f] < v_max_s_f[:, f]) & in_s_f_plan[:, f],
                                 bc_n_f[:, f], bc_s_f[:, f], act_n_f[:, f], act_s_f[:, f], f'night_num_incomplete_fields_{filt_name}', f'survey_num_incomplete_fields_{filt_name}')
                    
                    np.minimum.at(s_f_mins[:, f], v_bins, s_f_til[:, f])
                    np.minimum.at(n_f_mins[:, f], v_bins, n_f_til[:, f])
                    
                    # Apply sentinels exclusively to untouched bins (np.inf)
                    s_f_mins[~act_s_f[:, f] | np.isinf(s_f_mins[:, f]), f] = sentinel_val
                    n_f_mins[~act_n_f[:, f] | np.isinf(n_f_mins[:, f]), f] = sentinel_val
                    
                    # Cap over-visited fields safely at 1.0 without destroying sentinels
                    s_f_mins[:, f] = np.minimum(s_f_mins[:, f], 1.0)
                    n_f_mins[:, f] = np.minimum(n_f_mins[:, f], 1.0)
                    
                    if (sk := f'survey_min_tiling_{filt_name}') in historic_features: historic_features[sk][global_idx] = s_f_mins[:, f]
                    if (nk := f'night_min_tiling_{filt_name}') in historic_features: historic_features[nk][global_idx] = n_f_mins[:, f]
            else:
                # IV. Execute 1D States (No per-filter tracking, just overall visit counts)

                # Get valid visit/max_visit arrays for visited bins
                v_s_vis, v_n_vis = cur_s_vis[v_mask], cur_n_vis[v_mask]
                v_max_s, v_max_n = max_s_vis[v_mask], max_n_vis[v_mask]
                in_s_plan, in_n_plan = v_max_s > 0, v_max_n > 0

                # NUM UNVISITED/INCOMPLETE (normalized by number of fields in bin) 
                assign_state((v_n_vis == 0) & in_n_plan, (v_s_vis == 0) & in_s_plan, bc_n, bc_s, act_n, act_s, 'night_num_unvisited_fields', 'survey_num_unvisited_fields')
                assign_state((v_n_vis < v_max_n) & in_n_plan, (v_s_vis < v_max_s) & in_s_plan, bc_n, bc_s, act_n, act_s, 'night_num_incomplete_fields', 'survey_num_incomplete_fields')

                # MIN TILING
                s_til, n_til = np.full_like(v_s_vis, np.inf, dtype=np.float32), np.full_like(v_n_vis, np.inf, dtype=np.float32)
                np.divide(v_s_vis.astype(np.float32), v_max_s.astype(np.float32), out=s_til, where=in_s_plan)
                np.divide(v_n_vis.astype(np.float32), v_max_n.astype(np.float32), out=n_til, where=in_n_plan)
                
                s_mins, n_mins = np.full(n_bins, np.inf, dtype=np.float32), np.full(n_bins, np.inf, dtype=np.float32)
                np.minimum.at(s_mins, v_bins, s_til)
                np.minimum.at(n_mins, v_bins, n_til)
                
                # Apply sentinels exclusively to untouched bins (np.inf)
                s_mins[~act_s | np.isinf(s_mins)] = sentinel_val
                n_mins[~act_n | np.isinf(n_mins)] = sentinel_val
                
                # Cap over-visited fields safely at 1.0 without destroying sentinels
                s_mins = np.minimum(s_mins, 1.0)
                n_mins = np.minimum(n_mins, 1.0)
                
                if 'survey_min_tiling' in historic_features: historic_features['survey_min_tiling'][global_idx] = s_mins
                if 'night_min_tiling' in historic_features: historic_features['night_min_tiling'][global_idx] = n_mins

            global_idx += 1
            
    _validate_history_dependent_features(do_filt=do_filt, idx2filter=idx2filter, calculated_features=historic_features)
    logger.debug(f"Historic features generated: {list(historic_features.keys())}")
    
    return historic_features

def _validate_history_dependent_features(do_filt, idx2filter, calculated_features):
    # Build a list of feature groupings to validate (both survey and night levels)
    check_groups = []
    for scope in ['survey', 'night']:
        check_groups.append({
            'unv': f"{scope}_num_unvisited_fields",
            'inc': f"{scope}_num_incomplete_fields",
            'til': f"{scope}_min_tiling",
            'name': f"{scope} (base)"
        })
        if do_filt:
            for filt_name in idx2filter.values():
                check_groups.append({
                    'unv': f"{scope}_num_unvisited_fields_{filt_name}",
                    'inc': f"{scope}_num_incomplete_fields_{filt_name}",
                    'til': f"{scope}_min_tiling_{filt_name}",
                    'name': f"{scope} ({filt_name})"
                })

    for grp in check_groups:
        unv_key, inc_key, til_key = grp['unv'], grp['inc'], grp['til']
        
        # Only check if these features were actually requested and generated
        if all(k in calculated_features for k in [unv_key, inc_key, til_key]):
            unv = calculated_features[unv_key]
            inc = calculated_features[inc_key]
            til = calculated_features[til_key]
            
            # Identify valid entries (ignoring the -1.0 or -0.1 sentinels for inactive bins)
            v_unv, v_inc, v_til = (unv >= 0.0), (inc >= 0.0), (til >= 0.0)
            
            # 1. Bounds Check
            if np.any(unv[v_unv] > 1.0):
                bad_idx = np.where(v_unv & (unv > 1.0))
                ts, bn = bad_idx[0][0], bad_idx[1][0]
                raise RuntimeError(f"FATAL BOUNDS: {unv_key} > 1.0 at global_idx {ts}, bin {bn}. Val: {unv[ts, bn]}")
            
            if np.any(inc[v_inc] > 1.0):
                bad_idx = np.where(v_inc & (inc > 1.0))
                ts, bn = bad_idx[0][0], bad_idx[1][0]
                raise RuntimeError(f"FATAL BOUNDS: {inc_key} > 1.0 at global_idx {ts}, bin {bn}. Val: {inc[ts, bn]}")

            # 2. Subset Rule: Unvisited MUST be <= Incomplete
            both_valid = v_unv & v_inc
            subset_violation = both_valid & (unv > (inc + 1e-5))
            if np.any(subset_violation):
                bad_idx = np.where(subset_violation)
                ts, bn = bad_idx[0][0], bad_idx[1][0]
                raise RuntimeError(
                    f"FATAL LOGIC LEAK: {grp['name']} unvisited > incomplete at global_idx {ts}, bin {bn}.\n"
                    f"Unvisited: {unv[ts, bn]} | Incomplete: {inc[ts, bn]}"
                )

            # 3. Tiling Floor: If unvisited > 0, min_tiling MUST be 0.0
            both_valid_til = v_unv & v_til
            has_unvisited = unv > 1e-5
            tiling_violation = both_valid_til & has_unvisited & (til > 1e-5)
            if np.any(tiling_violation):
                bad_idx = np.where(tiling_violation)
                ts, bn = bad_idx[0][0], bad_idx[1][0]
                raise RuntimeError(
                    f"FATAL LOGIC LEAK: {grp['name']} has unvisited fields, but min_tiling > 0 at global_idx {ts}, bin {bn}.\n"
                    f"Unvisited: {unv[ts, bn]} | Min Tiling: {til[ts, bn]}"
                )