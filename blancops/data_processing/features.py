import pandas as pd
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.utils.sys_utils import get_workspace_dir
import numpy as np
from datetime import timezone, timedelta
import ephem
from astropy.time import Time
import torch

import fitsio
from pathlib import Path
from tqdm import tqdm

from blancops.math import units
from blancops.ephemerides import ephemerides
from blancops.data_processing.constants import *

import warnings
import logging
logger = logging.getLogger(__name__)


# Filter wavelengths (nm) according to obztak https://github.com/kadrlica/obztak/blob/c28fab23b09bcff1cf46746eae4ec7e40aeb7f7a/obztak/seeing.py#L22
FILTER2WAVE = {
    'u': 380,
    'g': 480,
    'r': 640,
    'i': 780,
    'z': 920,
    'Y': 990
}

FILTERWAVENORM = 1000.
FILTER2IDX = {k: i for i, k in enumerate(FILTER2WAVE.keys())}
IDX2WAVE = {i: FILTER2WAVE[k] for i, k in enumerate(FILTER2WAVE.keys())}
NUM_FILTERS = len(FILTER2IDX)

def get_nautical_twilight(timestamp, event_type='set', horizon='-10'):
    obs = ephemerides.blanco_observer(time=timestamp)
    obs.horizon = horizon
    sun = ephem.Sun()
    
    if event_type == 'rise':
        ephem_date = obs.next_rising(sun).datetime()
    elif event_type == 'set':
        ephem_date = obs.previous_setting(sun).datetime()
    else:
        raise NotImplementedError

    dt_utc = ephem_date.replace(tzinfo=timezone.utc)
    return dt_utc.timestamp()

def get_sun_rise_and_set_times(df):
    rise_times = df.groupby('night').apply(get_nautical_twilight, event_type='rise').values
    set_times = df.groupby('night').apply(get_nautical_twilight, event_type='set').values
    return rise_times, set_times
    
def get_sun_rise_and_set_azel(df):
    rise_times, set_times = get_sun_rise_and_set_times(df)
    rise_azels = np.empty(shape=(len(set_times), 2))
    set_azels = np.empty(shape=(len(set_times), 2))
    
    for i, time in enumerate(rise_times):
        ra, dec = ephemerides.get_source_ra_dec('sun', time=time)
        sun_az, sun_el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=time)
        rise_azels[i] = np.array([sun_az, sun_el])
    for i, time in enumerate(set_times):
        ra, dec = ephemerides.get_source_ra_dec('sun', time=time)
        sun_az, sun_el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=time)
        set_azels[i] = np.array([sun_az, sun_el])

    return rise_azels, set_azels

def get_inst_teff_rate(df, remove_large_time_diffs, next_state_idxs):
    if remove_large_time_diffs:
        next_state_df = df.iloc[next_state_idxs]
        current_state_df = df.iloc[next_state_idxs-1]
        t_diff = next_state_df['timestamp'].values - current_state_df['timestamp'].values
        teff_no_zen = next_state_df[['teff']].values[:, 0]
    else:
        t_diff = df.sort_values(['night', 'timestamp']).groupby('night')['timestamp'].diff().dropna().to_numpy()
        teff_no_zen = df[df['object'] != 'zenith'][['teff']].values[:, 0]

    teff_inst_rate = teff_no_zen / t_diff
    min_rate = np.min(teff_inst_rate)
    max_rate = np.max(teff_inst_rate)
    rewards = (teff_inst_rate - min_rate)/max_rate
    return rewards

def normalize_timestamp(timestamp, sunset_timestamp, sunrise_timestamp):
    return (timestamp - sunset_timestamp) / (sunrise_timestamp - sunset_timestamp)

def get_zenith_features(original_df):
    """
    Constructs dataframe with zenith features for each night in the original_df.
    Assumes zenith starts 10 seconds before the first observation.
    """
    zenith_datetimes = (original_df.groupby('night').head(1).datetime - pd.Timedelta(seconds=10)).values
    zenith_timestamps = zenith_datetimes.astype(np.int64) // 10 ** 9
    zenith_rows = []
    nights = original_df.night.unique()
    for i_row, time in tqdm(enumerate(zenith_timestamps), total=len(zenith_timestamps), desc='Calculating zenith states'):
        row_dict = {}
        row_dict['timestamp'] = time
        row_dict['night'] = nights[i_row]
        row_dict['datetime'] = zenith_datetimes[i_row]
        blanco = ephemerides.blanco_observer(time=time)
        row_dict['ra'], row_dict['dec'] = np.array(blanco.radec_of('0',  '90')) / units.deg
        zenith_rows.append(row_dict)

    zenith_df = pd.DataFrame(zenith_rows)
    zenith_df['az'] = 0
    zenith_df['el'] = 90
    zenith_df['airmass'] = 1
    zenith_df['zd'] = 0
    zenith_df['ha'] = 0
    zenith_df['object'] = 'zenith'
    zenith_df['field_id'] = -1
    zenith_df['filter'] = 0
    zenith_df['datetime'] = pd.to_datetime(zenith_df['datetime'], utc=True)
    zenith_df['night'] = pd.to_datetime(zenith_df['night'], utc=True)

    return zenith_df

def calculate_and_add_global_features(df, field2name, hpGrid, 
                      base_global_feature_names, cyclical_feature_names, do_cyclical_norm):
    """Processes and filters the dataframe to return a new dataframe with added columns for current global state features"""
    # Sort df by timestamp
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 1. Insert zenith states in dataframe
    zenith_df = get_zenith_features(original_df=df)
    df = pd.concat([df, zenith_df], ignore_index=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 2. Get coords in radians
    df.loc[:, ['ra', 'dec', 'az', 'zd', 'ha']] *= units.deg
    df['el'] = np.pi/2 - df['zd'].values

    # 2. Calculate LST for the whole column at once
    if 'lst' in base_global_feature_names:
        t_arr = Time(df['datetime'].values, format='datetime64', scale='utc')
        lst_obj = t_arr.sidereal_time('apparent', longitude="-70:48:23.49")  # Blanco longitude
        df['lst'] = lst_obj.radian
        df['lst_hours'] = lst_obj.hour # for debugging

    timestamps = df['timestamp'].values
    indices = df.index
    ra_arr = df['ra'].values
    dec_arr = df['dec'].values
    filt_arr = df['filter'].values

    # 3. Get time dependent features (sun and moon pos)
    for idx, time in tqdm(zip(indices, timestamps), total=len(timestamps), desc='Calculating sun and moon ra/dec and az/el'):
        sun_ra, sun_dec = ephemerides.get_source_ra_dec('sun', time=time)
        df.loc[idx, ['sun_ra', 'sun_dec']] = sun_ra, sun_dec
        df.loc[idx, ['sun_az', 'sun_el']] = ephemerides.equatorial_to_topographic(ra=sun_ra, dec=sun_dec, time=time)

        moon_ra, moon_dec = ephemerides.get_source_ra_dec('moon', time=time)
        df.loc[idx, ['moon_ra', 'moon_dec']] = moon_ra, moon_dec
        df.loc[idx, ['moon_az', 'moon_el']] = ephemerides.equatorial_to_topographic(ra=moon_ra, dec=moon_dec, time=time)

    # Use first and last observation in night of offline dataset as time start and end
    def normalize_times(time_series):
        sunset = get_nautical_twilight(time_series.median(), event_type='set')
        sunrise = get_nautical_twilight(time_series.median(), event_type='rise')
        total_time = sunrise - sunset

        time_series = (time_series - sunset) / total_time
        assert all(time_series.values > 0) and all(time_series.values < 1), "Time fractions should be between 0 and 1"
        return time_series
    # Using nautical twilight for time start and end
    df['time_fraction_since_start'] = df.groupby('night')['timestamp'].transform(normalize_times)
    assert all(df['time_fraction_since_start'].values > 0) and all(df['time_fraction_since_start'].values < 1), "Time fractions should be between 0 and 1"  

    # 6. Add bin and field id columns to dataframe
    df['field_id'] = df['object'].map({v: k for k, v in field2name.items()})
    if hpGrid is not None:
        if hpGrid.is_azel:
            lon = df['az']
            lat = df['el']
        else:
            lon = df['ra']
            lat = df['dec']
        df['bin'] = hpGrid.ang2idx(lon=lon, lat=lat)
        # df['bin'][df['object'] == 'zenith'] = -1  # assign zenith bin of -1
        df.loc[df['object'] == 'zenith', "bin"] = -1
        df.loc[df['object'] == 'zenith', "field_id"] = -1 # Need to re-assign zenith field_id bc df['object'].map(...) above will assign zenith the field_id of the field with object name 'zenith', but this field is mis-labelled and not actually the zenith field. #TODO should fix this in field2name

    # Add other feature columns for those not present in dataframe
    for feat_name in base_global_feature_names:
        if feat_name not in df.columns:
            if feat_name == 'filter_wave':
                df['filter_wave'] = df['filter'].map(FILTER2WAVE)
                df['filter_wave'] = df['filter_wave'].fillna(0.) / FILTERWAVENORM # zenith "filter" set to 0, then normalize
            elif feat_name == 'ra_vel':
                # Calculate both ra_vel and dec_vel
                zenith_idxs = df.index[df['object'] == 'zenith']
                delta_ts = df['timestamp'].diff().values
                delta_ts[0] = 1 # Change nan to 0
                delta_ts -= df['exptime'].values
                delta_ts[zenith_idxs] = 1 # First pointing for each night is zenith
                print('delta_ts is 0 at ', np.where(delta_ts == 0))

                delta_ras = df['ra'].diff().values
                delta_decs = df['dec'].diff().values
                delta_ras[zenith_idxs] = 0 # Assume zenith has 0 difference
                delta_decs[zenith_idxs] = 0

                dRAdt = delta_ras * np.cos(df['dec'].values)/ delta_ts
                dDECdt = delta_decs / delta_ts
                df['ra_vel'] = dRAdt
                df['dec_vel'] = dDECdt
            elif feat_name == 'sky_brightness':
                df['sky_brightness'] = estimate_sky_brightness(time=timestamps, ra=ra_arr, dec=dec_arr, band=filt_arr)
            else:
                raise NotImplementedError(f"Feature {feat_name} not found in dataframe columns. Check spelling. Or, this feature is not yet implemented.")

    # Normalize periodic features here and add as df cols
    if do_cyclical_norm:
        for feat_name in base_global_feature_names:
            if any(string in feat_name and 'frac' not in feat_name and 'bin' not in feat_name for string in cyclical_feature_names):
                logger.info(f'Applying cyclical norm to {feat_name}')
                df[f'{feat_name}_cos'] = np.cos(df[feat_name].values)
                df[f'{feat_name}_sin'] = np.sin(df[feat_name].values)

    # Ensure all data are 32-bit precision before training
    for bin_str, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
        cols = df.select_dtypes(include=[bin_str]).columns
        df[cols] = df[cols].astype(np_bit)
    return df

def calculate_and_add_bin_features(pt_df, datetimes, hpGrid, base_bin_feature_names, prenorm_bin_feature_names, 
                                   bin_feature_names, cyclical_feature_names, do_cyclical_norm, night2fieldvisits,
                                   night2filtervisithistory, fieldfilter2maxvisits, field2radec, field2maxvisits, bin_space):
    """
    Calculate bin features dynamically based on requested feature names.
    
    This version:
    1. Only calculates features that are present in base_bin_feature_names
    2. Dynamically constructs the output arrays based on which features are requested
    """
    
    # Create empty arrays 
    timestamps = pt_df['timestamp'].values
    n_timestamps = len(timestamps)
    n_bins = len(hpGrid.idx_lookup)
    
    # History based features
    history_based_features = [
        "num_unvisited_fields",
        "num_incomplete_fields",
        "min_tiling",
    ]
    do_history_based_features = any(hist_feat in base_feat for base_feat in base_bin_feature_names for hist_feat in history_based_features)

    # Instantaenous features
    do_angular_distance_to_pointing = "angular_distance_to_pointing" in base_bin_feature_names
    do_ha = "ha" in base_bin_feature_names
    do_airmass = "airmass" in base_bin_feature_names
    do_moon_dist = "moon_distance" in base_bin_feature_names
    do_ra = "ra" in base_bin_feature_names
    do_dec = "dec" in base_bin_feature_names
    do_az = "az" in base_bin_feature_names
    do_el = "el" in base_bin_feature_names
    do_coords = do_ra or do_dec or do_az or do_el
    
    # Initialize arrays only for features we need
    calculated_features = {}
    
    # Pre-allocate
    if do_ha:
        calculated_features['ha'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)
        logger.debug(f"Calculating ha for {n_timestamps} timestamps and {n_bins} bins")
    if do_airmass:
        calculated_features['airmass'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)
        logger.debug(f"Calculating airmass for {n_timestamps} timestamps and {n_bins} bins")
    if do_moon_dist:
        calculated_features['moon_distance'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)
        logger.debug(f"Calculating moon distance for {n_timestamps} timestamps and {n_bins} bins")
    if do_ra or do_dec:
        calculated_features['ra'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)  # az or ra
        calculated_features['dec'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)  # az or ra
        logger.debug(f"Calculating ra/dec for {n_timestamps} timestamps and {n_bins} bins")
    if do_az or do_el:
        calculated_features['az'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)  # el or dec
        calculated_features['el'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)  # el or dec
        logger.debug(f"Calculating az/el for {n_timestamps} timestamps and {n_bins} bins")
    if do_angular_distance_to_pointing:
        calculated_features['angular_distance_to_pointing'] = np.empty(shape=(n_timestamps, n_bins), dtype=np.float32)
        if hpGrid.is_azel:
            target_lons = pt_df['az'].values
            target_lats = pt_df['el'].values
        else:
            target_lons = pt_df['ra'].values
            target_lats = pt_df['dec'].values
            
    # Calculate per-timestamp features
    lon, lat = hpGrid.lon, hpGrid.lat
    for i, time in tqdm(enumerate(timestamps), total=n_timestamps, desc='Calculating bin features for all healpix bins and timestamps'):
        if do_ha:
            calculated_features['ha'][i] = hpGrid.get_hour_angle(time=time)
        if do_airmass:
            calculated_features['airmass'][i] = hpGrid.get_airmass(time)
        if do_moon_dist:
            calculated_features['moon_distance'][i] = hpGrid.get_source_angular_separations('moon', time=time)
        if do_angular_distance_to_pointing:
            calculated_features['angular_distance_to_pointing'][i] = hpGrid.get_angular_separations(lon=target_lons[i], lat=target_lats[i])
        
        # Coordinate transformations
        if do_coords:
            if hpGrid.is_azel:
                if 'az' in calculated_features: calculated_features['az'][i] = lon
                if 'el' in calculated_features: calculated_features['el'][i] = lat
                # ONLY calculate RA/DEC if they were actually requested
                if do_ra or do_dec: 
                    calculated_features['ra'][i], calculated_features['dec'][i] = ephemerides.topographic_to_equatorial(az=lon, el=lat, time=time)
            else:
                if 'ra' in calculated_features: calculated_features['ra'][i] = lon
                if 'dec' in calculated_features: calculated_features['dec'][i] = lat
                if do_az or do_el:
                    calculated_features['az'][i], calculated_features['el'][i] = ephemerides.equatorial_to_topographic(ra=lon, dec=lat, time=time)

    # Calculate night-based features if needed
    if do_history_based_features:
        calculated_night_history_features = calculate_history_dependent_bin_features(pt_df=pt_df, hpGrid=hpGrid, field2radec=field2radec, 
                                                                                     night2visithistory=night2fieldvisits, night2filtervisithistory=night2filtervisithistory,
                                                                                     field2maxvisits=field2maxvisits, fieldfilter2maxvisits=fieldfilter2maxvisits, bin_space=bin_space,
                                                                                     requested_features=base_bin_feature_names)
        calculated_features = calculated_features | calculated_night_history_features
    
    # Dynamically stack features in the order they appear in base_bin_feature_names
    # First, determine the unique feature types and their order
    feature_order = []
    seen_features = set()
    for name in prenorm_bin_feature_names:
        parts = name.split('_')
        if len(parts) >= 3 and parts[0] == 'bin':
            feature_type = '_'.join(parts[2:])
            if feature_type not in seen_features:
                feature_order.append(feature_type)
                seen_features.add(feature_type)

    # Stack the features in order
    feature_arrays = []
    for feat in feature_order:
        if feat in calculated_features:
            feature_arrays.append(calculated_features[feat].astype(np.float32))
        else:
            raise ValueError(f"Feature {feat} is in prenorm_bin_feature_names but was not calculated. Check if it's included in base_bin_feature_names and if the calculation code is implemented.")
    
    # Stack and reshape to create the final feature matrix
    if feature_arrays:
        stacked = np.stack(feature_arrays, axis=2)  # shape: (n_timestamps, n_bins, n_features)
        bin_states = stacked.reshape(n_timestamps, -1)  # shape: (n_timestamps, n_bins * n_features)
    else:
        bin_states = np.empty((n_timestamps, 0))
    
    final_col_names = prenorm_bin_feature_names.copy()

    if do_cyclical_norm:
        cyclical_cols = []
        cyclical_names = []
        # Identify indices of columns that need cyclic norm
        for i, feat_name in enumerate(prenorm_bin_feature_names):
            if any(string in feat_name and 'frac' not in feat_name for string in cyclical_feature_names):
                cyclical_cols.append(i)
                cyclical_names.extend([f'{feat_name}_cos', f'{feat_name}_sin'])
        
        if cyclical_cols:
            base_cyclical_data = bin_states[:, cyclical_cols]
            
            # Pre-allocate array for the new cos/sin features
            new_data = np.empty((n_timestamps, len(cyclical_cols) * 2), dtype=np.float32)
            new_data[:, 0::2] = np.cos(base_cyclical_data)
            new_data[:, 1::2] = np.sin(base_cyclical_data)
            
            # Horizontally stack the new data
            bin_states = np.hstack([bin_states, new_data])
            final_col_names.extend(cyclical_names)

    sort_idx = np.argsort(timestamps)
    bin_states = bin_states[sort_idx]
    sorted_timestamps = timestamps[sort_idx]
    
    # Map datetimes efficiently using np datetime arrays
    sorted_nights = (datetimes.values[sort_idx] - np.timedelta64(12, 'h')).astype('datetime64[D]')

    bin_df = pd.DataFrame(data=bin_states, columns=final_col_names, copy=False)
    bin_df['night'] = sorted_nights
    bin_df['timestamp'] = sorted_timestamps
    
    # Make sure there are no missing columns
    missing_cols = set(bin_feature_names) - set(bin_df.columns)
    assert len(missing_cols) == 0, f'Features {missing_cols} do not exist in dataframe. These are not yet implemented in method self._get_bin_features()'
    
    return bin_df

def calculate_history_dependent_bin_features(pt_df, hpGrid, field2radec, night2visithistory, 
                                             night2filtervisithistory, field2maxvisits, 
                                             fieldfilter2maxvisits, bin_space, requested_features):
    n_bins = len(hpGrid.idx_lookup)
    arr_shape = (len(pt_df), n_bins)
    field_ids = np.array(list(field2maxvisits.keys()))
    nfields, nfilters = len(field_ids), len(FILTER2IDX)
    idx2filter = {v: k for k, v in FILTER2IDX.items()}
    
    do_filt = 'filter' in bin_space
    is_azel = hpGrid.is_azel

    # ---------------------------------------------------------
    # 1. STRICT MEMORY ALLOCATION
    # ---------------------------------------------------------
    calculated_features = {}
    base_keys = ['night_num_unvisited_fields', 'night_num_incomplete_fields', 'night_min_tiling',
                 'survey_num_unvisited_fields', 'survey_num_incomplete_fields', 'survey_min_tiling']
    
    for key in base_keys:
        if key in requested_features:
            calculated_features[key] = np.full(arr_shape, -0.1, dtype=np.float32) if 'min_tiling' in key else np.zeros(arr_shape, dtype=np.float32)
        if do_filt:
            for filt_name in FILTER2IDX.keys():
                filt_key = f"{key}_{filt_name}"
                if filt_key in requested_features:
                    calculated_features[filt_key] = np.full(arr_shape, -0.1, dtype=np.float32) if 'min_tiling' in key else np.zeros(arr_shape, dtype=np.float32)

    # ---------------------------------------------------------
    # 2. SURVEY-WIDE SETUP
    # ---------------------------------------------------------
    ra_arr = np.array([field2radec[fid][0] for fid in field_ids])
    dec_arr = np.array([field2radec[fid][1] for fid in field_ids])
    max_s_vis_all = np.array([field2maxvisits[fid] for fid in field_ids], dtype=np.int32)
    if do_filt: 
        max_s_f_vis_all = np.array([fieldfilter2maxvisits[fid] for fid in field_ids], dtype=np.int32)

    pt_df['filt_idx'] = pt_df['filter'].map(FILTER2IDX).fillna(-1)

    # STATIC RADEC CACHE (Computed once if static coords)
    if not is_azel:
        bins_raw = hpGrid.ang2idx(lon=ra_arr, lat=dec_arr)
        bins_static = np.array([b if b is not None else -1 for b in bins_raw], dtype=np.int32)
        v_mask_static = bins_static != -1
        v_bins_static = bins_static[v_mask_static]

        bc_s_static = np.bincount(v_bins_static, weights=(max_s_vis_all[v_mask_static] > 0), minlength=n_bins)
        act_s_static = bc_s_static > 0

        if do_filt:
            bc_s_f_static = np.zeros((n_bins, nfilters), dtype=np.float64)
            for f in range(nfilters):
                bc_s_f_static[:, f] = np.bincount(v_bins_static, weights=(max_s_f_vis_all[v_mask_static, f] > 0), minlength=n_bins)
            act_s_f_static = bc_s_f_static > 0

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
        cur_s_vis = night2visithistory[night][field_ids].copy().astype(np.int32)
        cur_n_vis = np.zeros(nfields, dtype=np.int32)
        if do_filt:
            cur_s_f_vis, cur_n_f_vis = night2filtervisithistory[night].copy(), np.zeros((nfields, nfilters), dtype=np.int32)
        
        step_fids = group['field_id'].fillna(-1).to_numpy(dtype=np.int32)
        step_filts = group['filt_idx'].to_numpy(dtype=np.int32)
        step_times = group['timestamp'].to_numpy(dtype=np.int32)

        # B. Safely Calculate Target Max Arrays (Filtering out -1)
        valid_night = group['object'] != 'zenith'
        n_fids_raw = group['field_id'][valid_night].fillna(-1).to_numpy(dtype=np.int32)
        valid_fids = n_fids_raw != -1
        map_n_fids = field_ids[n_fids_raw[valid_fids]]
        valid_map = map_n_fids != -1
        final_fids = map_n_fids[valid_map]
        
        max_n_vis = np.bincount(final_fids, minlength=nfields)
        max_s_vis = np.maximum(max_n_vis, max_s_vis_all)

        if do_filt:
            n_filts_raw = group['filt_idx'][valid_night].fillna(-1).to_numpy(dtype=np.int32)
            aligned_filts = n_filts_raw[valid_fids][valid_map]
            valid_filt = aligned_filts != -1
            
            max_n_f_vis = np.zeros((nfields, nfilters), dtype=np.int32)
            np.add.at(max_n_f_vis, (final_fids[valid_filt], aligned_filts[valid_filt]), 1)
            max_s_f_vis = np.maximum(max_n_f_vis, max_s_f_vis_all)

        # C. If RaDec, inject static variables for the night
        if not is_azel:
            v_bins, v_mask = v_bins_static, v_mask_static
            bc_s, act_s = bc_s_static, act_s_static
            bc_n = np.bincount(v_bins, weights=(max_n_vis[v_mask] > 0), minlength=n_bins)
            act_n = bc_n > 0
            
            if do_filt:
                bc_s_f, act_s_f = bc_s_f_static, act_s_f_static
                bc_n_f = np.zeros((n_bins, nfilters), dtype=np.float64)
                for f in range(nfilters):
                    bc_n_f[:, f] = np.bincount(v_bins, weights=(max_n_f_vis[v_mask, f] > 0), minlength=n_bins)
                act_n_f = bc_n_f > 0

        # ---------------------------------------------------------
        # 4. TIMESTEP LOOP
        # ---------------------------------------------------------
        for timestamp, obs_fid, obs_filt in zip(step_times, step_fids, step_filts):
            
            # I. Update Tracking Counters
            if obs_fid != -1 and (idx := field_ids[obs_fid]) != -1:
                cur_s_vis[idx] += 1; cur_n_vis[idx] += 1
                if do_filt and obs_filt != -1:
                    cur_s_f_vis[idx, obs_filt] += 1; cur_n_f_vis[idx, obs_filt] += 1

            # II. If AzEl, do 5-minute dynamic cache updates
            if is_azel and abs(timestamp - cache_time) > 300:
                az, el = ephemerides.equatorial_to_topographic(ra_arr, dec_arr, time=timestamp)
                bins = np.array([b if b is not None else -1 for b in hpGrid.ang2idx(lon=az, lat=el)], dtype=np.int32)
                v_mask = (el > 0) & (bins != -1)
                v_bins = bins[v_mask]
                
                bc_s = np.bincount(v_bins, weights=(max_s_vis[v_mask] > 0), minlength=n_bins)
                bc_n = np.bincount(v_bins, weights=(max_n_vis[v_mask] > 0), minlength=n_bins)
                act_s, act_n = bc_s > 0, bc_n > 0
                
                if do_filt:
                    bc_s_f, bc_n_f = np.zeros((n_bins, nfilters)), np.zeros((n_bins, nfilters))
                    for f in range(nfilters):
                        bc_s_f[:, f] = np.bincount(v_bins, weights=(max_s_f_vis[v_mask, f] > 0), minlength=n_bins)
                        bc_n_f[:, f] = np.bincount(v_bins, weights=(max_n_f_vis[v_mask, f] > 0), minlength=n_bins)
                    act_s_f, act_n_f = bc_s_f > 0, bc_n_f > 0
                    
                cache_time = timestamp

            # III. State Assignment Helper
            def assign_state(mask_n, mask_s, count_n, count_s, act_n_msk, act_s_msk, key_n, key_s):
                res_n, res_s = np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
                np.divide(np.bincount(v_bins, weights=mask_n, minlength=n_bins), count_n, out=res_n, where=act_n_msk)
                np.divide(np.bincount(v_bins, weights=mask_s, minlength=n_bins), count_s, out=res_s, where=act_s_msk)
                res_n[~act_n_msk] = -1.0; res_s[~act_s_msk] = -1.0
                if key_n in calculated_features: calculated_features[key_n][global_idx] = res_n
                if key_s in calculated_features: calculated_features[key_s][global_idx] = res_s

            # IV. Execute 1D States 
            v_s_vis, v_n_vis = cur_s_vis[v_mask], cur_n_vis[v_mask]
            v_max_s, v_max_n = max_s_vis[v_mask], max_n_vis[v_mask]
            in_s_plan, in_n_plan = v_max_s > 0, v_max_n > 0

            assign_state((v_n_vis == 0) & in_n_plan, (v_s_vis == 0) & in_s_plan, bc_n, bc_s, act_n, act_s, 'night_num_unvisited_fields', 'survey_num_unvisited_fields')
            assign_state((v_n_vis < v_max_n) & in_n_plan, (v_s_vis < v_max_s) & in_s_plan, bc_n, bc_s, act_n, act_s, 'night_num_incomplete_fields', 'survey_num_incomplete_fields')

            s_til, n_til = np.full_like(v_s_vis, 2.0, dtype=np.float32), np.full_like(v_n_vis, 2.0, dtype=np.float32)
            np.divide(v_s_vis, v_max_s, out=s_til, where=in_s_plan)
            np.divide(v_n_vis, v_max_n, out=n_til, where=in_n_plan)
            s_mins, n_mins = np.full(n_bins, 2.0, dtype=np.float32), np.full(n_bins, 2.0, dtype=np.float32)
            np.minimum.at(s_mins, v_bins, s_til); np.minimum.at(n_mins, v_bins, n_til)
            s_mins[~act_s | (s_mins > 1.0)] = -1.0; n_mins[~act_n | (n_mins > 1.0)] = -1.0
            
            if 'survey_min_tiling' in calculated_features: calculated_features['survey_min_tiling'][global_idx] = s_mins
            if 'night_min_tiling' in calculated_features: calculated_features['night_min_tiling'][global_idx] = n_mins

            # V. Execute 2D States (If Filter Space Active)
            if do_filt:
                v_s_f_vis, v_n_f_vis = cur_s_f_vis[v_mask], cur_n_f_vis[v_mask]
                v_max_s_f, v_max_n_f = max_s_f_vis[v_mask], max_n_f_vis[v_mask]
                in_s_f_plan, in_n_f_plan = v_max_s_f > 0, v_max_n_f > 0
                
                s_f_mins, n_f_mins = np.full((n_bins, nfilters), 2.0, dtype=np.float32), np.full((n_bins, nfilters), 2.0, dtype=np.float32)
                s_f_til, n_f_til = np.full_like(v_s_f_vis, 2.0, dtype=np.float32), np.full_like(v_n_f_vis, 2.0, dtype=np.float32)
                np.divide(v_s_f_vis, v_max_s_f, out=s_f_til, where=in_s_f_plan)
                np.divide(v_n_f_vis, v_max_n_f, out=n_f_til, where=in_n_f_plan)

                for f, filt_name in idx2filter.items():
                    assign_state((v_n_f_vis[:, f] == 0) & in_n_f_plan[:, f], (v_s_f_vis[:, f] == 0) & in_s_f_plan[:, f], 
                                 bc_n_f[:, f], bc_s_f[:, f], act_n_f[:, f], act_s_f[:, f], f'night_num_unvisited_fields_{filt_name}', f'survey_num_unvisited_fields_{filt_name}')
                    assign_state((v_n_f_vis[:, f] < v_max_n_f[:, f]) & in_n_f_plan[:, f], (v_s_f_vis[:, f] < v_max_s_f[:, f]) & in_s_f_plan[:, f],
                                 bc_n_f[:, f], bc_s_f[:, f], act_n_f[:, f], act_s_f[:, f], f'night_num_incomplete_fields_{filt_name}', f'survey_num_incomplete_fields_{filt_name}')
                    
                    np.minimum.at(s_f_mins[:, f], v_bins, s_f_til[:, f])
                    np.minimum.at(n_f_mins[:, f], v_bins, n_f_til[:, f])
                    s_f_mins[~act_s_f[:, f] | (s_f_mins[:, f] > 1.0), f] = -1.0
                    n_f_mins[~act_n_f[:, f] | (n_f_mins[:, f] > 1.0), f] = -1.0
                    
                    if (sk := f'survey_min_tiling_{filt_name}') in calculated_features: calculated_features[sk][global_idx] = s_f_mins[:, f]
                    if (nk := f'night_min_tiling_{filt_name}') in calculated_features: calculated_features[nk][global_idx] = n_f_mins[:, f]

            global_idx += 1

    _validate_history_dependent_features(do_filt=do_filt, idx2filter=idx2filter, calculated_features=calculated_features)
    
    return calculated_features

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

# def calculate_history_dependent_bin_features(pt_df, hpGrid, night2visithistory, night2filtervisithistory, 
#                                              fieldfilter2maxvisits, field2radec, field2maxvisits, bin_space,
#                                              base_bin_feature_names):
#     n_bins = len(hpGrid.idx_lookup)
#     arr_shape = (len(pt_df), n_bins)

#     history_based_features = ['num_unvisited_fields', 'num_incomplete_fields', 'min_tiling']
        
#     calculated_features = {}
    
#     # Only allocate memory if the feature is explicitly requested in your config
#     for key in history_based_features:
#         if key in base_bin_feature_names:
#             calculated_features[key] = -.1 * np.ones(arr_shape, dtype=np.float32) if 'min_tiling' in key else np.zeros(arr_shape, dtype=np.float32)
            
#     if hpGrid.is_azel:
#         warnings.filterwarnings("error", category=RuntimeWarning)
#         calculated_features = calculate_history_dependent_bin_features_azel(pt_df=pt_df, hpGrid=hpGrid, field2radec=field2radec, calculated_features=calculated_features, 
#                                                                             night2visithistory=night2visithistory, night2filtervisithistory=night2filtervisithistory,
#                                                                             field2maxvisits=field2maxvisits, fieldfilter2maxvisits=fieldfilter2maxvisits, bin_space=bin_space,
#                                                                             base_bin_feature_names)
#     else:
#         warnings.filterwarnings("error", category=RuntimeWarning)
#         calculated_features = calculate_history_dependent_bin_features_radec(pt_df=pt_df, hpGrid=hpGrid, field2radec=field2radec, calculated_features=calculated_features, 
#                                                                             night2visithistory=night2visithistory, night2filtervisithistory=night2filtervisithistory,
#                                                                             field2maxvisits=field2maxvisits, fieldfilter2maxvisits=fieldfilter2maxvisits, bin_space=bin_space,
#                                                                             base_bin_feature_names)
    
#     for key, arr in calculated_features.items():
#         if arr.min() < -.1 and arr.max() > 1.:
#             logger.debug(f"{key} is not between 0 and 1. Array max/min={arr.max()}/{arr.min()}. Check normalization factor.")

#     return calculated_features


def calculate_history_dependent_bin_features_radec(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, 
                                                   night2filtervisithistory, field2maxvisits, fieldfilter2maxvisits, bin_space,
                                                    base_bin_feature_names):
    n_bins = len(hpGrid.idx_lookup)
    field_ids = np.array(list(field2maxvisits.keys()))
    nfields = len(field_ids)
    nfilters = len(FILTER2IDX)
    idx2filter = {v: k for k, v in FILTER2IDX.items()}

    # Before looping over nights, get survey-wide visits
    # Get bin membership of all fields in survey
    ra_arr = np.array([field2radec[fid][0] for fid in field_ids])
    dec_arr = np.array([field2radec[fid][1] for fid in field_ids])
    bins_membership_arr = hpGrid.ang2idx(lon=ra_arr, lat=dec_arr) # Bin membership of each field ordered by field idx

    # Get max visits per field and number of fields per bin for entire survey
    max_s_visits_arr_all = np.array([field2maxvisits[fid] for fid in field_ids], dtype=np.int32) # visits per field
    in_survey_plan = max_s_visits_arr_all > 0 # mask fields not in survey (field2maxvisits should be built such that field ids only include fields in survey)

    # Get max filter visits per field (Shape: nfields x n_filters)
    max_s_filter_visits_arr_all = np.array([fieldfilter2maxvisits[fid] for fid in field_ids], dtype=np.int32)
    in_survey_filter_plan = max_s_filter_visits_arr_all > 0
    
    nfields_s = np.bincount(bins_membership_arr, weights=in_survey_plan, minlength=n_bins) # number of fields per bin
    active_bins_s = nfields_s > 0
    
    global_idx = 0
    pt_df['filt_idx'] = pt_df['filter'].map(FILTER2IDX).fillna(-1)

    night_groups = pt_df.groupby('night')

    # if False:
    if 'filter' in bin_space:
        # Precompute survey-wide filters for each filter
        nfields_s_filter = np.zeros((n_bins, nfilters), dtype=np.int32)
        for f in range(nfilters):
            nfields_s_filter[:, f] = np.bincount(bins_membership_arr, weights=in_survey_filter_plan[:, f], minlength=n_bins)
        active_bins_s_filter = nfields_s_filter > 0

        for night, group in tqdm(night_groups, total=night_groups.ngroups, desc='Calculating night history bin features'):
            # Initialize 1D total visit counters
            cur_survey_visits = night2visithistory[night].copy()
            cur_night_visits = np.zeros(nfields, dtype=np.int32)
            
            # Initialize 2D filter visit counters
            cur_survey_filter_visits = night2filtervisithistory[night].copy()
            cur_night_filter_visits = np.zeros((nfields, nfilters), dtype=np.int32)
            
            # Get field ids and filter indices at each step before loop
            step_fids = group['field_id'].to_numpy(dtype=np.int32)
            step_filts = group['filt_idx'].to_numpy(dtype=np.int32)
                        
            # Get night visit limits
            # Get max visits to each field tonight
            valid_night_mask = group['object'] != 'zenith'
            night_fids_raw = group['field_id'][valid_night_mask].to_numpy().astype(np.int32)
            night_filts_raw = group['filt_idx'][valid_night_mask].to_numpy().astype(np.int32)

            mapped_night_fids = field_ids[night_fids_raw]
            valid_mapped_mask = mapped_night_fids != -1
            
            max_n_visits_arr = np.bincount(mapped_night_fids[valid_mapped_mask], minlength=nfields)
            in_night_plan = max_n_visits_arr > 0

            # 2D target matrix for tonight's filter visits
            max_n_filter_visits_arr = np.zeros((nfields, nfilters), dtype=np.int32)
            np.add.at(
                max_n_filter_visits_arr, 
                (mapped_night_fids[valid_mapped_mask], night_filts_raw[valid_mapped_mask]), 
                1
            )
            in_night_filter_plan = max_n_filter_visits_arr > 0

            # specific case of teff < .3
            max_s_visits_arr = np.maximum(max_n_visits_arr, max_s_visits_arr_all)
            max_s_filter_visits_arr = np.maximum(max_n_filter_visits_arr, max_s_filter_visits_arr_all)
            
            nfields_n = np.bincount(bins_membership_arr, weights=in_night_plan, minlength=n_bins)
            active_bins_n = nfields_n > 0

            # Precompute tonight's valid fields per bin for each filter
            nfields_n_filter = np.zeros((n_bins, nfilters), dtype=np.float64)
            for f in range(nfilters):
                nfields_n_filter[:, f] = np.bincount(bins_membership_arr, weights=in_night_filter_plan[:, f], minlength=n_bins)
            active_bins_n_filter = nfields_n_filter > 0
            
            for i in range(len(group)):
                obs_fid = step_fids[i]
                obs_filt = step_filts[i]
                
                if obs_fid != -1:
                    idx = field_ids[obs_fid]
                    if idx != -1: 
                        # Update 1D counters
                        cur_survey_visits[idx] += 1
                        cur_night_visits[idx] += 1
                        
                        # Update 2D filter counters
                        cur_survey_filter_visits[idx, obs_filt] += 1
                        cur_night_filter_visits[idx, obs_filt] += 1
                
                # --- Bin's historic features based on (FIELD, FILTER) --- #

                # Assign sentinel values
                for key in ['survey_num_unvisited_fields', 'night_num_unvisited_fields', 
                            'survey_num_incomplete_fields', 'night_num_incomplete_fields']:
                    for filt_name in idx2filter.values():
                        calculated_features[f'{key}_{filt_name}'][global_idx] = -1. # bins with no viable fields get sentinel value -1
                                
                for f in range(nfilters):
                    filt_name = idx2filter[f]
                    
                    # Unvisited specific filter
                    
                    s_unvisited_f = np.bincount(bins_membership_arr, weights=(cur_survey_filter_visits[:, f] == 0) & in_survey_filter_plan[:, f], minlength=n_bins)
                    n_unvisited_f = np.bincount(bins_membership_arr, weights=(cur_night_filter_visits[:, f] == 0) & in_night_filter_plan[:, f], minlength=n_bins)

                    # Incomplete specific filter
                    s_incomplete_f = np.bincount(bins_membership_arr, weights=(cur_survey_filter_visits[:, f] < max_s_filter_visits_arr[:, f]) & in_survey_filter_plan[:, f], minlength=n_bins)
                    n_incomplete_f = np.bincount(bins_membership_arr, weights=(cur_night_filter_visits[:, f] < max_n_filter_visits_arr[:, f]) & in_night_filter_plan[:, f], minlength=n_bins)

                    # Filter specific division
                    np.divide(s_unvisited_f, nfields_s_filter[:, f], out=calculated_features[f'survey_num_unvisited_fields_{filt_name}'][global_idx], where=active_bins_s_filter[:, f])
                    np.divide(n_unvisited_f, nfields_n_filter[:, f], out=calculated_features[f'night_num_unvisited_fields_{filt_name}'][global_idx], where=active_bins_n_filter[:, f])
                    np.divide(s_incomplete_f, nfields_s_filter[:, f], out=calculated_features[f'survey_num_incomplete_fields_{filt_name}'][global_idx], where=active_bins_s_filter[:, f])
                    np.divide(n_incomplete_f, nfields_n_filter[:, f], out=calculated_features[f'night_num_incomplete_fields_{filt_name}'][global_idx], where=active_bins_n_filter[:, f])
                
                # Min tiling
                s_filter_tiling_all = np.full_like(cur_survey_filter_visits, 2.0, dtype=np.float32)
                n_filter_tiling_all = np.full_like(cur_night_filter_visits, 2.0, dtype=np.float32)
                
                np.divide(cur_survey_filter_visits, max_s_filter_visits_arr, out=s_filter_tiling_all, where=in_survey_filter_plan)
                np.divide(cur_night_filter_visits, max_n_filter_visits_arr, out=n_filter_tiling_all, where=in_night_filter_plan)
                
                s_filter_mins = np.full((n_bins, nfilters), 2.0, dtype=np.float32) 
                n_filter_mins = np.full((n_bins, nfilters), 2.0, dtype=np.float32)
                
                for f in range(nfilters):
                    np.minimum.at(s_filter_mins[:, f], bins_membership_arr, s_filter_tiling_all[:, f])
                    np.minimum.at(n_filter_mins[:, f], bins_membership_arr, n_filter_tiling_all[:, f])
                
                s_filter_mins[s_filter_mins > 1.0] = -1.0
                n_filter_mins[n_filter_mins > 1.0] = -1.0
                
                for f in range(nfilters):
                    filt_name = idx2filter[f]
                    calculated_features[f'survey_min_tiling_{filt_name}'][global_idx] = s_filter_mins[:, f]
                    calculated_features[f'night_min_tiling_{filt_name}'][global_idx] = n_filter_mins[:, f]

                global_idx += 1
    else:
        # Get max visits per field and number of fields per bin for entire survey
        max_s_visits_arr_all = np.array([field2maxvisits[fid] for fid in field_ids], dtype=np.int32) # visits per field
        in_survey_plan = max_s_visits_arr_all > 0 # mask fields not in survey (field2maxvisits should be built such that field ids only include fields in survey)
        
        nfields_s = np.bincount(bins_membership_arr, weights=in_survey_plan, minlength=n_bins) # number of fields per bin
        active_bins_s = nfields_s > 0
        
        global_idx = 0
        night_groups = pt_df.groupby('night')
        for night, group in tqdm(night_groups, total=night_groups.ngroups, desc='Calculating night history bin features'):
            # Initialize visit counters
            cur_survey_visits = night2visithistory[night].copy()
            cur_night_visits = np.zeros(nfields, dtype=np.int32)
            
            # Get field ids at each step before loop
            step_fids = group['field_id'].to_numpy(dtype=np.int32)
            
            # Get max visits to each field tonight
            night_fids_raw = group['field_id'][group['object'] != 'zenith'].to_numpy().astype(np.int32)
            max_n_visits_arr = np.bincount(field_ids[night_fids_raw], minlength=nfields)
            in_night_plan = max_n_visits_arr > 0

            # If fields visited tonight multiple times and all have teff < .3, add these visits to survey wide counts (field2maxvisits only counts observations with teff < .3 once)
            max_s_visits_arr = np.maximum(max_n_visits_arr, max_s_visits_arr_all)
            
            # Get number of fields in each bin
            nfields_n = np.bincount(bins_membership_arr, weights=in_night_plan, minlength=n_bins)
            active_bins_n = nfields_n > 0
            
            for i in range(len(group)):
                obs_fid = step_fids[i]
                if obs_fid != -1:
                    idx = field_ids[obs_fid]
                    if idx != -1: # Make sure fid is a valid field (for case of sparse field ids)
                        cur_survey_visits[idx] += 1
                        cur_night_visits[idx] += 1
        
                # Get number of unvisited fields in each bin - bins below horizon have 0 fields unvisited
                s_unvisited = np.bincount(bins_membership_arr, weights=(cur_survey_visits == 0) & in_survey_plan, minlength=n_bins)
                n_unvisited = np.bincount(bins_membership_arr, weights=(cur_night_visits == 0) & in_night_plan, minlength=n_bins)

                # Get number of incomplete fields in each bin
                s_incomplete_mask = (cur_survey_visits < max_s_visits_arr) & in_survey_plan
                n_incomplete_mask = (cur_night_visits < max_n_visits_arr) & in_night_plan
                s_incomplete = np.bincount(bins_membership_arr, weights=s_incomplete_mask, minlength=n_bins)
                n_incomplete = np.bincount(bins_membership_arr, weights=n_incomplete_mask, minlength=n_bins)
        
                # Create a zero-filled array for the results
                for key in ['survey_num_unvisited_fields', 'night_num_unvisited_fields', 
                            'survey_num_incomplete_fields', 'night_num_incomplete_fields']:
                    calculated_features[key][global_idx] = -1. # bins with no viable fields get sentinel value -1
                
                # Do division in-place (bypasses runtimewarning error )
                np.divide(s_unvisited, nfields_s, out=calculated_features['survey_num_unvisited_fields'][global_idx], where=active_bins_s)
                np.divide(n_unvisited, nfields_n, out=calculated_features['night_num_unvisited_fields'][global_idx], where=active_bins_n)
                np.divide(s_incomplete, nfields_s, out=calculated_features['survey_num_incomplete_fields'][global_idx], where=active_bins_s)
                np.divide(n_incomplete, nfields_n, out=calculated_features['night_num_incomplete_fields'][global_idx], where=active_bins_n)
                
        
                # Min tiling
                s_tiling_all = np.full_like(cur_survey_visits, 2.0, dtype=np.float32)
                n_tiling_all = np.full_like(cur_night_visits, 2.0, dtype=np.float32)
                # current_num_visits_field / max_num_visits_field only where max_num_visits_field > 0 ie, in the plan
                np.divide(cur_survey_visits, max_s_visits_arr, out=s_tiling_all, where=in_survey_plan)
                np.divide(cur_night_visits, max_n_visits_arr, out=n_tiling_all, where=in_night_plan)
                
                s_mins = np.full(n_bins, 2.0, dtype=np.float32)
                n_mins = np.full(n_bins, 2.0, dtype=np.float32)
                np.minimum.at(s_mins, bins_membership_arr, s_tiling_all)
                np.minimum.at(n_mins, bins_membership_arr, n_tiling_all)
                
                # Reset bins with no fields back to -0.1
                s_mins[s_mins > 1.0] = -1.0
                n_mins[n_mins > 1.0] = -1.0
                calculated_features['survey_min_tiling'][global_idx] = s_mins
                calculated_features['night_min_tiling'][global_idx] = n_mins

                global_idx += 1
            
    return calculated_features
        
def calculate_history_dependent_bin_features_azel(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, 
                                                  night2filtervisithistory, field2maxvisits, fieldfilter2maxvisits, bin_space,
                                                  base_bin_feature_names):
    n_bins = len(hpGrid.idx_lookup)
    field_ids = np.array(list(field2maxvisits.keys()))
    nfields = len(field_ids)
    nfilters = len(FILTER2IDX)
    idx2filter = {v: k for k, v in FILTER2IDX.items()}

    ra_arr = np.array([field2radec[fid][0] for fid in field_ids])
    dec_arr = np.array([field2radec[fid][1] for fid in field_ids])
    
    max_s_visits_arr_all = np.array([field2maxvisits[fid] for fid in field_ids], dtype=np.int32)
    max_s_filter_visits_arr_all = np.array([fieldfilter2maxvisits[fid] for fid in field_ids], dtype=np.int32)

    # Add filter idx mapping to dataframe safely
    pt_df['filt_idx'] = pt_df['filter'].map(FILTER2IDX).fillna(-1)

    # --- TIME CACHING VARIABLES ---
    cache_time = -1e9
    v_bins_cache = None
    valid_mask_cache = None
    bc_s_cache = None
    bc_n_cache = None
    act_s_cache = None
    act_n_cache = None
    
    # Filter caching variables
    bc_s_f_cache = None
    bc_n_f_cache = None
    act_s_f_cache = None
    act_n_f_cache = None

    global_idx = 0
    for night, group in tqdm(pt_df.groupby('night'), desc='Calculating AzEl History'):
        # 1D counters
        cur_survey_visits = night2visithistory[night][field_ids].copy().astype(np.int32)
        cur_night_visits = np.zeros(nfields, dtype=np.int32)
        
        # 2D counters
        if 'filter' in bin_space:
            cur_survey_filter_visits = night2filtervisithistory[night].copy()
            cur_night_filter_visits = np.zeros((nfields, nfilters), dtype=np.int32)
        
        step_fids = group['field_id'].fillna(-1).to_numpy(dtype=np.int32)
        step_filts = group['filt_idx'].to_numpy(dtype=np.int32)
        step_times = group['timestamp'].to_numpy(dtype=np.int32)

        valid_night_mask = group['object'] != 'zenith'
        night_fids_raw = group['field_id'][valid_night_mask].fillna(-1).to_numpy().astype(np.int32)
        mapped_night_fids = field_ids[night_fids_raw]
        valid_mapped_mask = mapped_night_fids != -1
        
        # Night limits 1D
        max_n_visits_arr = np.bincount(mapped_night_fids[valid_mapped_mask], minlength=nfields)
        
        # Night limits 2D
        if 'filter' in bin_space:
            night_filts_raw = group['filt_idx'][valid_night_mask].to_numpy().astype(np.int32)
            max_n_filter_visits_arr = np.zeros((nfields, nfilters), dtype=np.int32)
            np.add.at(
                max_n_filter_visits_arr, 
                (mapped_night_fids[valid_mapped_mask], night_filts_raw[valid_mapped_mask]), 
                1
            )
            max_s_filter_visits_arr = np.maximum(max_n_filter_visits_arr, max_s_filter_visits_arr_all)
            
        max_s_visits_arr = np.maximum(max_n_visits_arr, max_s_visits_arr_all)

        for i in range(len(group)):
            timestamp = step_times[i]
            obs_fid = step_fids[i]
            obs_filt = step_filts[i]

            if obs_fid != -1:
                idx = field_ids[obs_fid]
                if idx != -1:
                    cur_survey_visits[idx] += 1
                    cur_night_visits[idx] += 1
                    
                    if 'filter' in bin_space and obs_filt != -1:
                        cur_survey_filter_visits[idx, obs_filt] += 1
                        cur_night_filter_visits[idx, obs_filt] += 1

            # 1. TIME CACHING: Refresh every 5 minutes (300s)
            if abs(timestamp - cache_time) > 300:
                az, el = ephemerides.equatorial_to_topographic(ra_arr, dec_arr, time=timestamp)
                bins_raw = hpGrid.ang2idx(lon=az, lat=el)
                
                # FIX: Explicitly handle None values and convert to numeric sentinel (-1)
                bins = np.array([b if b is not None else -1 for b in bins_raw], dtype=np.int32)
                valid_mask = (el > 0) & (bins != -1)
                
                v_bins = bins[valid_mask]
                
                # 1D Cache
                in_s_plan = max_s_visits_arr[valid_mask] > 0
                in_n_plan = max_n_visits_arr[valid_mask] > 0
                
                bin_count_s = np.bincount(v_bins, weights=in_s_plan, minlength=n_bins)
                bin_count_n = np.bincount(v_bins, weights=in_n_plan, minlength=n_bins)
                
                active_bins_s = bin_count_s > 0
                active_bins_n = bin_count_n > 0
                
                # Update 1D cache variables
                cache_time, v_bins_cache, valid_mask_cache = timestamp, v_bins, valid_mask
                bc_s_cache, bc_n_cache = bin_count_s, bin_count_n
                act_s_cache, act_n_cache = active_bins_s, active_bins_n
                
                # 2D Filter Cache
                if 'filter' in bin_space:
                    in_s_f_plan = max_s_filter_visits_arr[valid_mask] > 0
                    in_n_f_plan = max_n_filter_visits_arr[valid_mask] > 0

                    bin_count_s_filter = np.zeros((n_bins, nfilters), dtype=np.float64)
                    bin_count_n_filter = np.zeros((n_bins, nfilters), dtype=np.float64)

                    for f in range(nfilters):
                        bin_count_s_filter[:, f] = np.bincount(v_bins, weights=in_s_f_plan[:, f], minlength=n_bins)
                        bin_count_n_filter[:, f] = np.bincount(v_bins, weights=in_n_f_plan[:, f], minlength=n_bins)

                    active_bins_s_filter = bin_count_s_filter > 0
                    active_bins_n_filter = bin_count_n_filter > 0
                    
                    # Update 2D cache variables
                    bc_s_f_cache, bc_n_f_cache = bin_count_s_filter, bin_count_n_filter
                    act_s_f_cache, act_n_f_cache = active_bins_s_filter, active_bins_n_filter
            else:
                # Load from cache
                v_bins, valid_mask = v_bins_cache, valid_mask_cache
                bin_count_s, bin_count_n = bc_s_cache, bc_n_cache
                active_bins_s, active_bins_n = act_s_cache, act_n_cache
                
                if 'filter' in bin_space:
                    bin_count_s_filter, bin_count_n_filter = bc_s_f_cache, bc_n_f_cache
                    active_bins_s_filter, active_bins_n_filter = act_s_f_cache, act_n_f_cache

            # 2. CALCULATE 1D STATE
            v_survey_counts = cur_survey_visits[valid_mask]
            v_night_counts = cur_night_visits[valid_mask]
            
            v_max_v_survey = max_s_visits_arr[valid_mask]
            v_max_v_night = max_n_visits_arr[valid_mask]

            in_s_plan = v_max_v_survey > 0
            in_n_plan = v_max_v_night > 0

            for key_n, key_s, mask_n, mask_s in [
                ('night_num_unvisited_fields', 'survey_num_unvisited_fields', 
                 (v_night_counts == 0) & in_n_plan, 
                 (v_survey_counts == 0) & in_s_plan),
                 
                ('night_num_incomplete_fields', 'survey_num_incomplete_fields', 
                 (v_night_counts < v_max_v_night) & in_n_plan, 
                 (v_survey_counts < v_max_v_survey) & in_s_plan)
                ]:
                res_n, res_s = np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
                
                np.divide(np.bincount(v_bins, weights=mask_n, minlength=n_bins), bin_count_n, out=res_n, where=active_bins_n)
                np.divide(np.bincount(v_bins, weights=mask_s, minlength=n_bins), bin_count_s, out=res_s, where=active_bins_s)
                
                res_n[~active_bins_n] = -0.0
                res_s[~active_bins_s] = -0.0
                
                calculated_features[key_n][global_idx] = res_n
                calculated_features[key_s][global_idx] = res_s

            # Vectorized Min Tiling 1D
            s_tiling_all = np.full_like(v_survey_counts, 2.0, dtype=np.float32)
            n_tiling_all = np.full_like(v_night_counts, 2.0, dtype=np.float32)
            
            np.divide(v_survey_counts, v_max_v_survey, out=s_tiling_all, where=in_s_plan)
            np.divide(v_night_counts, v_max_v_night, out=n_tiling_all, where=in_n_plan)
            
            s_mins, n_mins = np.full(n_bins, 2.0, dtype=np.float32), np.full(n_bins, 2.0, dtype=np.float32)
            np.minimum.at(s_mins, v_bins, s_tiling_all)
            np.minimum.at(n_mins, v_bins, n_tiling_all)
            
            s_mins[~active_bins_s] = 0.0
            n_mins[~active_bins_n] = 0.0
            
            calculated_features['survey_min_tiling'][global_idx] = s_mins
            calculated_features['night_min_tiling'][global_idx] = n_mins
            
            # 3. CALCULATE 2D FILTER STATE
            if 'filter' in bin_space:
                v_survey_f_counts = cur_survey_filter_visits[valid_mask]
                v_night_f_counts = cur_night_filter_visits[valid_mask]
                
                v_max_v_survey_f = max_s_filter_visits_arr[valid_mask]
                v_max_v_night_f = max_n_filter_visits_arr[valid_mask]

                in_s_f_plan = v_max_v_survey_f > 0
                in_n_f_plan = v_max_v_night_f > 0
                
                for f in range(nfilters):
                    filt_name = idx2filter[f]
                    
                    # Unvisited specific filter
                    mask_n_unv_f = (v_night_f_counts[:, f] == 0) & in_n_f_plan[:, f]
                    mask_s_unv_f = (v_survey_f_counts[:, f] == 0) & in_s_f_plan[:, f]

                    # Incomplete specific filter
                    mask_n_inc_f = (v_night_f_counts[:, f] < v_max_v_night_f[:, f]) & in_n_f_plan[:, f]
                    mask_s_inc_f = (v_survey_f_counts[:, f] < v_max_v_survey_f[:, f]) & in_s_f_plan[:, f]
                    
                    res_n_unv, res_s_unv = np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
                    res_n_inc, res_s_inc = np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)

                    # Filter specific division
                    np.divide(np.bincount(v_bins, weights=mask_n_unv_f, minlength=n_bins), bin_count_n_filter[:, f], out=res_n_unv, where=active_bins_n_filter[:, f])
                    np.divide(np.bincount(v_bins, weights=mask_s_unv_f, minlength=n_bins), bin_count_s_filter[:, f], out=res_s_unv, where=active_bins_s_filter[:, f])
                    
                    np.divide(np.bincount(v_bins, weights=mask_n_inc_f, minlength=n_bins), bin_count_n_filter[:, f], out=res_n_inc, where=active_bins_n_filter[:, f])
                    np.divide(np.bincount(v_bins, weights=mask_s_inc_f, minlength=n_bins), bin_count_s_filter[:, f], out=res_s_inc, where=active_bins_s_filter[:, f])

                    # Sentinel values for inactive filter bins
                    res_n_unv[~active_bins_n_filter[:, f]] = -1.0
                    res_s_unv[~active_bins_s_filter[:, f]] = -1.0
                    res_n_inc[~active_bins_n_filter[:, f]] = -1.0
                    res_s_inc[~active_bins_s_filter[:, f]] = -1.0

                    calculated_features[f'night_num_unvisited_fields_{filt_name}'][global_idx] = res_n_unv
                    calculated_features[f'survey_num_unvisited_fields_{filt_name}'][global_idx] = res_s_unv
                    calculated_features[f'night_num_incomplete_fields_{filt_name}'][global_idx] = res_n_inc
                    calculated_features[f'survey_num_incomplete_fields_{filt_name}'][global_idx] = res_s_inc

                # Min tiling filter specific
                s_filter_tiling_all = np.full_like(v_survey_f_counts, 2.0, dtype=np.float32)
                n_filter_tiling_all = np.full_like(v_night_f_counts, 2.0, dtype=np.float32)
                
                np.divide(v_survey_f_counts, v_max_v_survey_f, out=s_filter_tiling_all, where=in_s_f_plan)
                np.divide(v_night_f_counts, v_max_v_night_f, out=n_filter_tiling_all, where=in_n_f_plan)
                
                s_filter_mins = np.full((n_bins, nfilters), 2.0, dtype=np.float32) 
                n_filter_mins = np.full((n_bins, nfilters), 2.0, dtype=np.float32)
                
                for f in range(nfilters):
                    np.minimum.at(s_filter_mins[:, f], v_bins, s_filter_tiling_all[:, f])
                    np.minimum.at(n_filter_mins[:, f], v_bins, n_filter_tiling_all[:, f])
                    
                    # Sentinel values
                    s_filter_mins[~active_bins_s_filter[:, f], f] = -1.0
                    n_filter_mins[~active_bins_n_filter[:, f], f] = -1.0
                    
                    filt_name = idx2filter[f]
                    calculated_features[f'survey_min_tiling_{filt_name}'][global_idx] = s_filter_mins[:, f]
                    calculated_features[f'night_min_tiling_{filt_name}'][global_idx] = n_filter_mins[:, f]

            global_idx += 1

    return calculated_features
# def calculate_history_dependent_bin_features_azel(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, 
#                                                   night2filtervisithistory, field2maxvisits, fieldfilter2maxvisits, bin_space):
#     n_bins = len(hpGrid.idx_lookup)
#     field_ids = np.array(list(field2maxvisits.keys()))
#     nfields = len(field_ids)
#     idx2filter = {v: k for k, v in FILTER2IDX.items()}

#     ra_arr = np.array([field2radec[fid][0] for fid in field_ids])
#     dec_arr = np.array([field2radec[fid][1] for fid in field_ids])
#     max_s_visits_arr = np.array([field2maxvisits[fid] for fid in field_ids], dtype=np.int32)
#     max_s_filter_visits_arr = np.array([fieldfilter2maxvisits[fid] for fid in field_ids], dtype=np.int32)

#     # --- TIME CACHING VARIABLES ---
#     cache_time = -1e9
#     v_bins_cache = None
#     active_bins_cache = None
#     bin_count_cache = None
#     valid_mask_cache = None

#     global_idx = 0
#     for night, group in tqdm(pt_df.groupby('night'), desc='Calculating AzEl History'):
#         cur_survey_visits = night2visithistory[night][field_ids].copy().astype(np.int32)
#         cur_night_visits = np.zeros(nfields, dtype=np.int32)
        
#         step_fids = group['field_id'].to_numpy(dtype=np.int32)
#         step_times = group['timestamp'].to_numpy(dtype=np.int32)

#         night_fids_raw = group['field_id'][group['object'] != 'zenith'].to_numpy().astype(np.int32)
#         max_n_visits_arr = np.bincount(field_ids[night_fids_raw], minlength=nfields)

#         for i in range(len(group)):
#             timestamp = step_times[i]
#             obs_fid = step_fids[i]

#             if obs_fid != -1:
#                 idx = field_ids[obs_fid]
#                 if idx != -1:
#                     cur_survey_visits[idx] += 1
#                     cur_night_visits[idx] += 1

#             # 1. TIME CACHING: Refresh every 5 minutes (300s)
#             if abs(timestamp - cache_time) > 300:
#                 az, el = ephemerides.equatorial_to_topographic(ra_arr, dec_arr, time=timestamp)
#                 bins_raw = hpGrid.ang2idx(lon=az, lat=el)
                
#                 # FIX: Explicitly handle None values and convert to numeric sentinel (-1)
#                 bins = np.array([b if b is not None else -1 for b in bins_raw], dtype=np.int32)
#                 valid_mask = (el > 0) & (bins != -1)
                
#                 v_bins = bins[valid_mask]
                
#                 # Check if the fields above horizon are actually in the plans
#                 in_s_plan = max_s_visits_arr[valid_mask] > 0
#                 in_n_plan = max_n_visits_arr[valid_mask] > 0
                
#                 # Count fields per bin for Survey vs Night
#                 bin_count_s = np.bincount(v_bins, weights=in_s_plan, minlength=n_bins)
#                 bin_count_n = np.bincount(v_bins, weights=in_n_plan, minlength=n_bins)
                
#                 active_bins_s = bin_count_s > 0
#                 active_bins_n = bin_count_n > 0
                
#                 cache_time, v_bins_cache, valid_mask_cache = timestamp, v_bins, valid_mask
                
#                 # Update cache variables
#                 bc_s_cache, bc_n_cache = bin_count_s, bin_count_n
#                 act_s_cache, act_n_cache = active_bins_s, active_bins_n
#             else:
#                 v_bins, valid_mask = v_bins_cache, valid_mask_cache
#                 bin_count_s, bin_count_n = bc_s_cache, bc_n_cache
#                 active_bins_s, active_bins_n = act_s_cache, act_n_cache

#             # 2. CALCULATE STATE
#             v_survey_counts = cur_survey_visits[valid_mask]
#             v_night_counts = cur_night_visits[valid_mask]
            
#             v_max_v_survey = max_s_visits_arr[valid_mask]
#             v_max_v_night = max_n_visits_arr[valid_mask]

#             # Re-create the plan masks for the state checks
#             in_s_plan = v_max_v_survey > 0
#             in_n_plan = v_max_v_night > 0

#             for key_n, key_s, mask_n, mask_s in [
#                 # Must be unvisited AND in the respective plan
#                 ('night_num_unvisited_fields', 'survey_num_unvisited_fields', 
#                  (v_night_counts == 0) & in_n_plan, 
#                  (v_survey_counts == 0) & in_s_plan),
                 
#                 # Must be incomplete AND in the respective plan
#                 ('night_num_incomplete_fields', 'survey_num_incomplete_fields', 
#                  (v_night_counts < v_max_v_night) & in_n_plan, 
#                  (v_survey_counts < v_max_v_survey) & in_s_plan)
#                 ]:
#                 res_n, res_s = np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
                
#                 # Use the correct denominators and active masks!
#                 np.divide(np.bincount(v_bins, weights=mask_n, minlength=n_bins), bin_count_n, out=res_n, where=active_bins_n)
#                 np.divide(np.bincount(v_bins, weights=mask_s, minlength=n_bins), bin_count_s, out=res_s, where=active_bins_s)
                
#                 res_n[~active_bins_n] = 0.
#                 res_s[~active_bins_s] = 0.
                
#                 calculated_features[key_n][global_idx] = res_n
#                 calculated_features[key_s][global_idx] = res_s

#             # Vectorized Min Tiling (With Safe Division)
#             s_tiling_all = np.full_like(v_survey_counts, 2.0, dtype=np.float32)
#             n_tiling_all = np.full_like(v_night_counts, 2.0, dtype=np.float32)
            
#             np.divide(v_survey_counts, v_max_v_survey, out=s_tiling_all, where=in_s_plan)
#             np.divide(v_night_counts, v_max_v_night, out=n_tiling_all, where=in_n_plan)
            
#             s_mins, n_mins = np.full(n_bins, 2.0, dtype=np.float32), np.full(n_bins, 2.0, dtype=np.float32)
#             np.minimum.at(s_mins, v_bins, s_tiling_all)
#             np.minimum.at(n_mins, v_bins, n_tiling_all)
            
#             s_mins[~active_bins_s] = 0.
#             n_mins[~active_bins_n] = 0.
            
#             calculated_features['survey_min_tiling'][global_idx] = s_mins
#             calculated_features['night_min_tiling'][global_idx] = n_mins
            
#             global_idx += 1

#     return calculated_features

def old_calculate_night_history_bin_features_radec(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, field2maxvisits):
    n_bins = len(hpGrid.idx_lookup)
    fids = np.array(list(field2maxvisits.keys()))
    nfields = len(fids)
    fid2idx = np.full(fids.max() + 1, -1, dtype=np.int32)
    for idx, fid in enumerate(fids):
        fid2idx[fid] = idx
    
    ra_arr = np.array([field2radec[fid][0] for fid in fids])
    dec_arr = np.array([field2radec[fid][1] for fid in fids])
    bins_arr = hpGrid.ang2idx(lon=ra_arr, lat=dec_arr) # Bin membership of each field ordered by field idx
    max_s_visits_arr = np.array([field2maxvisits[fid] for fid in fids], dtype=np.int32)
    has_survey_plan = max_s_visits_arr > 0
    
    global_idx = 0

    night_groups = pt_df.groupby('night')
    
    for night, group in tqdm(night_groups, total=night_groups.ngroups, desc='Calculating night history bin features'):
        cur_survey_visits = night2visithistory[night].copy()
        cur_night_visits = np.zeros(nfields, dtype=np.int32)
        
        step_fids = group['field_id'].to_numpy(dtype=np.int32)
        step_times = group['timestamp'].to_numpy(dtype=np.int32)
        
        night_fids_raw = group['field_id'][group['object'] != 'zenith'].to_numpy().astype(np.int32)
        max_n_visits_arr = np.bincount(fid2idx[night_fids_raw], minlength=nfields)
        has_night_plan = max_n_visits_arr > 0
        # max_s_visits_arr = np.maximum(max_n_visits_arr, max_s_visits_arr_all)

        for i in range(len(group)):
            timestamp = step_times[i]
            obs_fid = step_fids[i]
    
            # Get fields above horizon
            _, fields_el = ephemerides.equatorial_to_topographic(ra=ra_arr, dec=dec_arr, time=timestamp)
            valid_mask = fields_el > 0
    
            # Mask fields below horizon
            valid_bins = bins_arr[valid_mask]
            valid_night_counts = cur_night_visits[valid_mask]
            valid_night_max_visits = max_n_visits_arr[valid_mask]
            valid_has_n_plan = has_night_plan[valid_mask]

            valid_survey_counts = cur_survey_visits[valid_mask]
            valid_survey_max_visits = max_s_visits_arr[valid_mask]
            valid_has_s_plan = has_survey_plan[valid_mask]

            # Get number of fields in each bin
            nfields_s = np.bincount(valid_bins, weights=valid_has_s_plan, minlength=n_bins)
            nfields_n = np.bincount(valid_bins, weights=valid_has_n_plan, minlength=n_bins)
            active_bins_s = nfields_s > 0
            active_bins_n = nfields_n > 0
    
            # Get number of unvisited fields in each bin - bins below horizon have 0 fields unvisited
            s_unvisited = np.bincount(valid_bins, weights=(valid_survey_counts == 0) & valid_has_s_plan, minlength=n_bins)
            n_unvisited = np.bincount(valid_bins, weights=(valid_night_counts == 0) & valid_has_n_plan, minlength=n_bins)
    
            s_incomplete_mask = (valid_survey_counts < valid_survey_max_visits) & valid_has_s_plan
            n_incomplete_mask = (valid_night_counts < valid_night_max_visits) & valid_has_n_plan
            s_incomplete = np.bincount(valid_bins, weights=s_incomplete_mask, minlength=n_bins)
            n_incomplete = np.bincount(valid_bins, weights=n_incomplete_mask, minlength=n_bins)
    
            # Create a zero-filled array for the results
            for key in ['survey_num_unvisited_fields', 'night_num_unvisited_fields', 
                        'survey_num_incomplete_fields', 'night_num_incomplete_fields']:
                calculated_features[key][global_idx] = -0.1
            
            # Do division in-place (bypasses runtimewarning error )
            np.divide(s_unvisited, nfields_s, out=calculated_features['survey_num_unvisited_fields'][global_idx], where=active_bins_s)
            np.divide(n_unvisited, nfields_n, out=calculated_features['night_num_unvisited_fields'][global_idx], where=active_bins_n)
            np.divide(s_incomplete, nfields_s, out=calculated_features['survey_num_incomplete_fields'][global_idx], where=active_bins_s)
            np.divide(n_incomplete, nfields_n, out=calculated_features['night_num_incomplete_fields'][global_idx], where=active_bins_n)
    
            # Min tiling
            s_tiling_all = np.full_like(valid_survey_counts, 2.0, dtype=np.float32)
            n_tiling_all = np.full_like(valid_night_counts, 2.0, dtype=np.float32)
            np.divide(valid_survey_counts, valid_survey_max_visits, out=s_tiling_all, where=valid_has_s_plan)
            np.divide(valid_night_counts, valid_night_max_visits, out=n_tiling_all, where=valid_has_n_plan)
            
            s_mins = np.full(n_bins, 2.0, dtype=np.float32)
            n_mins = np.full(n_bins, 2.0, dtype=np.float32)
            np.minimum.at(s_mins, valid_bins, s_tiling_all)
            np.minimum.at(n_mins, valid_bins, n_tiling_all)
            
            # Reset bins with no fields back to -0.1
            s_mins[s_mins > 1.0] = -0.1
            n_mins[n_mins > 1.0] = -0.1
            calculated_features['survey_min_tiling'][global_idx] = s_mins
            calculated_features['night_min_tiling'][global_idx] = n_mins
            
            if obs_fid != -1:
                idx = fid2idx[obs_fid]
                if idx != -1: # Make sure fid is a valid field (for case of sparse field ids)
                    cur_survey_visits[idx] += 1
                    cur_night_visits[idx] += 1 
                    
            global_idx += 1
        
    return calculated_features

def old_calculate_historical_bin_features_azel(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, field2maxvisits):
    n_bins = len(hpGrid.idx_lookup)

    # Save (all) field radecs for quick access during loop
    fids = np.array(list(field2maxvisits.keys()))
    nfields = len(fids)
    max_fid = fids[-1]

    # Field to index mapping for sparse field ids; unused fields maps to -1
    fid2idx = np.full(max_fid + 1, -1, dtype=np.int32)
    for idx, fid in enumerate(fids):
        fid2idx[fid] = idx

    # Get compact radec arrays - ie, skip fields not present in field2maxvisits
    ra_arr = np.zeros(nfields)
    dec_arr = np.zeros(nfields)
    max_v_arr = np.zeros(nfields, dtype=np.int32)
    for idx, fid in enumerate(fids):
        ra_arr[idx], dec_arr[idx] = field2radec[fid]
        max_v_arr[idx] = field2maxvisits[fid]

    # Row index
    global_idx = 0

    for night, group in tqdm(pt_df.groupby('night'), total=pt_df.groupby('night').ngroups, desc='Calculating night history bin features'):
        
        # Get field visit counts at start of night
        cur_survey_visits = night2visithistory[night][fids].copy().astype(np.int32)
        cur_night_visits = np.zeros(nfields, dtype=np.int32)

        # Speed up loop by extracting dataframe values beforehand
        step_fids = group['field_id'].to_numpy(dtype=np.int32)
        step_times = group['timestamp'].to_numpy(dtype=np.int32)

        for i in range(len(group)):
            timestamp = step_times[i]
            obs_fid = step_fids[i]
            
            az, el = ephemerides.equatorial_to_topographic(ra_arr, dec_arr, time=timestamp)

            bins = hpGrid.ang2idx(lon=az, lat=el) # Bin membership of each field
            valid_mask = el > 0

            # Mask quantities whose associated field is below horizon
            v_bins = bins[valid_mask].astype(np.int32)
            v_survey_counts = cur_survey_visits[valid_mask].astype(np.int32)
            v_night_counts = cur_night_visits[valid_mask].astype(np.int32)
            v_max_v = max_v_arr[valid_mask].astype(np.int32)

            # Count total visible fields in each bin
            bin_count = np.bincount(v_bins, minlength=n_bins)
            active_bins = bin_count > 0

            # Num Unvisited fields
            s_unvisited = np.bincount(v_bins, weights=(v_survey_counts == 0), minlength=n_bins)
            n_unvisited = np.bincount(v_bins, weights=(v_night_counts == 0), minlength=n_bins)
            
            # Num Incomplete fields
            s_incomplete_mask = v_survey_counts < v_max_v
            s_incomplete = np.bincount(v_bins, weights=s_incomplete_mask, minlength=n_bins)
            n_incomplete_mask = v_night_counts < v_max_v
            n_incomplete = np.bincount(v_bins, weights=n_incomplete_mask, minlength=n_bins)
            
            # Create a zero-filled array for the results
            s_unvisited_frac = np.zeros_like(s_unvisited)
            n_unvisited_frac = np.zeros_like(n_unvisited)
            s_incomplete_frac = np.zeros_like(s_incomplete)
            n_incomplete_frac = np.zeros_like(n_incomplete)

            # Do division in-place (bypasses runtimewarning error )
            np.divide(s_unvisited, bin_count, out=s_unvisited_frac, where=active_bins)
            np.divide(n_unvisited, bin_count, out=n_unvisited_frac, where=active_bins)
            np.divide(s_incomplete, bin_count, out=s_incomplete_frac, where=active_bins)
            np.divide(n_incomplete, bin_count, out=n_incomplete_frac, where=active_bins)

            # Record to dictionary
            calculated_features['survey_num_unvisited_fields'][global_idx] = s_unvisited_frac
            calculated_features['night_num_unvisited_fields'][global_idx] = n_unvisited_frac
            calculated_features['survey_num_incomplete_fields'][global_idx] = s_incomplete_frac
            calculated_features['night_num_incomplete_fields'][global_idx] = n_incomplete_frac

            # # Min Tiling
            # unique_bins = np.where(active_bins)[0]
            # s_tiling_all = v_survey_counts / v_max_v
            # n_tiling_all = v_night_counts / v_max_v

            # for b in unique_bins:
            #     mask = v_bins == b
            #     calculated_features['survey_min_tiling'][global_idx, b] = np.min(s_tiling_all[mask])
            #     calculated_features['night_min_tiling'][global_idx, b] = np.min(n_tiling_all[mask])

            # --- VECTORIZED MIN TILING  --- #
            s_tiling_all = v_survey_counts / v_max_v
            n_tiling_all = v_night_counts / v_max_v

            # Init with sentinel -0.1, but use high value for intermediate min check
            s_mins = np.full(n_bins, 2.0, dtype=np.float32)
            n_mins = np.full(n_bins, 2.0, dtype=np.float32)
            
            np.minimum.at(s_mins, v_bins, s_tiling_all)
            np.minimum.at(n_mins, v_bins, n_tiling_all)
            
            # Reset bins with no fields back to -0.1
            s_mins[~active_bins] = -0.1
            n_mins[~active_bins] = -0.1
            
            calculated_features['survey_min_tiling'][global_idx] = s_mins
            calculated_features['night_min_tiling'][global_idx] = n_mins

            if obs_fid != -1:
                idx = fid2idx[obs_fid]
                if idx != -1: # Make sure fid is a valid field (for case of sparse field ids)
                    cur_survey_visits[idx] += 1
                    cur_night_visits[idx] += 1

            global_idx += 1
            
    return calculated_features

