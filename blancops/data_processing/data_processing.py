import fitsio
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from datetime import timezone, timedelta
import ephem
from astropy.time import Time
import torch

from blancops.math import units
from blancops.ephemerides import ephemerides
from tqdm import tqdm
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


# --- DATAFRAME PROCESSING --- #
# TODO make into class
def load_raw_data_to_dataframe(fits_path):
    d = fitsio.read(fits_path)
    df = pd.DataFrame(d.astype(d.dtype.newbyteorder('='))) # Big-endian/little-endian error

    sel = (df['propid'] == '2012B-0001') & (df['exptime'] > 40) & (df['exptime'] < 100) & (~np.isnan(df['teff']))
    df = df[sel].copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['night'] = (df['datetime'] - pd.Timedelta(hours=12)).dt.normalize()
    df['night'] = df['night'] + (timedelta(days=1) - pd.Timedelta(seconds=1))
    df = df[df['datetime'].dt.year > 2010] # There are some 1970 rows even after selecting propid

    # Add timestamp col
    utc = pd.to_datetime(df['datetime'], utc=True)
    timestamps = (utc.astype('int64') // 10**9).values
    df['timestamp'] = timestamps.copy()
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

def drop_rows_in_DECam_data(df, objects_to_remove, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Drops nights (1) in year 1970, and (2) with specific objects (ie, SN or GW followup which are observed for long stretches of time)"""
    df = remove_dates(df, specific_years, specific_months, specific_days, specific_filters)
    
    # Remove specific nights according to object name
    # df = remove_specific_objects(objects_to_remove=objects_to_remove, df=df)
    pattern = '|'.join(objects_to_remove)
    mask = ~df['object'].str.contains(pattern, case=False, na=False)

    # Filter the DataFrame
    df = df[mask]

    # Some fields are mis-labelled - add '(outlier)' to these object names so that they are treated as separate fields
    df = relabel_mislabelled_objects(df)
    df = remove_outliers(df)
    df.sort_values(by='timestamp').reset_index(drop=True, inplace=True)
    return df

def remove_outliers(df):
    """Removes objects that have (outlier) in its object name"""
    df = df[~df['object'].astype(str).str.contains('(outlier)', regex=False, na=False)]
    return df

def remove_specific_objects(df, objects_to_remove):
    nights_with_special_fields = set()
    for i, spec_obj in enumerate(objects_to_remove):
        for night, subdf in df.groupby('night'):
            if any(spec_obj in obj_name for obj_name in subdf['object'].values) or any(subdf['object'] == ""):
                nights_with_special_fields.add(night)
    nights_to_remove_mask = df['night'].isin(nights_with_special_fields)

    df = df[~nights_to_remove_mask]
    assert not df.empty, "All nights have special fields"
    return df

def relabel_mislabelled_objects(df):
    """Renames object columns with 'object_name (outlier)' if they are outside of a certain cutoff from the median RA/Dec.

    Args
    ----
    df (pd.DataFrame): The dataframe with object names and RA/Dec positions.

    Returns
    -------
    df_relabelled (pd.DataFrame): The dataframe with relabelled objects.
    """
    object_radec_df = df[['object', 'ra', 'dec']]
    object_radec_groups = object_radec_df.groupby('object')
    df_relabelled = df.copy(deep=True)

    outlier_indices = []
    for _, g in object_radec_groups:
        cutoff_deg = 3
        median_ra = g.ra.median()
        delta_ra = g.ra - median_ra
        delta_ra_shifted = np.remainder(delta_ra + 180, 360) - 180
        mask_outlier_ra = np.abs(delta_ra_shifted) > cutoff_deg

        median_dec = g.dec.median()
        delta_dec = g.dec - median_dec
        delta_dec_shifted = np.remainder(delta_dec + 180, 360) - 180
        mask_outlier_dec = np.abs(delta_dec_shifted) > cutoff_deg

        mask_outlier = mask_outlier_ra | mask_outlier_dec

        if np.count_nonzero(mask_outlier) > 0:
            indices = g.index[mask_outlier].values
            outlier_indices.extend(indices)

    df_relabelled.loc[outlier_indices, 'object'] = [f'{obj_name} (outlier)' for obj_name in df.loc[outlier_indices, 'object'].values]
    return df_relabelled

def remove_dates(df, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Processes and filters the dataframe to return a new dataframe with added columns for current global state features"""
    # Add column which indicates observing night (noon to noon)
    # Get observations for specific years, days, filters, etc.
    if specific_years is not None and specific_years is not []:
        df = df[df['night'].dt.year.isin(specific_years)]
        assert not df.empty, f"Years {specific_years} do not exist in dataset"
    if specific_months is not None and specific_months is not []:
        df = df[df['night'].dt.month.isin(specific_months)]
        assert not df.empty, f"Months {specific_months} do not exist in years {specific_years}"
    if specific_days is not None and specific_days is not []:
        df = df[df['night'].dt.day.isin(specific_days)]
        assert not df.empty, f"Days {specific_days} do not exist in months {specific_months}, and years {specific_years}"
    if specific_filters is not None and specific_filters is not []:
        df = df[df['filter'].isin(specific_filters)]
        assert not df.empty, f"Filters {specific_filters} do not exist in days {specific_days}, months {specific_months}, and years {specific_years}"
    assert not df.empty, "No observations found for the specified year/month/day/filter selections."
    return df

def normalize_noncyclic_features(state, 
                                state_feature_names,
                                max_norm_feature_names,
                                ang_distance_norm_feature_names,
                                do_inverse_norm, do_max_norm, do_ang_distance_norm,
                                bin_space=None,
                                fix_nans=True):
    is_torch = torch.is_tensor(state)
    # build masks (numpy boolean array)
    airmass_mask = np.array(['airmass' in feat for feat in state_feature_names], dtype=bool)
    max_norm_mask = np.array([any(max_feat in feat for max_feat in max_norm_feature_names) for feat in state_feature_names], dtype=bool)
    ang_distance_mask = np.array([any(dist_feat in feat for dist_feat in ang_distance_norm_feature_names) for feat in state_feature_names], dtype=bool)

    if is_torch:
        airmass_mask = torch.tensor(airmass_mask, dtype=torch.bool, device=state.device)
        max_norm_mask = torch.tensor(max_norm_mask, dtype=torch.bool, device=state.device)
        ang_distance_mask = torch.tensor(ang_distance_mask, dtype=torch.bool, device=state.device)

    do_reshape = False

    if state.ndim == 3: # ie, if is bin states
        do_reshape = True
        nrows, nbins, nfeats_per_bin = state.shape
        if is_torch:
            state = state.flatten(start_dim=1)
        else:
            state = state.reshape(state.shape[0], -1) 
    # logger.debug(f"airmass mask shape {airmass_mask.shape}")
    # logger.debug(f"state shape {state.shape}")  
    if do_inverse_norm:
        # logger.debug(f"state shape {state.shape}, airmass mask shape {airmass_mask.shape}")
        state[..., airmass_mask] = 1.0 / state[..., airmass_mask]
    if do_max_norm:
        state[..., max_norm_mask] = state[..., max_norm_mask] / (np.pi / 2)
    if do_ang_distance_norm:
        # logger.debug(f"DOING ANG DISTANCE NORM for {ang_distance_mask.sum()} number of elements")
        state[..., ang_distance_mask] = state[..., ang_distance_mask] / np.pi
    if fix_nans:
        if is_torch:
            state[torch.isnan(state)] = 1.2
        else:
            state[np.isnan(state)] = 1.2
    if do_reshape:
        state = state.reshape(nrows, nbins, nfeats_per_bin)

    return state

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

def expand_feature_names_for_cyclic_norm(feature_names, cyclical_feature_names):
    # periodic vars first
    feature_names = [
        element 
        for feat_name in feature_names
        for element in ([feat_name + '_cos', feat_name + '_sin'] 
                        if any(string in feat_name and 'frac' not in feat_name for string in cyclical_feature_names)
                        else [feat_name])
        ]
    return feature_names

def setup_feature_names(include_default_features, additional_global_features, additional_bin_features,
                        default_global_feature_names, cyclical_feature_names, hpGrid, do_cyclical_norm,
                        grid_network):

    # include_bin_features = len(additional_global_features) > 0
    # Any experiment will likely have at least these state features
    required_point_features = default_global_feature_names \
                                if include_default_features else []
    # required_bin_features = default_bin_feature_names \
                                # if (include_default_features and include_bin_features) else []
    
    # Include additional features not in default features above
    additional_global_features = [] if additional_global_features is None else additional_global_features

    base_global_feature_names = required_point_features + additional_global_features
    base_bin_feature_names = np.unique(np.array(additional_bin_features)).tolist()
    if grid_network == None:
        prenorm_bin_feature_names = np.array([ [f'bin_{bin_num}_{bin_feat}'
                                        for bin_feat in base_bin_feature_names]
                                        for bin_num in range(len(hpGrid.idx_lookup))])
        prenorm_bin_feature_names = prenorm_bin_feature_names.flatten().tolist()
    elif grid_network == 'single_bin_scorer':
        prenorm_bin_feature_names = np.array([ [f'bin_{bin_num}_{bin_feat}'
                            for bin_feat in base_bin_feature_names]
                            for bin_num in range(len(hpGrid.idx_lookup))])
        prenorm_bin_feature_names = prenorm_bin_feature_names.flatten().tolist()
    else:
        raise NotImplementedError(f"grid_network {grid_network} not supported")

    base_feature_names = base_global_feature_names + prenorm_bin_feature_names

    # Replace cyclical features with their cyclical transforms/normalizations if on  
    if do_cyclical_norm:
        global_feature_names = expand_feature_names_for_cyclic_norm(base_global_feature_names.copy(), cyclical_feature_names)
        bin_feature_names = expand_feature_names_for_cyclic_norm(prenorm_bin_feature_names, cyclical_feature_names)
    
    if grid_network is None:
        state_feature_names = global_feature_names + bin_feature_names
    elif grid_network == 'single_bin_scorer':
        state_feature_names = global_feature_names

    return base_global_feature_names, base_bin_feature_names, base_feature_names, global_feature_names, \
        bin_feature_names, state_feature_names, prenorm_bin_feature_names

def setup_feature_names(base_global_feature_names, base_bin_feature_names, cyclical_feature_names, nbins, do_cyclical_norm, grid_network):
    """
    Returns
    -------
    global_feature_names (list): feature names after circular normalization. If grid_network is None, returns [global_feature_names] + [bin_feature_names]
    bin_feature_names (list): feature names after circular normalization but before adding 'bin_{i}_{feat}' prefixes
    expanded_global_feature_names
    """
    if len(base_bin_feature_names) > 0:
        prenorm_expanded_bin_feature_names = np.array([ [f'bin_{bin_num}_{bin_feat}'
                                        for bin_feat in base_bin_feature_names]
                                        for bin_num in range(nbins) ])
        prenorm_expanded_bin_feature_names = prenorm_expanded_bin_feature_names.flatten().tolist()
    else:
        prenorm_expanded_bin_feature_names = []

    # Replace cyclical features with their cyclical transforms/normalizations if on  
    if do_cyclical_norm:
        global_feature_names = expand_feature_names_for_cyclic_norm(base_global_feature_names.copy(), cyclical_feature_names)
        bin_feature_names = expand_feature_names_for_cyclic_norm(prenorm_expanded_bin_feature_names.copy(), cyclical_feature_names)
    else:
        global_feature_names = base_global_feature_names
        bin_feature_names = prenorm_expanded_bin_feature_names
    return global_feature_names, bin_feature_names, prenorm_expanded_bin_feature_names

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

    # 3. Get time dependent features (sun and moon pos)
    for idx, time in tqdm(zip(df.index, df['timestamp'].values), total=len(df['timestamp']), desc='Calculating sun and moon ra/dec and az/el'):
        sun_ra, sun_dec = ephemerides.get_source_ra_dec('sun', time=time)
        df.loc[idx, ['sun_ra', 'sun_dec']] = sun_ra, sun_dec
        df.loc[idx, ['sun_az', 'sun_el']] = ephemerides.equatorial_to_topographic(ra=sun_ra, dec=sun_dec, time=time)

        moon_ra, moon_dec = ephemerides.get_source_ra_dec('moon', time=time)
        df.loc[idx, ['moon_ra', 'moon_dec']] = moon_ra, moon_dec
        df.loc[idx, ['moon_az', 'moon_el']] = ephemerides.equatorial_to_topographic(ra=moon_ra, dec=moon_dec, time=time)

    # 4. Get times
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
                                   field2radec, field2maxvisits, bin_space):
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
        calculated_night_history_features = calculate_history_dependent_bin_features(pt_df=pt_df, hpGrid=hpGrid, field2radec=field2radec, night2visithistory=night2fieldvisits, field2maxvisits=field2maxvisits, bin_space=bin_space)
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

def calculate_history_dependent_bin_features(pt_df, hpGrid, night2visithistory, field2radec, field2maxvisits, bin_space):
    n_bins = len(hpGrid.idx_lookup)
    calculated_features = {
        'night_num_visits': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'night_num_unvisited_fields': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'night_num_incomplete_fields': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'night_min_tiling': -.1 * np.ones((len(pt_df), n_bins), dtype=np.float32), # set bin's min tiling to -1 if there are no fields visible
        'survey_num_visits': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'survey_num_unvisited_fields': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'survey_num_incomplete_fields': np.zeros((len(pt_df), n_bins), dtype=np.float32),
        'survey_min_tiling': -.1 * np.ones((len(pt_df), n_bins), dtype=np.float32) # set bin's min tiling to -1 if there are no fields visible
    }

    if hpGrid.is_azel:
        calculated_features = calculate_history_dependent_bin_features_azel(pt_df=pt_df, hpGrid=hpGrid, field2radec=field2radec, calculated_features=calculated_features, 
                                                                            night2visithistory=night2visithistory, field2maxvisits=field2maxvisits, bin_space=bin_space)
    else:
        calculated_features = calculate_history_dependent_bin_features_radec(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, field2maxvisits, bin_space=bin_space)
    
    for key, arr in calculated_features.items():
        if arr.min() < -.1 and arr.max() > 1.:
            logger.debug(f"{key} is not between 0 and 1. Array max/min={arr.max()}/{arr.min()}. Check normalization factor.")

    return calculated_features

def calculate_history_dependent_bin_features_radec(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, field2maxvisits, bin_space):
    n_bins = len(hpGrid.idx_lookup)
    fids = np.array(list(field2maxvisits.keys()))
    nfields = len(fids)
    fid2idx = np.full(fids.max() + 1, -1, dtype=np.int32) # in case field indices are sparse
    for idx, fid in enumerate(fids):
        fid2idx[fid] = idx

    # Get bin membership of all fields in survey
    ra_arr = np.array([field2radec[fid][0] for fid in fids])
    dec_arr = np.array([field2radec[fid][1] for fid in fids])
    bins_membership_arr = hpGrid.ang2idx(lon=ra_arr, lat=dec_arr) # Bin membership of each field ordered by field idx

    # Get max visits per field and number of fields per bin for entire survey
    max_s_visits_arr_all = np.array([field2maxvisits[fid] for fid in fids], dtype=np.int32) # visits per field
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
        max_n_visits_arr = np.bincount(fid2idx[night_fids_raw], minlength=nfields)
        in_night_plan = max_n_visits_arr > 0

        # If fields visited tonight multiple times and all have teff < .3, add these visits to survey wide counts (field2maxvisits only counts observations with teff < .3 once)
        max_s_visits_arr = np.maximum(max_n_visits_arr, max_s_visits_arr_all)
        
        # Get number of fields in each bin
        nfields_n = np.bincount(bins_membership_arr, weights=in_night_plan, minlength=n_bins)
        active_bins_n = nfields_n > 0
        
        for i in range(len(group)):
            obs_fid = step_fids[i]
            if obs_fid != -1:
                idx = fid2idx[obs_fid]
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
        
def calculate_history_dependent_bin_features_azel(pt_df, hpGrid, field2radec, calculated_features, night2visithistory, field2maxvisits, bin_space):
    n_bins = len(hpGrid.idx_lookup)
    fids = np.array(list(field2maxvisits.keys()))
    nfields = len(fids)
    fid2idx = np.full(fids.max() + 1, -1, dtype=np.int32)
    for idx, fid in enumerate(fids):
        fid2idx[fid] = idx

    ra_arr = np.array([field2radec[fid][0] for fid in fids])
    dec_arr = np.array([field2radec[fid][1] for fid in fids])
    max_v_arr = np.array([field2maxvisits[fid] for fid in fids], dtype=np.int32)

    # --- TIME CACHING VARIABLES ---
    cache_time = -1e9
    v_bins_cache = None
    active_bins_cache = None
    bin_count_cache = None
    valid_mask_cache = None

    global_idx = 0
    for night, group in tqdm(pt_df.groupby('night'), desc='Calculating AzEl History'):
        cur_survey_visits = night2visithistory[night][fids].copy().astype(np.int32)
        cur_night_visits = np.zeros(nfields, dtype=np.int32)
        
        step_fids = group['field_id'].to_numpy(dtype=np.int32)
        step_times = group['timestamp'].to_numpy(dtype=np.int32)

        night_fids_raw = group['field_id'][group['object'] != 'zenith'].to_numpy().astype(np.int32)
        max_n_visits_arr = np.bincount(fid2idx[night_fids_raw], minlength=nfields)

        for i in range(len(group)):
            timestamp = step_times[i]
            obs_fid = step_fids[i]

            if obs_fid != -1:
                idx = fid2idx[obs_fid]
                if idx != -1:
                    cur_survey_visits[idx] += 1
                    cur_night_visits[idx] += 1

            # 1. TIME CACHING: Refresh every 5 minutes (300s)
            if abs(timestamp - cache_time) > 300:
                az, el = ephemerides.equatorial_to_topographic(ra_arr, dec_arr, time=timestamp)
                bins_raw = hpGrid.ang2idx(lon=az, lat=el)
                
                # FIX: Explicitly handle None values and convert to numeric sentinel (-1)
                bins = np.array([b if b is not None else -1 for b in bins_raw], dtype=np.int32)
                valid_mask = (el > 0) & (bins != -1)
                
                v_bins = bins[valid_mask]
                
                # Check if the fields above horizon are actually in the plans
                in_s_plan = max_v_arr[valid_mask] > 0
                in_n_plan = max_n_visits_arr[valid_mask] > 0
                
                # Count fields per bin for Survey vs Night
                bin_count_s = np.bincount(v_bins, weights=in_s_plan, minlength=n_bins)
                bin_count_n = np.bincount(v_bins, weights=in_n_plan, minlength=n_bins)
                
                active_bins_s = bin_count_s > 0
                active_bins_n = bin_count_n > 0
                
                cache_time, v_bins_cache, valid_mask_cache = timestamp, v_bins, valid_mask
                
                # Update cache variables
                bc_s_cache, bc_n_cache = bin_count_s, bin_count_n
                act_s_cache, act_n_cache = active_bins_s, active_bins_n
            else:
                v_bins, valid_mask = v_bins_cache, valid_mask_cache
                bin_count_s, bin_count_n = bc_s_cache, bc_n_cache
                active_bins_s, active_bins_n = act_s_cache, act_n_cache

            # 2. CALCULATE STATE
            v_survey_counts = cur_survey_visits[valid_mask]
            v_night_counts = cur_night_visits[valid_mask]
            
            v_max_v_survey = max_v_arr[valid_mask]
            v_max_v_night = max_n_visits_arr[valid_mask]

            # Re-create the plan masks for the state checks
            in_s_plan = v_max_v_survey > 0
            in_n_plan = v_max_v_night > 0

            for key_n, key_s, mask_n, mask_s in [
                # Must be unvisited AND in the respective plan
                ('night_num_unvisited_fields', 'survey_num_unvisited_fields', 
                 (v_night_counts == 0) & in_n_plan, 
                 (v_survey_counts == 0) & in_s_plan),
                 
                # Must be incomplete AND in the respective plan
                ('night_num_incomplete_fields', 'survey_num_incomplete_fields', 
                 (v_night_counts < v_max_v_night) & in_n_plan, 
                 (v_survey_counts < v_max_v_survey) & in_s_plan)
                ]:
                res_n, res_s = np.zeros(n_bins, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
                
                # Use the correct denominators and active masks!
                np.divide(np.bincount(v_bins, weights=mask_n, minlength=n_bins), bin_count_n, out=res_n, where=active_bins_n)
                np.divide(np.bincount(v_bins, weights=mask_s, minlength=n_bins), bin_count_s, out=res_s, where=active_bins_s)
                
                res_n[~active_bins_n] = 0.
                res_s[~active_bins_s] = 0.
                
                calculated_features[key_n][global_idx] = res_n
                calculated_features[key_s][global_idx] = res_s

            # Vectorized Min Tiling (With Safe Division)
            s_tiling_all = np.full_like(v_survey_counts, 2.0, dtype=np.float32)
            n_tiling_all = np.full_like(v_night_counts, 2.0, dtype=np.float32)
            
            np.divide(v_survey_counts, v_max_v_survey, out=s_tiling_all, where=in_s_plan)
            np.divide(v_night_counts, v_max_v_night, out=n_tiling_all, where=in_n_plan)
            
            s_mins, n_mins = np.full(n_bins, 2.0, dtype=np.float32), np.full(n_bins, 2.0, dtype=np.float32)
            np.minimum.at(s_mins, v_bins, s_tiling_all)
            np.minimum.at(n_mins, v_bins, n_tiling_all)
            
            s_mins[~active_bins_s] = 0.
            n_mins[~active_bins_n] = 0.
            
            calculated_features['survey_min_tiling'][global_idx] = s_mins
            calculated_features['night_min_tiling'][global_idx] = n_mins
            
            global_idx += 1

    return calculated_features

import json
import pickle
import copy

def save_DES_bin_and_field_mappings(
    data_fits_fn=None,
    outdir='../data/lookups/',
    field2radec_fn = 'field2radec.json',
    field2name_fn = 'field2name.json',
):
    if data_fits_fn is None:
        data_fits_fn = '../data/fits/decam-exposures-20251211.fits'

    # Filter data
    objects_to_remove = ["guide", "DES vvds","J0'","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec", "outlier"]
    df = load_raw_data_to_dataframe(fits_path=data_fits_fn)
    df = drop_rows_in_DECam_data(
        df,
        objects_to_remove=objects_to_remove
    )
    assert len(df) > 0, "No observations found for the specified year/month/day/filter selections."
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Convert degrees to radians and define field_ids
    df['el'] = np.pi/2 - df['zd'].values
    df.loc[:, ['ra', 'dec', 'az', 'el', 'zd']] *= units.deg
    df['field_id'] = pd.factorize(df['object'])[0]
    # df['field_id'] = df['object'].map({v: k for k, v in field2name.items()})

    # field2name: Save mapping from field id to `object` name
    field2name = {fid: g.loc[:, ['object']].values.tolist()[0][0] for fid, g in df.groupby('field_id')}
    with open(outdir + field2name_fn, "w") as f:
        json.dump(field2name, f)

    # 4. field2radec: Save mapping from field id to its respective ra, dec, defined by mean of tilings
    field2radec = {int(fid): (g.loc[:, ['ra', 'dec']]).mean(axis=0).values.tolist() for fid, g in df.groupby('field_id')}
    with open(outdir + field2radec_fn, "w") as f:
        json.dump(field2radec, f)

    # Only fields with teff > .3 are counted as a visit (as defined by obztak)
        # Fields with *only* teff < .3 have had 0 visits, but should still be present in the dictionary
        # Need a separate history for train vs eval
        # In training, when calculating field tilings, we need a minimum visit of one for all exposures, even if all exposures for a specific field do not meet the teff threshold
            # Otherwise, we get a division by 0 when normalizing by the total tilings thus far
        # In evaluating, we don't want to include exposures with teff < .3 in the visit history
     
    unique_field_ids = np.unique(df['field_id'])
    u_fid_counts = np.zeros(len(unique_field_ids), dtype=int)
    valid_unique_field_ids, valid_u_fid_counts = np.unique(df['field_id'][df['teff'] > .3], return_counts=True)
    u_fid_counts[valid_unique_field_ids] = valid_u_fid_counts

    num_fields = len(unique_field_ids)

    # 5. field2nvisits
    field2nvisits_default1 = {int(fid): 1 for fid in field2radec.keys()} # make sure fields which never have a good teff are at least present in the field2nvisits mapping
    field2nvisits_default1.update({int(fid): int(c) for fid, c in zip(unique_field_ids, u_fid_counts)})
    with open(outdir + 'field2nvisits_default1.json', "w") as f:
        json.dump(field2nvisits_default1, f)

    field2nvisits_default0 = {int(fid): 0 for fid in field2radec.keys()}
    field2nvisits_default0.update({int(fid): int(c) for fid, c in zip(unique_field_ids, u_fid_counts)})
    with open(outdir + 'field2nvisits_default0.json', "w") as f:
        json.dump(field2nvisits_default0, f)
        
    # 6. night2fieldhistory: field visits each night
    night2fieldhistory = {}
    prev_visit_history = np.zeros(shape=(df['field_id'].max() + 1))
    for i, (night, grouped) in enumerate(df.groupby('night')):
        night2fieldhistory[night] = prev_visit_history.copy()
        mask_teff = grouped['teff'] > .3
        fids, cs = np.unique(grouped['field_id'][mask_teff].values, return_counts=True)
        prev_visit_history[fids] += cs
        
    with open(outdir + 'night2fieldhistory.pkl', 'wb') as f:
        pickle.dump(night2fieldhistory, f)

    # 7. field2filter: save viable filter visits per field -- #TODO will probably have to also do default0 and default1 like with field2nvisits
    field2filters = {fid: g['filter'].unique() for fid, g in df.groupby('field_id')}
    with open(outdir + 'field2filters.pkl', "wb") as f:
        pickle.dump(field2filters, f)

    # 7. night2filterhistory: filter visits per field each night
    night2filterhistory = {}
    df['filt_idx'] = df['filter'].map(FILTER2IDX)

    # 2. Initialize the running tracker
    running_counts = np.zeros(shape=(num_fields, len(FILTER2IDX)), dtype=int)

    # 3. Iterate and accumulate
    mask_teff = df['teff'] > .3

    for night, grouped in df[mask_teff].groupby('night'):
        night2filterhistory[night] = running_counts.copy()
        np.add.at(
            running_counts, 
            (grouped['field_id'].values, grouped['filt_idx'].values), 
            1
        )

    with open(outdir + 'night2filterhistory.pkl', "wb") as f:
        pickle.dump(night2filterhistory, f)
    
    return df

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

