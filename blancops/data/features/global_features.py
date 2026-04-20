import pandas as pd
import torch
from tqdm import tqdm
from blancops.data.constants import ZENITH_BIN_NUM, np
from blancops.data.features.normalizations import np
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.math import units
from blancops.ephemerides import ephemerides
from datetime import timezone, timedelta
import numpy as np
import ephem
from astropy.time import Time
from blancops.data.constants import *

import logging
logger = logging.getLogger(__name__)

def calc_t_survey(survey_night_indices, survey_nights_max):
    t_survey = survey_night_indices / survey_nights_max
    if type(t_survey) == torch.Tensor or type(t_survey) == np.ndarray:
        assert t_survey.min() >= 0 and t_survey.max() <= 1, "t_survey should be between 0 and 1"
    return t_survey    

def calc_urgency(filter_counts_arr, filter_counts_max, survey_night_indices, survey_nights_max):
    survey_progress = filter_counts_arr / filter_counts_max
    t_survey = calc_t_survey(survey_night_indices, survey_nights_max)
    urgency = np.clip((1 - survey_progress) / (1 - t_survey + 1e-9), a_min=0.01, a_max=100.0)
    return urgency

def calc_twilight(timestamp, event_type='set', horizon='-10', buffer_in_seconds=10):
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
    if event_type == 'rise':
        dt_utc -= timedelta(seconds=buffer_in_seconds)
    else:
        dt_utc += timedelta(seconds=buffer_in_seconds)
    return dt_utc.timestamp()

def calculate_sun_rise_and_set(df):
    rise_times = df.groupby('night').apply(calc_twilight, event_type='rise').values
    set_times = df.groupby('night').apply(calc_twilight, event_type='set').values
    return rise_times, set_times
    
def calculate_sun_rise_and_set_azel(df):
    rise_times, set_times = calculate_sun_rise_and_set(df)
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

def calc_sun_and_moon_positions(time):
    sun_radec = ephemerides.get_source_ra_dec('sun', time=time)
    sun_azel = ephemerides.equatorial_to_topographic(ra=sun_radec[0], dec=sun_radec[1], time=time)
    moon_radec = ephemerides.get_source_ra_dec('moon', time=time)
    moon_azel = ephemerides.equatorial_to_topographic(ra=moon_radec[0], dec=moon_radec[1], time=time)
    return sun_radec, sun_azel, moon_radec, moon_azel

def calc_moon_phase(time):
    observer = ephemerides.blanco_observer(time=time)
    moon = ephem.Moon()
    moon.compute(observer)
    moon_phase = moon.phase / 100
    return np.float32(moon_phase)

def calc_lst(datetime_np64):
    t_arr = Time(datetime_np64, format='datetime64', scale='utc')
    lst_obj = t_arr.sidereal_time('apparent', longitude="-70:48:23.49")  # Blanco longitude
    return lst_obj.radian, lst_obj.hour # for debugging

def calculate_global_features(df, field2name, hpGrid, 
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
    
    # 2b. Back fill for zenith states (assume no change between zenith state and next state)
    df = _backfill_zenith_states(df)
    
    # 3. Vectorized LST
    if 'lst' in base_global_feature_names:
        df['lst'], df['lst_hours'] = calc_lst(df['datetime'].values)

    # 4. Get time dependent features (sun and moon pos)
    timestamps = df['timestamp'].values
    sun_ras, sun_decs, sun_azs, sun_els = [], [], [], []
    moon_ras, moon_decs, moon_azs, moon_els = [], [], [], []
    moon_phases = []
    
    for time in tqdm(timestamps, total=len(timestamps), desc='Calculating sun and moon ra/dec and az/el'):
        sun_radec, sun_azel, moon_radec, moon_azel = calc_sun_and_moon_positions(time=time)
        sun_ras.append(sun_radec[0]); sun_decs.append(sun_radec[1]); sun_azs.append(sun_azel[0]); sun_els.append(sun_azel[1])
        moon_ras.append(moon_radec[0]); moon_decs.append(moon_radec[1]); moon_azs.append(moon_azel[0]); moon_els.append(moon_azel[1])
        moon_phases.append(calc_moon_phase(time=time))

    df['sun_ra'], df['sun_dec'], df['sun_az'], df['sun_el'] = sun_ras, sun_decs, sun_azs, sun_els
    df['moon_ra'], df['moon_dec'], df['moon_az'], df['moon_el'] = moon_ras, moon_decs, moon_azs, moon_els
    df['moon_phase'] = moon_phases
        
    # Use first and last observation in night of offline dataset as time start and end

    
    ra_arr = df['ra'].values
    dec_arr = df['dec'].values
    
    # Using nautical twilight for time start and end
    df['t_night'] = df.groupby('night')['timestamp'].transform(normalize_times)
    assert all(df['t_night'].values > 0) and all(df['t_night'].values < 1), "Time fractions should be between 0 and 1"  

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
        df.loc[df['object'] == 'zenith', "bin"] = ZENITH_BIN_NUM
        df.loc[df['object'] == 'zenith', "field_id"] = ZENITH_FIELD_ID # Need to re-assign zenith field_id bc df['object'].map(...) above will assign zenith the field_id of the field with object name 'zenith', but this field is mis-labelled and not actually the zenith field. #TODO should fix this in field2name

    # Add other feature columns for those not present in dataframe
    sky_bright_done = False
    for feat_name in base_global_feature_names:
        if feat_name not in df.columns:
            if feat_name == 'filter_wave':
                df['filter_wave'] = df['filter'].map(FILTER2WAVE)
                df['filter_wave'] = df['filter_wave'].fillna(ZENITH_WAVELENGTH) / FILTERWAVENORM # zenith "filter" set to 0, then normalize
            elif feat_name == 'filter_idx':
                df['filter_idx'] = df['filter'].map(FILTER2IDX)
            elif feat_name.startswith('is_filter_'):
                filt_str = feat_name.split('_')[-1]
                df[feat_name] = (df['filter'] == filt_str).astype(np.float32)
            elif 'sky_brightness' in feat_name:
                if not sky_bright_done:
                    for filt in FILTER2WAVE.keys():
                        if filt != ZENITH_FILTER:
                            df[f'sky_brightness_{filt}'] = estimate_sky_brightness(time=timestamps, ra=ra_arr, dec=dec_arr, band=filt)
                    sky_bright_done = True
            else:
                raise NotImplementedError(f"Feature {feat_name} not found in dataframe columns. Check spelling. Or, this feature is not yet implemented.")

    # Normalize periodic features here and add as df cols
    if do_cyclical_norm:
        for feat_name in base_global_feature_names:
            if any(feat_name.endswith(string) for string in cyclical_feature_names):
            # if any(string in feat_name and 'frac' not in feat_name and 'bin' not in feat_name for string in cyclical_feature_names):
                logger.info(f'Applying cyclical norm to {feat_name}')
                df[f'{feat_name}_cos'] = np.cos(df[feat_name].values)
                df[f'{feat_name}_sin'] = np.sin(df[feat_name].values)

    # Ensure all data are 32-bit precision before training
    for bin_str, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]): 
        cols = df.select_dtypes(include=[bin_str]).columns
        df[cols] = df[cols].astype(np_bit)
    return df

def get_zenith_features(original_df):
    """
    Constructs dataframe with zenith features for each night in the original_df.
    Assumes zenith starts 10 seconds before the first observation.
    """
    zenith_datetimes = original_df.groupby('night').head(1).datetime - pd.Timedelta(seconds=20)
    zenith_timestamps = (zenith_datetimes - pd.Timestamp("1970-01-01", tz='utc')) // pd.Timedelta("1s")
    zenith_datetimes = zenith_datetimes.values
    # zenith_timestamps = zenith_datetimes.astype(np.int64) // 10 ** 9
    # df['timestamp'] = timestamps
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
    zenith_df['az'] = ZENITH_AZ
    zenith_df['el'] = ZENITH_EL
    zenith_df['airmass'] = ZENITH_AIRMASS
    zenith_df['zd'] = ZENITH_ZD
    zenith_df['ha'] = ZENITH_HA
    zenith_df['object'] = ZENITH_OBJECT
    zenith_df['field_id'] = ZENITH_FIELD_ID
    zenith_df['filter'] = ZENITH_FILTER
    zenith_df['datetime'] = pd.to_datetime(zenith_df['datetime'], utc=True)
    zenith_df['night'] = pd.to_datetime(zenith_df['night'], utc=True)

    return zenith_df

def _backfill_zenith_states(df):
    df['fwhm'] = df.groupby('night')['fwhm'].bfill()
    df['night_idx'] = df.groupby('night')['night_idx'].bfill()
    df['t_survey'] = df.groupby('night')['t_survey'].bfill()
    for f in FILTER2IDX.keys():
        df[f'raw_survey_progress_{f}'] = df.groupby('night')[f'raw_survey_progress_{f}'].bfill()
        df[f'survey_progress_{f}'] = df.groupby('night')[f'survey_progress_{f}'].bfill()
        df[f'urgency_{f}'] = df.groupby('night')[f'urgency_{f}'].bfill()
    return df


def normalize_times(time_series):
    sunset_ts = calc_twilight(time_series.median(), event_type='set')
    sunrise_ts = calc_twilight(time_series.median(), event_type='rise')
    total_time = sunrise_ts - sunset_ts

    time_series = (time_series - sunset_ts) / total_time
    assert all(time_series.values > 0) and all(time_series.values < 1), "Time fractions should be between 0 and 1"
    return time_series


def calc_inst_teff_rate(df, next_state_idxs):
    next_state_df = df.iloc[next_state_idxs]
    current_state_df = df.iloc[next_state_idxs-1]
    t_diff = next_state_df['timestamp'].values - current_state_df['timestamp'].values
    teff_no_zen = next_state_df[['teff']].values[:, 0]

    teff_inst_rate = teff_no_zen / t_diff
    min_rate = np.min(teff_inst_rate)
    max_rate = np.max(teff_inst_rate)
    rewards = (teff_inst_rate - min_rate)/max_rate
    return rewards


def time_until_set():
    pass