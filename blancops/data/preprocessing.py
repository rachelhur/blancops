from pathlib import Path

import pandas as pd
from blancops.data.features.glob_features import calc_t_survey, calc_urgency
import numpy as np
from datetime import timedelta
import re


from blancops.io.fits_io import fits_to_df
from blancops.math import units

import logging
logger = logging.getLogger(__name__)

from blancops.configs.constants import FILTER2IDX
from blancops.configs.constants import TRAIN_DATA_DIR, TRAIN_DATA_PATH

def preprocess_historic_data(fits_path: str | Path | None = None, df=None, sel=None) -> pd.DataFrame:
    """Loads data from fits file, performs hard coded cuts, adds universally required columns, and converts to degrees and utc timestamp
    
    Args:
    =====
    fits_path: str | Path
        Path to fits file
    sel: 
        NOT YET IMPLEMENTED. Will be selection criteria for the dataframe
    """
    df = fits_to_df(fits_path)
    
    if sel is None:
        sel = (df['propid'] == '2012B-0001') & (df['exptime'] > 40) & (df['exptime'] < 100) & (~np.isnan(df['teff']))
    # sel &= df['exptime'] == 90 # use only 90s exposures
    df = df[sel].copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['night'] = (df['datetime'] - pd.Timedelta(hours=12)).dt.normalize()
    df['night'] = df['night'] + (timedelta(days=1) - pd.Timedelta(seconds=1))
    df = df[df['datetime'].dt.year > 2010] # There are some 1970 rows even after selecting propid
    
    df = _convert_df_to_radians(df)

    timestamps = (df['datetime'] - pd.Timestamp("1970-01-01", tz='utc')) // pd.Timedelta("1s")
    df['timestamp'] = timestamps
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df = _add_essential_cols(df)
    return df

def _convert_df_to_radians(df: pd.DataFrame, key_list: list = []):
    key_list += ['ra', 'dec', 'az', 'zd', 'ha']
    key_list = list(set(key_list))
    df.loc[:, ['ra', 'dec', 'az', 'zd', 'ha']] *= units.deg
    return df

def _add_essential_cols(df):
    df['el'] = np.pi/2 - df['zd'].values
    # df['night_idx'] = pd.factorize(df['night'])[0]
    # df['t_survey'] = calc_t_survey(df['night_idx'].values, df['night_idx'].max() + 1)
    
    # # df['t_survey'] = df['night_idx']/(df['night_idx'].max() + 1) # normalize to [0, 1]
    
    # for f in FILTER2IDX.keys():
    #     condition = (df['filter'] == f) & (df['teff'] >= 0.3)
    #     filter_counts_arr = condition.cumsum()
    #     urgency = calc_urgency(filter_counts_arr, filter_counts_arr.max(), df['night_idx'].values, df['night_idx'].max() + 1)
    #     df[f'raw_survey_progress_{f}'] = filter_counts_arr
    #     df[f'survey_progress_{f}'] = filter_counts_arr / filter_counts_arr.max()
    #     df[f'urgency_{f}'] = urgency
    return df
    
def drop_rows_in_DECam_data(df, objects_to_remove=None, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Drops nights (1) in year 1970, and (2) with specific objects (ie, SN or GW followup which are observed for long stretches of time)"""
    if objects_to_remove is None:
        objects_to_remove = ["guide", "DES vvds","J0","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec", "(outlier)"]

    df = _keep_dates(df, specific_years, specific_months, specific_days, specific_filters)
    

    # Some fields are mis-labelled - add '(outlier)' to these object names so that they are treated as separate fields
    df = _remove_object_outliers(df)
    
    # Remove specific nights according to object name
    # df = remove_specific_objects(objects_to_remove=objects_to_remove, df=df)
    pattern = '|'.join(re.escape(obj) for obj in objects_to_remove)
    mask = ~df['object'].str.contains(pattern, case=False, na=False, regex=True) & (df['object'] != '')
    
    # Filter the DataFrame
    df = df[mask]

    df.sort_values(by='timestamp').reset_index(drop=True, inplace=True)
    return df

# def _remove_outliers(df):
#     """Removes objects that have (outlier) in its object name"""
#     df = df[~df['object'].astype(str).str.contains('(outlier)', regex=False, na=False)]
#     return df

def _remove_object_outliers(df, cutoff_dist=3.5) -> pd.DataFrame:
    """Remove rows if they are outside of a certain cutoff from the object's median RA/Dec.

    Args
    ----
    df (pd.DataFrame): The dataframe with object names and RA/Dec positions.
    cutoff_dist (float): The cutoff distance in degrees.

    Returns
    -------
    cleaned_df (pd.DataFrame): The cleaned dataframe with relabelled objects removed.
    """
    indices_to_drop = []
    for obj_name, g in df.groupby('object'):
        median_ra = _get_object_median_ra(g.ra)
        median_dec = g.dec.median()

        # Calculate true on-sky distance from the robust center
        distances = _get_haversine_dist(median_ra, median_dec, g.ra.values, g.dec.values)
        
        # Mask and re-label
        mask_outlier = distances > cutoff_dist

        if np.count_nonzero(mask_outlier) > 0:
            outlier_indices = g.index[mask_outlier].values
            indices_to_drop.extend(outlier_indices)
    cleaned_df = df.drop(index=indices_to_drop)
    return cleaned_df

def _keep_dates(df, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Filters dataframe for selected years, months, days, and filters"""
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


def _get_object_median_ra(dither_ras):
    ra_rad = np.radians(dither_ras)
    
    mean_x = np.mean(np.cos(ra_rad))
    mean_y = np.mean(np.sin(ra_rad))
    
    anchor_ra = np.degrees(np.arctan2(mean_y, mean_x)) % 360
    shifted_ra = (dither_ras - anchor_ra + 180) % 360 - 180
    
    return (np.median(shifted_ra) + anchor_ra) % 360

def _get_haversine_dist(ra_center, dec_center, ra_array, dec_array):
    """Calculates great-circle distance between a center point and an array of points."""
    ra1, dec1 = np.radians(ra_center), np.radians(dec_center)
    ra2, dec2 = np.radians(ra_array), np.radians(dec_array)
    
    d_ra = ra2 - ra1
    d_dec = dec2 - dec1
    
    a = np.sin(d_dec / 2.0)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2.0)**2
    a = np.clip(a, 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(np.sqrt(a)))
