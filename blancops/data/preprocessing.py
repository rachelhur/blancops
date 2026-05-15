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

def preprocess_train_df(fits_path, add_survey_progress_cols=True) -> pd.DataFrame:
    """Loads train data from fits file, performs hard coded cuts, adds universally required columns, converts to degrees and utc timestamp"""
    df = fits_to_df(fits_path)
    
    sel = (df['propid'] == '2012B-0001') & (df['exptime'] > 40) & (df['exptime'] < 100) & (~np.isnan(df['teff']))
    # sel &= df['exptime'] == 90 # use only 90s exposures
    df = df[sel].copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['night'] = (df['datetime'] - pd.Timedelta(hours=12)).dt.normalize()
    df['night'] = df['night'] + (timedelta(days=1) - pd.Timedelta(seconds=1))
    df = df[df['datetime'].dt.year > 2010] # There are some 1970 rows even after selecting propid
    
    df = _convert_df_to_deg(df)

    timestamps = (df['datetime'] - pd.Timestamp("1970-01-01", tz='utc')) // pd.Timedelta("1s")
    df['timestamp'] = timestamps
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    if add_survey_progress_cols:
        df = _add_cols_to_raw_dataframe(df)
    return df

def _convert_df_to_deg(df: pd.DataFrame, key_list: list = []):
    key_list += ['ra', 'dec', 'az', 'zd', 'ha']
    key_list = list(set(key_list))
    df.loc[:, ['ra', 'dec', 'az', 'zd', 'ha']] *= units.deg
    return df

def _add_cols_to_raw_dataframe(df):
    df['night_idx'] = pd.factorize(df['night'])[0]
    df['t_survey'] = calc_t_survey(df['night_idx'].values, df['night_idx'].max() + 1)
    df['el'] = np.pi/2 - df['zd'].values
    
    # df['t_survey'] = df['night_idx']/(df['night_idx'].max() + 1) # normalize to [0, 1]
    
    for f in FILTER2IDX.keys():
        condition = (df['filter'] == f) & (df['teff'] >= 0.3)
        filter_counts_arr = condition.cumsum()
        urgency = calc_urgency(filter_counts_arr, filter_counts_arr.max(), df['night_idx'].values, df['night_idx'].max() + 1)
        df[f'raw_survey_progress_{f}'] = filter_counts_arr
        df[f'survey_progress_{f}'] = filter_counts_arr / filter_counts_arr.max()
        df[f'urgency_{f}'] = urgency
    return df
    
def drop_rows_in_DECam_data(df, objects_to_remove=None, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Drops nights (1) in year 1970, and (2) with specific objects (ie, SN or GW followup which are observed for long stretches of time)"""
    if objects_to_remove is None:
        objects_to_remove = ["guide", "DES vvds","J0","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec", "(outlier)"]

    df = _keep_dates(df, specific_years, specific_months, specific_days, specific_filters)
    

    # Some fields are mis-labelled - add '(outlier)' to these object names so that they are treated as separate fields
    df = _relabel_mislabelled_objects(df)
    df = _remove_outliers(df)
    
    # Remove specific nights according to object name
    # df = remove_specific_objects(objects_to_remove=objects_to_remove, df=df)
    pattern = '|'.join(re.escape(obj) for obj in objects_to_remove)
    mask = ~df['object'].str.contains(pattern, case=False, na=False, regex=True) & (df['object'] != '')
    
    # Filter the DataFrame
    df = df[mask]

    df.sort_values(by='timestamp').reset_index(drop=True, inplace=True)
    return df

def _remove_outliers(df):
    """Removes objects that have (outlier) in its object name"""
    df = df[~df['object'].astype(str).str.contains('(outlier)', regex=False, na=False)]
    return df

# def _remove_specific_objects(df, objects_to_remove):
#     nights_with_special_fields = set()
#     for i, spec_obj in enumerate(objects_to_remove):
#         for night, subdf in df.groupby('night'):
#             if any(spec_obj in obj_name for obj_name in subdf['object'].values) or any(subdf['object'] == ""):
#                 nights_with_special_fields.add(night)
#     nights_to_remove_mask = df['night'].isin(nights_with_special_fields)

#     df = df[~nights_to_remove_mask]
#     assert not df.empty, "All nights have special fields"
#     return df

def _relabel_mislabelled_objects(df):
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

def _keep_dates(df, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Filters dataframe for selected years, months, days, and filters"""
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