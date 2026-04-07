import pandas as pd
from blancops.data_processing.features import calculate_t_survey, calculate_urgency
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

import warnings
import logging
logger = logging.getLogger(__name__)

from blancops.data_processing.constants import FILTER2IDX

import json
import pickle

def save_DES_bin_and_field_mappings(fits_path, outdir):
    if type(outdir) is not Path:
        outdir = Path(outdir).resolve()
    if type(fits_path) is not Path:
        fits_path = Path(fits_path).resolve()
    workspace = get_workspace_dir()
    with open(workspace / "configs" / "global_config.json", "r") as f:
        gcfg = json.load(f)

    # Filter data
    objects_to_remove = ["guide", "DES vvds","J0'","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec", "outlier"]
    df = load_raw_data_to_dataframe(fits_path=fits_path)
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
    with open(outdir / gcfg['files']['FIELD2NAME'], "w") as f:
        json.dump(field2name, f)

    # field2radec: Save mapping from field id to its respective ra, dec, defined by mean of tilings
    field2radec = {int(fid): (g.loc[:, ['ra', 'dec']]).mean(axis=0).values.tolist() for fid, g in df.groupby('field_id')}
    with open(outdir / gcfg['files']['FIELD2RADEC'], "w") as f:
        json.dump(field2radec, f)

    unique_field_ids = np.unique(df['field_id'])
    u_fid_counts = np.zeros(len(unique_field_ids), dtype=int)
    valid_unique_field_ids, valid_u_fid_counts = np.unique(df['field_id'][df['teff'] > .3], return_counts=True)
    u_fid_counts[valid_unique_field_ids] = valid_u_fid_counts

    num_fields = len(unique_field_ids)

    # 5. field2nvisits
    field2nvisits_default1 = {int(fid): 1 for fid in field2radec.keys()} # make sure fields which never have a good teff are at least present in the field2nvisits mapping
    field2nvisits_default1.update({int(fid): int(c) for fid, c in zip(unique_field_ids, u_fid_counts)})
    with open(outdir / gcfg['files']['FIELD2MAXVISITS_TRAIN'], "w") as f:
        json.dump(field2nvisits_default1, f)

    field2nvisits_default0 = {int(fid): 0 for fid in field2radec.keys()}
    field2nvisits_default0.update({int(fid): int(c) for fid, c in zip(unique_field_ids, u_fid_counts)})
    with open(outdir / gcfg['files']['FIELD2MAXVISITS_EVAL'], "w") as f:
        json.dump(field2nvisits_default0, f)

    # new_df.to_json(outdir + 'field_lookup.json', indent=2, orient='index')

    # 7. field2filter: save viable filter visits per field -- #TODO will probably have to also do default0 and default1 like with field2nvisits
    field2filters = {fid: g['filter'].unique() for fid, g in df.groupby('field_id')}
    with open(outdir / gcfg['files']['FIELD2FILTERS'], "wb") as f:
        pickle.dump(field2filters, f)

    # 7. night2filterhistory: filter visits per field each night
    night2filterhistory = {}
    night2fieldhistory = {}
    df['filt_idx'] = df['filter'].map(FILTER2IDX) #.fillna(-1)

    filt_running_counts = np.zeros(shape=(num_fields, len(FILTER2IDX)), dtype=np.int32)
    field_running_counts = np.zeros(shape=(num_fields), dtype=np.int32)
    # mask_teff = df['teff'] > .3

    for night, grouped in df.groupby('night'):
        night2filterhistory[night] = filt_running_counts.copy()
        night2fieldhistory[night] = field_running_counts.copy()

        fids = grouped['field_id'].values
        
        field_running_counts += np.bincount(fids, minlength=num_fields)
        np.add.at(
            filt_running_counts, 
            (grouped['field_id'].values, grouped['filt_idx'].values), 
            1
        )
    
    fieldfilter2nvisits = filt_running_counts.copy()
    
    with open(outdir / gcfg['files']['FIELDFILTER2MAXVISITS'], "wb") as f:
        pickle.dump(fieldfilter2nvisits, f)

    with open(outdir / gcfg['files']['NIGHT2FILTERVISITS'], "wb") as f:
        pickle.dump(night2filterhistory, f)
            
    with open(outdir / gcfg['files']['NIGHT2FIELDVISITS'], 'wb') as f:
        pickle.dump(night2fieldhistory, f)

    filter_target_counts = np.empty(shape=len(FILTER2IDX), dtype=int)
    # night2survey_progress = {}
    
    for f, idx in FILTER2IDX.items():
        # col_name = f'survey_progress_{f}'
        condition = (df['filter'] == f) & (df['teff'] > 0.3)
        cum_sum_arr = condition.cumsum()
        target_counts = int(cum_sum_arr.max())
        # df['raw_' + col_name] = cum_sum_arr
        filter_target_counts[idx] = target_counts
        # df[col_name] = cum_sum_arr / cum_sum_arr.max()
        # df[f'urgency_{f}'] = np.clip((1 - df[col_name].values) / (1 - df['t_survey'].values  + 1e-9), a_min=0.01, a_max=100.0)
    with open(outdir / gcfg['files']['FILTER_TARGET_COUNTS'], "wb") as f:
        pickle.dump(filter_target_counts, f)
    
    # for night, grouped in df.groupby('night'):
    #     night2survey_progress[night] = grouped[[f'survey_progress_{f}' for f in FILTER2IDX.keys()]].values   
         
    # raw_survey_progress_history = df[[f'survey_progress_{f}' for f in FILTER2IDX.keys()]].values
    # with open(outdir / gcfg['files']['TARGET_COUNTS_PER_FILTER'], "wb") as f:
    #     pickle.dump(filter_target_counts, f)
    # with open(outdir / gcfg['files']['SURVEY_PROGRESS_HISTORY_PER_FILTER'], "wb") as f:
    #     pickle.dump(filter_target_counts, f)
        
    return df

def load_raw_data_to_dataframe(fits_path, add_survey_progress_cols=True):
    d = fitsio.read(fits_path)
    df = pd.DataFrame(d.astype(d.dtype.newbyteorder('='))) # Big-endian/little-endian error

    sel = (df['propid'] == '2012B-0001') & (df['exptime'] > 40) & (df['exptime'] < 100) & (~np.isnan(df['teff']))
    df = df[sel].copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['night'] = (df['datetime'] - pd.Timedelta(hours=12)).dt.normalize()
    df['night'] = df['night'] + (timedelta(days=1) - pd.Timedelta(seconds=1))
    df = df[df['datetime'].dt.year > 2010] # There are some 1970 rows even after selecting propid

    timestamps = (df['datetime'] - pd.Timestamp("1970-01-01", tz='utc')) // pd.Timedelta("1s")
    df['timestamp'] = timestamps
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    if add_survey_progress_cols:
        df = add_cols_to_raw_dataframe(df)
    return df

def add_cols_to_raw_dataframe(df):
    df['night_idx'] = pd.factorize(df['night'])[0]
    df['t_survey'] = calculate_t_survey(df['night_idx'].values, df['night_idx'].max() + 1)
    # df['t_survey'] = df['night_idx']/(df['night_idx'].max() + 1) # normalize to [0, 1]
    
    for f in FILTER2IDX.keys():
        condition = (df['filter'] == f) & (df['teff'] >= 0.3)
        filter_counts_arr = condition.cumsum()
        urgency = calculate_urgency(filter_counts_arr, filter_counts_arr.max(), df['night_idx'].values, df['night_idx'].max() + 1)
        df[f'raw_survey_progress_{f}'] = filter_counts_arr
        df[f'survey_progress_{f}'] = filter_counts_arr / filter_counts_arr.max()
        df[f'urgency_{f}'] = urgency
    return df
    
def drop_rows_in_DECam_data(df, objects_to_remove=None, specific_years=None, specific_months=None, specific_days=None, specific_filters=None):
    """Drops nights (1) in year 1970, and (2) with specific objects (ie, SN or GW followup which are observed for long stretches of time)"""
    if objects_to_remove is None:
        objects_to_remove = ["guide", "DES vvds","J0'","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec", "(outlier)"]

    df = remove_dates(df, specific_years, specific_months, specific_days, specific_filters)
    
    # Remove specific nights according to object name
    # df = remove_specific_objects(objects_to_remove=objects_to_remove, df=df)
    pattern = '|'.join(objects_to_remove)
    mask = ~df['object'].str.contains(pattern, case=False, na=False, regex=True)

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

# def expand_feature_names_for_cyclic_norm(feature_names, cyclical_feature_names):
    # # periodic vars first
    # feature_names = [
    #     element 
    #     for feat_name in feature_names
    #     for element in ([feat_name + '_cos', feat_name + '_sin'] 
    #                     if any((string == feat_name) or (f"_{string}" in feat_name) for string in cyclical_feature_names)
    #                     else [feat_name])
    #     ]
    # return feature_names

def expand_feature_names_for_cyclic_norm(feature_names, cyclical_feature_names):
    feature_names_out = []
    for feat_name in feature_names:
        is_rel_feat = feat_name.startswith('rel_')
        is_cyclic = any((feat_name == cyc_feat) or feat_name.endswith(f"_{cyc_feat}") for cyc_feat in cyclical_feature_names)
        
        if is_cyclic and not is_rel_feat:
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
