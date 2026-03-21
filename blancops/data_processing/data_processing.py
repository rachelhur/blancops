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

    # new_df = df.groupby(['field_id']).agg(
    #     ra=('ra', 'mean'),
    #     dec=('dec', 'mean'),
    #     n_visits=('ra', 'count')           # Replaces your field2nvisits logic
    # ).reset_index()

    # ra_arr, dec_arr = new_df['ra'].values, new_df['dec'].values
    # field2radec = {str(fid): (ra_arr[fid], dec_arr[fid]) for fid in new_df['field_id'].values}
    # with open(outdir + 'field2radec.json', 'w') as f:
    #     json.dump(field2radec, f, indent=2)

    # 4. field2radec: Save mapping from field id to its respective ra, dec, defined by mean of tilings
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

    return df

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
    # utc = pd.to_datetime(df['datetime'], utc=True)
    # timestamps = (utc.astype('int64') // 10**9).values
    # timestamps = [int(t.timestamp()) for t in pd.to_datetime(df['datetime'], utc=True)]
    timestamps = (df['datetime'] - pd.Timestamp("1970-01-01", tz='utc')) // pd.Timedelta("1s")
    df['timestamp'] = timestamps
    df = df.sort_values(by='timestamp').reset_index(drop=True)
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
