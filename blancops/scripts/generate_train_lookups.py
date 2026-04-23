import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from astropy import units

from blancops.configs.constants import TRAIN_DATA_DIR, TRAIN_DATA_PATH
from blancops.configs.enums import LookupKeys
from blancops.data.constants import FILTER2IDX
from blancops.data.lookup import LookupTables
from blancops.data.preprocessing import drop_rows_in_DECam_data, preprocess_train_df
import logging

from blancops.utils.sys_utils import setup_logger
logger = logging.getLogger(__name__)

def save_DES_bin_and_field_mappings(fits_path=None, outdir=None):
    if fits_path is None:
        fits_path = TRAIN_DATA_PATH
    if type(fits_path) is not Path:
        fits_path = Path(fits_path).resolve()
    if outdir is None:
        outdir = TRAIN_DATA_DIR
    if type(outdir) is not Path:
        outdir = Path(outdir).resolve()
    
    df = preprocess_train_df(fits_path=fits_path)
    df = drop_rows_in_DECam_data(df)
    if len(df) == 0: # Fixed the logical bug here: len(df) == 0 means no obs found
        logger.warning("No observations found for the specified year/month/day/filter selections.")
        raise ValueError
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Convert degrees to radians and define field_ids
    df['el'] = np.pi/2 - df['zd'].values
    df.loc[:, ['ra', 'dec', 'az', 'el', 'zd']] *= units.deg
    df['field_id'] = pd.factorize(df['object'])[0]

    # 1. BUILD MAPPINGS (Ensure formats match the LookupTables expectations)
    fid2name = {int(fid): g['object'].iloc[0] for fid, g in df.groupby('field_id')}
    
    # Note: LookupTables expects fid2radec and fid2filters as numpy arrays, not dicts!
    # Sorting by field_id ensures the array index matches the field_id
    grouped = df.groupby('field_id')
    fid2radec = np.array([g[['ra', 'dec']].mean(axis=0).values for _, g in sorted(grouped)])
    fid2filters = np.array([g['filter'].unique() for _, g in sorted(grouped)], dtype=object)

    # 2. BUILD COUNTS
    unique_field_ids = np.unique(df['field_id'])
    num_fields = len(unique_field_ids)
    
    valid_unique_field_ids, valid_u_fid_counts = np.unique(df['field_id'][df['teff'] > .3], return_counts=True)
    target_fid_counts = np.zeros(num_fields, dtype=int)
    target_fid_counts[valid_unique_field_ids] = valid_u_fid_counts

    # 3. BUILD HISTORY & TARGETS
    df['filt_idx'] = df['filter'].map(FILTER2IDX) 
    filt_running_counts = np.zeros(shape=(num_fields, len(FILTER2IDX)), dtype=np.int32)
    field_running_counts = np.zeros(shape=(num_fields), dtype=np.int32)
    
    night2filterhistory = {}
    night2fieldhistory = {}

    for night, night_df in df.groupby('night'):
        night2filterhistory[night] = filt_running_counts.copy()
        night2fieldhistory[night] = field_running_counts.copy()
        
        field_running_counts += np.bincount(night_df['field_id'].values, minlength=num_fields)
        np.add.at(
            filt_running_counts, 
            (night_df['field_id'].values, night_df['filt_idx'].values), 
            1
        )
    
    target_fidfilt_counts = filt_running_counts.copy()

    # 4. FILTER TARGET COUNTS
    target_filt_counts = np.zeros(len(FILTER2IDX), dtype=int)
    for f, idx in FILTER2IDX.items():
        condition = (df['filter'] == f) & (df['teff'] > 0.3)
        target_filt_counts[idx] = int(condition.sum()) # .sum() is safer than cumsum().max() for booleans

    # 5. INSTANTIATE AND SAVE
    lookups = LookupTables(
        dir=outdir,
        fields_table=df, # Or however you want to store the base fields
        fid2name=fid2name,
        fid2radec=fid2radec,
        fid2filters=fid2filters,
        target_fid_counts=target_fid_counts,
        target_fidfilt_counts=target_fidfilt_counts,
        target_filt_counts=target_filt_counts,
        night2fid_visit_hist=night2fieldhistory,
        night2fidfilt_visit_hist=night2filterhistory
    )
    
    lookups.write_to_disk(outdir)
    logger.info(f"Successfully generated and saved all lookup tables in {outdir}")
    
    return lookups

    # # Convert degrees to radians and define field_ids
    # df['el'] = np.pi/2 - df['zd'].values
    # df.loc[:, ['ra', 'dec', 'az', 'el', 'zd']] *= units.deg
    # df['field_id'] = pd.factorize(df['object'])[0]

    # # FID2NAME
    # fid2name = {fid: g.loc[:, ['object']].values.tolist()[0][0] for fid, g in df.groupby('field_id')}
    # with open(outdir / LookupKeys.FID2NAME.value, "w") as f:
    #     json.dump(fid2name, f)

    # # FID2RADEC
    # fid2radec = {int(fid): (g.loc[:, ['ra', 'dec']]).mean(axis=0).values.tolist() for fid, g in df.groupby('field_id')}
    # with open(outdir / LookupKeys.FID2RADEC.value, "w") as f:
    #     json.dump(fid2radec, f)

    # unique_field_ids = np.unique(df['field_id'])
    # u_fid_counts = np.zeros(len(unique_field_ids), dtype=int)
    # valid_unique_field_ids, valid_u_fid_counts = np.unique(df['field_id'][df['teff'] > .3], return_counts=True)
    # u_fid_counts[valid_unique_field_ids] = valid_u_fid_counts

    # num_fields = len(unique_field_ids)

    # # FID2TARGET_VISITS_TRAIN
    # fid2target_visits_default1 = {int(fid): 1 for fid in fid2radec.keys()} 
    # fid2target_visits_default1.update({int(fid): int(c) for fid, c in zip(unique_field_ids, u_fid_counts)})
    # with open(outdir / LookupKeys.TARGET_FID2VISITS_TRAIN.value, "w") as f:
    #     json.dump(fid2target_visits_default1, f)

    # # FID2TARGET_VISITS_EVAL
    # fid2target_visits_default0 = {int(fid): 0 for fid in fid2radec.keys()}
    # fid2target_visits_default0.update({int(fid): int(c) for fid, c in zip(unique_field_ids, u_fid_counts)})
    # with open(outdir / LookupKeys.TARGET_FID2VISITS_EVAL.value, "w") as f:
    #     json.dump(fid2target_visits_default0, f)

    # # FID2FILTERS
    # fiD2filters = {fid: g['filter'].unique() for fid, g in df.groupby('field_id')}
    # with open(outdir / LookupKeys.FID2FILTERS.value, "wb") as f:
    #     pickle.dump(fiD2filters, f)

    # # NIGHT2VISIT_HISTORY AND FIDFILT_TARGET_COUNTS
    # night2filterhistory = {}
    # night2fieldhistory = {}
    # df['filt_idx'] = df['filter'].map(FILTER2IDX) 

    # filt_running_counts = np.zeros(shape=(num_fields, len(FILTER2IDX)), dtype=np.int32)
    # field_running_counts = np.zeros(shape=(num_fields), dtype=np.int32)

    # for night, grouped in df.groupby('night'):
    #     night2filterhistory[night] = filt_running_counts.copy()
    #     night2fieldhistory[night] = field_running_counts.copy()

    #     fids = grouped['field_id'].values
        
    #     field_running_counts += np.bincount(fids, minlength=num_fields)
    #     np.add.at(
    #         filt_running_counts, 
    #         (grouped['field_id'].values, grouped['filt_idx'].values), 
    #         1
    #     )
    
    # fieldfilter2nvisits = filt_running_counts.copy()
    
    # with open(outdir / LookupKeys.TARGET_FIDFILT_COUNTS.value, "wb") as f:
    #     pickle.dump(fieldfilter2nvisits, f)

    # with open(outdir / LookupKeys.NIGHT2FIDFILT_VISIT_HIST.value, "wb") as f:
    #     pickle.dump(night2filterhistory, f)
            
    # with open(outdir / LookupKeys.NIGHT2FID_VISIT_HIST.value, 'wb') as f:
    #     pickle.dump(night2fieldhistory, f)

    # # FILTER_TARGET_COUNTS
    # filter_target_counts = np.empty(shape=len(FILTER2IDX), dtype=int)
    
    # for f, idx in FILTER2IDX.items():
    #     condition = (df['filter'] == f) & (df['teff'] > 0.3)
    #     cum_sum_arr = condition.cumsum()
    #     target_counts = int(cum_sum_arr.max())
    #     filter_target_counts[idx] = target_counts

    # with open(outdir / LookupKeys.TARGET_FILT_COUNTS.value, "wb") as f:
    #     pickle.dump(filter_target_counts, f)
    
    # logger.info(f"Successfully generated all lookup tables in {outdir}")
    # return df


def main():
    parser = argparse.ArgumentParser(description="Generate Train Data Lookup Tables from Raw Data")
    parser.add_argument(
        '--fits_path', 
        type=Path, 
        default=TRAIN_DATA_PATH, 
        help="Path to the raw DECam exposures FITS file"
    )
    parser.add_argument(
        '--outdir', 
        type=Path, 
        default=TRAIN_DATA_DIR, 
        help="Directory to save the generated JSON/Pickle lookup tables"
    )
    
    args = parser.parse_args()
    logger = setup_logger(save_dir=None)
    
    # Ensure output directory exists
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting lookup generation...")
    save_DES_bin_and_field_mappings(fits_path=args.fits_path, outdir=args.outdir)
    logger.info(f"Successfully generated all lookup tables in {args.outdir}")

if __name__ == "__main__":
    main()