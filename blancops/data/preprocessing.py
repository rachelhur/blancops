from pathlib import Path

import pandas as pd
import numpy as np
from datetime import timedelta
import re


from blancops.data.features.glob_features import get_night_boundaries
from blancops.math import units

from blancops.configs.constants import TRAIN_DATA_DIR, TRAIN_DATA_PATH
from blancops.configs.constants import FILTER2IDX
from blancops.data.lookup_tables import LookupTables, TrainLookupTables
from blancops.io.fits_io import fits_to_df
from blancops.math import units

import logging
logger = logging.getLogger(__name__)

from blancops.configs.constants import FILTER2IDX
from blancops.configs.constants import TRAIN_DATA_DIR, TRAIN_DATA_PATH

_VALID_TEFF_THRESHOLD = 0.3

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
    df = _add_essential_cols(df)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
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
    
def remove_undesired_dates_and_objects(df, objects_to_remove=None, years_keep=None, months_keep=None, days_keep=None, filters_keep=None):
    """Drops nights (1) not within the years, months, and days specified, and (2) with specific objects (ie, SN or GW followup which are observed for long stretches of time)"""
    if objects_to_remove is None:
        objects_to_remove = ["guide", "DES vvds","J0","gwh","DESGW","Alhambra-8","cosmos","COSMOS hex","TMO","LDS","WD0","DES supernova hex","NGC","ec", "(outlier)"]

    df = _keep_dates(df, years_keep, months_keep, days_keep, filters_keep)

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
    if specific_years:
        df = df[df['night'].dt.year.isin(specific_years)]
        assert not df.empty, f"Years {specific_years} do not exist in dataset"
    if specific_months:
        df = df[df['night'].dt.month.isin(specific_months)]
        assert not df.empty, f"Months {specific_months} do not exist in years {specific_years}"
    if specific_days:
        df = df[df['night'].dt.day.isin(specific_days)]
        assert not df.empty, f"Days {specific_days} do not exist in months {specific_months}, and years {specific_years}"
    if specific_filters:
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

def build_DES_lookups(fits_path=None, outdir=None):
    fits_path = Path(fits_path or TRAIN_DATA_PATH).resolve()
    outdir = Path(outdir or TRAIN_DATA_DIR).resolve()
    
    df = preprocess_historic_data(fits_path=fits_path)
    df = remove_undesired_dates_and_objects(df)
    if len(df) == 0: # Fixed the logical bug here: len(df) == 0 means no obs found
        logger.warning("No observations found for the specified year/month/day/filter selections.")
        raise ValueError
    
    # field_id is 0..N-1 contiguous by construction (pd.factorize).
    df['field_id'] = pd.factorize(df['object'])[0]
    df["filt_idx"] = df["filter"].map(FILTER2IDX)

    num_fields = df["field_id"].nunique()
    nfilters = len(FILTER2IDX)
    
    
    # Quality threshold — only targets and per-night history derive
    # from this set, so completion checks and seeded state agree.
    valid_df = df[df["teff"] > _VALID_TEFF_THRESHOLD].copy()
    if len(valid_df) == 0:
        raise ValueError(
            f"No observations with teff > {_VALID_TEFF_THRESHOLD} in "
            f"{fits_path}; check input data quality."
        )
    
    # ---------- Per-field DataFrame ----------
    # One row per field_id. Aggregate over the FULL df (not teff-
    # filtered) so even fields with no valid observations still
    # appear; they'll have a target row of zeros and be excluded from
    # masks. Since field_id was factorized from `object`, all rows in
    # a group share the same name (taking iloc[0] is unambiguous).
    fields_rows = []
    for field_id, g in df.groupby("field_id"):
        fields_rows.append({
            "field_id": int(field_id),
            "object": g["object"].iloc[0],
            "ra": float(g["ra"].mean()),
            "dec": float(g["dec"].mean()),
        })
    fields = pd.DataFrame(fields_rows).set_index("field_id").sort_index()
    fields.index.name = "field_id"
    logger.info(" [+] Constructed Fields Lookup")

    # ---------- Per-(field, filter) matrices ----------
    # target_fidfilt_counts: total VALID observations per (field, filter).
    target_fidfilt_counts = np.zeros((num_fields, nfilters), dtype=np.int32)
    np.add.at(
        target_fidfilt_counts,
        (valid_df["field_id"].values, valid_df["filt_idx"].values),
        1,
    )
    logger.info(" [+] Constructed Target Counts Lookup")

    # fidfilt_exptime: most-common exptime per (field, filter).
    fidfilt_exptime = (
        df.pivot_table(
            index="field_id", columns="filt_idx", values="exptime",
            aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else 0,
        )
        .reindex(index=range(num_fields), columns=range(nfilters), fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    logger.info(" [+] Constructed Exposure Time Lookup")


    # ---------- Per-night visit history & last-visit timestamps ----------
    # Running counts and most-recent-visit timestamps, snapshotted at the
    # START of each night. Iterate over ALL nights in df (not just nights
    # with valid observations) so every observed night has a seedable
    # state, but only VALID observations contribute to the running totals
    # — keeping history consistent with the target-completion semantics.
    #
    # last_visit uses NaN as "no recorded visit yet"; we update with
    # `np.fmax` so existing recorded values win over NaN as new
    # observations come in. (np.maximum would propagate NaN.)
    field_running = np.zeros(num_fields, dtype=np.int32)
    fidfilt_running = np.zeros((num_fields, nfilters), dtype=np.int32)
    field_last_visit_ts = np.full(num_fields, np.nan, dtype=np.float64)
    fidfilt_last_visit_ts = np.full((num_fields, nfilters), np.nan, dtype=np.float64)
    field_last_visit_ot = np.full(num_fields, np.nan, dtype=np.float64)
    fidfilt_last_visit_ot = np.full((num_fields, nfilters), np.nan, dtype=np.float64)

    night2fid_visit_hist = {}
    night2fidfilt_visit_hist = {}
    night2fid_last_visit_ts = {}
    night2fidfilt_last_visit_ts = {}
    night2fid_last_visit_ot = {}
    night2fidfilt_last_visit_ot = {}
    night2ot_clock_seconds = {}     # night -> OT(sunset_n)

    cum_ot = 0.0
        
    for i, (night, night_df) in enumerate(df.groupby("night")):
        sunset_ts, sunrise_ts = get_night_boundaries(night, sun_el_limit=-10)
        night2ot_clock_seconds[night] = cum_ot
        night_dur = sunrise_ts - sunset_ts
        # night2idx[night] = i

        # Snapshot start-of-night state BEFORE adding this night's
        # contributions. Matches existing visit_hist semantics so
        # downstream consumers can pair (visit_hist[n], last_visit[n])
        # safely.
        night2fid_visit_hist[night] = field_running.copy()
        night2fidfilt_visit_hist[night] = fidfilt_running.copy()
        night2fid_last_visit_ts[night] = field_last_visit_ts.copy()
        night2fidfilt_last_visit_ts[night] = fidfilt_last_visit_ts.copy()
        night2fid_last_visit_ot[night] = field_last_visit_ot.copy()
        night2fidfilt_last_visit_ot[night] = fidfilt_last_visit_ot.copy()

 
        valid_night = night_df[night_df["teff"] > _VALID_TEFF_THRESHOLD]
        if len(valid_night):
            # Visit counts
            field_running += np.bincount(
                valid_night["field_id"].values, minlength=num_fields
            )
            np.add.at(
                fidfilt_running,
                (valid_night["field_id"].values, valid_night["filt_idx"].values),
                1,
            )

            # Last-visit timestamps: per (field) and per (field, filter),
            # we want the MAXIMUM timestamp across this night's valid
            # observations for that key.
            #
            # Per-field: groupby max is O(n log n) but n is small per
            # night; cleaner than np.maximum.at + temp array.
            fid_max = valid_night.groupby("field_id")["timestamp"].max()
            fid_ids = fid_max.index.to_numpy()
            fid_ts = fid_max.to_numpy(dtype=np.float64)
            # np.fmax: NaN-aware max — incoming value wins over existing NaN.
            field_last_visit_ts[fid_ids] = np.fmax(
                field_last_visit_ts[fid_ids], fid_ts
            )
 
            # Per (field, filter): same idea, 2-D index.
            ff_max = valid_night.groupby(["field_id", "filt_idx"])["timestamp"].max()
            if len(ff_max):
                ff_keys = np.array(ff_max.index.tolist(), dtype=np.int64)
                ff_ts = ff_max.to_numpy(dtype=np.float64)
                rows, cols = ff_keys[:, 0], ff_keys[:, 1]
                fidfilt_last_visit_ts[rows, cols] = np.fmax(
                    fidfilt_last_visit_ts[rows, cols], ff_ts
                )
                
            # Last-visit time: in unit of observing time seconds
            #       is per (field) and per (field, filter),
            valid_night = valid_night.assign(
                ot=cum_ot + (valid_night["timestamp"] - sunset_ts)
            )
            fid_max = valid_night.groupby("field_id")["ot"].max()
            fid_ids = fid_max.index.to_numpy()
            fid_ot = fid_max.to_numpy(dtype=np.float64)
            # np.fmax: NaN-aware max — incoming value wins over existing NaN.
            field_last_visit_ot[fid_ids] = np.fmax(
                field_last_visit_ot[fid_ids], fid_ot
            )
 
            # Per (field, filter): same idea, 2-D index.
            ff_max = valid_night.groupby(["field_id", "filt_idx"])["ot"].max()
            if len(ff_max):
                ff_keys = np.array(ff_max.index.tolist(), dtype=np.int64)
                ff_ot = ff_max.to_numpy(dtype=np.float64)
                rows, cols = ff_keys[:, 0], ff_keys[:, 1]
                fidfilt_last_visit_ot[rows, cols] = np.fmax(
                    fidfilt_last_visit_ot[rows, cols], ff_ot
                )

        cum_ot += night_dur
        
    total_observing_seconds = cum_ot

    logger.info(" [+] Constructed start-of-night 'Snapshots' Lookup (required for history-based feature construction) ")

    # ---------- Construct LookupTable and save to disk ----------
    lookups = TrainLookupTables(
        fields=fields,
        target_fidfilt_counts=target_fidfilt_counts,
        fidfilt_exptime=fidfilt_exptime,
        dir=outdir,
        night2fid_visit_hist=night2fid_visit_hist,
        night2fidfilt_visit_hist=night2fidfilt_visit_hist,
        night2fid_last_visit_ts=night2fid_last_visit_ts,
        night2fidfilt_last_visit_ts=night2fidfilt_last_visit_ts,
        night2fid_last_visit_ot=night2fid_last_visit_ot,
        night2fidfilt_last_visit_ot=night2fidfilt_last_visit_ot,
        night2ot_clock_seconds=night2ot_clock_seconds,
        total_ot_sec=total_observing_seconds,
    )
    lookups.write_to_disk(outdir)
    logger.info(f" [+] Successfully generated all lookup tables in {outdir}")

    return lookups
    