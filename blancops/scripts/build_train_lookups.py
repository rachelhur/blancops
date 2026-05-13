import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from blancops.math import units

from blancops.configs.constants import TRAIN_DATA_DIR, TRAIN_DATA_PATH
from blancops.configs.constants import FILTER2IDX
from blancops.data.lookup_tables import LookupTables
from blancops.data.preprocessing import drop_rows_in_DECam_data, preprocess_train_df
import logging
logger = logging.getLogger(__name__)

from blancops.io.logger_utils import setup_logger

def save_DES_bin_and_field_mappings(fits_path=None, outdir=None):
    fits_path = Path(fits_path or TRAIN_DATA_PATH).resolve()
    outdir = Path(outdir or TRAIN_DATA_DIR).resolve()
    
    df = preprocess_train_df(fits_path=fits_path)
    df = drop_rows_in_DECam_data(df)
    if len(df) == 0: # Fixed the logical bug here: len(df) == 0 means no obs found
        logger.warning("No observations found for the specified year/month/day/filter selections.")
        raise ValueError
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # field_id is 0..N-1 contiguous by construction (pd.factorize).
    df['field_id'] = pd.factorize(df['object'])[0]
    df["filt_idx"] = df["filter"].map(FILTER2IDX)

    num_fields = int(df["field_id"].max()) + 1
    nfilters = len(FILTER2IDX)
    
    
    # Quality threshold — only targets and per-night history derive
    # from this set, so completion checks and seeded state agree.
    valid_df = df[df["teff"] > 0.3].copy()
    if len(valid_df) == 0:
        raise ValueError(
            f"No observations with teff > 0.3 in {fits_path}; "
            f"check input data quality."
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

    # ---------- Per-(field, filter) matrices ----------
    # target_fidfilt_counts: total VALID observations per (field, filter).
    target_fidfilt_counts = np.zeros((num_fields, nfilters), dtype=np.int32)
    np.add.at(
        target_fidfilt_counts,
        (valid_df["field_id"].values, valid_df["filt_idx"].values),
        1,
    )

    # fidfilt_exptime: most-common exptime per (field, filter). Computed
    # over the FULL df (not teff-filtered) since exptime is a configured
    # parameter, not an outcome. Cells not in the survey get 0; the
    # action mask never lets the agent land on those.
    fidfilt_exptime = (
        df.pivot_table(
            index="field_id", columns="filt_idx", values="exptime",
            aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else 0,
        )
        .reindex(index=range(num_fields), columns=range(nfilters), fill_value=0)
        .to_numpy(dtype=np.float32)
    )


    # ---------- Per-night visit history ----------
    # Running counts snapshotted at the START of each night. Iterate
    # over all nights in df (not just those with valid observations)
    # so every observed night has a seedable state, but only valid
    # observations contribute to the running totals — keeping history
    # consistent with the target-completion semantics.
    field_running = np.zeros(num_fields, dtype=np.int32)
    fidfilt_running = np.zeros((num_fields, nfilters), dtype=np.int32)
    night2fid_visit_hist = {}
    night2fidfilt_visit_hist = {}
 
    for night, night_df in df.groupby("night"):
        night2fid_visit_hist[night] = field_running.copy()
        night2fidfilt_visit_hist[night] = fidfilt_running.copy()
 
        valid_night = night_df[night_df["teff"] > 0.3]
        if len(valid_night):
            field_running += np.bincount(
                valid_night["field_id"].values, minlength=num_fields
            )
            np.add.at(
                fidfilt_running,
                (valid_night["field_id"].values, valid_night["filt_idx"].values),
                1,
            )
    print(f'DOES NIGHT2FIDFILT_VISIT_HIST GET CONSTRUCTED? {night2fid_visit_hist is not None}')

    # ---------- Construct and persist ----------
    lookups = LookupTables(
        fields=fields,
        target_fidfilt_counts=target_fidfilt_counts,
        fidfilt_exptime=fidfilt_exptime,
        dir=outdir,
        night2fid_visit_hist=night2fid_visit_hist,
        night2fidfilt_visit_hist=night2fidfilt_visit_hist,
    )
    lookups.write_to_disk(outdir)
    logger.info(f"Successfully generated lookup tables in {outdir}")
    return lookups
 
def main():
    parser = argparse.ArgumentParser(
        description="Generate train-data lookup tables from raw DECam observations."
    )
    parser.add_argument(
        "--fits_path", type=Path, default=TRAIN_DATA_PATH,
        help="Path to the raw DECam exposures FITS file",
    )
    parser.add_argument(
        "--outdir", type=Path, default=TRAIN_DATA_DIR,
        help="Directory to save the generated lookup tables",
    )
    args = parser.parse_args()
 
    logger = setup_logger(save_dir=None)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting lookup generation...")
    save_DES_bin_and_field_mappings(fits_path=args.fits_path, outdir=args.outdir)
 
 
if __name__ == "__main__":
    main()