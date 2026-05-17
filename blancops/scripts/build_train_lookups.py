import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from blancops.data.features.glob_features import get_night_boundaries
from blancops.math import units

from blancops.configs.constants import TRAIN_DATA_DIR, TRAIN_DATA_PATH
from blancops.configs.constants import FILTER2IDX
from blancops.data.lookup_tables import LookupTables
from blancops.data.preprocessing import drop_rows_in_DECam_data, preprocess_train_df
import matplotlib.pyplot as plt
import warnings
import logging
logger = logging.getLogger(__name__)

from blancops.io.logger_utils import setup_logger_old

# Quality threshold for an observation to count as a "real" visit.
# Lives here because the same threshold gates both target-completion
# (target_fidfilt_counts) and visit history (visit_hist and last_visit
# dicts) — they must agree, otherwise a field could be "complete" but
# never appear to have been visited.
_VALID_TEFF_THRESHOLD = 0.3


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

    # fidfilt_exptime: most-common exptime per (field, filter). Computed
    # over the FULL df (not teff-filtered) since exptime is a configured
    # parameter, not an outcome. (field, filter) pairs not in the survey get 0; the
    # action mask never lets the agent land on those.
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
        
    for night, night_df in df.groupby("night"):
            
        sunset_ts, sunrise_ts = get_night_boundaries(night, sun_el_limit=-10)
        night2ot_clock_seconds[night] = cum_ot
        night_dur = sunrise_ts - sunset_ts

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
    lookups = LookupTables(
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
        total_ot_sec=total_observing_seconds
    )
    lookups.write_to_disk(outdir)
    logger.info(f" [+] Successfully generated all lookup tables in {outdir}")
    
    # --------- Validate constructed/saved == loaded lookup ----------- #
    # loaded_lookups = LookupTables.load_from_dir(outdir, is_historic=True)
    # print(f"Saved lookups: {lookups}")
    # print(f"Loaded lookups: {loaded_lookups}")
    # if not lookups_loaded == lookups:
    #     logger.fatal(" [!!!] Mismatch between saved and loaded versions of LookupTable.")
            
    
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
    parser.add_argument(
        '-p', '--save_plots', action="store_true",
        help="Whether or not to save plots. (Will be saved to outdir)"
    )
    args = parser.parse_args()
 
    logger = setup_logger_old(save_dir=None)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting lookup generation...")
    lookups = save_DES_bin_and_field_mappings(fits_path=args.fits_path, outdir=args.outdir)
    
    save = args.save_plots
    
    
    # --------------------------------------  
    # PLOTTING
    # --------------------------------------  
    if save:
        _FIGSIZE = (6,6)
        _ra = (lookups.fields.ra.values + np.pi) % 2 * np.pi - np.pi
        _dec = lookups.fields.dec.values
        
        # --------------------------------------  
        # Fields in ra/dec, colored by field id
        # --------------------------------------  
        fig, ax  = plt.subplots(figsize=_FIGSIZE)
        ax.scatter(_ra/units.deg, _dec/units.deg, cmap="plasma", c=lookups.fields.index.values, alpha=.5, s=5)
        ax.set_xlabel('ra (deg)')
        ax.set_ylabel('dec (deg)')
        if save:
            fig.savefig(args.outdir / "field_radecs.png")
            
        # Plot target counts
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        for filt, fidx in FILTER2IDX.items():
            ax.scatter(np.arange(len(lookups.fields)), lookups.target_fidfilt_counts[:, fidx], label=filt, s=5, alpha=.5)
        ax.set_xlabel('Field id')
        ax.set_ylabel('Counts')
        if save:
            fig.savefig(args.outdir / "target_counts_per_field_filter.png")
        
        # --------------------------------------  
        # Average Accumulated Visits over bins vs night
        # --------------------------------------  
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        visits = np.array(list(lookups.night2fidfilt_visit_hist.values()))
        mean_visits = visits.mean(axis=1)
        std_visits = visits.std(axis=1)
        for filt, fidx in FILTER2IDX.items():
            ax.plot(np.arange(len(mean_visits)), mean_visits[:, fidx], label=filt, color=f"C{fidx}")
            ax.fill_between(
                np.arange(len(lookups.night2fidfilt_visit_hist)), 
                y1=mean_visits[:, fidx] - std_visits[:, fidx],
                y2=mean_visits[:, fidx] + std_visits[:, fidx],
                alpha=.3,
                color=f"C{fidx}"
                )
        ax.set_xlabel('Night of survey')
        ax.set_ylabel("Visits")
        ax.legend()
        if save:
            fig.savefig(args.outdir / "average_visits_over_time.png")
        
        # --------------------------------------  
        # Average Time Since Last Visit over bins vs night FOR INCOMPLETE FIELDS ONLY
        # --------------------------------------  
        
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        last_visit_times = np.array(list(lookups.night2fidfilt_last_visit_ot.values())) / 60 / 60 
        ot_clock_hour = np.array(list(lookups.night2ot_clock_seconds.values())) / 60 / 60 
        t_since_last_visit = ot_clock_hour[:, None, None] - last_visit_times  # (n_nights, n_fields, n_filters)

        visit_hist = np.array(list(lookups.night2fidfilt_visit_hist.values()))   # (n_nights, n_fields, n_filters)
        incomplete_mask = visit_hist < lookups.target_fidfilt_counts[None, :, :]
        t_since_last_visit = np.where(incomplete_mask, t_since_last_visit, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_times = np.nanmean(t_since_last_visit, axis=1)
            std_times = np.nanstd(t_since_last_visit, axis=1)

        for filt, fidx in FILTER2IDX.items():
            night_idxs = np.arange(len(mean_times))
            _filt_means = mean_times[:, fidx]
            _filt_stds = std_times[:, fidx]
            ax.plot(_filt_means, label=filt, color=f"C{fidx}")
            ax.fill_between(
                x=night_idxs,
                y1=_filt_means - _filt_stds,
                y2=_filt_means + _filt_stds,
                alpha=.3,
                color=f"C{fidx}"
            )
        ax.set_xlabel('Night of survey')
        ax.set_ylabel("Time since last visit (hour)\n(incomplete fields only)")
        ax.legend()
        if save:
            fig.tight_layout()
            fig.savefig(args.outdir / "average_time_since_last_visit_incomplete.png")

        
        # --------------------------------------  
        # Median Time Since Last Visit over bins vs night FOR INCOMPLETE FIELDS ONLY
        # Includes number of (field, filter) pair contributes per night, filter
        # --------------------------------------  
        
        fig, axes = plt.subplots(
            nrows=len(FILTER2IDX), ncols=1,
            figsize=(8, 2 * len(FILTER2IDX)),
            sharex=True,
        )
                
        last_visit_times = np.array(list(lookups.night2fidfilt_last_visit_ot.values())) / 60 / 60 
        ot_clock_hour = np.array(list(lookups.night2ot_clock_seconds.values())) / 60 / 60 
        t_since_last_visit = ot_clock_hour[:, None, None] - last_visit_times   # (n_nights, n_fields, n_filters)

        # Mask out (field, filter) cells that were already complete at the
        # start of each night — same logic as the bin-feature change in
        # compute_bin_progress_features. The np.where converts masked cells
        # to NaN so nanpercentile / nanmean skip them.
        visit_hist = np.array(list(lookups.night2fidfilt_visit_hist.values()))
        incomplete_mask = visit_hist < lookups.target_fidfilt_counts[None, :, :]
        t_since_last_visit = np.where(incomplete_mask, t_since_last_visit, np.nan)

        # Percentile reduction over the fields axis. nanpercentile returns NaN
        # for slices that are all-NaN (a filter with no recorded visits yet,
        # or all-complete late in the survey); matplotlib will leave gaps,
        # which is the intended visual.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pct = np.nanpercentile(t_since_last_visit, [10, 50, 90], axis=1)
            # pct shape: (3, n_nights, n_filters)

        # Contributing-(field, filter) count per (night, filter) — sanity check for the
        # noisy tail. Counts (field, filter) that are both incomplete AND have a recorded
        # visit (i.e., contributed a finite value to the percentile).
        contributing = (incomplete_mask & ~np.isnan(t_since_last_visit)).sum(axis=1)
        # contributing shape: (n_nights, n_filters)

        night_idxs = np.arange(pct.shape[1])
        for (filt, fidx), ax in zip(FILTER2IDX.items(), axes):
            ax2 = ax.twinx()
            p10, p50, p90 = pct[0, :, fidx], pct[1, :, fidx], pct[2, :, fidx]
            ax.plot(night_idxs, p50, color=f"C{fidx}", label=f"{filt} median")
            ax.fill_between(night_idxs, p10, p90, alpha=0.25, color=f"C{fidx}")
            ax2.plot(
                night_idxs, contributing[:, fidx],
                color="gray", linestyle="--", alpha=0.6, linewidth=0.8,
            )
            ax.set_ylabel(f"{filt}\nhour since visit")
            ax2.set_ylabel("n (field, filter) pairs", color="gray")
            ax.set_ylim(bottom=0)
            ax2.tick_params(axis="y", colors="gray")

        axes[-1].set_xlabel("Night of survey")
        fig.suptitle("Time since last visit (incomplete fields only)\nmedian with 10–90 percentile band")
        fig.tight_layout()
        
        if save:
            fig.tight_layout()
            fig.savefig(args.outdir / "median_time_since_last_visit_incomplete.png")
if __name__ == "__main__":
    main()