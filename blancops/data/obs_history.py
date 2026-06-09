"""Seed offline-scheduler state from a prior observing history.

Reads either a schedule CSV (this package's ``run_offline_scheduler`` /
``OfflineRunner`` output) or a live JSONL observing log (``live_scheduler``
``ProgressManager`` output) and reconstructs the survey-progress seed values
consumed by ``OfflineBlancoEnv``:

- ``initial_counts``        per-(field, filter) visit counts
- ``initial_last_visit_ot`` per-(field, filter) last-visit observing-time (OT)
- ``initial_ot_at_sunset``  cumulative OT clock at the first new night's sunset

The per-night OT/counts accumulation mirrors ``build_DES_lookups`` in
``blancops/data/preprocessing.py`` (the FITS training pipeline) so seeded runs
share the same OT frame as the trained model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from blancops.configs.constants import FILTER2IDX
from blancops.data.features.glob_features import get_night_boundaries

import logging
logger = logging.getLogger(__name__)


# Candidate column names across the two supported formats. Schedule CSVs use the
# agent_* names from io.schedule_io.SCHEDULE_KEYS; live JSONL logs use the bare
# proposal-row names emitted by live_scheduler.model_runner.
_FIELD_ID_COLS = ("agent_field_id", "field_id")
_TIMESTAMP_COLS = ("agent_timestamp", "timestamp")
_FILTER_IDX_COLS = ("agent_filter_idx", "filter_idx")
_FILTER_NAME_COLS = ("agent_filter", "filter")


def _first_present(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Observation history is missing a {what} column "
        f"(looked for {list(candidates)}); found {list(df.columns)}."
    )


def _read_obs_history(path: Path) -> pd.DataFrame:
    """Load a schedule CSV or live JSONL log into a raw DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".jsonl", ".json"):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(
        f"Unsupported obs-history format '{suffix}' for {path}; "
        f"expected .csv (schedule) or .jsonl/.json (live log)."
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a raw history frame to columns: field_id, filt_idx, timestamp, night."""
    if df.empty:
        raise ValueError("Observation history is empty.")

    field_col = _first_present(df, _FIELD_ID_COLS, "field id")
    ts_col = _first_present(df, _TIMESTAMP_COLS, "timestamp")

    field_id = df[field_col].to_numpy(dtype=np.int64)
    timestamp = df[ts_col].to_numpy(dtype=np.float64)

    # Filter index may be stored directly (CSV) or as a name needing FILTER2IDX.
    idx_col = next((c for c in _FILTER_IDX_COLS if c in df.columns), None)
    if idx_col is not None:
        filt_idx = df[idx_col].to_numpy(dtype=np.int64)
    else:
        name_col = _first_present(df, _FILTER_NAME_COLS, "filter")
        mapped = df[name_col].map(FILTER2IDX)
        if mapped.isna().any():
            bad = sorted(set(df[name_col][mapped.isna()]))
            raise ValueError(f"Unrecognized filter name(s) in history: {bad}.")
        filt_idx = mapped.to_numpy(dtype=np.int64)

    out = pd.DataFrame(
        {"field_id": field_id, "filt_idx": filt_idx, "timestamp": timestamp}
    )
    # Night key: matches fits_io._add_night (datetime - 12h -> date) so night
    # grouping is consistent with the FITS training lookups.
    dt = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    out["night"] = (dt - pd.Timedelta(hours=12)).dt.date
    return out


def load_seed_state_from_obs_history(path, lookups, sun_el_limit):
    """Reconstruct survey-progress seeds from a prior observing history.

    Args:
        path: schedule CSV (.csv) or live JSONL log (.jsonl/.json).
        lookups: LookupTables; supplies the (n_fields, n_filters) shape.
        sun_el_limit: sun-elevation limit (deg) bounding each observing night.

    Returns:
        (initial_counts, initial_last_visit_ot, initial_ot_at_sunset)
        with shapes matching lookups.target_fidfilt_counts. Mirrors the
        per-night accumulation in blancops/data/preprocessing.py.
    """
    path = Path(path)
    df = _normalize(_read_obs_history(path))

    n_fields, n_filters = lookups.target_fidfilt_counts.shape
    counts = np.zeros((n_fields, n_filters), dtype=np.int64)
    last_visit_ot = np.full((n_fields, n_filters), np.nan, dtype=np.float64)

    cum_ot = 0.0
    for night, night_df in df.groupby("night"):
        sunset_ts, sunrise_ts = get_night_boundaries(night, sun_el_limit=sun_el_limit)
        night_dur = sunrise_ts - sunset_ts

        # Visit counts: every logged row is a real exposure (no teff filter,
        # unlike the FITS pipeline whose raw catalog includes invalid frames).
        np.add.at(
            counts,
            (night_df["field_id"].to_numpy(), night_df["filt_idx"].to_numpy()),
            1,
        )

        # Per-(field, filter) last-visit OT, in cumulative seconds since survey
        # start: ot = cum_ot + (timestamp - sunset_ts). np.fmax keeps the latest
        # visit and is NaN-aware (incoming value wins over an existing NaN).
        ot = cum_ot + (night_df["timestamp"].to_numpy() - sunset_ts)
        ff_max = night_df.assign(ot=ot).groupby(["field_id", "filt_idx"])["ot"].max()
        keys = np.array(ff_max.index.tolist(), dtype=np.int64)
        rows, cols = keys[:, 0], keys[:, 1]
        last_visit_ot[rows, cols] = np.fmax(last_visit_ot[rows, cols], ff_max.to_numpy())

        cum_ot += night_dur

    initial_ot_at_sunset = float(cum_ot)
    logger.info(
        "Seeded survey state from %s: %d visits across %d (field, filter) cells; "
        "initial_ot_at_sunset=%.1f s.",
        path.name, int(counts.sum()), int((counts > 0).sum()), initial_ot_at_sunset,
    )
    return counts, last_visit_ot, initial_ot_at_sunset
