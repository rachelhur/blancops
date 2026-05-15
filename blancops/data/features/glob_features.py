"""Global (per-timestep, single-row) feature computation.

The module-level helpers at the top define the canonical per-timestep
computation. Both ``BaseBlancoEnv._calculate_global_features`` (live,
1 timestep) and ``GlobalFeatureEngineer.transform`` (offline, batch) drive
them — the offline pipeline iterating rows and writing into DataFrame
columns; the live env returning a single dict.

Two helpers split the computation by data dependency:

- ``compute_global_time_only_features``: depends only on ``timestamp``.
  Sun/Moon positions, Moon phase, LST. Used by both pipelines.
- ``compute_global_pointing_features``: depends on ``(timestamp, ra, dec)``.
  Topocentric az/el/ha, plane-parallel airmass, per-filter sky brightness.
  Used by the live env only — the offline pipeline reads az/el/ha from the
  raw FITS measurements (which differ from ephemeris values due to
  atmospheric refraction and pointing error) and uses its own vectorized
  sky-brightness pass for batch efficiency.

A third helper, ``apply_cyclical_global_features``, expands cyclical
features to ``_cos``/``_sin`` pairs. It works on either a dict (live) or a
pandas DataFrame (offline) via duck-typed `in` / ``[k]`` / ``[k] = v``.
"""
from datetime import time, timezone, timedelta
import datetime
from typing import Union
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import ephem
from astropy.time import Time
from tqdm import tqdm
from datetime import date, datetime, timedelta, timezone

from blancops.data.features.normalizations import apply_cyclical_features
from blancops.ephemerides.time_utils import standardize_time, unix_to_datetime
from blancops.math import units
from blancops.ephemerides import ephemerides
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.data.constants import (
    BLANCO_LON, ZENITH_BIN_NUM, ZENITH_FIELD_ID, ZENITH_WAVELENGTH,
    FILTER2WAVE, FILTERWAVENORM, FILTER2IDX, ZENITH_FILTER,
    ZENITH_AZ, ZENITH_EL, ZENITH_AIRMASS, ZENITH_ZD, ZENITH_HA, ZENITH_OBJECT
)

import logging
logger = logging.getLogger(__name__)


# ============================================================================
# Canonical per-timestep helpers — single source of truth, shared between
# BaseBlancoEnv (live, single-step) and GlobalFeatureEngineer (offline batch).
# ============================================================================


def compute_global_time_only_features(*, timestamp) -> dict:
    """Time-only global ephemeris features (no pointing dependence).

    Returns a dict with: ``lst``, ``sun_ra``, ``sun_dec``, ``sun_az``,
    ``sun_el``, ``moon_ra``, ``moon_dec``, ``moon_az``, ``moon_el``,
    ``moon_phase`` — each a scalar at the given timestamp.

    Used by both pipelines: live calls this once per step; offline calls
    it inside its row loop to fill the sun/moon/lst columns.
    """
    features = {}

    astro_time = Time(timestamp, format='unix', scale='utc')
    features['lst'] = float(
        astro_time.sidereal_time('apparent', longitude=BLANCO_LON).radian
    )

    sun_radec, sun_azel, moon_radec, moon_azel = calc_sun_and_moon_positions(timestamp)
    features['sun_ra'], features['sun_dec'] = sun_radec
    features['sun_az'], features['sun_el'] = sun_azel
    features['moon_ra'], features['moon_dec'] = moon_radec
    features['moon_az'], features['moon_el'] = moon_azel
    features['moon_phase'] = calc_moon_phase(timestamp)

    return features


def compute_global_pointing_features(*, timestamp, ra, dec) -> dict:
    """Pointing-dependent global ephemeris features.

    Returns a dict with: ``az``, ``el`` (clipped to ``[0, π/2]``), ``ha``,
    ``airmass``, and ``sky_brightness_<filter>`` for each filter in
    ``FILTER2IDX``.

    The elevation clip absorbs the precision issue where ``el`` can be
    slightly negative just before sunrise/sunset and propagate into the
    airmass calculation as a divergent value.

    Used by the live env. The offline pipeline takes ``az``/``el``/``ha``
    from FITS measurements (which differ from ephemeris values due to
    atmospheric refraction and pointing error) and uses a vectorized
    sky-brightness pass for batch efficiency, so it does not call this
    helper.
    """
    features = {}

    az, el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=timestamp)
    el = max(min(el, np.pi / 2), 0.0)
    features['az'] = az
    features['el'] = el
    features['ha'] = ephemerides.equatorial_to_hour_angle(
        ra=ra, dec=dec, time=timestamp
    )
    features['airmass'] = 1.0 / np.cos(np.pi / 2 - el)

    for filt in FILTER2IDX.keys():
        features[f"sky_brightness_{filt}"] = estimate_sky_brightness(
            time=timestamp, ra=ra, dec=dec, band=filt
        )

    return features


# ============================================================================
# GlobalFeatureEngineer — offline batch pipeline.
# ============================================================================


class GlobalFeatureEngineer:
    """Pipeline for calculating global state features for blancops RL."""

    def __init__(self, lookups, hpGrid, base_features, cyclical_features,
                 do_cyclical_norm=True, do_filt=True):
        self.lookups = lookups
        self.hpGrid = hpGrid
        self.base_features = base_features
        self.cyclical_features = cyclical_features
        self.do_cyclical_norm = do_cyclical_norm
        self.do_filt = do_filt

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the full feature engineering pipeline."""
        return (df
            .pipe(self._add_zenith_rows)
            .pipe(self._add_time_dependent_features)
            .pipe(self._map_bins_and_fields)
            .pipe(self._add_filter_choice)
            .pipe(self._add_filter_dep_features)
            .pipe(self._apply_cyclical_norms)
            .pipe(self._ensure_32_bit)
        )

    def _add_zenith_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        zenith_df = get_zenith_features(original_df=df)
        df = _backfill_zenith_states(merge_zenith_df(df, zenith_df))
        return df

    def _add_time_dependent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill LST + sun/moon + moon_phase columns via the shared helper,
        then add ``t_night`` (which needs per-night normalization).
        """
        timestamps = df['timestamp'].values

        # Drive the shared helper per row; accumulate into per-column lists,
        # then assign back to df in one shot.
        feat_lists = defaultdict(list)
        for t in tqdm(timestamps, total=len(timestamps),
                      desc='Calculating sun/moon/lst ephemeris'):
            feats = compute_global_time_only_features(timestamp=t)
            for k, v in feats.items():
                feat_lists[k].append(v)

        for k, vs in feat_lists.items():
            df[k] = vs

        # Preserve the lst_hours debugging column when LST is requested —
        # it's not in any standard feature set, but several offline tools
        # consume it. Cheap to compute vectorized.
        if 'lst' in self.base_features:
            _, df['lst_hours'] = calc_lst(df['datetime'].values)

        df['t_night'] = df.groupby('night')['timestamp'].transform(normalize_times)
        return df

    def _map_bins_and_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps RA/Dec to field_id and bin number, using lookups and hpGrid if provided."""
        # df['field_id'] = df['object'].map({v: k for k, v in self.fid2name.items()})
        df['field_id'] = df['object'].map({v: k for k, v in self.lookups.fields['object'].to_dict().items()})

        if self.hpGrid is not None:
            lon = df['az'] if self.hpGrid.is_azel else df['ra']
            lat = df['el'] if self.hpGrid.is_azel else df['dec']

            df['bin'] = self.hpGrid.ang2idx(lon=lon, lat=lat)

            # Re-assign zenith specifics
            zenith_mask = df['object'] == 'zenith'
            df.loc[zenith_mask, "bin"] = ZENITH_BIN_NUM
            df.loc[zenith_mask, "field_id"] = ZENITH_FIELD_ID

        return df

    def _add_filter_choice(self, df: pd.DataFrame):
        df['filter_wave'] = df['filter'].map(FILTER2WAVE)
        df['filter_wave'] = df['filter_wave'].fillna(ZENITH_WAVELENGTH) / FILTERWAVENORM # zenith "filter" set to 0, then normalize
        df['filter_idx'] = df['filter'].map(FILTER2IDX)
        for feat_name in self.base_features:
            if feat_name == 'is_filter':
                for filt in FILTER2IDX.keys():
                    df[f'{feat_name}_{filt}'] = (df['filter'] == filt).astype(np.float32)
        return df

    def _add_filter_dep_features(self, df):
        """Vectorized per-filter sky brightness. Kept separate from the
        shared per-timestep helper because ``estimate_sky_brightness``
        accepts arrays — calling it five times with N-element arrays is
        much faster than calling it per row in the time-features loop.
        """
        # if any(('sky_brightness' in base_feat) for base_feat in self.base_features):
        for filt in FILTER2WAVE.keys():
            if filt != ZENITH_FILTER:
                if any(('sky_brightness' == base_feat) for base_feat in self.base_features):
                    df[f'sky_brightness_{filt}'] = estimate_sky_brightness(
                        time=df['timestamp'].values,
                        ra=df['ra'].values,
                        dec=df['dec'].values,
                        band=filt,
                    )
        return df
    
    def _ensure_32_bit(self, df):
        for bin_str, np_bit in zip(['float64', 'int64'], [np.float32, np.int32]):
            cols = df.select_dtypes(include=[bin_str]).columns
            df[cols] = df[cols].astype(np_bit)
        return df

    def _apply_cyclical_norms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cos/sin pairs for cyclical features via the shared helper.

        Uses the corrected matching rule (exact name OR ``_<cyc>`` suffix);
        the previous in-class version used a bare ``endswith`` that also
        matched ``rel_*`` features against their unprefixed cyclical roots.
        """
        apply_cyclical_features(
            df, self.base_features, self.cyclical_features
        )
        return df


# ============================================================================
# Other helpers (data-loading-time computations, not part of the shared
# per-timestep API).
# ============================================================================


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

def get_night_boundaries(
    anchor: Union[
    float, int, np.integer, np.floating,
    datetime, date, pd.Timestamp, str,
    pd.Series
    ],
    sun_el_limit: float = -14.0,
    observer_lon_rad: float = -70.8065,
) -> tuple[float, float]:
    """Compute (sunset_ts, sunrise_ts) UTC unix timestamps for the
    Blanco observing night identified by `anchor`.

    `anchor` accepts whatever is most natural at the call site:
      - **Unix timestamp** (numeric, including numpy scalars): a moment
        in time. The night is determined with a local-noon cutover —
        moments before local solar noon resolve to the night just
        ended, after to the upcoming. References inside an observing
        night always land on that night.
      - **Datetime / Timestamp / ISO string**: same — interpreted as a
        moment, reduced via the unix-timestamp path. (For
        `first_row["night"]` at 23:59:59 UTC, the moment is well
        inside the local-noon cutover region for its night, so this
        Just Works.)
      - **`date`**: treated as the canonical evening-date of the
        observing night and used directly, no cutover needed.
      - **`pd.Series`**: uses the first element. Convenient for
        `df.groupby('night').transform(...)` callbacks where every
        element of the group belongs to the same night by construction.
    """
    noon_ts = _noon_anchor_from_input(anchor, observer_lon_rad)
    sunset_ts = calc_twilight(noon_ts, "set", sun_el_limit)
    sunrise_ts = calc_twilight(noon_ts, "rise", sun_el_limit)

    if not (sunset_ts < sunrise_ts):
        raise RuntimeError(
            f"Computed sunset ({sunset_ts}) does not precede sunrise "
            f"({sunrise_ts}) for sun_el_limit={sun_el_limit}. Check that "
            f"sun_el_limit isn't above the local horizon for this date "
            f"(no twilight crossing) and that calc_twilight returns "
            f"`next_*`-style boundaries."
        )
    return sunset_ts, sunrise_ts

def _noon_anchor_from_input(
    anchor: Union[
        float, int, np.integer, np.floating,
        datetime, date, pd.Timestamp, str,
        pd.Series,
        ],
    observer_lon_deg: float = -70.8065,
) -> float:
    """Resolve a flexible `anchor` to the unix timestamp of local solar
    noon on the corresponding date."""
    # Approximate UTC offset from longitude (15° per hour). Solar time,
    # not civil — accurate to within the equation of time (~16 min) and
    # free of timezone/DST handling. Sufficient because noon is ~6 h
    # from any twilight boundary.
    utc_offset_sec = observer_lon_deg * (3600.0 / 15.0)  # degrees to seconds

    # pd.Series: every element belongs to the same night by groupby
    # construction at all current call sites; use the first.
    if isinstance(anchor, pd.Series):
        if len(anchor) == 0:
            raise ValueError(
                "Cannot derive a night anchor from an empty Series."
            )
        anchor = anchor.iloc[0]

    # `date` but NOT `datetime` (datetime is a subclass of date in
    # the stdlib): treat as canonical evening-date label, skip cutover.
    if isinstance(anchor, date) and not isinstance(anchor, datetime):
        anchor_date = anchor
    else:
        # Everything else reduces to a unix timestamp.
        if isinstance(anchor, (int, float, np.integer, np.floating)):
            ts_unix = float(anchor)
        elif isinstance(anchor, pd.Timestamp):
            ts = (
                anchor if anchor.tzinfo is not None
                else anchor.tz_localize("UTC")
            )
            ts_unix = ts.timestamp()
        elif isinstance(anchor, datetime):
            ts = (
                anchor if anchor.tzinfo is not None
                else anchor.replace(tzinfo=timezone.utc)
            )
            ts_unix = ts.timestamp()
        elif isinstance(anchor, str):
            ts = pd.Timestamp(anchor)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts_unix = ts.timestamp()
        else:
            raise TypeError(
                f"Unsupported anchor type {type(anchor).__name__}; "
                f"expected unix timestamp, datetime, date, pd.Timestamp, "
                f"ISO string, or pd.Series of timestamps."
            )

        # Local-noon cutover: shift to local time, back 12 h, take date.
        local_ref = datetime.fromtimestamp(
            ts_unix + utc_offset_sec, tz=timezone.utc
        )
        anchor_date = (local_ref - timedelta(hours=12)).date()

    return (
        datetime(
            anchor_date.year, anchor_date.month, anchor_date.day,
            hour=12, tzinfo=timezone.utc,
        ).timestamp()
        - utc_offset_sec
    )

def calc_twilight(ts, event_type='set', horizon='-14', buffer_in_seconds=10):
    obs = ephemerides.blanco_observer(time=ts)
    obs.horizon = str(horizon)
    sun = ephem.Sun()
    sun.compute(obs)

    if event_type == 'rise':
        ephem_date = obs.next_rising(sun)
    elif event_type == 'set':
        ephem_date = obs.next_setting(sun)
    else:
        raise NotImplementedError(f"Unsupported event_type: {event_type}")

    dt_utc = ephem_date.datetime().replace(tzinfo=timezone.utc)
    if event_type == 'rise':
        dt_utc -= timedelta(seconds=buffer_in_seconds)
    else:
        dt_utc += timedelta(seconds=buffer_in_seconds)
    return dt_utc.timestamp()

def calculate_sun_rise_and_set_times(df):
    rise_times = df.groupby('night').apply(calc_twilight, event_type='rise').values
    set_times = df.groupby('night').apply(calc_twilight, event_type='set').values
    return rise_times, set_times

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

def merge_zenith_df(df, zenith_df):
    df = pd.concat([df, zenith_df], ignore_index=True)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
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
        row_dict['ra'], row_dict['dec'] = np.array(blanco.radec_of('0',  '90'))
        zenith_rows.append(row_dict)

    zenith_df = pd.DataFrame(zenith_rows)
    zenith_df['az'] = ZENITH_AZ * units.deg
    zenith_df['el'] = ZENITH_EL * units.deg
    zenith_df['airmass'] = ZENITH_AIRMASS
    zenith_df['zd'] = ZENITH_ZD * units.deg
    zenith_df['ha'] = ZENITH_HA * units.deg
    zenith_df['object'] = ZENITH_OBJECT
    zenith_df['field_id'] = ZENITH_FIELD_ID
    zenith_df['filter'] = ZENITH_FILTER
    zenith_df['datetime'] = pd.to_datetime(zenith_df['datetime'], utc=True)
    zenith_df['night'] = pd.to_datetime(zenith_df['night'], utc=True)

    return zenith_df

def _backfill_zenith_states(df):
    """Back fills zenith state for relevant features"""
    df['fwhm'] = df.groupby('night')['fwhm'].bfill()
    df['night_idx'] = df.groupby('night')['night_idx'].bfill()
    df['t_survey'] = df.groupby('night')['t_survey'].bfill()
    for f in FILTER2IDX.keys():
        df[f'raw_survey_progress_{f}'] = df.groupby('night')[f'raw_survey_progress_{f}'].bfill()
        df[f'survey_progress_{f}'] = df.groupby('night')[f'survey_progress_{f}'].bfill()
        df[f'urgency_{f}'] = df.groupby('night')[f'urgency_{f}'].bfill()
    return df


def normalize_times(time_series, sun_el_limit=-10):
    sunset_ts, sunrise_ts = get_night_boundaries(time_series, sun_el_limit)

    # sunset_ts = calc_twilight(time_series.median(), event_type='set')
    # sunrise_ts = calc_twilight(time_series.median(), event_type='rise')
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


def calculate_sun_rise_and_set_azel(df):
    rise_times, set_times = calculate_sun_rise_and_set_times(df)
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