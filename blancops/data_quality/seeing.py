"""Seeing conversion and rolling-history prediction helpers."""
import warnings
import numpy as np
import pandas as pd
from blancops.math import units
from blancops.ephemerides.time_utils import (
    standardize_time,
    utc_now,
    standardize_timedelta,
)

__all__ = ["convert_seeing", "Seeing"]

# Hardcoded seeing conversion factors between bands
# Copied from obztak seeing.py, which in turn got them from Eric Neilsen
# Values are nominally (lambda_i / lambda_j)^(1/5), for effective band center lambda
# For more info, see https://doi.org/10.2172/1574836 Appendix C
_BAND_FACTORS = {
    "u": 0.86603,  # ( u (380nm) / i (780nm) ) ** (1/5)
    "g": 0.9067,  # ( g (480nm) / i (780nm) ) ** (1/5)
    "r": 0.9609,  # ( r (640nm) / i (780nm) ) ** (1/5)
    "i": 1.0,  # ( i (780nm) / i (780nm) ) ** (1/5)
    "z": 1.036,  # ( z (920nm) / i (780nm) ) ** (1/5)
    "Y": 1.0523,  # ( Y (990nm) / i (780nm) ) ** (1/5)
}

# Hardcoded contribution of DECam to the PSF
# From obztak seeing.py, which in turn got it from Eric Neilsen
_DECAM_FWHM = 0.5 * units.arcsec


def convert_seeing(
    seeing,
    to_band,
    to_el,
    to_instrument=_DECAM_FWHM,
    from_band="i",
    from_el=90 * units.deg,
    from_instrument=_DECAM_FWHM,
):
    """
    Convert seeing from one band and elevation to another band and elevation assuming a
    Kolmogorov turbulence model, wherein seeing ~ airmass^(3/5) / wavelength^(1/5). For
    more info, see https://doi.org/10.2172/1574836 Appendix C.

    Arguments
    ---------
    seeing : float or list-like of float
        Seeing value to convert.
    to_band : str or list-like of str
        Target band for conversion ('u', 'g', 'r', 'i', 'z', or 'Y').
    to_el : float or list-like of float
        Target elevation for conversion (in radians).
    to_instrument : float [0.5 * units.arcsec]
        Component of seeing due to the target instrument, summed in quadrature with the
        atmospheric component. Default is nominal DECam contribution.
    from_band : str or list-like of str ["i"]
        Original band of the seeing value ('u', 'g', 'r', 'i', 'z', or 'Y').
    from_el : float or list-like of float [90 * units.deg]
        Original elevation of the seeing value (in radians). Default is zenith.
    from_instrument : float [0.5 * units.arcsec]
        Component of seeing due to the instrument, summed in quadrature with the
        atmospheric component. Default is nominal DECam contribution.

    Returns
    -------
    float or np.ndarray of float
        Converted seeing value.
    """
    # check for scalar inputs to preserve output shape
    scalar_input = np.all(
        [np.ndim(x) == 0 for x in [seeing, to_band, to_el, from_band, from_el]]
    )

    # remove instrument contribution in quadrature
    seeing = np.sqrt(np.asarray(seeing) ** 2 - from_instrument**2)

    # airmass conversion factor
    from_am = 1.0 / np.sin(from_el)
    to_am = 1.0 / np.sin(to_el)
    airmass_factor = (to_am / from_am) ** (3 / 5)

    # wavelength conversion factor
    to_band_factor = np.asarray(
        [_BAND_FACTORS[band] for band in np.atleast_1d(to_band)]
    )
    from_band_factor = np.asarray(
        [_BAND_FACTORS[band] for band in np.atleast_1d(from_band)]
    )
    band_factor = (to_band_factor / from_band_factor) ** (-1)

    # convert airmass and wavelength contributions and add back instrument contribution
    seeing = np.hypot(seeing * airmass_factor * band_factor, to_instrument)
    return np.squeeze(seeing) if scalar_input else seeing


class Seeing:
    """
    Container for recent seeing measurements and a predictor for future values, adapted
    from obztak's seeing.py. Specifically:
    - Store incoming raw measurements (self.raw)
    - Convert raw values to i-band at zenith with no instrument contribution (self.data)
    - Provide a near-future prediction via a weighted-average heuristic

    Usage:
    - add : ingest one or more measurements of seeing
    - prune : drop history older than a specified retention window to manage memory
    - predict : return a single predicted observed seeing FWHM
    """

    def __init__(
        self,
        window="15m",
        retention_window=None,
        from_instrument=_DECAM_FWHM,
        to_instrument=_DECAM_FWHM,
    ):
        """
        Initialize the Seeing container.

        Arguments
        ---------
        window : float, str, Timedelta ["15m"]
            Time window of recent history to use for prediction. Float values are
            interpreted as seconds. Str values are parsed as Timedeltas.
        retention_window : float, str, Timedelta [None]
            Time window to retain history for; defaults to 2*window.
        from_instrument : float [0.5 * units.arcsec]
            Instrument component of measured seeing values, summed in quadrature with
            the atmospheric component. Default is expected for qc_fwhm values from DECam
        to_instrument : float [0.5 * units.arcsec]
            Component of seeing due to the target instrument, summed in quadrature with
            the atmospheric component. Default is expected for DECam
        """
        # algorithm parameters
        self.window = standardize_timedelta(window)
        self.retention_window = (
            2 * self.window
            if retention_window is None
            else standardize_timedelta(retention_window)
        )
        self.from_instrument = from_instrument
        self.to_instrument = to_instrument

        # initialize empty storage frames
        self.raw = pd.DataFrame(columns=["date", "seeing", "band", "el"])
        self.data = pd.DataFrame(columns=["date", "seeing", "band", "el"])

        self._warned_empty_history = False

    def add(self, date, seeing, band, el, prune=False):
        """
        Ingest previous seeing measurements.

        Arguments
        ---------
        date : float, str, datetime, or list-like
            Times of seeing measurements, to be standardized with ephemerides.time_utils
        seeing : float or list-like of float
            Seeing measurements
        band : str or list-like of str
            Bands of the measurements
        el : float or list-like of float
            Elevations of the measurements
        prune : bool [False]
            Whether to drop old history after adding new measurements
        """
        # make dataframe of new measurements
        new = pd.DataFrame(
            {
                "date": [standardize_time(t) for t in np.atleast_1d(date)],
                "seeing": np.asarray(seeing, dtype=float),
                "band": np.asarray(band, dtype=str),
                "el": np.asarray(el, dtype=float),
            }
        )

        # append new data to history table
        combined = new if self.raw.empty else pd.concat([self.raw, new])
        self.raw = (
            combined
            .sort_values("date")
            .reset_index(drop=True)
            .convert_dtypes()
        )

        # convert raw data to i-band at zenith, with no instrument contribution
        self.data = pd.DataFrame(
            {
                "date": self.raw["date"],
                "seeing": convert_seeing(
                    seeing=self.raw["seeing"].to_numpy(),
                    to_band="i",
                    to_el=90 * units.deg,
                    from_band=self.raw["band"].to_numpy(),
                    from_el=self.raw["el"].to_numpy(),
                    from_instrument=self.from_instrument,
                    to_instrument=0.0,
                ),
                "band": "i",
                "el": 90 * units.deg,
            }
        )

        # drop old history if requested
        if prune:
            self.prune()

        return

    def prune(self, now=None, retention_window=None):
        """
        Remove history older than the retention window.

        Arguments
        ---------
        now : float, str, datetime [None]
            Reference time for pruning, removing history before now - retention_window.
            Defaults to now.
        retention_window : float, str, Timedelta [None]
            Time window to use for pruning; overrides self.retention_window
        """
        # nothing to prune if data is empty
        if self.data.empty:
            return

        # format reference time and window
        reference_time = standardize_time(now) if now is not None else utc_now()
        window = (
            standardize_timedelta(retention_window)
            if retention_window is not None
            else self.retention_window
        )
        cutoff = reference_time - window

        # filter both raw and converted tables to keep only recent rows
        self.raw = self.raw.loc[self.raw["date"] >= cutoff].reset_index(drop=True)
        self.data = self.data.loc[self.data["date"] >= cutoff].reset_index(drop=True)
        return

    def predict(self, band, el, now=None):
        """
        Predict the observed seeing in the requested band and elevation. Uses a
        log-space weighted heuristic adapted from obztak.

        Arguments
        ---------
        band : str
            Target band for prediction ("u", "g", "r", "i", "z", "Y").
        el : float
            Target elevation for prediction (in radians).
        now : float, str, datetime [None]
            Reference time for prediction. Defaults to now.
        """

        # build recent and ancient windows
        reference_time = standardize_time(now) if now is not None else utc_now()
        recent = (
            self.data.loc[
                (self.data["date"] < reference_time)
                & (self.data["date"] >= (reference_time - self.window)),
                "seeing",
            ]
            / units.arcsec
        )
        ancient = (
            self.data.loc[
                (self.data["date"] < (reference_time - self.window))
                & (self.data["date"] >= (reference_time - self.retention_window)),
                "seeing",
            ]
            / units.arcsec
        )

        # no data: use nominal DECam median seeing value
        xmu = np.log10(np.sqrt(0.9**2 - ((_DECAM_FWHM / units.arcsec) ** 2)))
        if recent.empty and ancient.empty:
            if not self._warned_empty_history:
                warnings.warn(
                    "No seeing history available; using the nominal DECam median.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_empty_history = True
            xpred = xmu

        # weighted-median heuristic
        # NB: constants were derived for 5min time delta and may not hold for arb window
        else:
            if (not recent.empty) and (not ancient.empty):  # weighted median
                xpred = (
                    xmu
                    + 0.8 * (np.log10(np.median(recent)) - xmu)
                    + 0.14 * (np.log10(np.median(ancient)) - xmu)
                )
            elif not recent.empty:  # median of recent data
                xpred = np.log10(np.median(recent))
            else:  # most recent ancient data
                xpred = np.log10(ancient.iloc[-1])

        # convert predicted i-band zenith seeing to target band and elevation
        return convert_seeing(
            (10**xpred) * units.arcsec,
            to_band=band,
            to_el=el,
            from_band="i",
            from_el=90 * units.deg,
            from_instrument=0.0,
            to_instrument=self.to_instrument,
        )
