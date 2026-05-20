"""Pure-function helpers for angular/airmass/ephemerides calculations.

These functions are independent of the Evaluator/DataContainer plumbing so they
can be reused (and tested) on their own.
"""
from __future__ import annotations

import numpy as np

from blancops.math.geometry import angular_separation
from blancops.ephemerides.ephemerides import get_source_ra_dec
from blancops.data.features.glob_features import (
    calc_moon_phase as _calc_moon_phase,
    calc_sun_and_moon_positions as _calc_sun_and_moon_pos,
)

import logging
logger = logging.getLogger(__name__)


def calc_airmass(el):
    """Plane-parallel airmass approximation. `el` in radians."""
    return 1 / np.cos(np.pi / 2 - el)


def calc_slew_distance(prev_radecs, radecs):
    """Per-row angular separation between two arrays of (ra, dec) pairs."""
    n = len(prev_radecs)
    out = np.zeros(n)
    for i in range(n):
        out[i] = angular_separation(prev_radecs[i], radecs[i])
    return out


def calc_moon_dist(radecs, timestamps):
    """Angular distance from each (ra, dec) to the moon at the matching timestamp."""
    out = np.zeros(len(timestamps))
    for i, t in enumerate(timestamps):
        moon_radec = get_source_ra_dec('moon', time=t)
        out[i] = angular_separation(moon_radec, radecs[i])
    return out


def calc_moon_phase(timestamps):
    out = np.empty(len(timestamps))
    for i, t in enumerate(timestamps):
        out[i] = _calc_moon_phase(t)
    return out


def calc_sun_and_moon_pos(timestamps):
    """Returns (sun_az, sun_el, moon_az, moon_el) arrays."""
    sun_azel = np.empty((len(timestamps), 2))
    moon_azel = np.empty((len(timestamps), 2))
    for i, t in enumerate(timestamps):
        _, sun_azel[i], _, moon_azel[i] = _calc_sun_and_moon_pos(t)
    return sun_azel[:, 0], sun_azel[:, 1], moon_azel[:, 0], moon_azel[:, 1]
