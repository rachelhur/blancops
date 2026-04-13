import pandas as pd

import numpy as np
from datetime import timezone, timedelta
import ephem
from astropy.time import Time
import torch
from einops import rearrange

import fitsio
from pathlib import Path
from tqdm import tqdm

from blancops.math import units
from blancops.data_quality.sky_brightness import estimate_sky_brightness
from blancops.configs.constants import get_workspace_dir
from blancops.ephemerides import ephemerides
from blancops.data.constants import *
from blancops.features.normalizations import *

import warnings
import logging
logger = logging.getLogger(__name__)

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

def time_until_set():
    pass

from blancops.ephemerides.ephemerides import HealpixGrid

def calculate_distance_matrix(nside, is_azel):
    hpGrid = HealpixGrid(nside, is_azel)
    lons = hpGrid.lon
    lats = hpGrid.lat
    distance_matrix = np.zeros( (len(hpGrid.lon), len(hpGrid.lon)) )
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        distance_matrix[i] = hpGrid.get_angular_separations(lon, lat)
    return distance_matrix