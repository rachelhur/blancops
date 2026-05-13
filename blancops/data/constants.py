from enum import IntEnum
import numpy as np

class EnvSignal(IntEnum):
    WAIT = -2
    NO_FILTER = -1
    
"""
BLANCO CONSTS
"""

# BLANCO_LAT = -30.169
BLANCO_LON = "-70:48:23.49"
BLANCO_ELEV = 2200

"""

ZENITH CONSTANTS

"""

ZENITH_AZ = 0
ZENITH_EL = np.pi/2
ZENITH_AIRMASS = 1
ZENITH_ZD = 0
ZENITH_HA = 0
ZENITH_OBJECT = 'zenith'
ZENITH_FIELD_ID = -1
ZENITH_BIN_NUM = -1
ZENITH_WAVELENGTH = 0
ZENITH_FILTER_IDX = -1
ZENITH_FILTER = 'null'

"""

FILTER INFO 

"""

# Filter wavelengths (nm) according to obztak https://github.com/kadrlica/obztak/blob/c28fab23b09bcff1cf46746eae4ec7e40aeb7f7a/obztak/seeing.py#L22
FILTER2WAVE = {
    # 'u': 380, # not present in train data,
    'g': 480,
    'r': 640,
    'i': 780,
    'z': 920,
    'Y': 990
}

NUM_FILTERS = len(FILTER2WAVE)
IDX2WAVE = {i: FILTER2WAVE[k] for i, k in enumerate(FILTER2WAVE.keys())}
FILTERWAVENORM = 1000.

FILTER2IDX = {k: i for i, k in enumerate(FILTER2WAVE.keys())}
IDX2FILTER = {v: k for k, v in FILTER2IDX.items()}

"""

ENVIRONMENT SENTINEL VALS

"""
WAIT_SIGNAL = -2
NO_FILTER_SIGNAL = -1 # if action space is just bins, not filters
AZEL_BIN_FEAT_SENTINEL = -1.0 # no fields now, might be later
RADEC_BIN_FEAT_SENTINEL = -1.0

