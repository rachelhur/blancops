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
ZENITH_EL = 90
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
AZEL_BIN_FEAT_SENTINEL = 0.0 # no fields now, might be later
RADEC_BIN_FEAT_SENTINEL = -1.0

"""

NORMALIZATION CONSTANTS

"""

SKYBRIGHT_MAX = 23
SKYBRIGHT_MIN = 17
FWHM_MAX = 8

"""

IMPLEMENTED GRID NETWORK NAMES

"""

ACTION_ARCHITECTURES = ['simultaneous', 'multi_head_scorer', 'autoregressive']

"""

TEST SUITE CONSTS

"""

TEST_SUITE_NAMES = ['magic-spring', 'healpix-grid', 'delve', 'gw-followup']

MS_OBSERVING_DATES = ['2026-04-09-half1', '2026-04-10-half1', '2026-04-11-half1', '2026-04-12-half1', \
                      '2026-05-09-half1', '2026-05-10-half1', '2026-06-08-half1', '2026-06-09-half1']
HP_OBSERVING_DATES = ['2026-04-09-full', '2026-04-10-full', '2026-04-11-full', '2026-04-12-full', \
                      '2026-05-09-full', '2026-05-10-full', '2026-06-08-full', '2026-06-09-full']
DD_NIGHT = ['2026-04-10-half2']
GW_OBSERVING_DATES_GOOD = ['2026-02-10-full', '2026-02-11-full', '2026-02-12-full', '2026-02-13-full', \
                           '2026-02-14-full', '2026-02-15-full', '2026-02-16-full', '2026-02-17-full']
GW_OBSERVING_DATES_BAD = ['2026-05-09-full', '2026-05-10-full', '2026-05-11-full', '2026-05-12-full', \
                          '2026-05-13-full', '2026-05-14-full', '2026-05-15-full', '2026-05-16-full']
# DELVE_OBSERVING_DATES = ['2026-04-09-half1', '2026-04-10-half1', '2026-04-11-half1', '2026-04-12-half1', '2026-05-09-half1', '2026-05-10-half1', '2026-06-08-half1', '2026-06-09-half1']

"""

DEPLOYMENT CONSTS

"""

DEPLOYMENT_OBSERVING_DATES = ['2026-06-23-half1', '2026-06-24-half1']


"""

MODEL CONSTANTS

"""

import numpy as np

# Focal loss weights
FILTER_COUNTS_ORDERED = np.array([20574, 18450, 17312, 15984, 16221]) # entire train dataset
FILTER_ALPHA_WEIGHTS = 1 / FILTER_COUNTS_ORDERED * len(FILTER_COUNTS_ORDERED) / np.sum(1/FILTER_COUNTS_ORDERED)