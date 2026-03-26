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
FILTERWAVENORM = 1000.
FILTER2IDX = {k: i for i, k in enumerate(FILTER2WAVE.keys())}
IDX2WAVE = {i: FILTER2WAVE[k] for i, k in enumerate(FILTER2WAVE.keys())}

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

"""

IMPLEMENTED GRID NETWORK NAMES

"""

GRID_NETWORKS = ['single_bin_scorer', 'multi_dim_scorer', 'multi_head_scorer']

"""

TEST SUITES

"""

TEST_SUITE_NAMES = ['magic-spring', 'healpix-grid', 'delve', 'gw-followup']

"""

MAGIC-SPRING OBSERVING DATES

"""

MS_OBSERVING_DATES = ['2026-04-09-half1', '2026-04-10-half1', '2026-04-11-half1', '2026-04-12-half1', '2026-05-09-half1', '2026-05-10-half1', '2026-06-08-half1', '2026-06-09-half1']