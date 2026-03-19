# Filter wavelengths (nm) according to obztak https://github.com/kadrlica/obztak/blob/c28fab23b09bcff1cf46746eae4ec7e40aeb7f7a/obztak/seeing.py#L22
FILTER2WAVE = {
    # 'u': 380, # not present in train data,
    'g': 480,
    'r': 640,
    'i': 780,
    'z': 920,
    'Y': 990
}

FILTERWAVENORM = 1000.
FILTER2IDX = {k: i for i, k in enumerate(FILTER2WAVE.keys())}
IDX2WAVE = {i: FILTER2WAVE[k] for i, k in enumerate(FILTER2WAVE.keys())}
NUM_FILTERS = len(FILTER2IDX)

