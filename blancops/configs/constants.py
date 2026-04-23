from collections import OrderedDict
import os
from pathlib import Path
from typing import Dict, List, Literal

def get_workspace_dir() -> Path:
    """Determines the active workspace. Priority: (1) environment variable (2) pointer file (saved after running model-init) (3) default=`~/.blancops`
    """
    env_workspace = os.getenv("BLANCOPS_WORKSPACE")
    if env_workspace:
        return Path(env_workspace).resolve()
        
    pointer_file = Path.home() / ".blancops_profile"
    if pointer_file.exists():
        saved_path = pointer_file.read_text().strip()
        if saved_path:
            return Path(saved_path).resolve()
            
    # 3. Fallback to default
    return Path.home() / ".blancops"

WORKSPACE = get_workspace_dir()

PATHS = {
    "TRAIN_DIR": Path(WORKSPACE / "data" / "train"),
    "HEALPIX_GRID": Path(WORKSPACE / "data" / "test_suite" / "healpix-grid"),
    "MAGIC_SPRING": Path(WORKSPACE / "data" / "test_suite" / "magic-spring"),
    "SAMPLE_110825": Path(WORKSPACE / "data" / "test_suite" / "sample-110825")
}

TRAIN_DATA_DIR = PATHS["TRAIN_DIR"]
TRAIN_DATA_PATH = TRAIN_DATA_DIR / "decam-exposures-20251211.fits"

GLOBAL_FEATURES = [
    "t_night", 
    "t_survey", 
    "lst", 
    "ha", 
    "el", 
    "airmass", 
    "sun_ra", "sun_dec", "sun_az", "sun_el", 
    "moon_ra", "moon_dec", "moon_az", "moon_el",
    "survey_num_unvisited_fields",
    "survey_num_incomplete_fields",
    "survey_min_tiling",
    "filter_wave", "filter_idx",
    "is_filter_g", "is_filter_r", "is_filter_i", "is_filter_z", "is_filter_Y",
    "urgency_g", "urgency_r", "urgency_i", "urgency_z", "urgency_Y",
    "sky_brightness_g", "sky_brightness_r", "sky_brightness_i", "sky_brightness_z", "sky_brightness_Y",
    "moon_phase",
    "fwhm",
]

BIN_FEATURES = [
    "ha",
    # "time_till_set", 
    "airmass",
    "moon_distance", 
    "rel_ha", "rel_moon_distance", 
    # "is_rising",
    "delta_az", 
    "delta_el", 
    "az", 
    "el", 
    "ra", 
    "dec",
    "pointing_distance", 
    "night_num_unvisited_fields", "night_num_incomplete_fields", "night_min_tiling",
    "survey_num_unvisited_fields",
    "survey_num_unvisited_fields_r", "survey_num_unvisited_fields_g", "survey_num_unvisited_fields_i", "survey_num_unvisited_fields_z", "survey_num_unvisited_fields_Y", 
    "survey_num_incomplete_fields",
    "survey_num_incomplete_fields_r", "survey_num_incomplete_fields_g",  "survey_num_incomplete_fields_i", "survey_num_incomplete_fields_z", "survey_num_incomplete_fields_Y",
    "survey_min_tiling", 
    "survey_min_tiling_r", "survey_min_tiling_g",  "survey_min_tiling_i",  "survey_min_tiling_z", "survey_min_tiling_Y",
    "rel_survey_num_unvisited_fields", "rel_survey_num_unvisited_fields_r", "rel_survey_num_unvisited_fields_g", "rel_survey_num_unvisited_fields_i",  "rel_survey_num_unvisited_fields_z", "rel_survey_num_unvisited_fields_Y", 
    "rel_survey_num_incomplete_fields", "rel_survey_num_incomplete_fields_r", "rel_survey_num_incomplete_fields_g",  "rel_survey_num_incomplete_fields_i", "rel_survey_num_incomplete_fields_z",  "rel_survey_num_incomplete_fields_Y",
    "rel_survey_min_tiling", 
    "rel_survey_min_tiling_r", "rel_survey_min_tiling_g", "rel_survey_min_tiling_i", "rel_survey_min_tiling_z", "rel_survey_min_tiling_Y", 
]


# 1. Define allowed normalization strings to catch typos instantly
NORM_TYPES = Literal['cyclical', 'sin', 'log', 'fractional', 'z_score', 'local_mean_z']

# 2. Define the absolute physical rules of your domain
ALLOWED_NORMS_PER_FEATURE = {
    
    # Telescope coords
    'ra': {'cyclical', 'z_score'},
    'az': {'cyclical', 'z_score'},
    'ha': {'cyclical', 'z_score'},
    'lst': {'cyclical', 'z_score'},
    
    # Sun coords
    'sun_ra': {'cyclical', 'z_score'},
    'sun_az': {'cyclical', 'z_score'},
    'sun_el': {'z_score'},
    'sun_dec': {'z_score'},
    
    # Moon coords
    'moon_ra': {'cyclical', 'z_score'},
    'moon_az': {'cyclical', 'z_score'},
    'moon_el': {'z_score'},
    'moon_dec': {'z_score'},
    
    'moon_distance': {'z_score'},
    'airmass': {'log', 'z_score'},
    'pointing_distance': {'z_score'},
    'sky_brightness': {'z_score'},
    'delta_az': {'z_score'},
    'delta_el': {'z_score'},
    'el': {'z_score'},
    'dec': {'z_score'},
    
    'fwhm': {'log', 'z_score'},
    'urgency': {'log', 'z_score'},
    
    'survey_progress': {'fractional', 'sin', 'z_score'},
    'urgency': {'log', 'z_score'},
    
    'rel_num_unvisited_fields': {'local_mean_z'},
    'rel_num_incomplete_fields': {'local_mean_z'},
    'rel_min_tiling': {'local_mean_z'},
    'rel_moon_distance': {'local_mean_z'},
    'rel_ha': {'local_mean_z'},
    
    't_night': {'fractional'},
    't_survey': {'fractional'},
    'moon_phase': {'fractional'},
    'survey_num_visits_done': {'fractional'},
}

DEFAULT_NORM_MAPPING = {
    
    # Telescope coords
    'ra': ['cyclical'],
    'az': ['cyclical'],
    'ha':['cyclical'],
    'lst': ['cyclical'],
    'el': ['z_score'],
    'dec': ['z_score'],

    # Sun coords
    'sun_ra': ['cyclical'],
    'sun_az': ['cyclical'],
    'sun_el': ['z_score'],
    'sun_dec': ['z_score'],
    
    # Moon coords
    'moon_ra': ['cyclical'],
    'moon_az': ['cyclical'],
    'moon_el': ['z_score'],
    'moon_dec': ['z_score'],
    
    # Image quality
    'moon_distance': ['z_score'],
    'airmass': ['log', 'z_score'],
    'sky_brightness': ['z_score'],
    'delta_az': ['z_score'],
    'delta_el': ['z_score'],
    'fwhm': ['z_score'],
    'urgency': ['z_score'],
    
    # Bin features
    'pointing_distance': ['z_score'],
    'rel_num_unvisited_fields': ['local_mean_z'],
    'rel_num_incomplete_fields': ['local_mean_z'],
    'rel_min_tiling': ['local_mean_z'],
    'rel_moon_distance': ['local_mean_z'],
    'rel_ha': ['local_mean_z'],
    
    't_night': ['fractional'],
    't_survey': ['fractional'],
    'moon_phase': ['fractional'],
    'survey_num_visits_done': ['fractional'],
}

EMPTY_SISPI_DICT = OrderedDict([
    ("object",  None),
    ("seqnum",  None), # 1-indexed
    ("seqtot",  1),
    ("seqid",   ""),
    ("expTime", 90),
    ("RA",      None),
    ("dec",     None),
    ("filter",  None),
    ("count",   1),
    ("expType", "object"),
    ("program", None),
    ("wait",    "False"),
    ("propid",  None),
    ("comment", ""),
])

CYCLICAL_FEATURE_NAMES = ["ra", "az", "ha", "lst"]
# SIN_NORM_FEATURE_NAMES = []
# LOG_NORM_FEATURE_NAMES = ["fwhm", "urgency"]
# FRACTIONAL_FEATURE_NAMES = ["moon_phase", "t_night", "t_survey", "survey_num_visits_done"]
# Z_SCORE_NORM_FEATURE_NAMES = ["airmass", "pointing_distance", "moon_distance", "sky_brightness", "fwhm", "delta_az", "delta_el", "urgency", "el", "dec"]
# LOCAL_MEAN_Z_SCORE_FEATURE_NAMES = ["rel_num_unvisited_fields", "rel_num_incomplete_fields", "rel_min_tiling", "rel_moon_distance", "rel_ha"]