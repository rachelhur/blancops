import os
from pathlib import Path

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

LOOKUPS = {
    "DECFITS": "decam-exposures-20251211.fits",
    "FIELD2RADEC": "field2radec.json",
    "FIELD2NAME": "field2name.json",
    "FIELD2MAXVISITS_TRAIN": "field2nvisits_default1.json",
    "FIELD2MAXVISITS_EVAL": "field2nvisits_default0.json",
    "NIGHT2FIELDVISITS": "night2fieldhistory.pkl",
    "NIGHT2FILTERVISITS": "night2filterhistory.pkl",
    "FIELD2FILTERS": "field2filters.pkl",
    "FIELDFILTER2MAXVISITS": "fieldfilter2nvisits.pkl",
    "FILTER_TARGET_COUNTS": "target_counts_per_filter.pkl"
}

TRAIN_DATA_DIR = PATHS["TRAIN_DIR"]
TRAIN_DATA_PATH = TRAIN_DATA_DIR / LOOKUPS["DECFITS"]

GLOBAL_FEATURES = [
    "t_night", 
    "t_survey", 
    "lst", 
    "ha", 
    "el", 
    "airmass", 
    "sun_ra", "sun_dec", "sun_az", "sun_el", 
    "moon_ra", "moon_dec", "moon_az", "moon_el",
    "survey_num_unvisited_fields", # ie, survey progress dim
    "survey_num_incomplete_fields", # ie, survey progress dim
    "survey_min_tiling", # survey progress 
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

CYCLICAL_FEATURE_NAMES = ["ra", "az", "ha", "lst"]
SIN_NORM_FEATURE_NAMES = ["el", "dec"]
LOG_NORM_FEATURE_NAMES = ["fwhm", "urgency"]
FRACTIONAL_FEATURE_NAMES = ["moon_phase", "t_night", "t_survey", "survey_num_visits_done"]
Z_SCORE_NORM_FEATURE_NAMES = ["airmass", "pointing_distance", "moon_distance", "sky_brightness", "fwhm", "delta_az", "delta_el", "urgency"]
LOCAL_MEAN_Z_SCORE_FEATURE_NAMES = ["rel_num_unvisited_fields", "rel_num_incomplete_fields", "rel_min_tiling", "rel_moon_distance", "rel_ha"]