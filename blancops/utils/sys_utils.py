import sys
import logging
from pathlib import Path
import json 
import importlib.resources as pkg_resources
import os

import numpy as np
import random
import torch

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

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

def generate_src_global_config(output_fn="global_config.json"):
    workspace_dir = get_workspace_dir()
    workspace_dir_str = str(workspace_dir)
    outpath = workspace_dir / "blancops" / "configs" / output_fn
    # Contains 
    # (1) Substrings of features that follow respective normalization schemes (if turned on in cfg file)
    # (2) All features implemented so far
    # (3) Train data/lookup paths
    config_data = {
        "features": {
            "CYCLICAL_FEATURE_NAMES": [ # sin/cos transform
                "ra", "az", "ha", "lst"
            ],
            "SIN_NORM_FEATURE_NAMES": [ # sin transform only
                "el", "dec"
            ],
            "LOG_NORM_FEATURE_NAMES": [ # log transform
                "fwhm",
                "urgency"
                # "urgency_r",
                # "urgency_i",
                # "urgency_z",
                # "urgency_Y",
            ],
            "FRACTIONAL_FEATURE_NAMES": [ # transform = 2 * (value - .5) (features in [0,1] are transformed to [-1,1])
                'moon_phase',
                't_night', # unit seconds
                't_survey', # unit nights
                'survey_num_visits_done',
            ],
            "Z_SCORE_NORM_FEATURE_NAMES": [
                'airmass',
                "pointing_distance",
                "moon_distance",
                'sky_brightness',
                'fwhm',
                'delta_az', # delta relative to pointing
                'delta_el',
                'urgency'
            ],
            "LOCAL_MEAN_Z_SCORE_FEATURE_NAMES": [ # transform: ( value - mean(timestamp) ) / entire train data std
                'rel_num_unvisited_fields',
                'rel_num_incomplete_fields',
                'rel_min_tiling',
                'rel_moon_distance',
                'rel_ha'
            ],
            "GLOBAL_FEATURES": [
                "t_night",
                "t_survey",
                "lst", 
                # "ra", # remove to avoid memorization?
                # "dec", 
                # "az",
                "ha",
                "el", 
                "airmass", 
                "ha", 
                "sun_ra", 
                "sun_dec",
                "sun_az", 
                "sun_el", 
                "moon_ra",
                "moon_dec", 
                "moon_az", 
                "moon_el",
                "filter_wave",
                "filter_idx",
                "sky_brightness_g",
                "sky_brightness_r",
                "sky_brightness_i",
                "sky_brightness_z",
                "sky_brightness_Y",
                "survey_num_unvisited_fields", # ie, survey progress dim
                "survey_num_incomplete_fields", # ie, survey progress dim
                "survey_min_tiling", # survey progress 
                "is_filter_g",
                "is_filter_r",
                "is_filter_i",
                "is_filter_z",
                "is_filter_Y",
                "urgency_g",
                "urgency_r",
                "urgency_i",
                "urgency_z",
                "urgency_Y",
                "moon_phase",
                "fwhm",
            ],
            "BIN_FEATURES": [
                # "ha",
                "rel_ha",
                "time_till_set",
                "airmass",
                "moon_distance",
                "rel_moon_distance",
                "is_rising",
                "delta_az",
                "delta_el",
                # "az", 
                "el", 
                # "ra", 
                # "dec",
                "pointing_distance", 
                "night_num_unvisited_fields",
                "night_num_incomplete_fields", 
                "night_min_tiling",
                "survey_num_unvisited_fields", 
                "survey_num_unvisited_fields_r", # ie, survey progress dim
                "survey_num_unvisited_fields_g",
                "survey_num_unvisited_fields_i", 
                "survey_num_unvisited_fields_z",
                "survey_num_unvisited_fields_Y", 
                "survey_num_incomplete_fields",
                "survey_num_incomplete_fields_r", # ie, survey progress dim
                "survey_num_incomplete_fields_g", 
                "survey_num_incomplete_fields_i",
                "survey_num_incomplete_fields_z", 
                "survey_num_incomplete_fields_Y",
                "survey_min_tiling", 
                "survey_min_tiling_r", # survey progress 
                "survey_min_tiling_g", 
                "survey_min_tiling_i", 
                "survey_min_tiling_z",
                "survey_min_tiling_Y",
                "rel_survey_num_unvisited_fields", # ie, survey progress dim
                "rel_survey_num_unvisited_fields_r", # ie, survey progress dim
                "rel_survey_num_unvisited_fields_g",
                "rel_survey_num_unvisited_fields_i", 
                "rel_survey_num_unvisited_fields_z",
                "rel_survey_num_unvisited_fields_Y", 
                "rel_survey_num_incomplete_fields", # ie, survey progress dim
                "rel_survey_num_incomplete_fields_r", # ie, survey progress dim
                "rel_survey_num_incomplete_fields_g", 
                "rel_survey_num_incomplete_fields_i",
                "rel_survey_num_incomplete_fields_z", 
                "rel_survey_num_incomplete_fields_Y",
                "rel_survey_min_tiling", 
                "rel_survey_min_tiling_r", # survey progress 
                "rel_survey_min_tiling_g", 
                "rel_survey_min_tiling_i", 
                "rel_survey_min_tiling_z",
                "rel_survey_min_tiling_Y",
            ]
        },
        "files": {
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
        },
        "paths": {
            "TRAIN_DIR": f"{workspace_dir_str}/data/train/",
            "HEALPIX-GRID": f"{workspace_dir_str}/data/test_suite/healpix-grid/",
            "MAGIC-SPRING": f"{workspace_dir_str}/data/test_suite/magic-spring/",
            "SAMPLE-110825": f"{workspace_dir_str}/data/test_suite/sample-110825/"
        }
    }

    # Write the assembled dictionary to the specified JSON file
    with open(outpath, 'w') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"  [+] Constructed new global config at {outpath}")

def load_global_config(config_path=None):
    """Loads a custom config if provided, otherwise loads the default from the package."""
    if config_path is None:
        workspace_dir = get_workspace_dir()
        config_path = workspace_dir / "configs" / "global_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find global_config.json at {config_path}.\n"
            "Need to run `model-init` to set up workspace"
        )
    with open(config_path, 'r') as f:
        gcfg = json.load(f)
    logger.info(f"Loaded config file at {config_path}")
    return gcfg
    
def load_model_config(config_path=None):
    """Loads a custom config if provided, otherwise loads the default from the package."""
    if config_path:
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Load the default config bundled inside your package (e.g., blancops/global_config.json)
        # This works no matter where the package is installed!
        config_text = pkg_resources.files('blancops').joinpath('configs/default_model_config.json').read_text()
        return json.loads(config_text)

def save_config(args=None, config_dict=None, outdir=None):
    """Saves the experiment arguments as a nested JSON."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Convert argparse Namespace to nested dict
    if args is not None:
        config_dict = dict_to_nested(vars(args))
    
    with open(out_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

def dict_to_nested(data):
    """Converts {'model.lr': 0.1} to {'model': {'lr': 0.1}}"""
    nested = {}
    for key, value in data.items():
        keys = key.split('.')
        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested

def setup_logger(save_dir, logging_filename, logging_level='debug'):
    # Create logger
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger('blancops')
    logger.propagate = False
    
    if logging_level == 'debug':
        logger.setLevel(logging.DEBUG)
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    elif logging_level == 'info':
        logger.setLevel(logging.INFO)
        format = '%(asctime)s - %(levelname)s - %(message)s'
    else:
        raise NotImplementedError

    # Avoid duplicate handlers if called twice
    if logger.handlers:
        raise ValueError("Handler called twice")
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(save_dir / logging_filename, mode='w')
    # file_handler.setLevel(logging.DEBUG)
    
    # Create formatters and add to handlers
    # console_format = logging.Formatter('%(levelname)s - %(message)s')
    format = logging.Formatter(format, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(format)
    file_handler.setFormatter(format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"   
    )
    return device
