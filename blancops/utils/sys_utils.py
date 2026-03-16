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
    """Determines the active workspace. Priority: (1) environment variable (2) pointer file (saved after running model-init) (3) default=$HOME.blancops"""
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

def generate_global_config(output_fn="global_config.json"):
    workspace_dir = get_workspace_dir()
    workspace_dir_str = str(workspace_dir)
    outpath = workspace_dir / "configs" / output_fn
    config_data = {
        "features": {
            "CYCLICAL_FEATURE_NAMES": [
                "ra", "az", "ha", "lst"
            ],
            "MAX_NORM_FEATURE_NAMES": [
                "el", "dec"
            ],
            "INVERSE_NORM_FEATURES_NAMES": [
                "airmass"
            ],
            "ANG_DISTANCE_NORM_FEATURE_NAMES": [
                "distance"
            ],
            "GLOBAL_FEATURES": [
                "ra", "dec", "az", "el", "airmass", "ha", "sun_ra", "sun_dec",
                "sun_az", "sun_el", "moon_ra", "moon_dec", "moon_az", "moon_el",
                "filter_wave", "lst", "time_fraction_since_start"
            ],
            "BIN_FEATURES": [
                "ha", "airmass", "moon_distance", "az", "el", "ra", "dec",
                "angular_distance_to_pointing", "night_num_unvisited_fields",
                "night_num_incomplete_fields", "night_min_tiling",
                "survey_num_unvisited_fields", "survey_num_unvisited_fields_u",
                "survey_num_unvisited_fields_r", "survey_num_unvisited_fields_g",
                "survey_num_unvisited_fields_i", "survey_num_unvisited_fields_z",
                "survey_num_unvisited_fields_Y", "survey_num_incomplete_fields",
                "survey_num_incomplete_fields_u", "survey_num_incomplete_fields_r",
                "survey_num_incomplete_fields_g", "survey_num_incomplete_fields_i",
                "survey_num_incomplete_fields_z", "survey_num_incomplete_fields_Y",
                "survey_min_tiling", "survey_min_tiling_u", "survey_min_tiling_r",
                "survey_min_tiling_g", "survey_min_tiling_i", "survey_min_tiling_z",
                "survey_min_tiling_Y"
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
            "FIELDFILTER2MAXVISITS": "fieldfilter2nvisits.pkl"
        },
        "paths": {
            "TRAIN_DIR": f"{workspace_dir_str}/data/train/",
            "BLANCOPS": f"{workspace_dir_str}/",
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
    logger = logging.getLogger()
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
        return logger
    
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
