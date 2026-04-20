import sys
import logging
from pathlib import Path
import importlib.resources as pkg_resources
import os

import numpy as np
import random
import torch
import yaml

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

def load_model_config(config_path=None):
    """Loads a custom config if provided, otherwise loads the default from the package."""
    if config_path:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Load the default config bundled inside your package (e.g., blancops/global_config.json)
        config_text = pkg_resources.files('blancops').joinpath('configs/default_model_config.yaml').read_text()
        return yaml.safe_load(config_text)

def save_config(args=None, config_dict=None, outdir=None):
    """Saves the experiment arguments as YAML file."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Convert argparse Namespace to nested dict
    if args is not None:
        config_dict = dict_to_nested(vars(args))
    
    with open(out_path / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, indent=4)

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
