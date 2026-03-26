import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

import torch
import torch.nn.functional as F
import gymnasium as gym

import os
import pickle
import json

from blancops.plotting.plotting import plot_schedule_from_file
from blancops.core_rl.agent import Agent
from blancops.utils.sys_utils import seed_everything, load_global_config, load_model_config, get_workspace_dir
from blancops.algorithms.factory import setup_algorithm
from blancops.utils.sys_utils import setup_logger, get_device
from blancops.data_processing.data_processing import load_raw_data_to_dataframe, expand_feature_names_for_cyclic_norm
from blancops.data_processing.constants import *
from blancops.core_rl.environments import OnlineBlancoEnv
from blancops.data_processing.features import get_nautical_twilight
from datetime import datetime, timedelta
from blancops.plotting.schedule_viz import plot_static_diagnostics, save_schedule

import logging
logger = logging.getLogger(__name__)

import argparse
import re

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    # Test data selection
    # parser.add_argument('--SISPI_file', type=str, help='Path to a SISPI-like json file with a list of fields.')
    parser.add_argument('-t', '--trained_model_dir', type=Path, required=True, help='Relative path to trained model directory')
    parser.add_argument('-n', '--schedule_name', type=str, default='schedule', help='Name of schedule (acts as subdir in model directory)')
    parser.add_argument('-d', '--observing_nights', type=str, nargs='*', default='2026-06-23-full', help="First observing night. Format YY-MM-DD")
    parser.add_argument('-f', '--field_lookup_dir', type=Path, default=None, required=False, help='relative path to field lookup dir')
    parser.add_argument('-l', '--logging_level', type=str, default='info', help='Logging level. Options: info, debug')
    parser.add_argument('-c', '--field_choice_method', type=str, default='random', help="Options: random, interp")
    parser.add_argument('-g', '--make_gifs', action='store_true', help="Whether to create the set of gifs. Currently can only choose to make all or none.")

    # Evaluation hyperparameters
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of evaluation episodes to run')
    parser.add_argument('--max_nights', type=int, default=0, help='Maximum number of nights')
    # Parse args
    args = parser.parse_args()
   
    if (args.schedule_name not in TEST_SUITE_NAMES) and (args.field_lookup_dir is None):
        raise AssertionError(f"Must pass `field_lookup_dir` or specify a test from {TEST_SUITE_NAMES}")

    # Get configs
    gcfg = load_global_config()
    cfg_dir = args.trained_model_dir
    assert os.path.exists(cfg_dir), f"Directory {cfg_dir} does not exist"
    cfg = load_model_config(cfg_dir / "config.json")
    nside = cfg['data']['nside']
    
    # Define eval outdir
    schedule_name = f"{args.schedule_name}_v0"
    schedule_outdir = cfg_dir / schedule_name
    if not os.path.exists(schedule_outdir):
        os.makedirs(schedule_outdir)
    else:
        while os.path.exists(schedule_outdir):
            # Match any string ending in digits. 
            # Group 1 captures the prefix (e.g., "eval_2026-03-06_")
            # Group 2 captures the number suffix (e.g., "0")
            match = re.search(r"^(.*?)(\d+)$", schedule_name)
            
            if match:
                base_name = match.group(1)
                num_group = int(match.group(2))
                schedule_name = f"{base_name}{num_group + 1}"
            else:
                # Fallback just in case the user provided a custom name without a number
                schedule_name = f"{schedule_name}_v0"
            schedule_outdir = cfg_dir / schedule_name
        os.makedirs(schedule_outdir)
        
    # Set up logging
    logger = setup_logger(save_dir=Path(schedule_outdir).resolve(), logging_filename='eval.log', logging_level=args.logging_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pytorch").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    logging.getLogger("fontconfig").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)

    logger.info(f"Saving results in {schedule_outdir}")

    # Seed and get device
    seed_everything(args.seed)
    device = get_device()

    # Load lookup tables
    if args.schedule_name in TEST_SUITE_NAMES:
        test_name = args.schedule_name
        workspace_dir = get_workspace_dir()
        lookup_dirpath = workspace_dir / "data" / "test_suite" / test_name
        observing_night_strs = np.array(MS_OBSERVING_DATES)
    else:
        lookup_dirpath = args.field_lookup_dir.resolve()

    for f in ['field_lookup.json', 'field2radec.json']:
        filepath = lookup_dirpath / f
        assert os.path.exists(filepath), f"Path to {f} not found in {lookup_dirpath}"
    with open(lookup_dirpath / "field_lookup.json", 'r') as f:
        field_lookup = json.load(f)
    with open(lookup_dirpath / "field2radec.json") as f:
        field2radec = json.load(f)
    
    # Check that field_lookup has all required columns needed to run environment
    required_columns = ['field_id', 'exptime', 'ra', 'dec', 'n_visits', 'filter'] # 'dithers','object', 'priorities'
    for col in required_columns:
        assert col in field_lookup.keys(), f"Column '{col}' not found in field_lookup.json"
    

    logger.info("Setting up agent...")
    algorithm = setup_algorithm(algorithm_name=cfg['model']['algorithm'], 
                                num_actions=cfg['data']['num_actions'],
                                num_filters=NUM_FILTERS,
                                n_global_features = cfg['data']['state_dim'],
                                n_bin_features=cfg['data']['bin_state_dim'],
                                grid_network=cfg['model']['grid_network'],
                                loss_fxn=cfg['model']['loss_function'],
                                hidden_dim=cfg['train']['hidden_dim'], lr=cfg['train']['lr'], lr_scheduler=cfg['train']['lr_scheduler'], 
                                device=device, lr_scheduler_kwargs=cfg['train']['lr_scheduler_kwargs'], lr_scheduler_epoch_start=cfg['train']['lr_scheduler_epoch_start'], 
                                lr_scheduler_num_epochs=cfg['train']['lr_scheduler_num_epochs'],
                                gamma=cfg['model']['gamma'], 
                                tau=cfg['model']['tau'],
                                activation=cfg['model']['activation'],
                                cql_alpha=cfg['model'].get('cql_alpha', None),
                                nside=cfg['data'].get('nside', None),
                                bin_space=cfg['data']['bin_space']
                                )
    
    agent = Agent(
        algorithm=algorithm,
        train_outdir=cfg_dir,
    )
    agent.load(cfg_dir / 'best_weights.pt')

    # Initialize environment
    logger.info("Setting up environment...")
    env_name = 'OnlineDECamEnv-v0'
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=OnlineBlancoEnv,
    )

    # Creat env
    env = gym.make(id=f"gymnasium_env/{env_name}", cfg=cfg, gcfg=gcfg, lookup_path=lookup_dirpath / 'field_lookup.json',
                    observing_night_strs=observing_night_strs, horizon='-12', max_nights=args.max_nights)
    field2nvisits = {int(fid): n for fid, n in field_lookup['n_visits'].items()}
    field2radec = {int(fid): (field_lookup['ra'][fid], field_lookup['dec'][fid]) for fid in field_lookup['ra'].keys()}

    # Evaluate
    agent.evaluate(env=env, cfg=cfg, num_episodes=1, field_choice_method=args.field_choice_method, eval_outdir=schedule_outdir,
              field2nvisits=field2nvisits, field2radec=field2radec)

    # Load results
    with open(schedule_outdir / 'eval_metrics.pkl', 'rb') as f:
        eval_metrics = pickle.load(f)

    logger.info("Generating evaluation plots...")
    plot_static_diagnostics(eval_metrics=eval_metrics, observing_night_strs=observing_night_strs, schedule_outdir=schedule_outdir, grid_network=cfg['model']['grid_network'],
                            nside=nside, lookup_dirpath=lookup_dirpath, env=env, field_lookup=field_lookup, num_actions=cfg['data']['num_actions'], bin_space=cfg['data']['bin_space'])
if __name__ == "__main__":
    main()