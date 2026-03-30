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
from blancops.plotting.schedule_viz import *

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
    parser.add_argument('--sun_horizon', type=float, default=-12, help="Sun horizon in degrees for determining night start/end. Default is -12.")
    parser.add_argument('--airmass_lim', type=float, default=1.2, help="The agent will only observe if there exist *any* fields below the airmass_lim")
    parser.add_argument('--no_night_diagnostics', action='store_false', help="Whether to skip generating nightly diagnostics plots" )
    parser.add_argument('--do_night_gifs', action='store_true')

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
    bin_space = cfg['data']['bin_space']
    
    # Define eval outdir
    schedule_name = f"{args.schedule_name}_v0"
    schedule_outdir = cfg_dir / schedule_name
    if not os.path.exists(schedule_outdir):
        os.makedirs(schedule_outdir)
    else:
        while os.path.exists(schedule_outdir):
            # Match any string ending in digits. 
            # Group 1: prefix (e.g., "eval_2026-03-06_")
            # Group 2: number suffix
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
        print(f"Using test suite {args.schedule_name} with predefined lookup tables and observing nights")
        test_name = args.schedule_name
        workspace_dir = get_workspace_dir()
        lookup_dirpath = workspace_dir / "data" / "test_suite" / test_name
        if test_name == 'gw-followup':
            print(f"Using GW followup test suite with predefined observing nights based on {args.observing_nights} set")
            if args.observing_nights[0] == 'good':
                print('USING GOOD GW FOLLOWUP NIGHTS')
                observing_night_strs = GW_OBSERVING_DATES_GOOD
            elif args.observing_nights[0] == 'bad':
                observing_night_strs = GW_OBSERVING_DATES_BAD
            else:
                observing_night_strs = args.observing_nights
        elif test_name == 'healpix-grid':
            observing_night_strs = HP_OBSERVING_DATES
        elif test_name == 'magic-spring':
            observing_night_strs = MS_OBSERVING_DATES
        elif test_name == 'delve':
            observing_night_strs = args.observing_nights
        else:
            raise ValueError(f"Test suite {test_name} not recognized. Must be one of {TEST_SUITE_NAMES}")
    else:
        lookup_dirpath = args.field_lookup_dir.resolve()

    for f in ['field_lookup.json', 'field2radec.json']:
        filepath = lookup_dirpath / f
        assert os.path.exists(filepath), f"Path to {f} not found in {lookup_dirpath}"
    with open(lookup_dirpath / "field_lookup.json", 'r') as f:
        field_lookup = json.load(f)
    
    field2radec = pd.read_json(lookup_dirpath / "field2radec.json")
    field2radec = field2radec[['ra', 'dec']].to_numpy()
    field2nvisits = np.array([n for n in field_lookup['n_visits'].values()])

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
                                nside=nside,
                                bin_space=bin_space
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

    #pyephem requires sun horizon to be in string format if degrees (float if radians)
    sun_horizon = str(args.sun_horizon)

    # Creat env

    env = gym.make(id=f"gymnasium_env/{env_name}", cfg=cfg, gcfg=gcfg, data_dir=lookup_dirpath, field2radec=field2radec,
                    observing_night_strs=observing_night_strs, horizon=sun_horizon, max_nights=args.max_nights, airmass_limit=args.airmass_lim)
    # field2radec = np.array([[ra, dec] for ra, dec in zip(field_lookup['ra'].values(), field_lookup['dec'].values())])

    # Evaluate
    eval_metrics = agent.evaluate(env=env, cfg=cfg, num_episodes=1, field_choice_method=args.field_choice_method, eval_outdir=schedule_outdir,
              field2nvisits=field2nvisits, field2radec=field2radec)

    logger.info("Generating plots...")
    save_survey_diagnostics(eval_metrics, save_dir=schedule_outdir, field_lookup=field_lookup, nside=nside, bin_space=bin_space)
    save_gifs(schedule_path=schedule_outdir / 'survey_schedule.csv', save_dir=schedule_outdir, do_fieldbin=True, do_bin=False, do_mollefield=False, do_ortho=False, bin_space=bin_space, nside=nside, field2radec_filepath=lookup_dirpath / "field2radec.json")

    # # Load results
    # with open(schedule_outdir / 'eval_metrics.pkl', 'rb') as f:
    #     eval_metrics = pickle.load(f)

    if not args.no_night_diagnostics:
        save_nightly_diagnostics(eval_metrics=eval_metrics, observing_night_strs=observing_night_strs, schedule_outdir=schedule_outdir, grid_network=cfg['model']['grid_network'],
                            nside=nside, lookup_dirpath=lookup_dirpath, env=env, num_actions=cfg['data']['num_actions'], bin_space=bin_space,
                            do_gifs=not args.do_night_gifs)
        
if __name__ == "__main__":
    main()