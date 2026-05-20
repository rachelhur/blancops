import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch
import torch.nn.functional as F
import gymnasium as gym

import os

from blancops.configs.constants import get_workspace_dir
from blancops.configs.rl_schema import load_and_validate
from blancops.rl.offline_runner import OfflineRunner
from blancops.data.lookup_tables import LookupTables
from blancops.rl.agent import Agent
from blancops.utils.sys_utils import seed_everything
from blancops.io.logger_utils import configure_logger
from blancops.utils.sys_utils import get_system_device
from blancops.configs.constants import *
from blancops.environment.sim_env import SimBlancoEnv
from blancops.plotting.schedule_viz import *
from blancops.rl.registry import build_algorithm


import logging
logger = logging.getLogger(__name__)

import argparse
import re

from pathlib import Path

def _get_fieldfilter_nvisits(visit_history_df, field_lookup):
    fieldfilter2nvisits = np.zeros(shape=(len(field_lookup), len(FILTER2IDX)), dtype=np.int32)
    filt_idxs = visit_history_df['filter'].map(FILTER2IDX)
    fids = visit_history_df['field_id'].values
    len(fids)
    np.add.at(fieldfilter2nvisits, (fids, filt_idxs), 1)
    return fieldfilter2nvisits

def calc_visit_history_features(filepath, field_lookup):
    visit_history_df = pd.read_csv(filepath)
    try:
        fid_visit_history = visit_history_df['field_id'].values
        s_visits_cur = np.bincount(fid_visit_history, minlength=len(field_lookup))
        fieldfilter2nvisits = _get_fieldfilter_nvisits(visit_history_df, field_lookup)
        priorities = field_lookup['PRIORITY']
    except Exception as e:
        print(f"Exception error: {e}")
    return s_visits_cur, fieldfilter2nvisits, priorities

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
    parser.add_argument('--night1_ts_start', type=int, default=None, help="Timestamp at which to start the first night. Defaults to time defined by `sun_horzion`")
    parser.add_argument('--load_obs_history', action='store_true')
    parser.add_argument('--obs_history_filename', type=str, default='fake_night_2_observations.csv', help='A csv file with column `field_id` defined by field_lookup')
    parser.add_argument('--sun_horizon', type=float, default=-12, help="How low below horizon sun needs to be for observing (in deg). Default is -12.")
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
    cfg_dir = args.trained_model_dir
    assert os.path.exists(cfg_dir), f"Directory {cfg_dir} does not exist"
    cfg = load_and_validate(cfg_dir / "resolved_config.yaml")
    nside = cfg['data']['nside']
    action_space = cfg['data']['action_space']
    
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
        
    # ---------------------------------
    # SETUP LOGGER
    # ---------------------------------
    # logger = setup_logger_old(save_dir=Path(schedule_outdir).resolve(), logging_filename='eval.log', logging_level=args.logging_level)
    logger = configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=True,
        outdir=schedule_outdir,
        filename='offline_schedule.log',
        use_tqdm=True
    )

    logger.info(f"Saving results in {schedule_outdir}")

    # Seed and get device
    seed_everything(args.seed)
    device = get_system_device()
    s_visits_cur = None
    fieldfilter2nvisits = None
    
    # Load lookup tables
    if args.schedule_name in TEST_SUITE_NAMES:
        print(f"Using test suite {args.schedule_name} with predefined lookup table paths and observing nights")
        schedule_name = args.schedule_name
        workspace_dir = get_workspace_dir()
        lookup_dirpath = workspace_dir / "data" / "test_suite" / schedule_name
        for f in ['field_lookup.json', 'fid2radec.json']:
            filepath = lookup_dirpath / f
            assert os.path.exists(filepath), f"Path to {f} not found in {lookup_dirpath}"

        field_lookup = pd.read_json(lookup_dirpath / "field_lookup.json")
        # fid2radec = field_lookup[['ra', 'dec']].to_numpy()
        
        if schedule_name == 'gw-followup':
            print(f"Using GW followup test suite with predefined observing nights based on {args.observing_nights} set")
            if args.observing_nights[0] == 'good':
                print('USING GOOD GW FOLLOWUP NIGHTS')
                observing_night_strs = GW_OBSERVING_DATES_GOOD
            elif args.observing_nights[0] == 'DD-night':
                observing_night_strs = DD_NIGHT
            elif args.observing_nights[0] == 'bad':
                observing_night_strs = GW_OBSERVING_DATES_BAD
            else:
                observing_night_strs = args.observing_nights
        elif schedule_name == 'healpix-grid':
            observing_night_strs = HP_OBSERVING_DATES
        elif schedule_name == 'magic-spring':
            observing_night_strs = MS_OBSERVING_DATES
            lookups = LookupTables.load_from_dir(lookup_dirpath, include_historic=True)

        elif schedule_name == 'magic-spring-1':
            if args.load_obs_history:
                observing_night_strs = MS_OBSERVING_DATES[1:] # check second night onwards
                obs_history_filepath = lookup_dirpath / args.obs_history_filename
                s_visits_cur, fieldfilter2nvisits, priorities = calc_visit_history_features(obs_history_filepath, field_lookup)
        elif schedule_name == 'delve':
            observing_night_strs = args.observing_nights
        else:
            raise ValueError(f"Test suite {schedule_name} not recognized. Must be one of {TEST_SUITE_NAMES}")
    else:
        lookup_dirpath = args.field_lookup_dir.resolve()
        schedule_name = args.schedule_name

    for f in ['field_lookup.json', 'fid2radec.json']:
        filepath = lookup_dirpath / f
        assert os.path.exists(filepath), f"Path to {f} not found in {lookup_dirpath}"

    field_lookup = pd.read_json(lookup_dirpath / "field_lookup.json")

    # Check that field_lookup has all required columns needed to run environment
    required_columns = ['field_id', 'exptime', 'ra', 'dec', 'n_visits', 'filter'] # 'dithers','object', 'priorities'
    for col in required_columns:
        assert col in field_lookup.keys(), f"Column '{col}' not found in field_lookup.json"
    
    logger.info("Setting up agent...")
    algorithm = build_algorithm(cfg, device)
    algorithm.load(Path(args.trained_model_dir) / 'best_weights.pt')
    
    # BUILD AGENT
    agent = Agent(algorithm=algorithm, cfg=cfg, lookups=lookups, field_choice_method=args.field_choice_method)
    # BUILD RUNNER
    runner = OfflineRunner(
        agent=agent,
        algorithm=algorithm,
        cfg=cfg,
        lookups=lookups,
        num_episodes=1,
        outdir=schedule_outdir,
        save_SISPI=False,
        schedule_chunk_size=0,
    )
    # LOAD POLICY WEIGHTS
    algorithm = build_algorithm(cfg, device)
    algorithm.load(Path(args.trained_model_dir) / 'best_weights.pt')

    # agent.load(cfg_dir / 'best_weights.pt')

    # Initialize environment
    logger.info("Setting up environment...")
    env_name = 'SimBlancoEnv-v0'
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=SimBlancoEnv,
    )

    #pyephem requires sun horizon to be in string format if degrees (float if radians)
    sun_horizon = str(args.sun_horizon)

    # CREATE ENVIRONMENT
    env = gym.make(id=f"gymnasium_env/{env_name}", cfg=cfg, data_dir=lookup_dirpath,
                    observing_night_strs=observing_night_strs, horizon=sun_horizon, max_nights=args.max_nights, airmass_limit=args.airmass_lim,
                    s_visits_cur=s_visits_cur, s_filter_visits_cur=fieldfilter2nvisits, night1_ts_start=args.night1_ts_start, field_priorities_arr=None)
    # fid2radec = np.array([[ra, dec] for ra, dec in zip(field_lookup['ra'].values(), field_lookup['dec'].values())])

    # Evaluate
    diagnostics = runner.run(env=env)
    
    logger.info("Generating plots...")
    save_survey_diagnostics(diagnostics, save_dir=schedule_outdir, field_lookup=field_lookup, nside=nside, action_space=action_space)
    save_gifs(schedule_path=schedule_outdir / 'full_survey_schedule.csv', save_dir=schedule_outdir, 
              do_fieldbin=True, do_bin=False, do_mollefield=False, do_ortho=False, action_space=action_space, nside=nside, 
              fid2radec_filepath=lookup_dirpath / "fid2radec.json")

    if not args.no_night_diagnostics:
        save_nightly_diagnostics(eval_metrics=diagnostics, observing_night_strs=observing_night_strs, schedule_outdir=schedule_outdir, action_architecture=cfg['model']['action_architecture'],
                            nside=nside, lookup_dirpath=lookup_dirpath, env=env, num_actions=cfg['data']['num_actions'], action_space=action_space,
                            do_gifs=not args.do_night_gifs)
        
if __name__ == "__main__":
    main()