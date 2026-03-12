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
from blancops.core_rl.agents import Agent
from blancops.utils.sys_utils import seed_everything, load_global_config, load_model_config, get_workspace_dir
from blancops.algorithms.factory import setup_algorithm
from blancops.utils.sys_utils import setup_logger, get_device
from blancops.data_processing.data_processing import load_raw_data_to_dataframe, get_nautical_twilight, NUM_FILTERS
from blancops.core_rl.environments import OnlineDECamEnv
from blancops.data_processing.data_processing import expand_feature_names_for_cyclic_norm
from datetime import datetime, timedelta

import logging
logger = logging.getLogger(__name__)

import argparse
import re

from pathlib import Path

def save_schedule(eval_metrics, save_dir, night_idx, make_gifs=True, nside=None, is_azel=False, whole=False, field2radec_filepath=None):
    # Save timestamps, field_ids, and bin numbers
    bin_space = 'azel' if is_azel else 'radec'
    assert os.path.exists(save_dir)

    timestamps = np.array(eval_metrics['ep-0']['timestamp'][f'night-{night_idx}']).astype(np.int32)
    bins = np.array(eval_metrics['ep-0']['bin'][f'night-{night_idx}']).astype(np.int32)
    fids = np.array(eval_metrics['ep-0']['field_id'][f'night-{night_idx}']).astype(np.int32)

    real_obs_mask = (bins != -1) & (bins != -2)
    
    schedule_full = {
        'agent_timestamp': timestamps[real_obs_mask],
        'agent_field_id': fids[real_obs_mask],
        'agent_bin_id': bins[real_obs_mask],
    }

    df = pd.DataFrame(data={k: pd.Series(v) for k, v in schedule_full.items()}).fillna(0).astype(int)

    output_filepath = save_dir / "schedule.csv"
    df.to_csv(output_filepath, index=False)

    # schedule = pd.read_csv(output_filepath)
    logger.info("Creating fieldbin movies")
    # Create binfield movies

    plot_schedule_from_file(
        outfile=save_dir / "agent_fieldbin_schedule.gif",
        schedule_file=output_filepath,
        plot_type='fieldbin',
        nside=nside,
        fields_file=field2radec_filepath,
        whole=False,
        compare=False,
        expert=False,
        is_azel=bin_space=='azel',
        mollweide=False,
    )

    if make_gifs:
        # Create fields movies
        logger.info("Creating field movies")
        if not is_azel:
            plot_schedule_from_file(
                outfile=save_dir / "expert_field_schedule.gif",
                schedule_file=output_filepath,
                plot_type='field',
                nside=nside,
                fields_file=field2radec_filepath,
                whole=False,
                compare=False,
                expert=True,
                is_azel=bin_space=='azel',
                mollweide=False,
            )

        plot_schedule_from_file(
            outfile=save_dir / "agent_bin_schedule.gif",
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=field2radec_filepath,
            whole=False,
            compare=False,
            expert=False,
            is_azel=bin_space=='azel',
            mollweide=False,
        ) 

        if bin_space == 'radec':
            # Mollefield
            logger.info("Creating static plots")
            plot_schedule_from_file(
                outfile=save_dir / "mollweide.png",
                schedule_file=output_filepath,
                plot_type='bin',
                nside=nside,
                fields_file=field2radec_filepath,
                whole=True,
                compare=True,
                expert=True,
                is_azel=bin_space=='azel',
                mollweide=True,
            )  
            plot_schedule_from_file(
                outfile=save_dir / "ortho.png",
                schedule_file=output_filepath,
                plot_type='bin',
                nside=nside,
                fields_file=field2radec_filepath,
                whole=True,
                compare=True,
                expert=True,
                is_azel=bin_space=='azel',
                mollweide=False,
            )  

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    # Test data selection
    # parser.add_argument('--SISPI_file', type=str, help='Path to a SISPI-like json file with a list of fields.')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of model. Options are those in models directory')
    parser.add_argument('-n', '--schedule_name', type=str, default='schedule', help='Name of schedule (acts as subdir in model directory)')
    parser.add_argument('-d', '--observing_night', type=str, default='2026-06-23', help="First observing night. Format YY-MM-DD")
    parser.add_argument('-f', '--field_lookup_dir', type=str, required=True, help='field lookup dir')
    parser.add_argument('-l', '--logging_level', type=str, default='info', help='Logging level. Options: info, debug')
    parser.add_argument('-c', '--field_choice_method', type=str, default='random', help="Options: random, interp")
    parser.add_argument('-g', '--make_gifs', action='store_true', help="Whether to create the set of gifs. Currently can only choose to make all or none.")

    # Evaluation hyperparameters
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of evaluation episodes to run')
    parser.add_argument('--max_nights', type=int, default=5, help='Maximum number of nights')
    # Parse args
    args = parser.parse_args()
    args_dict = vars(args)

    # Get configs
    global_cfg = load_global_config()
    workspace = get_workspace_dir()
    cfg_dir = workspace / "models" / args.model_name 
    assert os.path.exists(cfg_dir), f"Directory {cfg_dir} does not exist"
    cfg = load_model_config(cfg_dir / "config.json")
    
    # Define eval outdir
    date_postfix = args.observing_night
    
    schedule_name = f"{args.schedule_name}_{date_postfix}_v0"
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
                schedule_name = f"{schedule_name}_1"
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

    # Print args
    logger.warning("Experiment parameters:")
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")

    # Seed and get device
    seed_everything(args.seed)
    device = get_device()

    # Load lookup tables
    for f in ['lookup.json', 'field2radec.json']:
        path = workspace / 'data' / 'lookups' / args.field_lookup_dir
        assert os.path.exists(path / f), f"Path to {f} not found in {path}"

    with open(args.field_lookup_dir + 'lookup.json', 'r') as f:
        field_lookup = json.load(f)
    with open(args.field_lookup_dir + 'field2radec.json') as f:
        field2radec = json.load(f)
    
    # Check that field_lookup has all required columns needed to run environment
    required_columns = ['field_id', 'exptime', 'ra', 'dec', 'n_visits', 'filters'] # 'dithers','object', 'priorities'
    for col in required_columns:
        assert col in field_lookup.keys(), f"{col} not found in lookup.json"
    
    nside = cfg['data']['nside']

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
                                activation=cfg['model']['activation']
                                )
    logger.debug('policy net structure: \n algorithm.policy_net')
    
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
        entry_point=OnlineDECamEnv,
    )

    # Creat env
    env = gym.make(id=f"gymnasium_env/{env_name}", cfg=cfg, gcfg=global_cfg, lookup_path=args.field_lookup_dir + 'lookup.json',
                    night_str=args.observing_night, horizon='-12', max_nights=args.max_nights)
    field2nvisits = {int(fid): n for fid, n in field_lookup['n_visits'].items()}
    field2radec = {int(fid): (field_lookup['ra'][fid], field_lookup['dec'][fid]) for fid in field_lookup['ra'].keys()}

    # Evaluate
    agent.evaluate(env=env, cfg=cfg, num_episodes=1, field_choice_method='random', eval_outdir=schedule_outdir,
              field2nvisits=field2nvisits, field2radec=field2radec)

    # Load results
    with open(schedule_outdir / 'eval_metrics.pkl', 'rb') as f:
        eval_metrics = pickle.load(f)

    logger.info("Generating evaluation plots...")
    
    ep_num = 0
    metrics = eval_metrics[f'ep-{ep_num}']
    current_night = datetime.strptime(args.observing_night, "%Y-%m-%d")
    for night_idx in range(len(metrics['timestamp'].keys())):
        date_str = f"{current_night.year}-{current_night.month}-{current_night.day}"
        logger.info(f'Drawing plots for night {date_str}')
        night_dir = schedule_outdir / date_str
        if not os.path.exists(night_dir):
            os.makedirs(night_dir)

        # Mask zenith observations in plotting
        real_obs_mask = np.array(metrics['field_id'][f'night-{night_idx}']) != -1
        real_obs_mask &= np.array(metrics['field_id'][f'night-{night_idx}']) != -2
        
        timestamps = metrics['timestamp'][f'night-{night_idx}']
        field_ids = metrics['field_id'][f'night-{night_idx}']
        bin_nums = metrics['bin'][f'night-{night_idx}']

        night_ts = env.unwrapped._night_dt.timestamp()
        sunset_time = math.ceil(get_nautical_twilight(night_ts, 'set', env.unwrapped.horizon))
        timestamps = (timestamps - sunset_time) / 3600
    
        # Plot bins vs timestamp        
        fig_b, axb = plt.subplots()
        axb.plot(timestamps[real_obs_mask],
                bin_nums[real_obs_mask],
                      marker='o', label='pred', alpha=.5)
        axb.legend()
        axb.set_xlabel('Hours since sunset \n (-10 deg)')
        axb.set_ylabel('bin')
        fig_b.suptitle(date_postfix)
        fig_b.tight_layout()
        fig_b.savefig(night_dir / 'bin_vs_step.png')
        plt.close()

        # Plot state features vs timestamp for first episode
        fig, axs = plt.subplots(len(env.unwrapped.global_feature_names), figsize=(10, len(env.unwrapped.global_feature_names)*5))
        for i, feature_row in enumerate(np.array(metrics['glob_observations'][f'night-{night_idx}']).T[:len(env.unwrapped.global_feature_names)]):
            feat_name = env.unwrapped.global_feature_names[i]
            if feat_name == 'airmass':
                feature_row = 1 / feature_row
            elif 'dec' in feat_name or 'el' in feat_name:
                feature_row = feature_row * (np.pi/2)
            elif 'distance' in feat_name:
                feature_row = feature_row * np.pi

            axs[i].plot(timestamps[real_obs_mask], feature_row[real_obs_mask], label='policy roll out', marker='o')
            axs[i].set_title(feat_name)
            axs[i].set_xlabel('Hours since sunset \n (-10 deg)')
            axs[i].legend()
        fig.tight_layout()
        fig.savefig(night_dir / 'state_features_vs_time.png')
        plt.close()

        # Plot most frequently visited bin features vs timestamp
        if cfg['model']['grid_network'] is not None:
            _bins_vis_tonight = np.array(bin_nums).astype(int)
            _bincounts = np.bincount(_bins_vis_tonight[real_obs_mask], minlength=cfg['data']['num_actions'])
            _most_common_bin = np.argmax(_bincounts)
            normed_feature_names = expand_feature_names_for_cyclic_norm(env.unwrapped.base_bin_feature_names, env.unwrapped.cyclical_feature_names)
            fig, axs = plt.subplots(len(normed_feature_names), figsize=(10, len(normed_feature_names)* 5))
            for i, feat_row in enumerate(np.array(metrics['bin_observations'][f'night-{night_idx}']).T[:, _most_common_bin, :]):
                feat_name = normed_feature_names[i]
                # unnormalize observations to compare to expert values
                if feat_name == 'airmass':
                    feat_row = 1 / feat_row
                elif 'dec' in feat_name or 'el' in feat_name:
                    feat_row = feat_row * (np.pi/2)
                elif 'distance' in feat_name:
                    feat_row = feat_row * np.pi
                axs[i].plot(timestamps[real_obs_mask], feat_row[real_obs_mask], label='policy roll out', marker='o')
                axs[i].set_title(feat_name)
                axs[i].set_xlabel('Hours since sunset \n (-10 deg)')
                axs[i].legend()
            fig.tight_layout()
            fig.savefig(night_dir / 'bin_features_vs_time.png')

        # Plot static bin and field radec scatter plots
        bin2coord = {int(i): (lon, lat) for i, (lon, lat) in enumerate(zip(env.unwrapped.hpGrid.lon, env.unwrapped.hpGrid.lat))}
        eval_bin_radecs = np.array([bin2coord[bin_num] for bin_num in bin_nums[real_obs_mask]])
        eval_field_radecs = np.array([[field_lookup['ra'][str(field_id)], field_lookup['dec'][str(field_id)]] for field_id in field_ids[real_obs_mask]])
        
        # Plot bins
        if len(eval_bin_radecs) == 0:
            current_night += timedelta(days=1)

            print(eval_bin_radecs)
            continue
        fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
        axs[1].scatter(eval_bin_radecs[:, 0], eval_bin_radecs[:, 1], label='agent', cmap='Blues', c=np.arange(len(eval_bin_radecs)))
        for ax in axs:
            ax.set_xlabel('x (ra or az)')
            ax.legend()
        axs[0].set_ylabel('y (dec or el)')
        fig.suptitle(f'Bins - night {night_idx}')
        fig.savefig(night_dir / "bins_ra_vs_dec.png")
        plt.close()
        
        # Plot fields
        fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
        axs[1].scatter(eval_field_radecs[:, 0], eval_field_radecs[:, 1], label='agent', cmap='Blues', c=np.arange(len(eval_field_radecs)), s=10)
        for ax in axs:
            ax.set_xlabel('ra')
            ax.legend() 
        axs[0].set_ylabel('dec')
        fig.suptitle(f'Fields - night {night_idx}')
        fig.savefig(night_dir / "fields_ra_vs_dec.png")
        plt.close()

        logger.info(f'Creating schedule gif for {night_idx}th night')
        save_schedule(eval_metrics=eval_metrics, save_dir=night_dir, night_idx=night_idx, nside=nside, make_gifs=args.make_gifs, 
                      is_azel='azel' in cfg['data']['bin_space'], field2radec_filepath=workspace / 'data' / 'lookups' / args.field_lookup_dir / 'field2radec.json')
        current_night += timedelta(days=1)

if __name__ == "__main__":
    main()