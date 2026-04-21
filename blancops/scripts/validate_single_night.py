import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import gymnasium as gym

import json
import pandas as pd
import logging

from blancops.configs.schema import load_and_validate
from blancops.plotting.plotting import plot_schedule_from_file
from blancops.rl.trainer import Trainer
from blancops.utils.sys_utils import seed_everything
from blancops.rl.algorithms.builder import build_algorithm
from blancops.utils.sys_utils import setup_logger, get_device
from blancops.data.preprocessing import load_train_data_to_dataframe
from blancops.data.dataset import load_field2radec_as_numpy
from blancops.environment.validation_env import ValidationBlancoEnv
from blancops.data.dataset import OfflineDataset
from blancops.data.constants import *
from blancops.math import units
from blancops.data.dataset import load_field2radec_as_numpy
from blancops.configs.constants import TRAIN_DATA_PATH, TRAIN_DATA_DIR, LOOKUPS

from blancops.data.features.glob_features import calc_twilight
import logging
logger = logging.getLogger(__name__)

import argparse
from datetime import datetime
import re

from pathlib import Path

def save_schedule(night_metrics, pd_group, save_dir, make_gifs=True, nside=None, is_azel=False, whole=False, bin2pos_filepath=None, field2radec_filepath=None):
    bin2pos_filepath=None
    # Save timestamps, field_ids, and bin numbers
    action_space = 'azel' if is_azel else 'radec'
    assert os.path.exists(save_dir)

    eval_timestamps = night_metrics['timestamp']
    expert_timestamps = pd_group['timestamp'].values
    
    _timestamps = eval_timestamps if len(eval_timestamps) > len(expert_timestamps) else expert_timestamps
    
    schedule_full = {
        'agent_timestamp': eval_timestamps,
        'agent_field_id': night_metrics['field_id'],
        'agent_bin_id': night_metrics['bin'],
        'expert_timestamp': expert_timestamps,
        'expert_field_id': pd_group['field_id'].values,
        'expert_bin_id': pd_group['bin'].values,
        'timestamp': _timestamps
    }

    df = pd.DataFrame(data={k: pd.Series(v) for k, v in schedule_full.items()}).fillna(0).astype(int)

    output_filepath = save_dir + 'schedule.csv'
    df.to_csv(output_filepath, index=False)

    # schedule = pd.read_csv(output_filepath)
    logger.info("Creating fieldbin movies")
    # Create binfield movies
    plot_schedule_from_file(
        outfile=save_dir + 'agent_fieldbin_schedule.gif',
        schedule_file=output_filepath,
        plot_type='fieldbin',
        nside=nside,
        fields_file=field2radec_filepath,
        bins_file=bin2pos_filepath,
        whole=False,
        compare=False,
        expert=False,
        is_azel=action_space=='azel',
        mollweide=False,
    )
    plot_schedule_from_file(
        outfile=save_dir + 'expert_fieldbin_schedule.gif',
        schedule_file=output_filepath,
        plot_type='fieldbin',
        nside=nside,
        fields_file=field2radec_filepath,
        bins_file=bin2pos_filepath,
        whole=False,
        compare=False,
        expert=True,
        is_azel=action_space=='azel',
        mollweide=False,
    )

    if make_gifs:
        # Create fields movies
        logger.info("Creating field movies")
        if not is_azel:
            plot_schedule_from_file(
                outfile=save_dir + 'expert_field_schedule.gif',
                schedule_file=output_filepath,
                plot_type='field',
                nside=nside,
                fields_file=field2radec_filepath,
                bins_file=bin2pos_filepath,
                whole=False,
                compare=False,
                expert=True,
                is_azel=action_space=='azel',
                mollweide=False,
            )
            plot_schedule_from_file(
                outfile=save_dir + 'agent_field_schedule.gif',
                schedule_file=output_filepath,
                plot_type='field',
                nside=nside,
                fields_file=field2radec_filepath,
                bins_file=bin2pos_filepath,
                whole=False,
                compare=False,
                expert=False,
                is_azel=action_space=='azel',
                mollweide=False,
            )

        plot_schedule_from_file(
            outfile=save_dir + 'agent_bin_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=field2radec_filepath,
            bins_file=bin2pos_filepath,
            whole=False,
            compare=False,
            expert=False,
            is_azel=action_space=='azel',
            mollweide=False,
        ) 

        # Create bin movies   
        logger.info("Creating bin movies")
        plot_schedule_from_file(
            outfile=save_dir + 'bin_comparison_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=field2radec_filepath,
            bins_file=bin2pos_filepath,
            whole=False,
            compare=True,
            expert=True,
            is_azel=action_space=='azel',
            mollweide=False,
        )
        plot_schedule_from_file(
            outfile=save_dir + 'expert_bin_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=field2radec_filepath,
            bins_file=bin2pos_filepath,
            whole=False,
            compare=False,
            expert=True,
            is_azel=action_space=='azel',
            mollweide=False,
        )

        if 'radec' in action_space:
            # Mollefield
            logger.info("Creating static plots")
            plot_schedule_from_file(
                outfile=save_dir + 'mollweide.png',
                schedule_file=output_filepath,
                plot_type='bin',
                nside=nside,
                fields_file=field2radec_filepath,
                bins_file=bin2pos_filepath,
                whole=True,
                compare=True,
                expert=True,
                is_azel='azel' in action_space,
                mollweide=True,
            )  
            plot_schedule_from_file(
                outfile=save_dir + 'ortho.png',
                schedule_file=output_filepath,
                plot_type='bin',
                nside=nside,
                fields_file=field2radec_filepath,
                bins_file=bin2pos_filepath,
                whole=True,
                compare=True,
                expert=True,
                is_azel='azel' in action_space,
                mollweide=False,
            )  

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    # Test data selection
    # parser.add_argument('--SISPI_file', type=str, help='Path to a SISPI-like json file with a list of fields.')
    parser.add_argument('-t', '--trained_model_dir', type=Path, required=True, help='Directory of the trained model to evaluate')
    parser.add_argument('-n', '--evaluation_name', type=str, default=None, help='Directory name for this evaluation run')
    parser.add_argument('-y', '--years', type=int, nargs='*', default=None, help='Specific years to include in the test dataset')
    parser.add_argument('-m','--months', type=int, nargs='*', default=None, help='Specific months to include in the test dataset')
    parser.add_argument('-d', '--days', type=int, nargs='*', default=None, help='Specific days to include in the test dataset')
    parser.add_argument('-f', '--filters', type=str, nargs='*', default=None, help='Specific days to include in the test dataset')
    parser.add_argument('-l', '--logging_level', type=str, default='info', help='Logging level. Options: info, debug')
    parser.add_argument('-z', '--start_at_zenith', action='store_true', help='Whether to start evaluation episodes at zenith or not. If False, starts at the first observation of the night.')
    parser.add_argument('--field_choice_method', type=str, default='interp', help="Options: random, interp")
    parser.add_argument('--fits_path', type=str, default='../data/decam-exposures-20251211.fits', help='Path to offline dataset file')
    parser.add_argument('--json_path', type=str, default='../data/decam-exposures-20251211.json', help='Path to offline dataset metadata json file')
    parser.add_argument('--make_gifs', action='store_true', help="Whether to create the set of gifs. Currently can only choose to make all or none.")

    # Evaluation hyperparameters
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of evaluation episodes to run')

    # Parse args
    args = parser.parse_args()
    # Get configs
    args.trained_model_dir = args.trained_model_dir.resolve()
    cfg = load_and_validate(args.trained_model_dir / "resolved_config.json")

    # Define eval outdir
    evaluation_name = args.evaluation_name
    if getattr(args, 'evaluation_name', None) is None:
        date_postfix = datetime.now().strftime("%Y-%m-%d")
        evaluation_name = f"eval_{date_postfix}_0"
    else:
        evaluation_name = args.evaluation_name

    assert os.path.exists(args.trained_model_dir / 'best_weights.pt'), f"There is no best_weights.pt file in {args.trained_model_dir}"
    eval_outdir = os.path.join(args.trained_model_dir, evaluation_name)

    if not os.path.exists(eval_outdir):
        os.makedirs(eval_outdir)
    else:
        while os.path.exists(eval_outdir):
            # Group 1 captures the prefix (e.g., "eval_2026-03-06_")
            # Group 2 captures the number suffix (e.g., "0")
            match = re.search(r"^(.*?)(\d+)$", evaluation_name)
            
            if match:
                base_name = match.group(1)
                num_group = int(match.group(2))
                evaluation_name = f"{base_name}{num_group + 1}"
            else:
                # Fallback just in case the user provided a custom name without a number
                evaluation_name = f"{evaluation_name}_1"
                
            eval_outdir = os.path.join(args.trained_model_dir, evaluation_name)
            
        os.makedirs(eval_outdir)
    eval_outdir += '/'
        
    # Set up logging
    logger = setup_logger(save_dir=Path(eval_outdir).resolve(), logging_filename='eval.log', logging_level=args.logging_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pytorch").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    logging.getLogger("fontconfig").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)

    logger.info("Saving results in " + eval_outdir)

    # Seed everything
    seed_everything(args.seed)

    device = get_device()
    logger.info("Loading raw data...")
    
    df = load_train_data_to_dataframe(TRAIN_DATA_PATH)
    survey_nights_total = df['night'].nunique()
        
    field2radec_filepath = TRAIN_DATA_DIR / LOOKUPS['FIELD2RADEC']
    FIELD2RADEC = load_field2radec_as_numpy(field2radec_filepath)
    
    # with open(field2radec_filepath, 'r') as f:
    #     FIELD2RADEC = json.load(f)

    nside = cfg.data.nside
    zscore_stats = torch.load(Path(args.trained_model_dir) / "z_score_stats.pt")
    rel_norm_stats = torch.load(Path(args.trained_model_dir) / "rel_norm_stats.pt")
    logger.info(f"Loaded z-score stats: {zscore_stats}")
    logger.info(f"Loaded rel_norm_stats: {rel_norm_stats}")
    
    logger.info("Loading test dataset with same config as training dataset...")
    test_dataset = OfflineDataset(
        df=df,
        cfg=cfg,
        specific_years=args.years,
        specific_months=args.months,
        specific_days=args.days,
        specific_filters=args.filters,
        z_score_stats=zscore_stats,
        rel_norm_stats=rel_norm_stats
        ) 
    from scipy.interpolate import CubicSpline

    fwhm_night_interps = []
    for n, ng in test_dataset._df.groupby('night'):
        _ts = ng['timestamp'].values
        _fwhms = ng['fwhm'].values
        cs = CubicSpline(_ts, _fwhms)
        # plt.scatter(_ts, _fwhms, s=1)
        # plt.plot(_ts, cs(_ts), color='red')
        fwhm_night_interps.append(cs)

    
    logger.info("Setting up agent...")
    algorithm = build_algorithm(cfg, device)
    agent = Trainer(
        algorithm=algorithm,
        train_outdir=args.trained_model_dir,
    )
    agent.load(Path(args.trained_model_dir) / 'best_weights.pt')

    # Initialize environment
    logger.info("Setting up environment...")
    env_name = 'OfflineDECamTestingEnv-v0'
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=ValidationBlancoEnv,
    )

    # Creat env
    global_pd_nightgroup = test_dataset._df.groupby('night')
    if not args.start_at_zenith:
        global_pd_nightgroup = global_pd_nightgroup.apply(lambda x: x.iloc[1:]).reset_index(drop=True).groupby('night')
    t_survey_arr = np.asarray([test_dataset._df['t_survey'].unique()[0]]).flatten()
    if len(cfg['data']['bin_features']) > 0:
        night_start_indices = (test_dataset._df.iloc[test_dataset.current_state_idxs].reset_index(drop=True)[test_dataset._df.iloc[test_dataset.current_state_idxs].reset_index(drop=True)['object'] == 'zenith']).index.values
        if not args.start_at_zenith:
            night_start_indices += 1
        night_start_bin_states = test_dataset._prenorm_bin_states[night_start_indices].detach().numpy()
    else:
        night_start_bin_states = None
        
    env = gym.make(id=f"gymnasium_env/{env_name}", cfg=cfg, gcfg=global_cfg, max_nights=None, global_pd_nightgroup=global_pd_nightgroup, \
                   zenith_bin_states=night_start_bin_states, z_score_stats=zscore_stats, rel_norm_stats=rel_norm_stats, t_survey_arr=t_survey_arr, survey_nights_total=survey_nights_total,
                   fwhm_night_interps=fwhm_night_interps)
    
    # Plot predicted action for each state
    cur_idxs = test_dataset.current_state_idxs
    with torch.no_grad():
        q_vals = agent.algorithm.policy.core_net(test_dataset.states[cur_idxs].to(device), test_dataset.bin_states[cur_idxs].to(device) if test_dataset.bin_states is not None else None)
        agent_actions = torch.argmax(q_vals, dim=1).to('cpu').detach().numpy()
    
    exp_actions = test_dataset.actions.detach().numpy()    
    exp_mask = test_dataset.actions != ZENITH_BIN_NUM
    ag_mask = agent_actions != ZENITH_BIN_NUM
    # Get expert and agent actions (bin and filter)
    if 'filter' in cfg['data']['action_space']:
        expert_filters = exp_actions[exp_mask] % NUM_FILTERS
        agent_filters = agent_actions[ag_mask] % NUM_FILTERS
        expert_bins = exp_actions[exp_mask] // NUM_FILTERS
        agent_bins = agent_actions[ag_mask] // NUM_FILTERS
        assert len(expert_filters) == len(agent_filters), f"Shape mismatch: expert filters {expert_filters.shape}, agent_filters {agent_filters.shape}"
    else:
        expert_bins = exp_actions[exp_mask]
        agent_bins = agent_actions[ag_mask]
    
    # Get expert times
    time_idx = np.where(np.array(test_dataset.global_feature_names) == 't_night')[0]
    expert_times = test_dataset.states[test_dataset.next_state_idxs, time_idx].detach().numpy()
    assert len(expert_bins) == len(agent_bins) == len(expert_times), f"Shape mismatch: expert bins {expert_bins.shape}, agent_bins {agent_bins.shape} expert_times {expert_times.shape}"
    
    # Plot expert vs agent actions
    fig, axs = plt.subplots(2, figsize=(10,5), sharex=True)
    axs[0].plot(expert_times, expert_bins, marker='*', alpha=.3, label='true')
    axs[0].plot(expert_times, agent_bins, marker='o', alpha=.3, label='pred')
    axs[0].legend()
    axs[0].set_ylabel('bin number')
    axs[1].plot(expert_times, agent_bins - expert_bins, marker='o', alpha=.5)
    axs[1].set_ylabel('Eval sequence - target sequence \n[bin number]')
    axs[1].set_xlabel('Time since sunrise (normalized)')
    fig.savefig(eval_outdir + 'single_step_bins_vs_time.png')

    if 'filter' in cfg['data']['action_space']:
        expert_filters_names = [IDX2FILTER[i] for i in expert_filters]
        agent_filters_names = [IDX2FILTER[i] for i in agent_filters]
        filter_residuals = agent_filters - expert_filters

        fig, axs = plt.subplots(2, figsize=(10,5), sharex=True)
        axs[0].plot(expert_times, expert_filters_names, marker='*', alpha=.3, label='true')
        axs[0].plot(expert_times, agent_filters_names, marker='o', alpha=.3, label='pred')
        axs[0].legend()
        axs[0].set_ylabel('filter')
        for filt in FILTER2IDX.keys():
            m = expert_filters_names == filt
            axs[1].plot(expert_times[m], filter_residuals[m], marker='o', alpha=.5, label='expert chose {filt}')
        axs[1].set_ylabel('Filter Index residuals (Agent - Expert)')
        axs[1].set_xlabel('Time since sunrise (normalized)')
        fig.savefig(eval_outdir + 'single_step_filters_vs_time.png')

    # Roll out policy
    logger.info("Starting evaluation...")
    agent.evaluate(env=env, cfg=cfg, num_episodes=args.num_episodes, field_choice_method=args.field_choice_method, eval_outdir=eval_outdir, field2radec=FIELD2RADEC)
    logger.info("Evaluation complete.")
    with open(eval_outdir + 'eval_metrics.pkl', 'rb') as handle:
        eval_metrics = pickle.load(handle)
    logger.info("Generating static plots...")

    ep_num = 0
    eval_metrics = eval_metrics[f'ep-{ep_num}']

    # 1. Create a DataFrame of ONLY the valid "current" states.
    # This perfectly aligns (1-to-1) with test_dataset.bin_actions and test_dataset.dones
    valid_transitions_df = test_dataset._df.iloc[test_dataset.current_state_idxs].reset_index(drop=True)

    # 2. Iterate cleanly over the unique nights
    for night_idx, night_name in enumerate(test_dataset.unique_nights):
        
        # Get date in string form for plots
        date = night_name.date()
        date_str = f"{date.year}-{date.month}-{date.day}"
        logger.info(f'Drawing plots for night {date_str}')
        night_dir = eval_outdir + date_str + '/'
        if not os.path.exists(night_dir):
            os.makedirs(night_dir)

        # 3. Get the DataLoader indices belonging strictly to this night
        night_mask = (valid_transitions_df['night'] == night_name).values
        night_dl_idxs = np.where(night_mask)[0]
        
        # 4. Extract Agent Metrics
        metrics = eval_metrics[f'night-{night_idx}']
        
        if len(night_dl_idxs) < 2 or len(metrics['timestamp']) < 2:
            logger.info(f"Night {night_idx} had no viable observations")
            continue
            
        night_compact_idxs = test_dataset.curr_compact_idxs[night_dl_idxs]
        
        night_df = valid_transitions_df.iloc[night_dl_idxs]
        
        if test_dataset._prenorm_bin_states is not None:
            expert_bin_states = test_dataset._prenorm_bin_states[night_compact_idxs]
    
        # Get plot time axis as hours since sunset
        sunset = calc_twilight(metrics['timestamp'][0], event_type='set')
        
        agent_zenith_mask = metrics['field_id'] != ZENITH_FIELD_ID
        expert_zenith_mask = night_df['field_id'].values != ZENITH_FIELD_ID
        
        agent_timestamps = np.array(metrics['timestamp'])
        agent_timestamps = (agent_timestamps - sunset) / 3600
        expert_timestamps = (night_df['timestamp'].values - sunset) / 3600
    
        # Plot bins vs timestamp        
        fig_b, axb = plt.subplots()
        axb.plot(agent_timestamps[agent_zenith_mask],
                      metrics['bin'][agent_zenith_mask],
                      marker='o', label='pred', alpha=.5)
        axb.plot(expert_timestamps[expert_zenith_mask],
                      night_df['bin'].values.astype(int)[expert_zenith_mask],
                      marker='o', label='true', alpha=.5)
        axb.legend()
        axb.set_xlabel('Hours since sunset \n (-10 deg)')
        axb.set_ylabel('bin')
        fig_b.suptitle(date_str)
        fig_b.tight_layout()
        fig_b.savefig(night_dir + f'bin_vs_step.png')
        plt.close()

        # # Plot state features vs timestamp for first episode
        # fig, axs = plt.subplots(len(test_dataset.global_feature_names), figsize=(10, len(test_dataset.global_feature_names)*5))
        # for i, feature_row in enumerate(metrics['glob_observations'].T[:len(test_dataset.global_feature_names)]):
        #     feat_name = test_dataset.global_feature_names[i]
        #     agent_data = feature_row.copy()
        #     # REVERSE NORMALIZATIONS
        #     if feat_name in global_cfg['features']['SIN_NORM_FEATURE_NAMES']:
        #         agent_data = np.arcsin(agent_data)
        #     elif feat_name in global_cfg['features']['LOG_NORM_FEATURE_NAMES']:
        #         agent_data = np.exp(agent_data) - 1e-9
        #     elif feat_name in global_cfg['features']['FRACTIONAL_FEATURE_NAMES']:
        #         agent_data = agent_data * (2*np.pi)
        #     elif feat_name in global_cfg['features']['LOCAL_MEAN_Z_SCORE_FEATURE_NAMES']:
        #         agent_data = agent_data * (2*np.pi)
        #     else:
        #         agent_data = agent_data
        #     if feat_name in global_cfg['features']['Z_SCORE_FEATURE_NAMES']:
        #         agent_data = agent_data * zscore_stats['global_features']['std'][] + zscore_stats[feat_name]['mean']

        #     axs[i].plot(agent_timestamps[agent_zenith_mask], agent_data[agent_zenith_mask], label='policy roll out', marker='o')
        #     axs[i].plot(expert_timestamps[expert_zenith_mask], night_df[feat_name].values[expert_zenith_mask], label='original schedule', marker='o')
        #     axs[i].set_title(feat_name)
        #     axs[i].set_xlabel('Hours since sunset \n (-10 deg)')
        #     axs[i].legend()
        # fig.tight_layout()
        # fig.savefig(night_dir + f'state_features_vs_time.png')
        # plt.close()

        # # Plot most frequently visited bin features vs timestamp
        # if cfg['model']['action_architecture'] is not None:
        #     _bins_vis_tonight = metrics['bin'].astype(int)
        #     _bincounts = np.bincount(_bins_vis_tonight[agent_zenith_mask], minlength=test_dataset.num_actions)
        #     _most_common_bin = np.argmax(_bincounts)
        #     normed_feature_names = test_dataset.bin_feature_names
        #     fig, axs = plt.subplots(len(normed_feature_names), figsize=(10, len(normed_feature_names)* 5))
        #     for i, feat_row in enumerate(metrics['bin_observations'].T[:, _most_common_bin, :]):
        #         feat_name = normed_feature_names[i]
        #         # unnormalize observations to compare to expert values
        #         if feat_name == 'airmass':
        #             agent_data = 1 / feat_row
        #         elif feat_name in global_cfg['features']['SIN_NORM_FEATURE_NAMES']:
        #             agent_data = feat_row * (np.pi/2)
        #         elif feat_name in global_cfg['features']['ANG_DISTANCE_NORM_FEATURE_NAMES']:
        #             agent_data = feat_row * (2 * np.pi)
        #         else:
        #             agent_data = feat_row
        #         axs[i].plot(agent_timestamps[agent_zenith_mask], agent_data[agent_zenith_mask], label='policy roll out', marker='o')
        #         axs[i].plot(expert_timestamps[expert_zenith_mask], expert_bin_states[expert_zenith_mask, _most_common_bin, i], label='original schedule', marker='o')
        #         axs[i].set_title(feat_name)
        #         axs[i].set_xlabel('Hours since sunset \n (-10 deg)')
        #         axs[i].legend()
        #     fig.suptitle(f"Bin {_most_common_bin}: (az, el) = ({test_dataset.hpGrid.lon[_most_common_bin]:.2f}, {test_dataset.hpGrid.lat[_most_common_bin]:.2f}")
        #     fig.tight_layout()
        #     fig.savefig(night_dir + f'bin_features_vs_time.png')


        # Plot static bin and field radec scatter plots
        bin2coord = {int(i): (lon, lat) for i, (lon, lat) in enumerate(zip(test_dataset.hpGrid.lon/units.deg, test_dataset.hpGrid.lat/units.deg))}

        agent_bin_radecs = np.array([bin2coord[bin_num] for bin_num in metrics['bin'].astype(int) if bin_num != ZENITH_BIN_NUM])
        orig_bin_radecs = np.array([bin2coord[bin_num] for bin_num in night_df['bin'].values if bin_num != ZENITH_BIN_NUM])
        
        agent_field_radecs = np.array([FIELD2RADEC[field_id] for field_id in metrics['field_id'].astype(int) if field_id != ZENITH_FIELD_ID])
        orig_field_radecs = np.array([FIELD2RADEC[field_id] for field_id in night_df['field_id'].values.astype(int) if field_id != ZENITH_FIELD_ID])
        
        if len(orig_field_radecs) != 1:
            # Plot bins
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            axs[0].scatter(orig_bin_radecs[:, 0], orig_bin_radecs[:, 1], label='expert', cmap='Reds', c=np.arange(len(orig_bin_radecs)))
            axs[1].scatter(agent_bin_radecs[:, 0], agent_bin_radecs[:, 1], label='agent', cmap='Blues', c=np.arange(len(agent_bin_radecs)))
            for ax in axs:
                ax.set_xlabel('x (ra or az)')
                ax.legend()
            axs[0].set_ylabel('y (dec or el)')
            fig.suptitle(f'Bins {night_name}')
            fig.savefig(night_dir + f'bins_ra_vs_dec.png')
            plt.close()
            
            # Plot fields
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            axs[0].scatter(orig_field_radecs[:, 0], orig_field_radecs[:, 1], label='expert', cmap='Reds', c=np.arange(len(orig_field_radecs)), s=10)
            axs[1].scatter(agent_field_radecs[:, 0], agent_field_radecs[:, 1], label='agent', cmap='Blues', c=np.arange(len(agent_field_radecs)), s=10)
            for ax in axs:
                ax.set_xlabel('ra')
                ax.legend() 
            axs[0].set_ylabel('dec')
            fig.suptitle(f'Fields {night_name}')
            fig.savefig(night_dir + f'fields_ra_vs_dec.png')
            plt.close()

        logger.info(f'Creating schedule gif for {night_idx}th night')
        
        bin2pos_filepath = global_cfg['paths']['TRAIN_DIR'] + f"nside{nside}_bin2{cfg['data']['action_space']}.json"
        bin2pos_filepath = bin2pos_filepath.replace("_filter", "")
        save_schedule(night_metrics=metrics, pd_group=night_df, save_dir=night_dir, nside=nside, make_gifs=args.make_gifs, 
                      is_azel=test_dataset.hpGrid.is_azel, bin2pos_filepath=bin2pos_filepath, field2radec_filepath=field2radec_filepath)
        
if __name__ == "__main__":
    main()