import gc
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import gymnasium as gym

import json
import pandas as pd
import logging

from blancops.configs.constants import _NUM_FILTERS
from blancops.configs.enums import LookupKeys
from blancops.configs.rl_schema import ActionConstraints, load_and_validate
from blancops.rl.agent_factory import AgentFactory
from blancops.rl.checkpointer import get_checkpoint
from blancops.rl.offline_runner import OfflineRunner
from blancops.data.lookup_tables import LookupTables
from blancops.plotting.plotting import plot_schedule_from_file
from blancops.utils.sys_utils import seed_everything
from blancops.utils.sys_utils import get_system_device
from blancops.io.logger_utils import setup_logger_old
from blancops.data.preprocessing import load_and_process_historic_data
from blancops.environment.historic_env import HistoricBlancoEnv
from blancops.data.dataset import TransitionDataset
from blancops.data.feature_cache import RawFeatureCache, ValDatasetCache
from blancops.configs.constants import *
from blancops.math import units
from blancops.configs.constants import DES_FITS_PATH, DES_DATA_DIR
from blancops.data.features.normalizations import build_normalizer, build_normalizer_kwargs, StateNormalizer

from blancops.data.features.glob_features import calc_twilight
import logging
logger = logging.getLogger(__name__)

import argparse
from datetime import datetime
import re

from pathlib import Path

def setup_eval_outdir(args):
    if getattr(args, 'evaluation_name', None) is None:
        date_postfix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        evaluation_name = f"val_test_{date_postfix}_0"
    else:
        evaluation_name = args.evaluation_name
    eval_outdir = os.path.join(args.trained_model_dir, evaluation_name)
    
    if not os.path.exists(eval_outdir):
        os.makedirs(eval_outdir)
    else:
        while os.path.exists(eval_outdir):
            match = re.search(r"^(.*?)(\d+)$", evaluation_name)
            if match:
                base_name = match.group(1)
                num_group = int(match.group(2))
                evaluation_name = f"{base_name}{num_group + 1}"
            else:
                evaluation_name = f"{evaluation_name}_1"
            eval_outdir = os.path.join(args.trained_model_dir, evaluation_name)
        os.makedirs(eval_outdir)
    return Path(eval_outdir)

def save_schedule(night_metrics, pd_group, save_dir, make_gifs=True, nside=None, is_azel=False, whole=False, fid2radec_filepath=None):
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

    output_filepath = save_dir / 'schedule.csv'
    df.to_csv(output_filepath, index=False)

    logger.info("Creating fieldbin movies")
    # Create binfield movies
    plot_schedule_from_file(
        outfile=save_dir /  'agent_fieldbin_schedule.gif',
        schedule_file=output_filepath,
        plot_type='fieldbin',
        nside=nside,
        fields_file=fid2radec_filepath,
        whole=False,
        compare=False,
        expert=False,
        is_azel=action_space=='azel',
        mollweide=False,
    )
    plot_schedule_from_file(
        outfile=save_dir /  'expert_fieldbin_schedule.gif',
        schedule_file=output_filepath,
        plot_type='fieldbin',
        nside=nside,
        fields_file=fid2radec_filepath,
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
                outfile=save_dir /  'expert_field_schedule.gif',
                schedule_file=output_filepath,
                plot_type='field',
                nside=nside,
                fields_file=fid2radec_filepath,
                whole=False,
                compare=False,
                expert=True,
                is_azel=action_space=='azel',
                mollweide=False,
            )
            plot_schedule_from_file(
                outfile=save_dir /  'agent_field_schedule.gif',
                schedule_file=output_filepath,
                plot_type='field',
                nside=nside,
                fields_file=fid2radec_filepath,
                whole=False,
                compare=False,
                expert=False,
                is_azel=action_space=='azel',
                mollweide=False,
            )

        plot_schedule_from_file(
            outfile=save_dir /  'agent_bin_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=fid2radec_filepath,
            whole=False,
            compare=False,
            expert=False,
            is_azel=action_space=='azel',
            mollweide=False,
        ) 

        # Create bin movies   
        logger.info("Creating bin movies")
        plot_schedule_from_file(
            outfile=save_dir /  'bin_comparison_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=fid2radec_filepath,
            whole=False,
            compare=True,
            expert=True,
            is_azel=action_space=='azel',
            mollweide=False,
        )
        plot_schedule_from_file(
            outfile=save_dir /  'expert_bin_schedule.gif',
            schedule_file=output_filepath,
            plot_type='bin',
            nside=nside,
            fields_file=fid2radec_filepath,
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
                outfile=save_dir /  'mollweide.png',
                schedule_file=output_filepath,
                plot_type='bin',
                nside=nside,
                fields_file=fid2radec_filepath,
                whole=True,
                compare=True,
                expert=True,
                is_azel='azel' in action_space,
                mollweide=True,
            )  
            plot_schedule_from_file(
                outfile=save_dir /  'ortho.png',
                schedule_file=output_filepath,
                plot_type='bin',
                nside=nside,
                fields_file=fid2radec_filepath,
                whole=True,
                compare=True,
                expert=True,
                is_azel='azel' in action_space,
                mollweide=False,
            )  

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
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
    parser.add_argument('--make_gifs', action='store_true', help="Whether to create the set of gifs.")
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of evaluation episodes to run')
    parser.add_argument('--data_dir', type=str, default=str(DES_DATA_DIR),
                        help='Data directory containing lookups/ and the feature cache.')

    # Parse args and get config
    args = parser.parse_args()
    args.trained_model_dir = args.trained_model_dir.resolve()
    cfg = load_and_validate(args.trained_model_dir / "configs" / "resolved_config.yaml")
    constraints_cfg = ActionConstraints()
    eval_outdir = setup_eval_outdir(args)
    
    # Setup eval_outdir and looger``
    logger = setup_logger_old(save_dir=Path(eval_outdir).resolve(), logging_filename='eval.log', logging_level=args.logging_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pytorch").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    logging.getLogger("fontconfig").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)
    logger.info("Saving results in %s", eval_outdir)
    
    # Set Seed and Device
    seed_everything(args.seed)
    device = get_system_device()
    
    # Load Weights and Norm Stats
    checkpoint = get_checkpoint(args.trained_model_dir, device=device)
    zscore_stats = checkpoint['norm_stats'].get('z_score', {})
    rel_norm_stats = checkpoint['norm_stats'].get('rel_norm', {})
    logger.info("Successfully extracted norm_stats from checkpoint.")

    # Load Lookups
    lookups = LookupTables.load_from_dir(DES_DATA_DIR, include_historic=True)

    # Load Validation Dataset (from cache if available, else reconstruct)
    val_cache_path = args.trained_model_dir / "checkpoints" / "val_dataset_cache.pt"
    data_dir = Path(args.data_dir)
    is_azel = 'azel' in cfg.data.action_space
    coord = 'azel' if is_azel else 'radec'
    cache_dir = data_dir / f"feature_cache_nside{cfg.data.nside}_{coord}"

    if ValDatasetCache.exists(val_cache_path):
        logger.info(f"Loading cached val dataset from {val_cache_path}")
        test_dataset = ValDatasetCache.load(val_cache_path)
    else:
        logger.info("Val dataset cache not found — reconstructing from feature cache.")
        if not RawFeatureCache.exists(cache_dir):
            raise FileNotFoundError(
                f"Neither val dataset cache ({val_cache_path}) nor feature cache "
                f"({cache_dir}) found. Run precompute-features and run-train first."
            )
        full_cache = RawFeatureCache.load(cache_dir, mmap_bin=True)
        val_nights = cfg.data.val_nights
        val_raw_cache = full_cache.filter_nights(val_nights)
        test_dataset = TransitionDataset(
            mode='test',
            cache=val_raw_cache,
            cfg=cfg,
            lookups=lookups,
            z_score_stats=zscore_stats,
            rel_norm_stats=rel_norm_stats,
        )
        ValDatasetCache.from_transition_dataset(test_dataset).save(val_cache_path)
        logger.info(f"Val dataset cache saved to {val_cache_path}")
        del full_cache, val_raw_cache
        gc.collect()
        logger.info("Released feature cache from memory.")
    
    # from scipy.interpolate import CubicSpline

    # fwhm_night_interps = []
    # for n, ng in test_dataset._df.groupby('night'):
    #     _ts = ng['timestamp'].values
    #     _fwhms = ng['fwhm'].values
    #     cs = CubicSpline(_ts, _fwhms)
    #     fwhm_night_interps.append(cs)

    logger.info("Setting up environment...")
    env_name = 'OfflineDECamTestingEnv-v0'
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=HistoricBlancoEnv,
    )
        
    global_pd_nightgroup = test_dataset._df.groupby('night')
    if not args.start_at_zenith:
        global_pd_nightgroup = global_pd_nightgroup.apply(lambda x: x.iloc[1:]).reset_index(drop=True).groupby('night')
    t_survey_arr = np.asarray([test_dataset._df['t_survey'].unique()[0]]).flatten()
    if cfg.data.bin_state_dim > 0:
        night_start_indices = (test_dataset._df.iloc[test_dataset.current_state_idxs].reset_index(drop=True)[test_dataset._df.iloc[test_dataset.current_state_idxs].reset_index(drop=True)['object'] == 'zenith']).index.values
        if not args.start_at_zenith:
            night_start_indices += 1
        night_start_bin_states = test_dataset._prenorm_bin_states[night_start_indices].detach().numpy()
    else:
        night_start_bin_states = None
    
    env = HistoricBlancoEnv(
        cfg=cfg,
        constraints_cfg=constraints_cfg,
        lookups=lookups,
        global_pd_nightgroup=global_pd_nightgroup, 
        night_start_bin_states=night_start_bin_states, 
        z_score_stats=zscore_stats, 
        rel_norm_stats=rel_norm_stats,
        t_survey_arr=t_survey_arr, 
        # survey_nights_total=survey_nights_total,
        # fwhm_night_interps=fwhm_night_interps
    )
    
    factory = AgentFactory(base_model_dir=args.trained_model_dir.parent)
    
    agent, cfg, norm_stats = factory.build_agent(
        model_path_or_alias=args.trained_model_dir,
        lookups=lookups,
        field_choice_method=args.field_choice_method,
        device=device
    )
    
    zscore_stats = norm_stats.get('z_score', {})
    rel_norm_stats = norm_stats.get('rel_norm', {})
    
    runner = OfflineRunner(
        agent=agent,
        policy=agent.policy,
        cfg=cfg,
        lookups=lookups,
        num_episodes=1,
        outdir=eval_outdir,
        save_SISPI=False,
        schedule_chunk_size=0,
    )

    cur_idxs = test_dataset.current_state_idxs
    with torch.no_grad():
        q_vals = agent.policy.core_net(test_dataset.states[cur_idxs].to(device), test_dataset.bin_states[cur_idxs].to(device) if test_dataset.bin_states is not None else None)
        agent_actions = torch.argmax(q_vals, dim=1).to('cpu').detach().numpy()
    
    exp_actions = test_dataset.actions.detach().numpy()    
    exp_mask = test_dataset.actions != ZENITH_BIN_NUM
    ag_mask = agent_actions != ZENITH_BIN_NUM

    if 'filter' in cfg.data.action_space:
        expert_filters = exp_actions[exp_mask] % _NUM_FILTERS
        agent_filters = agent_actions[ag_mask] % _NUM_FILTERS
        expert_bins = exp_actions[exp_mask] // _NUM_FILTERS
        agent_bins = agent_actions[ag_mask] // _NUM_FILTERS
        assert len(expert_filters) == len(agent_filters), f"Shape mismatch: expert filters {expert_filters.shape}, agent_filters {agent_filters.shape}"
    else:
        expert_bins = exp_actions[exp_mask]
        agent_bins = agent_actions[ag_mask]
    
    time_idx = np.where(np.array(test_dataset.global_feature_names) == 't_night')[0]
    expert_times = test_dataset.states[test_dataset.next_state_idxs, time_idx].detach().numpy()
    assert len(expert_bins) == len(agent_bins) == len(expert_times), f"Shape mismatch: expert bins {expert_bins.shape}, agent_bins {agent_bins.shape} expert_times {expert_times.shape}"
    
    fig, axs = plt.subplots(2, figsize=(10,5), sharex=True)
    axs[0].plot(expert_times, expert_bins, marker='*', alpha=.3, label='true')
    axs[0].plot(expert_times, agent_bins, marker='o', alpha=.3, label='pred')
    axs[0].legend()
    axs[0].set_ylabel('bin number')
    axs[1].plot(expert_times, agent_bins - expert_bins, marker='o', alpha=.5)
    axs[1].set_ylabel('Eval sequence - target sequence \n[bin number]')
    axs[1].set_xlabel('Time since sunrise (normalized)')
    fig.savefig(eval_outdir /'single_step_bins_vs_time.png')

    if 'filter' in cfg.data.action_space:
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
        fig.savefig(eval_outdir /'single_step_filters_vs_time.png')

    logger.info("Starting evaluation...")
    diagnostics = runner.run(env=env)
    logger.info("Evaluation complete.")
    with open(eval_outdir /'eval_metrics.pkl', 'rb') as handle:
        diagnostics = pickle.load(handle)
    logger.info("Generating static plots...")
    
    ep_num = 0
    diagnostics = diagnostics[f'ep-{ep_num}']

    valid_transitions_df = test_dataset._df.iloc[test_dataset.current_state_idxs].reset_index(drop=True)

    for night_idx, night_name in enumerate(test_dataset.unique_nights):
        date = night_name.date()
        date_str = f"{date.year}-{date.month}-{date.day}"
        logger.info(f'Drawing plots for night {date_str}')
        night_dir = eval_outdir / date_str 
        if not os.path.exists(night_dir):
            os.makedirs(night_dir)

        night_mask = (valid_transitions_df['night'] == night_name).values
        night_dl_idxs = np.where(night_mask)[0]
        
        metrics = diagnostics[f'night-{night_idx}']
        print(metrics['field_id'])
        
        if not any(night_dl_idxs > 0) and len(night_dl_idxs) > 2:
            logger.info(f"Night {night_idx} had no viable observations")
            continue
            
        night_compact_idxs = test_dataset.curr_compact_idxs[night_dl_idxs]
        night_df = valid_transitions_df.iloc[night_dl_idxs]
        
        if test_dataset._prenorm_bin_states is not None:
            expert_bin_states = test_dataset._prenorm_bin_states[night_compact_idxs]
    
        sunset = calc_twilight(metrics['timestamp'][0], event_type='set')
        
        agent_zenith_mask = metrics['field_id'] != ZENITH_FIELD_ID
        expert_zenith_mask = night_df['field_id'].values != ZENITH_FIELD_ID
        
        agent_timestamps = np.array(metrics['timestamp'])
        agent_timestamps = (agent_timestamps - sunset) / 3600
        expert_timestamps = (night_df['timestamp'].values - sunset) / 3600
    
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
        fig_b.savefig(night_dir / f'bin_vs_step.png')
        plt.close()

        fig, axs = plt.subplots(len(test_dataset.global_feature_names), figsize=(10, len(test_dataset.global_feature_names)*5))
        feature_mappings = cfg.data.norm.feature_norm_mappings
        
        global_z_stats = zscore_stats.get('global_features', {})
        global_rel_stats = rel_norm_stats.get('global_features', {})
        
        for i, feature_row in enumerate(metrics['glob_observations'].T[:len(test_dataset.global_feature_names)]):
            feat_name = test_dataset.global_feature_names[i]
            agent_data = feature_row.copy()
            
            norms_applied = feature_mappings.get(feat_name, [])
            
            if 'z_score' in norms_applied:
                mean = global_z_stats[feat_name]['mean']
                std = global_z_stats[feat_name]['std']
                agent_data = (agent_data * std) + mean
                
            if 'local_mean_z' in norms_applied:
                std = global_rel_stats[feat_name]['std']
                agent_data = agent_data * std
                
            if 'fractional' in norms_applied:
                agent_data = (agent_data / 2.0) + 0.5
            if 'log' in norms_applied:
                agent_data = np.exp(agent_data) - 1e-9
            if 'sin' in norms_applied:
                agent_data = np.arcsin(agent_data)

            axs[i].plot(agent_timestamps[agent_zenith_mask], agent_data[agent_zenith_mask], label='policy roll out', marker='o')
            axs[i].plot(expert_timestamps[expert_zenith_mask], night_df[feat_name].values[expert_zenith_mask], label='original schedule', marker='o')
            axs[i].set_title(feat_name)
            axs[i].set_xlabel('Hours since sunset \n (-10 deg)')
            axs[i].legend()
            
        fig.tight_layout()
        fig.savefig(night_dir / f'state_features_vs_time.png')
        plt.close()
        
        bin2coord = {int(i): (lon, lat) for i, (lon, lat) in enumerate(zip(test_dataset.hpGrid.lon/units.deg, test_dataset.hpGrid.lat/units.deg))}

        agent_bin_radecs = np.array([bin2coord[bin_num] for bin_num in metrics['bin'].astype(int) if bin_num != ZENITH_BIN_NUM and bin_num != WAIT_SIGNAL])
        orig_bin_radecs = np.array([bin2coord[bin_num] for bin_num in night_df['bin'].values if bin_num != ZENITH_BIN_NUM and bin_num != WAIT_SIGNAL])
        
        agent_field_radecs = np.array([lookups.fields[['ra', 'dec']].loc[field_id] for field_id in metrics['field_id'].astype(int) if field_id != ZENITH_FIELD_ID])
        orig_field_radecs = np.array([lookups.fields[['ra', 'dec']].loc[field_id] for field_id in night_df['field_id'].values.astype(int) if field_id != ZENITH_FIELD_ID])
        
        if len(orig_field_radecs) != 1:
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            axs[0].scatter(orig_bin_radecs[:, 0], orig_bin_radecs[:, 1], label='expert', cmap='Reds', c=np.arange(len(orig_bin_radecs)))
            axs[1].scatter(agent_bin_radecs[:, 0], agent_bin_radecs[:, 1], label='agent', cmap='Blues', c=np.arange(len(agent_bin_radecs)))
            for ax in axs:
                ax.set_xlabel('x (ra or az)')
                ax.legend()
            axs[0].set_ylabel('y (dec or el)')
            fig.suptitle(f'Bins {night_name}')
            fig.savefig(night_dir / f'bins_ra_vs_dec.png')
            plt.close()
            
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            axs[0].scatter(orig_field_radecs[:, 0], orig_field_radecs[:, 1], label='expert', cmap='Reds', c=np.arange(len(orig_field_radecs)), s=10)
            axs[1].scatter(agent_field_radecs[:, 0], agent_field_radecs[:, 1], label='agent', cmap='Blues', c=np.arange(len(agent_field_radecs)), s=10)
            for ax in axs:
                ax.set_xlabel('ra')
                ax.legend() 
            axs[0].set_ylabel('dec')
            fig.suptitle(f'Fields {night_name}')
            fig.savefig(night_dir / f'fields_ra_vs_dec.png')
            plt.close()

        logger.info(f'Creating schedule gif for {night_idx}th night')
        
        save_schedule(night_metrics=metrics, pd_group=night_df, save_dir=night_dir, nside=cfg.data.nside, make_gifs=args.make_gifs, 
                    is_azel=test_dataset.hpGrid.is_azel, fid2radec_filepath=DES_DATA_DIR / LookupKeys.FIELDS.value
        )
        
if __name__ == "__main__":
    main()