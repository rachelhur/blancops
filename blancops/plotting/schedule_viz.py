
from datetime import datetime, timedelta
import math
import os
from venv import logger

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from blancops.data_processing.constants import *
from blancops.data_processing.features import get_nautical_twilight
from blancops.plotting.plotting import plot_schedule_from_file

def save_schedule(night_metrics, save_dir, make_gifs=True, nside=None, is_azel=False, whole=False, field2radec_filepath=None):
    # Save timestamps, field_ids, and bin numbers
    bin_space = 'azel' if is_azel else 'radec'
    assert os.path.exists(save_dir)

    timestamps = np.array(night_metrics['timestamp']).astype(np.int32)
    bins = np.array(night_metrics['bin']).astype(np.int32)
    fids = np.array(night_metrics['field_id']).astype(np.int32)
    if len(timestamps) < 1:
        return

    real_obs_mask = (bins != ZENITH_BIN_NUM) & (bins != WAIT_SIGNAL)
    
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

from datetime import timezone
def plot_static_diagnostics(eval_metrics, observing_night_strs, schedule_outdir, grid_network, env, field_lookup, nside, lookup_dirpath, num_actions, bin_space, ep_num=0):
    ep_num = 0
    eval_metrics = eval_metrics[f'ep-{ep_num}']
    night_info = []
    for obs_n_str in observing_night_strs:
        str_split = obs_n_str.split('-', maxsplit=3)
        night_str = '-'.join(str_split[:3])
        night_portion = str_split[-1]
        night_dt = datetime.strptime(night_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        midnight_dt = night_dt + (timedelta(days=1) - pd.Timedelta(nanoseconds=1))
        night_info.append((midnight_dt, night_portion))

    for night_idx in range(len(eval_metrics.keys())):
        observing_night_str = observing_night_strs[night_idx]
        night_dt = night_info[night_idx][0]
        date_str = f"{night_dt.year}-{night_dt.month}-{night_dt.day}"
        logger.info(f'Drawing plots for night {date_str}')
        night_dir = schedule_outdir / date_str
        if not os.path.exists(night_dir):
            os.makedirs(night_dir)
        metrics = eval_metrics[f'night-{night_idx}']
        print(metrics['timestamp'].shape, metrics['glob_observations'].shape)
        if len(metrics['timestamp']) < 2:
            logger.info(f"Night {night_idx} had no viable observations")
            continue

        # Mask zenith observations in plotting
        real_obs_mask = np.array(metrics['field_id']) != ZENITH_FIELD_ID
        real_obs_mask &= np.array(metrics['field_id']) != WAIT_SIGNAL
        
        timestamps = metrics['timestamp']
        field_ids = metrics['field_id']
        bin_nums = metrics['bin']

        night_ts = night_dt.timestamp()
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
        fig_b.suptitle(observing_night_str)
        fig_b.tight_layout()
        fig_b.savefig(night_dir / 'bin_vs_step.png')
        plt.close()

        # Plot state features vs timestamp for first episode
        fig, axs = plt.subplots(len(env.unwrapped.global_feature_names), figsize=(10, len(env.unwrapped.global_feature_names)*5))
        for i, feature_row in enumerate(np.array(metrics['glob_observations']).T[:len(env.unwrapped.global_feature_names)]):
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
        if grid_network is not None:
            _bins_vis_tonight = np.array(bin_nums).astype(int)
            _bincounts = np.bincount(_bins_vis_tonight[real_obs_mask], minlength=num_actions)
            _most_common_bin = np.argmax(_bincounts)
            normed_feature_names = env.unwrapped.bin_feature_names
            fig, axs = plt.subplots(len(normed_feature_names), figsize=(10, len(normed_feature_names)* 5))
            for i, feat_row in enumerate(np.array(metrics['bin_observations']).T[:, _most_common_bin, :]):
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
            night_dt += timedelta(days=1)

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
        save_schedule(night_metrics=metrics, save_dir=night_dir, nside=nside, make_gifs=True, 
                    is_azel='azel' in bin_space, field2radec_filepath= lookup_dirpath / 'field2radec.json')
        
        night_dt += timedelta(days=1)