
import os
from venv import logger

import numpy as np
import pandas as pd

from blancops.data_processing.constants import WAIT_SIGNAL, ZENITH_BIN_NUM
from blancops.plotting.plotting import plot_schedule_from_file


def save_schedule(night_metrics, save_dir, make_gifs=True, nside=None, is_azel=False, whole=False, field2radec_filepath=None):
    # Save timestamps, field_ids, and bin numbers
    bin_space = 'azel' if is_azel else 'radec'
    assert os.path.exists(save_dir)

    timestamps = np.array(night_metrics['timestamp']).astype(np.int32)
    bins = np.array(night_metrics['bin']).astype(np.int32)
    fids = np.array(night_metrics['field_id']).astype(np.int32)
    if len(timestamps) > 1:
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
