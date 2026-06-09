import numpy as np
import gymnasium as gym

from blancops.configs.constants import get_workspace_dir
from blancops.configs.rl_schema import ActionConstraints, load_and_validate
from blancops.plotting.plotting import plot_bins_movie
from blancops.rl.agent_factory import AgentFactory
from blancops.rl.checkpointer import get_checkpoint
from blancops.rl.offline_runner import OfflineRunner
from blancops.data.lookup_tables import LookupTables
from blancops.utils.sys_utils import seed_everything
from blancops.io.logger_utils import configure_logger
from blancops.utils.sys_utils import get_system_device
from blancops.configs.constants import WORKSPACE
from blancops.environment.offline_env import OfflineBlancoEnv

import argparse
from datetime import datetime
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model choice
    parser.add_argument('-m', '--model_path_or_alias', type=Path, default="bc_v1", help='Model alias or relative path to trained model directory')
    parser.add_argument('-c', '--field_choice_method', type=str, default='interp', help="Options: random, interp")

    # Field and Schedule info
    parser.add_argument('--field_lookup_dir', type=Path, required=True, help='Relative path to field lookup dir')
    parser.add_argument('-d', '--observing_nights', type=str, nargs='*', default=['2026-06-23-half2', '2026-06-24-half2'],
                        help="List of observing nights. Format [YY-MM-DD-NIGHT, ...] (e.g. 2026-06-23-full)"
                        )
    parser.add_argument('--obs_history_filename', type=str, default=None, help='If provided, will load historical observations to initialize the first state.')

    # Output info
    parser.add_argument('-o', '--outdir', type=Path, default=None, required=True, help='Relative path to output directory')
    parser.add_argument('--schedule_prefix', type=str, default='schedule', help='Base filename prefix for the generated schedule output')
    parser.add_argument('--save_sispi', action='store_true', help='Whether to save SISPI-format json files.')
    parser.add_argument('--save_movie', action='store_true', help='Whether to save gif files.')
    parser.add_argument('--save_mollweide', action='store_true', help='Whether to save png files.')

    # Logging
    parser.add_argument('-l', '--logging_level', type=str, default='info', choices=['info', 'debug', 'warning', 'error'], help='Logging level.')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing schedule if name already exists.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for schedule generation')

    # Scheduling parameters
    parser.add_argument('--sun_el_limit', type=float, default=-12, help="How low below horizon sun needs to be for observing (in deg). Default is -12.")
    parser.add_argument('--airmass_limit', type=float, default=1.2, help="The agent will only observe if there exist *any* fields below the airmass_lim")

    # Evaluation hyperparameters
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of evaluation episodes to run')
    parser.add_argument('--max_nights', type=int, default=0, help='Maximum number of nights')

    return parser.parse_args()


def main():

    # Parse args
    args = get_args()
    
    # ------------------------------
    # LOAD TARGET FIELDS
    # ------------------------------

    lookup_dir = Path(args.field_lookup_dir)
    lookups = LookupTables.load_from_dir(data_dir=lookup_dir)

    # ---------------------------------
    # SETUP LOGGER AND OUTDIR
    # ---------------------------------
    device = get_system_device()
    seed_everything(args.seed)

    if args.outdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = lookup_dir.parents[0].mkdir(parents=True, exist_ok=True)
        outdir = outdir / f"{args.schedule_prefix}_run_{timestamp}"
    else:
        outdir = Path(args.outdir)

    logger = configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=True,
        outdir=outdir,
        filename='offline_schedule.log',
        use_tqdm=True
    )
    
    logger.info("Arguments:")
    for key, value in vars(args).items():
        logger.info(
            "\t" + f"{key}: {value}"
            )

    if args.outdir is None:
        logger.info(f"No output directory specified. Using {outdir} as output directory.")
    else:
        logger.info(f"Using {outdir} as output directory.")
        

    # ---------------------------------
    # LOAD AGENT, MODEL, AND OFFLINE RUNNER
    # ---------------------------------
    logger.info("Loading agent...")
    factory = AgentFactory()

    agent, model_cfg, _ = factory.build_agent(
        model_path_or_alias=args.model_path_or_alias,
        lookups=lookups,
        field_choice_method=args.field_choice_method,
        device=device,
    )
    runner = OfflineRunner(
        agent=agent, policy=agent.policy, cfg=model_cfg,
        lookups=lookups, num_episodes=args.num_episodes, outdir=outdir,
        save_SISPI=args.save_sispi, save_movie=args.save_movie,
        save_mollweide=args.save_mollweide
    )

    # ---------------------------------
    # CREATE ENVIRONMENT
    # ---------------------------------
    logger.info("Setting up environment...")
    env_name = 'OfflineBlanco-v0'
    gym.register(
        id=f"gymnasium_env/{env_name}",
        entry_point=OfflineBlancoEnv,
    )

    checkpoint = get_checkpoint(Path(model_cfg.outdir), device=device)
    zscore_stats = checkpoint['norm_stats'].get('z_score', {})
    rel_norm_stats = checkpoint['norm_stats'].get('rel_norm', {})

    initial_counts = np.zeros_like(lookups.target_fidfilt_counts)
    initial_last_visit_ot = np.full(shape=lookups.target_fidfilt_counts.shape, fill_value=np.nan)
    initial_ot_at_sunset = 0

    env = gym.make(
        id=f"gymnasium_env/{env_name}",
        cfg=model_cfg,
        constraints_cfg=ActionConstraints(sun_el_limit=args.sun_el_limit,
                                          airmass_limit=args.airmass_limit),
        lookups=lookups,
        z_score_stats=zscore_stats,
        rel_norm_stats=rel_norm_stats,
        observing_night_strs=args.observing_nights,
        initial_counts=initial_counts,
        initial_last_visit_ot=initial_last_visit_ot,
        initial_ot_at_sunset=initial_ot_at_sunset,
        initial_fwhm=args.initial_fwhm,
    )

    # ---------------------------------
    # RUN POLICY
    # ---------------------------------
    logger.info("Running policy rollout...")
    runner.run(env=env)

    logger.info(f"Done. Output written to: {outdir}")


if __name__ == "__main__":
    main()
