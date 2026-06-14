#%%

import argparse
import os
from pathlib import Path
from blancops.io.logger_utils import configure_logger
from blancops.rl.evaluations.evaluator import build_evaluators
from blancops.configs.rl_schema import load_and_validate
import logging

from blancops.utils.sys_utils import get_system_device


def main():

    # ------------------------------
    # ArgParse
    # ------------------------------

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_path', type=str, default=None, help="Path to config file. If passed, all other arguments are ignored")
    parser.add_argument('-l', '--logging_level', type=str, default='debug', help='Logging level. Options: info, debug')
    parser.add_argument('-f', '--force_overwrite', action='store_true', help='Whether to force overwrite previous rollout files.')
    parser.add_argument('--save_movies', action='store_true', help='Whether to save movie files.')
    parser.add_argument('--save_mollweides', action='store_true', help='Whether to save movie files.')

    args = parser.parse_args()

    # ------------------------------
    # Load config and device
    # ------------------------------

    cfg_path = args.cfg_path

    cfg = load_and_validate(cfg_path)
    device = get_system_device()

    # ------------------------------
    # Initialize logger
    # ------------------------------
    logger = configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=True,
        outdir=cfg.outdir,
        filename='validation.log',
        use_tqdm=True
    )

    # ------------------------------
    # Build and run evaluators
    # ------------------------------
    logger.info("Building evaluators...")
    s_eval, m_eval = build_evaluators(cfg, device=device, save_movie=args.save_movies, save_mollweide=args.save_mollweides)

    logger.info("Running evaluators...")
    s_eval.run()
    m_eval.run(overwrite=args.force_overwrite)

    # ------------------------------
    # Default plots
    # ------------------------------
    os.makedirs(Path(cfg.outdir) / 'ss', exist_ok=True)
    os.makedirs(Path(cfg.outdir) / 'ms', exist_ok=True)
    fig, ax = s_eval.plot_2dhist_res('ra', 'dec', normalization='probability')
    fig.savefig(Path(cfg.outdir) / 'ss' / 'ra_vs_dec_res.png')
    fig, ax =m_eval.plot_2dhist_res('ra', 'dec', normalization='probability')
    fig.savefig(Path(cfg.outdir) / 'ms' / 'ra_vs_dec_res.png')



