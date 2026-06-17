#%%

import argparse
import os
from pathlib import Path

from matplotlib import pyplot as plt
from blancops.io.logger_utils import configure_logger
from blancops.rl.evaluations.evaluator import build_evaluators, plot_metric_distributions_with_ss_overlay
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

    # Resolve the model dir from where the config was loaded (machine-portable)
    if cfg.orig_cfg_path:
        cfg_dir = Path(cfg.orig_cfg_path).parent
        outdir = cfg_dir.parent if cfg_dir.name == "configs" else cfg_dir
    else:
        outdir = Path(cfg.outdir)

    # ------------------------------
    # Initialize logger
    # ------------------------------
    logger = configure_logger(
        level=args.logging_level,
        log_to_stdout=True,
        log_to_file=True,
        outdir=outdir,
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
    os.makedirs(outdir / 'ss', exist_ok=True)
    os.makedirs(outdir / 'ms', exist_ok=True)
    fig, ax = s_eval.plot_2dhist_res('ra', 'dec', normalization='probability')
    fig.savefig(outdir / 'ss' / 'ra_vs_dec_res.png')
    fig, ax =m_eval.plot_2dhist_res('ra', 'dec', normalization='probability')
    fig.savefig(outdir / 'ms' / 'ra_vs_dec_res.png')

    m_eval.plot_violin_per_filter('moon_el')
    plt.savefig(outdir / 'ms' / 'filter_strategy_violins_moon_el.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_metric_distributions_with_ss_overlay(ms_evaluator=m_eval, ss_evaluator=s_eval)
    plt.savefig(outdir / 'ms' / 'survey_quality_distributions.png', dpi=300, bbox_inches='tight')





