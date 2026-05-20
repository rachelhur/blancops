#%%

import argparse
from pathlib import Path
from blancops.io.logger_utils import configure_logger
from blancops.rl.evaluations.evaluator import build_evaluators
from blancops.configs.rl_schema import load_and_validate
import logging

from blancops.utils.sys_utils import get_system_device

logger = logging.getLogger(__name__)


def build_and_run_evaluators(cfg):
    device = get_system_device()
    s_eval, m_eval = build_evaluators(cfg, device=device)
    s_eval.run(); m_eval.run()
    logger.info("Done")

def main():
    
    # ------------------------------
    # ArgParse
    # ------------------------------
    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_path', type=str, default=None, help="Path to config file. If passed, all other arguments are ignored")
    parser.add_argument('-l', '--logging_level', type=str, default='debug', help='Logging level. Options: info, debug')
    parser.add_argument('-f', '--force_overwrite', action='store_true', help='Whether to force overwrite previous rollout files.')

    args = parser.parse_args()
    
    # ------------------------------
    # Load config and device
    # ------------------------------
    
    if args.cfg_path is None:
        cfg_path = Path('./experiments/bc/TEST_FULL_FEATURE_SET/run_20260517_203006/configs/resolved_config.yaml')
        # cfg_path = '/home/hurra/Projects/blancops/experiments/bc/TEST_FULL_FEATURE_SET/run_20260517_162420/configs/resolved_config.yaml'
    else:
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
    s_eval, m_eval = build_evaluators(cfg, device=device)
    
    logger.info("Running evaluators...")
    s_eval.run(); m_eval.run(overwrite=args.force_overwrite)
    
    # ------------------------------
    # Default plots
    # ------------------------------
    
    #TODO
    
    
    
    