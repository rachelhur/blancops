import os
import numpy as np
import torch
import matplotlib

from blancops.data.lookup_tables import LookupTables
matplotlib.use('Agg')
import time

from blancops.rl.trainer import Trainer
# from blancops.rl.algorithms.builder import build_algorithm
from blancops.utils.sys_utils import get_device, seed_everything
from blancops.io.logger_utils import setup_logger
from blancops.data.preprocessing import preprocess_train_df 
from blancops.data.dataset import OfflineDataset
from blancops.plotting.training_viz import plot_bin_feature_distributions, plot_bin_membership, plot_global_feature_distributions, plot_train_metrics
from blancops.rl.registry import build_algorithm, build_network
from blancops.configs.rl_schema import ExperimentConfig, load_and_validate, resolve_and_save
from blancops.configs.constants import TRAIN_DATA_DIR, TRAIN_DATA_PATH, _BIN_FEATURES, WORKSPACE

import argparse
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
    
def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg', type=str, default=None, required=True, help="Path to config file. If passed, all other arguments are ignored")
    parser.add_argument('-l', '--logging_level', type=str, default='info', help='Logging level. Options: info, debug')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Whether to resume training from a checkpoint.')
    parser.add_argument('--overwrite', action='store_true', help='Whether to ignore existing history but keep files.')
    parser.add_argument('--hard_overwrite', action='store_true', help='Whether to completely overwrite existing results.')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top runs to keep. Default is 1.')

    args = parser.parse_args()
    return args

def setup_result_outdirs(cfg: ExperimentConfig):
    parent = Path(cfg.parent_dir)
    if not parent.is_absolute():
        parent = WORKSPACE / parent

    if cfg.outdir:
        outdir = Path(cfg.outdir)
        if not outdir.is_absolute():
            outdir = WORKSPACE / outdir
    else:
        outdir = parent / cfg.experiment_name

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'figures').mkdir(parents=True, exist_ok=True)
    (outdir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (outdir / 'metrics').mkdir(parents=True, exist_ok=True)
    (outdir / 'configs').mkdir(parents=True, exist_ok=True)
    (outdir / 'logs').mkdir(parents=True, exist_ok=True)

    cfg.outdir = str(outdir)
    
    return outdir

def main():

    # --- SETUP EXPERIMENT FROM ARGS AND CONFIG --- #
    args = get_args()
    cfg = load_and_validate(args.cfg)
    outdir = setup_result_outdirs(cfg)    
    # ---------------------- #

    # --- SET UP LOGGER --- #
    logger = setup_logger(save_dir=outdir / "logs", logging_filename='train.log')
    # ---------------------- #

    # --- SEED AND GET DEVICE --- #
    seed_everything(cfg.train.seed)
    device = get_device()
    # ---------------------- #

    # --- LOAD DATA AND CONSTRUCT DATASET --- #
    df = preprocess_train_df(TRAIN_DATA_PATH)
    train_lookups = LookupTables.load_from_dir(TRAIN_DATA_DIR, is_historic=True)
    train_dataset = OfflineDataset(
        mode='train',
        df=df,
        cfg=cfg,
        lookups=train_lookups
        )
    logger.info("Finished constructing train_dataset.")
    logger.info(f"Train dataset has {train_dataset.n_nights} nights and {train_dataset.num_transitions} transitions")
    # ---------------------- #

    #  --- BASIC PLOTTING --- #
    plot_bin_membership(train_dataset, outdir / "figures")
    plot_global_feature_distributions(train_dataset, outdir / "figures")
    plot_bin_feature_distributions(train_dataset, outdir / "figures")
    #  ---------------------- #

    # --- DATALOADERS --- #
    trainloader, valloader = train_dataset.get_dataloader(cfg.train.batch_size, num_workers=cfg.train.num_workers, pin_memory=True if device.type == 'cuda' else False, \
                                                              random_seed=cfg.train.seed)
    #  ---------------------- #
    
    # --- SAVE VAL DATA FOR VALIDATION --- #
    # cache_path = outdir / "val_cache.pt"
    # val_data_list = []
    # for batch in valloader:
    #     val_data_list.append(batch)
        # torch.save(val_data_list, cache_path)
        # logger.info(f"Validation cache saved to {cache_path}")
    # ---------------------- #
    
    # --- GET COSINE ANNEALING SCHEDULER Kwargs --- #
    def get_cosine_annealing_scheduler_kwargs(cfg, train_dataset):
        steps_per_epoch = np.max([int(len(train_dataset) // cfg.train.batch_size), 1])
        num_lr_scheduler_steps = np.int32(np.max([1, int(cfg.train.lr_sched_epoch_duration * steps_per_epoch)]))
        lr_scheduler_kwargs = {'T_max': int(num_lr_scheduler_steps), 'eta_min': float(cfg.train.lr_final)} if cfg.train.lr_scheduler == 'cosine_annealing' else {}
        return lr_scheduler_kwargs
    
    lr_scheduler_kwargs = get_cosine_annealing_scheduler_kwargs(cfg, trainloader.dataset)
    # ---------------------- #
    
    # --- BUILD TRAINER --- #
    cfg = resolve_and_save(cfg=cfg, dataset_dims=train_dataset.dataset_dims, dataset_feature_names=train_dataset.dataset_feature_names, 
                           lr_scheduler_kwargs=lr_scheduler_kwargs, val_nights=train_dataset.val_nights, outdir=outdir / "configs")
    algorithm = build_algorithm(cfg, device=device)

    latest_ckpt_path = outdir / "checkpoints" / "latest_checkpoint.pt"

    trainer = Trainer(
        algorithm=algorithm,
        train_outdir=outdir,
        top_k=args.top_k,
        overwrite=args.overwrite,
        hard_overwrite=args.hard_overwrite
    )
    
    if latest_ckpt_path.exists() and args.resume_from_checkpoint:
        start_epoch = trainer.resume_from_checkpoint(latest_ckpt_path)
    else:
        start_epoch = 0
        
    # ---------------------- #

    logger.info("Starting training...")

    # Train agent
    start_time = time.time()
    
    trainer.fit(
        start_epoch=start_epoch,
        num_epochs=cfg.train.max_epochs,
        trainloader=trainloader,
        valloader=valloader,
        batch_size=cfg.train.batch_size,
        patience=cfg.train.patience,
        hpGrid=train_dataset.hpGrid,
        norm_stats=train_dataset.get_norm_stats()
    )
    end_time = time.time()
    logger.info(f'Total train time = {end_time - start_time}s on {device}')
    logger.info("Training complete.")
    
    logger.info(f'Results saved in {outdir}')
    
    logger.info("Plotting metrics...")
    plot_train_metrics(outdir, dataset=train_dataset)

if __name__ == "__main__":
    main()
    
    
# from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR

# # 1. Setup Optimizer
# optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# # 2. Define the "Waiting Period" Scheduler (Factor=1.0 means don't change the LR)
# delay_epochs = 5
# waiting_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=delay_epochs)

# # 3. Define your actual active scheduler (e.g., decay the LR over the next 10 epochs)
# active_epochs = 10
# decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=active_epochs)

# # 4. Chain them together!
# # It will wait for 5 epochs, then decay for 10 epochs.
# main_scheduler = SequentialLR(
#     optimizer, 
#     schedulers=[waiting_scheduler, decay_scheduler], 
#     milestones=[delay_epochs] # The epoch where it switches from waiting to decaying
# )

# # 5. Pass it to your radically simplified trainer
# trainer = BehaviorCloning(
#     policy_net=policy, 
#     optimizer=optimizer,
#     lr_scheduler=main_scheduler, 
#     device='cuda'
# )