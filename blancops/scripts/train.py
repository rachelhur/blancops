import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import time

from blancops.rl.trainer import Trainer
# from blancops.rl.algorithms.builder import build_algorithm
from blancops.utils.sys_utils import setup_logger, get_device, seed_everything
from blancops.data.preprocessing import load_train_data_to_dataframe 
from blancops.data.offline_dataset import OfflineDataset
from blancops.utils.sys_utils import save_config
from blancops.plotting.training_viz import plot_bin_feature_distributions, plot_bin_membership, plot_global_feature_distributions, plot_train_metrics
from blancops.rl.registry import build_algorithm, build_network
from blancops.configs.schema import load_and_validate, resolve_and_save
from blancops.configs.constants import TRAIN_DATA_PATH, BIN_FEATURES

import argparse
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
    
def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg', type=str, default=None, help="Path to config file. If passed, all other arguments are ignored")
    parser.add_argument('-l', '--logging_level', type=str, default='info', help='Logging level. Options: info, debug')

    args = parser.parse_args()
    return args

def get_results_outdirs(cfg):
    # Define standard output directories based on the workspace
    results_outdir = Path(cfg.experiments_directory) / Path(cfg.experiment_name)
    results_outdir.mkdir(parents=True, exist_ok=True)
    fig_outdir = results_outdir / 'figures'
    return results_outdir, fig_outdir

def main():
    args = get_args()
    cfg = load_and_validate(args.cfg)
    exp_outdir, fig_outdir = get_results_outdirs(cfg)    
    exp_outdir.mkdir(parents=True, exist_ok=True)
    fig_outdir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logger(save_dir=exp_outdir, logging_filename='training.log')

    # Seed everything
    seed_everything(cfg.train.seed)
    device = get_device()

    df = load_train_data_to_dataframe(TRAIN_DATA_PATH)
    train_dataset = OfflineDataset(
        df=df,
        cfg=cfg,
        )
    logger.info("Finished constructing train_dataset.")
    logger.info(f"Train dataset has {train_dataset.n_nights} nights and {train_dataset.num_transitions} transitions")

    #  --- BASIC PLOTTING --- #
    plot_bin_membership(train_dataset, fig_outdir)
    plot_global_feature_distributions(train_dataset, fig_outdir)
    plot_bin_feature_distributions(train_dataset, fig_outdir)
    #  ---------------------- #

    # DATALOADERS
    trainloader, valloader = train_dataset.get_dataloader(cfg.train.batch_size, num_workers=cfg.train.num_workers, pin_memory=True if device.type == 'cuda' else False, \
                                                              random_seed=cfg.train.seed)
    
    # SAVE VAL DATA FOR VALIDATION
    cache_path = exp_outdir / "val_cache.pt"
    val_data_list = []
    for batch in valloader:
        val_data_list.append(batch)
        torch.save(val_data_list, cache_path)
        logger.info(f"Validation cache saved to {cache_path}")
    
    def get_cosine_annealing_scheduler_kwargs(cfg, train_dataset):
        steps_per_epoch = np.max([int(len(train_dataset) // cfg.train.batch_size), 1])
        num_lr_scheduler_steps = np.int32(np.max([1, int(cfg.train.lr_sched_epoch_duration * steps_per_epoch)]))
        lr_scheduler_kwargs = {'T_max': int(num_lr_scheduler_steps), 'eta_min': float(cfg.train.lr_final)} if cfg.train.lr_scheduler == 'cosine_annealing' else {}
        return lr_scheduler_kwargs
    
    lr_scheduler_kwargs = get_cosine_annealing_scheduler_kwargs(cfg, trainloader.dataset)
    cfg = resolve_and_save(cfg=cfg, dataset_dims=train_dataset.dataset_dims, dataset_feature_names=train_dataset.dataset_feature_names, 
                           lr_scheduler_kwargs=lr_scheduler_kwargs, outdir=exp_outdir)
    algorithm = build_algorithm(cfg, device=device)
    

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
    
    # cfg['train']['lr_scheduler_kwargs'] = {key: float(val) for key, val in lr_scheduler_kwargs.items()}

    agent = Trainer(
        algorithm=algorithm,
        train_outdir=str(exp_outdir) + '/',
    )

    logger.info("Starting training...")

    # Train agent
    start_time = time.time()
    agent.fit(
        num_epochs=cfg.train.max_epochs,
        trainloader=trainloader,
        valloader=valloader,
        batch_size=cfg.train.batch_size,
        patience=cfg.train.patience,
        hpGrid=train_dataset.hpGrid
    )
    end_time = time.time()
    logger.info(f'Total train time = {end_time - start_time}s on {device}')
    logger.info("Training complete.")
    
    logger.info(f'Results saved in {exp_outdir}')
    

    # if args.save_to_model_dir:
    #     if args.model_name is None:
    #         args.model_name = cfg['metadata']['exp_name']
    #     model_dir = workspace / 'models' / args.model_name
    #     model_dir.mkdir(parents=True, exist_ok=True)  # <--- Add this line
    #     logger.info(f"Saving weights and config to {model_dir}")
    #     save_config(config_dict=cfg, outdir=model_dir)
    #     agent.save(filepath=model_dir / 'best_weights.pt')

    logger.info("Plotting metrics...")
    plot_train_metrics(exp_outdir, dataset=train_dataset)

if __name__ == "__main__":
    main()