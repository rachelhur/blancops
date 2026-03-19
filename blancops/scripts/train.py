import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch

import time
import pickle

from blancops.core_rl.agents import Agent
from blancops.algorithms.factory import setup_algorithm
from blancops.math import geometry
from blancops.math import units
from blancops.utils.sys_utils import setup_logger, get_device, seed_everything
from blancops.data_processing.data_processing import load_raw_data_to_dataframe 
from blancops.data_processing.offline_dataset import OfflineDataset
from blancops.utils.sys_utils import save_config, load_global_config, dict_to_nested, get_workspace_dir

import argparse
import logging
import json

from pathlib import Path

def plot_metrics(results_outdir, dataset):
    with open(results_outdir / 'train_metrics.pkl', 'rb') as f:
        train_metrics = pickle.load(f)
    with open(results_outdir / 'val_metrics.pkl', 'rb') as f:
        val_metrics = pickle.load(f)
    with open(results_outdir / 'val_train_metrics.pkl', 'rb') as f:
        val_train_metrics = pickle.load(f)

    # Plot Loss, Accuracy, and Angular separation
    nrows = 3 if 'ang_sep' in val_metrics else 2
    fig, axs = plt.subplots(nrows, sharex=True, figsize=(4, 7))

    axs[0].plot(train_metrics['epoch'], train_metrics['train_loss'], label='train loss', color='black', linestyle='dotted')
    axs[0].plot(val_metrics['epoch'], val_metrics['val_loss'], label='val loss')
    axs[0].hlines(y=0, xmin=0, xmax=np.max(val_metrics['epoch']), color='red', linestyle='dashed')
    axs[0].set_ylabel('Loss', fontsize=14)
    axs[0].legend(fontsize=12)

    axs[1].plot(val_train_metrics['epoch'], val_train_metrics['accuracy'], label='train accuracy', color='black', linestyle='dotted')
    axs[1].plot(val_metrics['epoch'], val_metrics['accuracy'], label='val accuracy')
    axs[1].hlines(y=1, xmin=0, xmax=np.max(train_metrics['epoch']), color='red', linestyle='dotted')
    axs[1].set_ylabel('Accuracy', fontsize=14)
    axs[1].legend(fontsize=12)

    if 'ang_sep' in val_metrics:
        lonlat = np.array((dataset.hpGrid.lon, dataset.hpGrid.lat))
        pos1 = lonlat[:, :-1]
        pos2 = lonlat[:, 1:]
        ang_seps = geometry.angular_separation(pos1=pos1, pos2=pos2)
        average_bin_sep = np.mean(ang_seps)

        axs[2].plot(val_train_metrics['epoch'], np.array(val_train_metrics['ang_sep'])/units.deg, label='train', color='black', linestyle='dotted')
        axs[2].plot(val_metrics['epoch'], np.array(val_metrics['ang_sep'])/units.deg, label='val')
        axs[2].set_ylabel('Angular separation \n (deg)', fontsize=14)
        axs[2].set_xlabel('Epoch')
        axs[2].hlines(y=average_bin_sep/units.deg, xmin=0, xmax=np.max(val_train_metrics['epoch']), label='average bin sep', color='red', linestyle='dashed')
        axs[2].legend(fontsize=12)

    for ax in axs:
        ax.grid(True, alpha=.5)

    fig.tight_layout()
    fig.savefig(results_outdir / 'figures' / 'loss_and_metrics_history.png')    

    if 'filter_accuracy' in val_metrics:
        fig, ax = plt.subplots(1)
        ax.plot(val_train_metrics['epoch'], val_train_metrics['filter_accuracy'])
        ax.plot(val_metrics['epoch'], val_metrics['filter_accuracy'])
        ax.set_ylabel('Filter Accuracy', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.hlines(y=1., xmin=0, xmax=np.max(val_train_metrics['epoch']), color='red', linestyle='dashed')
        fig.tight_layout()
        fig.savefig(results_outdir / 'figures' / 'filter_accuracy.png')

    i = 0
    fig, ax = plt.subplots()
    for key in val_metrics.keys():
        if key != 'accuracy' and key != 'epoch' and 'loss' not in key and key != 'ang_sep':
            ax.plot(val_metrics['epoch'], val_metrics[key], label='val ' + key, color=f"C{i}")
            ax.plot(val_metrics['epoch'], val_train_metrics[key], color=f"C{i}", linestyle='dotted')
            i += 1
    ax.grid(True, alpha=.5)
    ax.legend()
    ax.set_xlabel('Epoch', fontsize=14)
    fig.tight_layout()
    fig.savefig(results_outdir / 'figures' / 'val_metrics.png')

    if 'unique_bins' in val_metrics:
        # Count bins with < 10 examples
        bin_ids, _ = np.unique(dataset.bin_actions.detach().numpy(), return_counts=True)
        total_bin_diversity = len(bin_ids)/dataset.num_actions
        fig, ax = plt.subplots()
        ax.plot(val_train_metrics['epoch'], val_train_metrics['unique_bins'], label='train', color='grey', alpha=.5, linestyle='dotted')
        ax.plot(val_metrics['epoch'], val_metrics['unique_bins'], label='val')
        ax.set_ylabel('Unique bins \n (normalized by total number of bins)', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.hlines(y=total_bin_diversity, xmin=0, xmax=np.max(val_train_metrics['epoch']), label='dataset-wide unique bin visit', color='black', linestyle='dotted')
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(results_outdir / 'figures' / 'unique_bins_history.png')
    
    fig, ax = plt.subplots()
    ax.grid(True, alpha=.5)
    ax.plot(train_metrics['epoch'], train_metrics['lr'])
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('LR', fontsize=14)
    fig.tight_layout()
    fig.savefig(results_outdir / 'figures' / 'lr_steps.png')
    
def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg', type=str, default=None, help="Path to config file. If passed, all other arguments are ignored")
    parser.add_argument('-s', '--save_to_model_dir', action='store_true', default=None, help="Whether or not to save to official models directory")
    parser.add_argument('--model_name', type=str, default=None, help='Name of model; used as name of model directory')
    
    # Data input and output file and dir setups
    parser.add_argument('--fits_path', type=str, default='../data/decam-exposures-20251211.fits', help='Path to offline dataset file')
    parser.add_argument('--json_path', type=str, default='../data/decam-exposures-20251211.json', help='Path to offline dataset metadata json file')
    parser.add_argument('--metadata.parent_results_dir', type=str, default=None, help='Path to results directory')
    parser.add_argument('--metadata.exp_name', type=str, default='test_experiment', help='Name of the experiment -- used to create the subdir in parents_results_dir')
    parser.add_argument('--metadata.seed', type=int, default=10, help='Random seed for reproducibility')
    
    # Algorithm setup
    parser.add_argument('--model.algorithm', type=str, default='ddqn', help='Algorithm to use for training (DDQN or BC)')
    parser.add_argument('--model.loss_function', type=str, default='cross_entropy', help='Loss function. Options: mse, cross_entropy, huber, mse')
    parser.add_argument('--model.contextual_gating', action='store_true', help='Whether or not to use contextual gating on global features. Only implemented for multi_dim_score grid network')
    parser.add_argument('--model.tau', type=float, default=0.005, help='Target network update rate for DDQN')
    parser.add_argument('--model.gamma', type=float, default=0.99, help='Discount factor for DDQN')
    parser.add_argument('--model.activation', type=str, default='relu', help='The activation function to use in the neural network. Options: relu, mish, swish ')

    # Data selection and setup
    parser.add_argument('--data.bin_method', type=str, default='healpix', help='Binning method to use (healpix or uniform)')
    parser.add_argument('--data.nside', type=int, default=16, help='Healpix nside parameter (only used if binning_method is healpix)')
    parser.add_argument('--data.num_bins_1d', type=int, default=16, help='Number of bins in 1dim (only used if binning_method is uniform)')
    parser.add_argument('--data.bin_space', type=str, default='radec', help='Binning space to use (azel or radec)')
    parser.add_argument('--data.specific_years', type=int, nargs='*', default=None, help='Specific years to include in the dataset')
    parser.add_argument('--data.specific_months', type=int, nargs='*', default=None, help='Specific months to include in the dataset')
    parser.add_argument('--data.specific_days', type=int, nargs='*', default=None, help='Specific days to include in the dataset')
    parser.add_argument('--data.specific_filters', type=str, nargs='*', default=None, help='Specific filters to include in the dataset')
    # parser.add_argument('--include_default_features', action='store_true', help='Whether to include default features in the dataset')
    parser.add_argument('--data.do_cyclical_norm', action='store_true', help='Whether to apply cyclical normalization to the features')
    parser.add_argument('--data.do_max_norm', action='store_true', help='Whether to apply max normalization to the features')
    parser.add_argument('--data.do_inverse_norm', action='store_true', help='Whether to include inverse normalizations to features')
    parser.add_argument('--data.remove_large_time_diffs', action='store_true', help='New method of calculating transitions which removes any transitions with time difference greater than 10 min')
    parser.add_argument('--data.bin_features', type=str, nargs='*', default=[], help='Bin feautures to include')
    parser.add_argument('--data.pointing_features', type=str, nargs='*', default=[], help='Pointing feautures to include')

    # Training hyperparameters
    parser.add_argument('--train.max_epochs', type=float, default=10, help='Maximum number of passes through train dataset')
    parser.add_argument('--train.batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--train.num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--train.use_train_as_val', action='store_true', help='Instead of using validation samples during training, use the training samples')
    parser.add_argument('--train.lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train.lr_scheduler', type=str, default=None, help='cosine_annealing or None')
    parser.add_argument('--train.lr_scheduler_num_epochs', type=int, default=0, help='Number of epochs to reach min lr (must be less than num_epochs)')
    parser.add_argument('--train.lr_scheduler_epoch_start', type=int, default=100, help='Epoch at which to start lr scheduler')
    parser.add_argument('--train.eta_min', type=float, default=1e-5, help='Minimum learning rate for cosine annealing scheduler')
    parser.add_argument('--train.hidden_dim', type=int, default=1024, help='Hidden dimension size for the model')
    parser.add_argument('--train.patience', type=int, default=0, help='Early stopping patience (in epochs). If 0, patience will not be used.')
    
    # Verbosity
    parser.add_argument('-l', '--logging_level', type=str, default='info', help='Logging level. Options: info, debug')

    args = parser.parse_args()

    # If a config file is passed, overwrite the argparse defaults
    if args.cfg is not None:
        assert Path(args.cfg).exists(), f"Config file at {args.cfg} does not exist."
            
        with open(args.cfg, 'r') as f:
            print(args.cfg)
            file_conf = json.load(f)
            for section, values in file_conf.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        setattr(args, f"{section}.{k}", v)
    return args

def main():
    args = get_args()
    cfg = dict_to_nested(vars(args))
    gcfg = load_global_config()
    workspace = get_workspace_dir()
    
    # Define standard output directories based on the workspace
    if cfg['metadata']['parent_results_dir'] is None:
        results_outdir = workspace / "experiments" / f"nside{cfg['data']['nside']}" / cfg['metadata']['exp_name']
    else:
        results_outdir = workspace / cfg['metadata']['parent_results_dir'] / cfg['metadata']['exp_name']
    results_outdir.mkdir(parents=True, exist_ok=True)
    fig_outdir = results_outdir / 'figures'
    fig_outdir.mkdir(parents=True, exist_ok=True)
    cfg['metadata']['outdir'] = str(results_outdir)

    # Set up logging
    logger = setup_logger(save_dir=results_outdir, logging_filename='training.log')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pytorch").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    logging.getLogger("fontconfig").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)
    
    # Make sure action space and grid networks align
    if 'filter' in cfg['data']['bin_space']:
        assert cfg['model']['grid_network'] == "multi_dim_scorer", "Only multi_dim_scorer can handle filter in action space right now"
    if len(cfg['data']['bin_features']) > 0:
        assert np.isin(cfg['model']['grid_network'], ["single_bin_scorer", "multi_dim_scorer"]), "Must use a grid_network if using bin features. Options: single_bin_scorer, multi_dim_scorer"

    # Get training configs used more than once
    batch_size = cfg['train']['batch_size'] #cfg.get('experiment.training.batch_size')
    max_epochs = cfg['train']['max_epochs'] #cfg.get('experiment.training.max_epochs')
    lr_scheduler = cfg['train']['lr_scheduler'] #cfg.get('experiment.training.lr_scheduler')
    lr_scheduler_epoch_start = cfg['train']['lr_scheduler_epoch_start'] #cfg.get('experiment.training.lr_scheduler_epoch_start')
    lr_scheduler_num_epochs = cfg['train']['lr_scheduler_num_epochs'] #cfg.get('experiment.training.lr_scheduler_num_epochs')
    for bin_feat in cfg['data']['bin_features']:
        assert bin_feat in gcfg['features']['BIN_FEATURES'], f"{bin_feat} has not yet been implemented. Check global config file for valid inputs."
    # assert errors dne before running rest of code
    if lr_scheduler is not None:
        assert max_epochs - lr_scheduler_epoch_start - lr_scheduler_num_epochs >= 0, "The number of epochs must be greater than lr_scheduler_epoch_start + lr_scheduler_num_epochs"

    logger.info("Saving results in " + str(results_outdir))

    # Seed everything
    seed_everything(cfg['metadata']['seed'])

    device = get_device()

    logger.info("Loading raw data...")

    df = load_raw_data_to_dataframe(Path(gcfg['paths']['TRAIN_DIR']) / Path(gcfg['files']['DECFITS']))

    logger.info("Processing raw data into OfflineDataset()...")
    # Need to include paths.lookup_dir in cfg before sending to offline dataset -- brittle
    # train_dataset = OfflineDELVEDataset(
    #     df=df,
    #     cfg=cfg,
    #     gcfg=gcfg,
    # )
    train_dataset = OfflineDataset(
        df=df,
        cfg=cfg,
        gcfg=gcfg,
        )
    logger.info("Finished constructing train_dataset.")
    logger.info(f"Train dataset has {train_dataset.n_nights} nights and {train_dataset.num_transitions} transitions")

    # Plot bin membership for fields in ra vs dec
    colors = [f'C{i}' for i in range(7)]
    for i, (bin_id, g) in enumerate(train_dataset._df.groupby('bin')):
        plt.scatter(g.ra, g.dec, label=bin_id, color=colors[i%len(colors)], s=1)
    plt.title("Fields in train data, colored by bin membership")
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.savefig(fig_outdir / 'train_data_fields_dec_vs_ra.png')

    # logger.info("Plotting S x A (state x action) space cornerplot (this will take some time...)")
    # # Plot State x action space via cornerplot
    # corner_plot = sns.pairplot(train_dataset._df,
    #          vars=train_dataset.global_feature_names + ['bin'],
    #          kind='hist',
    #          corner=True
    #         )
    # corner_plot.figure.savefig(fig_outdir / 'state_times_action_space_corner_plot.png')
    # logger.info("Corner plot saved")

    fig, axs = plt.subplots(len(train_dataset.global_feature_names), figsize=(4, len(train_dataset.global_feature_names)*3))
    next_states = train_dataset.next_states.T
    for i, feat_name in enumerate(train_dataset.global_feature_names):
        axs[i].hist(next_states[i])
        axs[i].set_title(f"Train distribution ({feat_name})")
    fig.tight_layout()
    fig.savefig(fig_outdir / 'train_data_global_feature_distributions.png')
        
    if cfg['train']['use_train_as_val']:
        trainloader = train_dataset.get_dataloader(batch_size, num_workers=cfg['train']['num_workers'], pin_memory=True if device.type == 'cuda' else False, random_seed=cfg.get('experiment.metadata.seed'), return_train_and_val=False)
        valloader = trainloader
    else:
        trainloader, valloader = train_dataset.get_dataloader(batch_size, num_workers=cfg['train']['num_workers'], pin_memory=True if device.type == 'cuda' else False, random_seed=cfg['metadata']['seed'], return_train_and_val=True)

    # Initialize algorithm and agent
    logger.info("Initializing agent...")

    steps_per_epoch = np.max([int(len(trainloader.dataset) // batch_size), 1])
    num_lr_scheduler_steps = np.int32(np.max([1, int(lr_scheduler_num_epochs * steps_per_epoch)]))
    lr_scheduler_kwargs = {'T_max': num_lr_scheduler_steps, 'eta_min': cfg['train']['eta_min']} if lr_scheduler == 'cosine_annealing' else {}

    algorithm = setup_algorithm(algorithm_name=cfg['model']['algorithm'], n_global_features=train_dataset.states.shape[-1], n_bin_features=0 if train_dataset.bin_states is None else train_dataset.bin_states.shape[-1],
                                num_actions=train_dataset.num_actions, num_filters=train_dataset.num_filters, loss_fxn=cfg['model']['loss_function'],
                                hidden_dim=cfg['train']['hidden_dim'], lr=cfg['train']['lr'], lr_scheduler=lr_scheduler, 
                                device=device, lr_scheduler_kwargs=lr_scheduler_kwargs, lr_scheduler_epoch_start=lr_scheduler_epoch_start, 
                                lr_scheduler_num_epochs=lr_scheduler_num_epochs, gamma=cfg['model']['gamma'], 
                                tau=cfg['model']['tau'], activation=cfg['model']['activation'], grid_network=cfg['model']['grid_network'],
                                use_contextual_gating=cfg['model']['contextual_gating'])

    agent = Agent(
        algorithm=algorithm,
        train_outdir=str(results_outdir) + '/',
    )

    # Save (or update) config file after updating
    cfg['data']['state_dim'] = train_dataset.state_dim
    cfg['data']['bin_state_dim'] = 0 if train_dataset._grid_network is None else train_dataset.bin_state_dim
    cfg['data']['num_actions'] = train_dataset.num_actions
    cfg['train']['lr_scheduler_kwargs'] = {key: float(val) for key, val in lr_scheduler_kwargs.items()}
    
    def check_cfg_dtypes(d):
        """Recursively check if all values in nested dict are 64-bit."""
        for k, v in d.items():
            if isinstance(v, dict):
                if not check_cfg_dtypes(v):
                    return False
            # Check for 64-bit integer or float specifically
            elif isinstance(v, (np.float64, np.int64, np.float32, np.int32)):
                # Optional: handle standard python types if necessary
                # For strictness, you may want: isinstance(v, (np.float64, np.int64))
                # Or check if dtype is 'float64'/'int64' if using numpy arrays
                logger.debug(f"{k} has np-bit precision with value {v}")
            else:
                logger.debug(f"{k} has value {v} with dtype {type(v)}")

    check_cfg_dtypes(cfg)
    save_config(config_dict=cfg, outdir=results_outdir)

    logger.info("Starting training...")

    # Train agent
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    start_time = time.time()
    agent.fit(
        num_epochs=max_epochs,
        trainloader=trainloader,
        valloader=valloader,
        batch_size=batch_size,
        patience=cfg['train']['patience'],
        hpGrid=train_dataset.hpGrid
    )
    end_time = time.time()
    logger.info(f'Total train time = {end_time - start_time}s on {device}')
    logger.info("Training complete.")

    if args.save_to_model_dir:
        if args.model_name is None:
            args.model_name = cfg['metadata']['exp_name']
        model_dir = workspace / 'models' / args.model_name
        model_dir.mkdir(parents=True, exist_ok=True)  # <--- Add this line
        logger.info(f"Saving weights and config to {model_dir}")
        save_config(config_dict=cfg, outdir=model_dir)
        agent.save(filepath=model_dir / 'best_weights.pt')

    logger.info("Plotting training loss curve...")
    plot_metrics(results_outdir, dataset=train_dataset)

    # Plot predicted action for each state in train dataset
    dataset = trainloader.dataset.dataset
    val_states, val_actions, _, _, _, _, val_bin_states, _ = dataset[valloader.dataset.indices]
    train_states, train_actions, _, _, _, _, train_bin_states, _ = dataset[trainloader.dataset.indices]

    do_bin_states = dataset._grid_network is not None
    for prefix, (states, bin_states, actions) in zip(['val_', 'train_'], [ (val_states, val_bin_states, val_actions), (train_states, train_bin_states, train_actions) ]):
        eval_actions_list = []
        # Process in smaller chunks to save VRAM
        plot_batch_size = 128 
        for i in range(0, len(states), plot_batch_size):
            with torch.no_grad():
                # Only send a slice to the device
                s_chunk = states[i:i + plot_batch_size].to(device)
                if do_bin_states:
                    b_chunk = bin_states[i:i + plot_batch_size].to(device)
                else:
                    b_chunk = None
                
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    q_vals = agent.algorithm.policy_net(x_glob=s_chunk, x_bin=b_chunk, y_data=None)
                
                chunk_actions = torch.argmax(q_vals, dim=1).cpu()
                eval_actions_list.append(chunk_actions)
        
        # Combine back into a single numpy array for your plotting function
        eval_actions = torch.cat(eval_actions_list).numpy()

        # Sequence of actions from target (original schedule) and policy
        target_sequence = actions.detach().numpy()
        eval_sequence = eval_actions
        first_night_indices = np.where(states[:, -1] == 0)

        fig, axs = plt.subplots(2, figsize=(10,5), sharex=True)
        
        axs[0].plot(target_sequence, marker='*', alpha=.3, label='true')
        axs[0].plot(eval_sequence, marker='o', alpha=.3, label='pred')
        axs[0].legend()
        axs[0].set_ylabel('bin number')
        axs[0].vlines(first_night_indices, ymin=0, ymax=len(dataset.hpGrid.lon), color='black', linestyle='--')
        axs[1].plot(eval_sequence - target_sequence, marker='o', alpha=.5)
        axs[1].set_ylabel('Eval sequence - target sequence \n[bin number]')
        axs[1].set_xlabel('observation index')
        fig.savefig(fig_outdir / (prefix + 'eval_and_target_bin_sequences.png'))

        logger.info(f'Results saved in {results_outdir}')

if __name__ == "__main__":
    main()