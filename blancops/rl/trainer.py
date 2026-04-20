from random import random
import gymnasium as gym
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
import time
from typing import Tuple
import os
import pickle
import random
from pathlib import Path

from blancops.math.interpolate import interpolate_on_sphere
from blancops.ephemerides import ephemerides
from blancops.data.constants import *
import logging

from blancops.utils.schedule_io import save_survey_schedule


# Get the logger associated with this module's name (e.g., 'my_module')
logger = logging.getLogger(__name__)
from tqdm.contrib.logging import logging_redirect_tqdm

class Trainer:
    """
    A simple, generic agent/wrapper for fitting and evaluating RL algorithms. 

    This class abstracts training loops, evaluation, saving/loading, and interaction with environment. It expects an underlying `algorithm` object for training.
    """
    def __init__(
            self,
            algorithm,
            train_outdir,
            cfg=None,
            # env: gym.Env = None,
            ):
        """
        Args
        ----
            algorithm (Algorithm): The Q-learning algorithm
            env (gymnasium.Env): The environment in which the agent will act.
            outdir (str): directory to save results
            normalize_obs (bool): Whether or not to normalize observations
        """
        if cfg is not None:
            self._setup_from_config(cfg)
        else:
            self.algorithm = algorithm
            self.device = algorithm.device
            if not os.path.exists(train_outdir):
                os.makedirs(train_outdir)
            self.train_outdir = train_outdir
        
    def fit(self, num_epochs, batch_size, trainloader, valloader, patience=10, train_log_freq=10, hpGrid=None):
        if len(valloader) == 0:
            raise ValueError("Validation dataloader is empty! Check dataset split logic.")

        val_metrics = defaultdict(list)
        train_metrics = defaultdict(list)
        
        # Set to train mode
        self.algorithm.policy.train()
        save_filepath = self.train_outdir + 'best_weights.pt'
        train_metrics_filepath = self.train_outdir + 'train_metrics.pkl'
        val_metrics_filepath = self.train_outdir + 'val_metrics.pkl'
        self.algorithm.policy.train()

        dataset_size = len(trainloader.dataset)
        steps_per_epoch = np.max([dataset_size // batch_size, 1])
        total_steps = int(num_epochs * steps_per_epoch) # ie, total number of times dataset is sampled
        loader_iter = iter(trainloader)  # create iterator

        use_best_val_loss = self.algorithm.name == 'BC'
        use_best_ang_sep = self.algorithm.name in ["DDQN", "DQN", "CQL"]
        assert use_best_val_loss or use_best_ang_sep, "Algorithm name is not valid. Check config file."
        best_val_loss = 1e5
        best_ang_sep = 1e5
        best_epoch = 0
        patience_cur = patience
        use_patience = patience != 0
        i_epoch = 0

        # total_lr_scheduler_steps = int(args.lr_scheduler_max_epochs * iterations_per_epoch // args.lr_scheduler_step_freq)
        logger.info(f"Total number of training steps: {total_steps}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.debug(f"Total number of lr scheduler steps: {self.algorithm.lr_scheduler_num_epochs if self.algorithm.lr_scheduler is not None else None}")
        logger.info(f"Number of transitions in dataset: {len(trainloader.dataset)}")

        with logging_redirect_tqdm():
            pbar = tqdm(total=total_steps, dynamic_ncols=True, desc="Starting training")
            for i_step in range(total_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(trainloader)
                    batch = next(loader_iter)

                if i_step % steps_per_epoch == 0:
                    i_epoch += 1
                    
                pbar.update(1)
                pbar.set_description(f"Epoch {i_epoch}/{int(num_epochs)} (step {i_step}/{total_steps})")

                # Train step
                log_metrics = i_step % steps_per_epoch == 0
                train_metrics_dict = self.algorithm.train_step(batch, epoch_num=i_epoch, hpGrid=hpGrid, compute_metrics=log_metrics) 
                if log_metrics:
                    for k, v in train_metrics_dict.items():
                        train_metrics[k].append(v)
                        
                    train_metrics['lr'].append(self.algorithm.optimizer.param_groups[0]["lr"])
                    train_metrics['epoch'].append(i_epoch)
                                   
                # Validation step
                with torch.no_grad():
                    if log_metrics:
                        val_metric_sums = defaultdict(float)
                        num_val_batches = len(valloader)
                        
                        for eval_batch in valloader:
                            batch_metrics = self.algorithm.val_step(eval_batch, hpGrid)
                            for k, v in batch_metrics.items():
                                val_metric_sums[k] += v
                            
                        # Average and save the metrics
                        val_log_str_parts = []
                        for k, total in val_metric_sums.items():
                            avg_val = total / num_val_batches
                            val_metrics[k].append(avg_val)
                            val_log_str_parts.append(f"{k} = {avg_val:.3f}")
                        val_metrics['epoch'].append(i_epoch)

                        # Log comparison
                        val_log_str = " | ".join(val_log_str_parts)
                        train_log_str = " | ".join(f"{k} = {v:.3f}" for k, v in train_metrics_dict.items())
                    
                        logger.info(
                            f"\nValidation check at train step {i_step} \n"
                            f" (val set)      {val_log_str} \n"
                            f" (train batch)  {train_log_str}"
                        )       

                        # Early stopping and model saving
                        val_loss_cur = val_metrics.get('val_loss', [1e5])[-1]
                        ang_sep_cur = val_metrics.get('ang_sep', [1e5])[-1]
                        improved = False
                        if val_loss_cur < best_val_loss and use_best_val_loss:
                            improved = True
                            best_val_loss = val_loss_cur
                            metric_str = f"val loss is {val_loss_cur:.3f}"
                        elif ang_sep_cur < best_ang_sep and use_best_ang_sep:
                            improved = True
                            best_ang_sep = ang_sep_cur
                            metric_str = f"angular separation is {ang_sep_cur:.3f}"
                        
                        if improved:
                            best_epoch = i_epoch
                            patience_cur = patience
                            logger.info(f'Improved model at step {i_step} (epoch {i_epoch}): {metric_str}. Saving weights')
                            self.save(save_filepath)
                            with open(train_metrics_filepath, 'wb') as handle:
                                pickle.dump(train_metrics, handle)
                            with open(val_metrics_filepath, 'wb') as handle:
                                pickle.dump(val_metrics, handle)
                        elif use_patience:
                            patience_cur -= 1
                            logger.debug(f"Patience left: {patience_cur}")
                            if patience_cur == 0:
                                logger.info("No patience left. Ending training.")
                                break
        if use_best_val_loss:
            logger.info(f"Best val loss was {best_val_loss:.3f} at epoch {best_epoch}")
        elif use_best_ang_sep:
            logger.info(f"Best angular separation was {best_ang_sep:.3f} at epoch {best_epoch}")
        with open(train_metrics_filepath, 'wb') as handle:
            pickle.dump(train_metrics, handle)
        with open(val_metrics_filepath, 'wb') as handle:
            pickle.dump(val_metrics, handle)
    
    def evaluate(self, env, cfg, num_episodes, lookups, field_choice_method='interp', eval_outdir=None, save_SISPI=False, SISPI_fn="survey_schedule"):
        """Evaluates the agent in an environment for multiple episodes.
        """
        eval_outdir = eval_outdir if eval_outdir is not None else self.train_outdir + 'evaluation/'
        if not os.path.exists(eval_outdir):
            os.makedirs(eval_outdir)
            
        # evaluation metrics
        self.algorithm.policy.eval()
        episode_rewards = []
        eval_metrics = {}

        field2nvisits = lookups.field2maxvisits
        field2radec = lookups.field2radec

        hpGrid = ephemerides.HealpixGrid(nside=cfg.data.nside, is_azel=('azel' in cfg.data.action_space))
        action_space = cfg.data.action_space

        FIELDS_CHOSEN = []

        with logging_redirect_tqdm():
            for episode in tqdm(range(num_episodes)):
                state, info = env.reset()
                episode_reward = 0
                terminated = False
                truncated = False
                num_nights = env.unwrapped.max_nights

                episode_data = {}
                reward = 0
                night_idx = 0
                current_night_key = f'night-{night_idx}'
                episode_data[current_night_key] = {
                    'glob_observations': [state['global_state']],
                    'bin_observations': [state['bin_state']],
                    'rewards': [reward],
                    'timestamp': [info.get('timestamp')],
                    'field_id': [ZENITH_FIELD_ID],
                    'bin': [ZENITH_BIN_NUM],
                    'filter_idx': [ZENITH_FILTER_IDX]
                }

                i = 0
                last_bin_idx = ZENITH_BIN_NUM
                field_id = ZENITH_FIELD_ID
                filter_idx = ZENITH_FILTER_IDX
                pbar = tqdm(total=250*num_nights, dynamic_ncols=True, desc=f"Rolling out policy for night {night_idx} step {i}")
                while not (terminated or truncated):
                    with torch.no_grad():
                        action_mask = info.get('action_mask', None)

                        # Catch the edge case where no fields are above the horizon - tell agent to wait
                        if not action_mask.any():
                            logger.warning(f"No valid fields available at step {i} (mask is all zeros).")
                            # bin_idx, field_id, filter_idx = WAIT_SIGNAL, WAIT_SIGNAL, WAIT_SIGNAL
                            bin_idx = WAIT_SIGNAL # do not update filter and field id since they should stay the same in wait state
                        else:
                            action = self.choose_action(x_glob=state['global_state'], x_bin=state['bin_state'], action_mask=action_mask, epsilon=None)
                            if 'filter' in action_space:
                                bin_idx = int(action // self.algorithm.policy.num_filters)
                                filter_idx = int(action % self.algorithm.policy.num_filters)
                            else:
                                bin_idx = action
                                filter_idx = NO_FILTER_SIGNAL

                            valid_fields_per_bin = info.get('valid_fields_per_bin', {})
                            fields_in_bin = np.array(valid_fields_per_bin.get(int(bin_idx), []))
                            if len(fields_in_bin) == 0:
                                raise ValueError(f"No valid fields in bin {action}.")
                            field_id = self.choose_field(obs=(state['global_state'], state['bin_state']), info=info, field2nvisits=field2nvisits, 
                                                        field2radec=field2radec, hpGrid=hpGrid, field_choice_method=field_choice_method, fields_in_bin=fields_in_bin,
                                                        filter_idx=filter_idx)#, num_filters=self.algorithm.num_filters)
                            FIELDS_CHOSEN.append(field_id)
                        is_first_wait = (bin_idx == WAIT_SIGNAL) and (last_bin_idx != WAIT_SIGNAL)
                        is_real_obs = bin_idx >= 0
                        if is_first_wait or is_real_obs:
                            current_night_dict = episode_data[current_night_key]
                            current_night_dict['glob_observations'].append(state['global_state'])
                            current_night_dict['bin_observations'].append(state['bin_state'])
                            current_night_dict['rewards'].append(reward)
                            current_night_dict['timestamp'].append(info.get('timestamp'))
                            current_night_dict['field_id'].append(field_id)
                            current_night_dict['bin'].append(bin_idx)
                            current_night_dict['filter_idx'].append(filter_idx)
                        
                        last_bin_idx = bin_idx
                        # Step environment
                        actions = {'bin': np.int32(bin_idx), 'field_id': np.int32(field_id), 'filter_idx': np.int32(filter_idx)}
                        state, reward, terminated, truncated, info = env.step(actions)
                        if terminated or truncated:
                            break

                        # Track total reward
                        episode_reward += reward

                        # Log zenith state if is new night
                        if info.get('night_idx') != night_idx:
                            night_idx = info.get('night_idx')
                            current_night_key = f'night-{night_idx}'
                            episode_data[current_night_key] = {
                                'glob_observations': [state['global_state']],
                                'bin_observations': [state['bin_state']],
                                'rewards': [0], # Re-initialize starting reward to 0
                                'timestamp': [info.get('timestamp')],
                                'field_id': [ZENITH_FIELD_ID],     # Reset to Zenith
                                'bin': [ZENITH_BIN_NUM],           # Reset to Zenith
                                'filter_idx': [ZENITH_FILTER_IDX]  # Reset to Zenith
                            }

                        # pbar update
                        i += 1
                        pbar.update(1)
                        pbar.set_description(f"Rolling out policy for night {night_idx} step {i}")
            pbar.close()
            # Convert all lists in the nested dictionary to numpy arrays
            for night_key, metrics in episode_data.items():
                for metric_name, values in metrics.items():
                    episode_data[night_key][metric_name] = np.array(values)

            # Store it in the master evaluation dictionary
            eval_metrics[f'ep-{episode}'] = episode_data
            episode_rewards.append(episode_reward)
            logger.info(f'terminated at step {i}')

        eval_metrics.update({
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards,
        })

        with open(Path(eval_outdir) / 'eval_metrics.pkl', 'wb') as handle:
            pickle.dump(eval_metrics, handle)
            logger.info(f'eval_metrics.pkl saved in {eval_outdir}')
        
        save_survey_schedule(
            eval_metrics=eval_metrics, 
            save_dir=eval_outdir, 
            field_lookup=lookups,
            save_SISPI=save_SISPI,
            SISPI_fn=SISPI_fn
            )   

        return eval_metrics

    def choose_action(self, x_glob, x_bin, action_mask, epsilon):
        """Selects an action using the underlying algorithm.

        Args:
            x_glob (array-like):
                Pointing and global state features (normalized if applicable).
            x_bin (array-like):
                Per-bin features (normalized if applicable).
            action_mask (array-like | None):
                Boolean mask indicating which actions are legal.
            epsilon (float | None):
                Epsilon for epsilon-greedy exploration. If None, selects greedily.

        Returns:
            int: Selected action index.
        """
        return self.algorithm.select_action(x_glob=x_glob, x_bin=x_bin, action_mask=action_mask, epsilon=epsilon)
    
    def save(self, filepath):
        """Saves algorithm parameters to a file.

        Args:
            filepath (str): Destination path for serialized model weights.
        """
        self.algorithm.save(filepath)
    
    def load(self, filepath):
        """Loads algorithm parameters from a file.

        Args:
            filepath (str): Path to previously saved model weights.
        """
        self.algorithm.load(filepath)

    def choose_field(self, obs, info, field2nvisits, field2radec, hpGrid, field_choice_method, fields_in_bin, filter_idx): 
        """
        Choose field in bin based on interpolated Q-values
        """
        assert len(fields_in_bin) != 0, "The agent is receiving an empty list for `fields_in_bin`."
        
        glob_state, bin_state = obs
        s_visited = info.get('s_visited', None)
        s_filter_visits = info.get('s_filter_visits', None)
        max_s_filter_visits = info.get('max_s_filter_visits', None)

        # 1. Filter out fields that have reached their visit limit
        if (s_filter_visits is not None) and (max_s_filter_visits is not None) and (filter_idx >= 0):
            field_ids_in_bin = [fid for fid in fields_in_bin if s_filter_visits[fid, filter_idx] < max_s_filter_visits[fid, filter_idx]]
        else:
            field_ids_in_bin = [fid for fid in fields_in_bin if s_visited[fid] < field2nvisits[fid]]
        
        assert len(field_ids_in_bin) != 0, "No valid fields are in bin...check environment's output mask."
        logger.debug(f'Chosen bin contains {len(field_ids_in_bin)} incomplete fields out of {len(fields_in_bin)} fields total')

        if field_choice_method == 'interp':
            with torch.no_grad():
                # Ensure tensors have the batch dimension expected by ScoreMLP
                glob_tensor = torch.as_tensor(glob_state, device=self.device, dtype=torch.float32).unsqueeze(0)
                bin_tensor = torch.as_tensor(bin_state, device=self.device, dtype=torch.float32).unsqueeze(0)
                
                # Get raw joint scores from MLP: shape (1, n_bins * n_filters)
                raw_scores = self.algorithm.policy.core_net(glob_tensor, bin_tensor)
                
                # [FIX 1]: Reshape scores to isolate the active filter's map
                n_bins = bin_tensor.shape[1]
                n_filters = raw_scores.shape[-1] // n_bins
                
                # Reshape to (n_bins, n_filters) and slice the specific filter
                q_map = raw_scores.view(n_bins, n_filters)[:, filter_idx].cpu().numpy()

            lon_data = hpGrid.lon 
            lat_data = hpGrid.lat

            # CHECK
            # target_coords = np.array([field2radec[fid] for fid in field_ids_in_bin])
            target_coords = np.array([field2radec[fid] for fid in field_ids_in_bin])
            
            if hpGrid.is_azel:
                # Project RA/Dec to local Az/El frame using the current timestamp
                timestamp = info.get('timestamp')
                target_lons, target_lats = ephemerides.equatorial_to_topographic(
                    ra=target_coords[:, 0], 
                    dec=target_coords[:, 1], 
                    time=timestamp
                )
            else:
                target_lons = target_coords[:, 0]
                target_lats = target_coords[:, 1]

            q_interpolated = interpolate_on_sphere(
                az=target_lons,
                el=target_lats,  # Target coordinates
                az_data=lon_data,
                el_data=lat_data,        # Bin centers (grid)
                values=q_map                      # Filter-specific Q-values
            )
            
            best_idx = np.argmax(q_interpolated)

            return field_ids_in_bin[best_idx]

        elif field_choice_method == 'random':
            return random.choice(field_ids_in_bin)