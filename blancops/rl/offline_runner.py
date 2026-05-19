from blancops.io.schedule_io import save_survey_schedule

import torch
import numpy as np
from tqdm import tqdm
import os
import gc
import pickle
from pathlib import Path

from blancops.ephemerides import ephemerides
from blancops.configs.constants import *
import logging

logger = logging.getLogger(__name__)
from tqdm.contrib.logging import logging_redirect_tqdm

class OfflineRunner:
    def __init__(self, agent, policy, cfg, lookups, num_episodes=1, 
                 outdir=None, save_SISPI=False, SISPI_fn="survey_schedule", schedule_chunk_size=None,
                 low_memory_mode=True,
                 ):
        self.agent = agent
        self.cfg = cfg
        self.policy = policy
        self.num_episodes = num_episodes
        self.lookups = lookups
        self.field_choice_method = self.agent.field_choice_method
        self.outdir = outdir
        self.save_SISPI = save_SISPI
        self.SISPI_fn = SISPI_fn
        # When True: per-night observation arrays are streamed to disk
        # (outdir/nights/) as they complete, and eval_metrics.pkl stores
        # a manifest of paths instead of the data itself. Bounds peak
        # rollout RAM to a single night's worth of (nbins, nfeats) data.
        self.low_memory_mode = low_memory_mode
        if (schedule_chunk_size is None) or (schedule_chunk_size <= 0):
            self.schedule_chunk_size = 1e5
        else:
            self.schedule_chunk_size = schedule_chunk_size
        self.schedules = {}
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if self.low_memory_mode:
            self._nights_dir = Path(self.outdir) / 'nights'
            self._nights_dir.mkdir(parents=True, exist_ok=True)

    def _flush_night_to_disk(self, night_dict, ep_num, night_key):
        """Convert a night's lists to float32 numpy arrays and save compressed.

        Returns the file path. Caller is expected to drop its reference
        to ``night_dict`` and let GC reclaim it.
        """
        arr_dict = {}
        for key, values in night_dict.items():
            arr = np.asarray(values)
            # Downcast f64 -> f32; observation features don't need f64 precision
            # and this halves the on-disk and reload memory footprint.
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32, copy=False)
            arr_dict[key] = arr
        path = self._nights_dir / f'ep-{ep_num}_{night_key}.npz'
        # savez_compressed adds CPU cost but ~3-5x smaller files; if rollout
        # I/O becomes a bottleneck switch to np.savez (uncompressed).
        np.savez_compressed(path, **arr_dict)
        return path
    
    def _update_current_night_dict(self, current_night_dict, obs, info, field_id, bin_idx, filter_idx, reward):
        current_night_dict['glob_observations'].append(obs['global_state'])
        current_night_dict['bin_observations'].append(obs['bin_state'])
        current_night_dict['rewards'].append(reward)
        current_night_dict['timestamp'].append(info.get('timestamp'))
        current_night_dict['field_id'].append(field_id)
        current_night_dict['bin'].append(bin_idx)
        current_night_dict['filter_idx'].append(filter_idx)

    def _log_zenith_state(self, obs, info):
        new_night_dict = {
            'glob_observations': [obs['global_state']],
            'bin_observations': [obs['bin_state']],
            'rewards': [0], # Re-initialize starting reward to 0
            'timestamp': [info.get('timestamp')],
            'field_id': [ZENITH_FIELD_ID],     # Reset to Zenith
            'bin': [ZENITH_BIN_NUM],           # Reset to Zenith
            'filter_idx': [ZENITH_FILTER_IDX]  # Reset to Zenith
        }
        return new_night_dict
    
    def run(self, env):
        self.policy.eval()
        episode_rewards = []

        hpGrid = ephemerides.HealpixGrid(nside=self.cfg.data.nside, is_azel=('azel' in self.cfg.data.action_space))

        with logging_redirect_tqdm():
            for ep_num in tqdm(range(self.num_episodes)):
                obs, info = env.reset()
                running_reward = 0
                terminated = False
                truncated = False
                num_nights = env.unwrapped.max_nights

                # In legacy mode, episode_data holds ALL nights' lists in memory
                # for the entire episode. In low_memory_mode, episode_data only
                # ever holds the CURRENT night being filled; completed nights
                # are flushed to disk and replaced by paths in episode_manifest.
                episode_data = {}
                episode_manifest = {}
                diagnostics = {}
                reward = 0
                night_idx = 0
                current_night_key = f'night-{night_idx}'
                episode_data[current_night_key] = self._log_zenith_state(obs, info)

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
                            bin_idx, filter_idx, field_id = self.agent.choose_bin_filter_field(obs, info, hpGrid, epsilon=None)
                            
                        # Step through environment
                        obs, reward, terminated, truncated, info = env.step({
                            'bin': np.int32(bin_idx), 
                            'field_id': np.int32(field_id), 
                            'filter_idx': np.int32(filter_idx)
                        })

                        # Log next obs if have not been waiting or if this is a non-zenith observation
                        is_first_wait = (bin_idx == WAIT_SIGNAL) and (last_bin_idx != WAIT_SIGNAL)
                        is_real_obs = bin_idx >= 0
                        if is_first_wait or is_real_obs:
                            self._update_current_night_dict(
                                current_night_dict=episode_data[current_night_key], 
                                obs=obs, info=info, field_id=field_id, bin_idx=bin_idx, filter_idx=filter_idx, reward=reward
                            )
                            
                        running_reward += reward
                        if terminated or truncated or i >= self.schedule_chunk_size:
                            break
                        
                        # Record last bin for logging check
                        last_bin_idx = bin_idx

                        # Log zenith state as previous state if is new night
                        if info.get('night_idx') != night_idx:
                            # In low_memory_mode, flush the just-completed night to
                            # disk and drop it from RAM before opening the next one.
                            # This is the key win: peak rollout RAM is bounded to
                            # one night instead of growing with nights observed.
                            if self.low_memory_mode:
                                path = self._flush_night_to_disk(
                                    episode_data[current_night_key], ep_num, current_night_key
                                )
                                episode_manifest[current_night_key] = str(path)
                                del episode_data[current_night_key]
                                gc.collect()

                            night_idx = info.get('night_idx')
                            current_night_key = f'night-{night_idx}'
                            episode_data[current_night_key] = self._log_zenith_state(obs, info)

                        # pbar update
                        i += 1
                        pbar.update(1)
                        pbar.set_description(f"Rolling out policy for night {night_idx} step {i}")
                logger.info(f'terminated at step {i}')

                # Flush the final night too in low_memory_mode
                if self.low_memory_mode and current_night_key in episode_data:
                    path = self._flush_night_to_disk(
                        episode_data[current_night_key], ep_num, current_night_key
                    )
                    episode_manifest[current_night_key] = str(path)
                    del episode_data[current_night_key]
                    gc.collect()

                diagnostics = self._construct_diagnostics(
                    diagnostics, episode_data, episode_rewards, running_reward, ep_num,
                    episode_manifest=episode_manifest,
                )
                
                pbar.close()
            diagnostics.update({
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episode_rewards': episode_rewards,
        })

        self._write_diagnostics_to_file(diagnostics)

        # save_survey_schedule expects the legacy in-memory format
        # ({ep-N: {night-K: {arrays...}}}). In low_memory_mode we only
        # have a manifest of paths, so skip the call unless explicitly
        # asked for via save_SISPI. Caller can re-run a dedicated
        # SISPI-writing step that loads nights one at a time.
        if self.low_memory_mode and not self.save_SISPI:
            logger.info(
                "low_memory_mode=True and save_SISPI=False; skipping "
                "save_survey_schedule. Per-night arrays are at %s.",
                self._nights_dir,
            )
        else:
            try:
                save_survey_schedule(
                    eval_metrics=diagnostics,
                    save_dir=self.outdir,
                    field_lookup=self.lookups,
                    save_SISPI=self.save_SISPI,
                    SISPI_fn=self.SISPI_fn,
                )
            except Exception as e:
                logger.warning(
                    "save_survey_schedule failed (likely incompatible with "
                    "manifest-format diagnostics): %s. Per-night arrays "
                    "still on disk at %s.",
                    e, getattr(self, "_nights_dir", self.outdir),
                )
        return diagnostics
    
    def _construct_diagnostics(self, diagnostics, episode_data, episode_rewards,
                               ep_running_reward, ep_num, episode_manifest=None):
        # Legacy path: episode_data holds the per-night dicts in memory.
        # Convert lists to np.arrays in place (briefly doubles each night's
        # footprint during the conversion).
        for night_key, metrics in episode_data.items():
            for metric_name, values in metrics.items():
                episode_data[night_key][metric_name] = np.array(values)

        if self.low_memory_mode and episode_manifest is not None:
            # Manifest format: episode_data is empty (already flushed).
            # Store paths instead. Downstream _process_eval_metrics
            # detects this 'manifest' key and reads per-night files.
            diagnostics[f'ep-{ep_num}'] = {
                'manifest': dict(episode_manifest),
                'total_reward': float(ep_running_reward),
            }
        else:
            diagnostics[f'ep-{ep_num}'] = episode_data

        episode_rewards.append(ep_running_reward)

        return diagnostics
    
    def _write_diagnostics_to_file(self, eval_metrics):
        with open(Path(self.outdir) / 'eval_metrics.pkl', 'wb') as handle:
            pickle.dump(eval_metrics, handle)
            logger.info(f'eval_metrics.pkl saved in {self.outdir}')