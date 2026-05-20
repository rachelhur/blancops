"""Orchestration: Evaluator hierarchy + factory.

`Evaluator` is the abstract base. `SingleStepEvaluator` runs one-step-ahead
inference; `MultiStepEvaluator` drives the offline runner.

`build_evaluators` wires everything together from a config path or object.
"""
from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops import rearrange
import logging
logger = logging.getLogger(__name__)

from blancops.configs.constants import (
    FILTER2IDX,
    TRAIN_DATA_DIR,
    TRAIN_DATA_PATH,
    _NUM_FILTERS,
)
from blancops.configs.rl_schema import ActionConstraints, load_and_validate
from blancops.data.dataset import OfflineDataset
from blancops.data.features.normalizations import build_normalizer
from blancops.data.lookup_tables import LookupTables
from blancops.data.preprocessing import preprocess_historic_data
from blancops.environment.historic_env import HistoricBlancoEnv
from blancops.rl.agent_factory import AgentFactory
from blancops.rl.checkpointer import get_checkpoint
from blancops.rl.offline_runner import OfflineRunner

from .data_container import (
    DataContainer,
    MultiStepDataContainer,
    SingleStepDataContainer,
)
from .plotters import FILTER_COLORS, EvaluationPlotter, PlotStyle


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def build_evaluators(
    cfg_or_cfg_path,
    device,
    eval_outdir: str = 'holdout_eval',
    style: PlotStyle = None,
) -> Tuple['SingleStepEvaluator', 'MultiStepEvaluator']:
    """Build SS and MS evaluators for the validation set from a config."""
    cfg = (
        load_and_validate(cfg_or_cfg_path, None)
        if isinstance(cfg_or_cfg_path, str)
        else cfg_or_cfg_path
    )
    style = style or PlotStyle()

    outdir = Path(cfg.outdir)
    ss_outdir = outdir / eval_outdir / 'ss'
    ms_outdir = outdir / eval_outdir / 'ms'

    # Data
    lookups = LookupTables.load_from_dir(TRAIN_DATA_DIR, include_historic=True)
    df = preprocess_historic_data(TRAIN_DATA_PATH)
    df_val = df[df['night'].isin(cfg.data.val_nights)]

    # Checkpoint + normalizers
    checkpoint = get_checkpoint(outdir, device=device)
    zscore_stats = checkpoint['norm_stats'].get('z_score', {})
    rel_norm_stats = checkpoint['norm_stats'].get('rel_norm', {})
    global_normalizer = build_normalizer(state_feature_names=cfg.data.global_features, cfg=cfg)
    

    val_dataset = OfflineDataset(
        df=df_val, cfg=cfg, lookups=lookups,
        z_score_stats=zscore_stats, rel_norm_stats=rel_norm_stats, mode='test',
    )
    

    # Agent + runner
    factory = AgentFactory(base_model_dir=outdir)
    agent, cfg, _ = factory.build_agent(
        model_path_or_alias=outdir,
        lookups=lookups,
        field_choice_method='interp',
        device=device,
    )
    runner = OfflineRunner(
        agent=agent, policy=agent.policy, cfg=cfg,
        lookups=lookups, num_episodes=1, outdir=ms_outdir,
        save_SISPI=False, schedule_chunk_size=0,
    )

    # Environment for MS evaluator
    nightgroup = val_dataset._df.groupby('night')
    nightgroup = nightgroup.apply(lambda x: x.iloc[1:]).reset_index(drop=True).groupby('night')

    night_start_bin_states = None
    if cfg.data.bin_state_dim > 0:
        cur = val_dataset._df.iloc[val_dataset.current_state_idxs].reset_index(drop=True)
        night_start_indices = cur.index[cur['object'] == 'zenith'].values + 1
        night_start_bin_states = val_dataset._prenorm_bin_states[night_start_indices].detach().numpy()

    env = HistoricBlancoEnv(
        cfg=cfg, constraints_cfg=ActionConstraints(), lookups=lookups,
        global_pd_nightgroup=nightgroup, night_start_bin_states=night_start_bin_states,
        z_score_stats=zscore_stats, rel_norm_stats=rel_norm_stats,
    )

    # Containers + plotters + evaluators
    action_space = cfg.data.action_space
    ss_data = SingleStepDataContainer(val_dataset, action_space, lookups,
                                     global_normalizer=global_normalizer)
    
    ms_data = MultiStepDataContainer(val_dataset, action_space, lookups, z_score_stats=zscore_stats, rel_norm_stats=rel_norm_stats,
                                     global_normalizer=global_normalizer)

    ss_plotter = EvaluationPlotter(ss_outdir, style=style)
    ms_plotter = EvaluationPlotter(ms_outdir, style=style)

    ss_eval = SingleStepEvaluator(policy=agent.policy, data_container=ss_data,
                                  plotter=ss_plotter, device=device)
    ms_eval = MultiStepEvaluator(runner=runner, env=env, policy=agent.policy,
                                 data_container=ms_data, plotter=ms_plotter, device=device)
    return ss_eval, ms_eval


# ----------------------------------------------------------------------
# Evaluator base
# ----------------------------------------------------------------------

class Evaluator(ABC):
    """Holds a policy, plotter, and data container to run inference.
    Can plot using convenience methods (wrapper around Plotter methods)
    or plot manually using data in self.data.expert_df and self.data.agent_df.
    """
    
    def __init__(self, policy, data_container: DataContainer,
                 plotter: EvaluationPlotter, device: str):
        self.policy = policy
        self.data = data_container
        self.plotter = plotter
        self.device = device

    @abstractmethod
    def run(self, *args, **kwargs) -> None: ...

    # ---- Diagnostics on the policy ----------------------------------

    def run_saliency_check(self):
        n = len(self.data.dataset.curr_compact_idxs)
        idx = np.random.randint(low=0, high=n)
        compact_idx = self.data.dataset.curr_compact_idxs[idx]

        x_glob = self.data.dataset.states[compact_idx].unsqueeze(0).to(self.device)
        x_bin  = self.data.dataset.bin_states[compact_idx].unsqueeze(0).to(self.device)
        self.policy.core_net.to(self.device)
        x_glob.requires_grad_(True)
        x_bin.requires_grad_(True)

        scores = self.policy.core_net(x_glob, x_bin)
        target = scores[0].max()
        self.policy.core_net.zero_grad()
        target.backward()

        bin_grads = x_bin.grad[0].abs().mean(dim=0)
        peak = bin_grads.max().item()
        for i, name in enumerate(self.data.dataset.bin_feature_names):
            print(f'Feature: {name:30} | Gradient: {bin_grads[i].item() / peak:.6f}')

    def plot_layer1_weights(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(20, 5))
        names = self.data.dataset.global_feature_names + self.data.dataset.bin_feature_names
        weights = self.policy.core_net.net[0].weight.data.cpu().detach().numpy()
        means = weights.mean(axis=0)
        stds  = weights.std(axis=0)
        xs = np.arange(len(names))
        ax.errorbar(xs, means[:len(names)], yerr=stds[:len(names)], color='black', fmt='none')
        ax.scatter(xs, means[:len(names)], color='C0')
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=45)
        return ax

    # ---- Common plot pass-throughs ----------------------------------

    def plot_mollweide_res(self):
        self.plotter.plot_mollweide_res(
            timestamps=self.data.agent_df['timestamp'].values,
            expert_bin_idxs=self.data.expert_df['bin_idx'],
            agent_bin_idxs=self.data.agent_df['bin_idx'],
            field_pos=np.array([(self.data.lookups.fields.ra[fid], self.data.lookups.fields.dec[fid])
                                for fid in range(len(self.data.lookups.fields.index))]),
            nside=self.data.hpGrid.nside,
        )

    def plot_hist_comparison(self, feature_name, density=False, bins=20, use_weights=False, ax=None):
        return self.plotter.plot_hist_comparison(
            feature_name,
            expert_arr=self.data.expert_df[feature_name],
            agent_arr=self.data.agent_df[feature_name],
            density=density, bins=bins, use_weights=use_weights, ax=ax,
        )
    
    
    def plot_2dhist(self, feature_x: str, feature_y: str, bins=25):
        return self.plotter.plot_2dhist(
            feature_x, feature_y,
            self.data.expert_df[feature_x].values, self.data.expert_df[feature_y].values,
            self.data.agent_df[feature_x].values,  self.data.agent_df[feature_y].values,
            bins=bins,
        )

    def plot_2dhist_res(self, feature_x, feature_y, bins=25):
        return self.plotter.plot_2dhist_res(
            feature_x, feature_y,
            self.data.expert_df[feature_x], self.data.expert_df[feature_y],
            self.data.agent_df[feature_x],  self.data.agent_df[feature_y],
            bins=bins,
        )
        
    def plot_2dhist_per_filter(self, feature_x, feature_y, bins=25):
        for filt in FILTER2IDX.keys():
            exp_f_mask = self.data.expert_df['filter'].values == filt
            agent_f_mask = self.data.agent_df['filter'].values == filt
            self.plotter.plot_2dhist(
                feature_x=feature_x,
                feature_y=feature_y,
                expert_x=self.data.expert_df[feature_x][exp_f_mask],
                expert_y=self.data.expert_df[feature_y][exp_f_mask],
                agent_x=self.data.agent_df[feature_x][agent_f_mask],
                agent_y=self.data.agent_df[feature_y][agent_f_mask]
            )
            plt.suptitle(f'{filt}-band', fontsize=16)
            


# ----------------------------------------------------------------------
# Single-step
# ----------------------------------------------------------------------

class SingleStepEvaluator(Evaluator):
    """One-step-ahead inference."""

    N_INFERENCE_SLICES = 8

    def __init__(self, policy, data_container: SingleStepDataContainer,
                 plotter: EvaluationPlotter, device: str = 'cuda'):
        super().__init__(policy, data_container, plotter, device)

    def run(self) -> None:
        agent_bin_idxs, agent_filter_idxs = self._batch_single_step_validation()
        timestamps = self.data.expert_df['timestamp'].astype(int)

        self.data.populate_agent_df(agent_bin_idxs, agent_filter_idxs, timestamps)
        # IMPORTANT: convert agent_df to deg BEFORE computing errors so both
        # dataframes share units inside populate_errors_df.
        self.data.convert_to_deg(self.data.agent_df)
        self.data.populate_errors_df()
        self.data.convert_to_deg(self.data.errors_df)

    def _batch_single_step_validation(self):
        n_slices = self.N_INFERENCE_SLICES
        chunk = len(self.data.dataset.curr_compact_idxs) // n_slices
        outputs = []
        for i in range(n_slices):
            sl = slice(i * chunk, None if i == n_slices - 1 else (i + 1) * chunk)
            idxs = self.data.dataset.curr_compact_idxs[sl]
            with torch.no_grad():
                outputs.append(self.policy.select_action(
                    self.data.dataset.states[idxs].to(self.device),
                    self.data.dataset.bin_states[idxs].to(self.device),
                    self.data.dataset.action_masks[idxs].to(self.device),
                ))
        bin_idxs = torch.cat(outputs).cpu().detach().numpy()
        if 'filter' in self.data.action_space:
            filter_idxs = bin_idxs % _NUM_FILTERS
            bin_idxs    = bin_idxs // _NUM_FILTERS
            return bin_idxs, filter_idxs
        return bin_idxs, None

    # ---- Convenience pass-throughs ----------------------------------

    def plot_line_comparison(self, feature_name, ax=None):
        self._check_input_features(feature_name, None)
        return self.plotter.plot_line_comparison(
            feature_name, self.data.expert_df[feature_name].values,
            self.data.agent_df[feature_name], ax=ax,
        )

    def plot_scatter_comparison(self, feature_y, feature_x=None, ax=None):
        self._check_input_features(feature_x, feature_y)
        expert_y = self.data.expert_df[feature_y].values
        agent_y  = self.data.agent_df[feature_y].values
        if feature_x is None:
            expert_x = np.arange(len(expert_y))
            agent_x  = np.arange(len(agent_y))
        else:
            expert_x = self.data.expert_df[feature_x].values
            agent_x  = self.data.agent_df[feature_x].values
        return self.plotter.plot_scatter_comparison(
            feature_y, expert_x, expert_y, agent_x, agent_y,
            feature_x=feature_x, ax=ax,
        )

    def plot_residual(self, feature_y, feature_x=None, plot_type='hist', bins=20, alpha=0.2,
                      density=False, ax=None):
        return self.plotter.plot_residual(
            feature_y,
            self.data.expert_df[feature_y], self.data.agent_df[feature_y],
            feature_x=feature_x,
            expert_x=self.data.expert_df.get(feature_x, None),
            agent_x=self.data.agent_df.get(feature_x, None),
            plot_type=plot_type, bins=bins, alpha=alpha, density=density, ax=ax,
        )

    def plot_filter_bin_cdf(self):
        return self.plotter.plot_filter_bin_cdf(self.data.expert_df, self.data.errors_df)

    def plot_quiver(self, feature_x, feature_y, ax=None):
        self._check_input_features(feature_x, feature_y)
        # prev_expert_df holds the PREVIOUS state; expert_df holds the NEXT state.
        return self.plotter.plot_quiver(
            feature_x, feature_y,
            expert_prev_x=self.data.prev_expert_df[feature_x],
            expert_prev_y=self.data.prev_expert_df[feature_y],
            expert_next_x=self.data.expert_df[feature_x],
            expert_next_y=self.data.expert_df[feature_y],
            agent_next_x=self.data.agent_df[feature_x],
            agent_next_y=self.data.agent_df[feature_y],
            ax=ax,
        )

    def plot_hist_comparison_per_filter(self, feature_name, bins=20):
        return self.plotter.plot_filter_histograms(
            feature_name,
            expert_feature_arr=self.data.expert_df[feature_name].values,
            expert_filters=self.data.expert_df['filter'].values,
            agent_filters=self.data.agent_df['filter'].values,
            bins=bins,
        )
    

    def calculate_filter_confusion(self) -> np.ndarray:
        n = len(FILTER2IDX)
        conf_mat = np.zeros((n, n))
        for filt, idx in FILTER2IDX.items():
            mask = self.data.expert_df['filter_idx'].values == idx
            ag = self.data.agent_df['filter'].values[mask]
            total = mask.sum()
            if total == 0:
                continue
            for _fname, _fidx in FILTER2IDX.items():
                conf_mat[idx, _fidx] = (ag == _fname).sum() / total
        return conf_mat

    def plot_filter_confusion(self, ax=None):
        return self.plotter.plot_filter_confusion(self.calculate_filter_confusion(), ax=ax)

    def _check_input_features(self, feature_x, feature_y):
        assert feature_x in self.data.agent_df.columns, f"{feature_x} not in agent_df"
        assert feature_y in self.data.agent_df.columns, f"{feature_y} not in agent_df"

# ----------------------------------------------------------------------
# Multi-step
# ----------------------------------------------------------------------

class MultiStepEvaluator(Evaluator):
    """Whole-episode rollout via the offline runner."""

    def __init__(self, runner: OfflineRunner, env, policy,
                 data_container: MultiStepDataContainer,
                 plotter: EvaluationPlotter, device: str = 'cuda'):
        super().__init__(policy, data_container, plotter, device)
        self.runner = runner
        self.env = env
        self.eval_metrics = None
        self.outdir: Path = Path(runner.outdir)

    def run(self, outdir: Union[str, Path, None] = None, overwrite=False) -> None:
        self.outdir = Path(outdir or self.runner.outdir)
        metrics_path = self.outdir / 'eval_metrics.pkl'

        if metrics_path.exists() and not overwrite:
            with open(metrics_path, 'rb') as f:
                self.eval_metrics = pickle.load(f)
                logger.info(f"Results already exist in {metrics_path}. \
                            Pass overwrite=True to re-run.")
        else:
            self.eval_metrics = self.runner.run(env=self.env)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'wb') as f:
                pickle.dump(self.eval_metrics, f)

        ts, bin_idxs, filter_idxs, field_ids, glob_df, bin_feat_dict = \
            self._process_eval_metrics(self.eval_metrics)

        self.data.populate_agent_df(
            bin_idxs=bin_idxs, filter_idxs=filter_idxs, timestamps=ts,
            field_ids=field_ids, glob_df=glob_df, bin_feat_dict=bin_feat_dict,
        )

    def _process_eval_metrics(self, eval_metrics):
        # Deterministic eval: single episode.
        episode = eval_metrics['ep-0']

        # Detect format. The streaming runner stores {'manifest': {...}, ...};
        # the legacy in-memory runner stores {night_key: {arr_dict}, ...}.
        if isinstance(episode, dict) and 'manifest' in episode:
            return self._process_eval_metrics_from_manifest(episode['manifest'])
        return self._process_eval_metrics_from_memory(episode)

    def _process_eval_metrics_from_memory(self, episode):
        """Legacy path: per-night arrays already in RAM.

        Memory note: the original implementation built `bin_feats` (a list of
        rearranged per-night views), concatenated it into a single full-size
        array, AND held the per-feature dict simultaneously — ~3x peak.
        Here we concatenate per feature and drop intermediates as we go.
        """
        # Build the small flat columns first; these are cheap.
        night_keys = list(episode.keys())
        bin_arr = np.concatenate([np.asarray(episode[n]['bin']) for n in night_keys])
        field_arr = np.concatenate([np.asarray(episode[n]['field_id']) for n in night_keys])
        ts_arr = np.concatenate([np.asarray(episode[n]['timestamp']) for n in night_keys])
        filter_arr = np.concatenate([np.asarray(episode[n]['filter_idx']) for n in night_keys])
        night_col = np.concatenate(
            [np.full(len(episode[n]['bin']), n) for n in night_keys]
        )

        # Drop zenith/wait states.
        valid = (bin_arr != -1) & (bin_arr != -2)

        # Global features: build the DataFrame once from a single concatenated
        # array rather than concat-of-per-night-DataFrames.
        glob_arr = np.concatenate(
            [np.asarray(episode[n]['glob_observations'], dtype=np.float32)
             for n in night_keys],
            axis=0,
        )
        glob_df = pd.DataFrame(glob_arr, columns=self.data.dataset.global_feature_names)
        glob_df['night'] = night_col
        del glob_arr

        # Bin features: stream per feature to avoid holding a full
        # (nfeats, total_rows, nbins) intermediate AND a full per-feature
        # dict at the same time.
        feat_names = self.data.dataset.bin_feature_names
        bin_feat_dict = {}
        # Materialize the full (total_rows, nbins, nfeats) array once in float32
        # — we still need it, but at least at fp32 not fp64.
        bin_obs_all = np.concatenate(
            [np.asarray(episode[n]['bin_observations'], dtype=np.float32)
             for n in night_keys],
            axis=0,
        )
        # Drop the per-night references from the upstream dict to free
        # ~50% of duplicated memory before slicing.
        for n in night_keys:
            episode[n].pop('bin_observations', None)
        import gc as _gc
        _gc.collect()

        for i, name in enumerate(feat_names):
            # arr[valid, :, i] copies; assign and move on.
            bin_feat_dict[name] = bin_obs_all[valid, :, i].copy()
        del bin_obs_all
        _gc.collect()

        return (
            ts_arr[valid],
            bin_arr[valid],
            filter_arr[valid],
            field_arr[valid],
            glob_df[valid],
            bin_feat_dict,
        )

    def _process_eval_metrics_from_manifest(self, manifest):
        """Streaming path: per-night arrays live in .npz files on disk.

        We do two passes:
          1) Cheap pass: load each night's small columns, build flat
             arrays and the global `valid` mask. Each .npz load uses
             ``mmap_mode='r'`` so big arrays aren't materialized yet.
          2) Big pass: for ``glob_observations``, concat once. For
             ``bin_observations``, allocate the output dict once and
             fill per-feature, only ever holding one night's bin
             observations in RAM at a time.
        """
        night_keys = list(manifest.keys())

        # ---- Pass 1: small per-step columns + valid mask ----
        # Open each .npz once; small columns are cheap to materialize.
        # Bigger arrays (`glob_observations`, `bin_observations`) stay
        # mmapped until we explicitly load them.
        bin_cols, field_cols, ts_cols, filter_cols, night_label_cols = [], [], [], [], []
        # Cache shapes so pass 2 can preallocate without re-opening.
        bin_obs_shapes = {}
        for n in night_keys:
            with np.load(manifest[n], mmap_mode='r') as npz:
                bin_cols.append(np.asarray(npz['bin']))
                field_cols.append(np.asarray(npz['field_id']))
                ts_cols.append(np.asarray(npz['timestamp']))
                filter_cols.append(np.asarray(npz['filter_idx']))
                night_label_cols.append(np.full(len(npz['bin']), n))
                # 'bin_observations' shape is (nrows, nbins, nfeats)
                bin_obs_shapes[n] = npz['bin_observations'].shape

        bin_arr = np.concatenate(bin_cols)
        field_arr = np.concatenate(field_cols)
        ts_arr = np.concatenate(ts_cols)
        filter_arr = np.concatenate(filter_cols)
        night_col = np.concatenate(night_label_cols)
        del bin_cols, field_cols, ts_cols, filter_cols, night_label_cols

        valid = (bin_arr != -1) & (bin_arr != -2)
        n_valid = int(valid.sum())

        # ---- Pass 2a: glob_observations (single concat) ----
        feat_names_glob = self.data.dataset.global_feature_names
        glob_parts = []
        for n in night_keys:
            with np.load(manifest[n]) as npz:
                glob_parts.append(np.asarray(npz['glob_observations'], dtype=np.float32))
        glob_arr = np.concatenate(glob_parts, axis=0)
        del glob_parts
        glob_df = pd.DataFrame(glob_arr, columns=feat_names_glob)
        glob_df['night'] = night_col
        del glob_arr

        # ---- Pass 2b: bin_observations per-feature ----
        # Preallocate the per-feature output arrays, then fill them by
        # streaming one night at a time. Peak extra RAM per iteration is
        # ONE night's (nrows, nbins, nfeats) — bounded and small.
        feat_names_bin = self.data.dataset.bin_feature_names
        # Infer (nbins, nfeats) from the first night.
        first_shape = next(iter(bin_obs_shapes.values()))
        nbins = first_shape[1]
        bin_feat_dict = {
            name: np.empty((n_valid, nbins), dtype=np.float32)
            for name in feat_names_bin
        }

        # Compute per-night offsets into the global valid index space.
        write_offset = 0
        row_offset = 0
        import gc as _gc
        for n in night_keys:
            with np.load(manifest[n]) as npz:
                bin_obs = np.asarray(npz['bin_observations'], dtype=np.float32)
            nrows = bin_obs.shape[0]
            night_valid = valid[row_offset:row_offset + nrows]
            n_take = int(night_valid.sum())
            if n_take > 0:
                # Slice once, then index per feature into preallocated output.
                bin_obs_valid = bin_obs[night_valid]  # (n_take, nbins, nfeats)
                for i, name in enumerate(feat_names_bin):
                    bin_feat_dict[name][write_offset:write_offset + n_take] = bin_obs_valid[:, :, i]
                del bin_obs_valid
                write_offset += n_take
            del bin_obs
            row_offset += nrows
            _gc.collect()

        return (
            ts_arr[valid],
            bin_arr[valid],
            filter_arr[valid],
            field_arr[valid],
            glob_df[valid],
            bin_feat_dict,
        )

    # ---- MS-specific plots ------------------------------------------

    def plot_kde_and_scatter(self, feature_x, feature_y, agent_alpha=0.2,
                             use_filter_coloring=True, s=5, ax=None):
        color_mapping = (
            self.data.agent_df['filter'].map(FILTER_COLORS)
            if use_filter_coloring else None
        )
        return self.plotter.plot_kde_and_scatter(
            feature_x, feature_y,
            self.data.expert_df[feature_x], self.data.expert_df[feature_y],
            self.data.agent_df[feature_x],  self.data.agent_df[feature_y],
            agent_alpha=agent_alpha, color_mapping=color_mapping, s=s, ax=ax,
        )

    def plot_filter_histograms(self, feature_name, bins=20):
        return self.plotter.plot_filter_histograms(
            feature_name,
            expert_feature_arr=self.data.expert_df[feature_name].values,
            agent_feature_arr=self.data.agent_df[feature_name].values,
            expert_filters=self.data.expert_df['filter'].values,
            agent_filters=self.data.agent_df['filter'].values,
            bins=bins,
        )

    def plot_hist_comparison(self, feature_name, density=False, bins=20, use_weights=False, ax=None):
        # MS expert_df has gaps masked via expert_valid_mask.
        expert_arr = self.data.expert_df[feature_name][self.data.expert_valid_mask]
        return self.plotter.plot_hist_comparison(
            feature_name, expert_arr=expert_arr,
            agent_arr=self.data.agent_df[feature_name],
            density=density, bins=bins, use_weights=use_weights, ax=ax,
        )
        
    def plot_2dhist_res(self, feature_x, feature_y, bins=25):
        return self.plotter.plot_2dhist_res(
            feature_x, feature_y,
            self.data.expert_df[feature_x], self.data.expert_df[feature_y],
            self.data.agent_df[feature_x],  self.data.agent_df[feature_y],
            bins=bins,
        )
        
    def _calculate_performance_metrics(self):
        """Placeholder for episode-level metrics (reward, coverage, etc.)."""
        metrics = {}
        
        pass
