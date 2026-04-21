from abc import ABC
import os
from pathlib import Path
import pickle

from einops import rearrange
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

from blancops.data.constants import *
from blancops.data.lookup import load_lookup_tables
from blancops.data.dataset import OfflineDataset
from blancops.data.features.normalizations import load_normalization_stats
from blancops.math.geometry import angular_separation
from blancops.ephemerides import ephemerides

from blancops.math import units
from blancops.ephemerides.ephemerides import get_source_ra_dec
from blancops.data.features.glob_features import calc_moon_phase as _calc_moon_phase
from blancops.data.features.glob_features import calc_sun_and_moon_positions as _calc_sun_and_moon_pos
from blancops.data.preprocessing import load_train_data_to_dataframe
from blancops.configs.constants import TRAIN_DATA_DIR, TRAIN_DATA_PATH

from blancops.plotting.plotting import plot_schedule_whole

from matplotlib.patches import Patch
import matplotlib.colors as colors
import seaborn as sns

FILTER_COLORS = {
    'g': '#00b25d', # Green
    'r': '#f97306', # Orange
    'i': '#e50000', # Red
    'z': '#8b0000', # Dark Red
    'Y': '#3b0000'  # Near-IR (Darkest)
}

FILTER_PATCHES = [Patch(color=v, label=k) for k, v in FILTER_COLORS.items()]

def get_validation_dataset(cfg):
    outdir = Path(cfg.experiment_outdir)
    df = load_train_data_to_dataframe(TRAIN_DATA_PATH)
    df_val = df[df['night'].isin(cfg.data.val_nights)]
    z_score_stats, rel_norm_stats = load_normalization_stats(outdir)
    val_dataset = OfflineDataset(
        df=df_val,
        cfg=cfg,
        z_score_stats=z_score_stats,
        rel_norm_stats=rel_norm_stats
    )
    return val_dataset

def build_evaluators(trainer, policy, val_dataset, action_space, device, agent_color='green', expert_color='purple', agent_cmap='Greens', expert_cmap='Purples', res_cmap='PRGn',
                     res_color='grey'):
    ss_data = DataContainer(val_dataset, action_space, eval_method='ss')
    ms_data = DataContainer(val_dataset, action_space, eval_method='ms')
    s_plotter = EvaluationPlotter(agent_color, expert_color, agent_cmap, expert_cmap, res_cmap, res_color, eval_method='ss')
    m_plotter = EvaluationPlotter(agent_color, expert_color, agent_cmap, expert_cmap, res_cmap, res_color, eval_method='ms')
    s_evaluator = SingleStepEvaluator(policy, ss_data, s_plotter, device)
    m_evaluator = MultiStepEvaluator(trainer, policy, ms_data, m_plotter, device)
    return s_evaluator, m_evaluator

def calc_airmass(el):
    return 1 / np.cos(np.pi/2 - el)
    
def calc_slew_distance(prev_radecs, radecs):
    slew_dists = np.zeros(shape=(len(prev_radecs)))
    for i in range(len(prev_radecs)):
        slew_dists[i] = angular_separation(prev_radecs[i], radecs[i])
    return slew_dists 

def calc_moon_dist(radecs, timestamps):
    moon_dists = np.zeros((len(timestamps)))
    for i, t in enumerate(timestamps):
        moon_radec = get_source_ra_dec('moon', time=t)
        moon_dists[i] = angular_separation(moon_radec, radecs[i]) 
    return moon_dists

def calc_moon_phase(timestamps):
    moon_phase_arr = np.empty(len(timestamps))
    for i, t in enumerate(timestamps):
        moon_phase_arr[i] = _calc_moon_phase(t)
    return moon_phase_arr

def calc_sun_and_moon_pos(timestamps):
    sun_azel = np.empty((len(timestamps), 2))
    moon_azel = np.empty((len(timestamps), 2))
    for i, t in enumerate(timestamps):
        _, sun_azel[i], _, moon_azel[i] = _calc_sun_and_moon_pos(t)
    return sun_azel[:, 0], sun_azel[:, 1], moon_azel[:, 0], moon_azel[:, 1]

class EvaluationPlotter:
    def __init__(self, agent_color='green', expert_color='purple', agent_cmap='Greens', expert_cmap='Purples', res_cmap='PRGn', res_color='slateblue', save_dir=None, eval_method=None):
        self.agent_color = agent_color
        self.expert_color = expert_color
        self.agent_cmap = agent_cmap
        self.expert_cmap = expert_cmap
        self.res_cmap = res_cmap
        self.res_color = res_color
        if save_dir is None:
            self.save_dir = Path('./eval_outdir').resolve()
        else:
            self.save_dir = (Path(save_dir) / eval_method).resolve()

    @staticmethod
    def _get_wrapped_ra(ra):
        return ((ra + 180) % 360 - 180)

    def _wrap_if_ra(self, feature_name, feature_arr):
        if feature_name is None:
            return
        if (feature_name == 'ra') or ('_ra' in feature_name):
            return self._get_wrapped_ra(feature_arr)
        else:
            return feature_arr
        
    def plot_2dhist(self, feature_x, feature_y, expert_x, expert_y, agent_x, agent_y, norm=colors.LogNorm(), bins=25, return_plt_objects=False):
        fig, axs = plt.subplots(1, 2, figsize=(14,5), sharex=True, sharey=True)
        expert_x = self._wrap_if_ra(feature_x, expert_x)
        expert_y = self._wrap_if_ra(feature_y, expert_y)
        agent_x = self._wrap_if_ra(feature_x, agent_x)
        agent_y = self._wrap_if_ra(feature_y, agent_y)
        exp_counts, xedges, yedges, im1 = axs[0].hist2d(x=expert_x, y=expert_y, bins=bins, cmap=self.expert_cmap, norm=norm)
        fig.colorbar(im1, ax=axs[0], location="right", label='Counts')
        axs[0].set_xlabel(feature_x)
        axs[0].set_ylabel(feature_y)
        axs[0].set_title('Expert')
        ag_counts, xedges, yedges, im2 = axs[1].hist2d(x=agent_x, y=agent_y, bins=bins, cmap=self.agent_cmap, norm=norm);
        axs[1].set_xlabel(feature_x)
        axs[1].set_ylabel(feature_y)
        axs[1].set_title('BC')
        fig.colorbar(im2, ax=axs[1], location="right", label='Counts')
        if return_plt_objects:
            return fig, axs, exp_counts, ag_counts
    
    def plot_2dhist_res(self, feature_x, feature_y, expert_x, expert_y, agent_x, agent_y, bins=25, return_plt_objects=False, density=False):
        expert_x = self._wrap_if_ra(feature_x, expert_x)
        expert_y = self._wrap_if_ra(feature_y, expert_y)
        agent_x = self._wrap_if_ra(feature_x, agent_x)
        agent_y = self._wrap_if_ra(feature_y, agent_y)
        
        bins = [bins, bins]
        
        # Dynamically match the auto-bounds from the normal plot
        xmin = min(expert_x.min(), agent_x.min())
        xmax = max(expert_x.max(), agent_x.max())
        ymin = min(expert_y.min(), agent_y.min())
        ymax = max(expert_y.max(), agent_y.max())
        hrange = [[xmin, xmax], [ymin, ymax]]

        exp_hist, xedges, yedges = np.histogram2d(expert_x, expert_y, bins=bins, range=hrange, density=density)
        agent_hist, _, _ = np.histogram2d(agent_x, agent_y, bins=bins, range=hrange, density=density)

        res = agent_hist - exp_hist
        lim = np.max(np.abs(res)) 
        
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(res.T, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap=self.res_cmap, aspect='auto',
                       vmin=-lim, vmax=lim)
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        fig.colorbar(im, ax=ax, label="Residual counts \n (agent - expert)")
        if return_plt_objects:
            return exp_hist, agent_hist

    def plot_mollweide_res(self, timestamps, expert_bin_idxs, agent_bin_idxs, field_pos, nside):
        plot_schedule_whole(
            outfile=self.save_dir / "mollweide_residuals",
            times=timestamps,
            field_pos=None,
            bin_idxs=expert_bin_idxs,
            alternate_bin_idxs=agent_bin_idxs,
            nside=nside,
            sky_bin_mapping=None,
            projection="mollweide",
            center_pos=(None, None),
            schedule_label="",
            )
        
    def plot_line_comparison(self, feature_name, expert_arr, agent_arr):
        expert_arr = self._wrap_if_ra(feature_name, expert_arr)
        agent_arr = self._wrap_if_ra(feature_name, agent_arr)

        plt.plot(expert_arr, label='expert', color=self.expert_color)
        plt.plot(agent_arr, label='agent', color=self.agent_color)
        plt.xlabel('Time', fontsize=16)
        plt.ylabel(feature_name, fontsize=16)
        plt.legend(fontsize=16)
    
    def plot_scatter_comparison(self, feature_y, expert_x, expert_y, agent_x, agent_y, feature_x=None):
        plt.scatter(expert_x, expert_y, label='expert', color=self.expert_color)
        plt.scatter(agent_x, agent_y, label='agent', color=self.agent_color)
        if feature_x is not None:
            plt.xlabel(feature_x, fontsize=16)
        plt.ylabel(feature_y, fontsize=16)
        plt.legend(fontsize=16)
        
    def plot_hist_comparison(self, feature_name, expert_arr, agent_arr, density=False, bins=20, alpha=.2, use_weights=False):
        if isinstance(bins, int) and (feature_name is not 'filter'):
            min_val = min(np.min(expert_arr), np.min(agent_arr))
            max_val = max(np.max(expert_arr), np.max(agent_arr))
            shared_bins = np.linspace(min_val, max_val, bins + 1)
        else:
            shared_bins = bins

        ag_weights = np.ones_like(agent_arr) / len(agent_arr) if use_weights else None
        exp_weights = np.ones_like(expert_arr) / len(expert_arr) if use_weights else None
        
        if use_weights:
            density = False

        plt.hist(expert_arr, bins=shared_bins, alpha=1, density=density, histtype='step', color=self.expert_color, weights=exp_weights)
        plt.hist(agent_arr, bins=shared_bins, alpha=1, density=density, histtype='step', color=self.agent_color, weights=ag_weights)
        
        # Plot fill (stepfilled)
        plt.hist(expert_arr, bins=shared_bins, label='expert', histtype='stepfilled', color=self.expert_color, alpha=alpha, density=density, weights=exp_weights)
        plt.hist(agent_arr, bins=shared_bins, label='agent', histtype='stepfilled', color=self.agent_color, alpha=alpha, density=density, weights=ag_weights)

        plt.xlabel(feature_name, fontsize=16)
        
        # 3. Accurate labeling
        if density:
            ylabel = 'Probability Density'
        elif use_weights:
            ylabel = 'Fraction of total'
        else:
            ylabel = 'Raw counts'
            
        plt.ylabel(ylabel, fontsize=16)
        plt.legend(fontsize=16)
        
    def plot_residual(self, feature_y, expert_y, agent_y, feature_x=None, expert_x=None, agent_x=None, plot_type='hist', bins=20, alpha=.2, density=False):
        expert_x = self._wrap_if_ra(feature_x, expert_x)
        expert_y = self._wrap_if_ra(feature_y, expert_y)
        agent_x = self._wrap_if_ra(feature_x, agent_x)
        agent_y = self._wrap_if_ra(feature_y, agent_y)
        
        assert plot_type in ['hist', 'line']
        res_y = agent_y - expert_y
        if expert_x is None:
            expert_x = np.arange(len(res_y))
            agent_x = np.arange(len(res_y))
        if plot_type == 'hist':
            plt.hist(res_y, bins=bins, alpha=1, density=density, histtype='step', color=self.res_color)
            plt.hist(res_y, bins=bins, histtype='stepfilled', color=self.res_color, alpha=alpha, density=density)
            plt.xlabel(feature_y)
            ylabel = 'Raw (residual) counts' if not density else 'Normalized (residual) counts'
            plt.ylabel(ylabel, fontsize=16)
        elif plot_type == 'line':
            plt.plot(expert_x, res_y, color=self.res_color, marker='o')    
    
    def plot_filter_confusion(self, conf_mat):
        plt.figure()
        sns.heatmap(conf_mat, annot=True, cmap=self.expert_cmap, 
                    xticklabels=FILTER2IDX.keys(), yticklabels=FILTER2IDX.keys())
        plt.xlabel('Expert')
        plt.ylabel('Agent')
        plt.title("Filter")
        plt.show()
    
    def plot_filter_bin_cdf(self, expert_df, errors_df):
        plt.figure(figsize=(5, 4))
        for i, filt in enumerate(FILTER2IDX.keys()):
            f_mask = filt == expert_df['filter']
            f_color = FILTER_COLORS[filt]
            sorted_errors = np.sort(errors_df['bin_angular_separation'][f_mask])
            y_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            
            # 3. Visualization
            plt.plot(sorted_errors, y_values, lw=4, alpha=.7, color=f"C{i}")
            
            # Add a "Tolerance" indicator (e.g., 5 degrees)
            tolerance = 5.0
            percentile = np.searchsorted(sorted_errors, tolerance) / len(sorted_errors)
            plt.axhline(y=percentile, color=f_color, linestyle='dotted', alpha=1, lw=1.5,
                        label=f'{filt}: {percentile*100:.1f}% within {tolerance}°')

            plt.xlabel('Angular Separation (deg)', fontsize=14)
            plt.ylabel('CDF', fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='lower right', fontsize=14)  
            plt.xlim(0, max(sorted_errors))
            plt.ylim(0, 1.05)
        plt.axvline(x=tolerance, color='Red', linestyle='solid', alpha=1, lw=1.5)
        plt.show()
        
    def plot_filter_histograms(self, feature_name, expert_feature_arr, expert_filters, agent_filters, agent_feature_arr=None, density=True, alpha=.2, bins=20):
        nrows = 2
        ncols = 3
        fig = plt.figure(figsize=(ncols * 4, nrows * 3))
        i = 1
        for filt in FILTER2IDX.keys():
            exp_m = expert_filters == filt
            ag_m = agent_filters == filt
            ax = fig.add_subplot(nrows, ncols, i)
            ax.hist(agent_feature_arr[ag_m], bins=bins, alpha=1, density=density, color=self.agent_color, histtype='step', lw=1.5)
            ax.hist(expert_feature_arr[exp_m], bins=bins, alpha=1, density=density, color=self.expert_color, histtype='step', lw=1.5)
            
            ax.hist(agent_feature_arr[ag_m], bins=bins, label=f'Agent chose {filt}', histtype='stepfilled', color=self.agent_color, alpha=alpha, density=density)
            ax.hist(expert_feature_arr[exp_m], bins=bins, label=f'Expert chose {filt}', histtype='stepfilled', color=self.expert_color, alpha=alpha, density=density)
            ax.set_xlabel(feature_name)
            ax.legend();
            i+=1
        fig.tight_layout()
        
    def plot_kde_and_scatter(self, feature_x, feature_y, expert_x, expert_y, agent_x, agent_y, agent_alpha=.2, color_mapping=None, s=5):
        expert_x = self._wrap_if_ra(feature_x, expert_x)
        expert_y = self._wrap_if_ra(feature_y, expert_y)
        agent_x = self._wrap_if_ra(feature_x, agent_x)
        agent_y = self._wrap_if_ra(feature_y, agent_y)
        
        if color_mapping is None:
            color_mapping = self.agent_color
            agent_label = 'agent'
            handles = None
        else:
            agent_label = None
            handles = FILTER_PATCHES
        ax = sns.kdeplot(x=expert_x, y=expert_y, 
                    thresh=0, levels=15, cmap=self.expert_cmap, alpha=1, label='expert', fill=False, norm=colors.CenteredNorm())
        plt.scatter(agent_x, agent_y, marker='*', c=color_mapping, label=agent_label, alpha=agent_alpha, s=s)
        plt.xlabel(feature_x, fontsize=16)
        plt.ylabel(feature_y, fontsize=16)
        plt.legend(handles=handles, fontsize=16)
        
    def plot_quiver(self, feature_x, feature_y, expert_prev_x, expert_prev_y, expert_next_x, expert_next_y, agent_next_x, agent_next_y):
        expert_next_x = self._wrap_if_ra(feature_x, expert_next_x)
        expert_next_y = self._wrap_if_ra(feature_y, expert_next_y)
        expert_prev_x = self._wrap_if_ra(feature_x, expert_prev_x)
        expert_prev_y = self._wrap_if_ra(feature_y, expert_prev_y)
        agent_next_x = self._wrap_if_ra(feature_x, agent_next_x)
        agent_next_y = self._wrap_if_ra(feature_y, agent_next_y)
        
        expert_u = expert_next_x - expert_prev_x
        expert_v = expert_next_y - expert_prev_y
        agent_u = agent_next_x - expert_prev_x
        agent_v = agent_next_y - expert_prev_y
        
        plt.quiver(expert_prev_x, expert_prev_y, expert_u, expert_v, color=self.expert_color, label='expert')
        plt.quiver(expert_prev_x, expert_prev_y, agent_u, agent_v, color=self.agent_color, label='agent')
        plt.xlabel(feature_x, fontsize=14)
        plt.ylabel(feature_y, fontsize=14)
        plt.legend()
    
class DataContainer():
    """Creates and processes dataframes holding expert and agent data. Right now, assumes all data comes from train data"""
    def __init__(self, val_dataset: OfflineDataset, action_space: str, eval_method: str = 'ss'):
        assert eval_method in ['ss', 'ms']
        self.eval_method = eval_method
        self.hpGrid = val_dataset.hpGrid
        self.is_azel = val_dataset.hpGrid.is_azel
        self.action_space = action_space
        self.dataset = val_dataset
        angular_features = ['ra', 'dec', 'az', 'el', 'slew_dist', 'moon_distance', 'angular_distance', 'bin_angular_separation']
        angular_features_prefixed = ['_' + feat for feat in angular_features]
        self._angle_features = angular_features + angular_features_prefixed
        self._populate_expert_df()
        self.segments = self._get_expert_idx_segments()
        
        self.lookup = load_lookup_tables(TRAIN_DATA_DIR)
        self.errors_df = pd.DataFrame()
            
    def populate_errors_df(self):
        angseps_arr = np.zeros(len(self.agent_df))
        for i, (pos1, pos2) in enumerate(zip(self.expert_df[['bin_ra', 'bin_dec']].to_numpy() * units.deg, self.agent_df[['bin_ra', 'bin_dec']].to_numpy())):
            angseps_arr[i] = angular_separation(pos1, pos2)
        self.errors_df['timestamp'] = self.expert_df['timestamp']
        self.errors_df['bin_angular_separation'] = angseps_arr

    def _populate_expert_df(self):
        """Populates expert df (and, for single-step, df containing previous state features). Converts radians to degrees for each radian defined column."""
        if self.eval_method == 'ss':
            self.expert_df, self.prev_expert_df = self._get_ss_expert_df()
            self._populate_expert_dfs_with_derived_cols(self.expert_df, self.prev_expert_df)
        elif self.eval_method == 'ms':
            self.expert_df = self._get_ms_expert_df()
            self._populate_expert_dfs_with_derived_cols(self.expert_df, prev_expert_df=None) # Assuming all states connected by action
        self.convert_df_to_deg(self.expert_df)
            
    def _get_ss_expert_df(self):
        expert_df = pd.DataFrame()
        expert_df = self._extract_expert_data(expert_df, desired_indices=self.dataset.next_state_idxs)
        prev_expert_df = pd.DataFrame()
        prev_expert_df = self._extract_expert_data(prev_expert_df, desired_indices=self.dataset.current_state_idxs)
        return expert_df, prev_expert_df
    
    def _get_ms_expert_df(self):
        expert_df = pd.DataFrame()
        expert_df = self._extract_expert_data(expert_df, desired_indices=self.dataset.state_idxs)
        self.expert_valid_mask = self._get_valid_state_mask(expert_df['timestamp'].values, max_time_diff_min=5)
        return expert_df
    
    def _extract_expert_data(self, expert_df, desired_indices):
        filtered_df =  self.dataset._df.iloc[desired_indices].reset_index(drop=True)
        
        bin_idxs = filtered_df['bin'].values.copy()
        z_mask = bin_idxs == -1
        if z_mask.any() and self.is_azel:
            zenith_bin = self.hpGrid.ang2idx(lon=0, lat=np.pi/2)
            bin_idxs[z_mask] = zenith_bin
        
        expert_df['bin_idx'] = bin_idxs
        expert_df['filter_idx'] = filtered_df['filt_idx'].values

        expert_df['timestamp'] = filtered_df['timestamp'].values
        expert_df['filter'] = expert_df['filter_idx'].map(IDX2FILTER).fillna(-1)
        
        expert_df['bin_az'], expert_df['bin_el'], expert_df['bin_ra'], expert_df['bin_dec'] = \
            self._get_bin_coords(expert_df['bin_idx'].values, timestamps=expert_df['timestamp'].values)
        expert_df['ra'], expert_df['dec'] = np.array([filtered_df.ra.values, filtered_df.dec.values])
        expert_df['az'], expert_df['el'] = np.array([filtered_df.az.values, filtered_df.el.values])
        
        # environmental params
        expert_df['airmass'] = filtered_df.airmass.values
        expert_df['moon_phase'] = filtered_df.get('moon_phase', None)
        expert_df['fwhm'] = filtered_df.get('fwhm', None)
        expert_df['sun_az'] = filtered_df.get('sun_az', None)
        expert_df['sun_el'] = filtered_df.get('sun_el', None)
        expert_df['moon_el'] = filtered_df.get('moon_el', None)
        expert_df['moon_az'] = filtered_df.get('moon_az', None)
        for feat in self.dataset.global_feature_names:
            if feat not in expert_df.columns:
                expert_df[feat] = filtered_df[feat]
        return expert_df

    def _populate_expert_dfs_with_derived_cols(self, expert_df, prev_expert_df=None):
        bin_radecs = expert_df[['bin_ra', 'bin_dec']].to_numpy()
        radecs = expert_df[['ra', 'dec']].to_numpy()
        bin_els = expert_df['bin_el'].to_numpy()
        timestamps = expert_df['timestamp'].values
        # State quantities
        expert_df['bin_moon_distance'] = calc_moon_dist(bin_radecs, timestamps)
        expert_df['moon_distance'] = calc_moon_dist(radecs, timestamps)
        expert_df['bin_airmass'] = calc_airmass(bin_els) # pointing airmass already included

        if self.eval_method == 'ss':
            # Uses prev_expert_df
            prev_bin_radecs = prev_expert_df[['bin_ra', 'bin_dec']].to_numpy()
            prev_radecs = prev_expert_df[['ra', 'dec']].to_numpy()
            prev_timestamps = prev_expert_df['timestamp'].values
            prev_bin_els = prev_expert_df['bin_el'].to_numpy()

            # Previous state quantities
            self.prev_expert_df['bin_moon_distance'] = calc_moon_dist(prev_bin_radecs, prev_timestamps)
            self.prev_expert_df['moon_distance'] = calc_moon_dist(prev_radecs, prev_timestamps)
            self.prev_expert_df['bin_airmass'] = calc_airmass(prev_bin_els)
            
            # Transition quantities
            expert_df['bin_slew_dist'] = calc_slew_distance(prev_radecs=prev_bin_radecs, radecs=bin_radecs)
            expert_df['slew_dist'] = calc_slew_distance(prev_radecs=prev_radecs, radecs=radecs)
            
        elif self.eval_method == 'ms':
            # Reconstruct previous quantities
            prev_bin_radecs = expert_df[['bin_ra', 'bin_dec']].shift(1).to_numpy()
            prev_radecs = expert_df[['ra', 'dec']].shift(1).to_numpy()
            
            bin_slew_dists = calc_slew_distance(prev_radecs=prev_bin_radecs, radecs=bin_radecs)
            slew_dists = calc_slew_distance(prev_radecs=prev_radecs, radecs=radecs)
            
            bin_slew_dists[~self.expert_valid_mask] = np.nan
            slew_dists[~self.expert_valid_mask] = np.nan

            # 5. Assign to dataframe
            expert_df['bin_slew_dist'] = bin_slew_dists
            expert_df['slew_dist'] = slew_dists

    def _get_valid_state_mask(self, timestamps, max_time_diff_min):
        """Calculates masks which discludes all invalid "next states"."""
        time_diffs = np.diff(timestamps)
        max_time_diff_sec = max_time_diff_min * 60
        valid_mask = time_diffs.astype(float) <= max_time_diff_sec
        valid_mask = np.insert(valid_mask, 0, False)
        return valid_mask
    
    def populate_agent_df(self, bin_idxs, filter_idxs, timestamps, field_ids=None, glob_df=None, bin_feat_dict=None):
        """Populates attribute agent_df (called after agent evaluation)"""
        if self.eval_method == 'ss':
            self.agent_df = self._get_ss_agent_df(bin_idxs, filter_idxs, timestamps)
        elif self.eval_method == 'ms':
            self.agent_df = self._get_ms_agent_df(bin_idxs, filter_idxs, timestamps, field_ids, glob_df, bin_feat_dict)
    
    def _get_base_agent_df(self, bin_idxs, filter_idxs, timestamps):
        """Agent df properties that are common for both single and multi-step evaluation. (Multi-step requires evaluating state at each timestamp)"""
        agent_df = pd.DataFrame()
        agent_df['bin_idx'] = bin_idxs
        agent_df['timestamp'] = timestamps
        agent_df['bin_az'], agent_df['bin_el'], agent_df['bin_ra'], agent_df['bin_dec'] = self._get_bin_coords(bin_idxs, timestamps=timestamps)
        agent_df['filter_idx'] = filter_idxs
        agent_df['filter'] = agent_df['filter_idx'].map(IDX2FILTER)
        agent_df['az'], agent_df['el'], agent_df['ra'], agent_df['dec'] = None, None, None, None
        return agent_df
    
    def _get_ss_agent_df(self, bin_idxs, filter_idxs, timestamps):
        """Agent df properties specific to single-step evaluation (equvialent to expert df, mostly)"""
        agent_df = self._get_base_agent_df(bin_idxs, filter_idxs, timestamps)    
        agent_df['bin_airmass'] = calc_airmass(agent_df['bin_el'])
        agent_df['moon_phase'] = self.expert_df['moon_phase']
        agent_df['fwhm'] = self.expert_df['fwhm'] # uses interpolation...
        agent_df['sun_az'] = self.expert_df['sun_az']
        agent_df['sun_el'] = self.expert_df['sun_el']
        agent_df['moon_az'] = self.expert_df['moon_az']
        agent_df['moon_el'] = self.expert_df['moon_el']
 
        bin_radecs = agent_df[['bin_ra', 'bin_dec']].to_numpy()
        prev_bin_radecs = self.prev_expert_df[['bin_ra', 'bin_dec']].to_numpy()
        
        # State quantities
        agent_df['bin_moon_distance'] = calc_moon_dist(bin_radecs, timestamps)

        # Transition quantities
        agent_df['bin_slew_dist'] = calc_slew_distance(prev_radecs=prev_bin_radecs, radecs=bin_radecs)
        return agent_df
    
    def _get_ms_agent_df(self, bin_idxs, filter_idxs, timestamps, field_ids, glob_df, bin_feat_dict):
        agent_df = self._get_base_agent_df(bin_idxs, filter_idxs, timestamps)
        agent_df['ra'], agent_df['dec'], agent_df['az'], agent_df['el'] = self._get_field_coords(field_ids, timestamps)
        agent_df['bin_airmass'] = calc_airmass(agent_df['bin_el'])
        agent_df['moon_phase'] = calc_moon_phase(timestamps)
        agent_df['sun_az'], agent_df['sun_el'], agent_df['moon_az'], agent_df['moon_el'] = calc_sun_and_moon_pos(timestamps)
        agent_df['bin_airmass'] = calc_airmass(el=agent_df['bin_el'].values)
        agent_df['airmass'] = calc_airmass(el=agent_df['el'].values)
        
        radecs = agent_df[['ra', 'dec']].to_numpy()
        bin_radecs = agent_df[['bin_ra', 'bin_dec']].to_numpy()
        
        # State quantities
        agent_df['bin_moon_distance'] = calc_moon_dist(bin_radecs, timestamps)
        agent_df['moon_distance'] = calc_moon_dist(radecs, timestamps)

        # Transition quantities
        bin_slew_dists = calc_slew_distance(prev_radecs=bin_radecs[:-1], radecs=bin_radecs[1:])
        bin_slew_dists = np.insert(bin_slew_dists, 0, np.nan)
        bin_slew_dists[~self.agent_valid_mask] = np.nan
        slew_dists = calc_slew_distance(prev_radecs=radecs[:-1], radecs=radecs[1:])
        slew_dists = np.insert(slew_dists, 0, np.nan)
        slew_dists[~self.agent_valid_mask] = np.nan
        
        agent_df['bin_slew_dist'] = bin_slew_dists
        agent_df['slew_dist'] = slew_dists
                
        
        # Rest of features...(in normalized form) #TODO: inverse normalize for direct comparison to expert for these features 
        for feat in self.dataset.global_feature_names:
            if feat not in agent_df.columns:
                agent_df[feat] = glob_df[feat]
        
        agent_df['t_survey'] = agent_df['t_survey'] / 2 + .5
        
        # Save bin feature dict
        self.agent_bin_feat_dict = bin_feat_dict
        self.agent_valid_mask = self._get_valid_state_mask(agent_df['timestamp'].values, max_time_diff_min=60)
        return agent_df
    
    def convert_df_to_deg(self, df):
        for key in df.keys():
            tokens = key.split('_')
            has_substr = any(feat in tokens for feat in self._angle_features)
            is_exact = any(feat == key for feat in self._angle_features)
            ends_with_circ = any(key.endswith(circ_suffix) for circ_suffix in ['sin', 'cos'])
            if (has_substr or is_exact) and not ends_with_circ:
                print(f"Converting {key} to degrees")
                df[key] /= units.deg
        return df

    def _get_bin_coords(self, bin_idxs, timestamps):
        if self.is_azel:
            ra_arr, dec_arr = np.zeros(len(bin_idxs)), np.zeros(len(bin_idxs))
            az_arr = np.array([self.hpGrid.lon[bid] for bid in bin_idxs])
            el_arr = np.array([self.hpGrid.lat[bid] for bid in bin_idxs])
            for i, (time, (az, el)) in enumerate(zip(timestamps, zip(az_arr, el_arr))):
                ra_arr[i], dec_arr[i] = ephemerides.topographic_to_equatorial(az=az, el=el, time=time)
        else:
            ra_arr = np.array([self.hpGrid.lon[bid] for bid in bin_idxs])
            dec_arr = np.array([self.hpGrid.lat[bid] for bid in bin_idxs])
            az_arr, el_arr = np.zeros(len(bin_idxs)), np.zeros(len(bin_idxs))
            for i, (time, bid) in enumerate(zip(timestamps, bin_idxs)):
                az_arr[i], el_arr[i] = ephemerides.equatorial_to_topographic(ra=self.hpGrid.lon[bid], dec=self.hpGrid.lat[bid], time=time)
        return az_arr, el_arr, ra_arr, dec_arr

    def _get_field_coords(self, field_ids, timestamps):
        ra_arr = np.array([self.lookup.field2radec[fid][0] for fid in field_ids])
        dec_arr = np.array([self.lookup.field2radec[fid][1] for fid in field_ids])
        az_arr, el_arr = np.zeros(len(field_ids)), np.zeros(len(field_ids))
        for i, (time, (ra, dec)) in enumerate(zip(timestamps, zip(ra_arr, dec_arr))):
            az_arr[i], el_arr[i] = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=time)
        return ra_arr, dec_arr, az_arr, el_arr
    
    def _get_expert_idx_segments(self):
        diffs = np.diff(self.dataset.current_state_idxs)
        break_indices = np.where(diffs > 1)[0] + 1
        segmented_expert_idxs = np.split(self.dataset.next_state_idxs, break_indices)
        return segmented_expert_idxs
    
    def _get_average_bin_properties_for_chosen_bins(self):
        pass
    
    def _check_key_exists(self, df, feature):
        if feature not in df.keys():
            raise ValueError
    
class Evaluator:
    def __init__(self, policy, data_container, plotter, device):
        self.policy = policy
        self.data = data_container
        self.plotter = plotter
        self.device = device
        
    def run_saliency_check(self):
        idx = np.random.randint(low=0, high=len(self.data.dataset.curr_compact_idxs))
        compact_idx = self.data.dataset.curr_compact_idxs[idx]
        
        x_glob, x_bin = self.data.dataset.states[compact_idx], self.data.dataset.bin_states[compact_idx]
        
        x_glob = x_glob.unsqueeze(0).to(self.device)
        x_bin = x_bin.unsqueeze(0).to(self.device)
        self.policy.core_net.to(self.device)
        
        x_glob.requires_grad_(True); x_bin.requires_grad_(True)
        
        scores = self.policy.core_net(x_glob, x_bin)
        target_score = scores[0].max()

        self.policy.core_net.zero_grad()
        target_score.backward()
        
        bin_feature_grads = x_bin.grad[0].abs().mean(dim=0)
        for i, name in enumerate(self.data.dataset.bin_feature_names):
            print(f"Feature: {name:30} | Gradient: {bin_feature_grads[i].item()/bin_feature_grads.max().item():.6f}")

    def plot_weights(self):
        fig, ax = plt.subplots(figsize=(20, 5))
        for i, feat in enumerate(self.data.dataset.global_feature_names + self.data.dataset.bin_feature_names):
            ax.errorbar(
                x=[i],
                y=self.policy.core_net.net[0].weight.data[:, i].mean(axis=0).cpu().detach().numpy(), 
                yerr=self.policy.core_net.net[0].weight.data[:, i].std(axis=0).cpu().detach().numpy(),
                color='black'
            )
            ax.scatter(
                x=[i],
                y=self.policy.core_net.net[0].weight.data[:, i].mean(axis=0).cpu().detach().numpy(), 
                color='C0'
            )
        ax.set_xticks(ticks=np.arange(len(self.data.dataset.global_feature_names + self.data.dataset.bin_feature_names)), labels=self.data.dataset.global_feature_names + self.data.dataset.bin_feature_names, rotation=45);
        return fig, ax
    
    def calculate_filter_confusion(self):
        conf_mat = np.zeros(shape=(len(FILTER2IDX), len(FILTER2IDX)))
        for filt, idx in FILTER2IDX.items():
            m = self.data.expert_df['filter_idx'].values == idx
            _ag_filt = self.data.agent_df['filter'].values[m]
            _exp_filt = self.data.expert_df['filter'].values[m]
            if len(_exp_filt) > 0:
                for _fname, _fidx in FILTER2IDX.items():
                    frac = sum(_ag_filt == _fname) / len(_exp_filt)
                    conf_mat[idx, _fidx] = frac
            else:
                frac = 0.0
        return conf_mat

    def plot_filter_confusion(self):
        conf_mat = self.calculate_filter_confusion()
        self.plotter.plot_filter_confusion(conf_mat)

    def plot_mollweide_res(self):
        timestamps = self.data.agent_df['timestamp'].values
        expert_bin_idxs = self.data.expert_df['bin_idx']
        agent_bin_idxs = self.data.agent_df['bin_idx']
        field_pos = np.array([self.data.lookup.field2radec[fid] for fid in np.arange(len(self.data.lookup.field2radec))])
        nside = self.data.hpGrid.nside
        self.plotter.plot_mollweide_res(timestamps, expert_bin_idxs, agent_bin_idxs, field_pos, nside)
        
    def plot_hist_comparison(self, feature_name, density=False, bins=20, use_weights=False):
        self.plotter.plot_hist_comparison(feature_name, expert_arr=self.data.expert_df[feature_name], agent_arr=self.data.agent_df[feature_name], 
                                          density=density, bins=bins, use_weights=use_weights)
        
class SingleStepEvaluator(Evaluator):
    def __init__(self, policy: torch.nn, data_container: DataContainer, plotter: EvaluationPlotter, device: str = 'cuda'):
        super().__init__(policy, data_container, plotter, device)
    
    def run(self):
        # RUN SINGLE STEP
        agent_bin_idxs, agent_filter_idxs = self._batch_single_step_validation()
        # POPULATE DATAFRAMES
        timestamps = self.data.expert_df['timestamp']
        self.data.populate_agent_df(agent_bin_idxs, agent_filter_idxs, timestamps)
        self.data.populate_errors_df()
        # CONVERT RAD TO DEG FOR EACH DF
        self.data.agent_df = self.data.convert_df_to_deg(self.data.agent_df)
        self.data.errors_df = self.data.convert_df_to_deg(self.data.errors_df)
        
    def _batch_single_step_validation(self):
        # in case memory limit, go in slices of data
        n_slices = 8
        best_actions = []
        dur = len(self.data.dataset.curr_compact_idxs) // 8
        for i in range(8):
            with torch.no_grad():
                cur_slice = slice(i * dur, (i+1) * dur)
                if i == n_slices-1:
                    cur_slice = slice(i * dur, None)
                best_actions.append(self.policy.select_action(self.data.dataset.states[self.data.dataset.curr_compact_idxs[cur_slice]].to(self.device),
                                                                    self.data.dataset.bin_states[self.data.dataset.curr_compact_idxs[cur_slice]].to(self.device), 
                                                                    self.data.dataset.action_masks[self.data.dataset.curr_compact_idxs[cur_slice]].to(self.device))
                                    )

        bin_idxs = torch.cat(best_actions).to('cpu').detach().numpy()
        if 'filter' in self.data.action_space:
            filter_idxs = bin_idxs % NUM_FILTERS
            bin_idxs = bin_idxs // NUM_FILTERS
            return bin_idxs, filter_idxs
        return bin_idxs, None

    def plot_2dhist(self, feature_x: str, feature_y: str, bins=25):
        return self.plotter.plot_2dhist(
            feature_x=feature_x,
            feature_y=feature_y,
            expert_x=self.data.expert_df[feature_x].values,
            expert_y=self.data.expert_df[feature_y].values,
            agent_x=self.data.agent_df[feature_x].values,
            agent_y=self.data.agent_df[feature_y].values,
            bins=bins
        )

    def plot_2dhist_res(self, feature_x, feature_y, bins=25):
        return self.plotter.plot_2dhist_res(
            feature_x=feature_x,
            feature_y=feature_y,
            expert_x=self.data.expert_df[feature_x],
            expert_y=self.data.expert_df[feature_y],
            agent_x=self.data.agent_df[feature_x],
            agent_y=self.data.agent_df[feature_y],
            bins=bins
        )

    def plot_line_comparison(self, feature_name):
        self.plotter.plot_line_comparison(feature_name=feature_name, expert_arr=self.data.expert_df[feature_name].values, agent_arr=self.data.agent_df[feature_name])
    
    def plot_scatter_comparison(self, feature_y, feature_x=None):
        expert_y = self.data.expert_df[feature_y].values
        agent_y = self.data.agent_df[feature_y].values
        if feature_x is None:
            expert_x = np.arange(len(expert_y))
            agent_x = np.copy(expert_x)
        else:
            expert_x = self.data.expert_df[feature_x].values
            agent_x = self.data.agent_df[feature_x].values
        self.plotter.plot_scatter_comparison(
            feature_x=feature_x,
            feature_y=feature_y,
            expert_x=expert_x,
            expert_y=expert_y,
            agent_x=agent_x,
            agent_y=agent_y
            )
    
    def plot_residual(self, feature_y, feature_x=None, plot_type='hist', bins=20, alpha=.2, density=False):
        expert_y = self.data.expert_df[feature_y]
        expert_x = self.data.expert_df.get(feature_x, None)
        agent_y = self.data.agent_df[feature_y]
        agent_x = self.data.agent_df.get(feature_x, None)
        self.plotter.plot_residual(feature_y, expert_y, agent_y, feature_x, expert_x, agent_x, plot_type, bins, alpha, density)
    
    def plot_filter_bin_cdf(self):
        self.plotter.plot_filter_bin_cdf(self.data.expert_df, self.data.errors_df)
        
    def plot_quiver(self, feature_x, feature_y):
        expert_prev_x = self.data.expert_df[feature_x]
        expert_prev_y = self.data.expert_df[feature_y]
        expert_next_x = self.data.prev_expert_df[feature_x]
        expert_next_y = self.data.prev_expert_df[feature_y]
        agent_next_x = self.data.agent_df[feature_x]
        agent_next_y = self.data.agent_df[feature_x]
        self.plotter.plot_quiver(feature_x, feature_y, expert_prev_x, expert_prev_y, expert_next_x, expert_next_y, agent_next_x, agent_next_y)
        
    def plot_filter_histograms(self, feature_name, bins=20):
        feature_arr = self.data.expert_df[feature_name].values
        expert_filters = self.data.expert_df['filter'].values
        agent_filters = self.data.agent_df['filter'].values
        self.plotter.plot_filter_histograms(feature_name, feature_arr, expert_filters, agent_filters, bins=bins)
        
class MultiStepEvaluator(Evaluator):
    def __init__(self, trainer, policy: torch.nn, data_container: DataContainer, plotter: EvaluationPlotter, device='cuda'):
        super().__init__(policy, data_container, plotter, device)
        self.trainer = trainer

    def run(self, env, cfg, field_choice_method, eval_outdir, lookups):
        self.eval_outdir = Path(eval_outdir)
        if os.path.exists(self.eval_outdir / "eval_metrics.pkl"):
            with open(self.eval_outdir / "eval_metrics.pkl", 'rb') as f:
                self.eval_metrics = pickle.load(f)
        else:
            self.eval_metrics = self.trainer.evaluate(env=env, cfg=cfg, num_episodes=1, field_choice_method=field_choice_method, eval_outdir=eval_outdir, lookups=lookups)
        timestamps, bin_idxs, filter_idxs, field_ids, glob_df, bin_feat_dict = self._process_eval_metrics(self.eval_metrics, eval_outdir)
        self.data.populate_agent_df(bin_idxs=bin_idxs, filter_idxs=filter_idxs, timestamps=timestamps, field_ids=field_ids, glob_df=glob_df, bin_feat_dict=bin_feat_dict)
        self.data.convert_df_to_deg(self.data.agent_df)

    def _process_eval_metrics(self, eval_metrics, eval_outdir):
        with open(Path(eval_outdir) / "eval_metrics.pkl", 'rb') as f:
            eval_metrics = pickle.load(f)
        # GET EPISODE (bc deterministic)
        eval_metrics = eval_metrics['ep-0']
        # GET BIN, FIELD, AND FILTERS
        eval_bin_df = pd.concat(
            [pd.DataFrame(eval_metrics[night]['bin'], columns=['bin']).assign(night=night)
            for night in eval_metrics],
            ignore_index=True
        )
        eval_field_df = pd.concat(
            [pd.DataFrame(eval_metrics[night]['field_id'], columns=['field_id']).assign(night=night)
            for night in eval_metrics],
            ignore_index=True
        )
        eval_timestamp_df = pd.concat(
            [pd.DataFrame(eval_metrics[night]['timestamp'], columns=['timestamp']).assign(night=night)
            for night in eval_metrics],
            ignore_index=True
        )
        eval_filter_idx_df = pd.concat(
            [pd.DataFrame(eval_metrics[night]['filter_idx'], columns=['filter_idx']).assign(night=night)
            for night in eval_metrics],
            ignore_index=True
        )
        # REMOVE ZENITH AND WAIT STATES
        valid_mask = ((eval_bin_df['bin'] != -1) & (eval_bin_df['bin'] != -2)).values
        glob_df = pd.concat(
            [pd.DataFrame(eval_metrics[night]['glob_observations'], columns=self.data.dataset.global_feature_names).assign(night=night) 
            for night in eval_metrics], 
            ignore_index=True
        )
        bin_feats = [rearrange(eval_metrics[night]['bin_observations'], "nrows nbins nfeats -> nfeats nrows nbins") for night in eval_metrics]
        bin_feats = np.concat(bin_feats, axis=1)
        
        bins_feats = {key: feat_row[valid_mask] for feat_row, key in zip(bin_feats, self.data.dataset.bin_feature_names)}
        
        agent_schedule = {
            'timestamp': eval_timestamp_df['timestamp'].values[valid_mask],
            'field_id': eval_field_df['field_id'].values[valid_mask],
            'bin_idx': eval_bin_df['bin'].values[valid_mask],
            'filter_idx': eval_filter_idx_df['filter_idx'].values[valid_mask],
        }
        return agent_schedule['timestamp'], agent_schedule['bin_idx'], agent_schedule['filter_idx'], agent_schedule['field_id'],\
            glob_df[valid_mask], bins_feats
            
    def plot_kde_and_scatter(self, feature_x, feature_y, agent_alpha=.2, use_filter_coloring=True, s=5):
        mapped_colors = self.data.agent_df['filter'].map(FILTER_COLORS)
        expert_x = self.data.expert_df[feature_x]
        expert_y = self.data.expert_df[feature_y]
        agent_x = self.data.agent_df[feature_x]
        agent_y = self.data.agent_df[feature_y]
        if use_filter_coloring:
            self.plotter.plot_kde_and_scatter(feature_x, feature_y, expert_x, expert_y, agent_x, agent_y, agent_alpha=agent_alpha, color_mapping=mapped_colors, s=s)
        else:
            self.plotter.plot_kde_and_scatter(feature_x, feature_y, expert_x, expert_y, agent_x, agent_y, agent_alpha=agent_alpha, s=s)
    
    def _calculate_performance_metrics():
        pass
    
    def plot_filter_histograms(self, feature_name, bins=20):
        expert_feature_arr = self.data.expert_df[feature_name].values
        agent_feature_arr = self.data.agent_df[feature_name].values
        expert_filters = self.data.expert_df['filter'].values
        agent_filters = self.data.agent_df['filter'].values
        self.plotter.plot_filter_histograms(feature_name=feature_name,
                                            expert_feature_arr=expert_feature_arr,
                                            agent_feature_arr=agent_feature_arr,
                                            expert_filters=expert_filters,
                                            agent_filters=agent_filters,
                                            bins=bins
                                            )
        
    def plot_hist_comparison(self, feature_name, density=False, bins=20, use_weights=False):
        self.plotter.plot_hist_comparison(feature_name, expert_arr=self.data.expert_df[feature_name][self.data.expert_valid_mask], agent_arr=self.data.agent_df[feature_name], 
                                          density=density, bins=bins, use_weights=use_weights)
        