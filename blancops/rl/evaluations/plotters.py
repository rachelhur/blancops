"""Plotting utilities for comparing expert vs agent behavior.

`EvaluationPlotter` methods accept an optional `ax` and return the matplotlib
artist(s) they create, so plots are composable into subplot grids.

All style choices live in `PlotStyle`. Pass one instance everywhere instead of
threading six color/cmap arguments through every constructor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from blancops.configs.constants import FILTER2IDX
from blancops.plotting.plotting import plot_schedule_whole


import logging
logger = logging.getLogger(__name__)

FILTER_COLORS = {
    'g': '#00b25d',  # Green
    'r': '#f97306',  # Orange
    'i': '#e50000',  # Red
    'z': '#8b0000',  # Dark Red
    'Y': '#3b0000',  # Near-IR (Darkest)
}
FILTER_PATCHES = [Patch(color=v, label=k) for k, v in FILTER_COLORS.items()]


@dataclass(frozen=True)
class PlotStyle:
    agent_color: str = 'green'
    expert_color: str = 'purple'
    agent_cmap: str = 'Greens'
    expert_cmap: str = 'Purples'
    res_cmap: str = 'PRGn'
    res_color: str = 'slateblue'


def _wrapped_ra(ra):
    return (ra + 180) % 360 - 180


def _wrap_if_ra(feature_name: Optional[str], arr):
    """Wrap RA-like features to [-180, 180]. Leaves everything else untouched."""
    if feature_name is None or arr is None:
        return arr
    if feature_name == 'ra' or '_ra' in feature_name:
        return _wrapped_ra(arr)
    return arr


class EvaluationPlotter:
    def __init__(self, outdir, style: Optional[PlotStyle] = None):
        self.outdir = Path(outdir)
        self.style = style or PlotStyle()

    # ------------------------------------------------------------------
    # 2D histograms
    # ------------------------------------------------------------------

    def plot_2dhist(self, feature_x, feature_y, expert_x, expert_y, agent_x, agent_y,
                    norm=None, bins=25, return_plt_objects=False, density=True):
        if norm is None:
            norm = mcolors.LogNorm()
        if density==False:
            cbar_label = 'Counts'
        else:
            cbar_label = 'Density'
        expert_x = _wrap_if_ra(feature_x, expert_x)
        expert_y = _wrap_if_ra(feature_y, expert_y)
        agent_x  = _wrap_if_ra(feature_x, agent_x)
        agent_y  = _wrap_if_ra(feature_y, agent_y)
        
        x_min = min(np.min(expert_x), np.min(agent_x))
        x_max = max(np.max(expert_x), np.max(agent_x))
        y_min = min(np.min(expert_y), np.min(agent_y))
        y_max = max(np.max(expert_y), np.max(agent_y))
        
                
        x_edges = np.linspace(x_min, x_max, bins + 1)
        y_edges = np.linspace(y_min, y_max, bins + 1)
        
        bins=[x_edges, y_edges]

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
        exp_counts, _, _, im1 = axs[0].hist2d(expert_x, expert_y, bins=bins,
                                              cmap=self.style.expert_cmap, norm=norm, density=density)
        fig.colorbar(im1, ax=axs[0], location='right', label=cbar_label)
        axs[0].set(xlabel=feature_x, ylabel=feature_y, title='Expert')

        ag_counts, _, _, im2 = axs[1].hist2d(agent_x, agent_y, bins=bins,
                                             cmap=self.style.agent_cmap, norm=norm, density=density)
        fig.colorbar(im2, ax=axs[1], location='right', label=cbar_label)
        axs[1].set(xlabel=feature_x, ylabel=feature_y, title='Agent')

        if return_plt_objects:
            return fig, axs, exp_counts, ag_counts
        return fig, axs

    def plot_2dhist_res(self, feature_x, feature_y, expert_x, expert_y, agent_x, agent_y,
                        bins=25, return_plt_objects=False, density=False, ax=None):
        expert_x = _wrap_if_ra(feature_x, expert_x)
        expert_y = _wrap_if_ra(feature_y, expert_y)
        agent_x  = _wrap_if_ra(feature_x, agent_x)
        agent_y  = _wrap_if_ra(feature_y, agent_y)

        xmin = min(expert_x.min(), agent_x.min())
        xmax = max(expert_x.max(), agent_x.max())
        ymin = min(expert_y.min(), agent_y.min())
        ymax = max(expert_y.max(), agent_y.max())
        hrange = [[xmin, xmax], [ymin, ymax]]
        nbins = [bins, bins]

        exp_hist, _, _ = np.histogram2d(expert_x, expert_y, bins=nbins, range=hrange, density=density)
        agent_hist, _, _ = np.histogram2d(agent_x, agent_y, bins=nbins, range=hrange, density=density)
        res = agent_hist - exp_hist
        lim = np.max(np.abs(res))

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure
        im = ax.imshow(res.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
                       cmap=self.style.res_cmap, aspect='auto', vmin=-lim, vmax=lim)
        ax.set(xlabel=feature_x, ylabel=feature_y)
        fig.colorbar(im, ax=ax, label='Residual counts\n(agent - expert)')

        if return_plt_objects:
            return fig, ax, exp_hist, agent_hist
        return fig, ax

    # ------------------------------------------------------------------
    # Mollweide / line / scatter / hist / residual
    # ------------------------------------------------------------------

    def plot_mollweide_res(self, timestamps, expert_bin_idxs, agent_bin_idxs, field_pos, nside):
        plot_schedule_whole(
            outfile=self.outdir / 'mollweide_residuals',
            times=timestamps,
            field_pos=None,
            bin_idxs=expert_bin_idxs,
            alternate_bin_idxs=agent_bin_idxs,
            nside=nside,
            sky_bin_mapping=None,
            projection='mollweide',
            center_pos=(None, None),
            schedule_label='',
        )

    def plot_line_comparison(self, feature_name, expert_arr, agent_arr, ax=None):
        ax = ax or plt.gca()
        expert_arr = _wrap_if_ra(feature_name, expert_arr)
        agent_arr  = _wrap_if_ra(feature_name, agent_arr)
        ax.plot(expert_arr, label='expert', color=self.style.expert_color)
        ax.plot(agent_arr, label='agent', color=self.style.agent_color)
        ax.set_xlabel('Time', fontsize=16)
        ax.set_ylabel(feature_name, fontsize=16)
        ax.legend(fontsize=16)
        return ax

    def plot_scatter_comparison(self, feature_y, expert_x, expert_y, agent_x, agent_y,
                                feature_x=None, ax=None):
        ax = ax or plt.gca()
        ax.scatter(expert_x, expert_y, label='expert', color=self.style.expert_color)
        ax.scatter(agent_x,  agent_y,  label='agent',  color=self.style.agent_color)
        if feature_x is not None:
            ax.set_xlabel(feature_x, fontsize=16)
        ax.set_ylabel(feature_y, fontsize=16)
        ax.legend(fontsize=16)
        return ax

    def plot_hist_comparison(self, feature_name, expert_arr, agent_arr,
                             density=False, bins=20, alpha=0.2, use_weights=False, ax=None):
        ax = ax or plt.gca()
        if isinstance(bins, int) and feature_name != 'filter':
            lo = min(np.min(expert_arr), np.min(agent_arr))
            hi = max(np.max(expert_arr), np.max(agent_arr))
            shared_bins = np.linspace(lo, hi, bins + 1)
        else:
            shared_bins = bins

        ag_weights  = np.ones_like(agent_arr)  / len(agent_arr)  if use_weights else None
        exp_weights = np.ones_like(expert_arr) / len(expert_arr) if use_weights else None
        if use_weights:
            density = False

        # Outlines
        ax.hist(expert_arr, bins=shared_bins, density=density, histtype='step',
                color=self.style.expert_color, weights=exp_weights)
        ax.hist(agent_arr,  bins=shared_bins, density=density, histtype='step',
                color=self.style.agent_color, weights=ag_weights)
        # Fills
        ax.hist(expert_arr, bins=shared_bins, label='expert', histtype='stepfilled',
                color=self.style.expert_color, alpha=alpha, density=density, weights=exp_weights)
        ax.hist(agent_arr,  bins=shared_bins, label='agent',  histtype='stepfilled',
                color=self.style.agent_color,  alpha=alpha, density=density, weights=ag_weights)

        ax.set_xlabel(feature_name, fontsize=16)
        ylabel = 'Probability Density' if density else ('Fraction of total' if use_weights else 'Raw counts')
        ax.set_ylabel(ylabel, fontsize=16)
        ax.legend(fontsize=16)
        return ax

    def plot_residual(self, feature_y, expert_y, agent_y, feature_x=None,
                      expert_x=None, agent_x=None, plot_type='hist',
                      bins=20, alpha=0.2, density=False, ax=None):
        assert plot_type in ('hist', 'line')
        ax = ax or plt.gca()

        expert_x = _wrap_if_ra(feature_x, expert_x)
        expert_y = _wrap_if_ra(feature_y, expert_y)
        agent_x  = _wrap_if_ra(feature_x, agent_x)
        agent_y  = _wrap_if_ra(feature_y, agent_y)

        res_y = np.asarray(agent_y) - np.asarray(expert_y)
        if expert_x is None:
            expert_x = np.arange(len(res_y))

        if plot_type == 'hist':
            ax.hist(res_y, bins=bins, density=density, histtype='step',       color=self.style.res_color)
            ax.hist(res_y, bins=bins, density=density, histtype='stepfilled', color=self.style.res_color, alpha=alpha)
            ax.set_xlabel(feature_y)
            ax.set_ylabel('Normalized (residual) counts' if density else 'Raw (residual) counts', fontsize=16)
        else:  # line
            ax.plot(expert_x, res_y, color=self.style.res_color, marker='o')
        return ax

    # ------------------------------------------------------------------
    # Filter-specific plots
    # ------------------------------------------------------------------

    def plot_filter_confusion(self, conf_mat, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.heatmap(conf_mat, annot=True, cmap=self.style.expert_cmap,
                    xticklabels=FILTER2IDX.keys(), yticklabels=FILTER2IDX.keys(), ax=ax)
        ax.set_xlabel('Expert')
        ax.set_ylabel('Agent')
        ax.set_title('Filter')
        return ax

    def plot_filter_bin_cdf(self, expert_df, errors_df, tolerance_deg=5.0):
        fig, ax = plt.subplots(figsize=(5, 4))
        max_x = 0.0
        for i, filt in enumerate(FILTER2IDX.keys()):
            mask = (expert_df['filter'] == filt).values
            sorted_errors = np.sort(errors_df['bin_angular_separation'][mask])
            if len(sorted_errors) == 0:
                continue
            y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            f_color = FILTER_COLORS[filt]

            ax.plot(sorted_errors, y, lw=4, alpha=0.7, color=f'C{i}')

            pct = np.searchsorted(sorted_errors, tolerance_deg) / len(sorted_errors)
            ax.axhline(pct, color=f_color, linestyle='dotted', lw=1.5,
                       label=f'{filt}: {pct*100:.1f}% within {tolerance_deg}°')
            max_x = max(max_x, sorted_errors.max())

        ax.axvline(tolerance_deg, color='red', lw=1.5)
        ax.set(xlabel='Angular Separation (deg)', ylabel='CDF',
               xlim=(0, max_x or 1), ylim=(0, 1.05))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=14)
        return ax

    def plot_hist_comparison_per_filter(self, feature_name, expert_feature_arr, expert_filters,
                               agent_filters, agent_feature_arr=None, density=True,
                               alpha=0.2, bins=20):
        if agent_feature_arr is None:
            agent_feature_arr = expert_feature_arr

        nrows, ncols = 2, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()
        for ax, filt in zip(axes, FILTER2IDX.keys()):
            exp_m = expert_filters == filt
            ag_m  = agent_filters  == filt
            ax.hist(agent_feature_arr[ag_m],  bins=bins, density=density,
                    color=self.style.agent_color,  histtype='step', lw=1.5)
            ax.hist(expert_feature_arr[exp_m], bins=bins, density=density,
                    color=self.style.expert_color, histtype='step', lw=1.5)
            ax.hist(agent_feature_arr[ag_m],  bins=bins, label=f'Agent chose {filt}',
                    histtype='stepfilled', color=self.style.agent_color,  alpha=alpha, density=density)
            ax.hist(expert_feature_arr[exp_m], bins=bins, label=f'Expert chose {filt}',
                    histtype='stepfilled', color=self.style.expert_color, alpha=alpha, density=density)
            ax.set_xlabel(feature_name)
            ax.legend()
        # Hide any unused subplots (only 5 filters but a 2x3 grid).
        for ax in axes[len(FILTER2IDX):]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------
    # KDE + scatter, quiver
    # ------------------------------------------------------------------

    def plot_kde_and_scatter(self, feature_x, feature_y, expert_x, expert_y,
                             agent_x, agent_y, agent_alpha=0.2, color_mapping=None,
                             s=5, ax=None):
        ax = ax or plt.gca()
        expert_x = _wrap_if_ra(feature_x, expert_x)
        expert_y = _wrap_if_ra(feature_y, expert_y)
        agent_x  = _wrap_if_ra(feature_x, agent_x)
        agent_y  = _wrap_if_ra(feature_y, agent_y)

        if color_mapping is None:
            color_mapping = self.style.agent_color
            agent_label = 'agent'
            handles = None
        else:
            agent_label = None
            handles = FILTER_PATCHES

        sns.kdeplot(x=expert_x, y=expert_y, thresh=0, levels=10,
                    cmap=self.style.expert_cmap, alpha=.5, label='expert', fill=False,
                    norm=mcolors.CenteredNorm(), ax=ax)
        ax.scatter(agent_x, agent_y, marker='*', c=color_mapping, label=agent_label,
                   alpha=agent_alpha, s=s)
        ax.set_xlabel(feature_x, fontsize=16)
        ax.set_ylabel(feature_y, fontsize=16)
        ax.legend(handles=handles, fontsize=16)
        return ax

    def plot_quiver(self, feature_x, feature_y, expert_prev_x, expert_prev_y,
                    expert_next_x, expert_next_y, agent_next_x, agent_next_y, ax=None):
        ax = ax or plt.gca()
        expert_prev_x = _wrap_if_ra(feature_x, expert_prev_x)
        expert_prev_y = _wrap_if_ra(feature_y, expert_prev_y)
        expert_next_x = _wrap_if_ra(feature_x, expert_next_x)
        expert_next_y = _wrap_if_ra(feature_y, expert_next_y)
        agent_next_x  = _wrap_if_ra(feature_x, agent_next_x)
        agent_next_y  = _wrap_if_ra(feature_y, agent_next_y)

        expert_u = expert_next_x - expert_prev_x
        expert_v = expert_next_y - expert_prev_y
        agent_u  = agent_next_x  - expert_prev_x
        agent_v  = agent_next_y  - expert_prev_y

        ax.quiver(expert_prev_x, expert_prev_y, expert_u, expert_v,
                  color=self.style.expert_color, label='expert')
        ax.quiver(expert_prev_x, expert_prev_y, agent_u, agent_v,
                  color=self.style.agent_color, label='agent')
        ax.set_xlabel(feature_x, fontsize=14)
        ax.set_ylabel(feature_y, fontsize=14)
        ax.legend()
        return ax
