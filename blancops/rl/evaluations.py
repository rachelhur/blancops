import pickle

import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

from blancops.data.constants import *
from blancops.math.geometry import angular_separation
from blancops.ephemerides import ephemerides

from blancops.math import units
from blancops.ephemerides.ephemerides import get_source_ra_dec


def build_evaluators(policy, val_dataset, action_space, device, agent_color='green', expert_color='purple', agent_cmap='Greens', expert_cmap='Purples', res_cmap='PRGn'):
    ss_data = DataContainer(val_dataset, action_space, eval_method='ss')
    ms_data = DataContainer(val_dataset, action_space, eval_method='ms')
    plotter = EvaluationPlotter(agent_color, expert_color, agent_cmap, expert_cmap, res_cmap)
    s_evaluator = SingleStepEvaluator(policy, ss_data, plotter, device)
    m_evaluator = MultiStepEvaluator(policy, ms_data, plotter, device)
    return s_evaluator, m_evaluator

def _stack_and_validate_radecs(ra, dec):
    radecs = np.stack((ra, dec), axis=1)
    if radecs.max() > 2*np.pi:
        radecs *= units.deg
    return radecs    

def calc_airmass(el):
    if el.max() > np.pi:
        el *= units.deg
    return 1 / np.cos(np.pi/2 - el)
    
def calc_slew_distance(ra, dec, timestamps):
    radecs = _stack_and_validate_radecs(ra, dec)
    slew_dists = np.zeros(shape=(len(timestamps)))
    for i in range(len(timestamps)-1):
        slew_dists[i] = angular_separation(radecs[i], radecs[i+1]) / units.deg
    return slew_dists

def calc_moon_dist(ra, dec, timestamps):
    radecs = _stack_and_validate_radecs(ra, dec)
        
    moon_dists = np.zeros((len(timestamps)))
    for i, t in enumerate(timestamps):
        moon_radec = get_source_ra_dec('moon', time=t)
        moon_dists[i] = angular_separation(moon_radec, radecs[i])
    return moon_dists

class EvaluationPlotter:
    def __init__(self, agent_color='green', expert_color='purple', agent_cmap='Greens', expert_cmap='Purples', res_cmap='PRGn'):
        self.agent_color = agent_color
        self.expert_color = expert_color
        self.agent_cmap = agent_cmap
        self.expert_cmap = expert_cmap
        self.res_cmap = res_cmap

    @staticmethod
    def _get_wrapped_ra(ra):
        return ((ra + np.pi) % 2 * np.pi - np.pi)
        
    def plot_radec_2dhist(self, expert_ra, expert_dec, agent_ra, agent_dec, norm=colors.LogNorm(), bins=25):
        fig, axs = plt.subplots(1, 2, figsize=(14,5), sharex=True, sharey=True)
        expert_ra = self._get_wrapped_ra(expert_ra)
        agent_ra = self._get_wrapped_ra(agent_ra)
        exp_counts, xedges, yedges, im1 = axs[0].hist2d(x=expert_ra/units.deg, y=expert_dec/units.deg, bins=bins, cmap=self.expert_cmap, norm=norm)
        fig.colorbar(im1, ax=axs[0], location="right", label='Counts')
        axs[0].set_xlabel('RA')
        axs[0].set_ylabel('Dec')
        axs[0].set_title('Expert')
        ag_counts, xedges, yedges, im2 = axs[1].hist2d(x=agent_ra/units.deg, y=agent_dec/units.deg, bins=bins, cmap=self.agent_cmap, norm=norm);
        axs[1].set_xlabel('RA')
        axs[1].set_ylabel('Dec')
        axs[1].set_title('BC')
        fig.colorbar(im2, ax=axs[1], location="right", label='Counts')
        return fig, axs, exp_counts, ag_counts
    
    def plot_radec_2dhist_res(self, expert_ra, expert_dec, agent_ra, agent_dec, bins=25):
        expert_ra = self._get_wrapped_ra(expert_ra)
        agent_ra = self._get_wrapped_ra(agent_ra)
        
        bins = [bins, bins]
        hrange = [[-180, 180], [-80, 5]]

        exp_hist, xedges, yedges = np.histogram2d(expert_ra/units.deg, expert_dec/units.deg, bins=bins, range=hrange, density=True)
        agent_hist, _, _ = np.histogram2d(agent_ra/units.deg, agent_dec/units.deg, bins=bins, range=hrange, density=True)

        res = agent_hist - exp_hist
        lim = np.max(np.abs(res)) 
        
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(res.T, origin='lower', extent=[-180, 180, -80, 5], cmap=self.res_cmap, aspect='auto',
                       vmin=-lim, vmax=lim)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        fig.colorbar(im, ax=ax, label="Residual counts \n (agent - expert)")
        return exp_hist, agent_hist

    def plot_1d_comparison(self, feature_name, expert_arr, agent_arr, timestamps, plot_type='hist', **kwargs):
        if plot_type=='hist':
            self._plot_hist(expert_arr, agent_arr)            
        elif plot_type=='line':
            self._plot_line(feature_name, expert_arr, agent_arr, timestamps, **kwargs)
        plt.legend(fontsize=16)
    
    def plot_line(self, feature_name, expert_arr, agent_arr, **kwargs):
        plt.plot(expert_arr, label='expert', color=self.expert_color, **kwargs)
        plt.plot(agent_arr, label='agent', color=self.agent_color, **kwargs)
        plt.xlabel('Time', fontsize=16)
        plt.ylabel(feature_name, fontsize=16)
        
    def plot_hist(self, feature_name, expert_arr, agent_arr, **kwargs):
        plt.hist(expert_arr, bins=20, alpha=.5, label='expert', density=True, histtype='step', color=self.expert_color, **kwargs)
        plt.hist(agent_arr, bins=20, alpha=.5, label='agent', density=True, histtype='step', color=self.agent_color, **kwargs)
        
        plt.hist(expert_arr, bins=30, histtype='stepfilled', color=self.expert_color, alpha=0.1)
        plt.hist(agent_arr, bins=30, histtype='stepfilled', color=self.agent_color, alpha=0.1)

        plt.xlabel(feature_name, fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
    
    def run_saliency_check(dataset, network, device):
        idx = np.random.randint(low=0, high=len(dataset.curr_compact_idxs))
        compact_idx = dataset.curr_compact_idxs[idx]
        np.random.randint(low=0, high=len(dataset))
        x_glob, x_bin = dataset.states[compact_idx], dataset.bin_states[compact_idx]
        x_glob.to(device); x_bin.to(device); network.to(device)
        x_glob.requires_grad_(True)
        x_bin.requires_grad_(True)
        scores = network(x_glob, x_bin)
        target_score = scores[0].max()

        network.zero_grad()
        target_score.backward()
        
        bin_feature_grads = x_bin.grad[0].abs().mean(dim=0)
        for i, name in enumerate(dataset.bin_feature_names):
            print(f"Feature: {name:30} | Gradient: {bin_feature_grads[i].item()/bin_feature_grads.max().item():.6f}")
    
    def plot_weights(dataset, policy):
        fig, ax = plt.subplots(figsize=(20, 5))
        for i, feat in enumerate(dataset.global_feature_names + dataset.bin_feature_names):
            ax.errorbar(
                x=[i],
                y=policy.core_net.net[0].weight.data[:, i].mean(axis=0).cpu().detach().numpy(), 
                yerr=policy.core_net.net[0].weight.data[:, i].std(axis=0).cpu().detach().numpy(),
                color='black'
            )
            ax.scatter(
                x=[i],
                y=policy.core_net.net[0].weight.data[:, i].mean(axis=0).cpu().detach().numpy(), 
                color='C0'
            )
        ax.set_xticks(ticks=np.arange(len(dataset.global_feature_names + dataset.bin_feature_names)), labels=dataset.global_feature_names + dataset.bin_feature_names, rotation=45);
        return fig, ax
    
class DataContainer():
    def __init__(self, val_dataset, action_space, eval_method='ss'):
        assert eval_method in ['ss', 'ms']
        self.eval_method = eval_method
        self.hpGrid = val_dataset.hpGrid
        self.is_azel = val_dataset.hpGrid.is_azel
        self.action_space = action_space
        self.dataset = val_dataset
        self._angle_features = ['ra', 'dec', 'az', 'el']
        self._angle_features += ['_ra', '_dec', '_az', '_el']
        self._populate_expert_df()
        
    def populate_agent_df(self, bin_idxs, filter_idxs, timestamps):
        if self.eval_method == 'ss':
            self.agent_df = self._get_ss_agent_df(bin_idxs, filter_idxs, timestamps)
        elif self.eval_method == 'ms':
            self.agent_df = self._get_ms_agent_df(bin_idxs, filter_idxs, timestamps)
            
    def _get_ss_agent_df(self, bin_idxs, filter_idxs, timestamps):
        agent_df = pd.DataFrame()
        agent_df['bin_az'], agent_df['bin_el'], agent_df['bin_ra'], agent_df['bin_dec'] = self._get_bin_coords(bin_idxs, timestamps=timestamps)
        agent_df['filter_idx'] = filter_idxs
        agent_df['filter'] = agent_df['filter_idx']
        agent_df['az'], agent_df['el'], agent_df['ra'], agent_df['dec'] = None, None, None, None
        
        agent_df['airmass'] = self.expert_df['airmass']
        agent_df['moon_phase'] = self.expert_df['moon_phase']
        agent_df['fwhm'] = self.expert_df['fwhm']
        agent_df['sun_az'] = self.expert_df['sun_az']
        agent_df['sun_el'] = self.expert_df['sun_el']
        agent_df['moon_el'] = self.expert_df['moon_el']
        agent_df['sun_az'] = self.expert_df['sun_az']
    
    def _get_ms_agent_df(bin_idxs, filter_idxs, timestamps):
        pass
    
    def _populate_expert_df(self):
        if self.eval_method == 'ss':
            self.expert_df = self._get_ss_expert_df()
        elif self.eval_method == 'ms':
            self.expert_df = self._get_ms_expert_df()
        
    def _get_ss_expert_df(self):
        expert_df = pd.DataFrame()
        expert_df = self._extract_expert_data(expert_df, desired_indices=self.dataset.next_state_idxs)
        expert_df = self._extract_expert_data(expert_df, desired_indices=self.dataset.current_state_idxs, key_pref='prev_')
        return expert_df
    
    def _get_ms_expert_df(self):
        expert_df = pd.DataFrame()
        expert_df = self._extract_expert_data(expert_df, desired_indices=self.dataset.state_idxs)
        return expert_df
    
    def _extract_expert_data(self, expert_df, desired_indices, key_pref=''):
        filtered_df =  self.dataset._df.iloc[desired_indices]
        
        if self.eval_method == 'ss':
            expert_df[key_pref+'bin_idx'] = self.dataset.actions.detach().numpy() // NUM_FILTERS
            expert_df[key_pref+'filter_idx'] = self.dataset.actions.detach().numpy() % NUM_FILTERS
        else:
            expert_df[key_pref+'bin_idx'] = filtered_df['bin'].values
            expert_df[key_pref+'filter_idx'] = filtered_df['filt_idx'].values

        expert_df[key_pref+'timestamp'] = filtered_df['timestamp'].values
        expert_df[key_pref+'filter'] = expert_df['filter_idx'].map(IDX2FILTER).fillna(-1)
        expert_df[key_pref+'bin_az'], expert_df[key_pref+'bin_el'], expert_df[key_pref+'bin_ra'], expert_df[key_pref+'bin_dec'] = \
            self._get_bin_coords(expert_df['bin_idx'].values, timestamps=expert_df[key_pref+'timestamp'].values)
        expert_df[key_pref+'ra'], expert_df[key_pref+'dec'] = np.array([filtered_df.ra.values, filtered_df.dec.values])
        expert_df[key_pref+'az'], expert_df[key_pref+'el'] = np.array([filtered_df.az.values, filtered_df.el.values])
        
        # environmental params
        expert_df[key_pref+'airmass'] = filtered_df.airmass.values
        expert_df[key_pref+'moon_phase'] = filtered_df.get('moon_phase', None)
        expert_df[key_pref+'fwhm'] = filtered_df.get('fwhm', None)
        expert_df[key_pref+'sun_az'] = filtered_df.get('sun_az', None) / units.deg
        expert_df[key_pref+'sun_el'] = filtered_df.get('sun_el', None) / units.deg
        expert_df[key_pref+'moon_el'] = filtered_df.get('moon_el', None) / units.deg
        expert_df[key_pref+'sun_az'] = filtered_df.get('moon_az', None) / units.deg
        return expert_df
    
    def _get_global_feature_dict(self):
        curr_state_df = self.dataset._df.iloc[self.dataset.current_state_idxs]
        input_state_properties = {}
        for key in self.dataset.global_feature_names:
            input_state_properties[key] = curr_state_df.get(key, None).values
            if any(angle_feat in key for angle_feat in self._angle_features) and not any(key.endswith(circ_suffix) for circ_suffix in ['sin', 'cos']):
                input_state_properties[key] /= units.deg
        return input_state_properties
    
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
    
    def _get_full_state_properties(self):
        full_state_properties = {}
        for key in self.dataset.global_feature_names:
            full_state_properties[key] = self.dataset._df.get(key, None).values
        return full_state_properties
    
    def _get_average_bin_properties_for_chosen_bins(self):
        pass

class SingleStepEvaluator:
    def __init__(self, agent, data_container: DataContainer, plotter: EvaluationPlotter, device='cuda'):
        super().__init__()
        self.agent = agent
        self.data = data_container
        self.plotter = plotter
        self.device = device
    
    def run(self):
        self.data.agent
        agent_bin_idxs, agent_filter_idxs = self._batch_single_step_validation()
        timestamps = self.data.expert_df['timestamp']
        self.data.populate_agent_data(agent_bin_idxs, agent_filter_idxs, timestamps)
        
    def _batch_single_step_validation(self):
        # in case memory limit, go in slices of data
        n_slices = 8
        best_actions = []
        dur = len(self.dataset.curr_compact_idxs) // 8
        for i in range(8):
            with torch.no_grad():
                cur_slice = slice(i * dur, (i+1) * dur)
                if i == n_slices-1:
                    cur_slice = slice(i * dur, None)
                best_actions.append(self.agent.choose_action(self.dataset.states[self.dataset.curr_compact_idxs[cur_slice]].to(self.device),
                                                                    self.dataset.bin_states[self.dataset.curr_compact_idxs[cur_slice]].to(self.device), 
                                                                    self.dataset.action_masks[self.dataset.curr_compact_idxs[cur_slice]].to(self.device))
                                    )

        bin_idxs = torch.cat(best_actions).to('cpu').detach().numpy()
        if 'filter' in self.action_space:
            filter_idxs = bin_idxs % NUM_FILTERS
            bin_idxs = bin_idxs // NUM_FILTERS
            return bin_idxs, filter_idxs
        return bin_idxs, None

    def plot_radec_2dhist(self):
        return self.plotter._plot_radec_2dhist(self.expert_ra, self.expert_dec, self.ra, self.dec)

    def plot_radec_2dhist_res(self):
        return self._plot_radec_2dhist_res(self.expert_ra, self.expert_dec, self.ra, self.dec)

    def plot_comparison(self, feature_name, expert_arr, agent_arr, timestamps, plot_type='hist', **kwargs):
        if plot_type=='hist':
            self._plot_hist(expert_arr, agent_arr)            
        elif plot_type=='line':
            self._plot_line(feature_name, expert_arr, agent_arr, timestamps, **kwargs)
        plt.legend(fontsize=16)
    
    def plot_line(self, feature_name, expert_arr, agent_arr, timestamps, **kwargs):
        self._plot_line(feature_name, expert_arr, agent_arr, timestamps, **kwargs)
        
    def plot_hist(self, feature_name, expert_arr, agent_arr, **kwargs):
        plt.hist(expert_arr, bins=20, alpha=.5, label='expert', density=True, histtype='step', color=self.expert_color, **kwargs)
        plt.hist(agent_arr, bins=20, alpha=.5, label='agent', density=True, histtype='step', color=self.agent_color, **kwargs)
        
        # Optional: Add a very faint fill under one or both for depth
        plt.hist(expert_arr, bins=30, histtype='stepfilled', color=self.expert_color, alpha=0.1)
        plt.hist(agent_arr, bins=30, histtype='stepfilled', color=self.agent_color, alpha=0.1)

        plt.xlabel(feature_name, fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
    
    def run_saliency_check(self):
        self._run_saliency_check(self.dataset, self.network, self.device)
    
    def plot_weights(self):
        fig, ax = plt.subplots(figsize=(20, 5))
        for i, feat in enumerate(self.dataset.global_feature_names + self.dataset.bin_feature_names):
            ax.errorbar(
                x=[i],
                y=self.agent.algorithm.policy.core_net.net[0].weight.data[:, i].mean(axis=0).cpu().detach().numpy(), 
                yerr=self.agent.algorithm.core_net.net[0].weight.data[:, i].std(axis=0).cpu().detach().numpy(),
                color='black'
            )
            ax.scatter(
                x=[i],
                y=self.policy.core_net.net[0].weight.data[:, i].mean(axis=0).cpu().detach().numpy(), 
                color='C0'
            )
        ax.set_xticks(ticks=np.arange(len(self.dataset.global_feature_names + self.dataset.bin_feature_names)), labels=self.dataset.global_feature_names + self.dataset.bin_feature_names, rotation=45);
        return fig, ax
    
class MultiStepEvaluator(EvaluationPlotter):
    def __init__(self, agent, data_container: DataContainer, plotter: EvaluationPlotter, device='cuda'):
        super().__init__()
        self.agent = agent
        self.data = data_container
        self.device = device

    def run(self, env, cfg, field_choice_method, eval_outdir):
        self.eval_metrics = self.agent.evaluate(env=env, cfg=cfg, num_episodes=1, field_choice_method=field_choice_method, eval_outdir=eval_outdir)
        agent_schedule, glob_df, bin_df = self._process_eval_metrics(self.eval_metrics)
        self.data.populate_agent_df()

    def _process_eval_metrics(self, eval_metrics):
        with open('eval_outdir/eval_metrics.pkl', 'rb') as f:
            eval_metrics = pickle.load(f)
        eval_metrics = eval_metrics['ep-0']
        glob_df = pd.concat(
            [pd.DataFrame(eval_metrics[night]['glob_observations'], columns=self.data.dataset.global_feature_names).assign(night=night) 
            for night in eval_metrics], 
            ignore_index=True
        )
        bin_df = pd.concat(
            [pd.DataFrame(eval_metrics[night]['bin_observations'], columns=self.data.dataset.global_feature_names).assign(night=night) 
            for night in eval_metrics], 
            ignore_index=True
        )
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
        eval_filter_df = pd.concat(
            [pd.DataFrame(eval_metrics[night]['filter'], columns=['filter']).assign(night=night)
            for night in eval_metrics],
            ignore_index=True
        )
        valid_mask = eval_bin_df['bin'] != -1 & eval_bin_df['bin'] != -2
        agent_schedule_full = {
            'timestamp': eval_timestamp_df['timestamp'].values[valid_mask],
            'field_id': eval_field_df['field_id'].values[valid_mask],
            'bin_idx': eval_bin_df['bin'].values[valid_mask],
            'filter_idx': eval_filter_df['filter'].values[valid_mask],
        }
        return agent_schedule_full, glob_df[valid_mask], bin_df[valid_mask]