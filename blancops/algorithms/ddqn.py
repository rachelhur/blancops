import numpy as np
import torch
import torch.nn.functional as F

from blancops.core_rl.neural_nets import MLP, SingleScoreMLP, BinEmbeddingDQN, MultiScoreMLP
from blancops.math import geometry
from blancops.algorithms.base import AlgorithmBase
import logging
logger = logging.getLogger(__name__)
from blancops.data_processing.constants import NUM_FILTERS

from pathlib import Path

class DDQN(AlgorithmBase):
    """
    Implementation of the DDQN algorithm. Uses AdamW optimizer and, optionally, a cosine annealing lr scheduler.

    Args
    ----
    obs_dim (int): size of each observation
    num_actions (int): number of total possible actions
    hidden_dim (int): hidden dimension size in DQN network
    gamma (float): 
    tau (float): 
    device (str): 
    lr (float): Learning rate
    loss_fxn (torch.nn.functional): Loss function (ie F.huber_loss, F.mse_loss)
    use_dqn (bool): 
    optimizer_kwargs (optional): 
    """

    def __init__(self, n_global_features, n_bin_features, num_actions, hidden_dim, num_filters=None, gamma=.99, tau=.005, target_update_freq=1000, loss_fxn=None, activation=None, lr=1e-3, lr_scheduler=None, lr_scheduler_kwargs=None, \
                    lr_scheduler_epoch_start=1, lr_scheduler_num_epochs=5, device='cpu', grid_network=None, use_contextual_gating=False,
                    embedding_dim=None, use_double=True, use_cql=True, cql_alpha=1., dist_matrix=None, dist_scaling_factor=1.):
        super().__init__()

        assert loss_fxn is not None, "loss_fxn needs to be passed"
        self.loss_fxn = loss_fxn

        if use_cql:
            self.name = "CQL"
        elif use_double:
            self.name = 'DDQN'
        else:
            self.name = 'DQN'
        self.num_filters = num_filters
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.use_cql = use_cql
        self.cql_alpha = cql_alpha
        penalty_matrix = 1.0 * dist_matrix * dist_scaling_factor
        self.cql_penalty_matrix = torch.tensor(
            penalty_matrix, 
            dtype=torch.float32, 
            device=self.device
        )

        if grid_network is None:
            obs_dim = n_global_features + n_bin_features
            self.policy_net = MLP(input_dim=obs_dim, output_dim=num_actions, hidden_dim=hidden_dim, activation=activation).to(device)
            self.target_net = MLP(input_dim=obs_dim, output_dim=num_actions, hidden_dim=hidden_dim, activation=activation).to(device)
        elif grid_network == 'single_bin_scorer':
            self.policy_net = SingleScoreMLP(input_dim=n_global_features + n_bin_features, hidden_dim=hidden_dim, activation=activation).to(device)
            self.target_net = SingleScoreMLP(input_dim=n_global_features + n_bin_features, hidden_dim=hidden_dim, activation=activation).to(device)
        elif grid_network == 'multi_dim_scorer':
            self.policy_net = MultiScoreMLP(global_dim=n_global_features, bin_feat_dim=n_bin_features, score_dim=num_filters, hidden_dim=hidden_dim, activation=activation, use_contextual_gating=use_contextual_gating).to(device)
            self.target_net = MultiScoreMLP(global_dim=n_global_features, bin_feat_dim=n_bin_features, score_dim=num_filters, hidden_dim=hidden_dim, activation=activation, use_contextual_gating=use_contextual_gating).to(device)
        else:
            raise NotImplementedError(f"grid_network {grid_network} not implemented")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Freeze params (and save memory)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.use_double = use_double
        self.target_update_freq = target_update_freq

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.lr_scheduler = self._initialize_scheduler(lr_scheduler, lr_scheduler_kwargs, self.optimizer)

        if lr_scheduler is not None:
            self.lr_scheduler_epoch_start = lr_scheduler_epoch_start
            self.lr_scheduler_num_epochs = lr_scheduler_num_epochs

        self.val_metrics = ['val_loss', 'td_error', 'q_std', 'q_policy', 'q_expert', 'accuracy', 'ang_sep', 'unique_bins', \
                            'filter_accuracy', 'cql_penalty', 'td_loss']
        
    def train_step(self, batch, epoch_num, step_num):
        
        state, actions, rewards, next_state, dones, action_masks, next_action_masks, bin_states, next_bin_states = batch

        state_dtype = torch.float32
        state = state.to(device=self.device, dtype=state_dtype)
        next_state = next_state.to(device=self.device, dtype=state_dtype)
        bin_states = bin_states.to(device=self.device, dtype=state_dtype)
        next_bin_states = next_bin_states.to(device=self.device, dtype=state_dtype)
        actions = actions.to(device=self.device, dtype=torch.long).unsqueeze(1)
        action_masks = action_masks.to(device=self.device, dtype=torch.bool)
        next_action_masks = next_action_masks.to(device=self.device, dtype=torch.bool)
        rewards = rewards.to(device=self.device, dtype=state_dtype)
        dones = dones.to(device=self.device, dtype=state_dtype)
            
        # # need to input (batch_size, obs_dim) into net - if obs_dim is 1, we get 1d tensor. Need to reshape
        # if state.dim() == 1:
        #     state = state.unsqueeze(1)
        #     next_state = next_state.unsqueeze(1)
        
        # Q(s_t, a_t) = r_t + gamma Q(s', a*) = r_t + gamma Q(s', argmax_a' Q'(s', a))
            # a* := argmax_a' Q'(s, a)
            # Q := target network
            # Q' := current (policy) network
        # Loss = q_val - expected_q_val

        # 1. Get q_val
        q_vals_all = self.policy_net(x_glob=state, x_bin=bin_states)
        q_val = q_vals_all.gather(1, actions).squeeze(1) # Do  I need these squeezes after gather?
        
        if self.use_cql:
            q_vals_all_masked = q_vals_all.clone()
            q_vals_all_masked[~action_masks] = -1e9

        with torch.no_grad():
            if self.use_double:
                # 2. Get a* = argmax_a'[Q'(s', a')] from current network
                q_vals = self.policy_net(x_glob=next_state, x_bin=next_bin_states)
                q_vals[~next_action_masks] = -1e9
                a_best = q_vals.argmax(1).type(torch.long)
                
                # 3. Get Q(s', a*) from target net
                q_next = self.target_net(x_glob=next_state, x_bin=next_bin_states)
                target_qval = q_next.gather(1, a_best.unsqueeze(1)).squeeze(1)
                
                # Get expected Q
                dones_mask = 1 - dones
                expected_q = rewards + self.gamma * target_qval * dones_mask
            else:    
                next_q = self.target_net(x_glob=next_state, x_bin=next_bin_states) 
                next_q[~next_action_masks] = -1e9 
                max_next_q = next_q.max(dim=1)[0]
                expected_q = rewards + self.gamma * max_next_q * (1 - dones)        

        loss = self.loss_fxn(q_val, expected_q)

        if self.use_cql:
            bin_idxs = actions // self.num_filters
            base_penalty_weights = self.cql_penalty_matrix[bin_idxs.squeeze(1)]
            penalty_weights = torch.repeat_interleave(base_penalty_weights, self.num_filters, dim=1)
            weighted_q_vals = q_vals_all_masked + penalty_weights # Q(s, a) + Penalty(a, a_exp)
            cql_logsumexp = torch.logsumexp(weighted_q_vals, dim=1)
            cql_penalty = (cql_logsumexp - q_val).mean() # log
            loss = loss + self.cql_alpha * cql_penalty
        # optimize w/ backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        do_lr_scheduler_step = (self.lr_scheduler is not None
                                and epoch_num >= self.lr_scheduler_epoch_start
                                and epoch_num <= self.lr_scheduler_num_epochs + self.lr_scheduler_epoch_start
        )
        if do_lr_scheduler_step:
            self.lr_scheduler.step()

        self._soft_update()

        return loss.item(), q_vals_all.mean().item()

    def _soft_update(self):
        # update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def select_action(self, state, bin_states, action_mask, epsilon=None):
        # if random sample less than epsilon, take random action
        if epsilon is not None:
            if np.random.random() < epsilon:
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
                return int(action)

        # greedy selection from policy
        with torch.no_grad():
            # 1. Convert to tensors efficiently
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            bin_states = torch.as_tensor(bin_states, dtype=torch.float32, device=self.device)
            
            # 2. Handle batch dimensions for single environment steps
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if bin_states.dim() == 1:  # Assuming bin_states also needs batch dim
                bin_states = bin_states.unsqueeze(0)
                
            # 3. Pass updated arguments to policy_net
            q_values = self.policy_net(x_glob=state, x_bin=bin_states).squeeze(0)
            
            # 4. Mask invalid actions
            action_mask_tensor = torch.as_tensor(action_mask, device=self.device, dtype=torch.bool)
            q_values[~action_mask_tensor] = -1e9 # Using -1e9 to match your train_step
            
            action = torch.argmax(q_values).item()
            
        return int(action)
    
    def val_step(self, eval_batch, hpGrid=None):
        state, actions, rewards, next_state, dones, action_masks, next_action_masks, bin_states, next_bin_states = eval_batch

        with torch.no_grad():      
            state_dtype = torch.float32
            state = state.to(device=self.device, dtype=state_dtype)
            next_state = next_state.to(device=self.device, dtype=state_dtype)
            bin_states = bin_states.to(device=self.device, dtype=state_dtype)
            next_bin_states = next_bin_states.to(device=self.device, dtype=state_dtype)
            actions = actions.to(device=self.device, dtype=torch.long).unsqueeze(1)
            action_masks = action_masks.to(device=self.device, dtype=torch.bool)
            next_action_masks = next_action_masks.to(device=self.device, dtype=torch.bool) # FIX: Added
            rewards = rewards.to(device=self.device, dtype=state_dtype)
            dones = dones.to(device=self.device, dtype=state_dtype)

            q_vals = self.policy_net(x_glob=state, x_bin=bin_states)
            q_current = q_vals.gather(1, actions).squeeze()

            q_vals_eval = q_vals.clone()
            q_vals_eval[~action_masks] = -1e9
            predicted_actions = q_vals_eval.argmax(1)

            if self.use_double:
                pol_next_q = self.policy_net(x_glob=next_state, x_bin=next_bin_states)
                pol_next_q[~next_action_masks] = -1e9
                pol_next_actions = pol_next_q.argmax(1)
                
                target_next_q = self.target_net(x_glob=next_state, x_bin=next_bin_states)
                next_q_vals = target_next_q.gather(1, pol_next_actions.unsqueeze(1)).squeeze()
            else:
                target_next_q = self.target_net(x_glob=next_state, x_bin=next_bin_states).clone()
                target_next_q[~next_action_masks] = float('-inf')
                next_q_vals = target_next_q.max(1)[0]
            
            q_expected = rewards + self.gamma * next_q_vals * (1 - dones)
            loss = self.loss_fxn(q_current, q_expected)
            td_loss = loss.item()

            if self.use_cql:
                q_vals_all_masked = q_vals.clone()
                q_vals_all_masked[~action_masks] = -1e9
                
                bin_idxs = actions // self.num_filters
                base_penalty_weights = self.cql_penalty_matrix[bin_idxs.squeeze(1)]
                penalty_weights = torch.repeat_interleave(base_penalty_weights, self.num_filters, dim=1)
                
                weighted_q_vals = q_vals_all_masked + penalty_weights
                cql_logsumexp = torch.logsumexp(weighted_q_vals, dim=1)
                cql_penalty = (cql_logsumexp - q_current).mean()
                cql_loss = cql_penalty.item()
                loss = loss + self.cql_alpha * cql_penalty

            td_error_mean = (q_current - q_expected).abs().mean()

            mean_accuracy = (predicted_actions == actions.squeeze(1)).float().mean() 
            q_dataset_mean = q_current.mean()
            q_policy_mean = q_vals.max(dim=1)[0].mean()
            q_std = q_vals.std()

            if hpGrid is not None:
                # Get angular separation
                predicted_actions = predicted_actions.cpu()
                actions = actions.cpu()

                if self.num_filters is not None and self.num_filters != 1:                  
                    predicted_bins = predicted_actions // self.num_filters
                    expert_bins = actions // self.num_filters
                    predicted_filters = predicted_actions % self.num_filters
                    expert_filters = actions % self.num_filters
                    filter_accuracy = (predicted_filters == expert_filters).float().mean().item()
                else:
                    predicted_bins = predicted_actions
                    expert_bins = actions
                    filter_accuracy = 0.

                predicted_coords = np.array((hpGrid.lon[predicted_bins], hpGrid.lat[predicted_bins]))
                actions_coords = np.array((hpGrid.lon[expert_bins], hpGrid.lat[expert_bins]))
                ang_seps = geometry.angular_separation(predicted_coords, actions_coords)
                ang_sep = ang_seps.mean()

                # Prediction diversity
                num_actions = len(hpGrid.lon)
                unique_preds = len(torch.unique(predicted_actions))
                unique_bins = unique_preds / num_actions
            else:
                ang_sep = 0.0
                unique_bins = 0.0
                filter_accuracy = 0.0

            return loss.item(), td_error_mean.item(), q_std.item(), q_policy_mean.item(), q_dataset_mean.item(), mean_accuracy.item(), ang_sep, \
                unique_bins, filter_accuracy, cql_loss, td_loss
