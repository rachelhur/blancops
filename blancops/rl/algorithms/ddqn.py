import numpy as np
import torch
import torch.nn.functional as F

from blancops.rl.neural_nets.neural_nets import MLP, MultiHeadMLP, BinEmbeddingDQN, ContextualScoreMLP
from blancops.math import geometry
from blancops.rl.algorithms.base import AlgorithmBase
import logging
logger = logging.getLogger(__name__)
from blancops.data.constants import NUM_FILTERS

from pathlib import Path
class DDQN(AlgorithmBase):
    def __init__(
        self, 
        policy,             # Inject the wrapped policy network
        target,             # Inject the wrapped target network
        gamma=0.99, 
        tau=0.005, 
        loss_function=None, 
        optimizer=None,
        lr_scheduler=None, 
        device='cpu', 
        use_double=True, 
        use_cql=True, 
        cql_alpha=1.0, 
        cql_margin=0.0,
        dist_matrix=None,
        dist_scaling_factor=1.0
    ):
        super().__init__()
        assert loss_function is not None, "loss_fxn needs to be passed"
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.loss_function = loss_function
        self.use_double = use_double
        self.use_cql = use_cql
        self.cql_alpha = cql_alpha
        self.cql_margin = cql_margin
        self.dist_matrix = dist_matrix
        self.dist_scaling_factor = dist_scaling_factor

        # 1. Store the injected networks
        self.policy = policy.to(device)
        self.target_net = target.to(device)
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # 2. Setup Optimizer (Injected from config)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
    def train_step(self, batch, epoch_num, step_num=None, hpGrid=None, compute_metrics=False):  
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
        
        with torch.amp.autocast(device_type='cuda'):
            q_vals_all = self.policy.get_q_values(state, bin_states)
            q_val = q_vals_all.gather(1, actions).squeeze(1) 
            
            with torch.no_grad():
                if self.use_double:
                    q_vals_next = self.policy.get_q_values(next_state, next_bin_states)
                    
                    mask_val = torch.finfo(q_vals_next.dtype).min
                    q_vals_next = q_vals_next.masked_fill(~next_action_masks, mask_val)
                    a_best = q_vals_next.argmax(1).type(torch.long)
                    
                    target_q_next = self.target_net.get_q_values(next_state, next_bin_states)
                    target_q_state = target_q_next.gather(1, a_best.unsqueeze(1)).squeeze(1)
                else:    
                    next_q = self.target_net.get_q_values(next_state, next_bin_states)
                    mask_val = torch.finfo(next_q.dtype).min
                    next_q = next_q.masked_fill(~next_action_masks, mask_val) 
                    target_q_state = next_q.max(dim=1)[0]
                q_expected = rewards + self.gamma * target_q_state * (1 - dones)

            loss = self.loss_function(q_val, q_expected)

            # CQL Penalty
            cql_loss_val = 0.0 
            if self.use_cql:
                cql_loss = self._calculate_cql_loss(q_vals_all, q_val, actions, action_masks, margin=self.cql_margin)
                loss = loss + cql_loss
                cql_loss_val = cql_loss.item()
            
        # 3. Optimize
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        do_lr_scheduler_step = (self.lr_scheduler is not None
                                and epoch_num >= self.lr_scheduler_epoch_start
                                and epoch_num <= self.lr_scheduler_num_epochs + self.lr_scheduler_epoch_start)
        if do_lr_scheduler_step:
            self.lr_scheduler.step()

        self._soft_update()

        # 4. Metrics
        metrics_dict = {}
        with torch.no_grad():
            if compute_metrics:
                q_vals_eval = q_vals_all.clone()
                mask_val = torch.finfo(q_vals_eval.dtype).min
                q_vals_eval = q_vals_eval.masked_fill(~action_masks, mask_val)
                predicted_actions = q_vals_eval.argmax(1)
                
                metrics_dict = {
                    'train_loss': loss.item(),
                    'td_error': (q_val - q_expected).abs().mean().item(),
                    'q_std': q_vals_all.std().item(),
                    'q_policy': q_vals_all.max(dim=1)[0].mean().item(),
                    'q_expert': q_val.mean().item(),
                    'accuracy': (predicted_actions == actions.squeeze(1)).float().mean().item(),
                    'cql_loss': cql_loss_val,
                }
                if hpGrid is not None:
                    heavy_metrics = self.policy._compute_heavy_metrics(predicted_actions, actions.squeeze(1), hpGrid)
                    metrics_dict.update(heavy_metrics)
                    
                

        return metrics_dict
        # if self.use_cql:
        #     bin_idxs = actions // self.num_filters
        #     base_penalty_weights = self.cql_penalty_matrix[bin_idxs.squeeze(1)]
        #     penalty_weights = torch.repeat_interleave(base_penalty_weights, self.num_filters, dim=1)
        #     weighted_q_vals = q_vals_all_masked + penalty_weights # Q(s, a) + Penalty(a, a_exp)
        #     cql_logsumexp = torch.logsumexp(weighted_q_vals, dim=1)
        #     cql_penalty = (cql_logsumexp - q_val).mean() # log
        #     loss = loss + self.cql_alpha * cql_penalty

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
            next_action_masks = next_action_masks.to(device=self.device, dtype=torch.bool)
            rewards = rewards.to(device=self.device, dtype=state_dtype)
            dones = dones.to(device=self.device, dtype=state_dtype)

            # Warning log
            expert_actions_squeezed = actions.squeeze(1)
            invalid_expert_mask = ~action_masks[torch.arange(action_masks.size(0)), expert_actions_squeezed]
            if invalid_expert_mask.any():
                logger.debug(f"WARNING: {invalid_expert_mask.sum().item()} expert actions in this batch are masked as INVALID!")
            
            with torch.amp.autocast(device_type='cuda'):
                q_vals_all = self.policy.core_net(x_glob=state, x_bin=bin_states)
                q_val = q_vals_all.gather(1, actions).squeeze(1)

                q_vals_eval = q_vals_all.clone()
                q_vals_eval[~action_masks] = -1e9
                predicted_actions = q_vals_eval.argmax(1)

                if self.use_double:
                    q_vals_next = self.policy.core_net(x_glob=next_state, x_bin=next_bin_states)
                    q_vals_next[~next_action_masks] = -1e9
                    a_best = q_vals_next.argmax(1)
                    
                    target_q_next = self.target_net(x_glob=next_state, x_bin=next_bin_states)
                    target_q_state = target_q_next.gather(1, a_best.unsqueeze(1)).squeeze(1) # FIX: squeeze(1)
                else:
                    target_q_next = self.target_net(x_glob=next_state, x_bin=next_bin_states).clone()
                    target_q_next[~next_action_masks] = float('-inf')
                    target_q_state = target_q_next.max(1)[0]
                
                q_expected = rewards + self.gamma * target_q_state * (1 - dones)
                loss = self.loss_function(q_val, q_expected)
                td_loss = loss.item()

                cql_loss_val = 0.0 
                if self.use_cql:
                    cql_loss = self._calculate_cql_loss(q_vals_all, q_val, actions, action_masks, margin=self.cql_margin)
                    loss = loss + cql_loss
                    cql_loss_val = cql_loss.item()


            td_error_mean = (q_val - q_expected).abs().mean()
            mean_accuracy = (predicted_actions == actions.squeeze(1)).float().mean() 
            q_dataset_mean = q_val.mean()
            q_policy_mean = q_vals_all.max(dim=1)[0].mean()
            q_std = q_vals_all.std()

            ang_sep = 0.0
            unique_bins = 0.0
            filter_accuracy = 0.0

            if hpGrid is not None:
                predicted_actions_cpu = predicted_actions.cpu()
                actions_cpu = actions.squeeze(1).cpu() # FIX: Squeeze before CPU math

                if self.num_filters is not None and self.num_filters != 1:                  
                    predicted_bins = predicted_actions_cpu // self.num_filters
                    expert_bins = actions_cpu // self.num_filters
                    predicted_filters = predicted_actions_cpu % self.num_filters
                    expert_filters = actions_cpu % self.num_filters
                    filter_accuracy = (predicted_filters == expert_filters).float().mean().item()
                else:
                    predicted_bins = predicted_actions_cpu
                    expert_bins = actions_cpu

                predicted_coords = np.array((hpGrid.lon[predicted_bins], hpGrid.lat[predicted_bins]))
                actions_coords = np.array((hpGrid.lon[expert_bins], hpGrid.lat[expert_bins]))
                ang_seps = geometry.angular_separation(predicted_coords, actions_coords)
                ang_sep = ang_seps.mean()

                num_actions_space = len(hpGrid.lon)
                unique_preds = len(torch.unique(predicted_actions_cpu))
                unique_bins = unique_preds / num_actions_space

            return loss.item(), td_error_mean.item(), q_std.item(), q_policy_mean.item(), q_dataset_mean.item(), mean_accuracy.item(), ang_sep, \
                unique_bins, filter_accuracy, cql_loss_val, td_loss

            ## distance
            # if self.use_cql:
            #     q_vals_all_masked = q_vals.clone()
            #     q_vals_all_masked[~action_masks] = -1e9
                
            #     bin_idxs = actions // self.num_filters
            #     base_penalty_weights = self.cql_penalty_matrix[bin_idxs.squeeze(1)]
            #     penalty_weights = torch.repeat_interleave(base_penalty_weights, self.num_filters, dim=1)
                
            #     weighted_q_vals = q_vals_all_masked + penalty_weights
            #     cql_logsumexp = torch.logsumexp(weighted_q_vals, dim=1)
            #     cql_penalty = (cql_logsumexp - q_current).mean()
            #     cql_loss = self.cql_alpha * cql_penalty
            #     loss = loss + cql_loss
            #     cql_loss = cql_loss.item()

    def _calculate_cql_loss(self, q_vals_all, q_val_expert, actions, action_masks, margin=0.):
        """
        Calculates the Conservative Q-Learning (CQL) penalty with a discrete margin.
        
        Returns:
            cql_loss_tensor (torch.Tensor): For backpropagation.
            cql_loss_val (float): For detached metric logging.
        """
        # 1. Clone to protect the forward computation graph
        q_vals_cql = q_vals_all.clone()
        q_vals_cql[~action_masks] = -1e9 
        num_total_actions = q_vals_all.shape[1]
        expert_mask = F.one_hot(actions.squeeze(1), num_classes=num_total_actions).bool()
        q_vals_cql[~expert_mask] += margin

        cql_logsumexp = torch.logsumexp(q_vals_cql, dim=1)
        cql_penalty = (cql_logsumexp - q_val_expert).mean()
        
        # Calculate cql loss
        cql_loss_tensor = self.cql_alpha * cql_penalty
        
        return cql_loss_tensor

    def _compute_metrics(self):
        pass
    # def _calculate_cql_loss(self, q_vals, action_masks, actions, q_current, penalty_choice):
        # if penalty_choice == 'pointing_distance':
        #     q_vals_all_masked = q_vals.clone()
        #     q_vals_all_masked[~action_masks] = -1e9
            
        #     bin_idxs = actions // self.num_filters
        #     base_penalty_weights = self.cql_penalty_matrix[bin_idxs.squeeze(1)]
        #     penalty_weights = torch.repeat_interleave(base_penalty_weights, self.num_filters, dim=1)
            
        #     weighted_q_vals = q_vals_all_masked + penalty_weights
        #     cql_logsumexp = torch.logsumexp(weighted_q_vals, dim=1)
        #     cql_penalty = (cql_logsumexp - q_current).mean()
        #     cql_loss = self.cql_alpha * cql_penalty
        #     loss = loss + cql_loss
        #     cql_loss = cql_loss.item()
        
    def _soft_update(self):
        # update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def select_action(self, x_glob, x_bin, action_mask, epsilon=None):
        # if random sample less than epsilon, take random action
        if epsilon is not None:
            if np.random.random() < epsilon:
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
                return int(action)

        # greedy selection from policy
        with torch.no_grad():
            # 1. Convert to tensors efficiently
            x_glob = torch.as_tensor(x_glob, dtype=torch.float32, device=self.device)
            x_bin = torch.as_tensor(x_bin, dtype=torch.float32, device=self.device)
            
            # 2. Handle batch dimensions for single environment steps
            if x_glob.dim() == 1:
                x_glob = x_glob.unsqueeze(0)
            if x_bin.dim() == 1:  # Assuming bin_states also needs batch dim
                x_bin = x_bin.unsqueeze(0)
                
            # 3. Pass updated arguments to policy
            q_values = self.policy.core_net(x_glob=x_glob, x_bin=x_bin).squeeze(0)
            
            # 4. Mask invalid actions
            action_mask_tensor = torch.as_tensor(action_mask, device=self.device, dtype=torch.bool)
            q_values[~action_mask_tensor] = -1e9 # Using -1e9 to match your train_step
            
            action = torch.argmax(q_values).item()
            
        return int(action)
    