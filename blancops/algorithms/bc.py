import numpy as np
import torch
import torch.nn.functional as F

from blancops.core_rl.neural_nets import MLP, MultiHeadMultiScoreNet, SingleScoreMLP, BinEmbeddingDQN, MultiScoreMLP
from blancops.math import geometry
from blancops.algorithms.base import AlgorithmBase
from blancops.data_processing.constants import GRID_NETWORKS
import logging
logger = logging.getLogger(__name__)

from pathlib import Path

class BehaviorCloning(AlgorithmBase):
    def __init__(self, n_global_features, n_bin_features, num_actions, hidden_dim, num_filters=None, loss_fxn=None, activation=None, lr=1e-3, lr_scheduler=None, lr_scheduler_kwargs=None, \
                    lr_scheduler_epoch_start=1, lr_scheduler_num_epochs=5, device='cpu', grid_network=None, use_contextual_gating=False,
                    embedding_dim=None
                    ):
        super().__init__()
        assert grid_network in [None] + GRID_NETWORKS
        
        assert loss_fxn is not None, "loss_fxn needs to be passed"
        self.loss_fxn = loss_fxn

        self.name = 'BC'
        self.device = device
        self.num_filters = num_filters
        if grid_network is None:
            obs_dim = n_global_features + n_bin_features
            self.policy_net = MLP(input_dim=obs_dim, output_dim=num_actions, hidden_dim=hidden_dim, activation=activation).to(device)
        elif grid_network == 'single_bin_scorer':
            self.policy_net = SingleScoreMLP(input_dim=n_global_features + n_bin_features, hidden_dim=hidden_dim, activation=activation).to(device)
        elif grid_network == 'multi_dim_scorer':
            self.policy_net = MultiScoreMLP(global_dim=n_global_features, bin_feat_dim=n_bin_features, score_dim=num_filters, hidden_dim=hidden_dim, activation=activation, use_contextual_gating=use_contextual_gating).to(device)
        elif grid_network == 'multi_head_scorer':
            self.policy_net = MultiHeadMultiScoreNet(global_dim=n_global_features, bin_feat_dim=n_bin_features, score_dim=num_filters, hidden_dim=hidden_dim, activation=activation, use_contextual_gating=use_contextual_gating).to(device)
        else:
            raise NotImplementedError(f"grid_network {grid_network} not implemented")
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.lr_scheduler = self._initialize_scheduler(lr_scheduler, lr_scheduler_kwargs, self.optimizer)
        if lr_scheduler is not None:
            logger.debug(f'lr_scheduler is {self.lr_scheduler}')
            self.lr_scheduler_epoch_start = lr_scheduler_epoch_start
            self.lr_scheduler_num_epochs = lr_scheduler_num_epochs
        
        self.val_metrics = ['val_loss', 'logp_expert_action', 'action_margin', 'entropy', 'ang_sep', 'unique_bins', 'accuracy', 'filter_accuracy']
        
    def train_step(self, batch, epoch_num, step_num=None, hpGrid=None, compute_metrics=False):
        """
        Train the policy to mimic expert actions from offline data
        """

        self.optimizer.zero_grad(set_to_none=True)

        state, expert_actions, rewards, next_state, dones, action_masks, next_action_masks, bin_states, next_bin_states = batch
         
        # Assume batch is already tensor
        state_dtype = torch.float32
        state = state.to(device=self.device, dtype=state_dtype)
        bin_states = bin_states.to(device=self.device, dtype=state_dtype)
        expert_actions = expert_actions.to(device=self.device, dtype=torch.long).view(-1)
        action_masks = action_masks.to(device=self.device, dtype=torch.bool)
         
        # Compute loss
        self.policy_net.train()
        with torch.amp.autocast('cuda', dtype=state_dtype):
            action_logits = self.policy_net(x_glob=state, x_bin=bin_states, y_data=None)
            loss = self.loss_fxn(action_logits, expert_actions)

        # Default return nothing
        metrics_dict = {}

        # 3. Fast GPU Metrics Calculation (Pre-Update)
        with torch.no_grad():
            if compute_metrics:
                # Mask invalid actions for accuracy calculation
                action_logits_masked = action_logits.clone()
                action_logits_masked[~action_masks] = -1e9
                predicted_actions = action_logits_masked.argmax(dim=1)
                
                accuracy = (predicted_actions == expert_actions).float().mean().item()

                # Log Probabilities & Entropy
                logp = F.log_softmax(action_logits, dim=-1)
                logp_expert_actions = logp.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
                
                p = F.softmax(action_logits, dim=-1)
                entropy = -(p * logp).sum(dim=-1).mean().item()

                # Margin (Expert vs Next Best)
                _, num_actions = action_logits.shape
                z_expert = action_logits.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
                expert_mask = F.one_hot(expert_actions, num_classes=num_actions).bool()
                z_max_other = action_logits.masked_fill(expert_mask, float("-inf")).max(dim=1).values
                margin = (z_expert - z_max_other).mean().item()

                # Default values for heavy metrics
                ang_sep = 0.0
                unique_bins = 0.0
                filter_accuracy = 0.0
                
                if hpGrid is not None:
                    # Get angular separation
                    predicted_actions = predicted_actions.cpu()
                    expert_actions = expert_actions.cpu()

                    if self.num_filters is not None and self.num_filters != 1:                  
                        predicted_bins = predicted_actions // self.num_filters
                        expert_bins = expert_actions // self.num_filters
                        predicted_filters = predicted_actions % self.num_filters
                        expert_filters = expert_actions % self.num_filters
                        filter_accuracy = (predicted_filters == expert_filters).float().mean().item()
                    else:
                        predicted_bins = predicted_actions
                        expert_bins = expert_actions
                        filter_accuracy = 0.

                    predicted_coords = np.array((hpGrid.lon[predicted_bins], hpGrid.lat[predicted_bins]))
                    expert_actions_coords = np.array((hpGrid.lon[expert_bins], hpGrid.lat[expert_bins]))
                    ang_seps = geometry.angular_separation(predicted_coords, expert_actions_coords)
                    ang_sep = ang_seps.mean()

                    # Prediction diversity
                    num_actions = len(hpGrid.lon)
                    unique_preds = len(torch.unique(predicted_actions))
                    unique_bins = unique_preds / num_actions
                
                metrics_dict = {
                    'train_loss': loss.item(),
                    'accuracy': accuracy,
                    'action_margin': margin,
                    'entropy': entropy,
                    'logp_expert_action': logp_expert_actions.mean().item(),
                    'ang_sep': ang_sep,
                    'unique_bins': unique_bins,
                    'filter_accuracy': filter_accuracy
                }

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        do_lr_scheduler_step = (self.lr_scheduler is not None
                                and epoch_num >= self.lr_scheduler_epoch_start
                                and epoch_num <= self.lr_scheduler_num_epochs + self.lr_scheduler_epoch_start
        )
        if do_lr_scheduler_step:
            self.lr_scheduler.step()

        return metrics_dict
    
    def val_step(self, batch, hpGrid=None):
        
        state, expert_actions, rewards, next_state, dones, action_masks, next_action_masks, bin_states, next_bin_states = batch

        # Assume batch is already tensor
        state_dtype = torch.float32
        state = state.to(device=self.device, dtype=state_dtype)
        expert_actions = expert_actions.to(device=self.device, dtype=torch.long).view(-1)
        bin_states = bin_states.to(device=self.device, dtype=state_dtype)
        action_masks = action_masks.to(device=self.device, dtype=torch.bool)

        with torch.amp.autocast('cuda', dtype=state_dtype):
            action_logits = self.policy_net(x_glob=state, x_bin=bin_states)
            action_logits_masked = action_logits.clone()
            action_logits_masked[~action_masks] = -1e9

            predicted_actions = action_logits_masked.argmax(dim=1)
            loss = self.loss_fxn(action_logits, expert_actions)

        accuracy = (predicted_actions == expert_actions).float().mean()

        # Get logp(a_expert|state)
        logp = F.log_softmax(action_logits, dim=-1)
        logp_expert_actions = logp.gather(1, expert_actions.unsqueeze(1)).squeeze(1)

        # Get action margin
        _, num_actions = action_logits.shape
        # expert logit: (B,)
        z_expert = action_logits.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
        expert_mask = F.one_hot(expert_actions, num_classes=num_actions).bool()
        # max logit among non-expert actions
        z_max_other = action_logits.masked_fill(expert_mask, float("-inf")).max(dim=1).values
        margin = (z_expert - z_max_other).mean()

        # Get policy entropy (p(a_i|s)logp(a_i|s))
        p = F.softmax(action_logits, dim=-1)
        entropy = -(p * logp).sum(dim=-1)

        if hpGrid is not None:
            # Get angular separation
            predicted_actions = predicted_actions.cpu()
            expert_actions = expert_actions.cpu()

            if self.num_filters is not None and self.num_filters != 1:                  
                predicted_bins = predicted_actions // self.num_filters
                expert_bins = expert_actions // self.num_filters
                predicted_filters = predicted_actions % self.num_filters
                expert_filters = expert_actions % self.num_filters
                filter_accuracy = (predicted_filters == expert_filters).float().mean().item()
            else:
                predicted_bins = predicted_actions
                expert_bins = expert_actions
                filter_accuracy = 0.

            predicted_coords = np.array((hpGrid.lon[predicted_bins], hpGrid.lat[predicted_bins]))
            expert_actions_coords = np.array((hpGrid.lon[expert_bins], hpGrid.lat[expert_bins]))
            ang_seps = geometry.angular_separation(predicted_coords, expert_actions_coords)
            ang_sep = ang_seps.mean()

            # Prediction diversity
            num_actions = len(hpGrid.lon)
            unique_preds = len(torch.unique(predicted_actions))
            unique_bins = unique_preds / num_actions
        else:
            ang_sep = 0

        # Return dictionary to cleanly pass to the fit() loop logger
        return loss.item(), logp_expert_actions.mean().item(), margin.mean().item(), entropy.mean().item(), ang_sep, unique_bins, accuracy.item(), filter_accuracy
    
    def select_action(self, x_glob, x_bin, action_mask, epsilon=None):
        with torch.no_grad():
            if not torch.is_tensor(x_glob):
                x_glob = torch.tensor(x_glob, dtype=torch.float32)
                x_bin = torch.tensor(x_bin, dtype=torch.float32)
                mask = torch.tensor(action_mask, dtype=torch.bool)
            x_glob = x_glob.to(self.device, dtype=torch.float32).unsqueeze(0)
            x_bin = x_bin.to(self.device, dtype=torch.float32).unsqueeze(0)
            mask = mask.to(self.device, dtype=torch.bool).unsqueeze(0)
            action_logits = self.policy_net(x_glob=x_glob, x_bin=x_bin, y_data=None)
            
            # mask invalid actions
            mask = mask.view_as(action_logits)
            action_logits[~mask] = -1e9
            action = torch.argmax(action_logits, dim=1)

            return action.cpu().numpy()[0] if action.size(0) == 1 else action.cpu().numpy()

    # def predict(self, state):
    #     """
    #     Get action for a given state
    #     """
    #     self.policy_net.eval()
    #     with torch.no_grad():
    #         state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
    #         if len(state_tensor.shape) == 1:
    #             state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                
    #         action_logits = self.policy_net(state_tensor)
    #         action = torch.argmax(action_logits, dim=1)
    #         return action.cpu().numpy()[0] if action.size(0) == 1 else action.cpu().numpy()
    
    # def evaluate(self, test_dataset):
    #     """
    #     Evaluate the trained policy on test data
    #     """
    #     states = torch.tensor(np.array([d['state'] for d in test_dataset]), dtype=torch.float32)
    #     expert_actions = torch.tensor([d['expert_action'] for d in test_dataset], dtype=torch.long)
        
    #     states = states.to(self.device)
    #     expert_actions = expert_actions.to(self.device)
        
    #     with torch.no_grad():
    #         action_logits = self.policy_net(states)
    #         predicted_actions = torch.argmax(action_logits, dim=1)
    #         accuracy = (predicted_actions == expert_actions).float().mean().item()
        
    #     return accuracy