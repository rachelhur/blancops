import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
import logging
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Deep Q-Network mapping observations to action-values.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation=None):
        super(MLP, self).__init__()
        self.activation = nn.ReLU if activation is None else activation 
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x_glob, x_bin=None, y_data=None):
        return self.net(x_glob)

class SingleScoreMLP(nn.Module):
    """
    Outputs one value for each input vector
    """
    def __init__(self, input_dim, hidden_dim, activation=None):
        super(SingleScoreMLP, self).__init__()
        self.activation = nn.ReLU if activation is None else activation
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x_glob, x_bin, y_data=None):
        x_glob = x_glob.unsqueeze(1) # (batch, 1, glob_dim)
        x_glob_exp = x_glob.expand(-1, x_bin.shape[1], -1) # (batch, n_bins, glob_dim)
        x = torch.cat((x_glob_exp, x_bin), dim=-1) # (batch, n_bins, glob_dim + bin_dim)
        return self.net(x).squeeze(-1)
    
class MultiScoreMLP(nn.Module):
    """
    Outputs multiple scores for each input
    """
    def __init__(self, global_dim, bin_feat_dim, hidden_dim, score_dim=1, use_contextual_gating=False, activation=None):
        super(MultiScoreMLP, self).__init__()
        self.score_dim = score_dim
        self.activation = nn.ReLU if activation is None else activation
        self.use_contextual_gating = use_contextual_gating
        if use_contextual_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(global_dim, bin_feat_dim),
                nn.Sigmoid()
                )
        input_dim = global_dim + bin_feat_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, score_dim)
        )
    def forward(self, x_glob, x_bin, y_data=None):
        batch_size, n_bins, _ = x_bin.shape
        x_glob = x_glob.unsqueeze(1) # (batch, 1, glob_dim)
        x_glob_exp = x_glob.expand(-1, n_bins, -1) # (batch, n_bins, glob_dim)
        if self.use_contextual_gating:
            gate_mask = self.gate_net(x_glob_exp)
            x_bin = x_bin * gate_mask
        else:
            pass
        x = torch.cat((x_glob_exp, x_bin), dim=-1) # (batch, n_bins, glob_dim + bin_dim)
        scores = self.net(x)
        joint_action_scores = scores.view(batch_size, -1) # flattens last dim (filter) first --> [bin0filter0, bin0filter1, ... bin1filter0, bin1filter1, ... binNfilterM]
        return joint_action_scores 
    
class MultiHeadNetwork(nn.Module):
    def __init__(self, global_dim, global_enc_dim, output_dim, num_heads, hidden_dim=128, activation=None):
        super(MultiHeadNetwork, self).__init__()
        self.activation = nn.ReuLu if activation is None else activation
        self.global_enc = MLP(global_dim, global_enc_dim)
        self.spatial_enc = MLP()

class BinEmbeddingDQN(nn.Module):
    """Deep Q-Network mapping observations to action-values.
    """
    def __init__(self, n_global_features, n_bin_features, action_dim, hidden_dim=128, activation=None, embedding_dim=None):
        super(BinEmbeddingDQN, self).__init__()

        self.activation = nn.ReLU if activation is None else activation

        self.bin_embedding = nn.Embedding(action_dim, embedding_dim)
        
        input_dim = (n_bin_features + embedding_dim) * action_dim + n_global_features

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, actions):
        local_features, global_features, bin_features = state

        bin_embeddings = self.bin_embedding(actions) # [batch, n_bins, emb_dim]
        bin_input = torch.cat([bin_features, bin_embeddings], dim=-1)  # [batch, n_bins, n_features + emb_dim]
        
        bin_flat = bin_input.flatten(start_dim=1)  # [batch, n_bins * (n_features + emb_dim)]
        full_input = torch.cat([bin_flat, local_features, global_features], dim=-1)
        
        return self.policy_net(full_input)

class SpatialEncoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
    def forward(self, x):
        # x shape: (Batch, Features, Lat, Lon)
        return self.cnn(x)


# def linear_schedule(eps_start: float, eps_end: float, duration: int, t: int):
#     """
#     Decreases epsilon linearly as a function of time step.

#     Args
#     ----
#     eps_start: float
#         starting epsilon value
#     eps_end: float
#         final epsilon value
#     duration: int
#         Number of time steps it takes to move from eps_start to eps_end
#     t: ind
#         Current time step
    
#     Returns
#     ------
#     epsilon: float
#         Current epsilon value
        
#     """
#     try:
#         slope = (eps_end - eps_start) / duration
#         return max(slope * t + eps_start, eps_end)
#     except:
#         raise Exception("Error in linear_schedule")

# def exponential_schedule(eps_start: float, eps_end: float, decay_rate: float, t: int):
#     try:
#         return eps_end + (eps_start - eps_end) * np.exp(-1. * t / decay_rate)
#     except:
#         raise Exception("Error in exponential_schedule")
