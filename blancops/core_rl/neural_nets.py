import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
import logging
logger = logging.getLogger(__name__)

def setup_network():
    pass

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

class ScoreMLP(nn.Module):
    """
    Outputs multiple scores for each input
    """
    def __init__(self, global_dim, bin_feat_dim, hidden_dim, score_dim=1, nlayers=3, use_contextual_gating=False, activation=None):
        super(ScoreMLP, self).__init__()
        self.score_dim = score_dim
        self.activation = nn.ReLU if activation is None else activation
        self.use_contextual_gating = use_contextual_gating
        if use_contextual_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(global_dim, bin_feat_dim),
                nn.Sigmoid()
                )
        input_dim = global_dim + bin_feat_dim
        
        layers = []
        if nlayers < 0:
            raise ValueError("Number of layers cannot be less than 1")
        if nlayers == 1:
            layers.append(nn.Linear(input_dim, score_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self.activation())
            layers.append(nn.Linear(hidden_dim, score_dim))
        self.net = nn.Sequential(*layers)

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
    
class MultiHeadMultiScoreNet(nn.Module):
    def __init__(self, global_dim, bin_feat_dim, hidden_dim, score_dim=1, activation=None, use_contextual_gating=False):
        super().__init__()
        self.activation = nn.ReLU if activation is None else activation
        self.glob_enc = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            self.activation()
        )
        self.bin_enc = nn.Sequential(
            nn.Linear(bin_feat_dim, hidden_dim),
            self.activation()
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, score_dim) 
        )

    def forward(self, x_glob, x_bin, y_data=None):
        batch_size, n_bins, _ = x_bin.shape
        
        # 1. Process independently
        g_emb = self.glob_enc(x_glob) 
        g_emb = g_emb.unsqueeze(1).expand(-1, n_bins, -1) # Shape: (Batch, Bins, Hidden)
        
        b_emb = self.bin_enc(x_bin) # Shape: (Batch, Bins, Hidden)
        
        # 2. Fuse deep in the network
        fused = torch.cat([g_emb, b_emb], dim=-1) # Shape: (Batch, Bins, Hidden * 2)
        
        # 3. Output scores
        scores = self.net(fused) 
        return scores.view(batch_size, -1)
    
class StateEncoder(nn.Module):
    def __init__(self, glob_dim, bin_dim, nbins, glob_hidden, bin_hidden, bin_out, output_dim, activation=None):
        super().__init__()
        self.activation = nn.ReLU if activation is None else activation
        # Global encoder
        self.glob_enc = nn.Sequential(
            nn.Linear(glob_dim, glob_hidden),
            self.activation(),
            nn.Linear(glob_hidden, glob_hidden),
            self.activation()
        )
        # Bin encoder - processes each bin's features independently and then flattens
        self.bin_enc = nn.Sequential(
            nn.Linear(bin_dim, bin_hidden),
            self.activation(),
            nn.Linear(bin_hidden, bin_out),
            self.activation()
        )
        # Concatenate global and entire sky features (ie, the flattened bin features) and output latent state representation
        self.fusion_net = nn.Sequential(
            nn.Linear(glob_hidden + bin_out * nbins, output_dim),
            self.activation()
        )
        
    def forward(self, x_glob, x_bin):
        x_context = self.glob_enc(x_glob) # shape (batch, glob_hidden)
        x_binfeats = self.bin_enc(x_bin) # shape (batch, nbinbs, bin_hidden)
        x_binfeats_flat = x_binfeats.view(x_binfeats.size(0), -1) # Shape: (batch_size, nbins * bin_hidden)
        x = torch.cat([x_context, x_binfeats_flat], dim=-1) #shape (batch, glob_hidden + nbins * bin_hidden)
        x_latent = self.fusion_net(x)
        return x_latent
    
from torch.distributions import Categorical

class AutoregressiveDiscreteNet(nn.Module):
    def __init__(self, glob_dim, bin_dim, action_dims, glob_hidden, bin_hidden, nbins, nfilters, bin_out, state_latent_dim, activation=None, hidden_dim=256, emb_dim=None, bin_first=False):
        """
        Args:
            state_dim (int): Dimension of the input state.
            action_dims (list of int): A list containing the number of discrete 
                                       choices for each action dimension.
            hidden_dim (int): Number of hidden units for the state encoder.
            emb_dim (int): Embedding size for previously selected actions.
        """
        super().__init__()
        self.action_dims = action_dims
        self.num_actions = len(action_dims)
        self.activation = nn.ReLU if activation is None else activation
        self._filt_idx = int(bin_first)
        self._bin_idx = int(not bin_first)
        
        if emb_dim is None:
            emb_dim = state_latent_dim // 8 # about 11% of latent state size - tune later

        # State encoder
        self.state_encoder = StateEncoder(glob_dim, bin_dim, nbins=nbins, glob_hidden=glob_hidden, bin_hidden=bin_hidden, bin_out=bin_out, output_dim=state_latent_dim, activation=self.activation)
        
        # 2. Embeddings for previously sampled actions (we don't need one for the final action)
        if bin_first:
            self.action_embeddings = nn.ModuleList([
                nn.Embedding(nbins, emb_dim)
            ])
        else: # Filter first
            self.action_embeddings = nn.ModuleList([
                nn.Embedding(nfilters, emb_dim)
            ])
        
        # Autoregressive action heads
        self.action_heads = nn.ModuleList()
        for i in range(self.num_actions):
            # Input to the i-th head is the state features + embeddings of all prior actions
            input_dim = hidden_dim + (i * emb_dim)
            self.action_heads.append(nn.Linear(input_dim, action_dims[i]))
        

    def forward(self, x_glob, x_bin, action_mask, action=None):
        # GET LATENT SPACE REPRESENTATION
        x_latent = self.state_encoder(x_glob, x_bin)
        
        pred_actions = []
        log_probs = []
        entropies = []
        
        x_current = x_latent
        
        # For each head, COMPUTE LOGITS, PRED ACTIONS, and APPEND EMBEDDING for next head
        for i in range(self.num_actions):
            # 1. COMPUTE LOGITS
            logits = self.action_heads[i](x_current)
            
            if action_mask is not None:
                # Unflatten mask (ie, flat_idx = (bin * nfilters) + filt)
                batch_size = logits.size(0)
                nfilters = self.action_dims[self._filt_idx]
                nbins = self.action_dims[self._bin_idx]
                
                mask_2d = action_mask.view(batch_size, nbins, nfilters)
                
                # For first head, mask is just the original mask. For second head, must index into the mask based on the first head's sampled action
                if i == 0:
                    if self._filt_idx == 0: 
                        step_mask = mask_2d.any(dim=1) # shape: (batch, nfilters)
                    else:
                        step_mask = mask_2d.any(dim=2) # shape: (batch, nbins)
                else:
                    first_choice = pred_actions[0]
                    batch_idx = torch.arange(batch_size, device=logits.device)
                    
                    if self._bin_idx == 1:
                        step_mask = mask_2d[batch_idx, :, first_choice]
                    else:
                        step_mask = mask_2d[batch_idx, first_choice, :]

                # 2. PREDICT ACTIONS
                mask_value = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~step_mask, mask_value)
            
            dist = Categorical(logits=logits)
            
            if action is None: # ie, inference
                a_i = logits.argmax(dim=-1)
            else:
                a_i = action[:, i] # training
                
            pred_actions.append(a_i)
            log_probs.append(dist.log_prob(a_i))
            entropies.append(dist.entropy())
            
            # 3. APPEND EMBEDDING
            if i < self.num_actions - 1:
                emb = self.action_embeddings[i](a_i)
                x_current = torch.cat([x_current, emb], dim=-1)
                
        pred_actions = torch.stack(pred_actions, dim=1)
        
        # Joint log probability and entropy are the sum of the individual step calculations
        joint_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        joint_entropy = torch.stack(entropies, dim=1).sum(dim=1)
        
        return pred_actions, joint_log_prob, joint_entropy
    
class BinEmbeddingDQN(nn.Module):
    """Deep Q-Network mapping observations to action-values.
    """
    def __init__(self, n_global_features, n_bin_features, action_dim, hidden_dim=128, activation=None, embedding_dim=None):
        super(BinEmbeddingDQN, self).__init__()

        self.activation = nn.ReLU if activation is None else activation

        self.bin_embedding = nn.Embedding(action_dim, embedding_dim)
        
        input_dim = (n_bin_features + embedding_dim) * action_dim + n_global_features

        self.net = nn.Sequential(
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
        
        return self.net(full_input)

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
