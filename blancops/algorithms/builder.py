import logging

from blancops.data_processing.features import calculate_distance_matrix
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from blancops.algorithms.ddqn import DDQN
from blancops.algorithms.bc import BehaviorCloning
import copy
import torch
import torch.nn as nn
from blancops.core_rl.neural_nets import SingleScoreMLP, ScoreMLP, AutoregressiveDiscreteNet
from blancops.algorithms.policies import FlatActionPolicy, AutoregressiveActionPolicy, FlatQNetWrapper, AutoregressiveQNetWrapper
from blancops.algorithms.bc import BehaviorCloning
from blancops.algorithms.ddqn import DDQN

def get_activation(name):
    activations = {'relu': nn.ReLU, 'mish': nn.Mish, 'swish': nn.SiLU}
    if name not in activations:
        raise ValueError(f"Activation {name} not supported.")
    return activations[name]

def build_neural_network(config):
    activation_fn = get_activation(config['model']['activation'])
    
    if config['model']['grid_network'] == 'single_bin_scorer':
        return SingleScoreMLP(
            input_dim=config['data']['n_global_features'] + config['data']['n_bin_features'],
            hidden_dim=config['train']['hidden_dim'],
            activation=activation_fn
        )
    elif config['model']['grid_network'] == 'multi_dim_scorer':
        return ScoreMLP(
            global_dim=config['data']['n_global_features'],
            bin_feat_dim=config['data']['n_bin_features'],
            score_dim=config['data']['num_filters'],
            hidden_dim=config['train']['hidden_dim'],
            activation=activation_fn
        )
    elif config['model']['grid_network'] == 'autoregressive':
        action_dims = [config['data']['num_filters'], config['data']['nbins']] if not config['model']['bin_first'] else [config['data']['nbins'], config['data']['num_filters']]
        return AutoregressiveDiscreteNet(
            glob_dim=config['data']['n_global_features'],
            bin_dim=config['data']['n_bin_features'],
            action_dims=action_dims,
            glob_hidden=config['model']['glob_hidden'],
            bin_hidden=config['model']['bin_hidden'],
            bin_out=config['model']['bin_out'],
            state_latent_dim=config['model']['state_latent_dim'],
            activation=activation_fn,
            bin_first=config['model']['bin_first'],
            nbins=config['data']['nbins'],
            nfilters=config['data']['num_filters']
        )
    else:
        raise NotImplementedError(f"Network {config['model']['grid_network']} not implemented.")

def build_algorithm(config, device):
    core_net = build_neural_network(config).to(device)
    
    optimizer = torch.optim.Adam(core_net.parameters(), lr=config['train']['lr'])
    if config['model']['algorithm'] == 'BC':
        if config['model']['grid_network'] == 'autoregressive':
            policy = AutoregressiveActionPolicy(core_net, config['data']['num_filters'])
        else:
            # e.g., CrossEntropyLoss
            loss_fxn = nn.CrossEntropyLoss(reduction='mean') 
            policy = FlatActionPolicy(core_net, loss_fxn, config['data']['num_filters'])
            
        return BehaviorCloning(
            policy=policy,
            optimizer=optimizer,
            device=device
        )
    elif config['model']['algorithm'] in ['DQN', 'DDQN', 'CQL']:
        # Create the Target Network by deeply copying the core net
        target_net = copy.deepcopy(core_net).to(device)
        
        # Select the right Q-Value Adapters
        if config['model']['grid_network'] == 'autoregressive':
            policy = AutoregressiveQNetWrapper(core_net, config['data']['num_filters'])
            target = AutoregressiveQNetWrapper(target_net, config['data']['num_filters'])
        else:
            policy = FlatQNetWrapper(core_net)
            target = FlatQNetWrapper(target_net)

        loss_fxn = nn.HuberLoss(reduction='mean') if config['model']['loss_function'] == 'huber' else nn.MSELoss(reduction='mean')
        
        # CQL Specific Math
        use_cql = (config['model']['algorithm'] == 'CQL')
        dist_matrix = None
        dist_scaling_factor = 0
        if use_cql:
            # Assuming calculate_distance_matrix is imported
            dist_matrix = calculate_distance_matrix(nside=config['data']['nside'], is_azel='azel' in config['data']['bin_space'])
            Q_max = 1 / (1 - config['model']['gamma'])
            dist_scaling_factor = Q_max / torch.pi

        return DDQN(
            policy=policy,
            target=target,
            optimizer=optimizer,
            gamma=config['model']['gamma'],
            tau=config['model']['tau'],
            loss_fxn=loss_fxn,
            use_double=(config['algorithm_name'] in ['DDQN', 'CQL']),
            use_cql=use_cql,
            cql_alpha=config['model']['cql_alpha'],
            cql_margin=config['model']['cql_margin'],
            dist_matrix=dist_matrix,
            dist_scaling_factor=dist_scaling_factor,
            device=device
        )
    else:
        raise ValueError(f"Algorithm {config['algorithm_name']} unknown.")