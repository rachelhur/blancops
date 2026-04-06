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
from blancops.core_rl.neural_nets import ScoreMLP, AutoregressiveDiscreteNet
from blancops.algorithms.policies import AutoregressiveActionPolicy, FlatQNetWrapper, AutoregressiveQNetWrapper, HybridMarginalPolicy, PseudoAutoregressivePolicy, PureJointPolicy
from blancops.algorithms.policies import FocalLoss
from blancops.algorithms.bc import BehaviorCloning
from blancops.algorithms.ddqn import DDQN

def get_activation(name):
    activations = {'relu': nn.ReLU, 'mish': nn.Mish, 'swish': nn.SiLU}
    if name not in activations:
        raise ValueError(f"Activation {name} not supported.")
    return activations[name]

def build_neural_network(config):
    activation_fn = get_activation(config['model']['activation'])
    
    if config['data']['action_space'] == 'filter':
        # return MLP(
        #     input_dim=config['data']['n_global_features'],
        #     output_dim=config['data']['num_filters'],
        #     hidden_dim=config['train']['hidden_dim'],
        #     activation=activation_fn
        # )
        raise NotImplementedError
    # if config['model']['action_architecture'] is None:
    #     return MLP(
            
    #     )
    if config['model']['action_architecture'] == 'simultaneous':
        return ScoreMLP(
            global_dim=config['data']['n_global_features'],
            bin_feat_dim=config['data']['n_bin_features'],
            score_dim=config['data']['num_filters'],
            hidden_dim=config['train']['hidden_dim'],
            activation=activation_fn
        )
    elif config['model']['action_architecture'] == 'autoregressive':
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
        raise NotImplementedError(f"Network {config['model']['action_architecture']} not implemented.")

def build_algorithm(config, device):
    core_net = build_neural_network(config).to(device)
    pretrained_path = config['metadata'].get('pretrained_model_path', None)
    if pretrained_path:
        core_net.load_state_dict(torch.load(pretrained_path, map_location=device))
        logger.info(f"Loaded pretrained model from {pretrained_path}")
    optimizer = torch.optim.Adam(core_net.parameters(), lr=config['train']['lr'])
    
    if config['model']['algorithm'] == 'BC':
        loss_function = get_loss_function(config['model']['loss_function'], config['model'].get('reduction', 'mean'), config['model'].get('gamma_focal', 2.0), use_alpha_focal=True)
        arch = config['model'].get('action_architecture', 'simultaneous')
        
        if config['data']['action_space'] == 'filter':
            policy = PureJointPolicy(
                core_net=core_net, 
                loss_function=loss_function, 
                num_filters=config['data']['num_filters']
            )
            
        elif arch == 'autoregressive':
            policy = AutoregressiveActionPolicy(core_net, config['data']['num_filters'])
        
        elif arch == 'simultaneous':
            loss_strat = config['model'].get('loss_strategy', 'pure_joint')
            ce_loss_function = nn.CrossEntropyLoss(reduction='mean')
            
            if loss_strat == 'hybrid_marginal':
                if config['model'].get('loss_function', None) == 'focal_loss':
                    joint_loss_function = get_loss_function('focal_loss', use_alpha_focal=False)
                else:
                    joint_loss_function = get_loss_function('cross_entropy', config['model'].get('reduction', 'mean'))
                policy = HybridMarginalPolicy(
                    core_net=core_net,
                    num_filters=config['data']['num_filters'],
                    bin_loss_function=ce_loss_function,
                    filter_loss_function=loss_function,
                    joint_loss_function=joint_loss_function,
                    alpha_bin=config['model'].get('alpha_bin', 1.0),
                    beta_filter=config['model'].get('beta_filter', 5.0),
                    zeta_joint=config['model'].get('zeta_joint', 0.1)
                )
                
            elif loss_strat == 'pseudo_autoregressive':
                policy = PseudoAutoregressivePolicy(
                    core_net=core_net,
                    num_filters=config['data']['num_filters'],
                    filter_penalty=config['model'].get('filter_penalty', 5.0)
                )
            elif loss_strat == 'pure_joint':
                policy = PureJointPolicy(core_net, nn.CrossEntropyLoss(), config['data']['num_filters'])
            else:
                raise NotImplementedError(f"`{loss_strat}` loss strategy is not implemented.")
        else:
            raise NotImplementedError(f"`{arch}` architecture is not implemented.")
            
        return BehaviorCloning(
            policy=policy,
            optimizer=optimizer,
            lr_scheduler=config['train']['lr_scheduler'],
            lr_scheduler_epoch_start=config['train']['lr_scheduler_epoch_start'],
            lr_scheduler_num_epochs=config['train']['lr_scheduler_num_epochs'],
            lr_scheduler_kwargs=config['train']['lr_scheduler_kwargs'],
            device=device
        )
    elif config['model']['algorithm'] in ['DQN', 'DDQN', 'CQL']:
        # Create the Target Network by deeply copying the core net
        target_net = copy.deepcopy(core_net).to(device)
        
        # Select the right Q-Value Adapters
        if config['model']['action_architecture'] == 'autoregressive':
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
            dist_matrix = calculate_distance_matrix(nside=config['data']['nside'], is_azel='azel' in config['data']['action_space'])
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