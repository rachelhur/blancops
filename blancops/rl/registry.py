import copy
import logging
logger = logging.getLogger(__name__)

import torch
from torch import nn

from blancops.configs.schema import ExperimentConfig
from blancops.configs.enums import Algorithm, LossStrategy, Network
from blancops.rl.neural_nets.neural_nets import ContextualScoreMLP, MLP
from blancops.rl.algorithms.bc import BehaviorCloning
from blancops.rl.algorithms.ddqn import DDQN
from blancops.rl.policies.policies import FilterFocalLoss, FlatQNetWrapper, FocalLoss, HybridMarginalPolicy, PseudoAutoregressivePolicy, PureJointPolicy, SlewDistanceFocalLoss, AutoregressiveActionPolicy

BC_POLICY_REGISTRY = {
    LossStrategy.PURE_JOINT: PureJointPolicy,
    LossStrategy.AUTOREGRESSIVE: AutoregressiveActionPolicy,
    LossStrategy.PSEUDO_AUTOREGRESSIVE: PseudoAutoregressivePolicy,
    LossStrategy.HYBRID_MARGINAL: HybridMarginalPolicy,
}

ALGORITHM_REGISTRY = {
    Algorithm.BC: BehaviorCloning,
    Algorithm.DDQN: DDQN
}

NETWORK_REGISTRY = {
    Network.CONTEXTUAL_SCORE_MLP: ContextualScoreMLP,
    Network.MLP: MLP
}

ACTIVATION_REGISTRY = {
    'relu': nn.ReLU,
    'mish': nn.Mish,
    'swish': nn.SiLU
}

def get_activation(name: str):
    activations = {'relu': nn.ReLU, 'mish': nn.Mish, 'swish': nn.SiLU}
    if name.lower() not in activations:
        raise ValueError(f"Activation {name} not supported.")
    return activations[name.lower()]

def get_loss_function(name: str, reduction='mean', gamma_focal=2, alpha=None):
    loss_map = {
        'cross_entropy': lambda: nn.CrossEntropyLoss(reduction=reduction),
        'huber': lambda: nn.HuberLoss(reduction=reduction),
        'mse': lambda: nn.MSELoss(reduction=reduction),
        'focal_loss_filter': lambda: FilterFocalLoss(gamma=gamma_focal, reduction=reduction),
        'focal_loss_slew': lambda: SlewDistanceFocalLoss(gamma=gamma_focal, reduction=reduction),
        'focal_loss': lambda: FocalLoss(gamma=gamma_focal, reduction=reduction, alpha=alpha),
    }
    if name not in loss_map:
        raise NotImplementedError(f"Loss function '{name}' is not registered.")
    return loss_map[name]()

def build_network(cfg: ExperimentConfig):
    activation_name = cfg.model.activation.lower()
    if activation_name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Activation {activation_name} not supported.")
    
    activation_fn = ACTIVATION_REGISTRY[activation_name]
    network_class = NETWORK_REGISTRY.get(cfg.model.network)
    
    if network_class is None:
        raise NotImplementedError(f"Network {cfg.model.network} not implemented.")
        
    if cfg.model.network == Network.CONTEXTUAL_SCORE_MLP:
        return network_class(
            global_dim=cfg.data.state_dim,
            bin_feat_dim=cfg.data.bin_state_dim,
            score_dim=cfg.data.num_filters,
            hidden_dim=cfg.model.hidden_dim,
            activation=activation_fn,
            nlayers=cfg.model.nlayers,
            use_contextual_gating=cfg.model.contextual_gating
        )
    return network_class() # Add kwargs for standard MLP if needed

def _build_bc_policy(cfg: ExperimentConfig, core_net: nn.Module):
    policy_class = BC_POLICY_REGISTRY.get(cfg.model.loss_strategy)
    if policy_class is None:
        raise NotImplementedError(f"`{cfg.model.loss_strategy}` strategy not implemented for BC.")
        
    # Some policies require different kwargs, handle that routing here
    if cfg.model.loss_strategy == LossStrategy.PURE_JOINT:
        return policy_class(core_net, nn.CrossEntropyLoss(), cfg.data.num_filters)
    elif cfg.model.loss_strategy == LossStrategy.PSEUDO_AUTOREGRESSIVE:
        return policy_class(core_net, cfg.data.num_filters, filter_penalty=5.0)
    elif cfg.model.loss_strategy == LossStrategy.HYBRID_MARGINAL:
        base_loss = get_loss_function(cfg.model.loss_function)
        return policy_class(core_net, cfg.data.num_filters, base_loss, base_loss, base_loss)
    
    return policy_class(core_net)

def build_algorithm(cfg: ExperimentConfig, device: torch.device):
    core_net = build_network(cfg).to(device)
    optimizer = torch.optim.Adam(core_net.parameters(), lr=cfg.train.lr_init)
    
    if cfg.model.algorithm == Algorithm.BC:
        policy = _build_bc_policy(cfg, core_net)
        return BehaviorCloning(
            policy=policy,
            optimizer=optimizer,
            lr_scheduler=cfg.train.lr_scheduler,
            lr_scheduler_kwargs=cfg.train.lr_scheduler_kwargs,
            lr_scheduler_epoch_start=cfg.train.lr_sched_epoch_start,
            lr_scheduler_num_epochs=cfg.train.lr_sched_epoch_duration,
            device=device
        )

    elif cfg.model.algorithm == Algorithm.DDQN:
        target_net = copy.deepcopy(core_net).to(device)
        policy = FlatQNetWrapper(core_net)
        target = FlatQNetWrapper(target_net)
        loss_fxn = get_loss_function(cfg.model.loss_function)
        
        return DDQN(
            policy=policy,
            target=target,
            optimizer=optimizer,
            gamma=cfg.model.gamma,  
            tau=cfg.model.tau,      
            loss_fxn=loss_fxn,
            use_double=True,
            use_cql=False,          
            device=device
        )
        
    raise ValueError(f"Algorithm {cfg.model.algorithm} unknown.")

# def build_network(cfg: ExperimentConfig):
#     """Dynamically builds the network using the registry."""
#     activation_name = cfg.model.activation.lower()
#     if activation_name not in ACTIVATION_REGISTRY:
#         raise ValueError(f"Activation {activation_name} not supported.")
    
#     activation_fn = ACTIVATION_REGISTRY[activation_name]
#     network_class = NETWORK_REGISTRY.get(cfg.model.network)
    
#     if network_class is None:
#         raise NotImplementedError(f"Network {cfg.model.network} not implemented.")
        
#     if cfg.model.network == Network.CONTEXTUAL_SCORE_MLP:
#         return network_class(
#             global_dim=cfg.data.state_dim,
#             bin_feat_dim=cfg.data.bin_state_dim,
#             score_dim=cfg.data.num_filters,
#             hidden_dim=cfg.model.hidden_dim,
#             activation=activation_fn,
#             nlayers=cfg.model.nlayers,
#             use_contextual_gating=cfg.model.contextual_gating
#         )
#     return network_class() # Add kwargs for standard MLP if needed

# def _build_bc_policy(cfg: ExperimentConfig, core_net: nn.Module):
#     """Helper to cleanly build BC policies using the registry."""
#     policy_class = BC_POLICY_REGISTRY.get(cfg.model.loss_strategy)
#     if policy_class is None:
#         raise NotImplementedError(f"`{cfg.model.loss_strategy}` strategy not implemented for BC.")
        
#     # Some policies require different kwargs, handle that routing here
#     if cfg.model.loss_strategy == LossStrategy.PURE_JOINT:
#         return policy_class(core_net, nn.CrossEntropyLoss(), cfg.data.num_filters)
#     elif cfg.model.loss_strategy == LossStrategy.PSEUDO_AUTOREGRESSIVE:
#         return policy_class(core_net, cfg.data.num_filters, filter_penalty=5.0)
#     elif cfg.model.loss_strategy == LossStrategy.HYBRID_MARGINAL:
#         base_loss = get_loss_function(cfg.model.loss_function)
#         return policy_class(core_net, cfg.data.num_filters, base_loss, base_loss, base_loss)
    
#     return policy_class(core_net)

# def build_algorithm(cfg: ExperimentConfig, device: torch.device):
#     """The main factory for your RL algorithms."""
    
#     core_net = build_network(cfg).to(device)
#     optimizer = torch.optim.Adam(core_net.parameters(), lr=cfg.train.lr_init)
    
#     if cfg.model.algorithm == Algorithm.BC:
#         policy = _build_bc_policy(cfg, core_net)
#         return BehaviorCloning(
#             policy=policy,
#             optimizer=optimizer,
#             lr_scheduler=cfg.train.lr_scheduler,
#             lr_scheduler_kwargs=cfg.train.lr_scheduler_kwargs,
#             lr_scheduler_epoch_start=cfg.train.lr_sched_epoch_start,
#             lr_scheduler_num_epochs=cfg.train.lr_sched_epoch_duration,
#             device=device
#         )

#     elif cfg.model.algorithm == Algorithm.DDQN:
#         target_net = copy.deepcopy(core_net).to(device)
#         policy = FlatQNetWrapper(core_net)
#         target = FlatQNetWrapper(target_net)
#         loss_fxn = get_loss_function(cfg.model.loss_function)
        
#         return DDQN(
#             policy=policy,
#             target=target,
#             optimizer=optimizer,
#             gamma=cfg.model.gamma,  
#             tau=cfg.model.tau,      
#             loss_fxn=loss_fxn,
#             use_double=True,
#             use_cql=False,          
#             device=device
#         )
        
#     raise ValueError(f"Algorithm {cfg.model.algorithm} unknown.")

# def get_loss_function(name, reduction='mean', gamma_focal=2, alpha=None):
#     if name == 'cross_entropy':
#         loss_function = nn.CrossEntropyLoss(reduction=reduction)
#     elif name == 'focal_loss_filter':
#         loss_function = FilterFocalLoss(gamma=gamma_focal, reduction=reduction)
#     elif name == 'focal_loss_slew':
#         loss_function = SlewDistanceFocalLoss(gamma=gamma_focal, reduction=reduction)
#     elif name == 'focal_loss':
#         loss_function = FocalLoss(gamma=gamma_focal, reduction=reduction, alpha=alpha)
#     else:
#         raise NotImplementedError
#     return loss_function

# def build_network(cfg: ExperimentConfig):
#     """Builds the core network based on the Pydantic config."""
#     activation_fn = get_activation(cfg.model.activation)
    
#     if cfg.model.network == Network.CONTEXTUAL_SCORE_MLP:
#         return ContextualScoreMLP(
#             global_dim=cfg.data.state_dim,
#             bin_feat_dim=cfg.data.bin_state_dim,
#             score_dim=cfg.data.num_filters,
#             hidden_dim=cfg.model.hidden_dim,
#             activation=activation_fn,
#             nlayers=cfg.model.nlayers,
#             use_contextual_gating=cfg.model.contextual_gating
#         )
#     # Add other network architectures here (e.g., AUTOREGRESSIVE)
#     else:
#         raise NotImplementedError(f"Network {cfg.model.network} not implemented.")

# def build_algorithm(cfg: ExperimentConfig, device: torch.device):
#     """Builds the RL algorithm, branching safely based on the model schema."""
    
#     core_net = build_network(cfg).to(device)
#     optimizer = torch.optim.Adam(core_net.parameters(), lr=cfg.train.lr_init)
#     loss_function = get_loss_function(cfg.model.loss_function, reduction='mean', gamma_focal=2, alpha=None)
    
#     if cfg.model.algorithm == Algorithm.BC:
#         # Note: cfg.model is guaranteed to be a BCModelConfig here
        
#         if cfg.model.loss_strategy == LossStrategy.PURE_JOINT:
#             ce_loss_function = nn.CrossEntropyLoss()
#             policy = PureJointPolicy(core_net, ce_loss_function, cfg.data.num_filters)
            
#         elif cfg.model.loss_strategy == LossStrategy.PSEUDO_AUTOREGRESSIVE:
#             policy = PseudoAutoregressivePolicy(
#                 core_net=core_net,
#                 num_filters=cfg.data.num_filters,
#                 filter_penalty=5.0 # If this should be configurable, add to BCModelConfig
#             )
#         elif cfg.model.loss_strategy == LossStrategy.HYBRID_MARGINAL:
#             policy = HybridMarginalPolicy(
#                 core_net=core_net,
#                 num_filters=cfg.data.num_filters,
#                 bin_loss_function=loss_function,
#                 filter_loss_function=loss_function,
#                 joint_loss_function=loss_function
#             )
#         else:
#              raise NotImplementedError(f"`{cfg.model.loss_strategy}` strategy not implemented for BC.")

#         return BehaviorCloning(
#             policy=policy,
#             optimizer=optimizer,
#             lr_scheduler=cfg.train.lr_scheduler,
#             lr_scheduler_kwargs=cfg.train.lr_scheduler_kwargs,
#             lr_scheduler_epoch_start=cfg.train.lr_sched_epoch_start,
#             lr_scheduler_num_epochs=cfg.train.lr_sched_epoch_duration,
#             device=device
#         )

#     elif cfg.model.algorithm == Algorithm.DDQN:
#         # Note: cfg.model is guaranteed to be a DDQNModelConfig here
        
#         target_net = copy.deepcopy(core_net).to(device)
#         policy = FlatQNetWrapper(core_net)
#         target = FlatQNetWrapper(target_net)

#         loss_fxn = nn.HuberLoss(reduction='mean') if cfg.model.loss_function == 'huber' else nn.MSELoss(reduction='mean')
        
#         return DDQN(
#             policy=policy,
#             target=target,
#             optimizer=optimizer,
#             gamma=cfg.model.gamma,  # Safely accessed because we know it's a DDQN model
#             tau=cfg.model.tau,      # Safely accessed because we know it's a DDQN model
#             loss_fxn=loss_fxn,
#             use_double=True,
#             use_cql=False,          # If you add CQL later, add CQLModelConfig to your schema union
#             device=device
#         )
        
#     else:
#         raise ValueError(f"Algorithm {cfg.model.algorithm} unknown.")