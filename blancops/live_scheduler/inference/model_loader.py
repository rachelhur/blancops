from typing import Tuple
from pydantic import BaseModel
import torch
import yaml
from pathlib import Path

# Import your domain-specific modules
from blancops.configs.constants import WORKSPACE
from blancops.configs.enums import Algorithm
from blancops.configs.schema import ExperimentConfig, load_and_validate
from blancops.rl.policies.policies import FlatQNetWrapper
from blancops.rl.registry import _build_bc_policy, build_algorithm, build_network
from blancops.rl.agent import Agent

import logging
logger = logging.getLogger(__name__)

class DeploymentAgentLoader:
    def __init__(self, base_model_dir: str = None):
        
        self.base_dir = Path(
            base_model_dir or WORKSPACE / "deployable_models"
            )
        self.alias_file = self.base_dir / "aliases.yaml"
        self.aliases = self._load_aliases()

    def _load_aliases(self) -> dict:
        if self.alias_file.exists():
            with open(self.alias_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def resolve_model_dir(self, model_name_or_alias: str) -> Path:
        target_directory_name = self.aliases.get(model_name_or_alias, model_name_or_alias)
        model_path = self.base_dir / target_directory_name
        
        if not model_path.is_dir():
            raise FileNotFoundError(f"Resolved model directory not found: {model_path}")
        return model_path
    def load_policy(self, model_dir: Path, cfg: ExperimentConfig, device: str) -> torch.nn.Module:
        core_net = build_network(cfg)
        
        if cfg.model.algorithm == Algorithm.BC:
            policy = _build_bc_policy(cfg, core_net)
        elif cfg.model.algorithm == Algorithm.DDQN:
            policy = FlatQNetWrapper(core_net)
            
        weights_path = model_dir / "model.pt"
        state_dict = torch.load(weights_path, map_location=device)
        policy.load_state_dict(state_dict, strict=True)
        
        policy.eval()
        return policy.to(device)

    def build_agent(
        self, 
        model_path_or_alias: str, 
        lookups: Path, 
        field_choice_method: str,
        device: str = 'cpu'
    ) -> Tuple[Agent, BaseModel]:
        
        model_dir = self.resolve_model_dir(model_path_or_alias)
        logger.info(f"[Model] Building Agent for deployment from: {model_dir}")
        
        config_path = model_dir / "resolved_config.yaml"
        weights_path = model_dir / "best_weights.pt" # Updated based on your snippet

        cfg = load_and_validate(config_path)
        
        loaded_policy = self.load_policy(model_dir, cfg, device)

        agent = Agent(
            policy=loaded_policy,
            cfg=cfg, 
            lookups=lookups, 
            field_choice_method=field_choice_method
        )
        
        logger.info("[Model] Agent successfully built and ready for scheduling.")
        return agent, cfg