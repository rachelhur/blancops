import json
from typing import Tuple
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

from typing import Tuple

class AgentFactory:
    def __init__(self, base_model_dir: str = None):
        self.base_dir = Path(base_model_dir or WORKSPACE / "deployable_models")
        self.alias_file = self.base_dir / "aliases.yaml"
        self.aliases = self._load_aliases()

    def build_agent(
        self, 
        model_path_or_alias: str, 
        lookups: Path, 
        field_choice_method: str,
        device: str = 'cpu',
        weights_filename: str = None # Now defaults to None for auto-detection
    ) -> Tuple[Agent, ExperimentConfig, dict]:
        
        model_dir = self.resolve_model_dir(model_path_or_alias)
        
        config_path = model_dir / "resolved_config.yaml"
        if not config_path.exists():
            config_path = model_dir / "configs" / "resolved_config.yaml"
            
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find resolved_config.yaml in {model_dir} or {model_dir}/configs/"
            )
            
        config_path = model_dir / "resolved_config.yaml"
        cfg = load_and_validate(config_path)
        
        # 1. Resolve which weights file to actually use
        weights_path = self._resolve_weights_path(model_dir, weights_filename)
        
        # 2. Load the policy
        loaded_policy, norm_stats = self.load_policy(weights_path, cfg, device)

        agent = Agent(
            policy=loaded_policy,
            cfg=cfg, 
            lookups=lookups, 
            field_choice_method=field_choice_method
        )
        
        return agent, cfg, norm_stats

    def _resolve_weights_path(self, model_dir: Path, filename: str = None) -> Path:
        """Logic to find the best weights file automatically."""
        # If user explicitly asked for a file, give it to them
        if filename:
            # Check root (deployment) or checkpoints/ (training)
            path = model_dir / filename if (model_dir / filename).exists() else model_dir / "checkpoints" / filename
            return path

        # Check for deployment model first (Standard for live scheduling)
        if (model_dir / "model.pt").exists():
            return model_dir / "model.pt"

        # Auto-detect best from training history (Standard for validation)
        history_file = model_dir / "checkpoints" / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            if history:
                # Top-K logic ensures index 0 is the best
                best_path = Path(history[0]['filepath'])
                logger.info(f"Auto-detected best weights: {best_path.name}")
                return best_path

        # Ultimate fallback
        fallback = model_dir / "checkpoints" / "latest_checkpoint.pt"
        if fallback.exists():
            return fallback
            
        raise FileNotFoundError(f"Could not find any valid weights in {model_dir}")

    @staticmethod
    def load_policy(weights_path: Path, cfg: ExperimentConfig, device: str) -> Tuple[torch.nn.Module, dict]:
        core_net = build_network(cfg)
        
        if cfg.model.algorithm == Algorithm.BC:
            policy = _build_bc_policy(cfg, core_net)
        elif cfg.model.algorithm == Algorithm.DDQN:
            policy = FlatQNetWrapper(core_net)
        
        checkpoint = torch.load(weights_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['model_state_dict'])
            norm_stats = checkpoint.get('norm_stats', {})
        else:
            policy.load_state_dict(checkpoint)
            norm_stats = {}
            
        policy.eval()
        return policy.to(device), norm_stats
    
    def _load_aliases(self) -> dict:
        if self.alias_file.exists():
            with open(self.alias_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def resolve_model_dir(self, model_path_or_alias: str) -> Path:
        target_directory_name = self.aliases.get(model_path_or_alias, model_path_or_alias)
        model_path = self.base_dir / target_directory_name
        
        if not model_path.is_dir():
            raise FileNotFoundError(f"Resolved model directory not found: {model_path}")
        return model_path