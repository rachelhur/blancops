import json
from typing import Tuple
import torch
import yaml
from pathlib import Path

# Import your domain-specific modules
from blancops.configs.constants import WORKSPACE
from blancops.configs.enums import Algorithm
from blancops.configs.rl_schema import ExperimentConfig, load_and_validate
from blancops.rl.registry import _build_bc_policy, _build_q_adapter, build_network
from blancops.rl.agent import Agent

import logging
logger = logging.getLogger(__name__)

from typing import Tuple

class AgentFactory:
    def __init__(self, base_model_dir: str = WORKSPACE / "deployable_models"):
        """Factory for building scheduling agents
        
        Args:
            base_model_dir (str, optional): _description_. Defaults to WORKSPACE / "deployable_models".
        """
        self.base_dir = Path(base_model_dir)
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
        
        # If model_path_or_alias is an absolute path or a path that exists, prefer that
        if isinstance(model_path_or_alias, (str,)) and Path(model_path_or_alias).is_absolute() and Path(model_path_or_alias).exists():
            model_dir = Path(model_path_or_alias)
        elif isinstance(model_path_or_alias, Path) and model_path_or_alias.exists():
            model_dir = model_path_or_alias
        else:
            model_dir = self.resolve_model_dir(model_path_or_alias)

        config_path = model_dir / "resolved_config.yaml"
        if not config_path.exists():
            config_path = model_dir / "configs" / "resolved_config.yaml"
            
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find resolved_config.yaml in {model_dir} or {model_dir}/configs/"
            )
            
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
        elif cfg.model.algorithm in (Algorithm.DDQN, Algorithm.CQL, Algorithm.IQL):
            # For IQL, algorithm.policy is the policy_net (QFlatPolicy), not the Q-adapter.
            policy = _build_q_adapter(cfg, core_net)
        
        try:
            checkpoint = torch.load(weights_path, map_location=device)
        except Exception as e:
            logger.warning(f"torch.load failed with default settings when loading policy: {e}. Retrying with weights_only=False.")
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and ('model_state_dict' in checkpoint or 'policy_state_dict' in checkpoint or 'state_dict' in checkpoint):
            # Support multiple checkpoint key names
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('policy_state_dict') or checkpoint.get('state_dict')
            policy.load_state_dict(state_dict)
            norm_stats = checkpoint.get('norm_stats', {})
        else:
            # If checkpoint is raw state dict
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