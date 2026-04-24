import datetime

from pydantic import BaseModel, Field, computed_field, model_validator
import yaml
from pathlib import Path
from typing import Any, List, Union, Literal, Dict
import numpy as np

from typing import Optional
from blancops.configs.enums import *
from blancops.configs.constants import DEFAULT_NORM_MAPPING, TRAIN_DATA_PATH, BIN_FEATURES
from blancops.data.constants import FILTER2IDX

from blancops.configs.constants import ALLOWED_NORMS_PER_FEATURE, NORM_TYPES

class EnvConfig(BaseModel):
    sun_el_lim: float = -12
    airmass_lim: float = 2.0
    
    
    
class NormalizationConfig(BaseModel):
    feature_mappings: Dict[str, List[NORM_TYPES]] = Field(
        default_factory=lambda: {k: v.copy() for k, v in DEFAULT_NORM_MAPPING.items()})
    fix_nans: bool = True

    @model_validator(mode='after')
    def validate_legal_normalizations(self) -> 'NormalizationConfig':
        for feature, requested_norms in self.feature_mappings.items():
            allowed_norms = ALLOWED_NORMS_PER_FEATURE.get(feature)
            if allowed_norms is None:
                raise ValueError(f"Feature '{feature}' is not recognized in allowed normalization rules.")
                
            if not set(requested_norms).issubset(allowed_norms): # allowed_norms is already a set!
                raise ValueError(
                    f"Normalization {requested_norms} is not allowed for feature '{feature}'. "
                    f"Allowed: {list(allowed_norms)}"
                )
        return self
    
class BaseDataConfig(BaseModel):
    name: str = 'des-data-v0'
    path: str = str(TRAIN_DATA_PATH)
    # cache_in_memory: bool = False
    
    # Data configuration
    nside: int = 16
    action_space: str
    
    # Normalization configuration
    norm: NormalizationConfig = Field(default_factory=NormalizationConfig)
    
    # Configurations calculated after data processing (required for model instantiation)
    state_dim: Optional[int] = None
    bin_state_dim: Optional[int] = None
    num_bins: Optional[int] = None
    num_filters: Optional[int] = None
    num_actions: Optional[int] = None

    # Features
    global_features: List[str]
    bin_features: List[str]
    
    @model_validator(mode='after')
    def validate_features(self) -> 'TrainDataConfig':
        for bin_feat in self.bin_features:
            if bin_feat not in BIN_FEATURES:
                raise ValueError(f"{bin_feat} is not implemented.")
        return self
      
class TrainDataConfig(BaseDataConfig):
    years: List[int] = [2013, 2014, 2015, 2016, 2017, 2018, 2019] # full set of data
    months: List[int] = [i+1 for i in range(12)]
    days: List[int] = [i+1 for i in range(31)]
    filters: List[str] = [filt for filt in FILTER2IDX.keys()]
    
    # Configurations required for validation
    train_nights: Optional[List[str]] = None
    val_nights: Optional[List[str]] = None
    train_val_split: float  = 0.9
    
class BaseModelConfig(BaseModel):
    network: Network = Network.CONTEXTUAL_SCORE_MLP
    loss_strategy: LossStrategy = LossStrategy.PURE_JOINT
    hidden_dim: int = 128
    nlayers: int = 4
    loss_function: str
    contextual_gating: bool = False
    activation: str = "relu"
    
class BCModelConfig(BaseModelConfig):
    algorithm: Literal[Algorithm.BC]

class DDQNModelConfig(BaseModelConfig):
    algorithm: Literal[Algorithm.DDQN]
    tau: float = 0.005 # DDQN specific parameter
    gamma: float = 0.99 # DDQN specific parameter
    
class RewardConfig(BaseModel):
    reward_choice: str = 'expert_actions'
    
class TrainConfig(BaseModel):
    max_epochs: int = 50
    batch_size: int   = 1024
    lr_scheduler: str = "cosine_annealing"
    num_workers: int = 0
    lr_init: float = .001
    lr_final: float = 1e-5
    lr_sched_epoch_start: int = 10
    lr_sched_epoch_duration: int = 30
    patience: int = 20
    device:         str   = "cuda"
    seed:           int   = 42
    
    lr_scheduler_kwargs: Optional[dict] = None
    
    @model_validator(mode='after')
    def validate_lr_scheduler(self):
        # This will now run automatically when model_copy adds the kwargs
        if self.lr_scheduler_kwargs:
            assert self.max_epochs - self.lr_sched_epoch_start - self.lr_sched_epoch_duration >= 0, "The number of epochs must be greater than lr_scheduler_epoch_start + lr_scheduler_dur_epochs"
        return self
    
AnyModelConfig = Union[BCModelConfig, DDQNModelConfig]

class ExperimentConfig(BaseModel):
    experiment_name: str
    parent_dir: str = "experiments/"
    outdir: Optional[str] = None
    data: TrainDataConfig
    model: AnyModelConfig = Field(discriminator="algorithm")
    reward: RewardConfig
    train: TrainConfig
    device: str = 'cuda'

    @model_validator(mode='before')
    @classmethod
    def set_outdir(cls, data: Any) -> Any:
        """Intercepts the raw dictionary to compute outdir before validation."""
        if isinstance(data, dict) and data.get('outdir') is None:
            exp_name = data.get('experiment_name')
            parent = data.get('parent_dir', 'experiments/')
            
            if exp_name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                data['outdir'] = str(Path(parent) / exp_name / f"run_{timestamp}")
                
                data['outdir'] = str(Path(parent) / exp_name)
                    
        return data # Return the modified dictionary
        
    # @model_validator(mode='before')
    # @classmethod
    # def set_outdir(self) -> 'ExperimentConfig':
    #     if self.outdir is None:
    #         self.outdir = str(Path(self.parent_dir) / self.experiment_name)
    #     return self
    
class ScheduleConfig(BaseModel):
    model_dir: str
    data: BaseDataConfig

def load_and_validate(yaml_path: str | Path) -> ExperimentConfig:
    """Loads the YAML config and validates it against the ExperimentConfig schema."""
    with open(yaml_path, "r") as f:
        raw_data = yaml.safe_load(f)
    return ExperimentConfig(**raw_data) # Added dictionary unpacking

def resolve_and_save(cfg: ExperimentConfig, dataset_dims: dict, dataset_feature_names: dict, lr_scheduler_kwargs: dict, val_nights: List[str], outdir: str | Path) -> ExperimentConfig:
    """Resolves config by filling in fields calculated after data processing. Saves the resolved config to the output directory."""
    # UPDATE CONFIG.DATA
    data_updates = {
        "state_dim": int(dataset_dims['state_dim']),
        "bin_state_dim": int(dataset_dims['bin_state_dim']),
        "num_bins": int(dataset_dims['num_bins']),
        "num_filters": int(dataset_dims['num_filters']),
        "num_actions": int(dataset_dims['num_actions']),
        "global_features": dataset_feature_names['global_features'],
        "bin_features": dataset_feature_names['bin_features'],
        "val_nights": val_nights
    }
    updated_data = cfg.data.model_copy(update=data_updates)
    # UPDATE CONFIG.TRAIN
    train_updates = {"lr_scheduler_kwargs": lr_scheduler_kwargs}
    updated_train = cfg.train.model_copy(update=train_updates)
    
    # COPY CONFIG WITH UPDATES
    resolved_cfg = cfg.model_copy(update={
        "data": updated_data,
        "train": updated_train
    })
    
    # CONSTRUCT EXPERIMENT_OUTDIR CONFIG FIELD AND SAVE RESOLVED CONFIG
    if resolved_cfg.outdir is None:
        resolved_cfg.outdir = str(Path(resolved_cfg.outdir))
    Path(Path(resolved_cfg.outdir) / "configs" ).mkdir(parents=True, exist_ok=True)
    with open(Path(resolved_cfg.outdir) / "configs" /"resolved_config.yaml", "w") as f:
        # Use mode='json' to force Pydantic to convert complex types (like Enums) to strings
        resolved_dict = resolved_cfg.model_dump(mode='json') 
        yaml.dump(resolved_dict, f, default_flow_style=False, sort_keys=False)
        
    return resolved_cfg
