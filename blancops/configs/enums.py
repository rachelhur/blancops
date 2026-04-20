from enum import Enum

class Algorithm(str, Enum):
    BC = "bc"
    DQN = "dqn"
    DDQN = "ddqn"
    CQL = "cql"

class LossStrategy(str, Enum):
    PURE_JOINT = "pure_joint"
    HYBRID_MARGINAL = "hybrid_marginal"
    PSEUDO_AUTOREGRESSIVE = "pseudo_ar"
    AUTOREGRESSIVE = "autoregressive"
    # MARGINAL = "marginal"
    
class Network(str, Enum):
    MLP = "mlp"
    CONTEXTUAL_SCORE_MLP = "context_score_mlp"
    MULTI_HEAD_MLP = "multi_head_mlp"
    AUTOREGRESSIVE = "autoregressive"

class Reward(str, Enum):
    EXPERT_ACTION = "expert_action"
    SURVEY_UNIFORMITY = "survey_uniformity"

class Lookup(str, Enum):
    FIELD2NAME = "field2name"
    FIELD2RADEC = "field2radec"
    FIELD2MAXVISITS_TRAIN = "field2maxvisits_train"
    NIGHT2FIELDVISITS = "night2fieldvisits"
    NIGHT2FILTERVISITS = "night2filtervisits"
    FIELDFILTER2MAXVISITS = "fieldfilter2maxvisits"
    FILTER_TARGET_COUNTS = "target_counts_per_filter"