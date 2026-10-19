from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GoalPredictorArguments:
    """
    Arguments for constructing MLLMGoalPredictor. Only used when training a
    predictor from scratch (model_args.load_checkpoint=False); a resumed run
    reconstructs the predictor from the config saved in its checkpoint.
    """
    mllm_hidden_dim: Optional[int] = field(
        default=None,
        metadata={'help': "Hidden size of the MLLM backbone. Defaults to the loaded "
                          "model's llm_config.hidden_size if unset."},
    )
    attn_dim: int = field(
        default=256,
        metadata={'help': 'Attention dimension for the section-aware encoder.'},
    )
    num_sections: int = field(
        default=4,
        metadata={'help': 'Number of semantic sections in the section-aware encoder.'},
    )
    map_size: int = field(
        default=224,
        metadata={'help': 'Raster map size (pixels) the predicted goal is expressed in.'},
    )
    dropout: float = field(
        default=0.1,
        metadata={'help': 'Dropout rate used throughout the goal predictor.'},
    )
    ablate_mllm: bool = field(
        default=False,
        metadata={'help': 'Diagnostic: skip the MLLM and feed zeros in place of its features, '
                          'so the heads predict from the numeric ego history alone.'},
    )
    goal_predictor_type: str = field(
        default='regression',
        metadata={'help': "'regression' (MLLMGoalPredictor, scalar x/y heads) or 'heatmap' "
                          "(MLLMHeatmapGoalPredictor, goal-cell classification + in-cell offset)."},
    )
    num_goal_queries: int = field(
        default=8,
        metadata={'help': 'heatmap only: number of learnable <GOAL> query tokens appended to the prompt.'},
    )
    llm_tap_layers: str = field(
        default='-1,-5,-9,-13',
        metadata={'help': 'heatmap only: comma-separated indices into the LLM hidden_states tuple, '
                          'mixed with learned weights.'},
    )
    heatmap_size: int = field(
        default=56,
        metadata={'help': 'heatmap only: goal grid resolution G (7 * 2**n); cells are 224 / G pixels.'},
    )
    heatmap_sigma: float = field(
        default=1.0,
        metadata={'help': 'heatmap only: std (in cells) of the Gaussian soft target for the goal-cell CE.'},
    )
    offset_loss_weight: float = field(
        default=1.0,
        metadata={'help': 'heatmap only: weight of the in-cell offset SmoothL1 loss.'},
    )
    hist_dropout: float = field(
        default=0.3,
        metadata={'help': 'heatmap only: per-sample probability of replacing the ego-history features '
                          'with a learned "missing" embedding during training.'},
    )
