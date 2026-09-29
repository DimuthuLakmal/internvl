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
