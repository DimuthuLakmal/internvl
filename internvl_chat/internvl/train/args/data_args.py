from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )
