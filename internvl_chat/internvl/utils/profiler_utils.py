# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
from contextlib import contextmanager

import torch
from transformers import TrainerCallback


class ProfilerCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()


@contextmanager
def build_profiler(profiler_args, local_rank):
    """Context manager yielding a configured torch profiler, or None if disabled."""
    if not profiler_args.enable_profiler:
        yield None
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiler_output_dir = os.path.join(profiler_args.profiler_output_dir, f'rank{local_rank}')
    os.makedirs(profiler_output_dir, exist_ok=True)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=profiler_args.profiler_wait_steps,
            warmup=profiler_args.profiler_warmup_steps,
            active=profiler_args.profiler_active_steps,
            repeat=profiler_args.profiler_repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
        record_shapes=profiler_args.profiler_record_shapes,
        profile_memory=profiler_args.profiler_profile_memory,
        with_stack=profiler_args.profiler_with_stack,
    ) as prof:
        yield prof
