from dataclasses import dataclass, field


@dataclass
class ProfilerArguments:
    """
    Arguments controlling torch.profiler tracing of the training loop.
    Disabled by default; enable with --enable_profiler true.
    """
    enable_profiler: bool = field(
        default=False,
        metadata={'help': 'Whether to profile training with torch.profiler.'},
    )
    profiler_output_dir: str = field(
        default='./prof_trace',
        metadata={'help': 'Directory to write profiler traces (viewable in TensorBoard).'},
    )
    profiler_wait_steps: int = field(
        default=1,
        metadata={'help': 'Number of steps the profiler skips before warmup.'},
    )
    profiler_warmup_steps: int = field(
        default=1,
        metadata={'help': 'Number of warmup steps before active tracing starts.'},
    )
    profiler_active_steps: int = field(
        default=3,
        metadata={'help': 'Number of steps actively traced per cycle.'},
    )
    profiler_repeat: int = field(
        default=1,
        metadata={'help': 'Number of wait/warmup/active cycles to repeat. 0 repeats indefinitely.'},
    )
    profiler_record_shapes: bool = field(
        default=False,
        metadata={'help': 'Whether to record tensor shapes in the profiler trace.'},
    )
    profiler_profile_memory: bool = field(
        default=True,
        metadata={'help': 'Whether to track tensor memory allocation in the profiler trace.'},
    )
    profiler_with_stack: bool = field(
        default=False,
        metadata={'help': 'Whether to record source code information (file and line) in the profiler trace.'},
    )
