from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MLflowArguments:
    """
    Arguments controlling experiment tracking. MLflow (activated via "mlflow"
    in --report_to) tracks configs and training/eval metrics. GPU utilization
    and DeepSpeed communication timing are infra/perf metrics rather than
    experiment results, so they're written to S3 as CSV instead, for Grafana
    to read (see PerfMetricsS3Callback in internvl.utils.mlflow_utils).
    """
    mlflow_tracking_uri: Optional[str] = field(
        default=None,
        metadata={'help': 'MLflow tracking URI. Defaults to a local file store under '
                          '<output_dir>/mlruns if unset.'},
    )
    mlflow_experiment_name: str = field(
        default='goal_predictor',
        metadata={'help': 'MLflow experiment name.'},
    )
    mlflow_run_name: Optional[str] = field(
        default=None,
        metadata={'help': 'MLflow run name. Defaults to TrainingArguments.run_name (output_dir).'},
    )
    log_perf_metrics: bool = field(
        default=True,
        metadata={'help': 'Whether to collect GPU utilization (NVML) and DeepSpeed '
                          'communication timer breakdown and upload them to S3 for Grafana.'},
    )
    s3_bucket: Optional[str] = field(
        default='internvl-tracking-gpu-2',
        metadata={'help': 'S3 bucket perf metrics CSV is uploaded to. AWS credentials are '
                          'read from the environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY '
                          '/ AWS_DEFAULT_REGION) -- never hardcode them.'},
    )
    s3_prefix: str = field(
        default='metrics',
        metadata={'help': 'Key prefix perf metrics are stored under: '
                          '<s3_prefix>/<run_name>/perf_metrics.csv'},
    )
    metrics_log_interval: int = field(
        default=50,
        metadata={'help': 'Collect and upload GPU/DeepSpeed perf metrics every N training steps.'},
    )
