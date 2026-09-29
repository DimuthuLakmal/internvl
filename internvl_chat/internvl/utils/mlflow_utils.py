# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import csv
import io
import logging
import os
from datetime import datetime, timezone

from transformers import TrainerCallback


logger = logging.getLogger(__name__)

_MAX_PARAM_VALUE_LEN = 250
_PARAMS_PER_BATCH = 100


def setup_mlflow_env(mlflow_args, training_args) -> None:
    """Point transformers' built-in MLflowCallback at the right tracking
    store/experiment/run. Must run before Trainer.train() so the callback
    picks these up in on_train_begin(). Credentials (MLFLOW_TRACKING_USERNAME/
    PASSWORD) are expected to already be set in the environment (e.g. sourced
    from an untracked ~/.dagshub_env) — never hardcode them here."""
    tracking_uri = mlflow_args.mlflow_tracking_uri or f'file:{os.path.join(training_args.output_dir, "mlruns")}'
    os.environ.setdefault('MLFLOW_TRACKING_URI', tracking_uri)
    os.environ.setdefault('MLFLOW_EXPERIMENT_NAME', mlflow_args.mlflow_experiment_name)
    if mlflow_args.mlflow_run_name:
        training_args.run_name = mlflow_args.mlflow_run_name


def _flatten_dataclass(obj, prefix=''):
    params = {}
    for key, value in vars(obj).items():
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            value = str(value)
        if isinstance(value, str) and len(value) > _MAX_PARAM_VALUE_LEN:
            value = value[:_MAX_PARAM_VALUE_LEN - 3] + '...'
        params[f'{prefix}{key}'] = value
    return params


class MLflowConfigCallback(TrainerCallback):
    """Logs the non-TrainingArguments configs (model/data/goal-predictor/etc.)
    to the active MLflow run. transformers' own MLflowCallback already logs
    TrainingArguments; this fills in the rest so full run config is captured."""

    def __init__(self, **named_dataclasses):
        self.named_dataclasses = named_dataclasses
        self._logged = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self._logged:
            return
        try:
            import mlflow
        except ImportError:
            logger.warning('mlflow is not installed; skipping extra config logging.')
            return
        if mlflow.active_run() is None:
            return

        params = {}
        for prefix, dataclass_obj in self.named_dataclasses.items():
            params.update(_flatten_dataclass(dataclass_obj, prefix=f'{prefix}.'))

        items = list(params.items())
        for i in range(0, len(items), _PARAMS_PER_BATCH):
            try:
                mlflow.log_params(dict(items[i:i + _PARAMS_PER_BATCH]))
            except Exception as exc:
                logger.warning(f'Failed to log MLflow params batch: {exc}')

        self._logged = True


class PerfMetricsS3Callback(TrainerCallback):
    """Collects per-GPU NVML utilization/memory/power and DeepSpeed wall-clock
    timer breakdown (forward/backward/communication/step) every `log_interval`
    steps, and periodically re-uploads them as one long-format CSV to S3 so
    Grafana (via an S3 CSV/JSON datasource plugin) can visualize them.

    These are infra/perf metrics, not experiment results, so they're kept out
    of MLflow. Only rank 0 collects and uploads; assumes a single-node job so
    NVML on rank 0 can see every GPU regardless of CUDA_VISIBLE_DEVICES.

    AWS credentials are read by boto3 from the environment (AWS_ACCESS_KEY_ID
    / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION) — never hardcode them here.
    """

    CSV_HEADER = ['timestamp', 'run_id', 'global_step', 'metric_name', 'metric_value', 'gpu_index']

    def __init__(self, trainer, bucket: str, prefix: str = 'metrics', log_interval: int = 50):
        self.trainer = trainer
        self.bucket = bucket
        self.prefix = prefix
        self.log_interval = log_interval
        self.run_id = None
        self.s3_key = None
        self._pynvml = None
        self._handles = []
        self._s3 = None
        self._rows = []
        self._deepspeed_warned = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self.run_id = args.run_name or os.path.basename(args.output_dir.rstrip('/'))
        self.s3_key = f'perf_metrics.csv'

        try:
            import boto3
            self._s3 = boto3.client('s3')
        except Exception as exc:
            logger.warning(f'Failed to init boto3 S3 client, disabling perf-metrics upload: {exc}')
            self._s3 = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as exc:
            logger.warning(f'NVML init failed, disabling GPU metrics: {exc}')
            self._pynvml = None

    def _get_timer_dict(self):
        engine = getattr(self.trainer, 'deepspeed', None) or getattr(self.trainer, 'model_wrapped', None)
        timers = getattr(engine, 'timers', None)
        return getattr(timers, 'timers', None)

    def _collect_gpu_rows(self, timestamp, step):
        if self._pynvml is None:
            return
        pynvml = self._pynvml
        for idx, handle in enumerate(self._handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._rows.append([timestamp, self.run_id, step, 'gpu.utilization_pct', float(util.gpu), idx])
                self._rows.append([timestamp, self.run_id, step, 'gpu.mem_utilization_pct', float(util.memory), idx])
                self._rows.append([timestamp, self.run_id, step, 'gpu.mem_used_gb', mem.used / (1024 ** 3), idx])
                self._rows.append([timestamp, self.run_id, step, 'gpu.mem_total_gb', mem.total / (1024 ** 3), idx])
            except Exception as exc:
                logger.warning(f'Failed to read NVML utilization/memory for GPU {idx}: {exc}')
                continue
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                self._rows.append([timestamp, self.run_id, step, 'gpu.power_w', power_w, idx])
            except Exception:
                pass
            try:
                temp_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                self._rows.append([timestamp, self.run_id, step, 'gpu.temperature_c', temp_c, idx])
            except Exception:
                pass

    def _collect_deepspeed_rows(self, timestamp, step):
        timer_dict = self._get_timer_dict()
        if not timer_dict:
            if not self._deepspeed_warned:
                logger.warning('DeepSpeed wall-clock timers not available; set '
                               '"wall_clock_breakdown": true in the DeepSpeed config '
                               'to enable communication-time logging.')
                self._deepspeed_warned = True
            return

        for name, timer in timer_dict.items():
            try:
                elapsed_ms = timer.elapsed(reset=True)
                if elapsed_ms:
                    self._rows.append([timestamp, self.run_id, step, f'deepspeed.{name}_ms', float(elapsed_ms), None])
            except Exception:
                continue

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step == 0 or state.global_step % self.log_interval != 0:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        self._collect_gpu_rows(timestamp, state.global_step)
        self._collect_deepspeed_rows(timestamp, state.global_step)
        self._upload()

    def _upload(self):
        if self._s3 is None or not self._rows:
            return
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.CSV_HEADER)
        writer.writerows(self._rows)
        try:
            self._s3.put_object(Bucket=self.bucket, Key=self.s3_key,
                                Body=buffer.getvalue().encode('utf-8'), ContentType='text/csv')
        except Exception as exc:
            logger.warning(f'Failed to upload perf metrics CSV to s3://{self.bucket}/{self.s3_key}: {exc}')

    def on_train_end(self, args, state, control, **kwargs):
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
