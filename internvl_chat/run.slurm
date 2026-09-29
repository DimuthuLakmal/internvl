#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Jul 10 2024 23:21:58 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=feit-gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="ctg"

# The project ID which this job should run under:
#SBATCH --account="punim2030"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

# Number of GPUs requested per node:
#SBATCH --gres=gpu:4
# Slurm QoS:
#SBATCH --qos=feit

# The amount of memory in megabytes per node:
#SBATCH --mem=160000

# Use this email address:
#SBATCH --mail-user=dkariyawasan@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=2-0:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load Anaconda3/2022.10

# The job command(s):
module load CUDA/11.8.0; eval "$(conda shell.bash hook)"; conda activate llm
export PYTHONPATH="$PYTHONPATH:/data/gpfs/projects/punim2030/mllm/InternVL/internvl_chat";export CUDA_HOME="/apps/easybuild-2022/easybuild/software/Core/CUDA/12.4.1";export CUDA_VISIBLE_DEVICES="0,1,2,3";export MASTER_PORT="34229";export LAUNCHER="pytorch"

# DagsHub MLflow auth: keep the token out of this tracked script. Create
# ~/.dagshub_env (outside the repo) containing:
#   export MLFLOW_TRACKING_USERNAME="DimuthuLakmal"
#   export MLFLOW_TRACKING_PASSWORD="<dagshub-token>"   # https://dagshub.com/user/settings/tokens
[ -f ~/.dagshub_env ] && source ~/.dagshub_env

# AWS credentials for perf-metrics (GPU/DeepSpeed) upload to S3 for Grafana.
# Create ~/.aws_env (outside the repo) containing:
#   export AWS_ACCESS_KEY_ID="<access-key>"
#   export AWS_SECRET_ACCESS_KEY="<secret-key>"
#   export AWS_DEFAULT_REGION="<bucket-region>"
[ -f ~/.aws_env ] && source ~/.aws_env

torchrun --nproc_per_node 4 internvl/train/internvl_chat_pretrain_target.py --model_name_or_path ./results/results_pretrained_stage2_logits --conv_style internvl2_5 --use_fast_tokenizer False --output_dir ./results/results_pretrained_1b_stage2_target --meta_path ./data/meta_1b_target_pred.json --overwrite_output_dir True --force_image_size 224 --down_sample_ratio 0.5 --drop_path_rate 0.1 --min_num_frame 8 --max_num_frame 32 --freeze_llm False --freeze_mlp False --freeze_backbone False --vision_select_layer -1 --dataloader_num_workers 8 --bf16 True --max_steps 115000 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 --evaluation_strategy no --save_strategy steps --save_steps 500 --save_total_limit 1 --learning_rate 4e-5 --weight_decay 0.05 --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --max_seq_length 4096 --do_train True --grad_checkpoint True --group_by_length False --dynamic_image_size True --use_thumbnail True --ps_version v2 --deepspeed zero_stage1_config.json --report_to mlflow --mlflow_tracking_uri https://dagshub.com/DimuthuLakmal/internvl.mlflow --s3_bucket internvl-tracking-gpu --use_packed_ds False --num_images_expected 48 --max_packed_tokens 4096 --max_buffer_size 20 --log_freq 100 --strict_mode False --replacement False --allow_overflow False --remove_unused_columns False --loss_reduction square --loss_reduction_all_gather True

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s