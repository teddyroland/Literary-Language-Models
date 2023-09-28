#! /bin/bash

# Assign variables

bert_model=${1:-"base"}
pct_smash=${2:-0}
sample_id=${3:-0}

sample_label=${bert_model}-${pct_smash}-${sample_id}

gpu_count=2

# Job Script for Workspace

config_1="#! /bin/bash
#SBATCH -N 1 -n 1 -c 1
#SBATCH --time=1-00:00:00
#SBATCH -e slurm/$sample_label/stdout/1_workspace.out
#SBATCH -o slurm/$sample_label/stdout/1_workspace.out
#SBATCH --partition=largemem
"

env_1="cd ~/BERT-LLM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_model
"

cmd_1="$config_1
$env_1
srun bash code/pipeline/create_workspace.sh $bert_model $pct_smash $sample_id"

# Job Script for Training

config_2="#! /bin/bash
#SBATCH -N 1 -n 1 -c 8
#SBATCH --time=7-00:00:00
#SBATCH -e slurm/$sample_label/stdout/2_train.out
#SBATCH -o slurm/$sample_label/stdout/2_train.out
#SBATCH --open-mode=append
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$gpu_count
"

env_2="cd ~/BERT-LLM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hf_model
"

resource_2="srun hostname
echo $SLURM_JOB_GPUS
srun --gres=gpu:$gpu_count /usr/bin/nvidia-smi
"

cmd_2="$config_2
$env_2
$resource_2
srun --gres=gpu:$gpu_count bash code/pipeline/run_pretrain.sh $bert_model $pct_smash $sample_id $gpu_count"

# Job Script for GLUE

config_3="#! /bin/bash
#SBATCH -N 1 -n 1 -c 8
#SBATCH --time=1-00:00:00
#SBATCH -e slurm/$sample_label/stdout/3_glue.out
#SBATCH -o slurm/$sample_label/stdout/3_glue.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$gpu_count
"

cmd_3="$config_3
$env_2
$resource_2
srun --gres=gpu:$gpu_count bash code/pipeline/run_glue.sh $bert_model $pct_smash $sample_id"

# Job Script for Prediction

config_4="#! /bin/bash
#SBATCH -N 1 -n 1 -c 8
#SBATCH --time=1-00:00:00
#SBATCH -e slurm/$sample_label/stdout/4_predict.out
#SBATCH -o slurm/$sample_label/stdout/4_predict.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:$gpu_count
"

cmd_4="$config_4
$env_2
$resource_2
srun --gres=gpu:$gpu_count bash code/pipeline/run_predict.sh $bert_model $pct_smash $sample_id"

# Job Script for Cleanup

config_5="#! /bin/bash
#SBATCH -N 1 -n 1 -c 1
#SBATCH --time=01:00:00
#SBATCH -e slurm/$sample_label/stdout/5_cleanup.out
#SBATCH -o slurm/$sample_label/stdout/5_cleanup.out
#SBATCH --partition=batch
"

env_5="cd ~/BERT-LLM
"

cmd_5="$config_5
$env_1
srun bash code/pipeline/cleanup_workspace.sh $bert_model $pct_smash $sample_id"

# Create Directory, Write Files

job_base_dir=slurm/$bert_model-$pct_smash-$sample_id
mkdir -m 777 -p $job_base_dir/job-script/
mkdir -m 777 -p $job_base_dir/stdout/

job_dir=$job_base_dir/job-script

printf "$cmd_1" >> $job_dir/$sample_label-wk.job
printf "$cmd_2" >> $job_dir/$sample_label-tr.job
printf "$cmd_3" >> $job_dir/$sample_label-ft.job
printf "$cmd_4" >> $job_dir/$sample_label-pr.job
printf "$cmd_5" >> $job_dir/$sample_label-cl.job

# Queue Job

train_iters=6
if [ "$bert_model" == "small" ] ; then
    train_iters=4
fi

JOB_1=$(sbatch --parsable $job_dir/$sample_label-wk.job)
JOB_PREV=$JOB_1
for i in $(seq 1 $train_iters); do
    JOB_PREV=$(sbatch --parsable --dependency=afterok:$JOB_PREV $job_dir/$sample_label-tr.job)
done
JOB_3=$(sbatch --parsable --dependency=afterok:$JOB_PREV $job_dir/$sample_label-ft.job)
JOB_4=$(sbatch --parsable --dependency=afterok:$JOB_3 $job_dir/$sample_label-pr.job)
JOB_5=$(sbatch --parsable --dependency=afterok:$JOB_4 $job_dir/$sample_label-cl.job)