#! /bin/bash
set -eo pipefail

# Assign variables

bert_model=${1:-"base"}
pct_smash=${2:-0}
sample_id=${3:-0}

# Transform arguments into forms taken by python scripts

model_label=${bert_model}-${pct_smash}-${sample_id}

# Set directories

model_dir=workspace/$model_label/model/
fine_dir=workspace/$model_label/glue
cache_dir=$fine_dir/cache

# Assemble command for run_glue.py 

CMD="python code/programs/run_glue.py"
CMD+=" --do_train --do_eval"
CMD+=" --max_seq_length 128"
CMD+=" --per_device_train_batch_size 32"
CMD+=" --learning_rate 5e-5"
CMD+=" --fp16"
CMD+=" --overwrite_output_dir"
CMD+=" --model_name_or_path $model_dir"
CMD+=" --disable_tqdm TRUE"

# Iterate over tasks & execute run_glue.py

ft_tasks="cola sst2 mrpc stsb qqp mnli qnli rte wnli"

for task in $ft_tasks; do
    CMD_TASK=$CMD
    CMD_TASK+=" --task_name $task"
    CMD_TASK+=" --output_dir $fine_dir/$task"
    CMD_TASK+=" --cache_dir $cache_dir"
    if [ "$bert_model" == "small" ] ; then
        CMD_TASK+=" --num_train_epochs 4"
    else
        CMD_TASK+=" --num_train_epochs 3"
    fi
    set -x
    $CMD_TASK
    set +x
done