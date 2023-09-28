#! /bin/bash
set -eo pipefail

# Arguments

bert_model=${1:-"base"}
pct_smash=${2:-0}
sample_id=${3:-0}
gpu_count=${4:-0}

# Filepaths

model_label=$bert_model-$pct_smash-$sample_id

model_dir=workspace/$model_label/model
vocab_file=$model_dir/vocab.txt
config_file=$model_dir/config.json

data_dir=workspace/$model_label/data
train_dir=$data_dir/train-shards
eval_dir=$data_dir/eval-shards

# Get Current Checkpoint

check_dir=$(find $model_dir -maxdepth 1  -type d -name 'checkpoint-*')

current_step=0

if [ ${#check_dir} -gt 0 ]; then
    current_step=${check_dir:${#check_dir}-6:${#check_dir}}
    if [ $(( current_step % 200000 )) -ne 0 -a $current_step -ne 900000 ]; then
        echo "ERROR: Incorrect step count."
        exit 1
    fi
fi

# Training Management & Hyperparameters

if [ "$bert_model" == "base" ]; then
    end_step=$(( $current_step + 200000 ))
else
    end_step=$(( $current_step + 400000 ))
fi

end_step=$(( $end_step < 900000 ? $end_step : 900000 ))

if [ $current_step -eq 900000 ]; then
    end_step=100000

    seq_length=512
    batch_size=32
    max_steps=100000
else
    seq_length=128
    batch_size=128
    max_steps=900000
fi


grad_accum=8
grad_accum=$(( $grad_accum / $gpu_count ))
warmup_steps=$(( $max_steps / 10 ))
data_workers=4

# NOTE: Batch size & gradient accumulation steps are tuned to the memory available to
# GPU clusters at UCSB's high-performance computing center. Test values on new hardware.

if [ "$bert_model" == "small" ] ; then
    batch_size=$(( $batch_size * 2 ))
    grad_accum=$(( $grad_accum / 2 ))
	data_workers=$(( $data_workers * 2 ))
fi

# Assemble arguments for run_pretrain.py

CMD="python code/programs/run_pretrain.py"
CMD+=" --model_dir=$model_dir"
CMD+=" --config_file=$config_file"
CMD+=" --vocab_file=$vocab_file"
CMD+=" --train_data=$train_dir"
CMD+=" --eval_data=$eval_dir"
CMD+=" --seq_length=$seq_length"
CMD+=" --batch_size=$batch_size"
CMD+=" --grad_accum=$grad_accum"
CMD+=" --max_steps=$max_steps"
CMD+=" --warmup_steps=$warmup_steps"
CMD+=" --end_step=$end_step"
CMD+=" --data_workers=$data_workers"
CMD+=" --rand_seed=$sample_id"

if [ ${#check_dir} -gt 0 -a $end_step -ne 100000 ]; then
    CMD+=" --check_dir=$check_dir"
fi

if [ $end_step -eq 100000 ]; then
    CMD+=" --load_model"
fi

if [ $end_step -eq 900000 -o $end_step -eq 100000 ]; then
    CMD+=" --save_model"
fi

# Execute run_pretrain.py

set -x
$CMD
set +x