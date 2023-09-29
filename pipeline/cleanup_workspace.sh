#! /bin/bash
set -eo pipefail

# Assign variables

bert_model=${1:-"base"}
pct_smash=${2:-0}
sample_id=${3:-0}

# Transform arguments into useable forms

model_label=${bert_model}-${pct_smash}-${sample_id}
base_dir=workspace/$model_label

glue_tasks="cola sst2 mrpc stsb qqp mnli qnli rte wnli"

# Confirm Pipeline Executed Correctly

if [[ ! -f $base_dir/model/pytorch_model.bin ]]; then
    echo "ERROR: Trained model does not exist."
    exit 1
fi

if [[ ! -d $base_dir/model/checkpoint-100000 ]]; then
    echo "ERROR: Incorrect step count."
    exit 1
fi

for task in $glue_tasks; do
    if [[ ! -f $base_dir/glue/$task/eval_results.json ]]; then
        echo "ERROR: Fine-tune failed on $task."
        exit 1
    fi
done

if [[ ! -f results/model-eval/$model_label/predict/smash_ngram.csv ]]; then
    echo "ERROR: Prediction failed."
    exit 1
fi

# Execute

for task in $glue_tasks; do
    set -x
    cp $base_dir/glue/$task/eval_results.json results/model-eval/$model_label/glue/${task}_results.json
    set +x
done
set -x
rm -r $base_dir/model/checkpoint*
tar -czf results/model-archive/model-${model_label}.tar.gz -C workspace $model_label/model
rm -r $base_dir
set +x
