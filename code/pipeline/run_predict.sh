#! /bin/bash
set -eo pipefail

# Arguments

bert_model=${1:-"base"}
pct_smash=${2:-0}
sample_id=${3:-0}

# Filepaths

model_label=$bert_model-$pct_smash-$sample_id

model_dir=workspace/$model_label/model
vocab_file=$model_dir/vocab.txt

pred_dir=results/model-eval/$model_label/predict

# Assemble arguments for run_predict.py and executue

CMD="python code/programs/run_predict.py"
CMD+=" --model_dir=$model_dir"
CMD+=" --vocab_file=$vocab_file"

data_sets="wiki smash"
tasks="mlm_nsp sent ngram"

for subset in $data_sets; do
    for task in $tasks; do
        CMD_SUB=$CMD
        CMD_SUB+=" --example_file=data/text-examples/${subset}_${task}.jsonl"
        CMD_SUB+=" --output_file=$pred_dir/${subset}_${task}.csv"
        set -x
        $CMD_SUB
        set +x
    done
done