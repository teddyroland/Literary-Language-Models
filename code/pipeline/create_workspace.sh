#! /bin/bash
set -eo pipefail

# Assign variables

bert_model=${1:-"base"}
pct_smash=${2:-0}
sample_id=${3:-0}

# Transform arguments into forms taken by python scripts

model_label=$bert_model-$pct_smash-$sample_id
vocab_file=workspace/$model_label/model/vocab.txt
config_file=workspace/$model_label/model/config.json

if [ "$bert_model" == "small" ] ; then
	max_file_num=24
else
    max_file_num=99
fi

# Make directories for sampled texts & fine-tuned model

model_dir=workspace/$model_label/model

data_dir=workspace/$model_label/data
train_dir=$data_dir/train-shards
eval_dir=$data_dir/eval-shards

fine_dir=workspace/$model_label/glue
cache_dir=$fine_dir/cache

results_dir=results/model-eval/$model_label
glue_dir=$results_dir/glue
pred_dir=$results_dir/predict

mkdir -m 777 -p $model_dir
mkdir -m 777 -p $train_dir
mkdir -m 777 -p $eval_dir
mkdir -m 777 -p $cache_dir
mkdir -m 777 -p $glue_dir
mkdir -m 777 -p $pred_dir

# Assemble command to run create_pretrain_sample.py

CMD_SAMPLE="python code/programs/create_pretrain_sample.py"
CMD_SAMPLE+=" --metadata_file=data/text-metadata/data-sample.csv"
CMD_SAMPLE+=" --model_label=$model_label"

# ... over training data

CMD_SAMPLE_TRAIN="$CMD_SAMPLE"
CMD_SAMPLE_TRAIN+=" --split=train"
CMD_SAMPLE_TRAIN+=" --path_in=data/text-dataset/train/"
CMD_SAMPLE_TRAIN+=" --path_out=$train_dir"

# ... over eval data

CMD_SAMPLE_EVAL="$CMD_SAMPLE"
CMD_SAMPLE_EVAL+=" --split=eval"
CMD_SAMPLE_EVAL+=" --path_in=data/text-dataset/eval/"
CMD_SAMPLE_EVAL+=" --path_out=$eval_dir"

# Assemble command to run create_vocab.py

CMD_VOCAB="python code/programs/create_vocab.py"
CMD_VOCAB+=" --bert_model=$bert_model"
CMD_VOCAB+=" --vocab_file=$vocab_file"
CMD_VOCAB+=" --input_dir=$train_dir"

# Execute Python scripts

set -x
$CMD_SAMPLE_TRAIN
$CMD_SAMPLE_EVAL
cp data/model-config/config-$bert_model.json $config_file
$CMD_VOCAB
cp $vocab_file results/model-vocab/vocab-$model_label.txt
set +x