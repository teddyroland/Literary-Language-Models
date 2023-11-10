#! /bin/bash

# Set Current Directory

cd /literary-language-models/

# Files & Directories

test_data_dir=data/text-dataset/wiki

example_file=data/text-examples/wiki_dummy.jsonl
parse_file=workspace/parse/wiki_parse.csv

wiki_model_dir=model/base-0-0/model/
wiki_mlm_file=workspace/mlm-loss/wiki_model_wiki_data.csv

full_model_dir=model/base-25-0/model/
full_mlm_file=workspace/mlm-loss/full_model_wiki_data.csv

exp_loss_file=results/expected_loss_wiki_data.csv

# Sample Passages from Test Set

python code/create_sample.py \
--data_dir $test_data_dir \
--vocab_file $wiki_model_dir/vocab.txt \
--output_file $example_file \
--sample_size 100000 \
--pass_text

# Parse Linguistic Features in Samples

python code/run_parse.py
--working_dir workspace/ \
--example_file $example_file \
--output_file $parse_file

# Evaluate Wiki Model on MLM & NSP

python code/run_predict.py \
--model_dir $wiki_model_dir \
--vocab_file $wiki_model_dir/vocab.txt \
--example_file $example_file \
--output_file $wiki_mlm_file

# Repeat with Full Model

python code/run_predict.py \
--model_dir $full_model_dir \
--vocab_file $full_model_dir/vocab.txt \
--example_file $example_file \
--output_file $full_mlm_file

# Compute Expected Loss By Tagged Feature

python code/run_expected_loss.py \
--parse_file $parse_file \
--wiki_model_mlm_file $wiki_mlm_file \
--full_model_mlm_file $full_mlm_file \
--output_file $exp_loss_file
