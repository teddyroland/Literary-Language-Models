#! /bin/bash

# Set Current Directory

cd /literary-language-models/

# Files & Directories

wiki_model_dir=model/base-0-0/model/
wiki_output=results/generative/wiki_gen.txt

full_model_dir=model/base-25-0/model/
full_output=results/generative/full_gen.txt


# Generate Text from Wiki Model

python code/run_gsn.py \
--model_dir $wiki_model_dir \
--vocab_file $wiki_model_dir/vocab.txt \
--output_file $wiki_output

# Repeat with Full Model

python code/run_gsn.py \
--model_dir $full_model_dir \
--vocab_file $full_model_dir/vocab.txt \
--output_file $full_output
