# Import Libraries

import copy
import argparse
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from transformers.tokenization_utils_base import BatchEncoding
import torch
import torch.nn.functional as F

# User Arguments

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--model_dir', type=str, required=True)
arg_parser.add_argument('--vocab_file', type=str, required=True)
arg_parser.add_argument('--output_file', type=str, required=True)
arg_parser.add_argument('--random_seed', type=int, required=False)

args = arg_parser.parse_args()

# Set random seed; defaults to zero, i.o.t. replicate values in paper

random_seed = args.random_seed if args.random_seed else 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Set device

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Model

bert_tokenizer = BertTokenizer(args.vocab_file)
bert_model = BertForMaskedLM.from_pretrained(args.model_dir).to(device)
bert_model.eval()

# MCMC Parameter

chain_len = 1000
num_chains = 100
sent_len = 21

chain_index = np.arange(num_chains)

# Median WP-Tokens / Sent (under full-model tokenizer), estimated
# BookCorpus: 13
# Wikipedia: 26
# Overall: 21

# Define Text Processing Functions

def init_sentence(token_len, chain_count):

    sent_ids = ['[CLS]'] + ['[MASK]'] * token_len + ['[SEP]']
    sent_id_list = bert_tokenizer.convert_tokens_to_ids(sent_ids)

    example = {key_:[[] for _ in range(chain_count)] for key_ in ['input_ids', 'token_type_ids', 'attention_mask']}

    for chain_i in range(chain_count):
        for id_ in sent_id_list:
            example['input_ids'][chain_i].append(id_)
            example['token_type_ids'][chain_i].append(0)
            example['attention_mask'][chain_i].append(1)

    for k in example.keys():
        example[k] = torch.tensor(example[k])
    
    example = BatchEncoding(example).to(device)

    return example

    # Resample token positions

def draw_sample(example, token_len, chain_count):

    sample_pos = np.arange(1,token_len+1)
    rand_array = np.random.choice(sample_pos, size = (chain_count,token_len), replace = True)

    # Gibbs-style sampling on individual tokens
    for col_i in range(token_len):
        rand_col = rand_array[:,col_i]
        example['input_ids'][chain_index,rand_col] = bert_tokenizer.convert_tokens_to_ids('[MASK]')
        logits_ = bert_model(**example).logits
        probs_ = F.softmax(logits_[chain_index,rand_col],dim=1)
        new_id = torch.multinomial(probs_, num_samples = 1).squeeze(dim=-1)
        example['input_ids'][chain_index,rand_col] = new_id
    
    return example


# Main Loop

sentence = init_sentence(sent_len,num_chains)

for step_i in range(chain_len):
    sentence = draw_sample(sentence,sent_len,num_chains)

# Main Loop

sentence = init_sentence(sent_len,num_chains)

for step_i in range(chain_len):
    sentence = draw_sample(sentence,sent_len,num_chains)

# Process into human-readable form

cpu_sentence = sentence.to('cpu') if device != 'cpu' else sentence
sentence_list = [bert_tokenizer.decode(cpu_sentence['input_ids'][chain_i]) for chain_i in range(num_chains)]

# Export Results

with open(args.output_file, 'a') as file_out:
    for sent_i,sentence_str in enumerate(sentence_list):
        print(f"{sent_i}\t{sentence_str}", file=file_out)
