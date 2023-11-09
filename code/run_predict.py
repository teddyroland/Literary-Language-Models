import csv
import json
import argparse
import warnings
from transformers import BertTokenizer, BertForPreTraining

import sys
sys.dont_write_bytecode = True # This keeps the folder tidy
from utils import reprocess_example

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--model_dir', type=str, required=True)
arg_parser.add_argument('--vocab_file', type=str, required=True)
arg_parser.add_argument('--example_file', type=str, required=True)
arg_parser.add_argument('--output_file', type=str, required=True)

args = arg_parser.parse_args()

bert_tokenizer = BertTokenizer(args.vocab_file)
bert_model = BertForPreTraining.from_pretrained(args.model_dir).to(device)
bert_model.eval()

mlm_list,token_list,nsp_list,loss_list = [],[],[],[]
i = 0

with torch.no_grad():
    with open(args.example_file,'r') as file_in:
        for json_line in file_in.readlines():
            ex_dict = json.loads(json_line)
            inputs = reprocess_example(ex_dict, bert_tokenizer).to(device)

            mlm_count = int(sum(inputs.labels[0]!=-100))
            token_count = inputs.labels.shape[1]
            nsp_bool = int(inputs.next_sentence_label[0])
            
            if token_count <= bert_model.config.max_position_embeddings:
                outputs = bert_model(**inputs)
                loss_total = float(outputs.loss)
            else:
                loss_total = float('nan')
                warnings.warn(f'Example {i} in {args.example_file} exceeds {bert_model.config.max_position_embeddings} tokens, using the tokenizer at {args.vocab_file}')

            mlm_list.append(mlm_count)
            token_list.append(token_count)
            nsp_list.append(nsp_bool)
            loss_list.append(loss_total)

            i += 1


fields = ['MLM_count','TOKEN_count','NSP_bool','Loss']
rows = list(zip(mlm_list,token_list,nsp_list,loss_list))

with open(args.output_file, 'w') as file_out:
    writer = csv.writer(file_out,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)

    writer.writerow(fields)
    writer.writerows(rows)