import json
import argparse
import random
from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

import sys
sys.dont_write_bytecode = True # This keeps the folder tidy
from utils import create_examples_from_batch, create_examples_with_text, DataCollatorForWholeWordMaskAndNSP, deprocess_example

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--data_dir', type=str, required=True)
arg_parser.add_argument('--vocab_file', type=str, required=True)
arg_parser.add_argument('--output_file', type=str, required=True)
arg_parser.add_argument('--sample_size', type=int, required=True)
arg_parser.add_argument('--pass_text', action='store_true')

args = arg_parser.parse_args()

def create_dataloader(data_dir, tokenizer, pass_text = False):

    test_stream = load_dataset(
        path = 'text',
        data_dir=data_dir,
        streaming=True,
        split='train'
    )

    example_fn = create_examples_from_batch
    if pass_text:
        example_fn = create_examples_with_text
    
    test_dataset = test_stream.map(
        lambda batch: example_fn(
            batch,
            tokenizer=tokenizer,
            block_size=128,
            nsp_probability=0.5,
            short_seq_probability=0.1
        ),
        batched=True,
        batch_size=10000,
        remove_columns='text'
    )

    data_collator = DataCollatorForWholeWordMaskAndNSP(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=False,
    )

    return test_dataloader

bert_tokenizer = BertTokenizer(args.vocab_file)
test_dataloader = create_dataloader(args.data_dir, bert_tokenizer, pass_text = args.pass_text)

ex_list = []
for step, inputs in enumerate(test_dataloader):
    if step % 10 == 0:
        input_dict = deprocess_example(inputs, bert_tokenizer, pass_text = args.pass_text)
        ex_list.append(input_dict)

random.shuffle(ex_list)
ex_list = ex_list[:args.sample_size]

with open(args.output_file, 'w') as file_out:
    for input_dict in ex_list:
        print(json.dumps(input_dict), file=file_out)