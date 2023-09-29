## Import Libraries

import argparse
from transformers import BertTokenizer, BertConfig, BertForPreTraining
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

import sys
sys.dont_write_bytecode = True # This keeps the folder tidy
from utils import create_examples_from_batch, DataCollatorForWholeWordMaskAndNSP, StopAfterCallback

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Parse Arguments for I/O & Parameters

arg_parser = argparse.ArgumentParser()

# Filepaths
arg_parser.add_argument('--model_dir', type=str, required=True)
arg_parser.add_argument('--vocab_file', type=str, required=True)
arg_parser.add_argument('--config_file', type=str, required=True)
arg_parser.add_argument('--train_data', type=str, required=True)
arg_parser.add_argument('--eval_data', type=str, required=True)

# Training Parameters
arg_parser.add_argument('--seq_length', type=int, required=True)
arg_parser.add_argument('--batch_size', type=int, required=True)
arg_parser.add_argument('--grad_accum', type=int, required=True)
arg_parser.add_argument('--max_steps', type=int, required=True)
arg_parser.add_argument('--warmup_steps', type=int, required=True)

# Training Management
arg_parser.add_argument('--end_step', type=int, required=True)
arg_parser.add_argument('--data_workers', type=int, required=True)
arg_parser.add_argument('--rand_seed', type=int, required=True)
arg_parser.add_argument('--check_dir', type=str, required=False)
arg_parser.add_argument('--load_model', action='store_true')
arg_parser.add_argument('--save_model', action='store_true')

args = arg_parser.parse_args()

## Initialize Model

bert_tokenizer = BertTokenizer(args.vocab_file)
if args.load_model:
    model = BertForPreTraining.from_pretrained(args.model_dir).to(device)
else:
    config = BertConfig.from_json_file(args.config_file)
    model = BertForPreTraining(config).to(device)

## Pass Dataset for NSP and MLM Tasks

train_stream = load_dataset(
    path = 'text',
    data_dir=args.train_data,
    streaming=True,
    split='train'
)

train_dataset = train_stream.map(
    lambda batch: create_examples_from_batch(
        batch,
        tokenizer=bert_tokenizer,
        block_size = args.seq_length,
        nsp_probability=0.5,
        short_seq_probability=0.1
    ),
    batched=True,
    batch_size=10000,
    remove_columns='text'
)

eval_stream = load_dataset(
    path = 'text',
    data_dir=args.eval_data,
    streaming=True,
    split='train'
)

eval_dataset = eval_stream.map(
    lambda batch: create_examples_from_batch(
        batch,
        tokenizer=bert_tokenizer,
        block_size = args.seq_length,
        nsp_probability=0.5,
        short_seq_probability=0.1
    ),
    batched=True,
    batch_size=10000,
    remove_columns='text'
)

data_collator = DataCollatorForWholeWordMaskAndNSP(
    tokenizer=bert_tokenizer,
    mlm=True,
    mlm_probability= 0.15
)

## Train Model

training_args = TrainingArguments(
    per_device_train_batch_size = args.batch_size,
    per_device_eval_batch_size = args.batch_size,
    gradient_accumulation_steps = args.grad_accum,
    max_steps = args.max_steps,
    warmup_steps = args.warmup_steps,
    dataloader_num_workers = args.data_workers,
    seed = args.rand_seed,
    data_seed = args.rand_seed,
    optim = 'adamw_torch',
    learning_rate = 1e-4,
    weight_decay = 0.01,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon = 1e-06,
    fp16 = True,
    evaluation_strategy = 'steps',
    prediction_loss_only = True,
    eval_steps = 1000,
    save_steps = 1000,
    save_total_limit = 1,
    disable_tqdm = True,
    ignore_data_skip = True,
    output_dir = args.model_dir,
    overwrite_output_dir = True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset = eval_dataset,
    callbacks = [StopAfterCallback(end_step = args.end_step)]
)

trainer.train( args.check_dir if args.check_dir else None )
if args.save_model:
    trainer.save_model(args.model_dir)