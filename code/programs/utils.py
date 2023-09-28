## Import Libraries

import copy
import random
from typing import Dict, List, Mapping, Union, Any, Optional

from transformers import DataCollatorForWholeWordMask, DefaultFlowCallback
from transformers.tokenization_utils_base import BatchEncoding
import torch

## Define Classes & Functions for BERT Pre-Training

# 'create_examples_from_batch'

# In order to stream the dataset from disk (rather than load fully into memory),
# we define a function that will handle tokenization and NSP randomization over batches.
# Our function 'create_examples_from_batch' combines the initialization and
# the 'create_examples_from_document' method in the class TextDatasetForNextSentencePrediction
# from the standard transformers library.

# NOTE: The standard TextDatasetForNextSentencePrediction has a bug that omits the final truncation
# of training examples to max_num_tokens, despite the function being used in related classes.
# As a result, sentences may run much longer than the maximum token length permitted by the
# model. 'create_examples_from_batch' includes a function 'truncate_seq_pair'.

def create_examples_from_batch(batch, tokenizer, block_size, short_seq_probability, nsp_probability):
    
    # Import batch of texts
    # Adapted from __init__ method in TextDatasetForNextSentencePrediction -tr

    documents = [[]]
    for line in iter(batch['text']):
        line = line.strip()
        if not line and len(documents[-1]) != 0:
            documents.append([])
        tokens = tokenizer.tokenize(line)
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        if tokens:
            documents[-1].append(tokens)
    
    # Remove empty documents, due to line breaks in source file -tr
    documents = [d for d in documents if len(d)>0]

    examples = []

    # Process batch of texts
    # Adapted from create_examples_from_document method in TextDatasetForNextSentencePrediction -tr

    max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)
    for doc_index, document in enumerate(documents):

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(0, len(documents) - 1)
                            if random_document_index != doc_index:
                                break

                        random_document = documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    # Insert function to truncate sentences
                    # Function copied from LineByLineWithSOPTextDataset class, (also in language_modeling.py) -tr
                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            if not (len(trunc_tokens) >= 1):
                                raise ValueError("Sequence length to be truncated must be no less than one")
                            # We want to sometimes truncate from the front and sometimes from the
                            # back to add more randomness and avoid biases.
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()
            
                    # Call to function copied from LineByLineWithSOPTextDataset class -tr
                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    if not (len(tokens_a) >= 1):
                        raise ValueError(f"Length of sequence a is {len(tokens_a)} which must be no less than 1")
                    if not (len(tokens_b) >= 1):
                        raise ValueError(f"Length of sequence b is {len(tokens_b)} which must be no less than 1")

                    # add special tokens
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long),
                    }

                    examples.append(example)

                current_chunk = []
                current_length = 0

            i += 1
    
    # Examples must be formatted as a dictionary for processing by model -tr
    ex_keys = ["input_ids", "token_type_ids", "next_sentence_label"]
    return { k:[examples[i][k] for i in range(len(examples))] for k in ex_keys}


def create_one_sentence_examples(batch, tokenizer, block_size, **kwargs):
    
    # Import batch of texts
    # Adapted from __init__ method in TextDatasetForNextSentencePrediction
    # Passes plaintext for each example, in addition to standard features -tr

    doc_token = [[]]
    doc_texts = [[]]
    for line in iter(batch['text']):
        line = line.strip()
        if not line and len(doc_token[-1]) != 0:
            doc_token.append([])
            doc_texts.append([])
        tokens = tokenizer.tokenize(line)
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        if tokens:
            doc_token[-1].append(tokens)
            doc_texts[-1].append(line)
    
    # Remove empty documents, due to line breaks in source file -tr
    documents, texts = [], []
    for doc_index,document in enumerate(doc_token):
        if len(document)>0:
            documents.append(document)
            texts.append(doc_texts[doc_index])


    examples = []

    # Process batch of texts
    # Adapted from create_examples_from_document method in TextDatasetForNextSentencePrediction -tr

    # Only count two special tokens toward max_num_tokens -tr
    # max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)
    max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=False)
    for doc_index,document in enumerate(documents):

        # Bring in the plaintext strings for a given document. -tr
        text = texts[doc_index]

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        
        # Disallow short sequences -tr
        # if random.random() < short_seq_probability:
        #     target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_text = []
        current_length = 0
        i = 0

        while i < len(document):
            segment,plain_text = document[i], text[i]
            current_chunk.append(segment)
            current_text.append(plain_text)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.

                    # Omit block for two-sentence case -tr
                    
                    a_end = max(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    
                    if len(tokens_a) > max_num_tokens:
                        tokens_a = tokens_a[:max_num_tokens]

                    text_a = " ".join(current_text[:a_end])
                    ascii_a = [ord(c) for c in text_a]

                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments

                    tokens_b = None
                    is_random_next = False


                    # add special tokens
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long),
                        "ascii": torch.tensor(ascii_a, dtype=torch.long),
                    }

                    examples.append(example)

                current_chunk = []
                current_length = 0
                current_text = []

            i += 1
    
    # Examples must be formatted as a dictionary for processing by model -tr
    ex_keys = ["input_ids", "token_type_ids", "next_sentence_label", "ascii"]
    return { k:[examples[i][k] for i in range(len(examples))] for k in ex_keys}

# 'DataCollatorForWholeWordMaskAndNSP'

# The main branch of transformers has a version of DataCollatorForWholeWordMask that is not
# compatible with TextDatasetForNextSentencePrediction. Specifically, it sheds any model
# inputs created by the Dataset class, returning only labels for word masking. The class here
# passes existing labels in addition to adding ones for MLM.

class DataCollatorForWholeWordMaskAndNSP(DataCollatorForWholeWordMask):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        
        # Pad existing labels to the appropriate length;
        # insert 'inputs', 'labels' into full dictionary of labels -tr
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch["input_ids"], batch["labels"] = inputs, labels
        return batch

# Supporting Functions

# We import DataCollatorForLanguageModeling from the library's highest level, so we don't get access
# to two functions called in DataCollatorForWholeWordMask: '_torch_collate_batch', 'tolist'
# They are copied here from 'transformers/data/data_collator.py'

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()

# 'StopAfterCallback'

# Training time on the wall clock is much longer than time allotments on the university
# computing cluster. In order to control when training will be interrupted, the class
# 'StopAfterCallback' takes an argument for the specific step at which to stop.

class StopAfterCallback(DefaultFlowCallback):
    def __init__(self, end_step: int = 1000000):
        self.end_step = end_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.end_step:
            control.should_training_stop = True

        return control

# 'deprocess_example' & 'reprocess_example'

# Complementary functions that allow examples to be saved and re-loaded across models.
# Their key feature is to represent words as complete strings, rather than wordpieces
# or token ids which are tokenizer dependent. 'deprocess_example' is used in
# 'create_prediction_data.py' and 'repreocess_example' is used in 'run_predict.py' and
# 'run_backprop.py'

def deprocess_example(example,tokenizer,pass_text=False):
    
    sent_list = [[],[]]
    label_list = [[],[]]
    token_count = example['input_ids'].shape[1]
    i = 0
    while i < token_count:
        in_key = 'input_ids'
        out_label = int(example['labels'][0][i] != -100)
        if out_label:
            in_key = 'labels'
        token_id = example[in_key][0][i]
        current_word = tokenizer.convert_ids_to_tokens(token_id.item())
        sent_index = example['token_type_ids'][0][i].item()
        while i < token_count - 1:
            if current_word in ['[SEP]','[CLS]'] or current_word.startswith('##'):
                break
            next_label = int(example['labels'][0][i+1] != -100)
            if next_label != out_label:
                break
            next_id = example[in_key][0][i+1].item()
            next_word = tokenizer.convert_ids_to_tokens(next_id)
            if not next_word.startswith('##'):
                break
            current_word += next_word[2:]
            i+=1
        sent_list[sent_index].append(current_word)
        label_list[sent_index].append(out_label)
        i+=1

    nsp_bool = example['next_sentence_label'][0].item()

    example_dict = {'sent_a':{'tokens':sent_list[0],'mask':label_list[0]},
                    'sent_b':{'tokens':sent_list[1],'mask':label_list[1]},
                    'nsp':nsp_bool}

    if pass_text:
        example_dict['sent_a']['text'] = "".join([chr(c) for c in example['ascii'][0]])
        example_dict['sent_b']['text'] = ""

    return example_dict


def reprocess_example(example_dict,tokenizer):
    example = {'input_ids':[[]], 'token_type_ids':[[]], 'attention_mask':[[]], 'labels':[[]]}
    example['next_sentence_label'] = [example_dict['nsp']]

    sent_keys = ['sent_a','sent_b']
    for sent_i,sk in enumerate(sent_keys):
        for word_i, word in enumerate(example_dict[sk]['tokens']):

            if not word.startswith('##'):
                token_list = tokenizer.tokenize(word)
            else:
                token_list = [word]

            token_id_list = tokenizer.convert_tokens_to_ids(token_list)

            mask = example_dict[sk]['mask'][word_i]

            for token in token_id_list:
                if not mask:
                    token_id = token
                    label_id = -100
                else:
                    label_id = token
                    token_id = tokenizer.mask_token_id

                example['input_ids'][0].append(token_id)
                example['labels'][0].append(label_id)
                example['attention_mask'][0].append(1)
                example['token_type_ids'][0].append(sent_i)

    for k in example.keys():
        example[k] = torch.tensor(example[k])

    return BatchEncoding(example)

def cascade_example(example_dict, tokenizer, max_len=128):

    if len(example_dict['sent_b']['tokens'])>0:
        raise ValueError('Cascade masking can only be used for single-sentence inputs.')

    word_count = len(example_dict['sent_a']['tokens'])
    assert word_count <= max_len

    example = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[], 'labels':[]}

    for mask_i in range( 1, word_count - 1 ):
        ex_ids, ex_labels, ex_attn, ex_type = [], [], [], []
        token_count = 0
        for word_i, word in enumerate(example_dict['sent_a']['tokens']):

            if not word.startswith('##'):
                token_list = tokenizer.tokenize(word)
            else:
                token_list = [word]

            token_id_list = tokenizer.convert_tokens_to_ids(token_list)

            mask = mask_i == word_i

            for token in token_id_list:
                if not mask:
                    token_id = token
                    label_id = -100
                else:
                    label_id = token
                    token_id = tokenizer.mask_token_id

                ex_ids.append(token_id)
                ex_labels.append(label_id)
                ex_type.append(0)
                ex_attn.append(1)
                token_count += 1

        # Slice or pad "sentence" to 128 tokens
        
        while token_count > max_len:
            del ex_ids[-2]
            del ex_labels[-2]
            del ex_type[-2]
            del ex_attn[-2]
            token_count -= 1
        
        while token_count < max_len:
            ex_ids.append(0)
            ex_labels.append(-100)
            ex_type.append(0)
            ex_attn.append(0)
            token_count += 1

        example['input_ids'].append(ex_ids)
        example['labels'].append(ex_labels)
        example['token_type_ids'].append(ex_type)
        example['attention_mask'].append(ex_attn)
    
    # Pad the batch with additional "sentences" up to 128

    for _ in range(max_len - word_count + 2):
        example['input_ids'].append([0]*max_len)
        example['labels'].append([-100]*max_len)
        example['token_type_ids'].append([0]*max_len)
        example['attention_mask'].append([0]*max_len)

    for k in example.keys():
        example[k] = torch.tensor(example[k])
    
    return  BatchEncoding(example)