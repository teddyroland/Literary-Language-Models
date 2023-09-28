# This script extends the BERT workflow distirbuted by Google,
# by using the recently distributed tensorflow_text library
# to learn a new BERT vocabulary.

import os
from absl import app
from absl import flags
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

# Arguments

FLAGS = flags.FLAGS

flags.DEFINE_string("bert_model", None,
    "BERT model configuration name.")

flags.DEFINE_string("input_dir", None,
    "Directory of text files from which to build vocabulary.")

flags.DEFINE_string("vocab_file", None,
    "File in which vocabulary will be stored.")

def main(_):
    # Vocab parameters

    vocab_size = 30000
    max_file_num = 99
    if FLAGS.bert_model == 'small':
        max_file_num = 24
    file_paths = [os.path.join(FLAGS.input_dir,str(fn)) for fn in range(max_file_num + 1)]

    bert_tokenizer_params=dict(lower_case=True)
    reserved_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    bert_vocab_args = dict(
        vocab_size = vocab_size,
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=bert_tokenizer_params )

    # Learn vocab

    dataset = tf.data.TextLineDataset(file_paths)
    ds_vocab = bert_vocab.bert_vocab_from_dataset( dataset, **bert_vocab_args )

    # Print vocab to file

    with open( FLAGS.vocab_file , 'w') as file_out:
        for token in ds_vocab:
            print(token, file=file_out)

if __name__ == "__main__":
    app.run(main)
