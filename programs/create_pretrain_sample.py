# This script takes pre-selected samples from the dataset and
# returns a list of files in the format required by run_pretraining.py

import os, shutil
from absl import app
from absl import flags
import pandas as pd
import random


# Arguments

FLAGS = flags.FLAGS

flags.DEFINE_string("metadata_file", None,
    "File containing pre-training data. CSV format.")

flags.DEFINE_string("model_label", None,
    "Unique label for pre-training dataset, from model config name,"
    "percent drawn from smashwords corpus, and id number.")

flags.DEFINE_string("split", None,
    "Unique label for pre-training dataset, from model config name,"
    "percent drawn from smashwords corpus, and id number.")

flags.DEFINE_string("path_in", None,
    "Directory containing pre-training data. Expected to contain each "
    "each sub-corpus in its own folder, where txt files are stored.")

flags.DEFINE_string("path_out", None,
    "Directory containing pre-training data. Expected to contain each "
    "each sub-corpus in its own folder, where txt files are stored.")


def main(_):

    # Data Selection

    sample_df = pd.read_csv(FLAGS.metadata_file)
    this_sample = sample_df[ (sample_df[FLAGS.model_label] ) & (sample_df['Data Split']==FLAGS.split)][['Corpus','Shard ID']].values
    num_chunks = 100

    model_ids = FLAGS.model_label.split('-')
    if model_ids[0] == 'small':
        num_chunks = 25
    
    if FLAGS.split=='eval':
        num_chunks=1

    random.seed(model_ids[2])
    random.shuffle(this_sample)

    j = 0
    for i in range(num_chunks):

        fn_out = os.path.join( FLAGS.path_out, str(i))
        with open(fn_out, 'wb' ) as file_out:

            while j < int( this_sample.shape[0] / num_chunks * ( i+1 ) ):

                corpus,shard = this_sample[j]

                fn_in = os.path.join( FLAGS.path_in, corpus, str(shard))
                with open(fn_in,'rb') as file_in:
                    shutil.copyfileobj(file_in, file_out)
                
                file_out.write(b"\n\n")

                j += 1

if __name__ == "__main__":
    app.run(main)
