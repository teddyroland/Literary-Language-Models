# Literary-Language-Models
It is a commonplace in the literature on Large Language Models that they are biased by their training data. This project explores one aspect of that biasing. Since GPT-1, novels and other creative writing have been included in most major transformer-based language models. What difference do they make? We test that question by revisiting the BERT model, which was trained on Wikipedia and BookCorpus. How does a model trained on both corpora behave differently than a model trained only on Wikipedia?<br><br>
This repository contains code used to (1) assemble a sampled training dataset, (2) train a full BERT model from scratch, and (3) perform a few basic evaluations of the model. Scripts are contained two sub-directories: "programs" which contains Python scripts for model training and "pipeline" which constains bash scripts to manage directories and call Python scripts with appropriate parameters.<br><br>
The full training pipeline includes the creation and submission of Slurm jobs, for use on a high performance computing (HPC) cluster. The pipeline begins with a call to the "create_job.sh" script, which generates a series of job files that call the other bash scripts in sequence:
* `create_workspace.sh` : Create directories for model training, collect sharded data in local directory, create wordpiece vocabulary from scratch
* `run_pretrain.sh` : Train BERT model. Note that HPC's often limit the time that jobs may run, so run_pretrain.sh is designed to be called multiple times, with restarts from a given checkpoint.
* `run_glue.sh` : Calls the Huggingface [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) script. This script does not grid-search over hyper-parameters and should be understood as a first pass evaluation on the [GLUE benchmark](https://gluebenchmark.com).
* `run_predict.sh` : Evaluate BERT by making predictions on new data and measuring loss. Examples must be created in advance, using the script create_prediction_sample.py
* `cleanup_workspace.sh` : Save prediction results and vocabulary, gzip the model directory, delete workspace folder

The pipeline expects to find the following folder structure:
```
 |-code
 | |-pipeline
 | |-programs
 |-data
 | |-model-config
 | |-text-dataset
 | | |-eval
 | | | |-smash
 | | | |-wiki
 | | |-test
 | | | |-smash
 | | | |-wiki
 | | |-train
 | | | |-smash
 | | | |-wiki
 | |-text-examples
 | |-text-metadata
 |-results
 | |-model-archive
 | |-model-eval
 | |-model-vocab
 |-slurm
 |-workspace
```

Model training and prediction uses the PyTorch/HuggingFace framework. Wordpiece vocabulary is learned using TensorFlow Text.
