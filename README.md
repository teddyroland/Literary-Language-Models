# Literary-Language-Models
It is a commonplace in the literature on Large Language Models that they are biased by their training data. This project explores one aspect of that biasing. Since GPT-1, novels and other creative writing have been included in most major transformer-based language models. What difference do they make? We test that question by revisiting the BERT model, which was trained on Wikipedia and BookCorpus. How does a model trained on both corpora behave differently than a model trained only on Wikipedia?<br><br>
This repository contains code used to (1) assemble a sampled training dataset, (2) train a full BERT model from scratch, and (3) perform a few basic evaluations of the model. The "code" directory contains two sub-directories: "programs" which contains Python scripts for model training and "pipeline" which constains bash scripts to manage directories and call scripts with appropriate parameters. Model training primarily uses the PyTorch/HuggingFace framework with some functions from TensorFlow Text. The training pipeline includes the creation and submission of Slurm jobs, for use on a high performance computing cluster.
