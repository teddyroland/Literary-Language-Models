# Literary Language Models
It is a commonplace in the literature on Large Language Models that they are biased by their training data. This project explores one aspect of that biasing. Since GPT-1, novels and other creative writing have been included in most major transformer-based language models. What difference do they make? We test that question by revisiting the BERT model, which was trained on Wikipedia and BookCorpus. How does a model trained on both corpora behave differently than a model trained only on Wikipedia?<br><br>
This repository tests those questions using two BERT-base models trained from scratch: one trained on 2B words from Wikipedia and another trained on 2B words sampled from Wikipedia (1.5B words) and BookCorpus (500M words). The models are referred to as the "Wiki Model" and "Full Model" respectively, through this repository. Models can be downloaded here: [Wiki](https://drive.google.com/file/d/1-iF6J4xvj_zprM1Tl6iVQ7EKYEAHMYsU/), [Full](https://drive.google.com/file/d/1ldOqb8qE-JOs_mai_BWqrmoYV7HmW1St/).<br><br>
This repository contains code used to (1) compare predictive accuracy on various linguistic features (morphology, part-of-speech, supersense) and (2) generate text from each model. Scripts can be found in the 'code' folder. An example workflow for each task can be found in the bash scripts: 'run_mlm.sh' and 'run_generate.sh'<br><br>
Main results for the article [CITATION FORTHCOMING] are found in the folder 'results'
- 'expected_loss.csv' contains the expected loss per tagged feature, computed on held-out Wikipedia articles
- 'full_gen.txt' and 'wiki_gen.txt' each contain 100 samples drawn from their respective language models<br><br>
<!-- end of the list -->
Analyses of model predictions that appear in the article are found in the 'bert_loss.ipynb' notebook.
Supporting data and additional results can be found in the folder 'workspace'.
- 'expected_loss_book_data.csv' contains the expected loss per tagged feature, on held-out BookCorpus novels
- 'parse' directory contains csv files with counts of tagged words per pre-processed example
- 'predictive-loss' directory contains files with BERT's predictive loss (MLM + NSP) by  'full' and 'wiki' models on test examples from both Wikipedia and BookCorpus
<!-- end of the list -->
Due to copyright and file size restrictions, the Wikipedia and BookCorpus data cannot be made available in full. The 'data' folder contains demonstration files to show formatting and directory structure expected by the scripts in 'code'.
- 'text-dataset' contains subdirectories for the BookCorpus and Wikipedia test sets, stored individually. Files are preprocessed with one sentence per line and a double line-break between documents, per BERT standard
- 'text-examples' contain files with "deprocessed" versions of formatted BERT inputs. Wordpieces are reassembled into whole words and masked words are indicated. (The BERT model uses whole-word masking.)<br><br>
In order to prevent data leakage from one model to another, each model in this project uses a unique vocabulary learned from its training dataset. Deprocessing for storage enables the same example to be re-processed using different wordpiece vocabularies. 
<!-- end of the list -->
Models are implemented in the [PyTorch/HuggingFace](https://github.com/huggingface/transformers/tree/main) framework. Wordpiece vocabulary is learned using [Tensorflow Text](https://www.tensorflow.org/text). Morphological features are tagged using [SpaCy](https://spacy.io). Part-of-Speech tags and Supersense tags are from [BookNLP](https://github.com/booknlp/booknlp/tree/main).<br><br>
Wikipedia articles are from the English-language dump in January 2020. BookCorpus was replicated from Smashwords in March 2020.
