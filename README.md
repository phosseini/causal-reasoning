# Commonsense Causal Reasoning

<p align="center">
  <img src='method.png' width='500' height='350' style="vertical-align:middle;margin:100px 50px">
</p>

## Converting ATOMIC-to-Text
Triples in ATOMIC are stored in form of: `(subject, relation, target)`. We convert (verbalize) these triples to natural language text to later use them in training/fine-tuning some Pretrained Language Models (PLMs).
#### Steps:
1. Download ATOMIC 2020 [here](https://allenai.org/data/atomic-2020), put it in the `/data` folder, and unzip it.
2. Run the following code: [`src/convert_atomic_to_text.py`](https://github.com/phosseini/causal-reasoning/blob/main/src/convert_atomic_to_text.py)
3. Output will be stored as `.txt` and `.csv` files in the `/data` folder.


## Continual Pretraining
Once we converted the ATOMIC triples to text, we can continually pretrain a Pretrained Language Model (PLM), BERT here, using the converted text. We call this pretraining step a **continual pretraining** since we use one of the objectives, Masked Language Modeling (MLM), that was originally used in pretraining BERT. There are two steps for running the pretraining:
* Setting the parameters in the [`pretraining_config.json`](https://github.com/phosseini/causal-reasoning/blob/main/config/pretraining_config.json): Even though most of these parameters are self-descriptive, we give a brief explanation about some of them for clarification purposes:
  * `pretraining_input`: path to the `.csv` file that is generated as the result of `ATOMIC-to-Text` process.
  * `relation_category`: a list of triple types (strings) with which we want to continually pretrain our model. There are three main categories of triples in ATOMIC2020 including: `event`, `social`, and `physical`. These categories may deal with different types of knowledge. And, models pretrained with each of these categories or a combination of them may give us different results when fine-tuned and tested on downstream tasks. As a result, we added an option for choosing the triple type(s) with which we want to run the pretraining.
