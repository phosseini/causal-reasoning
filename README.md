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
Once we converted the ATOMIC triples to text, we can continually pretrain a Pretrained Language Model (PLM), BERT here, using the converted text. We call this pretraining step a **continual pretraining** since we use one of the objectives, Masked Language Modeling (MLM) that was originally used in pretraining BERT, to further train the PLM. There are two steps for running the pretraining:
* Setting the parameters in the [`config/pretraining_config.json`](https://github.com/phosseini/causal-reasoning/blob/main/config/pretraining_config.json): Even though most of these parameters are self-descriptive, we give a brief explanation about some of them for clarification purposes:
  * `train_data`: Path to the `.csv` train file that is generated as the result of `ATOMIC-to-Text` process.
  * `dev_data`: Path to the `.csv` development file that is generated as the result of `ATOMIC-to-Text` process. If not using `dev` data, we alternatively split the `train` data into train and dev sets.
  * `relation_category`: A list of triple types (strings) with which we want to continually pretrain our model. There are three main categories of triples in ATOMIC2020 including: `event`, `social`, and `physical`. These categories may deal with different types of knowledge. And, models pretrained with each of these categories or a combination of them may give us different results when fine-tuned and tested on downstream tasks. As a result, we added an option for choosing the triple type(s) with which we want to run the pretraining.
* Runnig the pretraining code: [`src/pretraining.py`](https://github.com/phosseini/causal-reasoning/blob/main/src/pretraining.py)

### Using our continually pretrained model on HuggingFace🤗
Our model is deployed on the [HuggingFace's model hub](https://huggingface.co/models). Here is an example of how you can load the model:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")

model = AutoModel.from_pretrained("phosseini/atomic-bert-v1")
```

## Fine-tuning
After pretraining the PLM with the new data, it is time to fine-tune and evaluate its performance on downstream tasks. There are three steps for running the experiments related to model fine-tuning and evaluation:
* Running the [`src/convert_copa.py`](https://github.com/phosseini/causal-reasoning/blob/main/src/convert_copa.py) to generate all the required COPA-related data files (`train/test`) for fine-tuning.
* Setting the parameters in [`config/fine_tuning_config.json`](https://github.com/phosseini/causal-reasoning/blob/main/config/fine_tuning_config.json). Description of some parameters:
  * `tuning_backend`: For choosing the hyperparameter tuning backend, `ray` or `optuna`
  * `hyperparameter_search`: Whether to run hyperparameter search or not. `1` for running and `0` for not running, respectively.
  * `cross_validation`: Whether running the cross-validation on development set or not. If the `hyperparameter_search` is `0`, this parameter will be ignored since if we do not want to fine-tune the model with a known set of hyperparameters there is no need to run cross-validation on the development set.
  * `tuning_*`: All parameters related to hyperparameter tuning. `tuning_learning_rate_do_range` is set to `1` when we want to search learning rates within a range instead of a predefined list of learning rate values. `tuning_learning_rate_start` and `tuning_learning_rate_end` are to specify the start and end of such a range. Alternatively, we can set the `tuning_learning_rate_do_range` to `0` and learning rates for hyperparameter tuning will be selected from the `tuning_learning_rate` list.
 * Running the fine-tuning code: [`src/fine_tuning.py`](https://github.com/phosseini/causal-reasoning/blob/main/src/fine_tuning.py)

### Citation
```bibtex
@article{hosseini2021commonsense,
  title={Commonsense Knowledge-Augmented Pretrained Language Models for Causal Reasoning Classification},
  author={Hosseini, Pedram and Broniatowski, David A and Diab, Mona},
  journal={arXiv preprint arXiv:2112.08615},
  year={2021}
}
```
