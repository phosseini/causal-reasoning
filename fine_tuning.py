import copy
import json
import torch

from ray import tune
from typing import Optional, Union
from dataclasses import dataclass
from datasets import (DatasetDict, Dataset)
from transformers import (AutoModelForSequenceClassification, AutoModelForMultipleChoice,
                          AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerBase)
from transformers.tokenization_utils_base import PaddingStrategy
from ray.tune.schedulers import PopulationBasedTraining
from sklearn.model_selection import KFold

from utils import compute_metrics

# ------------------------------
# loading parameters
with open('fine_tuning_config.json') as f:
    params = json.load(f)
task_type = params['task_type']
n_fold = params['n_fold']
test_run_path = params['test_run_path']
random_seeds = params['random_seeds']

# ending0 and ending1 are the two choices for each question
ending_names = ["ending0", "ending1"]

output = []
best_ray_trials = []

# converting splits to Dataset objects and saving them in a DatasetDict
splits = DatasetDict()
splits['train'] = Dataset.from_csv(params['train_data_path'])
splits['test'] = Dataset.from_csv(params['test_data_path'])

# ------------------------------
# loading the model and tokenizer
model_name = params['model_name']
tokenizer_name = params['tokenizer_name']

if task_type == 'multi':
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
elif task_type == 'seq':
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def preprocess_function(examples, prompt=False, task=params['task_type']):
    # checking task value:
    if task not in ['seq', 'multi']:
        print("Task value should be one of the following: \'seq\' or \'multi\'")
        return

    if task == 'multi':
        # Repeat each first sentence two times to go with the two possibilities of second sentences.
        first_sentences = [[context] * 2 for context in examples["sent1"]]
        # Grab all second sentences possible for each context.
        question_headers = examples["sent2"]
        if prompt:
            second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in
                                enumerate(question_headers)]
        else:
            second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i, header in
                                enumerate(question_headers)]
    elif task == 'seq':
        first_sentences = [examples["sent1"]]
        second_sentences = [examples["sent2"]]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Un-flatten
    if task == 'multi':
        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences)
        return {k: [v[i:i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    elif task == 'seq':
        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, padding=True)
        return {k: [v[i:i + 1] for i in range(0, len(v), 1)] for k, v in tokenized_examples.items()}


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


encoded_datasets = splits.map(preprocess_function, batched=True)
columns_to_return = ['input_ids', 'label', 'attention_mask']
encoded_datasets.set_format(type='torch', columns=columns_to_return)


def tune_config_optuna(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", params['learning_rate_start'],
                                             params['learning_rate_end'], log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 4, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size",
                                                                 params['per_device_train_batch_size']),
    }


# some values for BERT
# "learning_rate_start": 1e-5,
# "learning_rate_end": 1e-2,
def tune_config_ray(trial):
    return {
        "learning_rate": tune.loguniform(params['learning_rate_start'], params['learning_rate_end']),
        # "learning_rate": tune.choice(params['learning_rate']),
        "num_train_epochs": tune.choice(params['tuning_num_train_epochs']),
        "per_device_train_batch_size": tune.choice(params['tuning_per_device_train_batch_size']),
    }


pbt_scheduler = PopulationBasedTraining(
    metric='eval_accuracy',
    mode='max',
)


def model_init():
    if task_type == 'multi':
        return AutoModelForMultipleChoice.from_pretrained(model_name)
    elif task_type == 'seq':
        return AutoModelForSequenceClassification.from_pretrained(model_name)


# since we don't have training set in COPA, we run cross-validation for hyperparameter tuning
# obviously, we DON'T do the hyperparameter tuning on test set to avoid leakage

kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
best_objective = 0
best_model_params = {}

for train_index, dev_index in kf.split(encoded_datasets['train']):
    train_index = [int(idx) for idx in list(train_index)]
    dev_index = [int(idx) for idx in list(dev_index)]
    train_set = torch.utils.data.dataset.Subset(encoded_datasets['train'], train_index)
    dev_set = torch.utils.data.dataset.Subset(encoded_datasets['train'], dev_index)

    args = TrainingArguments(
        test_run_path,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        disable_tqdm=True,
    )

    if task_type == 'multi':
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_set,
            eval_dataset=dev_set,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer),
            compute_metrics=compute_metrics,
        )
    elif task_type == 'seq':
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_set,
            eval_dataset=dev_set,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    # Defaut objective is the sum of all metrics when metrics are provided, so we have to maximize it.
    # best_trial = trainer.hyperparameter_search(direction="maximize", hp_space=tune_config_optuna)

    # if we want to specify hyperparameters: pass hp_space=tune_config_ray
    best_trial = trainer.hyperparameter_search(hp_space=tune_config_ray,
                                               backend=params['tuning_backend'],
                                               direction='maximize',
                                               scheduler=pbt_scheduler,
                                               keep_checkpoints_num=1,  # if using Ray and PopulationBasedTraining
                                               n_trials=params['n_trials'],
                                               # resources_per_trial=params['resources_per_trial'],
                                               )
    # check if the model is the best model so far, if yes, save it
    if best_trial.objective > best_objective:
        best_objective = copy.deepcopy(best_trial.objective)
        best_model_params = copy.deepcopy(best_trial)

    # saving best trial
    best_ray_trials.append(best_trial)

print('=========================================')
print(" **** Hyperparameter search results **** ")
print('=========================================')
print(" **** All trials **** ")
for trial in best_ray_trials:
    print(trial)
print('==========================================')
print("Best accuracy: {}".format(best_objective))
print('==========================================')
print("Best run hyperparameters")
print(best_model_params)
print('==========================================')

random_seed_results = []

# now, fine-tuning the model with the best set of hyperparameters and evaluate it on the test set
for random_seed in random_seeds:
    args = TrainingArguments(
        test_run_path,
        learning_rate=best_model_params.hyperparameters['learning_rate'],
        num_train_epochs=best_model_params.hyperparameters['num_train_epochs'],
        per_device_train_batch_size=best_model_params.hyperparameters['per_device_train_batch_size'],
        do_train=True,
        do_eval=True,
        seed=random_seed,
    )

    if task_type == 'multi':
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=encoded_datasets['train'] if args.do_train else None,
            eval_dataset=encoded_datasets['test'] if args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer),
            compute_metrics=compute_metrics,
        )
    elif task_type == 'seq':
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=encoded_datasets['train'] if args.do_train else None,
            eval_dataset=encoded_datasets['test'] if args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    result = trainer.evaluate()

    random_seed_results.append(result['eval_accuracy'])

print('====================================')
print(" *** Report on random seed runs *** ")
print(random_seed_results)
print('\n\nAverage performance: {}'.format(round(sum(random_seed_results) / len(random_seed_results), 3)))
