import copy
import json
import torch

import numpy as np
from ray import tune
from typing import Optional, Union
from dataclasses import dataclass
from datasets import (DatasetDict, Dataset)
from transformers import (AutoModelForSequenceClassification, AutoModelForMultipleChoice,
                          AutoModelForNextSentencePrediction, DataCollatorWithPadding,
                          AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerBase)
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.trainer_utils import IntervalStrategy
from ray.tune.schedulers import PopulationBasedTraining
from sklearn.model_selection import KFold

from utils import compute_metrics, lower_nth

# ------------------------------
# loading parameters
with open('fine_tuning_config.json') as f:
    params = json.load(f)

task_type = params['task_type']
n_fold = params['n_fold']
tuning_output_path = params['hp_tuning_output_path']
running_output_path = params['running_output_path']
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
elif task_type == 'nsp':
    model = AutoModelForNextSentencePrediction.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def preprocess_function(examples, prompt=True, task=params['task_type']):
    # checking task value:
    if task not in ['seq', 'multi', 'nsp']:
        print("Task value should be one of the following: \'seq\' or \'multi\' or \'nsp\'")
        return

    if task == 'multi':
        # Repeat each first sentence two times to go with the two possibilities of second sentences.
        first_sentences = [[context] * 2 for context in examples["sent1"]]
        # Grab all second sentences possible for each context.
        question_headers = examples["sent2"]
        if prompt:
            second_sentences = [[f"{header} {lower_nth(examples[end][i], 0)}" for end in ending_names] for i, header in
                                enumerate(question_headers)]
        else:
            second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i, header in
                                enumerate(question_headers)]
    elif task in ['seq', 'nsp']:
        first_sentences = [examples["sent1"]]
        second_sentences = [examples["sent2"]]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Un-flatten
    if task == 'multi':
        tokenized_examples = tokenizer(first_sentences, second_sentences, max_length=params['max_length'],
                                       truncation=True)
        return {k: [v[i:i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    elif task in ['seq', 'nsp']:
        tokenized_examples = tokenizer(first_sentences, second_sentences, max_length=params['max_length'],
                                       truncation=True)
        return tokenized_examples


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
columns_to_return = ['input_ids', 'label', 'attention_mask', 'token_type_ids']
encoded_datasets.set_format(type='torch', columns=columns_to_return)


def tune_config_optuna(trial):
    optuna_config = {}
    optuna_config["num_train_epochs"] = trial.suggest_int("num_train_epochs", 3, 4, 5)
    optuna_config["per_device_train_batch_size"] = trial.suggest_categorical("per_device_train_batch_size",
                                                                             params['per_device_train_batch_size'])
    if params['learning_rate_range'] == 1:
        optuna_config["learning_rate"] = trial.suggest_float("learning_rate", params['learning_rate_start'],
                                                             params['learning_rate_end'], log=True)
    return optuna_config


def tune_config_ray(trial):
    ray_config = {}
    if params['learning_rate_range'] == 1:
        ray_config["learning_rate"] = tune.loguniform(params['learning_rate_start'], params['learning_rate_end'])
    else:
        ray_config["learning_rate"] = tune.choice(params['tuning_learning_rate'])

    ray_config["num_train_epochs"] = tune.choice(params['tuning_num_train_epochs'])
    ray_config["per_device_train_batch_size"] = tune.choice(params['tuning_per_device_train_batch_size'])
    return ray_config


pbt_scheduler = PopulationBasedTraining(
    metric='eval_accuracy',
    mode='max',
)


def model_init():
    if task_type == 'multi':
        return AutoModelForMultipleChoice.from_pretrained(model_name)
    elif task_type == 'seq':
        return AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task_type == 'nsp':
        return AutoModelForNextSentencePrediction.from_pretrained(model_name)


def run_hyperparameter_tuning(data_train, data_dev):
    args = TrainingArguments(
        tuning_output_path,
        do_train=True,
        do_eval=True,
        evaluation_strategy=IntervalStrategy.STEPS,
        # save_strategy=IntervalStrategy.EPOCH,
        # save_total_limit=1,
        disable_tqdm=True,
    )

    if task_type == 'multi':
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=data_train,
            eval_dataset=data_dev,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer),
            compute_metrics=compute_metrics,
        )
    elif task_type in ['seq', 'nsp']:
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=data_train,
            eval_dataset=data_dev,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )

    # Defaut objective is the sum of all metrics when metrics are provided, so we have to maximize it.
    # best_trial = trainer.hyperparameter_search(direction="maximize", hp_space=tune_config_optuna)

    # if we want to specify hyperparameters: pass hp_space=tune_config_ray
    best_trial = trainer.hyperparameter_search(hp_space=tune_config_ray,
                                               backend=params['tuning_backend'],
                                               direction='maximize',
                                               # scheduler=pbt_scheduler,
                                               keep_checkpoints_num=0,  # if using Ray and PopulationBasedTraining
                                               n_trials=params['n_trials'],
                                               resources_per_trial=params['resources_per_trial'],
                                               )
    return best_trial


# since we don't have training set in COPA, we run cross-validation for hyperparameter tuning
# obviously, we DON'T do the hyperparameter tuning on test set to avoid leakage

if params['hyperparameter_search'] == 1:
    best_objective = 0
    best_model_params = {}
    if params['cross_validation'] == 1:
        kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)

        for train_index, dev_index in kf.split(encoded_datasets['train']):
            train_index = [int(idx) for idx in list(train_index)]
            dev_index = [int(idx) for idx in list(dev_index)]
            train_set = torch.utils.data.dataset.Subset(encoded_datasets['train'], train_index)
            dev_set = torch.utils.data.dataset.Subset(encoded_datasets['train'], dev_index)

            best_trial = run_hyperparameter_tuning(train_set, dev_set)

            # check if the model is the best model so far, if yes, save it
            if best_trial.objective > best_objective:
                best_objective = copy.deepcopy(best_trial.objective)
                best_model_params = copy.deepcopy(best_trial)

            # saving best trial
            best_ray_trials.append(best_trial)
    else:
        shuffled_data = encoded_datasets['train'].shuffle(seed=42)
        splitted_data = shuffled_data.train_test_split(test_size=0.1)

        train_set = splitted_data['train']
        dev_set = splitted_data['test']

        best_trial = run_hyperparameter_tuning(train_set, dev_set)

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

    # setting hyperparameters based on best trial
    learning_rate = best_model_params.hyperparameters['learning_rate']
    num_train_epochs = best_model_params.hyperparameters['num_train_epochs']
    per_device_train_batch_size = best_model_params.hyperparameters['per_device_train_batch_size']

random_seed_results = []

# shuffling train and test before running with random seeds
shuffled_train = encoded_datasets['train'].shuffle(seed=42)
shuffled_test = encoded_datasets['test'].shuffle(seed=42)

# now, fine-tuning the model with the best set of hyperparameters and evaluate it on the test set
for random_seed in random_seeds:
    args = TrainingArguments(
        running_output_path,
        learning_rate=learning_rate if params['hyperparameter_search'] == 1 else params['learning_rate'],
        num_train_epochs=num_train_epochs if params['hyperparameter_search'] == 1 else params['num_train_epochs'],
        per_device_train_batch_size=per_device_train_batch_size if params['hyperparameter_search'] == 1 else params[
            'per_device_train_batch_size'],
        do_train=True,
        do_eval=True,
        seed=random_seed,
    )

    if task_type == 'multi':
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=shuffled_train if args.do_train else None,
            eval_dataset=shuffled_test if args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer),
            compute_metrics=compute_metrics,
        )
    elif task_type in ['seq', 'nsp']:
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=shuffled_train if args.do_train else None,
            eval_dataset=shuffled_test if args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )

    trainer.train()

    result = trainer.evaluate()

    random_seed_results.append(result['eval_accuracy'])

print('====================================')
print(" *** Report on random seed runs *** ")
print(random_seed_results)
print('\n\nAverage performance: {}'.format(round(sum(random_seed_results) / len(random_seed_results), 3)))
print('==========================================')
print("*** Best hyperparameter tuning run ***")
print(best_model_params)
