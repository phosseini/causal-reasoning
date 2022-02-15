import json
import torch
import numpy as np
from ray import tune
from typing import Optional, Union
from dataclasses import dataclass
from datasets import (DatasetDict, Dataset)
from transformers import (AutoModelForSequenceClassification, AutoModelForMultipleChoice,
                          AutoModelForNextSentencePrediction, DataCollatorWithPadding,
                          AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerBase, EarlyStoppingCallback)
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.trainer_utils import IntervalStrategy
from ray.tune.schedulers import PopulationBasedTraining
from sklearn.model_selection import KFold

from utils import lower_nth


def compute_metrics(eval_predictions):
    # predictions, label_ids = eval_predictions
    predictions = eval_predictions.predictions[0] if isinstance(eval_predictions.predictions,
                                                                tuple) else eval_predictions.predictions
    label_ids = eval_predictions.label_ids
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


# loading parameters
with open('config/fine_tuning_config.json') as f:
    params = json.load(f)

n_fold = params['n_fold']
task_type = params['task_type']
model_checkpoint = params['model_checkpoint']
random_seeds = params['random_seeds']
tokenizer_name = params['tokenizer_name']
experiment_name = params['experiment_name']
running_output_path = params['running_output_path']
tuning_output_path = params['tuning_output_path']

# ending0 and ending1 are the two choices for each question
ending_names = ["ending0", "ending1"]

output = []
random_seed_results = []
tuning_vars = {'best_tuning_objective': 0,
               'best_tuning_trials': [],
               'best_tuning_params': {}}

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def preprocess_function(examples, task=params['task_type'], prompt=params['add_prompt_to_test']):
    # checking task value:
    if task not in ['seq', 'multi', 'nsp']:
        print("Task value should be one of the following: \'seq\' or \'multi\' or \'nsp\'")
        return

    if task == 'multi':
        # Repeat each first sentence two times to go with the two possibilities of second sentences.
        first_sentences = [[context] * 2 for context in examples["sent1"]]
        # Grab all second sentences possible for each context.
        question_headers = examples["sent2"]
        if prompt == 1:
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


def tune_config_optuna(trial):
    optuna_config = {}
    optuna_config["num_train_epochs"] = trial.suggest_int("num_train_epochs", 3, 4, 5)
    optuna_config["per_device_train_batch_size"] = trial.suggest_categorical("per_device_train_batch_size",
                                                                             params['batch_size'])
    if params['learning_rate_range'] == 1:
        optuna_config["learning_rate"] = trial.suggest_float("learning_rate", params['learning_rate_start'],
                                                             params['learning_rate_end'], log=True)
    return optuna_config


def tune_config_ray(trial):
    ray_config = {}
    if params['learning_rate_range'] == 1:
        ray_config["learning_rate"] = tune.loguniform(params['learning_rate_start'], params['learning_rate_end'])
    else:
        ray_config["learning_rate"] = tune.grid_search(params['tuning_learning_rate'])

    ray_config["num_train_epochs"] = tune.grid_search(params['tuning_num_train_epochs'])
    ray_config["per_device_train_batch_size"] = tune.grid_search(params['tuning_batch_size'])
    return ray_config


def model_init():
    if task_type == 'multi':
        return AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
    elif task_type == 'seq':
        return AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    elif task_type == 'nsp':
        return AutoModelForNextSentencePrediction.from_pretrained(model_checkpoint)


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that dynamically pads the inputs for multiple choice received.
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


encoded_datasets = DatasetDict()
encoded_datasets['train'] = Dataset.from_csv(params['train_data_path']).map(preprocess_function, batched=True)
encoded_datasets['test'] = Dataset.from_csv(params['test_data_path']).map(preprocess_function, batched=True)

columns_to_return = ['input_ids', 'label', 'attention_mask']
if 'roberta' not in tokenizer_name:
    columns_to_return.append('token_type_ids')

encoded_datasets.set_format(type='torch', columns=columns_to_return)

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="eval_accuracy",
    mode="max",
    perturbation_interval=1,
    hyperparam_mutations={
        "weight_decay": tune.uniform(0.0, 0.3),
        "learning_rate": tune.uniform(params['tuning_learning_rate_start'], params['tuning_learning_rate_end']),
        "per_device_train_batch_size": params['tuning_batch_size'],
    })


# since we don't have training set in COPA, we run cross-validation for hyperparameter tuning
# obviously, we DON'T do the hyperparameter tuning on test set to avoid leakage

def run_tuning(data_train, data_dev):
    tuning_args = TrainingArguments(
        tuning_output_path,
        do_train=True,
        do_eval=True,
        evaluation_strategy=IntervalStrategy.EPOCH,
        disable_tqdm=True,
    )

    tuning_trainer = Trainer(
        model_init=model_init,
        args=tuning_args,
        train_dataset=data_train,
        eval_dataset=data_dev,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    best_tuning_trial = tuning_trainer.hyperparameter_search(hp_space=tune_config_ray,
                                                             backend=params['tuning_backend'],
                                                             direction='maximize',
                                                             keep_checkpoints_num=0,
                                                             # if using Ray and PopulationBasedTraining
                                                             n_trials=params['n_trials'],
                                                             # scheduler=scheduler,
                                                             )

    if best_tuning_trial.objective > tuning_vars['best_tuning_objective']:
        tuning_vars['best_tuning_objective'] = best_tuning_trial.objective
        tuning_vars['best_tuning_params'] = best_tuning_trial

    # saving best trial
    tuning_vars['best_tuning_trials'].append(best_tuning_trial)


if params['hyperparameter_search'] == 1:
    if params['cross_validation'] == 1:
        kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
        for train_index, dev_index in kf.split(encoded_datasets['train']):
            train_index = [int(idx) for idx in list(train_index)]
            dev_index = [int(idx) for idx in list(dev_index)]
            train_set = torch.utils.data.dataset.Subset(encoded_datasets['train'], train_index)
            dev_set = torch.utils.data.dataset.Subset(encoded_datasets['train'], dev_index)
            run_tuning(train_set, dev_set)
    else:
        encoded_datasets = encoded_datasets['train'].train_test_split(test_size=0.1)
        train_set = encoded_datasets['train']
        dev_set = encoded_datasets['test']
        run_tuning(train_set, dev_set)

    print('==========================================')
    print("*** all trials ***")
    [print(trial) for trial in tuning_vars['best_tuning_trials']]
    print("*** best accuracy: {}".format(tuning_vars['best_tuning_objective']))
    print("*** best run hyperparameters ***")
    print(tuning_vars['best_tuning_params'])
    print('==========================================')

for random_seed in random_seeds:
    args = TrainingArguments(
        running_output_path,
        learning_rate=params['learning_rate'],
        num_train_epochs=params['num_train_epochs'],
        per_device_train_batch_size=params['batch_size'],
        evaluation_strategy="steps",
        seed=random_seed,
    )

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=encoded_datasets['train'],
        eval_dataset=encoded_datasets['test'],
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    if params['hyperparameter_search'] == 1:
        for n, v in tuning_vars['best_tuning_params'].hyperparameters.items():
            setattr(trainer.args, n, v)

    trainer.train()

    result = trainer.evaluate()

    random_seed_results.append(result['eval_accuracy'])

print("*** random seed runs ***")
print(random_seed_results)
print('\n\n*** average performance: {}'.format(round(sum(random_seed_results) / len(random_seed_results), 3)))
print('*** standard deviation: {}'.format(round(np.std(random_seed_results), 3)))
if params['hyperparameter_search'] == 1:
    print("*** best tuning run ***")
    print(tuning_vars['best_tuning_params'])
