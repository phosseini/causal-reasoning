import os
import json
import torch
import statistics
import numpy as np
import pandas as pd
from ray import tune
from dataclasses import dataclass
from typing import Optional, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import (DatasetDict, Dataset)
from transformers import (AutoModelForMultipleChoice, PreTrainedTokenizerBase,
                          AutoTokenizer, TrainingArguments, Trainer)
from transformers.tokenization_utils_base import PaddingStrategy

from utils import lower_nth


def compute_metrics_v1(pred):
    """
    a sightly different implementation of computing metrics, essentially same as compute_metrics
    :param pred:
    :return:
    """
    predictions = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    label_ids = pred.label_ids
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


# loading parameters
with open('config/fine_tuning_config.json') as f:
    params = json.load(f)

os.environ["WANDB_API_KEY"] = params['WANDB_API_KEY']

# creating a dataframe to save results
df_results = pd.DataFrame()
task_type = params['task_type']
model_checkpoint = params['model_checkpoint']
random_seeds = params['random_seeds']
tokenizer_name = params['tokenizer_name']
running_output_path = params['running_output_path']
tuning_output_path = params['tuning_output_path']

# ending0 and ending1 are the two choices for each question
ending_names = ["ending0", "ending1"]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)


def preprocess_function(examples, task=params['task_type'], prompt=params['add_prompt_to_test']):
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


raw_datasets = DatasetDict()
raw_datasets['train'] = Dataset.from_csv(params['train_data'])
raw_datasets['test'] = Dataset.from_csv(params['test_data'])
train_dataset = raw_datasets['train']
test_dataset = raw_datasets['test']

le = LabelEncoder()
le.fit_transform(train_dataset['label'])

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
)
test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
)

# since COPA doesn't have separate train and dev sets, we split its dev set into train and dev
train_dev_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)


def get_model():
    return AutoModelForMultipleChoice.from_pretrained(model_checkpoint)


training_args = TrainingArguments(
    output_dir=running_output_path,  # output directory
    evaluation_strategy="steps",
    report_to="wandb",
    disable_tqdm=True,
    seed=42
)

trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dev_dataset['train'],
    eval_dataset=train_dev_dataset['test'],
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    model_init=get_model,
    compute_metrics=compute_metrics
)

tune_config = {
    "per_device_train_batch_size": tune.grid_search(params['tuning_batch_size']),
    "num_train_epochs": tune.grid_search(params['tuning_num_train_epochs']),
    "learning_rate": tune.grid_search(params['tuning_learning_rate'])
}

best_trial = trainer.hyperparameter_search(
    hp_space=lambda _: tune_config,
    backend="ray",
    direction='maximize',
    n_trials=params['n_trials'],
    verbose=1,
    resources_per_trial={
        "cpu": params['resources_per_trial']['cpu'],  # make sure to change your resources accordingly
        "gpu": params['resources_per_trial']['gpu']
    },
    keep_checkpoints_num=0,
    local_dir="./ray_results/",
    log_to_file=True)

# updating hyperparameters using best trial
for n, v in best_trial.hyperparameters.items():
    setattr(training_args, n, v)

print("*** best hyperparameter values ***")
print(training_args)

print("*** best trial ***")
print(best_trial)

test_results = {'test_accuracy': list()}
df_test_dataset = test_dataset.to_pandas()
df_results['text'] = df_test_dataset['startphrase'] + " [0] " + df_test_dataset['ending0'] + " [1] " + df_test_dataset[
    'ending1']

for run in params['random_seeds']:
    setattr(training_args, 'seed', run)
    trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dev_dataset['train'],
        eval_dataset=train_dev_dataset['test'],
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        model_init=get_model,
        compute_metrics=compute_metrics
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)

    accuracy = compute_metrics(predictions)
    test_results['test_accuracy'].append(accuracy['accuracy'])
    predicted = le.inverse_transform(predictions.predictions.argmax(-1))
    labels = le.inverse_transform(test_dataset['label'])
    df_results['predicted_{}'.format(run)] = predicted
    df_results['labels_{}'.format(run)] = labels

    assert accuracy_score(labels, predicted) == accuracy['accuracy']

    trainer.save_model('{}/model_seed_{}'.format(running_output_path, run))

# saving prediction results
df_results.to_csv('{}/predictions.csv'.format(running_output_path))

print("=======================")
print("*** results on test ***")
print(test_results)
for metric, values in test_results.items():
    print('{}: mean = {}'.format(metric, statistics.mean(values)))
    if len(values) > 1:  # variance requires at least two data points
        print('{}: std = {}'.format(metric, statistics.stdev(values)))
