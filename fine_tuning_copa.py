import json
import torch
import numpy as np
from ray import tune
from typing import Optional, Union
from dataclasses import dataclass
from datasets import (DatasetDict, Dataset)
from transformers import (AutoModelForMultipleChoice, PreTrainedTokenizerBase,
                          AutoTokenizer, TrainingArguments, Trainer, set_seed)
from transformers.tokenization_utils_base import PaddingStrategy

from utils import lower_nth


def compute_metrics(eval_predictions):
    predictions = eval_predictions.predictions[0] if isinstance(eval_predictions.predictions,
                                                                tuple) else eval_predictions.predictions
    label_ids = eval_predictions.label_ids
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


# loading parameters
with open('config/fine_tuning_config.json') as f:
    params = json.load(f)

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

set_seed(42)

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

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
)
test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
)

train_dev_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)


def model_init():
    return AutoModelForMultipleChoice.from_pretrained(model_checkpoint)


training_args = TrainingArguments(
    tuning_output_path,
    evaluation_strategy="steps",
    disable_tqdm=True,
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dev_dataset['train'],
    eval_dataset=train_dev_dataset['test'],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
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
    resources_per_trial={
        "cpu": 1,
        "gpu": 1
    },
    keep_checkpoints_num=0,
    log_to_file=True)

for random_seed in random_seeds:
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    for n, v in best_trial.hyperparameters.items():
        setattr(trainer.args, n, v)
    setattr(trainer.args, 'seed', random_seed)

    trainer.train()

    result = trainer.evaluate()

    random_seed_results.append(result['eval_accuracy'])

print("*** random seed runs ***")
print(random_seed_results)
print('\n\n*** average performance: {}'.format(round(sum(random_seed_results) / len(random_seed_results), 3)))
print('*** standard deviation: {}'.format(round(np.std(random_seed_results), 3)))
print(best_trial)
