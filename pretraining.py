import json

import comet_ml
from datasets import Dataset, DatasetDict
from transformers import (Trainer, TrainingArguments, EarlyStoppingCallback,
                          AutoTokenizer, AutoModelForMaskedLM, set_seed,
                          RobertaForCausalLM, DataCollatorForLanguageModeling,
                          )

# loading parameters
with open('config/pretraining_config.json') as f:
    params = json.load(f)

# special tokens used when converting KG-to-Text
with open('data/special_tokens.txt', 'r') as in_file:
    special_tokens = [line.strip() for line in in_file.readlines()]

model_name = params['model_checkpoint']
tokenizer_name = params['tokenizer_name']
pretraining_method = params['pretraining_method']
train_data = params['train_data']
dev_data = params['dev_data']
max_length = params['max_length']
min_length = params['min_length']
block_size = params['block_size']
output_dir = params['output_dir']
kg_name = params['kg_name']
relation_category = params['relation_category']
create_dev = params['create_dev']
text_field = params['text_field']

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_example(example):
    example[text_field] = example[text_field].replace('\n', '')
    return example


def encode(examples):
    # since the data collator (DataCollatorForLanguageModeling) dynamically pads the input examples,
    # we skip padding when we are tokenizing the input examples and only do the truncation
    return tokenizer(examples[text_field], max_length=max_length, truncation=True, padding=False)


dataset = DatasetDict()
dataset['train'] = Dataset.from_csv(train_data)

if create_dev == 1:
    train_eval = dataset["train"].train_test_split(test_size=0.1)
    dataset["train"] = train_eval['train']
    dataset["dev"] = train_eval['test']
else:
    dataset['dev'] = Dataset.from_csv(dev_data)

if kg_name == "atomic":
    dataset = dataset.filter(lambda example: example['relation_category'] in relation_category and example[
        text_field] != '' and "[MASK]" not in example[text_field] and (
                                                     min_length < len(example[text_field].split(' '))))

dataset = dataset.map(preprocess_example)
dataset = dataset.map(encode, batched=True)

if params['batch_training'] == 1:
    dataset = dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
    )

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=params['num_train_epochs'],
    learning_rate=params['learning_rate'],
    weight_decay=params['weight_decay'],
    evaluation_strategy=params['eval_save_strategy'],
    save_strategy=params['eval_save_strategy'],
    eval_steps=params['eval_steps'],
    save_steps=params['save_steps'],
    logging_steps=params['logging_steps'],
    per_device_train_batch_size=params['batch_size'],
    per_device_eval_batch_size=params['batch_size'],
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    gradient_accumulation_steps=params['gradient_accumulation_steps'],
)


def get_model():
    if pretraining_method == 'mlm':
        return AutoModelForMaskedLM.from_pretrained(model_name)
    elif pretraining_method == "clm":
        return RobertaForCausalLM.from_pretrained(model_name)


trainer = Trainer(
    model_init=get_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=params['early_stopping_patience'])]
)

trainer.train()

trainer.save_model()
