import json
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForMaskedLM, AutoTokenizer, RobertaForCausalLM
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask

# loading parameters
with open('../config/pretraining_config.json') as f:
    params = json.load(f)
# special tokens used when converting KG-to-Text
with open('../data/special_tokens.txt', 'r') as in_file:
    special_tokens = [line.strip() for line in in_file.readlines()]

model_name = params['model_checkpoint']
tokenizer_name = params['tokenizer_name']
pretraining_method = params['pretraining_method']
train_data = params['train_data']
dev_data = params['dev_data']
max_length = params['max_length']
output_dir = params['output_dir']
relation_category = params['relation_category']
num_train_epochs = params['num_train_epochs']
learning_rate = params['learning_rate']
save_steps = params['save_steps']
logging_steps = params['logging_steps']
per_device_train_batch_size = params['per_device_train_batch_size']
early_stopping_patience = params['early_stopping_patience']

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

text_field_name = 'modified_text'


def model_init():
    if pretraining_method == 'mlm':
        return AutoModelForMaskedLM.from_pretrained(model_name)
    elif pretraining_method == "clm":
        return RobertaForCausalLM.from_pretrained(model_name)


def remove_newline(example):
    example[text_field_name] = example[text_field_name].replace('\n', '')
    return example


def encode(examples):
    # since the data collator (DataCollatorForLanguageModeling) dynamically pads the input examples,
    # we skip padding when we are tokenizing the input examples and only do the truncation
    return tokenizer(examples[text_field_name], max_length=max_length, truncation=True)


# preparing the data for pretraining
dataset = DatasetDict()
dataset['train'] = Dataset.from_csv(train_data)
dataset['dev'] = Dataset.from_csv(dev_data)
dataset = dataset.filter(lambda example: example['relation_category'] in relation_category and example[
    text_field_name] != '' and "[MASK]" not in example[text_field_name])

# train and dev have the same column names
remove_columns = dataset['train'].column_names  # saving column names before tokenizations
dataset = dataset.map(remove_newline)
dataset = dataset.map(encode, batched=True)
dataset = dataset.remove_columns(remove_columns)

# shuffle and select
dataset = dataset.shuffle(seed=42)

if 'roberta' in model_name:
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
else:
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

data_collator_lm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

data_collator_wwm = DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

training_args = TrainingArguments(
    output_dir=output_dir,
    do_train=True,
    do_eval=True,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    evaluation_strategy='steps',
    save_strategy='steps',
    save_steps=save_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    prediction_loss_only=True,
    load_best_model_at_end=True
)

# if we want to split either train or dev further
train_eval = dataset["train"].train_test_split(test_size=0.1)
# for now, we did not use ATOMIC's dev set, and we split the train set into train and dev
# if we want to use ATOMIC's dev set, we replace train_eval["test"] with dataset['dev'] in trainer

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_eval["train"],
    eval_dataset=train_eval["test"],
    data_collator=data_collator_lm,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
)

trainer.train()

trainer.save_model()
