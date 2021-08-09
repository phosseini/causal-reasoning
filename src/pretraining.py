import json
import torch
# torch_xla.* is related to using TPU (we used Google Colab TPU v2)
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, AutoTokenizer, BertForPreTraining, BertForMaskedLM
from transformers import TextDatasetForNextSentencePrediction, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask

tpu_device = xm.xla_device()
print(tpu_device)

# ------------------------------
# loading parameters
with open('../config/fine_tuning_config.json') as f:
    params = json.load(f)

model_name = params['model_name']
tokenizer_name = params['tokenizer_name']
pretraining_method = params['pretraining_method']
pretraining_input = params['pretraining_input']
max_length = params['max_length']
output_dir = params['output_dir']
relation_category = params['relation_category']
num_train_epochs = params['num_train_epochs']
learning_rate = params['learning_rate']
save_steps = params['save_steps']
logging_steps = params['logging_steps']
per_device_train_batch_size = params['per_device_train_batch_size']

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]", "[unused1]"]})

if pretraining_method == 'mlm':
    model = AutoModelForMaskedLM.from_pretrained(model_name)
elif pretraining_method == 'mlm_nsp':
    model = BertForPreTraining.from_pretrained(model_name)

"""
If using Next Sentence Prediction (NSP) too, for now use an input TXT file, instead of the CSV file, with the following format: 
every line is a sentence from a document. Documents are separated by an empty line. For example:

sentence 1.1
sentence 1.2
[empty line]
sentence 2.1
sentence 2.2
[empty line]
...
"""

# preparing the data for pretraining
if pretraining_method == 'mlm_nsp':
    dataset = TextDatasetForNextSentencePrediction(
        tokenizer=tokenizer,
        file_path=pretraining_input,
        block_size=max_length,
    )
elif pretraining_method == 'mlm':
    def remove_newline(example):
        example['text'] = example['text'].replace('\n', '')
        return example


    def encode(examples):
        # since the data collator (DataCollatorForLanguageModeling) dynamically pads the input examples,
        # we skip padding when we are tokenizing the input examples and only do the truncation
        return tokenizer(examples['text'], max_length=max_length, truncation=True)


    dataset = Dataset.from_csv(pretraining_input)
    dataset = dataset.filter(
        lambda example: example['relation_category'] in relation_category and example['text'] != '' and "[MASK]" not in
                        example['text'])
    dataset = dataset.map(remove_newline)
    dataset = dataset.map(encode, batched=True)
    dataset = dataset.remove_columns(['text', 'relation_category', 'relation_type', 'modified'])

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

# ==========================================
# ========== starting pretraining ==========
# ==========================================

shuffled_dataset = dataset.shuffle(seed=42)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    save_steps=save_steps,
    logging_steps=logging_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=shuffled_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator_lm,
)

trainer.train()

trainer.save_model()
