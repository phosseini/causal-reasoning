import os
import sys
import csv
import ast
import json
import random
from utils import convert_copa_data

# Original COPA (`dev + test`): https://people.ict.usc.edu/~gordon/copa.html
# Balanced-COPA (`dev`): https://github.com/Balanced-COPA/Balanced-COPA
# BCOPA-CE (`test`): https://github.com/badbadcode/weakCOPA


# the 'm' extension is for the multi-choice formatted data

dir_path = '../data/copa/'

copa_dev, copa_dev_ordered, copa_dev_m = convert_copa_data(dir_path + 'original/copa-dev.xml')
bcopa_dev, bcopa_dev_ordered, bcopa_dev_m = convert_copa_data(dir_path + 'original/balanced-copa-dev-all.xml')
copa_test, copa_test_ordered, copa_test_m = convert_copa_data(dir_path + 'original/copa-test.xml')
bcopa_ce_test, bcopa_ce_test_ordered, bcopa_ce_test_m = convert_copa_data(dir_path + 'original/BCOPA-CE.xml')

# ---------------------------------------------
# creating easy and hard subsets from COPA-test
# these subsets are defined by:
# Kavumba, Pride, et al. "When Choosing Plausible Alternatives, Clever Hans can be Clever."
# Proceedings of the First Workshop on Commonsense Inference in Natural Language Processing. 2019.

easy_hard_path = 'original/easy_hard_subsets.json'
with open(dir_path + easy_hard_path) as f:
    easy_hard_data = json.load(f)

easy_test_m = []
hard_test_m = []

for item in copa_test_m:
    item_id = int(item['id'])
    if item_id in easy_hard_data['easy']:
        easy_test_m.append(item)
    elif item_id in easy_hard_data['hard']:
        hard_test_m.append(item)
# ---------------------------------------------

all_data = {"copa_dev": copa_dev, "copa_dev_ordered": copa_dev_ordered, "copa_dev_m": copa_dev_m,
            "bcopa_dev": bcopa_dev, "bcopa_dev_ordered": bcopa_dev_ordered, "bcopa_dev_m": bcopa_dev_m,
            "copa_test": copa_test, "copa_test_ordered": copa_test_ordered, "copa_test_m": copa_test_m,
            "bcopa_ce_test": bcopa_ce_test, "bcopa_ce_test_ordered": bcopa_ce_test_ordered,
            "bcopa_ce_test_m": bcopa_ce_test_m,
            "easy_test_m": easy_test_m, "hard_test_m": hard_test_m}

for data_name, data in all_data.items():
    data = list(map(str, data))
    with open(dir_path + "{}.txt".format(data_name), "w") as output:
        output.writelines(f'{s}\n' for s in data)

# ---------------------------------------------

col_names_seq = ['sent1', 'sent2', 'label']

col_names_multi = ['startphrase', 'sent1', 'sent2', 'ending0', 'ending1', 'label']


def create_csv_seq(examples, file_path):
    with open('{}.csv'.format(file_path), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(col_names_seq)
        for example in examples:
            sent1 = example['sent1']
            sent2 = example['sent2']
            label = int(example['label'])
            writer.writerow([sent1, sent2, label])


def create_csv_multi(examples, file_path):
    with open('{}.csv'.format(file_path), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(col_names_multi)
        for example in examples:
            example = ast.literal_eval(example)
            sent1 = example['premise']
            ending0 = example['choice0']
            ending1 = example['choice1']
            label = int(example['label'])
            if 'cause' in example['question']:
                sent2 = 'It is because'
                startphrase = sent1 + ' ' + sent2
            else:
                sent2 = 'As a result,'
                startphrase = sent1 + ' ' + sent2

            writer.writerow([startphrase, sent1, sent2, ending0, ending1, label])


for data_name, data in all_data.items():
    if data_name.endswith('_m'):
        data = list(map(str, data))
        create_csv_multi(data, dir_path + '{}'.format(data_name))
    else:
        create_csv_seq(data, dir_path + '{}'.format(data_name))
