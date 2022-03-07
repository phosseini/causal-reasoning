import csv
import json

# OpenBookQA Dataset: https://allenai.org/data/open-book-qa

label2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

data_path = 'data/obqa/Additional/'

all_data = {'train': data_path + 'train_complete.jsonl',
            'dev': data_path + 'dev_complete.jsonl',
            'test': data_path + 'test_complete.jsonl'}

col_names = ['question', 'ending0', 'ending1', 'ending2', 'ending3', 'label']

for data_split, data_split_path in all_data.items():
    with open(data_path + '{}_obqa.csv'.format(data_split), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(col_names)
        with open(data_split_path, 'r') as f_in:
            lines = f_in.readlines()
            for line in lines:
                line = json.loads(line.strip())
                question = line['question']['stem']
                choices = dict()
                for choice in line['question']['choices']:
                    choices[choice['label']] = choice['text']
                choice0 = choices['A']
                choice1 = choices['B']
                choice2 = choices['C']
                choice3 = choices['D']
                label = label2id[line['answerKey']]
                writer.writerow([question, choice0, choice1, choice2, choice3, label])
