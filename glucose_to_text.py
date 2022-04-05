import re
import csv
import random
import pandas as pd

from utils import lower_nth


def clean_text(txt):
    txt = txt.replace('. .', '.')
    txt = re.sub(' +', ' ', txt)
    return txt


log_step = 50000
file_path = 'data/glucose_train.csv'
df = pd.read_csv('data/glucose/GLUCOSE_training_data_final.csv')

# quality can be 1-3 with 1 being the lowest and 3 being the best.
df = df[df['worker_quality_rating'].isin([3])]

# "specific" columns contain the actual sentences from the story
# general columns are the general patterns of a relation
# for each entry, there are 10 specific and 10 general columns, respectively
specific_cols = ['{}_specificNL'.format(i) for i in range(1, 11)]
general_cols = ['{}_generalNL'.format(i) for i in range(1, 11)]
templates = [['because', 'since'], ['because', 'since'], ['because', 'since'], ['because', 'since'],
             ['because', 'since'], ['causes', 'caused', 'results in'], ['. As a result'], ['. As a result'],
             ['. As a result'], ['. As a result']]
connectives = ['>Causes/Enables>', '>Motivates>', '>Enables>', '>Enables>', '>Enables>', '>Causes/Enables>', '>Causes>',
               '>Results in>', '>Results in>', '>Results in>']

# X: selected sentence
# 0: X is effect
# 1: X is cause
X_idx = {}
idxs = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

specific_templates = {}
for i in range(len(specific_cols)):
    specific_templates[specific_cols[i]] = [templates[i], connectives[i]]
    X_idx[specific_cols[i]] = idxs[i]

n = 1

with open(file_path, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["text", "connective"])
    for col in specific_cols:
        for idx, row in df[['selected_sentence', col]].iterrows():
            connective = specific_templates[col][1]
            if row[col] != 'escaped' and connective in row[col]:
                template = random.choice(specific_templates[col][0])
                selected_sentence = row['selected_sentence']
                sents = row[col].split(connective)
                if len(sents) == 2:
                    sent_a = sents[0].strip()
                    sent_b = sents[1].strip()

                    # removing '.' from end of sentences
                    sent_a = sent_a[:-1] if sent_a.endswith('.') else sent_a
                    sent_b = sent_b[:-1] if sent_b.endswith('.') else sent_b
                    selected_sentence = selected_sentence[:-1] if selected_sentence.endswith('.') else selected_sentence

                    if X_idx[col] == 0:
                        X, Y = sent_a, sent_b
                    else:
                        X, Y = sent_b, sent_a

                    # verbalizing the triple
                    verbalized_texts = ['{} {} {}'.format(X, template, lower_nth(Y, 0))]

                    verbalized_texts = [clean_text(text) for text in verbalized_texts]

                    for verbalized_text in verbalized_texts:
                        csv_writer.writerow([verbalized_text, connective])
                        n += 1
                        # show progress and flush records
                        if n % log_step == 0:
                            csv_file.flush()
                            print(n)

df_glucose = pd.read_csv(file_path)
print('# records [glucose]: {}'.format(len(df_glucose)))
df_glucose = df_glucose.drop_duplicates()
print('# deduplicated records: {}'.format(len(df_glucose)))
df_glucose.to_csv(file_path)
