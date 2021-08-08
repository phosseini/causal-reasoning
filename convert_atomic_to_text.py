# prerequisite
# ! pip install --upgrade language_tool_python

import re
import csv
import copy
import spacy
import pandas as pd
import language_tool_python

from utils import get_atomic_relation_map

nlp = spacy.load("en_core_web_sm")
grammar_tool = language_tool_python.LanguageTool('en-US')


def check_pos(doc, pos_list=[]):
    for token in doc:
        if token.pos_ in pos_list:
            return True
    return False


def normalize_string(a, replace_with_special_tokens=False):
    """
    normalizing an element in ATOMIC's triple
    :param a:
    :param replace_with_special_tokens:
    :return:
    """
    a = a.replace("'d", 'would')

    replacements = {'[.+]': ' ', ' +': ' ', '___': '[MASK]',
                    'person x': 'PersonX', 'person y': 'PersonY', 'Person x': 'PersonX', 'Person y': 'PersonY',
                    'person X': 'PersonX', 'person Y': 'PersonY', 'Person X': 'PersonX', 'Person Y': 'PersonY',
                    'personX': 'PersonX', 'personY': 'PersonY', 'personx': 'PersonX', 'persony': 'PersonY',
                    ' X ': ' PersonX ', ' Y ': ' PersonY '}

    for k, v in replacements.items():
        a = re.sub(k, v, a)

    # checking if there's any X or Y at the start or end of tuple elements
    text_starts = ['X ', 'Y ']
    text_ends = [' X', ' Y']

    for text_start in text_starts:
        if a.startswith(text_start):
            a = re.sub(text_start, 'Person{} '.format(text_start.strip()), a)

    for text_end in text_ends:
        if a.endswith(text_end):
            a = re.sub(text_end, ' Person{}'.format(text_end.strip()), a)

    if replace_with_special_tokens:
        # replacing PersonX and PersonY with [unused1] and [unused2] tokens, respectively
        # Based on Jacob Devlin's recommendation (https://github.com/google-research/bert/issues/9),
        # we use these [unused*] tokens since they are randomly initialized and may be a good replacement
        # for PersonX and PersonY that are not in BERT's vocabulary.
        a = re.sub('PersonX', '[unused1]', a)
        a = re.sub('PersonY', '[unused2]', a)

    return a.strip()


def lower_nth(s, n):
    return s[:n] + s[n].lower() + s[n + 1:]


def capitalize_nth(s, n):
    return s[:n] + s[n].capitalize() + s[n + 1:]


def normalize_for_grammar_check(text_in):
    text_in = text_in.replace('PersonX', 'Person')
    text_in = text_in.replace('PersonY', 'Person')
    return capitalize_nth(text_in, 0)


rel_map = get_atomic_relation_map(return_modified_templates=True)

# there are three splits for ATOMIC-2020: train.tsv, dev.tsv, test.tsv
data_path = 'data/atomic2020_data-feb2021'
data_splits = ['train']

pattern = re.compile("([P|p]erson[A-Z|a-z])")

grammar_errors = []
relation_type_filter = []
# three possible categories: event, physical, social
relation_category_filter = ['event', 'social', 'physical']
rels = {}

check_grammar = True
check_pos_flag = False

num_records = 0
count_duplicates = 0
saving_step = 10000  # using to flush data into the csv/txt files
logging_step = 20000  # using just to show progress

tmp = set()  # using as a temporary memory to check duplicate rows

with open('data/pretraining.txt', 'w') as txt_file, open('data/pretraining.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)

    for data_split in data_splits:
        sents_1 = []
        sents_2 = []
        relations = []

        # loading data (triples) from ATOMIC
        df = pd.read_csv('{}/{}.tsv'.format(data_path, data_split), sep='\t', header=None)
        df = df.sample(frac=1)

        print('data is loaded from {} split.'.format(data_split))

        # file header
        csv_writer.writerow(["text", "relation_category", "relation_type", "modified"])

        # ---------------------------------------------------
        for idx, row in df.iterrows():
            row = [str(r) for r in row]
            relation_type = copy.deepcopy(row[1])
            relation_category = rel_map[row[1]][2]

            if row[2].strip() != 'none' \
                    and not any(item.isupper() for item in row) \
                    and (len(relation_category_filter) == 0 or (
                    len(relation_category_filter) != 0 and relation_category in relation_category_filter)) \
                    and (len(relation_type_filter) == 0 or (
                    len(relation_type_filter) != 0 and relation_type in relation_type_filter)):

                # normalizing triple elements
                row[0] = normalize_string(row[0], replace_with_special_tokens=False)
                row[1] = normalize_string(rel_map[row[1]][0], replace_with_special_tokens=False)
                row[2] = normalize_string(row[2], replace_with_special_tokens=False)

                # checking duplicate values
                if str(row) not in tmp:
                    tmp.add(str(row))
                    relations.append(relation_type)
                    sents_1.append('{}'.format(capitalize_nth(row[0], 0)))
                    sents_2.append('{} {}'.format(row[1], lower_nth(row[2], 0)))
                else:
                    count_duplicates += 1

        del tmp
        # ---------------------------------------------------
        # since we wanted to batch process sentences/docs in spacy, we stored them all in sents_1 and sents_2
        if check_pos_flag:
            # now, batch processing documents for POS tagging
            print('batch preprocessing documents...')
            docs_1 = list(nlp.pipe(sents_1, n_process=2))
            docs_2 = list(nlp.pipe(sents_2, n_process=2))
            print('batch preprocessing documents is done.')
        else:
            docs_1 = copy.deepcopy(sents_1)
            docs_2 = copy.deepcopy(sents_2)

        assert len(docs_1) == len(sents_1)
        assert len(docs_2) == len(sents_2)

        # free the memory since we don't need these two lists any more
        del sents_1
        del sents_2

        for j in range(len(docs_1)):
            if not check_pos_flag:
                sent_1_text = docs_1[j]
                sent_2_text = docs_2[j]
            else:
                sent_1_text = docs_1[j].text
                sent_2_text = docs_2[j].text

            sent_1 = normalize_string(sent_1_text, replace_with_special_tokens=True)
            sent_2 = normalize_string(sent_2_text, replace_with_special_tokens=True)

            if pattern.search(sent_1) is None and pattern.search(sent_2) is None:

                # if we want to check existence of a VERB in a sentence
                # we do the VERB checking if our goal is to prepare the data for Next Sentence Prediction (NSP) training
                # the reason is that in NSP, we ideally want actual sentences not just some random chunks of text
                if not check_pos_flag or (
                        check_pos_flag and check_pos(docs_1[j], pos_list=['VERB']) and check_pos(docs_2[j],
                                                                                                 pos_list=['VERB'])):

                    # ------------------------------------------------------------------------------
                    # check the grammar to make sure sentences are most likely grammatically correct
                    if rel_map[relations[j]][1] == 0:
                        example = '{} {}\n\n'.format(sent_1, sent_2)
                    elif rel_map[relations[j]][1] == 1:
                        example = sent_1 + '. ' + capitalize_nth(sent_2, 0) + '.\n\n'

                    # correct possible grammatical errors
                    if check_grammar:
                        corrected_example = grammar_tool.correct(example)
                    else:
                        corrected_example = copy.deepcopy(example)

                    modified = 0  # to flag grammatically corrected examples

                    if example != corrected_example:
                        modified = 1
                        example = copy.deepcopy(corrected_example)
                    # ------------------------------------------------------------------------------

                    if relations[j] in rels:
                        rels[relations[j]] += 1
                    else:
                        rels[relations[j]] = 1

                    # TODO: adding a flag for the output format
                    # writing into the text file
                    # txt_file.write(example)

                    # writing into the csv file
                    csv_writer.writerow([example, rel_map[relations[j]][2], relations[j], modified])

                    num_records += 1

            # saving records every saving_step steps
            if num_records % saving_step == 0:
                txt_file.flush()
                csv_file.flush()

            if j % logging_step == 0:
                print('step {}'.format(j))