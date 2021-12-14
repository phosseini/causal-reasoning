import os
import re
import csv
import json
import copy
import pandas as pd
import language_tool_python

from utils import get_atomic_relation_templates
from utils import lower_nth, capitalize_nth

# loading parameters
config_path = '../config/atomic_conversion_config.json'
if os.path.exists(config_path):
    with open(config_path) as f:
        params = json.load(f)
else:
    raise FileNotFoundError('Please put the config file in the following path: ./config/atomic_conversion_config.json')

# loading models
grammar_tool = language_tool_python.LanguageTool('en-US')

special_tokens = {'[unused1]': '[unused1]'}
pattern = re.compile("([P|p]erson[A-Z|a-z])")

max_samples = params['max_samples']  # will be considered only if greater than 0
data_path = params['data_path']
check_grammar = params['check_grammar']
saving_step = params['saving_step']  # using to flush data into the csv/txt files
logging_step = params['logging_step']  # using just to show progress
output_file_directory = params['output_file_directory']
data_splits = params['data_splits']  # ATOMIC-2020 has three splits: train.tsv, dev.tsv, test.tsv
relation_filter = params['relation_filter']
relation_category_filter = params['relation_category_filter']  # three possible categories: event, physical, social

num_records = 0
count_duplicates = 0
relations_count = {}
grammar_errors = []
names_replacement = {'PersonX': 'Tracy', 'PersonY': 'Riley', 'PersonZ': 'Jordan'}


def get_special_token(w):
    """
    this method is taken from SpanBERT's repo: https://github.com/facebookresearch/SpanBERT/blob/main/code/run_tacred.py#L120
    :param w: a token/word
    :return:
    """
    if w not in special_tokens:
        special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
    return special_tokens[w]


def normalize_string(element: str):
    """
    normalizing an element in an ATOMIC triple where element is head, relation, or target
    :param element: either head or tail in a KG triple
    :return:
    """

    replacements = {'[.+]': ' ', ' +': ' ', '___': '[MASK]',
                    'person x': 'PersonX', 'person y': 'PersonY', 'Person x': 'PersonX', 'Person y': 'PersonY',
                    'person X': 'PersonX', 'person Y': 'PersonY', 'Person X': 'PersonX', 'Person Y': 'PersonY',
                    'personX': 'PersonX', 'personY': 'PersonY', 'personx': 'PersonX', 'persony': 'PersonY',
                    'Personx': 'PersonX', 'Persony': 'PersonY',
                    ' X ': ' PersonX ', ' Y ': ' PersonY ', ' x ': ' PersonX ', ' y ': ' PersonY '}

    for k, v in replacements.items():
        element = re.sub(k, v, element)

    # checking if there's any X or Y at the start or end of tuple elements
    text_starts = ['X ', 'x ', 'Y ', 'y ']
    text_ends = [' X', ' x', ' Y', ' y']

    for text_start in text_starts:
        if element.startswith(text_start):
            element = 'Person{} '.format(capitalize_nth(text_start.strip(), 0)) + element[len(text_start):]

    for text_end in text_ends:
        if element.endswith(text_end):
            element = element[:len(text_end)] + ' Person{}'.format(capitalize_nth(text_end.strip(), 0))

    return element.strip()


def replace_tokens(text):
    """
    replacing tokens in a verbalized KG triple
    :return:
    """
    for token_word, token_replacement in special_tokens.items():
        text = text.replace(token_word, token_replacement)
    return text


relation_templates = get_atomic_relation_templates()

tmp = set()  # using as a temporary memory to check duplicate rows

for data_split in data_splits:
    with open(output_file_directory + "atomic2020_{}.txt".format(data_split), 'w') as txt_file, open(
            output_file_directory + "atomic2020_{}.csv".format(data_split),
            'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        # csv file header
        # all *_text fields are created by concatenating head, relation, and tail in a knowledge graph triple
        # verbalized_text: verbalized KG triple
        # modified_text: the modified verbalized_text. Modification includes grammar correction, token replacement, etc.
        # triple_text: non-verbalized triple (simple concatenation of head, relation, and tail)
        # relation_category: one of the following categories: ['event', 'social', 'physical']
        # relation_type: relation type in ATOMIC 2020
        # grammar_modified: whether the text is grammatically corrected (1) or not (0)

        csv_writer.writerow(["verbalized_text", "modified_text", "triple_text", "relation_category", "relation_type",
                             "corrected_grammar"])

        # loading data (triples) from ATOMIC
        df = pd.read_csv('{}/{}.tsv'.format(data_path, data_split), sep='\t', header=None)
        df = df.sample(frac=1)

        print('data is loaded successfully from [{}] splits.'.format(data_split))

        i = 1
        for idx, row in df.iterrows():
            row = [str(r) for r in row]
            relation_type = copy.deepcopy(row[1])
            relation_category = relation_templates[relation_type][3]

            if row[2].strip() != 'none' \
                    and not any(item.isupper() for item in row) \
                    and (len(relation_category_filter) == 0 or (
                    len(relation_category_filter) != 0 and relation_category in relation_category_filter)) \
                    and (len(relation_filter) == 0 or (
                    len(relation_filter) != 0 and relation_type in relation_filter)):

                start_exception = ['PersonX', 'PersonY', 'PersonZ']

                # normalizing triple elements
                head = normalize_string(row[0])
                tail = normalize_string(row[2])
                verbalized_relation = relation_templates[row[1]][1]

                # triple_text will be used when we want to do MLM only using the triples with no KG-to-text conversion
                triple_text = '{} {} {}'.format(head, get_special_token(row[1]), tail)

                # checking duplicate values
                if str(triple_text) not in tmp:
                    tmp.add(str(triple_text))

                    # verbalizing the triple
                    segment_a = '{}'.format(capitalize_nth(head, 0))
                    if any(tail.startswith(exception) for exception in start_exception):
                        segment_b = '{} {}'.format(verbalized_relation, tail)
                    else:
                        segment_b = '{} {}'.format(verbalized_relation, lower_nth(tail, 0))

                    if relation_templates[relation_type][2] == 0:
                        verbalized_triple = '{} {}\n\n'.format(segment_a, segment_b)
                    elif relation_templates[relation_type][2] == 1:
                        verbalized_triple = segment_a + '. ' + capitalize_nth(segment_b, 0) + '.\n\n'

                    # modifying the verbalized triple (e.g., grammar correction, etc.)
                    modified_text = copy.deepcopy(verbalized_triple)

                    # replacing the names
                    for token, name_replacement in names_replacement.items():
                        modified_text = modified_text.replace(token, name_replacement)

                    if pattern.search(modified_text) is None:
                        if relation_type in relations_count:
                            relations_count[relation_type] += 1
                        else:
                            relations_count[relation_type] = 1

                        # ------------------------------------------------------------------------------
                        # correct possible grammatical errors
                        corrected_text = grammar_tool.correct(modified_text) if check_grammar == 1 else modified_text

                        # flag grammatically corrected examples
                        corrected = 1 if corrected_text != modified_text else 0
                        # ------------------------------------------------------------------------------

                        # writing into the text file
                        txt_file.write(corrected_text)

                        # writing into the csv file
                        csv_writer.writerow(
                            [verbalized_triple.strip(), corrected_text.strip(), triple_text.strip(), relation_category,
                             relation_type, corrected])

                        num_records += 1

                        # saving records every saving_step steps
                        if num_records % saving_step == 0:
                            txt_file.flush()
                            csv_file.flush()
                else:
                    count_duplicates += 1

            if i % logging_step == 0:
                print('step {}'.format(i))
            i += 1
            # in case we do not want to convert ALL triples and only want a small sample of converted triples
            if 0 < max_samples < num_records:
                break

        del tmp

# writing the special tokens into a file
special_tokens_file_path = "../data/special_tokens.txt"
relations_count_file_path = "../data/relations_count.txt"

with open(special_tokens_file_path, 'w') as out_file:
    for token, special_token in special_tokens.items():
        out_file.write(special_token + '\n')

# writing relations' count into a file
relations_count = {k: v for k, v in sorted(relations_count.items(), key=lambda item: item[1])}
with open(relations_count_file_path, 'w') as out_file:
    for relation, count in relations_count.items():
        out_file.write('{}: {}\n'.format(relation, count))
    out_file.write('\ntotal: {}\n'.format(sum(relations_count.values())))

print('ATOMIC2020-to-text conversion is done successfully.')
print('number of all converted triples: {}'.format(num_records))
print('number of duplicates (final output is deduplicated): {}'.format(count_duplicates))
