import ast
import json
import copy
import spacy
import pandas as pd


def crest2tacred(df, output_file_name, split=[], source=[], no_order=False, save_json=False):
    """
    converting CREST-formatted data to TACRED (https://nlp.stanford.edu/projects/tacred/)
    :param df: pandas data frame of the CREST-formatted excel file
    :param output_file_name: name of output file without extension
    :param no_order: True if we want to remove spans order, False, otherwise
    :param save_json: binary value, True, if want to save result in a JSON file, False, otherwise
    :param split: split of the data, value is a list of numbers such as 0: train, 1: dev, test: 2. will return all data by default
    :param source: source of the data, a list of integer numbers
    :return: list of dictionaries
    """

    def get_token_indices(i_idx, t_idx, span_end_idx, all_tokens):
        span_tokens = []
        while t_idx < span_end_idx:
            span_tokens.append(all_tokens[i_idx])
            t_idx += len(all_tokens[i_idx])
            i_idx += 1
        return span_tokens

    nlp = spacy.load("en_core_web_sm")

    if not type(df) == pd.core.frame.DataFrame:
        print("first parameter should be a pandas data frame")
        raise TypeError

    records = list()
    excluded_rows = list()
    excluded_records = list()
    records_df = list()
    for index, row in df.iterrows():
        try:
            x = str(row['idx']).split('\n')[0].split(' ')
            y = str(row['idx']).split('\n')[1].split(' ')
            if x[0] == 'span1':
                span1 = x[1]
                span2 = y[1]
            else:
                span2 = x[1]
                span1 = y[1]

            record = dict()
            span1_start = int(span1.split(':')[0])
            span1_end = int(span1.split(':')[1])
            span2_start = int(span2.split(':')[0])
            span2_end = int(span2.split(':')[1])

            if no_order:
                if span2_start < span1_start:
                    span1_start, span2_start = span2_start, span1_start
                    span1_end, span2_end = span2_end, span1_end

            label = int(row['label'])
            direction = int(row['direction'])

            # creating list of tokens in context and finding spans' start and end indices
            doc = nlp(row['context'])
            doc_tokens = []
            sent_start = 0
            token_idx = 0
            doc_token_idx = 0
            flag = False
            for j, sent in enumerate(doc.sents):
                sent_tokens = [token.text_with_ws for token in sent]
                sent_len = sum([len(token) for token in sent_tokens])
                sent_end = sent_start + sent_len
                if not flag and ((sent_start < span1_start < sent_end) or (sent_start < span2_start < sent_end)):
                    doc_tokens.extend(sent_tokens)
                    len_tokens = len(sent_tokens)
                    flag = True
                for i, sent_token in enumerate(sent_tokens):
                    if token_idx == span1_start:
                        record['span1_start'] = copy.deepcopy(i + doc_token_idx)
                        span1_tokens = get_token_indices(i, token_idx, span1_end, sent_tokens)
                        record['span1_end'] = record['span1_start'] + len(span1_tokens) - 1
                    elif token_idx == span2_start:
                        record['span2_start'] = copy.deepcopy(i + doc_token_idx)
                        span2_tokens = get_token_indices(i, token_idx, span2_end, sent_tokens)
                        record['span2_end'] = record['span2_start'] + len(span2_tokens) - 1
                    token_idx += len(sent_token)
                if flag:
                    doc_token_idx += len_tokens
                    flag = False
                sent_start = copy.deepcopy(sent_end)

            # getting the label and span type
            if direction == 0 or direction == -1:
                record['direction'] = 'RIGHT'
                record['span1_type'] = 'SPAN1'
                record['span2_type'] = 'SPAN2'
            elif direction == 1:
                record['direction'] = 'LEFT'
                record['span1_type'] = 'SPAN2'
                record['span2_type'] = 'SPAN1'

            record['id'] = str(row['global_id'])
            record['token'] = doc_tokens
            record['relation'] = label
            features = ['id', 'token', 'span1_start', 'span1_end', 'span2_start', 'span2_end', 'relation']

            # check if record has all the required fields
            if all(feature in record for feature in features) and (
                    len(split) == 0 or int(row['split']) in split) and (
                    len(source) == 0 or int(row['source']) in source) and record['span1_end'] <= record[
                'span2_start'] and record['span2_end'] < len(doc_tokens) and ''.join(
                doc_tokens[record['span1_start']:record['span1_end'] + 1]) == ''.join(span1_tokens) and ''.join(
                doc_tokens[record['span2_start']:record['span2_end'] + 1]) == ''.join(span2_tokens):
                records.append(record)
                records_df.append(row)
            else:
                excluded_records.append([record, row])
        except Exception as e:
            print("error in converting the record. global id: {}. detail: {}".format(row['global_id'], str(e)))
            pass

    # saving records into a JSON file
    if save_json and len(records) > 0:
        with open(str(output_file_name) + '.json', 'w') as fout:
            json.dump(records, fout)

    return records, records_df, excluded_records, excluded_rows
