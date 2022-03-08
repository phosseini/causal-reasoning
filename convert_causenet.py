import csv
import json

data_path = 'data/causenet/causenet-precision.jsonl'

with open(data_path, 'r') as file:
    lines = file.readlines()

types = ['wikipedia_sentence', 'clueweb12_sentence']

with open('data/causenet/causenet.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["text", "cause", "effect", "type"])
    for line in lines:
        line = json.loads(line)
        for source in line['sources']:
            if source['type'] in types:
                sentence = source['payload']['sentence']
                csv_writer.writerow(
                    [sentence,
                     line['causal_relation']['cause']['concept'],
                     line['causal_relation']['effect']['concept'],
                     source['type']])
