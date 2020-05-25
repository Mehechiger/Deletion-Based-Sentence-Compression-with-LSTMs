# https://github.com/google-research-datasets/sentence-compression/issues/1#issuecomment-520114905

import json
import csv
from glob import glob
import spacy

file_path = "/Users/mehec/nlp/prj_m1/Google_dataset_news/"

SpaCy_EN = spacy.load("en_core_web_sm")

fieldnames = ['original', 'compressed']


def tokenizer(text):
    return [tok.text for tok in SpaCy_EN.tokenizer(text)]


def to_csv_record(writer, buffer):
    record = json.loads(buffer)
    original = tokenizer(record['graph']['sentence'])
    compressed = tokenizer(record['compression']['text'])
    if set(original).issuperset(compressed):
        writer.writerow(dict(original=" ".join(original), compressed=" ".join(compressed)))


with open(file_path + 'B_0_training_data.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for json_file in glob(file_path + '**train**.json'):
        with open(json_file) as raw_contents:
            buffer = ''
            for line in raw_contents:
                if line.strip() == '':
                    to_csv_record(writer, buffer)
                    buffer = ''
                else:
                    buffer += line
            if len(buffer) > 0:
                to_csv_record(writer, buffer)

with open(file_path + 'B_0_eval_data.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['original', 'compressed'])
    writer.writeheader()

    with open(file_path + 'comp-data.eval.json') as raw_contents:
        buffer = ''
        for line in raw_contents:
            if line.strip() == '':
                to_csv_record(writer, buffer)
                buffer = ''
            else:
                buffer += line
        if len(buffer) > 0:
            to_csv_record(writer, buffer)
