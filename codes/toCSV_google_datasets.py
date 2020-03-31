# https://github.com/google-research-datasets/sentence-compression/issues/1#issuecomment-520114905

import json
import csv
from glob import glob

file_path = "/Users/mehec/nlp/prj_m1/Google_dataset_news/"

fieldnames = ['original', 'compressed']


def to_csv_record(writer, buffer):
    record = json.loads(buffer)
    writer.writerow(dict(
        original=record['graph']['sentence'],
        compressed=record['compression']['text']))


with open(file_path+'training_data.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for json_file in glob(file_path+'**train**.json'):
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

with open(file_path+'eval_data.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['original', 'compressed'])
    writer.writeheader()

    with open(file_path+'comp-data.eval.json') as raw_contents:
        buffer = ''
        for line in raw_contents:
            if line.strip() == '':
                to_csv_record(writer, buffer)
                buffer = ''
            else:
                buffer += line
        if len(buffer) > 0:
            to_csv_record(writer, buffer)
