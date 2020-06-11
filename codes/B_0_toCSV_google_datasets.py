# https://github.com/google-research-datasets/sentence-compression/issues/1#issuecomment-520114905
import copy
import json
import csv
from glob import glob
import spacy

file_path = "/Users/mehec/nlp/prj_m1/Google_dataset_news/"

SpaCy_EN = spacy.load("en_core_web_sm")

fieldnames = ['original', 'lemma', 'pos', 'tag', 'dep', 'head', 'head_text', 'depth', 'compressed']


def tokenizer(text):
    return [tok.text for tok in SpaCy_EN.tokenizer(text)]


def bottom_up_tree_depth(heads, head):
    head = heads[head]
    if head == -1:
        return 1
    else:
        return bottom_up_tree_depth(heads, head) + 1


def tokenizer_parser(text):
    doc = SpaCy_EN(text)
    processed = {"text": [],
                 "lemma": [],
                 "pos": [],
                 "tag": [],
                 "dep": [],
                 "head": [],
                 "head_text": [],
                 "depth": []
                 }
    for token in doc:
        processed["text"].append(token.text)
        processed["lemma"].append(token.lemma_)
        processed["pos"].append(token.pos_)
        processed["tag"].append(token.tag_)
        processed["dep"].append(token.dep_)
        processed["head"].append(token.head.i if token.dep_ != "ROOT" else -1)
    heads = copy.deepcopy(processed["head"])
    for i in range(len(heads)):
        head = heads[i]
        processed["head_text"].append(processed["text"][head])
        processed["depth"].append(str(bottom_up_tree_depth(heads, head)))
        processed["head"][i] = str(head)
    return processed


def to_csv_record(writer, buffer):
    record = json.loads(buffer)
    original_processed = tokenizer_parser(record['graph']['sentence'])
    original = original_processed['text']
    lemma = original_processed['lemma']
    pos = original_processed['pos']
    tag = original_processed['tag']
    dep = original_processed['dep']
    head = original_processed['head']
    head_text = original_processed['head_text']
    depth = original_processed['depth']
    compressed = tokenizer(record['compression']['text'])
    if set(original).issuperset(compressed):
        writer.writerow(dict(original=" ".join(original),
                             lemma=" ".join(lemma),
                             pos=" ".join(pos),
                             tag=" ".join(tag),
                             dep=" ".join(dep),
                             head=" ".join(head),
                             head_text=" ".join(head_text),
                             depth=" ".join(depth),
                             compressed=" ".join(compressed)
                             ))


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
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
