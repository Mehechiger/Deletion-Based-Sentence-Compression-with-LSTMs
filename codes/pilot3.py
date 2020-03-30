# Filippova 2015
import json


def deletion_tagger(pair):  # naive version
    sent, compr = pair
    j = 0
    labels = []
    for i in range(len(sent)):
        if sent[i] == compr[j]:
            labels.append(0)
            j += 1
        else:
            labels.append(1)
    return sent, labels


def compressor(sent, labels):
    return " ".join(sent[i] for i in range(len(sent)) if not labels[i])


with open("../Google_dataset_news/pilot_dataset.json", "r") as f:
    dataset = json.load(f)

# tokenization
training_data_orig = [(sent.lower().split(), compr.lower().split())
                      for sent, compr in dataset
                      ]
# tagging
training_data = list(map(deletion_tagger, training_data_orig))
