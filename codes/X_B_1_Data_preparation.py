# tokenize and eliminate unviable examples
# base structure: https://www.jianshu.com/p/dbf00b590c70

import spacy
import pickle
import dill
from torchtext.data import Field, TabularDataset

data_path = "../Google_dataset_news/"


# SpaCy_EN = spacy.load("en_core_web_sm")


def tokenizer(text):
    return text.split()
    # return [tok.text for tok in SpaCy_EN.tokenizer(text)]


def give_label(tabular_dataset):
    to_pop = []  # to store bad exemples indices
    for i in range(len(tabular_dataset.examples)):
        orig = tabular_dataset.examples[i].original
        compr = tabular_dataset.examples[i].compressed
        if not set(orig).issuperset(compr):
            to_pop.append(i)
            continue
        k = 0
        labels = []
        for j in range(len(orig)):
            if k >= len(compr):
                labels.append(0)
            elif orig[j] == compr[k]:
                labels.append(1)
                k += 1
            else:
                labels.append(0)
        tabular_dataset.examples[i].compressed = labels
        if len(tabular_dataset.examples[i].compressed) != len(tabular_dataset.examples[i].original):
            pass
    for i in to_pop[::-1]:
        tabular_dataset.examples.pop(i)


ORIG = Field(lower=True, tokenize=tokenizer, init_token="<eos>", eos_token="<eos>")
COMPR = Field(lower=True, tokenize=tokenizer, init_token="<eos>", eos_token="<eos>", unk_token=None)

train_val = TabularDataset(
    path=data_path + "B_0_training_data.csv",
    format="csv",
    fields=[("original", ORIG), ("compressed", COMPR)],
    skip_header=True,
)
give_label(train_val)
train, val = train_val.split(split_ratio=0.9)

test = TabularDataset(
    path=data_path + "B_0_eval_data.csv",
    format="csv",
    fields=[("original", ORIG), ("compressed", COMPR)],
    skip_header=True
)
give_label(test)

with open(data_path + "B_1_prepared_data.pickle", 'w') as f:
    # pickle.dump([train, val, test], f)
    # pickle.dump(val,f)
    dill.dump(val, f)
