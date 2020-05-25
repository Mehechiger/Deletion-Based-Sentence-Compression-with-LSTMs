# https://www.pytorchtutorial.com/pytorch-sequence-model-and-lstm-networks/
import json
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def deletion_tagger(pair):  # naive version
    sent, compression = pair
    j = 0
    tags = []
    for i in range(len(sent)):
        if sent[i] == compression[j]:
            tags.append(0)
            j += 1
        else:
            tags.append(1)
    return sent, tags


def compressor(sent, tags):
    return " ".join(sent[i] for i in range(len(sent)) if not tags[i])


torch.manual_seed(1)

with open("../Google_dataset_news/A_pilot_dataset.json", "r") as f:
    dataset = json.load(f)

# tokenization
training_data_orig = [(sent.lower().split(), compression.lower().split())
                      for sent, compression in dataset
                      ]
# tagging
training_data = list(map(deletion_tagger, training_data_orig))

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {0: 0, 1: 1}


EMBEDDING_DIM = 128
HIDDEN_DIM = 128

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(training_data[0][0])
print(inputs)
print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)
tag_scores = tag_scores.detach()
predicted_tags = np.argmax(tag_scores, axis=1)
print(predicted_tags)
predicted_compression = compressor(training_data[0][0], predicted_tags)
print(predicted_compression, "\n", " ".join(training_data_orig[0][1]))
