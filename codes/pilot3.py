# https://www.jianshu.com/p/dbf00b590c70
import json
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import optim
from torchtext.data import Field, BucketIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def give_label(pair):  # naive version
    sent, compr = pair
    j = 0
    labels = []
    for i in range(len(sent)):
        if sent[i] == compr[j]:
            labels.append(0)
            j += 1
        else:
            labels.append(1)
    return " ".join(sent), " ".join(map(str, labels))


def compress(sent, labels):
    return " ".join(sent[i] for i in range(len(sent)) if not labels[i])


with open("../Google_dataset_news/pilot_dataset.json", "r") as f:
    dataset = json.load(f)

training_data = [(sent.lower().split(), compr.lower().split())
                 for sent, compr in dataset
                 ]
training_data = list(zip(*map(give_label, training_data)))

word2ix = {}
for sent in training_data[0]:
    for word in sent:
        if word not in word2ix:
            word2ix[word] = len(word2ix)
word2ix['<sos>'] = len(word2ix)
word2ix['<eos>'] = len(word2ix)
label2ix = {0: 0, 1: 1, '<sos>': word2ix['<sos>'], '<eos>': word2ix['<eos>']}


def prepare_seq(seq, to_ix):
    idxs = ['<sos>', ]
    idxs.extend([to_ix[w] for w in seq])
    idxs.append('<eos>')
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size,
                              trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        return outputs


INPUT_DIM = len(word2ix)
OUTPUT_DIM = len(label2ix)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 3
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device)
