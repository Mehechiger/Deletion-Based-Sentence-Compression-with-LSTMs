# filippova 2015
# base structure: https://www.jianshu.com/p/dbf00b590c70
# gpu performance improvement: https://zhuanlan.zhihu.com/p/65002487

import os
import time
import random
import math
import torch
import torch.nn as nn
from torch import optim
from torchtext.data import Field, BucketIterator, TabularDataset
import spacy


def outputter(*content, verbose=False):
    if verbose:
        try:
            content = "".join(content)
        except TypeError:
            content = "".join(map(str, content))
        content += "\n"

        if verbose == 1:
            print(content)
        elif verbose == 2:
            with open("output.log", "a") as f:
                f.write(content)
        elif verbose == 3:
            print(content)
            with open("output.log", "a") as f:
                f.write(content)
        elif verbose == 4 and content == None:
            with open("output.log", "w") as f:
                f.write()


VERBOSE = 3
TRAIN_VERBOSE = 2
TEST_VERBOSE = 3

# clear output.log
outputter(None, verbose=4)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
outputter("using device: %s\n" % DEVICE, verbose=VERBOSE)
#SpaCy_EN = spacy.load("en_core_web_sm")


def tokenizer(text):
    # return [tok.text for tok in SpaCy_EN.tokenizer(text)]
    return text.split()


def give_label(tabular_dataset):  # naive version
    for i in range(len(tabular_dataset.examples)):
        orig = tabular_dataset.examples[i].original
        compr = tabular_dataset.examples[i].compressed
        k = 0
        labels = []
        for j in range(len(orig)):
            if k >= len(compr):
                break
            elif orig[j] == compr[k]:
                labels.append(1)
                k += 1
            else:
                labels.append(0)
        tabular_dataset.examples[i].compressed = labels


def compress_with_labels(sent, trg, labels, orig_itos, compr_itos, out=False):
    # temporary fix of the list index out of range
    # pb caused by tokenization
    res = []
    for i in range(sent.shape[1]):
        try:
            orig = [orig_itos[sent[j, i]]
                    for j in range(sent.shape[0])
                    ]
            labels_ = [compr_itos[labels.max(2)[1][j, i]]
                       for j in range(labels.shape[0])
                       ]
            trg_ = [compr_itos[trg[j, i]]
                    for j in range(trg.shape[0])
                    ]
            compr = []
            compr_trg = []
            for j in range(len(orig)):
                if labels_[j] == 1:
                    compr.append(orig[j])
                elif labels_[j] == 0:
                    compr.append("<del>")
                else:
                    compr.append(labels_[j])
                if trg_[j] == 1:
                    compr_trg.append(orig[j])
                elif trg_[j] == 0:
                    compr_trg.append("<del>")
                else:
                    compr_trg.append(trg_[j])
        except IndexError:
            orig = compr = compr_trg = ["INDEX_ERROR", ]
        res.append((orig, compr, compr_trg))
        outputter("original:   ", " ".join(orig), "\n",
                  "compressed: ", " ".join(compr), "\n",
                  "gold:       ", " ".join(compr_trg), "\n\n",
                  verbose=out
                  )
    return res


ORIG = Field(
    lower=True,
    tokenize=tokenizer,
    init_token='<eos>',
    eos_token='<eos>'
)
COMPR = Field(
    lower=True,
    tokenize=tokenizer,
    init_token='<eos>',
    eos_token='<eos>',
    unk_token=None
)

path_data = "../Google_dataset_news/"

train_val = TabularDataset(
    path=path_data+"training_data.csv",
    format="csv",
    fields=[("original", ORIG), ("compressed", COMPR)],
    skip_header=True
)
give_label(train_val)
train, val = train_val.split(split_ratio=0.9)

test = TabularDataset(
    path=path_data+"eval_data.csv",
    format="csv",
    fields=[("original", ORIG), ("compressed", COMPR)],
    skip_header=True
)
give_label(test)

"""
"""
# for testing use only small amount of data
train, _ = train.split(split_ratio=0.01)
val, _ = val.split(split_ratio=0.01)
test, _ = test.split(split_ratio=0.01)
#test, _ = train.split(split_ratio=0.1)
#val = test = train

outputter("train: %s examples" % len(train.examples), verbose=VERBOSE)
outputter("val: %s examples" % len(val.examples), verbose=VERBOSE)
outputter("test: %s examples" % len(test.examples), verbose=VERBOSE)
"""
"""

vectors_cache = "/Users/mehec/Google Drive/vector_cache" \
    if not os.path.isdir("/content/")\
    else "/content/drive/My Drive/vector_cache"
ORIG.build_vocab(train,
                 min_freq=1,
                 vectors="glove.840B.300d",
                 vectors_cache=vectors_cache
                 )
COMPR.build_vocab(train, min_freq=1)

# real batch size = BATCH_SIZE * ACCUMULATION_STEPS
# -> gradient descend every accumulation_steps batches
BATCH_SIZE = 32
ACCUMULATION_STEPS = 8

train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train, val, test),
    batch_size=BATCH_SIZE,
    sort=False,
    device=DEVICE
)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, emb_src_dim, emb_input_dim, hid_dim, n_layers, dropout, device):
        super().__init__()

        self.emb_src_dim = emb_src_dim
        self.emb_input_dim = emb_input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device

        self.embedding_src = nn.Embedding(input_dim, emb_src_dim)
        #self.embedding_input = nn.Embedding(output_dim, emb_input_dim)
        self.embedding_input = lambda l: torch.eye(
            emb_input_dim)[l.view(-1)].unsqueeze(0).to(self.device)
        self.rnn = nn.LSTM(emb_src_dim+emb_input_dim,
                           hid_dim,
                           n_layers,
                           dropout=dropout
                           )
        self.out = nn.Linear(hid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src, input, hidden, cell):
        print(input.shape)
        exit()
        input = input.unsqueeze(0)
        src = src.unsqueeze(0)
        embedded = self.embedding_src(src)
        embedded = torch.cat((embedded, self.embedding_input(input)), axis=2)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.softmax(self.out(output.squeeze(0)))
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

    def forward(self, src, trg, teacher_forcing_ratio, n):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len,
                              batch_size,
                              trg_vocab_size
                              ).to(self.device)
        hidden, cell = self.encoder(torch.flip(src[1:, :], [0, ]))
        input = trg[0, :]
        src_ = src[0, :]
        output, hidden, cell = self.decoder(src_, input, hidden, cell)
        prob = torch.zeros(batch_size, 1).to(self.device)
        beam = [(hidden, cell, input, prob, [output, ]), ]
        for t in range(1, max_len):
            teacher_force = random.random() < teacher_forcing_ratio
            src_ = src[t]
            next_beam = []
            for hidden, cell, input, prob, labels in beam:
                output, hidden, cell = self.decoder(src_, input, hidden, cell)
                if teacher_force:
                    input = trg[t]
                    # if LogSoftmax then prob+0 else *1
                    next_beam.append((hidden,
                                      cell,
                                      input,
                                      prob,
                                      labels+[output, ]
                                      ))
                else:
                    output_tops = torch.topk(output, n, 1)
                    for i in range(output_tops[1].shape[1]):
                        # if LogSoftmax then prob+ else *
                        next_beam.append((hidden,
                                          cell,
                                          output_tops[1][:, i],
                                          prob+output_tops[0][:, i],
                                          labels+[output, ]
                                          ))
            beam = sorted(next_beam, key=lambda x: x[3].mean())[:n]
        labels = sorted(beam, key=lambda x: x[3].mean())[0][4]
        for i in range(len(labels)):
            outputs[i] = labels[i]
        return outputs


INPUT_DIM = len(ORIG.vocab)
OUTPUT_DIM = len(COMPR.vocab)
ENC_EMB_DIM = 256
DEC_EMB_SRC_DIM = 256
DEC_EMB_INPUT_DIM = OUTPUT_DIM
HID_DIM = ENC_EMB_DIM
N_LAYERS = 3
ENC_DROPOUT = 0
DEC_DROPOUT = 0.2
enc = Encoder(INPUT_DIM,
              ENC_EMB_DIM,
              HID_DIM,
              N_LAYERS,
              ENC_DROPOUT
              )
dec = Decoder(INPUT_DIM,
              OUTPUT_DIM,
              DEC_EMB_SRC_DIM,
              DEC_EMB_INPUT_DIM,
              HID_DIM,
              N_LAYERS,
              DEC_DROPOUT,
              DEVICE
              )
model = Seq2Seq(enc, dec, DEVICE)
model.to(DEVICE)


"""
LR = 2
optimizer = optim.SGD(model.parameters(), lr=LR)
STEP_SIZE = 300000/(BATCH_SIZE*ACCUMULATION_STEPS)
GAMMA = 0.96
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=STEP_SIZE,
                                      gamma=GAMMA
                                      )
"""
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()


def train(model, iterator, optimizer, criterion, verbose=False, accumulation_steps=1):

    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.original
        trg = batch.compressed

        try:
            output = model(src, trg, 1, 1)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        if verbose:
            compress_with_labels(
                src,
                trg,
                output,
                ORIG.vocab.itos,
                COMPR.vocab.itos,
                out=verbose
            )

        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)

        loss = criterion(output, trg)
        outputter("batch %s, loss: " % i, loss.item(),
                  verbose=3 if verbose == 2 else verbose
                  )

        loss.backward()

        if ((i+1) % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, beam=3, verbose=False):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.original
            trg = batch.compressed

            try:
                output = model(src, trg, 0, beam)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            if verbose:
                compress_with_labels(
                    src,
                    trg,
                    output,
                    ORIG.vocab.itos,
                    COMPR.vocab.itos,
                    out=verbose
                )

            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 20
BEAM = 3

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model,
                       train_iterator,
                       optimizer,
                       criterion,
                       verbose=TRAIN_VERBOSE,
                       accumulation_steps=ACCUMULATION_STEPS
                       )

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    outputter(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s',
              verbose=VERBOSE
              )
    outputter(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}',
              verbose=VERBOSE
              )

    val_loss = evaluate(model,
                        val_iterator,
                        criterion,
                        beam=BEAM,
                        verbose=TRAIN_VERBOSE
                        )

    outputter(f'\tVal Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}',
              verbose=VERBOSE
              )

    """
    if val_loss <= 0.01:
        break
    """


test_loss = evaluate(model,
                     test_iterator,
                     criterion,
                     beam=BEAM,
                     verbose=TEST_VERBOSE
                     )
outputter(f'\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}',
          verbose=VERBOSE
          )
