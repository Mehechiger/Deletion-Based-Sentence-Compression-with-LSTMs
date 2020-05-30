# filippova 2015
# base structure: https://www.jianshu.com/p/dbf00b590c70
# gpu performance improvement: https://zhuanlan.zhihu.com/p/65002487
# beam search:
# - priority queue: github.com/budzianowski/PyTorch-Beam-Search-Decoding
# - TODO batch beam search: https://medium.com/the-artificial-impostor/implementing-beam-search-part-1-4f53482daabe
# - beam search prob normalizations:
# - - https://www.youtube.com/watch?v=gb__z7LlN_4
# - - https://opennmt.net/OpenNMT/translation/beam_search/
# - - https://arxiv.org/pdf/1609.08144.pdf

import os
import time
import math
from queue import PriorityQueue
import json
import torch
import torch.nn as nn
from torch import optim
from torchtext.data import Field, BucketIterator, TabularDataset

if not os.path.isdir("/content/"):
    VECTORS_CACHE = "/Users/mehec/Google Drive/Colab_tmp/vector_cache"
    PATH_LOG = "../outputs/"
    PATH_OUTPUT = "../outputs/"
else:
    VECTORS_CACHE = "/content/drive/My Drive/Colab_tmp/vector_cache"
    PATH_LOG = "/content/drive/My Drive/Colab_tmp/"
    PATH_OUTPUT = "/content/drive/My Drive/Colab_tmp/"


def logger(*content, verbose=False, path_log=PATH_LOG):
    log = "%soutput%s.log" % (path_log, AFFIX)
    if verbose:
        try:
            content = "".join(content)
        except TypeError:
            content = "".join(map(str, content))
        content += "\n"

        if verbose == 1:
            pass
        elif verbose == 2:
            with open(log, "a") as f:
                f.write(content)
        elif verbose == 3:
            print(content)
            with open(log, "a") as f:
                f.write(content)
        elif verbose == 4:
            with open(log, "w") as f:
                f.write("")


# Output Verbose modes:
# 1 = silent
# 2 = write only
# 3 = print and write
# 4, None, False, 0 = clear log file
VERBOSE = 3
TRAIN_VERBOSE = 1
VAL_VERBOSE = 2
TEST_VERBOSE = 3

# define AFFIX
AFFIX = ""

# clear output.log
logger(None, verbose=4)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger("using device: %s\n" % DEVICE, verbose=VERBOSE)


def splitter(text):
    return text.split(" ")


def give_label(tabular_dataset):
    for i in range(len(tabular_dataset.examples)):
        orig = tabular_dataset.examples[i].original
        compr = tabular_dataset.examples[i].compressed
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
            raise IndexError('Original and Compressed sentences do not match in length:\nOrig: %s\nCompr: %s' % (orig,
                                                                                                                 compr
                                                                                                                 ))


def compress_with_labels(sent, trg, labels, orig_itos, compr_itos, out=False):
    res = []
    for i in range(sent.shape[1]):
        orig = [orig_itos[sent[j, i]] for j in range(sent.shape[0])]
        labels_ = [compr_itos[labels.max(2)[1][j, i]] for j in range(labels.shape[0])]
        trg_ = [compr_itos[trg[j, i]] for j in range(trg.shape[0])]
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
        res.append((orig, compr, compr_trg))
        logger(
            "original:   ", " ".join(orig), "\n",
            "compressed: ", " ".join(compr), "\n",
            "gold:       ", " ".join(compr_trg), "\n\n",
            verbose=out,
        )
    return res


def res_outputter(res, file_name, show_spe_token=False, path_output=PATH_OUTPUT):
    file = path_output + file_name + ".json"
    to_dump = []
    for orig, compr, compr_trg in res:
        if show_spe_token:
            orig = " ".join(orig)
            compr = " ".join(compr)
            compr_trg = " ".join(compr_trg)
        else:
            orig = " ".join(s for s in orig if s != "<eos>" and s != "<pad>" and s != "<del>")
            compr = " ".join(s for s in compr if s != "<eos>" and s != "<pad>" and s != "<del>")
            compr_trg = " ".join(s for s in compr_trg if s != "<eos>" and s != "<pad>" and s != "<del>")
        to_dump.append(dict(orig=orig, hyp=compr, ref=compr_trg))
    with open(file, "w") as f:
        json.dump(to_dump, f)


ORIG = Field(lower=True, tokenize=splitter, init_token="<eos>", eos_token="<eos>")
COMPR = Field(lower=True, tokenize=splitter, init_token="<eos>", eos_token="<eos>", unk_token=None)

path_data = "../Google_dataset_news/"

train_val = TabularDataset(
    path=path_data + "B_0_training_data.csv",
    format="csv",
    fields=[("original", ORIG), ("compressed", COMPR)],
    skip_header=True,
)
give_label(train_val)
train, val = train_val.split(split_ratio=0.9)

test = TabularDataset(
    path=path_data + "B_0_eval_data.csv",
    format="csv",
    fields=[("original", ORIG), ("compressed", COMPR)],
    skip_header=True
)
give_label(test)

ORIG.build_vocab(train, min_freq=1, vectors="glove.840B.300d", vectors_cache=VECTORS_CACHE)
COMPR.build_vocab(train, min_freq=1)

"""
"""
# for testing use only small amount of data
train, _ = train.split(split_ratio=0.01)
# val, _ = val.split(split_ratio=0.005)
_, val = train.split(split_ratio=0.9995)
test, _ = test.split(split_ratio=0.005)
# test, _ = train.split(split_ratio=0.1)
# val = test = train
"""
"""

logger("train: %s examples" % len(train.examples), verbose=VERBOSE)
logger("val: %s examples" % len(val.examples), verbose=VERBOSE)
logger("test: %s examples" % len(test.examples), verbose=VERBOSE)

# split log files by epoch if train too big
if len(train.examples) + len(val.examples) + len(test.examples) >= 2000:
    AFFIX = "_epoch_1"
    logger(None, verbose=4)

# real batch size = BATCH_SIZE * ACCUMULATION_STEPS
# -> gradient descend every accumulation_steps batches
BATCH_SIZE = 32
ACCUMULATION_STEPS = 1

# for batch beam search
"""
train_iterator, val_iterator, test_iterator = BucketIterator.splits((train, val, test),
                                                                    batch_size=BATCH_SIZE,
                                                                    sort=False,
                                                                    device=DEVICE
                                                                    )
"""
(train_iterator,) = BucketIterator.splits((train,), batch_size=BATCH_SIZE, sort=False, device=DEVICE)

# batch size = 1 for val/test
val_iterator, test_iterator = BucketIterator.splits((val, test), batch_size=1, sort=False, device=DEVICE)


class PriorityEntry(object):  # prevent queue from comparing data
    """
    source: https://stackoverflow.com/a/40205431
    """

    def __init__(self, priority, data):
        self.data = data
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


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
        # self.embedding_input = nn.Embedding(output_dim, emb_input_dim)
        self.embedding_input = lambda l: torch.eye(
            emb_input_dim)[l.view(-1)].unsqueeze(0).to(self.device)
        self.rnn = nn.LSTM(emb_src_dim + emb_input_dim,
                           hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src, input_, hidden, cell):
        input_ = input_.unsqueeze(0)
        src = src.unsqueeze(0)
        embedded = self.embedding_src(src)
        embedded = torch.cat((embedded, self.embedding_input(input_)), axis=2)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.softmax(self.out(output.squeeze(0)))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def beam_predict(self, src, input_, hidden, cell, beam_width, lp_alpha=1):
        def normalize(prob, l, alpha=lp_alpha):
            lp = (5 + l) ** alpha / 6 ** alpha
            cp = 0  # coverage penalty - for attention mechanism
            return prob / lp + cp

        max_len = src.shape[0]
        src_ = src[0, :]
        output, hidden, cell = self.decoder(src_, input_, hidden, cell)
        prob = 0.0
        beam = PriorityQueue()
        beam.put(PriorityEntry(-normalize(prob, 1), (hidden, cell, input_, prob, output)))
        for t in range(1, max_len):
            src_ = src[t, :]
            next_beam = PriorityQueue()
            while beam.qsize() > 0:
                hidden, cell, input_, prob, labels = beam.get().data
                output, hidden, cell = self.decoder(src_, input_, hidden, cell)
                output_tops = torch.topk(output, output.shape[1], lp_alpha)  # to get indices
                for i in range(output_tops[1].shape[1]):
                    #prob_i = prob + float(output_tops[0][:, i])
                    prob_i = float(output_tops[0][:, i])
                    # prob_i = (prob * t + float(output_tops[0][:, i])) / (t + 1)
                    next_beam.put(PriorityEntry(-normalize(prob_i, t + 1),
                                                (hidden,
                                                 cell,
                                                 output_tops[1][:, i],
                                                 prob_i,
                                                 torch.cat((labels, output), dim=1)
                                                 )
                                                ))
            for i in range(min(beam_width, next_beam.qsize())):
                beam.put(next_beam.get())
        labels = beam.get().data[4] if beam.qsize() > 0 else None
        return labels.view(max_len, 1, -1)

    # for batch beam search
    """
    def batch_beam_predict(self, beam_width, src, input_, hidden, cell):
        hidden = hidden.repeat(1, beam_width, 1)
        cell = cell.repeat(1, beam_width, 1)
        input_ = input_.view(1, -1)
        pass
    """

    def forward(self, src, trg, beam_width):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        hidden, cell = self.encoder(torch.flip(src[1:, :], [0, ]))
        input_ = trg[0, :]
        src_ = src[0, :]
        if not beam_width:  # teacher forcing mode
            outputs = torch.zeros(max_len, batch_size,
                                  trg_vocab_size).to(self.device)
            for t in range(max_len):
                output, hidden, cell = self.decoder(src_, input_, hidden, cell)
                outputs[t] = output
                if t + 1 < max_len:
                    input_ = trg[t + 1]
                    src_ = src[t + 1]
        else:  # beam predicting mode
            outputs = self.beam_predict(src, input_, hidden, cell, beam_width, LP_ALPHA)
            # for batch beam search
            # outputs = self.batch_beam_predict(beam_width, src, input_, hidden, cell)
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
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(INPUT_DIM, OUTPUT_DIM, DEC_EMB_SRC_DIM,
              DEC_EMB_INPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, DEVICE)
model = Seq2Seq(enc, dec, DEVICE)
model.to(DEVICE)


"""
LR = 2
optimizer = optim.SGD(model.parameters(), lr=LR)
STEP_SIZE = 300000 / (BATCH_SIZE * ACCUMULATION_STEPS)
GAMMA = 0.96
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=STEP_SIZE,
                                      gamma=GAMMA
                                      )
"""
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()


def train(model,
          iterator,
          optimizer,
          criterion,
          accumulation_steps,
          verbose=False,
          val_in_epoch=None,
          in_epoch_steps=None
          ):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.original
        trg = batch.compressed

        try:
            output = model(src, trg, None)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                raise exception

        if verbose:
            compress_with_labels(src, trg, output, ORIG.vocab.itos, COMPR.vocab.itos, out=verbose)

        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)

        loss = criterion(output, trg)
        logger("batch %s, loss: " % i, loss.item(), verbose=3)

        loss.backward()

        if ((i + 1) % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
            #scheduler.step()

        if val_in_epoch and ((i + 1) % in_epoch_steps) == 0:
            val_loss, val_res = evaluate(model, val_in_epoch, criterion, beam_width=BEAM_WIDTH, verbose=VAL_VERBOSE)
            logger(f"\tVal Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}", verbose=VERBOSE)
            model.train()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, beam_width=3, verbose=False):
    model.eval()
    epoch_loss = 0
    res = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.original
            trg = batch.compressed

            try:
                output = model(src, trg, beam_width)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            if verbose:
                res += compress_with_labels(src, trg, output, ORIG.vocab.itos, COMPR.vocab.itos, out=verbose)
            else:
                res += compress_with_labels(src, trg, output, ORIG.vocab.itos, COMPR.vocab.itos, out=1)

            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), res


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 20
BEAM_WIDTH = 10
LP_ALPHA = 1

best_valid_loss = float("inf")

# TODO add checkpoint to google drive (colab) or local (local) at each epoch end
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model,
                       train_iterator,
                       optimizer,
                       criterion,
                       accumulation_steps=ACCUMULATION_STEPS,
                       verbose=TRAIN_VERBOSE,
                       val_in_epoch=val_iterator,
                       in_epoch_steps=512 // BATCH_SIZE
                       )

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    logger(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s", verbose=VERBOSE)
    logger(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}", verbose=VERBOSE)

    val_loss, val_res = evaluate(model, val_iterator, criterion, beam_width=BEAM_WIDTH, verbose=VAL_VERBOSE)
    res_outputter(val_res, "val_res_epoch%s" % (epoch + 1))

    logger(f"\tVal Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}", verbose=VERBOSE)

    # update AFFIX if necessary
    if AFFIX:
        AFFIX = "_epoch_%s" % (epoch + 2) if epoch < N_EPOCHS - 1 else "_test"
        logger(None, verbose=4)

    """
    if val_loss <= 0.01:
        break
    """

test_loss, test_res = evaluate(model, test_iterator, criterion, beam_width=BEAM_WIDTH, verbose=TEST_VERBOSE)
res_outputter(test_res, "test_res")

logger(f"\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}", verbose=VERBOSE)
