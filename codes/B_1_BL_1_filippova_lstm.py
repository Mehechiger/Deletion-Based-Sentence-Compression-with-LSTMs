# filippova 2015
# base structure: https://www.jianshu.com/p/dbf00b590c70
# gpu performance improvement: https://zhuanlan.zhihu.com/p/65002487
# beam search:
# - priority queue: github.com/budzianowski/PyTorch-Beam-Search-Decoding
# - batch beam search: https://medium.com/the-artificial-impostor/implementing-beam-search-part-1-4f53482daabe
# - beam search prob normalizations:
# - - https://www.youtube.com/watch?v=gb__z7LlN_4
# - - https://opennmt.net/OpenNMT/translation/beam_search/
# - - https://arxiv.org/pdf/1609.08144.pdf

import os
import time
import math
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset

if not os.path.isdir("/content/"):
    VECTORS_CACHE = "/Users/mehec/Google Drive/Colab_tmp/vector_cache"
    PATH_DATA = "/Users/mehec/Google Drive/Colab_tmp/data/"
    PATH_LOG = "../outputs/"
    PATH_OUTPUT = "../outputs/"
else:
    VECTORS_CACHE = "/content/drive/My Drive/Colab_tmp/vector_cache"
    PATH_DATA = "/content/drive/My Drive/Colab_tmp/data/"
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

checkpoints = sorted([file_ for file_ in os.listdir(PATH_OUTPUT) if file_.split('.')[0][:17] == 'checkpoint_epoch_'],
                     key=lambda x: int(x[:-3].split('_')[-1])
                     )

if checkpoints:
    logger('\n\nresume from checkpoint: %s\n%s\n' % (checkpoints[-1], datetime.now()), verbose=3)
else:
    # clear output.log
    logger(None, verbose=4)

CUDA = torch.cuda.is_available()
if CUDA:
    DEVICE = torch.device("cuda")
    APEX_OPT_LV = 'O1'
    from apex import amp
else:
    DEVICE = torch.device("cpu")

logger("using device: %s\n" % DEVICE, verbose=VERBOSE)


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


SPE_IDX = 2000

ORIG = Field(lower=True, init_token="<eos>", eos_token="<eos>")
LEMMA = Field(lower=True, init_token="<eos>", eos_token="<eos>", unk_token=None)
POS = Field(lower=True, init_token="<eos>", eos_token="<eos>", unk_token=None)
TAG = Field(lower=True, init_token="<eos>", eos_token="<eos>", unk_token=None)
DEP = Field(lower=True, init_token="<eos>", eos_token="<eos>", unk_token=None)
HEAD = Field(use_vocab=False, init_token=SPE_IDX, eos_token=SPE_IDX, pad_token=SPE_IDX, unk_token=None)
HEAD_TEXT = Field(lower=True, init_token="<eos>", eos_token="<eos>", unk_token=None)
DEPTH = Field(use_vocab=False, init_token=-1, eos_token=-1, pad_token=-1, unk_token=None)
COMPR = Field(lower=True, init_token="<eos>", eos_token="<eos>", unk_token=None)

FIELDS = {"original": ("original", ORIG),
          # "lemma":("lemma", LEMMA),
          # "pos":("pos", POS),
          # "tag":("tag", TAG),
          "dep": ("dep", DEP),
          "head": ("head", HEAD),
          # "head_text": ("head_text", HEAD_TEXT),
          "depth": ("depth", DEPTH),
          "compressed": ("compressed", COMPR)
          }

train, val, test = TabularDataset.splits(
    path=PATH_DATA,
    train="B_0_train_data.ttjson",
    validation="B_0_val_data.ttjson",
    test="B_0_test_data.ttjson",
    format="json",
    fields=FIELDS
)
give_label(train)
give_label(val)
give_label(test)

ORIG.build_vocab(train, min_freq=1, vectors="glove.840B.300d", vectors_cache=VECTORS_CACHE)
DEP.build_vocab(train, min_freq=1)
# TODO add <*ROOT*>
COMPR.build_vocab(train, min_freq=1)

"""
"""
"""
# for testing use only small amount of data
train, _ = train.split(split_ratio=0.0001)
val, _ = val.split(split_ratio=0.05)
# _, val = train.split(split_ratio=0.9995)
test, _ = test.split(split_ratio=0.05)
# test, _ = train.split(split_ratio=0.1)
# val = test = train
"""

logger("train: %s examples" % len(train.examples), verbose=VERBOSE)
logger("val: %s examples" % len(val.examples), verbose=VERBOSE)
logger("test: %s examples" % len(test.examples), verbose=VERBOSE)

# split log files by epoch if train too big -- when no checkpoints
if len(train.examples) + len(val.examples) + len(test.examples) >= 2000 and not checkpoints:
    AFFIX = "_epoch_1"
    logger(None, verbose=4)
if checkpoints:
    AFFIX = "_epoch_" + str(int(checkpoints[-1][:-3].split('_')[-1]) + 1)
    logger(None, verbose=4)

# real batch size = BATCH_SIZE * ACCUMULATION_STEPS
# -> gradient descend every accumulation_steps batches
BATCH_SIZE = 32
ACCUMULATION_STEPS = 1

# https://www.jianshu.com/p/e5adb235399e
train_iterator, val_iterator, test_iterator = BucketIterator.splits((train, val, test),
                                                                    batch_size=BATCH_SIZE,
                                                                    sort_key=lambda x: len(x.original),
                                                                    sort_within_batch=False,
                                                                    device=DEVICE
                                                                    )


class PositionalEncoding(nn.Module):
    """
    source: attention is all you need & https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, spe_idx=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        if spe_idx:
            max_len += 1

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        if spe_idx:
            pe[max_len - 1] = 0.

    def forward(self, x):
        x = self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):

    def __init__(self, pretrained_vectors, embedding_head, dep_dim, dep_emb_dim, depth_emb_dim, n_layers, dropout,
                 device):
        super().__init__()
        self.dep_dim = dep_dim
        self.src_emb_dim = pretrained_vectors.shape[1]
        self.dep_emb_dim = dep_emb_dim
        self.depth_emb_dim = depth_emb_dim
        self.emb_dim = self.src_emb_dim + self.dep_emb_dim + self.depth_emb_dim
        self.hid_dim = self.emb_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.embedding_text = nn.Embedding.from_pretrained(pretrained_vectors)
        self.embedding_dep = nn.Embedding(self.dep_dim, self.dep_emb_dim)
        self.embedding_head = embedding_head
        self.embedding_depth = lambda l: torch.eye(depth_emb_dim)[l.view(-1)].unsqueeze(0).to(self.device)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout)

    def forward(self, src):
        text_embedded = self.embedding_text(src[0])
        dep_embedded = self.embedding_dep(src[1])
        head_embedded = self.embedding_head(src[2])
        depth_embedded = torch.cat([self.embedding_depth(src[3][i].unsqueeze(0)) for i in range(src.shape[1])], dim=0)
        embedded = torch.cat((text_embedded,
                              dep_embedded + head_embedded,
                              depth_embedded
                              ),
                             dim=2
                             )
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, pretrained_vectors, embedding_head, dep_dim, dep_emb_dim, depth_emb_dim, n_layers,
                 dropout, device):
        super().__init__()

        self.dep_dim = dep_dim
        self.src_emb_dim = pretrained_vectors.shape[1]
        self.dep_emb_dim = dep_emb_dim
        self.depth_emb_dim = depth_emb_dim
        self.emb_dim = self.src_emb_dim + self.dep_emb_dim + self.depth_emb_dim
        self.hid_dim = self.emb_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.embedding_src = nn.Embedding.from_pretrained(pretrained_vectors)
        self.embedding_dep = nn.Embedding(self.dep_dim, self.dep_emb_dim)
        self.embedding_head = embedding_head
        self.embedding_depth = lambda l: torch.eye(depth_emb_dim)[l.view(-1)].unsqueeze(0).to(self.device)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout)
        self.out = nn.Linear(self.hid_dim, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src, hidden, cell):
        src = src.unsqueeze(1)
        text_embedded = self.embedding_src(src[0])
        dep_embedded = self.embedding_dep(src[1])
        head_embedded = self.embedding_head(src[2])
        depth_embedded = self.embedding_depth(src[3])
        embedded = torch.cat((text_embedded,
                              dep_embedded + head_embedded,
                              depth_embedded
                              ),
                             dim=2
                             )
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

    def batch_beam_predict(self, src, hidden, cell, beam_width, lp_alpha=1):
        def normalize(prob, l, alpha=lp_alpha):
            lp = (5 + l) ** alpha / 6 ** alpha
            cp = 0  # coverage penalty - for attention mechanism
            return prob / lp + cp

        max_len = src.shape[1]
        batch_size = src.shape[2]
        output_dim = self.decoder.output_dim
        src_ = src[:, 0, :].repeat(1, beam_width)
        hidden = hidden.repeat(1, beam_width, 1)
        cell = cell.repeat(1, beam_width, 1)
        output, hidden, cell = self.decoder(src_, hidden, cell)

        outputs = torch.zeros(max_len, batch_size * beam_width, output_dim).to(self.device)
        outputs[0, :, :] = output
        backtrack = torch.zeros(max_len, batch_size * beam_width, 1, dtype=torch.long).to(self.device)
        backtrack[0, :, :] = output.topk(k=1, dim=1).indices

        for t in range(1, max_len):
            src_ = src[:, t, :].repeat(1, beam_width)
            output, hidden, cell = self.decoder(src_, hidden, cell)

            probs = outputs[:t, :, :].gather(dim=2, index=backtrack[:t, :, :]).sum(dim=0).repeat(1, output_dim)
            probs += output
            probs = torch.cat(probs.chunk(beam_width, dim=0), dim=1)
            probs = normalize(probs, t, lp_alpha)
            top_indices = probs.topk(k=beam_width, dim=1).indices

            beams = top_indices // output_dim
            beams = beams.t().reshape(-1)
            beams = beams * batch_size + torch.LongTensor(range(batch_size)).repeat(beam_width).to(self.device)
            beams = beams.unsqueeze(0).unsqueeze(2)

            states_beams = beams.repeat(hidden.shape[0], 1, hidden.shape[2])
            hidden = hidden.gather(dim=1, index=states_beams)
            cell = cell.gather(dim=1, index=states_beams)

            outputs_beams = beams.repeat(t, 1, output_dim)
            outputs[:t, :, :] = outputs[:t, :, :].gather(dim=1, index=outputs_beams)
            outputs[t, :, :] = output.gather(dim=0, index=outputs_beams[0, :, :])
            if t + 1 < max_len:
                backtrack[:t, :, :] = backtrack[:t, :, :].gather(dim=1, index=outputs_beams[:, :, :1])

                top_indices = top_indices.fmod(output_dim).t().reshape(-1)
                backtrack[t, :, 0] = top_indices

        return outputs[:, :batch_size, :].contiguous()

    def forward(self, src, trg, beam_width, teacher_force):
        hidden, cell = self.encoder(torch.flip(src[:, 1:, :], [1, ]))
        if teacher_force:  # teacher forcing mode
            batch_size = trg.shape[1]
            max_len = trg.shape[0]
            output_dim = self.decoder.output_dim
            outputs = torch.zeros(max_len, batch_size, output_dim).to(self.device)
            for t in range(max_len):
                src_ = src[:, t, :]
                output, hidden, cell = self.decoder(src_, hidden, cell)
                outputs[t] = output
        else:
            outputs = self.batch_beam_predict(src, hidden, cell, beam_width, LP_ALPHA)
        return outputs


OUTPUT_DIM = len(COMPR.vocab)
DEP_DIM = len(DEP.vocab)
DEP_EMB_DIM = 50
DEPTH_EMB_DIM = 40
N_LAYERS = 3
ENC_DROPOUT = 0
DEC_DROPOUT = 0.2
EMBEDDING_HEAD = PositionalEncoding(DEP_EMB_DIM, dropout=0, max_len=SPE_IDX, spe_idx=True)
enc = Encoder(ORIG.vocab.vectors, EMBEDDING_HEAD, DEP_DIM, DEP_EMB_DIM, DEPTH_EMB_DIM, N_LAYERS, ENC_DROPOUT, DEVICE)
dec = Decoder(OUTPUT_DIM, ORIG.vocab.vectors, EMBEDDING_HEAD, DEP_DIM, DEP_EMB_DIM, DEPTH_EMB_DIM, N_LAYERS,
              DEC_DROPOUT, DEVICE)
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
if CUDA:
    model, optimizer = amp.initialize(model, optimizer, opt_level=APEX_OPT_LV)
criterion = nn.NLLLoss()


# TODO choose better loss func for seq2seq + beamsearch
# - NLLLoss can't reflect beam search's corrections over time step


def train(model,
          iterator,
          optimizer,
          criterion,
          clip,
          accumulation_steps,
          beam_width,
          verbose=False,
          val_in_epoch=None,
          in_epoch_steps=None
          ):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = torch.stack((batch.original, batch.dep, batch.head, batch.depth), dim=0)
        trg = batch.compressed

        try:
            output = model(src, trg, beam_width, True)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                raise exception

        if verbose:
            compress_with_labels(src[0, :, :], trg, output, ORIG.vocab.itos, COMPR.vocab.itos, out=verbose)

        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)

        loss = criterion(output, trg)
        logger("batch %s, loss: " % i, loss.item(), verbose=3)

        if CUDA:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if ((i + 1) % accumulation_steps) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        if val_in_epoch and ((i + 1) % in_epoch_steps) == 0:
            val_loss, val_res = evaluate(model, val_in_epoch, criterion, beam_width=BEAM_WIDTH, verbose=TRAIN_VERBOSE)
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
            src = torch.stack((batch.original, batch.dep, batch.head, batch.depth), dim=0)
            trg = batch.compressed

            try:
                output = model(src, trg, beam_width, False)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            if verbose:
                res += compress_with_labels(src[0, :, :], trg, output, ORIG.vocab.itos, COMPR.vocab.itos, out=verbose)
            else:
                res += compress_with_labels(src[0, :, :], trg, output, ORIG.vocab.itos, COMPR.vocab.itos, out=1)

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


N_EPOCHS = 50
CLIP = float("inf")  # TODO adjust clip value
BEAM_WIDTH = 10
LP_ALPHA = 1

if checkpoints:
    checkpoint = torch.load(PATH_OUTPUT + checkpoints[-1])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    useless_epochs = checkpoint['useless_epochs']
    val_losses = checkpoint['val_losses']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if CUDA:
        amp.load_state_dict(checkpoint['amp'])
    if useless_epochs > 5 or start_epoch >= N_EPOCHS - 1:
        if AFFIX:
            AFFIX = "_test"
            logger(None, verbose=4)
        logger('\nbest epoch at %s / %s with val loss at %s\n' % (best_epoch + 1, start_epoch, best_val_loss),
               verbose=TEST_VERBOSE)

        test_loss, test_res = evaluate(model, test_iterator, criterion, beam_width=BEAM_WIDTH, verbose=TEST_VERBOSE)
        res_outputter(test_res, "test_res")

        logger(f"\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}", verbose=VERBOSE)

        exit()
else:
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    useless_epochs = 0
    val_losses = []

# TODO add checkpoint to google drive (colab) or local (local) at each epoch end
# TODO iterator.init_epoch and shuffle data at each epoch start
for epoch in range(start_epoch, N_EPOCHS):
    start_time = time.time()

    train_loss = train(model,
                       train_iterator,
                       optimizer,
                       criterion,
                       CLIP,
                       accumulation_steps=ACCUMULATION_STEPS,
                       beam_width=BEAM_WIDTH,
                       verbose=TRAIN_VERBOSE,
                       # val_in_epoch=val_iterator,
                       # in_epoch_steps=512 // BATCH_SIZE
                       )

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    logger(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s", verbose=VERBOSE)
    logger(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}", verbose=VERBOSE)

    val_loss, val_res = evaluate(model, val_iterator, criterion, beam_width=BEAM_WIDTH, verbose=VAL_VERBOSE)
    val_losses.append(val_loss)
    res_outputter(val_res, "val_res_epoch%s" % (epoch + 1))

    logger(f"\tVal Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}", verbose=VERBOSE)

    if val_loss < best_val_loss:
        if best_val_loss - val_loss >= 0.001:
            useless_epochs = 0
        else:
            useless_epochs += 1
        best_val_loss = val_loss
        best_epoch = epoch
    else:
        useless_epochs += 1

    logger('\nbest epoch so far at %s / %s with val loss at %s\n' % (best_epoch + 1, epoch + 1, best_val_loss),
           verbose=TEST_VERBOSE)

    if CUDA:
        torch.save({
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'useless_epochs': useless_epochs,
            'val_losses': val_losses,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'amp': amp.state_dict()
        }, PATH_OUTPUT + 'checkpoint_epoch_' + str(epoch + 1) + '.pt')
    else:
        torch.save({
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'useless_epochs': useless_epochs,
            'val_losses': val_losses,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, PATH_OUTPUT + 'checkpoint_epoch_' + str(epoch + 1) + '.pt')

    if useless_epochs > 5 or epoch == N_EPOCHS - 1:
        if val_loss > best_val_loss:
            checkpoint = torch.load(PATH_OUTPUT + 'checkpoint_epoch_' + str(best_epoch + 1) + '.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        break

    # update AFFIX if necessary
    if AFFIX:
        AFFIX = "_epoch_%s" % (epoch + 2)
        logger(None, verbose=4)

if AFFIX:
    AFFIX = "_test"
    logger(None, verbose=4)

logger('\nbest epoch at %s / %s with val loss at %s\n' % (best_epoch + 1, epoch + 1, best_val_loss),
       verbose=TEST_VERBOSE)

test_loss, test_res = evaluate(model, test_iterator, criterion, beam_width=BEAM_WIDTH, verbose=TEST_VERBOSE)
res_outputter(test_res, "test_res")

logger(f"\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}", verbose=VERBOSE)
