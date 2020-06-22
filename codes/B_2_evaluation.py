from os import walk
from collections import defaultdict
import rouge
import json
import re
import pandas as pd

PATH_INPUT = "../outputs/to_be_eval/"
PATH_OUTPUT = "../outputs/"


def get_fd(dataset):
    fd = []
    for (dirpath, dirnames, filenames) in walk(dataset):
        fd.append((dirpath, filenames))
    return fd


def get_data(fd):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ele in fd:
        if ele[1] and ele[1] != ['.DS_Store']:
            for doc in ele[1]:
                if re.match('val_res', doc):
                    data[ele[0].split("/")[-1].split("]")[-1]]["val"][int(doc.split('.')[0][13:])] = ele[0] + '/' + doc
                if re.match('test_', doc):
                    data[ele[0].split("/")[-1].split("]")[-1]]["test"] = ele[0] + '/' + doc
    return data


def get_hyps_refs_origs(filepath):
    doc = json.load(open(filepath))
    all_hyp = []
    all_ref = []
    all_orig = []

    for example in doc:
        all_hyp.append(example['hyp'])
        all_ref.append(example['ref'])
        all_orig.append(example['orig'])

    return all_hyp, all_ref, all_orig


def get_avg_rouge(all_hypothesis, all_references, metrics=['rouge-n', 'rouge-l', 'rouge-w']):
    evaluator = rouge.Rouge(metrics=metrics,
                            max_n=4,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg='Avg',
                            # apply_best=apply_best,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    scores = evaluator.get_scores(all_hypothesis, all_references)
    return scores


def get_avg_cr(all_hypothesis, all_originals):
    hyp_avg_len = sum(map(lambda x: len(x), all_hypothesis))
    orig_avg_len = sum(map(lambda x: len(x), all_originals))
    return hyp_avg_len / orig_avg_len


def evaluate_model(model):
    scores = {}
    for type_k, type_v in model.items():
        if type_k == "test":
            hyps, refs, origs = get_hyps_refs_origs(type_v)
            rouge = get_avg_rouge(hyps, refs, metrics=['rouge-l', 'rouge-1'])
            f_rl = rouge['rouge-l']['f']
            f_r1 = rouge['rouge-1']['f']
            cr = get_avg_cr(hyps, origs)
            scores["test"] = {"F1_RL": f_rl, "F1_R1": f_r1, "CR": cr}
        elif type_k == "val":
            for epoch_k, epoch_v in type_v.items():
                hyps, refs, origs = get_hyps_refs_origs(epoch_v)
                rouge = get_avg_rouge(hyps, refs, metrics=['rouge-l', 'rouge-1'])
                f_rl = rouge['rouge-l']['f']
                f_r1 = rouge['rouge-1']['f']
                cr = get_avg_cr(hyps, origs)
                scores[epoch_k] = {"F1_RL": f_rl, "F1_R1": f_r1, "CR": cr}
    return scores


def evaluate(dt):
    scores = pd.DataFrame(columns=["model", "epoch", "F1_RL", "F1_R1", "CR"])
    for model_k, model_v in dt.items():
        score = evaluate_model(model_v)
        for epoch_k, epoch_v in score.items():
            scores.loc[scores.shape[0] + 1] = pd.Series({"model": model_k,
                                                         "epoch": epoch_k,
                                                         "F1_RL": epoch_v["F1_RL"],
                                                         "F1_R1": epoch_v["F1_R1"],
                                                         "CR": epoch_v["CR"]
                                                         })
    return scores


dt = get_data(get_fd(PATH_INPUT))
df = evaluate(dt)
df.to_csv(PATH_OUTPUT + "scores.csv")
df_test = df[df.epoch == "test"]
df_test.to_csv(PATH_OUTPUT + "scores_test.csv")
