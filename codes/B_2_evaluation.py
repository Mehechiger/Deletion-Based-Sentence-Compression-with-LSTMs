"""
from os import walk

import rouge
import json
import re


def get_fd(dataset):
    fd = []
    for (dirpath, dirnames, filenames) in walk(dataset):
        # print(dirpath, dirnames, filenames)
        fd.append((dirpath, filenames))
    # print(fd)
    return fd


def get_data(fd):
    data = []
    for ele in fd:
        if ele[1] and ele[1] != ['.DS_Store']:
            for doc in ele[1]:
                if re.match('val_res', doc):
                    data.append((ele[0].split("/")[-1], doc, ele[0] + '/' + doc))
    return data


def get_data_test(fd):
    data = []
    for ele in fd:
        if ele[1] and ele[1] != ['.DS_Store']:
            for doc in ele[1]:
                if re.match('test_', doc):
                    data.append((ele[0].split("/")[-1], doc, ele[0] + '/' + doc))
    return data


def get_hyps_refs(filepath):
    doc = json.load(open(filepath))
    all_hyp = []
    all_ref = []

    for example in doc:
        all_hyp.append(example['hyp'])
        all_ref.append(example['ref'])

    return all_hyp, all_ref


def save_data(data, outputname):
    json_str = json.dumps(data, indent=4)
    with open(outputname + '.json', 'w') as json_file:
        json_file.write(json_str)


def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1',
                                                                 100.0 * f)


def get_avg(all_hypothesis, all_references):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
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

    # for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    #        print(prepare_results(metric, results['p'], results['r'], results['f']))
    return scores


def print_scores(all_hypothesis, all_references):
    # for aggregator in ['Avg', 'Best', 'Individual']:
    for aggregator in ['Avg', 'Best']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=4,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=0.5,  # Default F1_score
                                weight_factor=1.2,
                                stemming=True)

        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(metric, results_per_ref['p'][reference_id],
                                                     results_per_ref['r'][reference_id],
                                                     results_per_ref['f'][reference_id]))
                print()
            else:
                print(prepare_results(metric, results['p'], results['r'], results['f']))
        print()


def evaluate(data):
    results = []
    for d in data:
        model, epoque, filepath = d
        all_hyp, all_ref = get_hyps_refs(filepath)
        # print(model)
        # print(epoque)
        scores = get_avg(all_hyp, all_ref)
        results.append((model, epoque, scores))
    return results


# data = get_data(get_fd("./dataset/"))
# print(data)
# results = evaluate(data)
# save_data(results, 'results')

# data = get_data_test(get_fd("./dataset/"))
# results = evaluate(data)
# save_data(results, 'results_test')
results = json.load(open('results_test.json'))
re2 = []
for model in results:
    m, t, r = model
    re2.append((m, r['rouge-l']['f']))
re_sorted = sorted(re2, key=lambda x: x[1], reverse=True)
for e in re_sorted:
    print(e)
# hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"
# references_1 = ["Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\nKing Sihanouk declined to chair talks in either place.\nA U.S. House resolution criticized Hun Sen's regime while the opposition tried to cut off his access to loans.\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\nLeft out, Sam Rainsy sought the King's assurance of Hun Sen's promise of safety and freedom for all politicians.",
#                "Cambodian prime minister Hun Sen rejects demands of 2 opposition parties for talks in Beijing after failing to win a 2/3 majority in recent elections.\nSihanouk refuses to host talks in Beijing.\nOpposition parties ask the Asian Development Bank to stop loans to Hun Sen's government.\nCCP defends Hun Sen to the US Senate.\nFUNCINPEC refuses to share the presidency.\nHun Sen and Ranariddh eventually form a coalition at summit convened by Sihanouk.\nHun Sen remains prime minister, Ranariddh is president of the national assembly, and a new senate will be formed.\nOpposition leader Rainsy left out.\nHe seeks strong assurance of safety should he return to Cambodia.\n",
#                ]

# hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of China '' to incite the subversion of state power , '' according to court documents given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"
# references_2 = "Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\n"

# all_hypothesis = [hypothesis_1, hypothesis_2]
# all_references = [references_1, references_2]


# print_scores(all_hypothesis, all_references)

# data3 = json.load(open('val_res_epoch3.json', 'r'))
# data20 = json.load(open('val_res_epoch20.json','r'))

# data1 = json.load(open('./COPIES_to_be_Rouged/batchsize2_train0.001_val_res_epoch1_0.489.json'))
# data2 = json.load(open('./COPIES_to_be_Rouged/greedy_batch32_train0.1_val_res_epoch2_0.351.json'))
# data3 = json.load(open('./COPIES_to_be_Rouged/teacherforceoff_train0.1_val_res_epoch1_0.383.json'))

# for epoque in range(1, 4):
#  print("File = %d" %epoque)
#  all_hyp, all_ref = make_examples(eval('data%d' %epoque))
#  print_scores(all_hyp, all_ref)

# data1 = json.load(open('val_res_epoch1.json'))
# all_hyp, all_ref = get_evaluation_data(data1)
# print_scores(all_hyp, all_ref)
"""




from os import walk
from collections import defaultdict
import rouge
import json
import re

PATH_INPUT = "../outputs/"
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
                    data[ele[0].split("/")[-1]]["val"][int(doc.split('.')[0][13:])] = ele[0] + '/' + doc
                if re.match('test_', doc):
                    data[ele[0].split("/")[-1]]["test"] = ele[0] + '/' + doc
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
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for type_k, type_v in model.items():
        if type_k == "test":
            hyps, refs, origs = get_hyps_refs_origs(type_v)
            f = get_avg_rouge(hyps, refs, metrics=['rouge-l'])['rouge-l']['f']
            cr = get_avg_cr(hyps, origs)
            scores["test"]['f1'] = f
            scores["test"]['cr'] = cr
        elif type_k == "val":
            for epoch_k, epoch_v in type_v.items():
                hyps, refs, origs = get_hyps_refs_origs(epoch_v)
                f = get_avg_rouge(hyps, refs, metrics=['rouge-l'])['rouge-l']['f']
                cr = get_avg_cr(hyps, origs)
                scores["val"][epoch_k]['f1'] = f
                scores["val"][epoch_k]['cr'] = cr
    return scores


dt = get_data(get_fd(PATH_INPUT))

for model_k, model_v in dt.items():
    print(model_k, evaluate_model(model_v))
