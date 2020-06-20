from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import re

PATH_SCORES = "../outputs/"
PATH_PLOTS = "../plots/"


def count_features(label):
    feat_pattern = re.compile(r"[\+-]")
    return len(re.findall(feat_pattern, label))


def mask(df, ns):
    res = []
    for e in df:
        if type(e) != str or count_features(e) in ns:
            res.append(True)
        else:
            res.append(False)
    return res


def plot(df, file_name):
    markers = ["o", "^", "s", "p", "d", "x", "P", "X", "D", "8", "*", "v"]
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ":"]
    plt.figure(figsize=(25, 15))
    sns.pointplot(x="epoch", y="F1", hue='model', kind='line', data=df,
                  markers=markers,
                  linestyles=linestyles,
                  scale=0.8
                  )
    ax = plt.gca()
    xticks = ax.get_xticks()
    ax.set_xticklabels(["test" if x == len(xticks) - 1 else x + 1 for x in xticks])
    plt.savefig(PATH_PLOTS + file_name + "_f1.png")
    plt.clf()

    plt.figure(figsize=(25, 15))
    sns.pointplot(x="epoch", y="CR", hue='model', kind='line', data=df,
                  markers=markers,
                  linestyles=linestyles,
                  scale=0.8
                  )
    ax = plt.gca()
    xticks = ax.get_xticks()
    ax.set_xticklabels(["test" if x == len(xticks) - 1 else x + 1 for x in xticks])
    plt.savefig(PATH_PLOTS + file_name + "_cr.png")
    plt.clf()


sns.set(style="whitegrid")

df = pd.read_csv(PATH_SCORES + "scores.csv")
max_ = max(map(lambda x: int(x) if x != "test" else -1, df.epoch))
df.loc[df.shape[0] + 1] = pd.Series({"model": None, "epoch": max_ + 1, "F1": None, "CR": None})
df.epoch[df.epoch == "test"] = max_ + 2
df.epoch = df.epoch.astype(int)
df = df.sort_values(by=["model", "epoch"])

plot(df, "all")

df_1feat = df[mask(df.model, {0, 1})]
plot(df_1feat, "1feat")

df_2feat = df[mask(df.model, {0, 2})]
plot(df_2feat, "2feat")

df_3feat = df[mask(df.model, {0, 3})]
plot(df_3feat, "3feat")
