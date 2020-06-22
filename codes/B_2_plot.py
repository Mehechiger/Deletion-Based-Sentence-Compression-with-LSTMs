from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import re

PATH_SCORES = "../outputs/"
PATH_PLOTS = "../plots/"


def count_features(label):
    feat_pattern = re.compile(r"[\+-]")
    return len(re.findall(feat_pattern, label))


def mask_count_features(df, ns):
    res = []
    for e in df:
        if type(e) != str or count_features(e) in ns:
            res.append(True)
        else:
            res.append(False)
    return res


def mask(df, plot):
    res = []
    for e in df:
        if type(e) != str or e in plot:
            res.append(True)
        else:
            res.append(False)
    return res


def plot(df, file_name):
    markers = ["o", "^", "s", "p", "d", "x", "P", "X", "D", "8", "*", "v"]
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ":"]

    plt.figure(figsize=(25, 15))
    sns.pointplot(x="epoch", y="F1_RL", hue='model', kind='line', data=df,
                  markers=markers,
                  linestyles=linestyles,
                  scale=0.8
                  )
    ax = plt.gca()
    xticks = ax.get_xticks()
    ax.set_xticklabels(["test" if x == len(xticks) - 1 else x + 1 for x in xticks])
    plt.savefig(PATH_PLOTS + file_name + "_f1rl.png")
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
df.loc[df.shape[0] + 1] = pd.Series({"model": None, "epoch": max_ + 1, "F1_RL": None, "CR": None})
df.epoch[df.epoch == "test"] = max_ + 2
df.epoch = df.epoch.astype(int)
df = df.sort_values(by=["model", "epoch"])

models = list(set(df.model))
for i, v in enumerate(models):
    if type(v) == str:
        print("%d - %s" % (i, v))
print("choose to plot, separate model with 1 space, return to separate plot, return again to end:")
to_plots = []
while True:
    input_ = input()
    if input_ == "":
        break
    to_plots.append(list(map(lambda x: models[int(x)], input_.split(" "))))
print("ploting:")
for to_plot in to_plots:
    print(to_plot, end="...")
    df_plot = df[mask(df.model, to_plot)]
    plot(df_plot, "|".join(to_plot))
    print(" done")

print("1feat", end="...")
df_1feat = df[mask_count_features(df.model, [0, 1])]
plot(df_1feat, "1feat")
print(" done")

print("all", end="...")
plot(df, "all")
print(" done")
