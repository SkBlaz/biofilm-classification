# visualize classifications
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import matplotlib
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "sans-serif",
#    "font.sans-serif": "Helvetica",
#})

results_folder = "/imagine/results"

for result in glob.glob(results_folder + "/classification*"):

    dfx = pd.read_csv(result, sep="\t")
    dfx = dfx[dfx['n_components'] != 580]
    models = dfx.model.values.tolist()
    models_tmp = []
    for model in models:
        if "autogluon" in model:
            models_tmp.append("AutoGluon")
        else:
            models_tmp.append(model)            
    models = [x.split("(")[0] for x in models_tmp]
    models = [x.replace("GridSearchCV", "KNN-grid") for x in models]
    models = [x.replace("DummyClassifier", "MajorityClassifier") for x in models]
    dfx.model = models
    dfx = dfx.sort_values(by=['accuracy'])
    plt.clf(); plt.cla()
    plt.title(result.split("/")[-1].replace(".tsv", ""))
    order = ['MajorityClassifier', 'LogisticRegression', 'DecisionTreeClassifier', 'KNN-grid', 'RandomForestClassifier','XGBClassifier', 'AutoGluon']
    sns.barplot(y=dfx.model, x=dfx.accuracy, color="black", errwidth=0.5, capsize=0.5, palette="colorblind", hue=dfx.n_components, alpha=0.5, order=order)    
    plt.ylabel("")
    plt.xlabel("Accuracy")
    plt.legend(loc='lower left')
    plt.tight_layout()
    img_name = result.split("/")[-1].replace(".tsv", "")
    plt.savefig(f"{results_folder}/visualizations/{img_name}.pdf", dpi=300)
    print(f"{img_name}.pdf")

plt.clf();plt.cla()
ablation_df = pd.read_csv(os.path.join(results_folder, "ablation_ranking_all.tsv"), sep="\t")
ablation_df = ablation_df[ablation_df.top_n < 1000]
max_top_n = ablation_df[ablation_df.accuracy == max(ablation_df.accuracy)]

sns.lineplot(x=ablation_df.top_n, y=ablation_df.accuracy)
plt.vlines(max_top_n.top_n, 0, max_top_n.accuracy, color="red", linestyle='dashed')
plt.plot(max_top_n.top_n, max_top_n.accuracy, "ro")
plt.text(max_top_n.top_n + 15, max_top_n.accuracy, f"Num features: {int(max_top_n.top_n)}, accuracy: {round(float(max_top_n.accuracy), 3)}")
plt.xlabel("Top n features considered (RF ranking)")
plt.ylabel("Mean accuracy (3-fold cross validation)")
plt.tight_layout()
plt.savefig(f"{results_folder}/visualizations/ablation_rf.pdf", dpi=300)
