import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
import umap

# Sample identifier groups
bioloska_paralelke = {
    "bioloska_paralelka_1": [
        "13042023_s_Lm_st_L628", "18042203_s_Lm_st_L394",
        "28032023_s_Lm_st_L455", "25042023_s_Lm_st_19115",
        "30052023_s_Lm_st_L634", "04072023_s_Lm_st_L1323",
        "30052023_s_Lm_st_L1764", "04072023_s_Lm_st_L1823"
    ],
    "bioloska_paralelka_2": [
        "23052023_s_Lm_st_L628", "16052023_s_Lm_st_L394",
        "16052023_s_Lm_st_L455", "23052023_s_Lm_st_19115",
        "13062023_s_Lm_st_L634", "18072023_s_Lm_st_L1323",
        "13062023_s_Lm_st_L1764", "18072023_s_Lm_st_L1823"
    ],
    "bioloska_paralelka_3": [
        "14112023_s_Lm_st_L628", "07112023_s_Lm_st_L394",
        "14112023_s_Lm_st_L455", "14112023_s_Lm_st_19115",
        "07112023_s_Lm_st_L634", "14112023_s_Lm_st_L1323",
        "14112023_s_Lm_st_L1764", "07112023_s_Lm_st_L1823"
    ]
}

def get_indices(sample_names, valid_samples):
    valid = set(valid_samples)
    return [i for i, name in enumerate(sample_names) if name in valid]

def plot_umap(df, title, filename):
    labels = df.pop('label').tolist()
    # Use columns starting at index 2 and drop columns with any NaNs
    X = df.iloc[:, 2:].dropna(axis=1)
    embedding = umap.UMAP().fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="colorblind")
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontsize=24)
    plt.legend(ncol=4)
    plt.savefig(filename, dpi=300)
    plt.clf(); plt.cla()

def process_and_plot_samples(group_name, valid_samples, data):
    parsed_names = ["_".join(x.split("--")[:5]) for x in data.sampleName.tolist()]
    indices = get_indices(parsed_names, valid_samples)
    subset = data.iloc[indices].copy()
    plot_umap(subset, f'UMAP visualization ({group_name})', f"umap_proj_{group_name}.tif")

def main():
    # Read and preprocess dendrogram input data
    dfx = pd.read_csv("dendrogrami_input.csv", sep="\t").replace({"+": 1, "-": 0})
    print(dfx)

    # Configure LaTeX rendering in matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Computer Modern')

    # Rename columns with descriptive labels
    row1 = ["L628", "L455", "L1764", "L394", "L634", "L1323", "L1823", "ATCC 19115"]
    row2 = ["CC1", "CC1", "CC121", "CC9", "CC4", "CC8", "CC6", "CC2"]
    row3 = [s.lower() for s in ["Animal clinical", "Animal feed", "Food", "Food",
                                 "Animal subclinical", "Human clinical", "Food", "Human clinical"]]
    dfx.columns = [f"{a} - {b} ({c})" for a, b, c in zip(row1, row2, row3)]

    # Set dendrogram link colors to black and generate dendrograms for each method
    hierarchy.set_link_color_palette(["black"])
    for method in ['single', 'weighted', 'complete', 'average', 'median']:
        Z = hierarchy.linkage(dfx.T, method=method)
        plt.figure(figsize=(10, 6))
        hierarchy.dendrogram(
            Z, labels=dfx.columns, color_threshold=1.2,
            link_color_func=lambda x: 'black', orientation='left'
        )
        plt.title(f"Dendrogram ({method})")
        plt.tight_layout()
        plt.savefig(f"dendrogram_{method}.pdf", dpi=300)
        plt.savefig(f"dendrogram_{method}.tif", dpi=300)
        print(f"dendrogram_{method}.pdf")
        plt.close()

    # UMAP projections using a separate dataset
    data2 = pd.read_csv("datafile.tsv", sep="\t")
    dates = {x.split("-")[0] for x in data2.sampleName.tolist()} | {"all"}
    
    # UMAP for predefined sample groups
    for group, samples in bioloska_paralelke.items():
        process_and_plot_samples(group, samples, data2)
    
    # UMAP for each date (or all data)
    for date in dates:
        subset = data2.copy() if date == "all" else data2[data2.sampleName.str.contains(date)]
        plot_umap(subset.copy(), f'UMAP visualization ({date})', f"umap_proj_{date}.tif")

if __name__ == "__main__":
    main()
