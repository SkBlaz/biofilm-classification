#!/usr/bin/env python3
"""
UMAP + Dendrograms with a HIGH-CONTRAST, stable palette.
- Consistent colors across all plots
- Legend shows only labels present in each plot
- Optional distinct markers per label to improve separability
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.cluster import hierarchy

# -----------------------------
# Sample identifier groups
# -----------------------------
bioloska_paralelke = {
    "bioloska_paralelka_1": [
        "13042023_s_Lm_st_L628",
        "18042203_s_Lm_st_L394",
        "28032023_s_Lm_st_L455",
        "25042023_s_Lm_st_19115",
        "30052023_s_Lm_st_L634",
        "04072023_s_Lm_st_L1323",
        "30052023_s_Lm_st_L1764",
        "04072023_s_Lm_st_L1823",
    ],
    "bioloska_paralelka_2": [
        "23052023_s_Lm_st_L628",
        "16052023_s_Lm_st_L394",
        "16052023_s_Lm_st_L455",
        "23052023_s_Lm_st_19115",
        "13062023_s_Lm_st_L634",
        "18072023_s_Lm_st_L1323",
        "13062023_s_Lm_st_L1764",
        "18072023_s_Lm_st_L1823",
    ],
    "bioloska_paralelka_3": [
        "14112023_s_Lm_st_L628",
        "07112023_s_Lm_st_L394",
        "14112023_s_Lm_st_L455",
        "14112023_s_Lm_st_19115",
        "07112023_s_Lm_st_L634",
        "14112023_s_Lm_st_L1323",
        "14112023_s_Lm_st_L1764",
        "07112023_s_Lm_st_L1823",
    ],
}


# -----------------------------
# Helpers
# -----------------------------
def _norm_label(x):
    return str(x).strip()


def get_indices(sample_names, valid_samples):
    valid = set(valid_samples)
    return [i for i, name in enumerate(sample_names) if name in valid]


def _tab20_max_contrast(n):
    """Reorder tab20 to maximize contrast across adjacent picks."""
    base = sns.color_palette("tab20", 20)
    order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    colors = [base[i] for i in order]
    return colors[:n]


def _concat_max_contrast(n):
    """
    Build up to 60 high-contrast colors by concatenating tab20, tab20b, tab20c,
    each reordered to maximize contrast.
    """
    pools = []
    for name in ("tab20", "tab20b", "tab20c"):
        base = sns.color_palette(name, 20)
        order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        pools.extend([base[i] for i in order])
    return pools[:n]


def _husl_fallback(n):
    # Perceptually uniform HUSL for large N (works well for color spacing)
    return sns.husl_palette(n, s=0.90, l=0.55)


def make_high_contrast_palette(all_labels):
    """
    Build a dict[label] -> color with strong separation and colorblind-minded choices.
    - Up to 60 highly contrasting categorical colors (tab20/b/c mixed)
    - Beyond 60, fall back to HUSL spacing
    """
    labs = pd.Series(all_labels).dropna().map(_norm_label)
    uniq = sorted(labs.unique().tolist())
    n = len(uniq)

    try:
        # If distinctipy is installed, it's excellent for many distinct colors
        import distinctipy  # type: ignore

        colors = distinctipy.get_colors(n, pastel_factor=0.2)
    except Exception:
        if n <= 20:
            colors = _tab20_max_contrast(n)
        elif n <= 60:
            colors = _concat_max_contrast(n)
        else:
            colors = _husl_fallback(n)

    return {lab: colors[i] for i, lab in enumerate(uniq)}


def make_marker_map(all_labels):
    """
    Optional: distinct markers per label to further aid differentiation.
    Seaborn supports a limited set; we cycle them deterministically.
    """
    markers_cycle = ["o", "s", "^", "P", "X", "D", "v", ">", "<", "*", "h", "H"]
    labs = pd.Series(all_labels).dropna().map(_norm_label)
    uniq = sorted(labs.unique().tolist())
    return {lab: markers_cycle[i % len(markers_cycle)] for i, lab in enumerate(uniq)}


def plot_umap(df, title, filename, global_palette, global_markers=None, label_col_norm="label_norm"):
    """
    Plot a UMAP projection using:
    - sub-palette of only present labels (consistent colors)
    - optional per-label markers for additional separability
    """
    if label_col_norm not in df.columns:
        print(f"[WARN] Skipping '{title}': '{label_col_norm}' column missing.")
        return

    X = df.select_dtypes(include=[np.number]).dropna(axis=1)
    if X.shape[0] < 2 or X.shape[1] < 1:
        print(f"[WARN] Skipping '{title}': insufficient data for UMAP (n_samples={X.shape[0]}, n_features={X.shape[1]}).")
        return

    embedding = umap.UMAP(random_state=42).fit_transform(X)

    plot_df = pd.DataFrame({"UMAP1": embedding[:, 0], "UMAP2": embedding[:, 1], label_col_norm: df[label_col_norm].map(_norm_label).values})

    present = sorted(plot_df[label_col_norm].dropna().unique().tolist())
    sub_palette = {lab: global_palette[lab] for lab in present if lab in global_palette}
    sub_markers = {lab: global_markers[lab] for lab in present} if global_markers is not None else True

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x="UMAP1",
        y="UMAP2",
        hue=label_col_norm,
        palette=sub_palette,
        style=label_col_norm if global_markers is not None else None,
        markers=sub_markers if global_markers is not None else True,
        s=60,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.4,
        legend=True,
    )
    plt.gca().set_aspect("equal", "datalim")
    plt.title(title, fontsize=24)
    plt.legend(title="Strain", ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(filename)


def process_and_plot_samples(group_name, valid_samples, data, global_palette, global_markers):
    parsed_names = ["_".join(str(x).split("--")[:5]) for x in data.sampleName.astype(str).tolist()]
    indices = get_indices(parsed_names, valid_samples)
    subset = data.iloc[indices].copy()

    if subset.empty:
        print(f"[INFO] No samples found for group '{group_name}'. Skipping.")
        return

    plot_umap(subset, f"UMAP visualization ({group_name})", f"umap_proj_{group_name}.tif", global_palette, global_markers)


# -----------------------------
# Main
# -----------------------------
def main():
    # =======================
    # Dendrograms
    # =======================
    dfx = pd.read_csv("dendrogrami_input.csv", sep="\t").replace({"+": 1, "-": 0})
    print(dfx)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif="Computer Modern")

    row1 = ["L628", "L455", "L1764", "L394", "L634", "L1323", "L1823", "ATCC 19115"]
    row2 = ["CC1", "CC1", "CC121", "CC9", "CC4", "CC8", "CC6", "CC2"]
    row3 = [
        s.lower()
        for s in ["Animal clinical", "Animal feed", "Food", "Food", "Animal subclinical", "Human clinical", "Food", "Human clinical"]
    ]
    dfx.columns = [f"{a} - {b} ({c})" for a, b, c in zip(row1, row2, row3)]

    hierarchy.set_link_color_palette(["black"])
    for method in ["single", "weighted", "complete", "average", "median"]:
        Z = hierarchy.linkage(dfx.T, method=method)
        plt.figure(figsize=(10, 6))
        hierarchy.dendrogram(Z, labels=dfx.columns, color_threshold=1.2, link_color_func=lambda x: "black", orientation="left")
        plt.title(f"Dendrogram ({method})")
        plt.tight_layout()
        plt.savefig(f"dendrogram_{method}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"dendrogram_{method}.tif", dpi=300, bbox_inches="tight")
        print(f"dendrogram_{method}.pdf")
        plt.close()

    # =======================
    # UMAP projections
    # =======================
    data2 = pd.read_csv("datafile.tsv", sep="\t")

    if "label" not in data2.columns:
        raise ValueError("Column 'label' not found in datafile.tsv")
    if "sampleName" not in data2.columns:
        raise ValueError("Column 'sampleName' not found in datafile.tsv")

    # Normalize once for consistent palette keys
    data2["label_norm"] = data2["label"].map(_norm_label)

    # Global high-contrast palette (+ optional markers)
    global_palette = make_high_contrast_palette(data2["label_norm"])
    # Comment this line if you want colors only:
    global_markers = make_marker_map(data2["label_norm"])

    # Dates (or 'all')
    dates = {str(x).split("-")[0] for x in data2.sampleName.tolist()} | {"all"}

    # UMAP for predefined sample groups
    for group, samples in bioloska_paralelke.items():
        process_and_plot_samples(group, samples, data2, global_palette, global_markers)

    # UMAP for each date (or all data)
    for date in sorted(dates):
        subset = data2.copy() if date == "all" else data2[data2.sampleName.astype(str).str.contains(date)]
        if subset.empty:
            print(f"[INFO] No samples for date '{date}'. Skipping.")
            continue

        plot_umap(subset, f"UMAP visualization ({date})", f"umap_proj_{date}.tif", global_palette, global_markers)


if __name__ == "__main__":
    main()
