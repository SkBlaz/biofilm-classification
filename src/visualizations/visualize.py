import itertools
import os
import warnings
from math import ceil

import pandas as pd
import plotly.express as px

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


def draw(feature, df_original, x_col, facet_col):
    colwrap = 3

    if facet_col:
        strains = list(df_original[facet_col].unique())
        rows = list(range(1, ceil(len(strains) / colwrap) + 1))[::-1]
    else:
        rows = [1]

    cfg = {"data_frame": df_original, "x": x_col, "y": feature, "points": "all", "title": "Feature: " + feature}

    if facet_col:
        cfg["facet_col"] = facet_col
        cfg["facet_col_wrap"] = colwrap
        cfg["facet_row_spacing"] = 0.03
        cfg["height"] = 400 * len(rows)
        cfg["color"] = "pool"
    else:
        cfg["color"] = x_col

    fig = px.box(**cfg)
    fig.update_yaxes(title="")

    if facet_col:
        facets = list(itertools.product(rows, list(range(1, colwrap + 1))))

        for strain, facet in zip(strains, facets):
            d = df_original[df_original[facet_col] == strain]
            mdn = d[feature].median()
            fig.add_hline(
                y=mdn,
                line_dash="dot",
                annotation_text=f"{strain} median",
                row=facet[0],
                col=facet[1],
            )
    else:
        mdn = df_original[feature].median()
        fig.add_hline(y=mdn, line_dash="dot", annotation_text="median")

    return fig


def top_n_visualization(data_path, rankings_path, output_folder, print_top_n=None, x_col=None, facet_strategy=False):
    """
    This script visualizes topN features in the input dataset by using rankings.

    :param data_path: rel/abs PATH to the dataset
    :param rankings_path: rel/abs PATH to the ranking file
    :param output_folder: output folder NAME (it will be created in the same directory as this script)
    :param print_top_n: how many features to visualize? None = ALL or [1, nbfeatures]
    :param x_col: column for x-axis
    :param facet_strategy: how should data be sliced; valid values 'strain', 'strain_date', None
    """

    ranking_col = "RandomForest(n=200, p=1.0)"

    # read rankings
    df_rankings = pd.read_csv(rankings_path, sep="\t")
    df_rankings = df_rankings.sort_values(by=[ranking_col], ascending=False)

    # check top n parameter
    if print_top_n is None:
        limit = df_rankings.shape[0]
    elif int(print_top_n) < 1 or int(print_top_n) > df_rankings.shape[0]:
        raise ValueError(f"print_top_n should be in the range of [1, {df_rankings.shape[0]}]")
    else:
        limit = int(print_top_n)

    if not facet_strategy:
        output_folder += f"/top{limit}_{x_col}"
    else:
        output_folder += f"/top{limit}_{x_col}_vs_{facet_strategy}"

    os.makedirs(output_folder, exist_ok=True)

    # read data
    df_original = pd.read_csv(data_path, sep="\t")

    # add visualization features
    sample_names = df_original["sampleName"].str.split("--", expand=True)
    strains = sample_names[4]
    strains_dates = sample_names[4] + " @ " + sample_names[0]
    pools = sample_names[6]
    # positions = sample_names[7]

    df_original["pool"] = pools
    df_original["strain"] = strains
    df_original["strain_date"] = strains_dates
    # df_original["position"] = positions

    threshold = 10**-3

    print(f"Creating visualizations for top {limit} features.\n\n")

    for i in range(0, limit):
        r = df_rankings[["feature", ranking_col]].iloc[i]
        feature, score = r[0], r[1]

        if score < threshold:
            print(f"Importance score {score} ({ranking_col}) lower than {threshold}. Quitting...")
            break

        newname = make_feature_name_disk_writable(feature)
        if newname != feature:
            print(f"\t{i}\t{score}\t{feature}\t-> RENAMED TO:\t{newname}")

        fig = draw(feature=feature, df_original=df_original, x_col=x_col, facet_col=facet_strategy)

        fig.write_html(f"{output_folder}/{i:03}_{newname}.html", include_plotlyjs="cdn")


def make_feature_name_disk_writable(feature: str):
    return feature.replace("(", "-").replace("<", "_lt_")


if __name__ == "__main__":
    cfg = [
        (
            "../../prepared_data/2023-09-27-2d.tsv",
            "../ranking_results/rankings_2023-09-27-2d.tsv",
            "2023-09-27/2d",
        ),
        (
            "../../prepared_data/2023-09-27-3d.tsv",
            "../ranking_results/rankings_2023-09-27-3d.tsv",
            "2023-09-27/3d",
        ),
        (
            "../../prepared_data/2023-09-27-2dd.tsv",
            "../ranking_results/rankings_2023-09-27-2dd.tsv",
            "2023-09-27/2dd",
        ),
        (
            "../../prepared_data/2023-09-27-3dd.tsv",
            "../ranking_results/rankings_2023-09-27-3dd.tsv",
            "2023-09-27/3dd",
        ),
    ]
    for data, rankings, root in cfg:
        top_n_visualization(
            data_path=data,
            rankings_path=rankings,
            output_folder=root,
            print_top_n=10,
        )
