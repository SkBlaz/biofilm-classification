"""
Code that should be used as a part of the analysis pipeline.
"""

import argparse

from visualize import top_n_visualization


class VisualizationPipeline:
    def __init__(self, data_path: str, rankings_path: str, output_folder: str, nb_features: int):
        self.data_path = data_path
        self.rankings_path = rankings_path
        self.output_folder = output_folder
        self.nb_features = nb_features

    def run(self):
        # top N based on training set; slice by strain and date
        top_n_visualization(
            data_path=self.data_path,
            rankings_path=self.rankings_path,
            output_folder=self.output_folder,
            print_top_n=self.nb_features,
            x_col="pool",
            facet_strategy="strain_date",
        )

        # top N based on training set; slice by strain
        top_n_visualization(
            data_path=self.data_path,
            rankings_path=self.rankings_path,
            output_folder=self.output_folder,
            print_top_n=self.nb_features,
            x_col="pool",
            facet_strategy="strain",
        )

        # top N based on training set; slice by strain
        top_n_visualization(
            data_path=self.data_path,
            rankings_path=self.rankings_path,
            output_folder=self.output_folder,
            print_top_n=self.nb_features,
            x_col="strain",
            facet_strategy=None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data",
        help="A TSV file with a prepared dataset to analyze (path to it)",
        required=True,
    )
    parser.add_argument(
        "--rankings",
        help="A TSV file with rankings calculated from the input dataset (path to it)",
        required=True,
    )
    parser.add_argument(
        "--fout",
        help="The results will be placed in this folder",
        required=True,
    )
    parser.add_argument(
        "--nbfeatures",
        help="How many top N features to visualize?",
        default=10,
        required=True,
    )
    try:
        arguments = parser.parse_args()
    except SystemExit:
        parser.print_help()
        exit(999)

    helper = VisualizationPipeline(
        data_path=arguments.data,
        rankings_path=arguments.rankings,
        output_folder=arguments.fout,
        nb_features=arguments.nbfeatures,
    )
    helper.run()
