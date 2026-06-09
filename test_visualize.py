#!/usr/bin/env python3
"""
Unit tests for src/visualizations/visualize.py.
"""

import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, "src/visualizations")

from visualize import make_feature_name_disk_writable, require_columns, top_n_visualization


class TestTopNVisualization(unittest.TestCase):
    def test_require_columns_reports_missing_and_available_columns(self):
        df = pd.DataFrame({"present": [1]})

        with self.assertRaisesRegex(ValueError, "Missing required columns.*missing.*present"):
            require_columns(df, ["present", "missing"], "demo")

    def test_make_feature_name_disk_writable_sanitizes_problem_characters(self):
        self.assertEqual(make_feature_name_disk_writable("counts(inten<0.1"), "counts-inten_lt_0.1")

    def test_top_n_visualization_uses_named_ranking_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.tsv")
            rankings_path = os.path.join(tmpdir, "rankings.tsv")
            output_path = os.path.join(tmpdir, "out")

            pd.DataFrame(
                {
                    "sampleName": ["13042023--s--Lm--st--L628--p--C03--pos001--tm--24--ch--Syto9--z--21"],
                    "feat1": [0.42],
                }
            ).to_csv(data_path, sep="\t", index=False)

            pd.DataFrame(
                {
                    "feature": ["feat1"],
                    "RandomForest(n=200, p=1.0)": [0.9],
                }
            ).to_csv(rankings_path, sep="\t", index=False)

            top_n_visualization(
                data_path=data_path,
                rankings_path=rankings_path,
                output_folder=output_path,
                print_top_n=1,
                x_col="strain",
                facet_strategy=None,
            )

            expected_html = os.path.join(output_path, "top1_strain", "000_feat1.html")
            self.assertTrue(os.path.exists(expected_html))

    def test_top_n_visualization_requires_ranking_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.tsv")
            rankings_path = os.path.join(tmpdir, "rankings.tsv")
            output_path = os.path.join(tmpdir, "out")

            pd.DataFrame(
                {
                    "sampleName": ["13042023--s--Lm--st--L628--p--C03--pos001--tm--24--ch--Syto9--z--21"],
                    "feat1": [0.42],
                }
            ).to_csv(data_path, sep="\t", index=False)

            pd.DataFrame({"feature": ["feat1"]}).to_csv(rankings_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                top_n_visualization(
                    data_path=data_path,
                    rankings_path=rankings_path,
                    output_folder=output_path,
                    print_top_n=1,
                    x_col="strain",
                    facet_strategy=None,
                )

    def test_top_n_visualization_rejects_out_of_range_top_n(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.tsv")
            rankings_path = os.path.join(tmpdir, "rankings.tsv")
            output_path = os.path.join(tmpdir, "out")

            pd.DataFrame(
                {
                    "sampleName": ["13042023--s--Lm--st--L628--p--C03--pos001--tm--24--ch--Syto9--z--21"],
                    "feat1": [0.42],
                }
            ).to_csv(data_path, sep="\t", index=False)

            pd.DataFrame({"feature": ["feat1"], "RandomForest(n=200, p=1.0)": [0.9]}).to_csv(rankings_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, r"range of \[1, 1\]"):
                top_n_visualization(
                    data_path=data_path,
                    rankings_path=rankings_path,
                    output_folder=output_path,
                    print_top_n=2,
                    x_col="strain",
                    facet_strategy=None,
                )

    def test_top_n_visualization_rejects_empty_rankings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.tsv")
            rankings_path = os.path.join(tmpdir, "rankings.tsv")
            output_path = os.path.join(tmpdir, "out")

            pd.DataFrame({"sampleName": ["13042023--s--Lm--st--L628--p--C03--pos001--tm--24--ch--Syto9--z--21"], "feat1": [0.42]}).to_csv(
                data_path, sep="\t", index=False
            )
            pd.DataFrame(columns=["feature", "RandomForest(n=200, p=1.0)"]).to_csv(rankings_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "No ranking data"):
                top_n_visualization(
                    data_path=data_path,
                    rankings_path=rankings_path,
                    output_folder=output_path,
                    print_top_n=1,
                    x_col="strain",
                    facet_strategy=None,
                )

    def test_top_n_visualization_rejects_invalid_sample_name_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.tsv")
            rankings_path = os.path.join(tmpdir, "rankings.tsv")
            output_path = os.path.join(tmpdir, "out")

            pd.DataFrame({"sampleName": ["invalid"], "feat1": [0.42]}).to_csv(data_path, sep="\t", index=False)
            pd.DataFrame({"feature": ["feat1"], "RandomForest(n=200, p=1.0)": [0.9]}).to_csv(rankings_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "invalid sampleName format"):
                top_n_visualization(
                    data_path=data_path,
                    rankings_path=rankings_path,
                    output_folder=output_path,
                    print_top_n=1,
                    x_col="strain",
                    facet_strategy=None,
                )


if __name__ == "__main__":
    unittest.main()
