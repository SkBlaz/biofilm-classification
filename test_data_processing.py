#!/usr/bin/env python3
"""
Unit tests for data processing utilities.
"""

import os
import sys
import tempfile
import unittest

import pandas as pd

# Add src directory to path
sys.path.insert(0, "src")

from analysis import aggregate_raw_features
from create_final_df_from_results import create_final_dataframe
from create_joint_df import create_joint_dataframe, extract_data


class TestExtractData(unittest.TestCase):
    """Test data extraction functionality."""

    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test CSV files with different identifiers
        self.custom_algos_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        self.diff_global_data = pd.DataFrame({"feature_a": [7, 8, 9], "feature_b": [10, 11, 12]})

        # Save files
        self.custom_file = os.path.join(self.temp_dir, "test_CustomAlgos_data.txt")
        self.custom_algos_data.to_csv(self.custom_file, sep="\t", index=False)

        self.diff_file = os.path.join(self.temp_dir, "test_DiffGlobal_data.txt")
        self.diff_global_data.to_csv(self.diff_file, sep="\t", index=False)

        self.other_file = os.path.join(self.temp_dir, "test_Other_data.txt")
        pd.DataFrame({"col1": [1, 2]}).to_csv(self.other_file, sep="\t", index=False)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_extract_custom_algos(self):
        """Test extracting CustomAlgos data."""
        namespace_identifiers = ["CustomAlgos", "DiffGlobal"]
        identifier, df = extract_data(self.custom_file, namespace_identifiers)

        self.assertEqual(identifier, "CustomAlgos")
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, self.custom_algos_data.shape)

    def test_extract_diff_global(self):
        """Test extracting DiffGlobal data."""
        namespace_identifiers = ["CustomAlgos", "DiffGlobal"]
        identifier, df = extract_data(self.diff_file, namespace_identifiers)

        self.assertEqual(identifier, "DiffGlobal")
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, self.diff_global_data.shape)

    def test_no_matching_identifier(self):
        """Test when no identifier matches."""
        namespace_identifiers = ["CustomAlgos", "DiffGlobal"]
        identifier, df = extract_data(self.other_file, namespace_identifiers)

        self.assertIsNone(identifier)
        self.assertIsNone(df)

    def test_empty_identifiers(self):
        """Test with empty identifier list."""
        namespace_identifiers = []
        identifier, df = extract_data(self.custom_file, namespace_identifiers)

        self.assertIsNone(identifier)
        self.assertIsNone(df)

    def test_identifier_priority(self):
        """Test that first matching identifier is returned."""
        # Create a file with multiple identifiers
        multi_file = os.path.join(self.temp_dir, "test_CustomAlgos_DiffGlobal.txt")
        pd.DataFrame({"col": [1]}).to_csv(multi_file, sep="\t", index=False)

        # CustomAlgos should be found first
        namespace_identifiers = ["CustomAlgos", "DiffGlobal"]
        identifier, df = extract_data(multi_file, namespace_identifiers)

        self.assertEqual(identifier, "CustomAlgos")
        self.assertIsNotNone(df)

    def test_file_read_error(self):
        """Test handling of file read errors."""
        namespace_identifiers = ["CustomAlgos"]

        with self.assertRaises(FileNotFoundError):
            extract_data("/nonexistent/CustomAlgos_file.txt", namespace_identifiers)


class TestDataProcessingUtilities(unittest.TestCase):
    """Test additional data processing utilities."""

    def test_dataframe_groupby_operations(self):
        """Test that pandas groupby operations work as expected."""
        # This tests the operations used in analysis.py
        df = pd.DataFrame({"sampleName": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        # Test median
        median_df = df.groupby(["sampleName"]).median().reset_index()
        self.assertEqual(len(median_df), 2)
        self.assertEqual(median_df[median_df["sampleName"] == "A"]["value"].values[0], 1.5)

        # Test mean
        mean_df = df.groupby(["sampleName"]).mean().reset_index()
        self.assertEqual(len(mean_df), 2)
        self.assertEqual(mean_df[mean_df["sampleName"] == "A"]["value"].values[0], 1.5)

        # Test std
        std_df = df.groupby(["sampleName"]).std().reset_index()
        self.assertEqual(len(std_df), 2)

    def test_dataframe_quantile_operations(self):
        """Test quantile operations used in analysis.py."""
        df = pd.DataFrame({"sampleName": ["A"] * 10, "value": range(10)})

        # Test q10 (10th percentile)
        q10_df = df.groupby(["sampleName"]).quantile(0.10).reset_index()
        self.assertEqual(len(q10_df), 1)
        self.assertAlmostEqual(q10_df["value"].values[0], 0.9, places=1)

        # Test q75 (75th percentile)
        q75_df = df.groupby(["sampleName"]).quantile(0.75).reset_index()
        self.assertEqual(len(q75_df), 1)
        self.assertAlmostEqual(q75_df["value"].values[0], 6.75, places=1)

    def test_dataframe_concat(self):
        """Test dataframe concatenation used in create_joint_df.py."""
        df1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})

        result = pd.concat([df1, df2], axis=0)

        self.assertEqual(len(result), 4)
        self.assertEqual(result.shape[1], 2)
        self.assertListEqual(result["col1"].tolist(), [1, 2, 5, 6])

    def test_dataframe_pivot(self):
        """Test pivot operations used in create_final_df_from_results.py."""
        df = pd.DataFrame({"sampleName": ["A", "A", "B", "B"], "variable": ["x", "y", "x", "y"], "value": [1, 2, 3, 4]})

        pivoted = df.pivot(index="sampleName", columns="variable", values="value")

        self.assertEqual(pivoted.shape, (2, 2))
        self.assertEqual(pivoted.loc["A", "x"], 1)
        self.assertEqual(pivoted.loc["B", "y"], 4)

    def test_create_joint_dataframe_writes_namespace_files(self):
        """Test joining feature generator outputs into raw namespace TSVs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            feature_dir = os.path.join(temp_dir, "features")
            raw_dir = os.path.join(temp_dir, "raw")
            os.makedirs(feature_dir)
            os.makedirs(raw_dir)

            custom_file = os.path.join(feature_dir, "date_s_Lm_st_L628_p_C03_pos001_tm_24_ch_Syto9_z_21CustomAlgos.txt")
            diff_file = os.path.join(feature_dir, "date_s_Lm_st_L628_p_C03_pos001_tm_24_ch_Syto9_z_21DiffGlobal.txt")
            ignored_file = os.path.join(feature_dir, "date_s_Lm_st_L628_p_C03_pos001_tm_24_ch_Syto9_z_21Other.txt")

            pd.DataFrame({"layer_feature": [1.0, 2.0]}).to_csv(custom_file, sep="\t", index=False)
            pd.DataFrame({"global_feature": [3.0]}).to_csv(diff_file, sep="\t", index=False)
            pd.DataFrame({"ignored": [4.0]}).to_csv(ignored_file, sep="\t", index=False)

            outputs = create_joint_dataframe(feature_dir, raw_dir)

            self.assertEqual(set(outputs), {"CustomAlgos", "DiffGlobal"})
            custom_output = pd.read_csv(os.path.join(raw_dir, "CustomAlgos.tsv"), sep="\t", index_col=0)
            diff_output = pd.read_csv(os.path.join(raw_dir, "DiffGlobal.tsv"), sep="\t", index_col=0)
            self.assertEqual(
                custom_output["sampleName"].unique().tolist(), ["date--s--Lm--st--L628--p--C03--pos001--tm--24--ch--Syto9--z--21"]
            )
            self.assertEqual(len(custom_output), 2)
            self.assertEqual(diff_output["global_feature"].tolist(), [3.0])

    def test_aggregate_raw_features_writes_requested_statistics(self):
        """Test analysis aggregation over raw namespace files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            analysis_dir = os.path.join(temp_dir, "analysis")
            os.makedirs(raw_dir)
            os.makedirs(analysis_dir)

            pd.DataFrame(
                {
                    "sampleName": ["sample_a", "sample_a", "sample_b"],
                    "feature": [1.0, 3.0, 10.0],
                }
            ).to_csv(os.path.join(raw_dir, "CustomAlgos.tsv"), sep="\t", index=False)

            output_files = aggregate_raw_features(raw_dir, analysis_dir, statistics=["mean", "max", "q25"])

            self.assertEqual(len(output_files), 3)
            mean_df = pd.read_csv(os.path.join(analysis_dir, "SUMMARYCustomAlgos.tsv_mean.txt"), sep="\t")
            max_df = pd.read_csv(os.path.join(analysis_dir, "SUMMARYCustomAlgos.tsv_max.txt"), sep="\t")
            q25_df = pd.read_csv(os.path.join(analysis_dir, "SUMMARYCustomAlgos.tsv_q25.txt"), sep="\t")

            self.assertEqual(mean_df.loc[mean_df["sampleName"] == "sample_a", "feature"].iloc[0], 2.0)
            self.assertEqual(max_df.loc[max_df["sampleName"] == "sample_a", "feature"].iloc[0], 3.0)
            self.assertEqual(q25_df.loc[q25_df["sampleName"] == "sample_a", "feature"].iloc[0], 1.5)

    def test_create_final_dataframe_with_labels_preserves_training_format(self):
        """Test final feature datafile creation still includes labels by default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis_dir = os.path.join(temp_dir, "analysis")
            os.makedirs(analysis_dir)
            outfile = os.path.join(temp_dir, "training_features.tsv")

            pd.DataFrame(
                {
                    "sampleName": ["2023--s--Lm--st--L1323--p"],
                    "feature_a": [1.5],
                }
            ).to_csv(os.path.join(analysis_dir, "SUMMARYCustomAlgos.tsv_mean.txt"), sep="\t", index=False)

            result = create_final_dataframe(analysis_dir, outfile)

            self.assertTrue(os.path.exists(outfile))
            self.assertIn("label", result.columns)
            self.assertIn("feature_a-SUMMARYCustomAlgos.tsv_mean.txt", result.columns)
            self.assertEqual(result.loc["2023--s--Lm--st--L1323--p", "label"], "L1323")

    def test_create_final_dataframe_without_labels_for_unknown_samples(self):
        """Test final feature datafile creation for unknown inference images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis_dir = os.path.join(temp_dir, "analysis")
            os.makedirs(analysis_dir)
            outfile = os.path.join(temp_dir, "unknown_features.tsv")

            pd.DataFrame(
                {
                    "sampleName": ["unknown_sample"],
                    "feature_a": [1.5],
                    "feature_b": [2.5],
                }
            ).to_csv(os.path.join(analysis_dir, "SUMMARYCustomAlgos.tsv_mean.txt"), sep="\t", index=False)

            result = create_final_dataframe(analysis_dir, outfile, include_label=False)

            self.assertTrue(os.path.exists(outfile))
            self.assertEqual(result.index.tolist(), ["unknown_sample"])
            self.assertNotIn("label", result.columns)
            self.assertIn("feature_b-SUMMARYCustomAlgos.tsv_mean.txt", result.columns)

    def test_create_final_dataframe_rejects_empty_analysis_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "No aggregated feature files"):
                create_final_dataframe(temp_dir, os.path.join(temp_dir, "data.tsv"))

    def test_string_operations(self):
        """Test string operations used in data processing."""
        # Test string split and join operations
        sample_name = "test_CustomAlgos_file_name.txt"

        # Remove identifier
        cleaned = sample_name.replace("CustomAlgos", "")
        self.assertNotIn("CustomAlgos", cleaned)

        # Remove extension
        no_ext = cleaned.replace(".txt", "")
        self.assertNotIn(".txt", no_ext)

        # Test split and join
        parts = "a_b_c".split("_")
        joined = "--".join(parts)
        self.assertEqual(joined, "a--b--c")


if __name__ == "__main__":
    unittest.main()
