"""Tests for the preflight reports used by the GUI and CLI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.input_validation import replication_group, validate_feature_table, validate_image_directory


class TestInputValidation(unittest.TestCase):
    def test_image_report_counts_labels_by_date(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in (
                "04072023_s_Lm_st_L1323_p_C06_pos004_tm_24_ch_Syto9_z_21.tif",
                "04072023_s_Lm_st_L1323_p_C08_pos004_tm_24_ch_Syto9_z_21.tif",
                "16052023_s_Lm_st_L394_p_D03_pos001_tm_24_ch_Syto9_z_21.tif",
                "bad-name.tif",
            ):
                (root / name).touch()

            report = validate_image_directory(root)

            self.assertFalse(report["ok"])
            self.assertEqual(report["images_per_label"], {"L1323": 2, "L394": 1})
            self.assertEqual(report["images_per_label_per_date"]["04072023"]["L1323"], 2)
            self.assertEqual(report["invalid_filenames"], ["bad-name.tif"])

    def test_feature_report_stops_on_empty_and_unparsed_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.tsv"
            pd.DataFrame(
                {
                    "sampleName": ["a", "b"],
                    "label": ["A", "B"],
                    "CustomAlgos_mean": [1.0, None],
                    "external_score": ["not-a-number", "2"],
                }
            ).to_csv(path, sep="\t", index=False)

            report = validate_feature_table(path)

            self.assertFalse(report["ok"])
            self.assertEqual(report["microics_features"], 1)
            self.assertEqual(report["external_features"], 1)
            self.assertEqual(report["unparsed_feature_names"], ["external_score"])
            self.assertGreater(report["nan_cells"], 0)

    def test_generated_microics_feature_families_are_not_external(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.tsv"
            pd.DataFrame(
                {
                    "sampleName": ["a"],
                    "label": ["A"],
                    "RoughnessThreshold=0.1-SUMMARYCustomAlgos.tsv_mean.txt": [1.0],
                    "ThicknessThreshold=0.1-SUMMARYCustomAlgos.tsv_mean.txt": [2.0],
                    "eigen-SUMMARYDiffGlobal.tsv_mean.txt": [3.0],
                    "globalMean-SUMMARYDiffGlobal.tsv_mean.txt": [4.0],
                    "mdiffs-SUMMARYDiffGlobal.tsv_mean.txt": [5.0],
                    "external_score": [6.0],
                }
            ).to_csv(path, sep="\t", index=False)

            report = validate_feature_table(path)

            self.assertEqual(report["microics_features"], 5)
            self.assertEqual(report["external_features"], 1)
            self.assertTrue(report["ok"])

    def test_replication_group_is_explicit(self):
        sample = "04072023--s--Lm--st--L1323--p--C06--pos004--tm--24--ch--Syto9--z--21"
        self.assertEqual(replication_group(sample, "date"), "04072023")
        self.assertEqual(replication_group(sample, "plate"), "L1323")
        self.assertEqual(replication_group(sample, "well"), "C06")
        self.assertEqual(replication_group(sample, "position"), "004")

    def test_feature_report_requires_sample_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.tsv"
            pd.DataFrame({"label": ["A"], "external_score": [1.0]}).to_csv(path, sep="\t", index=False)

            report = validate_feature_table(path)

            self.assertFalse(report["ok"])
            self.assertIn("Required sampleName column is missing", report["errors"])

    def test_complete_feature_table_rejects_duplicate_samples_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.tsv"
            pd.DataFrame(
                {
                    "sampleName": ["same", "same"],
                    "label": ["A", "missing"],
                    "external_score": [1.0, 2.0],
                }
            ).to_csv(path, sep="\t", index=False)

            report = validate_feature_table(path)

            self.assertFalse(report["ok"])
            self.assertIn("2 rows have duplicate sampleName values", report["errors"])
            self.assertIn("1 missing/unknown label values found", report["errors"])


if __name__ == "__main__":
    unittest.main()
