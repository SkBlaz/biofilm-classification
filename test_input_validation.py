"""Tests for the preflight reports used by the GUI and CLI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.input_validation import (
    assess_replication_units,
    merge_feature_tables,
    read_feature_table,
    replication_group,
    validate_feature_table,
    validate_image_directory,
)


class TestInputValidation(unittest.TestCase):
    def test_duplicate_headers_fail_before_pandas_renames_them(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".xlsx", ".csv", ".tsv", ".txt"):
                for column in ("sampleName", "assay", "NA"):
                    with self.subTest(suffix=suffix, column=column):
                        path = Path(directory) / f"features{suffix}"
                        columns = ["sampleName", "label", column, column]
                        frame = pd.DataFrame([["sample", "A", 1, 2]], columns=columns)
                        if suffix == ".xlsx":
                            frame.to_excel(path, index=False)
                        else:
                            frame.to_csv(path, sep="," if suffix == ".csv" else "\t", index=False)
                        report = validate_feature_table(path)
                        self.assertFalse(report["ok"])
                        self.assertIn("Duplicate column names", report["errors"][0])

    def test_excel_numeric_headers_match_delimited_names(self):
        with tempfile.TemporaryDirectory() as directory:
            primary = pd.DataFrame({"sampleName": ["a"], "label": ["A"], "1": [1]})
            additional = pd.DataFrame({"sampleName": ["a"], 1: [2]})
            path = Path(directory) / "measurements.xlsx"
            additional.to_excel(path, index=False)
            with self.assertRaisesRegex(ValueError, "overlap: 1"):
                merge_feature_tables(primary, read_feature_table(path))

    def test_join_preserves_exact_names_and_agreeing_labels(self):
        primary = pd.DataFrame({"sampleName": ["001", "1"], "label": ["a", "b"], "feature": [1, 2]})
        additional = pd.DataFrame({"sampleName": ["1", "001"], "label": ["b", "a"], "assay": [20, 10]})
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".xlsx", ".csv", ".tsv", ".txt"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"measurements{suffix}"
                    if suffix == ".xlsx":
                        additional.to_excel(path, index=False)
                    else:
                        additional.to_csv(path, sep="," if suffix == ".csv" else "\t", index=False)
                    merged = merge_feature_tables(primary, read_feature_table(path))
                    self.assertEqual(merged["sampleName"].tolist(), ["001", "1"])
                    self.assertEqual(merged["assay"].tolist(), [10, 20])
                    self.assertEqual(merged["label"].tolist(), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "Primary.*duplicate"):
            merge_feature_tables(pd.concat([primary, primary.iloc[:1]]), additional)

    def test_excel_csv_and_tsv_have_the_same_feature_contract(self):
        frame = pd.DataFrame({"sampleName": ["image1", "image2"], "label": ["a", "b"], "measured_area": [2.5, 7.0]})
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".xlsx", ".csv", ".tsv"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"features{suffix}"
                    if suffix == ".xlsx":
                        frame.to_excel(path, index=False)
                    else:
                        frame.to_csv(path, sep="," if suffix == ".csv" else "\t", index=False)
                    pd.testing.assert_frame_equal(read_feature_table(path), frame)
                    self.assertTrue(validate_feature_table(path)["ok"])

    def test_boolean_feature_values_are_read_as_numeric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.tsv"
            pd.DataFrame(
                {
                    "sampleName": ["a", "b"],
                    "label": ["True", "False"],
                    "NoBiomassDetected-SUMMARYImageFeaturesCustomAlgos": ["T", "F"],
                }
            ).to_csv(path, sep="\t", index=False)

            frame = read_feature_table(path)

            self.assertEqual(
                frame["NoBiomassDetected-SUMMARYImageFeaturesCustomAlgos"].tolist(),
                [1.0, 0.0],
            )
            self.assertEqual(frame["label"].tolist(), [True, False])
            self.assertTrue(validate_feature_table(path, require_label=False)["ok"])

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

    def test_unlabelled_images_accept_arbitrary_names_and_report_them(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            filenames = [
                "plain-image.tif",
                "05032024_s_Lm_st_L1764_p_E02_pos001_tm_24_ch_Syto9_z_21_treat_milkcoat.tif",
            ]
            for filename in filenames:
                (root / filename).touch()

            report = validate_image_directory(root, labelled=False)

            self.assertTrue(report["ok"])
            self.assertEqual(report["valid_images"], 2)
            self.assertEqual(report["invalid_filenames"], [])
            self.assertEqual(report["image_filenames"], sorted(filenames))
            self.assertIn("filename convention was not enforced", report["message"])

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

    def test_generated_nan_features_are_reported_and_imputed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy-generated.tsv"
            pd.DataFrame(
                {
                    "sampleName": ["a", "b"],
                    "label": ["A", "B"],
                    "CustomAlgos_mean": [1.0, None],
                    "eigen-SUMMARYDiffGlobal.tsv_var.txt": [None, None],
                }
            ).to_csv(path, sep="\t", index=False)

            report = validate_feature_table(path)

            self.assertTrue(report["ok"])
            self.assertEqual(report["empty_cells"], 3)
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("will impute them", report["warnings"][0])

    def test_generated_infinite_features_are_reported_and_imputed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy-generated.tsv"
            pd.DataFrame(
                {
                    "sampleName": ["a", "b"],
                    "label": ["A", "B"],
                    "external_score": [float("inf"), float("-inf")],
                }
            ).to_csv(path, sep="\t", index=False)

            report = validate_feature_table(path)

            self.assertTrue(report["ok"])
            self.assertEqual(report["infinite_cells"], 2)
            self.assertIn("infinite feature values", report["warnings"][0])

    def test_replication_group_is_explicit(self):
        sample = "04072023--s--Lm--st--L1323--p--C06--pos004--tm--24--ch--Syto9--z--21"
        self.assertEqual(replication_group(sample, "date"), "04072023")
        self.assertEqual(replication_group(sample, "well"), "04072023::C06")
        self.assertEqual(replication_group(sample, "position"), "04072023::C06::004")
        with self.assertRaisesRegex(ValueError, "date, well, or position"):
            replication_group(sample, "plate")

    def test_replication_groups_do_not_merge_leaf_values_across_parents(self):
        first = "04072023--s--Lm--st--L1--p--C06--pos004--tm--24--ch--Syto9--z--21"
        other_date = "05072023--s--Lm--st--L1--p--C06--pos004--tm--24--ch--Syto9--z--21"
        other_well = "04072023--s--Lm--st--L1--p--D06--pos004--tm--24--ch--Syto9--z--21"

        self.assertNotEqual(replication_group(first, "well"), replication_group(other_date, "well"))
        self.assertNotEqual(replication_group(first, "position"), replication_group(other_well, "position"))

    def test_replication_assessment_blocks_one_date_but_allows_balanced_wells(self):
        names = []
        labels = []
        for label, prefix in (("L1", "C"), ("L2", "D")):
            for number in range(1, 6):
                names.append(f"01012026--s--Lm--st--{label}--p--{prefix}{number:02d}--pos001--tm--24--ch--Syto9--z--21")
                labels.append(label)

        report = assess_replication_units(names, labels, selected_unit="date", require_nested_cv=True)

        self.assertFalse(report["ok"])
        self.assertEqual(report["units"]["date"]["status"], "unavailable")
        self.assertTrue(report["units"]["well"]["ok"])
        self.assertEqual(report["units"]["well"]["groups_per_class"], {"L1": 5, "L2": 5})
        self.assertIn("Date cannot be used", report["errors"][0])

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
