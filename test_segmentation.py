import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, "src")

from segmentation import (
    environment_settings,
    filter_mask,
    generate_mask,
    mask_for_image,
    normalize_settings,
    read_mask,
    read_tiff_stack,
    to_uint8,
)


class TestSegmentationSettings(unittest.TestCase):
    def test_legacy_otsu_with_manual_value_is_a_manual_override(self):
        settings = normalize_settings({"threshold_method": "otsu", "threshold_value": 0.78, "threshold_scale": "raw"})
        self.assertEqual(settings["segmentation_approach"], "manual")
        mask, result = generate_mask(np.array([0, 1, 255], dtype=np.uint8), settings)
        np.testing.assert_array_equal(mask, [False, True, True])
        self.assertEqual(result["threshold_uint8"], 0.78)
        self.assertEqual(result["method"], "manual")

    def test_explicit_automatic_mode_rejects_a_manual_cutoff(self):
        with self.assertRaisesRegex(ValueError, "Manual / tuned"):
            normalize_settings({"segmentation_approach": "automatic", "threshold_value": 100})

    def test_automatic_otsu_keeps_zero_background_out(self):
        image = np.array([0, 0, 0, 200], dtype=np.uint8)
        mask, result = generate_mask(image, normalize_settings({}))
        np.testing.assert_array_equal(mask, [False, False, False, True])
        self.assertEqual(result["threshold_uint8"], 1)

    def test_sensitivity_records_base_and_adjusted_threshold(self):
        image = np.array([0, 40, 100, 200], dtype=np.uint8)
        for method in ("otsu", "robust_background", "multi_otsu"):
            with self.subTest(method=method):
                settings = normalize_settings({"threshold_method": method, "threshold_sensitivity": 0.5})
                _, result = generate_mask(image, settings)
                self.assertAlmostEqual(result["base_threshold_uint8"] * 0.5, result["threshold_uint8"])

    def test_invalid_scales_groups_and_sizes_are_rejected(self):
        for raw in (
            {"threshold_method": "manual", "threshold_value": 256},
            {"feature_groups": []},
            {"minimum_object_volume_um3": -1},
            {"connectivity_3d": 26.1},
            {"threshold_method": "bem"},
            {"threshold_from": "brightest"},
            {"substratum_slices": 1.5},
            {"substratum_slices": 0},
            {"threshold_method": "robust_background", "threshold_from": "substratum"},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_settings(raw)

    def test_substratum_otsu_uses_first_n_slices_for_full_mask(self):
        image = np.array([[[0, 10], [10, 0]], [[80, 200], [200, 80]], [[80, 200], [200, 80]]], dtype=np.uint8)
        whole, whole_result = generate_mask(image, normalize_settings({}))
        for count in (1, 2, 20):
            settings = normalize_settings({"threshold_from": "substratum", "substratum_slices": count})
            with patch("segmentation.otsu_threshold", wraps=__import__("segmentation").otsu_threshold) as otsu:
                mask, result = generate_mask(image, settings)
            np.testing.assert_array_equal(otsu.call_args.args[0], image[:count])
            np.testing.assert_array_equal(mask, image >= result["threshold_uint8"])
        self.assertGreater(whole_result["threshold_uint8"], 1)
        bottom, _ = generate_mask(image, normalize_settings({"threshold_from": "substratum"}))
        self.assertGreater(bottom.sum(), whole.sum())

    def test_fractional_size_accepts_comma_and_dot(self):
        for text in ("0.015", "0,015"):
            settings = normalize_settings({"minimum_object_area_um2": text})
            mask = np.array([[[True, False, True], [False, False, True]]])
            filtered, _ = filter_mask(mask, (0.1, 0.1, 99), settings)
            self.assertEqual(filtered.sum(), 2)

    def test_removed_preprocessing_is_absent_from_settings(self):
        settings = normalize_settings({})
        self.assertNotIn("rolling_ball_radius_px", settings)
        self.assertNotIn("rolling_ball_radius_px", environment_settings({"IMAGINE_ROLLING_BALL_RADIUS_PX": "50"}))

    def test_three_class_chooses_one_comstat2_cutoff(self):
        image = np.array([0, 50, 100, 200], dtype=np.uint8)
        with patch("segmentation.multi_otsu_thresholds", return_value=(50, 100)):
            for dim_class, expected in (("foreground", [False, True, True, True]), ("background", [False, False, True, True])):
                mask, result = generate_mask(
                    image, normalize_settings({"threshold_method": "multi_otsu", "dim_class_assignment": dim_class})
                )
                np.testing.assert_array_equal(mask, expected)
                np.testing.assert_array_equal(result["class_labels"] > (0 if dim_class == "foreground" else 1), expected)


class TestThresholdConversion(unittest.TestCase):
    def test_12bit_range_rounding_clipping_and_identity(self):
        raw = np.array([0, 16, 2048, 4095, 5000], dtype=np.uint16)
        converted, metadata = to_uint8(raw, conversion_max=4095)
        np.testing.assert_array_equal(converted, [0, 1, 128, 255, 255])
        self.assertEqual(metadata["clipped_fraction"], 0.2)
        self.assertEqual(metadata["upper"], 4095)
        np.testing.assert_array_equal(to_uint8(converted)[0], converted)

    def test_conversion_uses_metadata_or_whole_stack_observed_range(self):
        image = np.array([[[1024]], [[2048]]], dtype=np.uint16)
        np.testing.assert_array_equal(to_uint8(image, significant_bits=12)[0].ravel(), [64, 128])
        converted, metadata = to_uint8(image)
        np.testing.assert_array_equal(converted.ravel(), [0, 255])
        self.assertEqual(metadata["lower"], 1024)
        self.assertEqual(metadata["upper"], 2048)
        self.assertEqual(to_uint8(image, range_mode="storage")[1]["upper"], 65535)

    def test_ome_significant_bits_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack.ome.tif"
            tifffile.imwrite(
                path,
                np.zeros((2, 3, 4), dtype=np.uint16),
                ome=True,
                photometric="minisblack",
                metadata={"axes": "ZYX", "SignificantBits": 12},
            )
            stack, bits = read_tiff_stack(path)
            self.assertEqual(bits, 12)
            self.assertEqual(to_uint8(stack, significant_bits=bits)[1]["upper"], 4095)


class TestImportedMasks(unittest.TestCase):
    def test_touching_labels_remain_distinct(self):
        labels = np.array([[[1, 2], [0, 0]], [[0, 0], [0, 0]]])
        mask, objects = filter_mask(labels > 0, (1, 1, 1), normalize_settings({}), labels)
        self.assertEqual(len(np.unique(objects[mask])), 2)
        _, binary_objects = filter_mask(mask, (1, 1, 1), normalize_settings({}))
        self.assertEqual(len(np.unique(binary_objects[mask])), 1)

    def test_size_filter_and_connectivity_use_physical_units(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = mask[1, 1, 1] = True
        for connectivity, expected in ((6, 0), (26, 2)):
            filtered, _ = filter_mask(
                mask, (2, 2, 0.5), normalize_settings({"connectivity_3d": connectivity, "minimum_object_volume_um3": 3})
            )
            self.assertEqual(filtered.sum(), expected)

    def test_pairing_dimensions_and_negative_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample_mask.tif"
            tifffile.imwrite(path, np.array([[0, 1], [1, 0]], dtype=np.uint8))
            self.assertEqual(mask_for_image("sample.tif", root), path)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                read_mask(path, (2, 2, 2))
            tifffile.imwrite(root / "sample.tiff", np.zeros((2, 2), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                mask_for_image("sample.tif", root)
            tifffile.imwrite(path, np.array([[-1, 1]], dtype=np.int16))
            with self.assertRaisesRegex(ValueError, "non-negative"):
                read_mask(path, (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
