#!/usr/bin/env python3
"""
Unit tests for feature_generator.py utility functions.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

# Add src directory to path
sys.path.insert(0, "src")

from feature_generator import (
    calculate_spatial_spreading,
    fractal_dimension,
    get_cell_count,
    get_homogenity,
    get_transition_matrix,
    read_voxel_dimensions,
    rgb2gray,
    segment,
)


class TestRgb2Gray(unittest.TestCase):
    """Test RGB to grayscale conversion."""

    def test_basic_conversion(self):
        """Test basic RGB to gray conversion."""
        # Create a simple RGB image (3x3 pixels)
        rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
                [[100, 150, 200], [50, 75, 100], [200, 100, 50]],
            ]
        )

        gray = rgb2gray(rgb)

        # Check shape is correct (should be 2D)
        self.assertEqual(gray.shape, (3, 3))

        # Check specific values using the formula: 0.2989*R + 0.5870*G + 0.1140*B
        # For pure red (255, 0, 0)
        expected_red = 0.2989 * 255
        self.assertAlmostEqual(gray[0, 0], expected_red, places=1)

        # For pure green (0, 255, 0)
        expected_green = 0.5870 * 255
        self.assertAlmostEqual(gray[0, 1], expected_green, places=1)

        # For pure blue (0, 0, 255)
        expected_blue = 0.1140 * 255
        self.assertAlmostEqual(gray[0, 2], expected_blue, places=1)

        # For white (255, 255, 255)
        self.assertAlmostEqual(gray[1, 0], 255.0, places=1)

        # For black (0, 0, 0)
        self.assertAlmostEqual(gray[1, 1], 0.0, places=1)

    def test_uniform_color(self):
        """Test conversion of uniform color image."""
        rgb = np.ones((5, 5, 3)) * 100
        gray = rgb2gray(rgb)

        # All pixels should have the same gray value (0.2989 + 0.5870 + 0.1140 = 1.0)
        # So gray value should be 100 * 1.0 = 100
        expected_value = 100 * (0.2989 + 0.5870 + 0.1140)
        self.assertTrue(np.allclose(gray, expected_value, rtol=0.01))


class TestVoxelDimensions(unittest.TestCase):
    def test_defaults_match_the_legacy_physical_scale(self):
        self.assertEqual(read_voxel_dimensions({}), (0.13, 0.13, 0.5))

    def test_custom_dimensions_are_read_per_axis(self):
        dimensions = read_voxel_dimensions(
            {
                "IMAGINE_VOXEL_SIZE_X": "0.21",
                "IMAGINE_VOXEL_SIZE_Y": "0.22",
                "IMAGINE_VOXEL_SIZE_Z": "0.8",
            }
        )

        self.assertEqual(dimensions, (0.21, 0.22, 0.8))

    def test_invalid_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "IMAGINE_VOXEL_SIZE_Y must be a positive number"):
            read_voxel_dimensions({"IMAGINE_VOXEL_SIZE_Y": "nan"})


class TestCalculateSpatialSpreading(unittest.TestCase):
    """Test spatial spreading calculation."""

    def test_single_point(self):
        """Test with a single non-zero point."""
        image_stack = np.zeros((3, 5, 5))
        image_stack[1, 2, 2] = 1

        Sxy, Sz, Sxyz = calculate_spatial_spreading(image_stack)

        # Single point should have zero variance
        self.assertEqual(Sxy, 0.0)
        self.assertEqual(Sz, 0.0)
        self.assertEqual(Sxyz, 0.0)

    def test_vertical_line(self):
        """Test with a vertical line of points."""
        image_stack = np.zeros((5, 3, 3))
        # Create a vertical line along z-axis
        for z in range(5):
            image_stack[z, 1, 1] = 1

        Sxy, Sz, Sxyz = calculate_spatial_spreading(image_stack)

        # z should have variance, x and y should not
        self.assertEqual(Sxy, 0.0)
        self.assertGreater(Sz, 0.0)
        self.assertGreater(Sxyz, 0.0)

    def test_spreading_uses_mask_voxel_calibration_per_axis(self):
        mask = np.zeros((1, 1, 3), dtype=bool)
        mask[0, 0, 0] = True
        mask[0, 0, 2] = True
        horizontal, vertical, total = calculate_spatial_spreading(mask, dimensions=(2.0, 3.0, 0.5))
        self.assertEqual(horizontal, 2.0)
        self.assertEqual(vertical, 0.0)
        self.assertEqual(total, 2.0)

    def test_horizontal_spread(self):
        """Test with horizontal spread."""
        image_stack = np.zeros((3, 5, 5))
        # Create horizontal spread on middle layer
        image_stack[1, :, :] = 1

        Sxy, Sz, Sxyz = calculate_spatial_spreading(image_stack)

        # x and y should have variance
        self.assertGreater(Sxy, 0.0)
        # z should have no variance (all on same layer)
        self.assertEqual(Sz, 0.0)
        self.assertGreater(Sxyz, 0.0)

    def test_empty_image(self):
        """Test with an empty (all zeros) image."""
        image_stack = np.zeros((3, 3, 3))

        # This should handle empty image gracefully
        # Empty arrays result in nan variance
        Sxy, Sz, Sxyz = calculate_spatial_spreading(image_stack)

        # With no non-zero pixels, we expect nan values
        self.assertTrue(np.isnan(Sxy))
        self.assertTrue(np.isnan(Sz))
        self.assertTrue(np.isnan(Sxyz))

    def test_anisotropic_spacing_is_applied_per_axis(self):
        image_stack = np.zeros((1, 1, 2))
        image_stack[0, 0, :] = 1
        horizontal, vertical, total = calculate_spatial_spreading(image_stack, (2.0, 3.0, 0.5))
        self.assertEqual(horizontal, 1.0)
        self.assertEqual(vertical, 0.0)
        self.assertEqual(total, 1.0)


class TestSegmentationIntegration(unittest.TestCase):
    def test_rolling_ball_preserves_raw_features_and_exact_qc(self):
        stack = np.full((2, 16, 16), 20, dtype=np.uint8)
        stack[:, 7:9, 7:9] = 120
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "background.tif"
            tifffile.imwrite(path, stack, photometric="minisblack")
            qc = root / "segmentation"
            qc.mkdir()
            for suffix in ("_object_labels.tif", "_segmentation_qc.png"):
                (qc / f"background{suffix}").touch()
            record = segment(str(path), directory, manual_threshold=50, rolling_ball_radius_px=3, feature_groups=["intensity", "geometry"])
            converted = tifffile.imread(qc / record["thresholding_image"])
            mask = tifffile.imread(qc / record["mask"]) > 0
            self.assertEqual(converted[0, 0, 0], 0)
            self.assertTrue(record["rolling_ball_applied"])
            np.testing.assert_array_equal(mask, converted >= 50)
            values = pd.read_csv(root / "backgroundCustomAlgos.txt", sep="\t")
            np.testing.assert_allclose(values["RawIntensityMean"], stack.mean(axis=(1, 2)))
            self.assertIn("FractalDimension", values)
            self.assertIn("FractalGrayscaleBoxDimension", values)
            self.assertNotIn("FractalDimension", record["scalar_feature_names"])
            for suffix in ("_object_labels.tif", "_segmentation_qc.png"):
                self.assertFalse((qc / f"background{suffix}").exists())

    def test_fractal_dimension_matches_legacy_box_counts(self):
        image = np.random.default_rng(5).integers(0, 256, (32, 32))
        binary = image < np.median(image)
        counts = []
        for size in (32, 16, 8, 4):
            boxes = [binary[y : y + size, x : x + size] for y in range(0, 32, size) for x in range(0, 32, size)]
            counts.append(sum(0 < box.sum() < size * size for box in boxes))
        expected = -np.polyfit(np.log([32, 16, 8, 4]), np.log(counts), 1)[0]
        self.assertAlmostEqual(fractal_dimension(image), expected)
        for image in (np.ones((16, 16)), np.zeros((2, 2)), np.eye(16)):
            self.assertTrue(np.isnan(fractal_dimension(image)))

    def test_raw_manual_threshold_and_qc_metadata(self):
        stack = np.array(
            [
                [[0, 100], [200, 0]],
                [[0, 100], [200, 0]],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.tif"
            tifffile.imwrite(image_path, stack, photometric="minisblack", metadata={"axes": "ZYX"})
            segment(
                str(image_path),
                directory,
                threshold_mode="comstat2",
                threshold_method="manual",
                threshold_scale="uint8",
                manual_threshold=100,
            )
            record_path = Path(directory) / "segmentation" / "sample_segmentation.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["selected_threshold_value"], 100)
            self.assertEqual(record["selected_threshold_units"], "uint8")
            self.assertEqual(record["threshold_boundary_mode"], "comstat2")
            self.assertFalse(record["no_biomass_detected"])
            self.assertFalse((Path(directory) / "segmentation" / "sample_segmentation_qc.png").exists())
            self.assertFalse((Path(directory) / "segmentation" / "sample_object_labels.tif").exists())

    def test_manual_otsu_override_controls_mask_and_all_slice_qc(self):
        stack = np.array([[[0, 10], [100, 200]]] * 3, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "override.tif"
            tifffile.imwrite(path, stack, photometric="minisblack")
            record = segment(
                str(path), directory, threshold_method="otsu", manual_threshold=0.78, threshold_scale="stack_normalized", qc_only=True
            )
            expected = stack >= 0.78 * 255
            np.testing.assert_array_equal(tifffile.imread(Path(directory) / "segmentation" / record["mask"]) > 0, expected)
            self.assertEqual(record["method"], "manual")
            self.assertEqual(record["submitted_threshold_value"], 0.78)
            self.assertAlmostEqual(record["threshold_uint8"], 198.9)
            qc = Path(directory) / "segmentation"
            overlay = tifffile.imread(qc / record["overlay"])
            self.assertEqual(overlay.shape, (*stack.shape, 3))
            self.assertFalse(list(qc.glob("*_mask_*.png")))
            self.assertFalse(list(qc.glob("*_overlay_*.png")))
            self.assertTrue((qc / record["object_labels_rgb"]).is_file())
            self.assertFalse(list(Path(directory).glob("*CustomAlgos.txt")))

    def test_intensity_only_skips_geometry_and_serial_calculation(self):
        stack = np.array([[[0, 10], [100, 200]]] * 2, dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intensity.tif"
            tifffile.imwrite(path, stack, photometric="minisblack")
            with (
                patch("feature_generator.comstat_metrics", side_effect=AssertionError("unselected geometry")),
                patch("feature_generator.get_cell_count", side_effect=AssertionError("unselected serial features")),
            ):
                segment(str(path), directory, feature_groups=["intensity"])
            table = pd.read_csv(Path(directory) / "intensityCustomAlgos.txt", sep="\t", index_col=0)
            self.assertEqual(table["RawIntensityMax"].iloc[0], 200)
            self.assertNotIn("ObjectCount3D", table)
            self.assertNotIn("Homogeneity", table)

    def test_imported_labels_survive_and_size_filter_updates_qc(self):
        stack = np.zeros((3, 3, 4), dtype=np.uint8)
        labels = np.zeros_like(stack, dtype=np.uint16)
        labels[1, 1, 1:3] = [9, 25]
        labels[2, 1, 2] = 25
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "masks"
            masks.mkdir()
            path = root / "labelled.tif"
            tifffile.imwrite(path, stack, photometric="minisblack")
            tifffile.imwrite(masks / "labelled_mask.tif", labels, photometric="minisblack")
            with (
                patch("feature_generator.voxel_size_x", 1),
                patch("feature_generator.voxel_size_y", 1),
                patch("feature_generator.voxel_size_z", 1),
            ):
                record = segment(
                    str(path),
                    directory,
                    segmentation_approach="import",
                    masks_dir=str(masks),
                    feature_groups=["objects"],
                    minimum_object_volume_um3=2,
                )
            self.assertIsNone(record["threshold_uint8"])
            self.assertIsNone(record["threshold_curve"])
            self.assertEqual(record["foreground_fraction"], 2 / stack.size)
            table = pd.read_csv(root / "labelledCustomAlgos.txt", sep="\t", index_col=0)
            self.assertEqual(table["ObjectCount3D"].iloc[0], 1)
            self.assertEqual(table["ObjectVolume3DMean_um3"].iloc[0], 2)

    def test_2d_package_has_objects_pores_and_texture_without_3d(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1:4, 1:4] = 200
        image[2, 2] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plane.tif"
            tifffile.imwrite(path, image)
            record = segment(str(path), directory, manual_threshold=100, image_dimension="2d")
            table = pd.read_csv(Path(directory) / "planeCustomAlgos.txt", sep="\t", index_col=0)
            self.assertIn("SubstratumObjectCount", table)
            self.assertIn("SubstratumInternalPoreAreaMean_um2", table)
            self.assertIn("GLCMContrast_d1_mean", table)
            self.assertNotIn("ObjectCount3D", table)
            self.assertNotIn("BiomassVolume_um3", table)
            self.assertEqual(record["image_dimension"], "2d")

    def test_imported_2d_mask_produces_scalar_training_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "feature_generator"
            masks = root / "masks"
            for folder in (generated, masks, root / "raw", root / "analysis"):
                folder.mkdir()
            image = np.zeros((12, 12), dtype=np.uint8)
            mask = np.zeros_like(image, dtype=np.uint16)
            mask[2:10, 2:10] = 50000
            mask[0, 3] = 9
            stem = "04072023_s_Lm_st_L1323_p_C08_pos004_tm_24_ch_Syto9_z_21"
            path = root / f"{stem}.tif"
            tifffile.imwrite(path, image)
            tifffile.imwrite(masks / f"{stem}_mask.tif", mask)
            record = segment(str(path), str(generated), segmentation_approach="import", masks_dir=str(masks), image_dimension="2d")
            self.assertAlmostEqual(record["foreground_touching_image_border_fraction"], 1 / 65)
            rgb = tifffile.imread(generated / "segmentation" / record["object_labels_rgb"])
            self.assertTrue(rgb[0, 5, 5].any())
            self.assertFalse(np.array_equal(rgb[0, 5, 5], rgb[0, 0, 3]))
            for script, args in (
                ("create_joint_df.py", [generated, root / "raw"]),
                ("analysis.py", [root / "raw", root / "analysis"]),
                ("create_final_df_from_results.py", [root / "analysis", root / "datafile.tsv"]),
            ):
                result = subprocess.run([sys.executable, str(Path("src") / script), *map(str, args)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            table = pd.read_csv(root / "datafile.tsv", sep="\t")
            self.assertEqual(len(table), 1)
            self.assertEqual(table["label"].iloc[0], "L1323")
            count_columns = [name for name in table if name.startswith("SubstratumObjectCount-")]
            self.assertEqual(len(count_columns), 1)
            self.assertEqual(table[count_columns[0]].iloc[0], 2)
            self.assertTrue(any(name.startswith("FractalDimension-") for name in table))
            self.assertTrue(any(name.startswith("FractalGrayscaleBoxDimension-") for name in table))
            self.assertTrue((root / "feature_generation_settings.json").is_file())

    def test_3d_scalar_features_export_once_and_heights_vary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "feature_generator"
            masks = root / "masks"
            for folder in (generated, masks, root / "raw", root / "analysis"):
                folder.mkdir()
            for label, depth in (("L1323", 2), ("L394", 4)):
                stem = f"04072023_s_Lm_st_{label}_p_C08_pos004_tm_24_ch_Syto9_z_21"
                stack = np.zeros((6, 4, 4), dtype=np.uint8)
                mask = np.zeros_like(stack)
                mask[:depth, 1:3, 1:3] = 255
                path = root / f"{stem}.tif"
                tifffile.imwrite(path, stack, photometric="minisblack")
                tifffile.imwrite(masks / f"{stem}_mask.tif", mask, photometric="minisblack")
                record = segment(
                    str(path),
                    str(generated),
                    segmentation_approach="import",
                    masks_dir=str(masks),
                    feature_groups=["basic", "objects"],
                    rolling_ball_radius_px=12.5,
                    threshold_from="substratum",
                    substratum_slices=2,
                )
                self.assertFalse(record["rolling_ball_applied"])
            for script, args in (
                ("create_joint_df.py", [generated, root / "raw"]),
                ("analysis.py", [root / "raw", root / "analysis"]),
                ("create_final_df_from_results.py", [root / "analysis", root / "datafile.tsv"]),
            ):
                result = subprocess.run([sys.executable, str(Path("src") / script), *map(str, args)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            table = pd.read_csv(root / "datafile.tsv", sep="\t")
            self.assertEqual(len(table), 2)
            settings = json.loads((root / "feature_generation_settings.json").read_text())
            self.assertEqual(settings["feature_schema_version"], 3)
            self.assertEqual(settings["rolling_ball_radius_px"], 12.5)
            self.assertEqual(settings["threshold_from"], "substratum")
            self.assertEqual(settings["substratum_slices"], 2)
            for feature in (
                "BiofilmExtentHeight_um",
                "BiomassCenterOfMassHeight_um",
                "BiomassVolume_um3",
                "FilledThicknessMean_um",
                "Roughness_RaStar",
                "SpreadingHorizontal",
                "ObjectCount3D",
                "ObjectDensity3D_per_um3",
                "ObjectNearestNeighbor3DIQR_um",
                "ObjectVolume3DIQR_um3",
                "MeanThickness_COMSTAT_um",
                "MeanThickness_BiomassColumns_um",
                "MaxCompactedColumnThickness_um",
            ):
                columns = [name for name in table if name.startswith(f"{feature}-")]
                self.assertEqual(len(columns), 1, feature)
                if feature.startswith(("MeanThickness", "MaxCompacted")):
                    self.assertEqual(table[columns[0]].nunique(), 2, feature)

    def test_advanced_package_keeps_maps_out_of_numeric_ml_table(self):
        stack = np.zeros((3, 5, 5), dtype=np.uint8)
        stack[:, 1:4, 1:4] = 200
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.tif"
            tifffile.imwrite(path, stack, photometric="minisblack")
            segment(str(path), directory, feature_groups=["advanced"])
            table = pd.read_csv(Path(directory) / "geometryCustomAlgos.txt", sep="\t", index_col=0)
            self.assertNotIn("LocalBiomassDensityMap", table)
            self.assertIn("LocalBiomassDensityMean", table)
            self.assertIn("LocalThicknessMean_um", table)
            self.assertTrue(all(pd.api.types.is_numeric_dtype(table[column]) for column in table))


class TestGetCellCount(unittest.TestCase):
    """Test cell counting functionality."""

    def test_single_component(self):
        """Test with a single connected component."""
        # Create a simple 3x3 matrix with one component
        intensity_matrix = np.array([[100, 100, 0], [100, 100, 0], [0, 0, 0]])

        ncomponents, labeled = get_cell_count(intensity_matrix, threshold=0.3, visualizations=False)

        # Should detect 2 components: background (0) and the object (1)
        # Background is counted as a component
        self.assertGreaterEqual(ncomponents, 1)
        self.assertEqual(labeled.shape, intensity_matrix.shape)

    def test_multiple_components(self):
        """Test with multiple separated components."""
        intensity_matrix = np.array([[100, 0, 100], [0, 0, 0], [100, 0, 100]])

        ncomponents, labeled = get_cell_count(intensity_matrix, threshold=0.3, visualizations=False)

        # Should detect background (0) plus 4 separate objects
        self.assertGreaterEqual(ncomponents, 1)
        self.assertEqual(labeled.shape, intensity_matrix.shape)

    def test_uniform_high_intensity(self):
        """Test with uniform high intensity (all cells)."""
        intensity_matrix = np.ones((5, 5)) * 100

        ncomponents, labeled = get_cell_count(intensity_matrix, threshold=0.3, visualizations=False)

        # Should detect 1 large component
        self.assertGreaterEqual(ncomponents, 1)
        self.assertEqual(labeled.shape, intensity_matrix.shape)

    def test_empty_matrix(self):
        """Test with all zeros (no cells)."""
        intensity_matrix = np.zeros((5, 5))

        ncomponents, labeled = get_cell_count(intensity_matrix, threshold=0.3, visualizations=False)

        # Should detect only background or handle gracefully
        self.assertGreaterEqual(ncomponents, 0)
        self.assertEqual(labeled.shape, intensity_matrix.shape)


class TestTransitionMatrix(unittest.TestCase):
    """Test transition matrix calculation."""

    def test_uniform_image(self):
        """Test with uniform intensity image."""
        raw_image = np.ones((3, 3, 3)) * 100

        tm = get_transition_matrix(raw_image, n_bins=10)

        # Should be a square matrix
        self.assertEqual(tm.shape[0], tm.shape[1])
        self.assertEqual(tm.shape[0], 10)

        # Should sum to 1 (probability distribution)
        self.assertAlmostEqual(np.sum(tm), 1.0, places=5)

    def test_binary_image(self):
        """Test with binary (0 and 1) image."""
        raw_image = np.random.choice([0, 100], size=(4, 4, 4))

        tm = get_transition_matrix(raw_image, n_bins=10)

        # Should be a square matrix
        self.assertEqual(tm.shape, (10, 10))

        # Should sum to 1
        self.assertAlmostEqual(np.sum(tm), 1.0, places=5)

        # All values should be non-negative
        self.assertTrue(np.all(tm >= 0))

    def test_custom_bins(self):
        """Test with custom number of bins."""
        raw_image = np.random.randint(0, 256, size=(3, 3, 3))

        for n_bins in [5, 10, 20]:
            tm = get_transition_matrix(raw_image, n_bins=n_bins)
            self.assertEqual(tm.shape, (n_bins, n_bins))
            self.assertAlmostEqual(np.sum(tm), 1.0, places=5)


class TestHomogenity(unittest.TestCase):
    """Test homogenity calculation."""

    def test_uniform_image(self):
        """Test homogenity of uniform image (should be high)."""
        raw_image = np.ones((5, 5, 5)) * 100

        homogenity = get_homogenity(raw_image)

        # Uniform image should have high homogenity
        self.assertIsInstance(homogenity, (float, np.floating))
        self.assertGreater(homogenity, 0.0)

    def test_random_image(self):
        """Test homogenity of random image."""
        np.random.seed(42)
        raw_image = np.random.randint(0, 256, size=(5, 5, 5))

        homogenity = get_homogenity(raw_image)

        # Should return a valid number
        self.assertIsInstance(homogenity, (float, np.floating))
        self.assertFalse(np.isnan(homogenity))
        self.assertFalse(np.isinf(homogenity))

    def test_binary_image(self):
        """Test homogenity of binary image."""
        raw_image = np.random.choice([0, 255], size=(4, 4, 4))

        homogenity = get_homogenity(raw_image)

        # Should return a valid number
        self.assertIsInstance(homogenity, (float, np.floating))
        self.assertGreater(homogenity, 0.0)


if __name__ == "__main__":
    unittest.main()
