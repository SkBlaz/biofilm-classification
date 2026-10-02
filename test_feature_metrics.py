import sys
import unittest

import numpy as np
from scipy import ndimage

sys.path.insert(0, "src")

from feature_metrics import (
    bem_threshold,
    comstat_metrics,
    connected_object_metrics_2d,
    connected_object_metrics_3d,
    internal_pore_metrics_2d,
    internal_pore_metrics_3d,
    local_biomass_density,
    local_thickness_metrics,
    multi_otsu_biomass_mask,
    multi_otsu_labels,
    robust_background_threshold,
    threshold_mask,
)


class TestThresholdBoundaryModes(unittest.TestCase):
    def test_comstat_modes_differ_at_equality(self):
        image = np.array([0, 1, 2], dtype=np.uint8)
        np.testing.assert_array_equal(threshold_mask(image, 1, "comstat1"), [False, False, True])
        np.testing.assert_array_equal(threshold_mask(image, 1, "comstat2"), [False, True, True])

    def test_robust_background_matches_biofilmq_trimmed_gaussian(self):
        image = np.arange(100, dtype=float)
        capped = image[5:-5]
        self.assertAlmostEqual(robust_background_threshold(image), capped.mean() + 2 * capped.std(ddof=1))
        self.assertGreater(robust_background_threshold(np.zeros(10)), 0)

    def test_three_class_labels_remain_intensity_only(self):
        values = np.array([0.0, 0.3, 0.7, 1.0])
        labels = multi_otsu_labels(values, 0.3, 0.7)
        np.testing.assert_array_equal(labels, [0, 1, 2, 2])
        labels_comstat1 = multi_otsu_labels(values, 0.3, 0.7, mode="comstat1")
        np.testing.assert_array_equal(labels_comstat1, [0, 0, 1, 2])
        np.testing.assert_array_equal(multi_otsu_biomass_mask(labels, "foreground"), [False, True, True, True])
        np.testing.assert_array_equal(multi_otsu_biomass_mask(labels, "background"), [False, False, True, True])

    def test_bem_requires_its_documented_8_bit_input(self):
        image = np.concatenate((np.zeros(10000, dtype=np.uint8), np.arange(1, 200, dtype=np.uint8)))
        self.assertIn(bem_threshold(image), range(1, 255))
        with self.assertRaisesRegex(ValueError, "saturated"):
            bem_threshold(np.array([0, 255], dtype=np.uint8))

    def test_bem_matches_the_power_curve_slope_criterion(self):
        thresholds = np.arange(1, 256, dtype=float)
        volume = 10000 * (thresholds**-0.5 - 255**-0.5)
        histogram = np.r_[10000, np.maximum(0, np.rint(volume[:-1] - volume[1:])).astype(int), 0]
        image = np.repeat(np.arange(256, dtype=np.uint8), histogram)
        expected = next(threshold for threshold in range(1, 255) if 1 - ((threshold + 1) / threshold) ** -1.5 < 0.1)
        self.assertLessEqual(abs(bem_threshold(image) - expected), 1)


class TestComstatMetrics(unittest.TestCase):
    def test_height_and_compacted_thickness_are_distinct(self):
        mask = np.zeros((4, 1, 2), dtype=bool)
        mask[0, 0, 0] = True
        mask[3, 0, 0] = True
        result = comstat_metrics(mask, (2, 3, 0.5))
        self.assertEqual(result["MaxBiofilmHeight_COMSTAT_um"], 1.5)
        self.assertEqual(result["MaxCompactedColumnThickness_um"], 1.0)
        self.assertEqual(result["VerticalFillRatio"], 0.5)

    def test_roughness_includes_empty_columns(self):
        mask = np.zeros((3, 1, 2), dtype=bool)
        mask[2, 0, 0] = True
        result = comstat_metrics(mask, (1, 1, 1))
        self.assertAlmostEqual(result["MeanThickness_COMSTAT_um"], 1.0)
        self.assertAlmostEqual(result["Roughness_RaStar"], 1.0)

    def test_empty_mask_has_nan_metrics_and_flag(self):
        result = comstat_metrics(np.zeros((2, 2, 3)), (1, 1, 1))
        self.assertTrue(result["NoBiomassDetected"])
        self.assertTrue(np.isnan(result["MaxBiofilmHeight_COMSTAT_um"]))
        self.assertTrue(np.isnan(result["BiomassCenterOfMassHeight_um"]))

    def test_physical_surface_and_diffusion_metrics_use_anisotropic_spacing(self):
        mask = np.ones((1, 1, 1), dtype=bool)
        result = comstat_metrics(mask, (2, 3, 0.5))
        self.assertEqual(result["BiomassVolume_um3"], 3.0)
        self.assertEqual(result["SurfaceArea_voxel_faces_um2"], 17.0)
        self.assertEqual(result["MeanDiffusionDistance_um"], 0.5)


class TestConnectedObjects(unittest.TestCase):
    def test_nearest_neighbors_use_anisotropic_xy_coordinates(self):
        mask = np.zeros((4, 5), dtype=bool)
        mask[0, 0] = True
        mask[0, 1] = True
        mask[3, 4] = True
        result = connected_object_metrics_2d(mask, (2, 3, 1))
        self.assertEqual(result["ObjectCount"], 2)
        self.assertAlmostEqual(result["NearestNeighborDistances_um"][0], np.sqrt(9 * 9 + 7 * 7))

    def test_3d_connectivity_is_explicit_and_defaults_to_26(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        mask[1, 1, 1] = True
        self.assertEqual(connected_object_metrics_3d(mask, (1, 1, 1), connectivity=6)["ObjectCount3D"], 2)
        self.assertEqual(connected_object_metrics_3d(mask, (1, 1, 1))["ObjectCount3D"], 1)

    def test_internal_pores_exclude_border_connected_background(self):
        mask = np.ones((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = False
        result = internal_pore_metrics_3d(mask, (1, 1, 1))
        self.assertEqual(result["InternalPoreCount3D"], 1)
        self.assertAlmostEqual(result["InternalPorosity3D"], 1 / 27)
        self.assertEqual(internal_pore_metrics_3d(np.zeros_like(mask), (1, 1, 1))["InternalPoreCount3D"], 0)

    def test_2d_pores_exclude_border_connected_background(self):
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        result = internal_pore_metrics_2d(mask, (2, 3, 1))
        self.assertEqual(result["InternalPoreCount2D"], 1)
        self.assertEqual(result["InternalPoreAreas_um2"].tolist(), [6.0])
        self.assertEqual(result["InternalPorosity2D"], 6.0 / 150.0)

    def test_local_thickness_uses_calibrated_inscribed_spheres(self):
        mask = np.ones((3, 3, 3), dtype=bool)
        result = local_thickness_metrics(mask, (1, 1, 1))
        self.assertEqual(result["LocalThicknessMax_um"], 4.0)
        self.assertTrue(np.isnan(local_thickness_metrics(np.zeros_like(mask), (1, 1, 1))["LocalThicknessMean_um"]))

    def test_local_density_uses_physical_ball(self):
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        result = local_biomass_density(mask, (1, 1, 1), radius_um=0.5)
        self.assertEqual(result["LocalBiomassDensityMean"], 1.0)

    def test_fft_density_matches_direct_spherical_counts_at_edges(self):
        mask = np.random.default_rng(3).random((12, 17, 19)) > 0.7
        dz, dy, dx = 0.8, 0.4, 0.3
        radius = 1.5
        rz, ry, rx = [int(radius / step) for step in (dz, dy, dx)]
        z, y, x = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
        kernel = ((z * dz) ** 2 + (y * dy) ** 2 + (x * dx) ** 2 <= radius**2).astype(float)
        count = ndimage.convolve(mask.astype(float), kernel, mode="constant")
        total = ndimage.convolve(np.ones(mask.shape), kernel, mode="constant")
        actual = local_biomass_density(mask, (dx, dy, dz), radius)
        np.testing.assert_allclose(actual["LocalBiomassDensityMap"], count / total, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
