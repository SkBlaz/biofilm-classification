import sys
import unittest

sys.path.insert(0, "src")

from feature_taxonomy import classify_feature, feature_catalog, is_threshold_derived_feature


class TestFeatureTaxonomy(unittest.TestCase):
    def test_segmentation_metrics_are_threshold_derived(self):
        self.assertTrue(is_threshold_derived_feature("MaxBiofilmHeight_COMSTAT_um"))
        self.assertTrue(is_threshold_derived_feature("SpreadingTotal"))
        self.assertTrue(is_threshold_derived_feature("BioVolumeThr0.1"))
        self.assertTrue(is_threshold_derived_feature("SubstratumObjectCount"))
        self.assertFalse(is_threshold_derived_feature("mean"))

    def test_primary_groups_are_biologist_facing(self):
        self.assertEqual(classify_feature("SubstratumObjectCount"), "2D substratum structure")
        self.assertEqual(classify_feature("MaxBiofilmHeight_COMSTAT_um"), "3D basic biofilm metrics")
        self.assertEqual(classify_feature("SurfaceArea_voxel_faces_um2"), "3D advanced geometry")
        self.assertEqual(classify_feature("GLCMContrast_d1_angle0"), "Threshold-free geometry and texture")

    def test_catalog_omits_sample_identifiers(self):
        catalog = feature_catalog(["sampleName", "label", "mean", "MaxBiofilmHeight_COMSTAT_um"])
        self.assertEqual(catalog["Threshold-free intensity"], ["mean"])
        self.assertEqual(catalog["3D basic biofilm metrics"], ["MaxBiofilmHeight_COMSTAT_um"])


if __name__ == "__main__":
    unittest.main()
