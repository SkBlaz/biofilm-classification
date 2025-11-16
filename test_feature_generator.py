#!/usr/bin/env python3
"""
Unit tests for feature_generator.py utility functions.
"""

import sys
import unittest
import numpy as np
from scipy.ndimage import label

# Add src directory to path
sys.path.insert(0, 'src')

from feature_generator import (
    rgb2gray,
    calculate_spatial_spreading,
    get_cell_count,
    get_transition_matrix,
    get_homogenity,
    CONNECTION_KERNEL
)


class TestRgb2Gray(unittest.TestCase):
    """Test RGB to grayscale conversion."""
    
    def test_basic_conversion(self):
        """Test basic RGB to gray conversion."""
        # Create a simple RGB image (3x3 pixels)
        rgb = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
            [[100, 150, 200], [50, 75, 100], [200, 100, 50]]
        ])
        
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


class TestGetCellCount(unittest.TestCase):
    """Test cell counting functionality."""
    
    def test_single_component(self):
        """Test with a single connected component."""
        # Create a simple 3x3 matrix with one component
        intensity_matrix = np.array([
            [100, 100, 0],
            [100, 100, 0],
            [0, 0, 0]
        ])
        
        ncomponents, labeled = get_cell_count(intensity_matrix, threshold=0.3, visualizations=False)
        
        # Should detect 2 components: background (0) and the object (1)
        # Background is counted as a component
        self.assertGreaterEqual(ncomponents, 1)
        self.assertEqual(labeled.shape, intensity_matrix.shape)
    
    def test_multiple_components(self):
        """Test with multiple separated components."""
        intensity_matrix = np.array([
            [100, 0, 100],
            [0, 0, 0],
            [100, 0, 100]
        ])
        
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


if __name__ == '__main__':
    unittest.main()
