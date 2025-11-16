#!/usr/bin/env python3
"""
Run all unit tests for the biofilm classification project.
"""

import sys
import unittest

# Discover and run all tests
loader = unittest.TestLoader()
suite = loader.discover(".", pattern="test_*.py")

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Exit with appropriate code
sys.exit(0 if result.wasSuccessful() else 1)
