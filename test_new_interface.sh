#!/bin/bash
# Simple test script to validate the new inference interface

set -e

echo "=== Testing New Inference Interface ==="

# Test 1: Check help output with no parameters
echo "Test 1: Help output"
bash src/run_analysis.sh 2>&1 | grep -q "docker compose run --rm imagine" && echo "✓ Help shows new interface" || echo "✗ Help missing new interface"

# Test 2: Check parameter parsing for new interface  
echo "Test 2: Parameter parsing for new interface"
bash src/run_analysis.sh 4 - 10 inference /test/models /test/images /test/output 2>&1 | grep -q "Using inference interface with CLI arguments" && echo "✓ New interface detected" || echo "✗ New interface not detected"

# Test 3: Check legacy interface still works
echo "Test 3: Legacy interface compatibility"
bash src/run_analysis.sh 4 data.tsv 10 learning_benchmark 2>&1 | grep -q "learning_benchmark" && echo "✓ Legacy interface works" || echo "✗ Legacy interface broken"

# Test 4: Check Python script accepts new parameter
echo "Test 4: Python script parameter support"
python3 src/inference.py --help | grep -q "images_folder" && echo "✓ Python script supports images_folder parameter" || echo "✗ Python script missing parameter"

echo "=== Test Summary ==="
echo "Basic interface tests completed."
echo "For full end-to-end testing, run with actual model and image data."