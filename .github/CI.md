# Continuous Integration (CI)

This repository includes automated CI testing for Docker images using GitHub Actions.

## CI Workflow

The CI workflow (`.github/workflows/docker-tests.yml`) automatically runs on:
- Pushes to `main` or `master` branches
- Pull requests targeting `main` or `master` branches

## Tests Performed

The CI pipeline performs the following tests:

1. **Docker Image Build**: Builds the Docker image defined in `Dockerfile`
2. **Feature Generation Test**: Runs `docker compose run --rm imagine-test-generate-features`
3. **Learning Benchmark Test**: Runs `docker compose run --rm imagine-test-learning-benchmark`
4. **Data Visualization Test**: Runs `docker compose run --rm imagine-test-data-visualization`
5. **Result Verification**: Confirms that test results are generated successfully
6. **Artifact Upload**: Uploads test results as GitHub artifacts for inspection

## Test Data

The CI uses test data located in `examples/test_images/` directory, which contains 6 biofilm image files (.tif format).

Test results are generated in `examples/test_images_results/` directory.

## Timeouts

- Overall job timeout: 60 minutes
- Individual test timeouts: 15 minutes each

## Viewing Results

Test results are uploaded as GitHub Actions artifacts and can be downloaded from the workflow run page for up to 30 days.