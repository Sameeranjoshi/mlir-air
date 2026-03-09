#!/bin/bash
# Helper script to run tests with verbose output showing test names

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TEST_DIR="${BUILD_DIR}/mlir/test"

if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory not found at $TEST_DIR"
    exit 1
fi

# Get the test suite from argument (default to all)
TEST_SUITE="${1:-all}"

# Determine which tests to run
case "$TEST_SUITE" in
    csld)
        TESTS="Dialect/CSL"
        ;;
    csl-rt)
        TESTS="Dialect/CSLRuntime"
        ;;
    csld-to-csl-rt)
        TESTS="Conversion/CSLToCSLRuntime"
        ;;
    csl-rt-to-csl)
        TESTS="Targets/CSLRuntimeToCSL"
        ;;
    air-to-csld)
        TESTS="Conversion/AIRToCSL"
        ;;
    all)
        TESTS="Dialect/CSL Dialect/CSLRuntime Conversion/CSLToCSLRuntime Targets/CSLRuntimeToCSL Conversion/AIRToCSL"
        ;;
    *)
        echo "Usage: $0 {csld|csl-rt|csld-to-csl-rt|csl-rt-to-csl|air-to-csld|all}"
        exit 1
        ;;
esac

# Run tests with verbose output
cd "$TEST_DIR"
/uufs/chpc.utah.edu/common/home/u1418973/other/amd-air/mlir-air-gpu/sandbox/bin/python3.10 \
    /uufs/chpc.utah.edu/common/home/u1418973/other/amd-air/mlir-air-gpu/sandbox/bin/lit \
    -v --timeout=30 $TESTS
