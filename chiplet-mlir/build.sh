#!/usr/bin/env bash
# Build chiplet-mlir against mlir-air's sibling LLVM install.
#
# Usage:
#   ./build.sh            # configure + build
#   ./build.sh clean      # remove build/ first

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"

# LLVM install is expected at ../llvm/install (mlir-air sibling layout).
MLIR_INSTALL="${MLIR_INSTALL:-${SCRIPT_DIR}/../llvm/install}"

if [[ ! -f "${MLIR_INSTALL}/lib/cmake/mlir/MLIRConfig.cmake" ]]; then
  echo "error: MLIRConfig.cmake not found at ${MLIR_INSTALL}/lib/cmake/mlir/"
  echo "Set MLIR_INSTALL to the directory containing lib/cmake/mlir/MLIRConfig.cmake."
  exit 1
fi

if [[ "${1:-}" == "clean" ]]; then
  rm -rf build
fi

mkdir -p build
cd build

cmake -G Ninja .. \
  -DMLIR_DIR="${MLIR_INSTALL}/lib/cmake/mlir" \
  -DLLVM_DIR="${MLIR_INSTALL}/lib/cmake/llvm" \
  -DLLVM_EXTERNAL_LIT="$(command -v lit || echo "${MLIR_INSTALL}/../build/bin/llvm-lit")" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang

ninja
