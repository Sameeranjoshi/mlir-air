#!/usr/bin/env bash
# Rebuild mlir-air-gpu from scratch (assumes git clone is already done).
# Usage: cd mlir-air-gpu && bash rebuild_from_scratch.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== Step 1: Load modules ==="
#module load python/3.10.3 cmake/3.26.0 rocm/6.2.1 singularity

echo "=== Step 2: Create & activate Python venv ==="
source /home/bricklib_dataflow/air-csl/venv_air_csl/bin/activate

# Conda-forge toolchain installed at venv_air_csl/tools (gcc 13, git)
# used because the base OS ships neither clang nor g++.
TOOLS_PREFIX=/home/bricklib_dataflow/air-csl/venv_air_csl/tools
export PATH="$TOOLS_PREFIX/bin:$PATH"
export CC="$TOOLS_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$TOOLS_PREFIX/bin/x86_64-conda-linux-gnu-g++"
# Conda python has Development headers; venv's system python3.9 does not.
export PYTHON3_EXECUTABLE="$TOOLS_PREFIX/bin/python3.9"

echo "=== Step 3: Install Python dependencies ==="
pip install --upgrade pip
pip install cmake ninja lit pybind11 numpy "nanobind>=2.9" typing_extensions psutil

echo "=== Step 4: Build LLVM (this takes a long time) ==="
./utils/build-llvm-local.sh llvm

echo "=== Step 5: Build mlir-air-gpu ==="
./utils/build-mlir-air-gpu.sh llvm

echo "=== Step 6: Set up environment ==="
# env_setup_gpu.sh appends to PYTHONPATH/LD_LIBRARY_PATH; pre-init them for set -u.
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source utils/env_setup_gpu.sh install llvm/install

echo "=== Step 7: Run tests ==="
cd build && ninja check-air-mlir

echo ""
echo "=== Done! ==="
echo "To use in future shells, run:"
echo "  module load python/3.10.3 cmake/3.26.0 rocm/6.2.1 singularity"
echo "  source sandbox/bin/activate"
echo "  source utils/env_setup_gpu.sh install llvm/install"
