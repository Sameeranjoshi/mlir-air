#!/bin/bash

# get machine
salloc -p amd-arad -N 1 --gres=gpu:mi300x:1 -t 1:00:00

# setup env, this sets the paths and initialized the binaries into the PATH, aircc/air-opt is now in path.
cd /uufs/chpc.utah.edu/common/home/u1418973/other/amd-air/mlir-air-gpu
module load python/3.10.3 cmake/3.26.0 rocm/6.2.1
source sandbox/bin/activate
source utils/env_setup_gpu.sh install llvm/install


# test
# gfx942 = mi300x
# compile
# aircc.py --target gpu --gpu-arch gfx942 -v -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir
aircc.py --target gpu --gpu-arch gfx942 -v --tmpdir output_intermediate -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir

# run
mlir-runner --entry-point-result=void --shared-libs=$LLVM_INSTALL_DIR/lib/libmlir_rocm_runtime.so output.mlir