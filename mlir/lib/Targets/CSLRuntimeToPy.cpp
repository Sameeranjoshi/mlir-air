//===- CSLRuntimeToPy.cpp - Emit Python from csl_rt dialect -----*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Translation backend: csl_rt dialect → Python (layout.py and run.py).
// Walks csl_rt ops and emits corresponding SdkLayout and SdkRuntime API calls.
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSLRuntime/CSLRuntimeDialect.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;
using namespace xilinx::csl_rt;

namespace {

/// Translates a csl_rt module to Python (layout.py and run.py).
class CSLRuntimeToPyTranslator {
public:
  CSLRuntimeToPyTranslator(StringRef outputDir) : outputDir(outputDir) {}

  LogicalResult translate(ModuleOp module) {
    // For now, generate minimal template files.
    // TODO: Walk csl_rt ops and emit actual layout.py and run.py

    emitLayoutPy();
    emitRunPy();

    return success();
  }

private:
  StringRef outputDir;
  std::string layoutPy;
  std::string runPy;

  void emitLayoutPy() {
    layoutPy = R"(#!/usr/bin/env python3
"""
Auto-generated layout.py using SdkLayout API.
This file configures the spatial placement and compilation for the Cerebras WSE.
"""

from cerebras.sdk.sdk_runtime import SdkLayout

# Create layout
layout = SdkLayout()

# TODO: Add code regions and exports from csl_rt ops

# Compile layout
compile_artifacts = layout.compile(out_prefix='out')

print(f"Compilation artifacts: {compile_artifacts}")
)";
  }

  void emitRunPy() {
    runPy = R"(#!/usr/bin/env python3
"""
Auto-generated run.py template using SdkRuntime API.
This file implements the host-side control flow for running the WSE kernel.
"""

import argparse
from cerebras.sdk.sdk_runtime import SdkRuntime, SdkRuntimeContext
from cerebras.sdk.common import MemcpyDirection
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Run WSE kernel')
    parser.add_argument('--kernel-dir', type=str, default='out',
                        help='Directory with compiled kernel artifacts')
    parser.add_argument('--platform', type=str, default='wse3',
                        help='Platform (wse3 or wse2)')
    args = parser.parse_args()

    # Create runtime from compiled artifacts
    runtime = SdkRuntime(args.kernel_dir, platform=args.platform)

    with SdkRuntimeContext(runtime) as ctx:
        # Load kernel
        runtime.load()

        # TODO: Add memcpy_h2d, launch, get_id calls from csl_rt ops

        # Run kernel
        runtime.run()

        # TODO: Add memcpy_d2h calls to read results

    runtime.stop()
    print("Kernel execution completed")

if __name__ == '__main__':
    main()
)";
  }

public:
  const std::string& getLayoutPy() const { return layoutPy; }
  const std::string& getRunPy() const { return runPy; }
};

/// Translation function for csl_rt → Python.
static LogicalResult translateCSLRuntimeToPy(ModuleOp module, raw_ostream &os) {
  // For now, emit a template message to stdout
  os << "# CSL Runtime to Python translation (template)\n";
  os << "# This would generate layout.py and run.py from csl_rt dialect IR\n";
  return success();
}

} // namespace

void xilinx::csl_rt::registerCSLRuntimeToPyTranslation() {
  TranslateFromMLIRRegistration registration(
      "emit-csl-rt", "Emit CSL Runtime to Python",
      [](ModuleOp module, raw_ostream &os) {
        return translateCSLRuntimeToPy(module, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<CSLRuntimeDialect>();
      });
}
