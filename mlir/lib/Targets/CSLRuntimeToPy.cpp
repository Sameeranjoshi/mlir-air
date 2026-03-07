//===- CSLRuntimeToPy.cpp - Emit Python from csl_rt dialect -----*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Translation backend: csl_rt dialect → Python (layout.py and run.py).
// Also extracts PE programs from csl.kernel ops.
//
// Pipeline:
//   csl.kernel (PE-level) + csl_rt ops (SDK-level)
//   → pe_program.csl (extracted from csl.kernel)
//   → layout.py (generated from csl_rt ops)
//   → run.py (template)
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeDialect.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

using namespace mlir;
using namespace xilinx::csl_rt;

namespace {

/// Extracts and emits CSL kernel bodies and generates Python layout API.
class CSLRuntimeToPyTranslator {
public:
  CSLRuntimeToPyTranslator(raw_ostream &out) : out(out) {}

  LogicalResult translate(ModuleOp module) {
    // Step 1: Find and extract all csl.kernel ops
    std::vector<Operation*> kernels;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "csl.kernel") {
        kernels.push_back(op);
      }
    });

    // Step 2: Generate PE program files from csl.kernel ops
    if (!kernels.empty()) {
      emitKernelPrograms(kernels);
      out << "\n";
    }

    // Step 3: Generate layout.py from csl_rt ops
    emitLayoutPy(module);
    out << "\n";

    // Step 4: Generate run.py template
    emitRunPy();

    return success();
  }

private:
  raw_ostream &out;

  void emitKernelPrograms(const std::vector<Operation*> &kernels) {
    out << "# ============================================================================\n";
    out << "# PE Program (pe_program.csl)\n";
    out << "# ============================================================================\n";
    out << "# Extracted from csl.kernel ops in the input MLIR\n";
    out << "# This should be written to: pe_program.csl\n";
    out << "\n";

    for (size_t i = 0; i < kernels.size(); ++i) {
      // Get the kernel filename if available (would be in csl.kernel sourceFile)
      // For now, just label them
      if (kernels.size() > 1) {
        out << "# Kernel " << (i + 1) << ":\n";
      }

      // Walk the kernel body to extract structure
      out << "// Generated PE Program\n";
      out << "// Contains: variables, functions, tasks, comptime blocks\n";
      out << "\n";

      // Extract key information from kernel op
      // Note: In a full implementation, this would walk the kernel's region
      // and emit proper CSL code for each op type
      out << "// param declarations from csl.param ops\n";
      out << "// var declarations from csl.var ops\n";
      out << "// function definitions from csl.func ops\n";
      out << "// task definitions from csl.task ops\n";
      out << "// comptime blocks from csl.comptime ops\n";
      out << "\n";
    }
  }

  void emitLayoutPy(ModuleOp module) {
    out << "# ============================================================================\n";
    out << "# Layout Configuration (layout.py)\n";
    out << "# ============================================================================\n";
    out << "# Generated from csl_rt dialect ops\n";
    out << "# This should be written to: layout.py\n";
    out << "\n";

    out << "#!/usr/bin/env python3\n";
    out << "\"\"\"Auto-generated layout.py using SdkLayout API.\"\"\"\n";
    out << "\n";

    // Emit imports
    out << "from cerebras.geometry.geometry import IntVector, IntRectangle\n";
    out << "from cerebras.sdk.runtime.sdkruntimepybind import (\n";
    out << "    Color, Edge, Route, RoutingPosition, get_edge_routing,\n";
    out << "    SdkLayout,\n";
    out << ")\n";
    out << "\n";

    // Emit layout creation
    out << "# Create layout\n";
    out << "layout = SdkLayout()\n";
    out << "\n";

    // Walk module to find csl_rt ops
    std::vector<Operation*> cslRtOps;
    module.walk([&](Operation *op) {
      // Check if operation is in csl_rt dialect
      auto opName = op->getName().getStringRef();
      if (opName.starts_with("csl_rt.")) {
        cslRtOps.push_back(op);
      }
    });

    if (!cslRtOps.empty()) {
      out << "# Generated from csl_rt dialect ops:\n";
      for (Operation *op : cslRtOps) {
        out << "# - " << op->getName().getStringRef() << "\n";
      }
      out << "\n";

      out << "# TODO: Emit layout.create_code_region() calls\n";
      out << "# TODO: Emit color, route, paint operations\n";
      out << "# TODO: Emit input/output port creation\n";
      out << "# TODO: Emit parameter setting calls\n";
    } else {
      out << "# No csl_rt ops found - layout is empty\n";
    }

    out << "\n";
    out << "# Compile layout\n";
    out << "compile_artifacts = layout.compile(out_prefix='out')\n";
    out << "print(f'Compilation artifacts: {compile_artifacts}')\n";
  }

  void emitRunPy() {
    out << "# ============================================================================\n";
    out << "# Host Runtime Program (run.py)\n";
    out << "# ============================================================================\n";
    out << "# This should be written to: run.py\n";
    out << "\n";

    out << "#!/usr/bin/env cs_python\n";
    out << "\"\"\"Auto-generated run.py using SdkRuntime API.\"\"\"\n";
    out << "\n";

    out << "import argparse\n";
    out << "import numpy as np\n";
    out << "from cerebras.sdk.runtime.sdkruntimepybind import (\n";
    out << "    SdkRuntime, MemcpyDataType, MemcpyOrder\n";
    out << ")\n";
    out << "\n";

    out << "# Parse command-line arguments\n";
    out << "parser = argparse.ArgumentParser()\n";
    out << "parser.add_argument('--name', help='Compiled kernel directory')\n";
    out << "parser.add_argument('--cmaddr', help='IP:port for CS system')\n";
    out << "args = parser.parse_args()\n";
    out << "\n";

    out << "# Create runtime\n";
    out << "runner = SdkRuntime(args.name, cmaddr=args.cmaddr)\n";
    out << "\n";

    out << "try:\n";
    out << "    # Load and run the program\n";
    out << "    runner.load()\n";
    out << "    runner.run()\n";
    out << "\n";
    out << "    # TODO: Add memcpy_h2d, launch, memcpy_d2h calls\n";
    out << "\n";
    out << "finally:\n";
    out << "    # Stop the program\n";
    out << "    runner.stop()\n";
    out << "\n";
    out << "print('SUCCESS!')\n";
  }
};

/// Translation function for csl_rt → Python.
static LogicalResult translateCSLRuntimeToPy(ModuleOp module, raw_ostream &os) {
  CSLRuntimeToPyTranslator translator(os);
  return translator.translate(module);
}

} // namespace

void xilinx::csl_rt::registerCSLRuntimeToPyTranslation() {
  TranslateFromMLIRRegistration registration(
      "emit-csl-rt", "Emit CSL Runtime to Python",
      [](ModuleOp module, raw_ostream &os) {
        return translateCSLRuntimeToPy(module, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect, CSLRuntimeDialect,
                        func::FuncDialect, arith::ArithDialect,
                        memref::MemRefDialect>();
      });
}
