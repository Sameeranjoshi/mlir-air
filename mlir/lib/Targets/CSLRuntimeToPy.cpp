//===- CSLRuntimeToPy.cpp - Emit Python from csl_rt dialect -----*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Translation backend: csl_rt dialect → Python (single run.py file).
// Also extracts PE programs from csl.kernel ops.
//
// Pipeline:
//   csl.kernel (PE-level) + csl_rt ops (SDK-level)
//   → pe_program.csl (extracted from csl.kernel)
//   → run.py (complete script with inline build_layout function)
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

/// Simple Python code generation utility
class PythonWriter {
public:
  explicit PythonWriter(llvm::raw_ostream &os) : os(os), indentLevel(0) {}

  void indent() { indentLevel++; }
  void dedent() {
    if (indentLevel > 0)
      indentLevel--;
  }

  llvm::raw_ostream &line() {
    os << std::string(indentLevel * 4, ' ');
    return os;
  }

  void blankLine() { os << "\n"; }
  llvm::raw_ostream &raw() { return os; }

private:
  llvm::raw_ostream &os;
  int indentLevel;
};

/// Extracts and emits CSL kernel bodies and generates Python runtime.
class CSLRuntimeToPyTranslator {
public:
  CSLRuntimeToPyTranslator(llvm::raw_ostream &out) : out(out), writer(out) {}

  LogicalResult translate(ModuleOp module) {
    // Step 1: Find and extract all csl.kernel ops
    std::vector<mlir::Operation*> kernels;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "csl.kernel") {
        kernels.push_back(op);
      }
    });

    // Step 2: Generate PE program files from csl.kernel ops
    if (!kernels.empty()) {
      emitKernelPrograms(kernels);
      out << "\n";
    }

    // Step 3: Generate single run.py file with inline build_layout()
    emitRunPy(module);

    return mlir::success();
  }

private:
  llvm::raw_ostream &out;
  PythonWriter writer;

  void emitKernelPrograms(const std::vector<mlir::Operation*> &kernels) {
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

  void emitRunPy(ModuleOp module) {
    out << "# ============================================================================\n";
    out << "# run.py - Complete Host Runtime Program\n";
    out << "# ============================================================================\n";
    out << "\n";

    out << "#!/usr/bin/env cs_python\n";
    out << "\"\"\"Auto-generated run.py using SdkLayout and SdkRuntime API.\"\"\"\n";
    out << "\n";

    out << "import argparse\n";
    out << "import numpy as np\n";
    out << "# sdkruntimepybind: SdkRuntime API + SdkLayout API\n";
    out << "from cerebras.sdk.runtime.sdkruntimepybind import (\n";
    out << "    SdkRuntime, SdkTarget, SdkLayout, SimfabConfig, get_platform,\n";
    out << ")\n";
    out << "\n";

    // Emit build_layout function
    out << "def build_layout(platform):\n";
    out << "    \"\"\"\n";
    out << "    Build and compile the layout for the WSE.\n";
    out << "    \n";
    out << "    Args:\n";
    out << "        platform: SdkRuntime platform object\n";
    out << "    \n";
    out << "    Returns:\n";
    out << "        compile_artifacts: Result of layout.compile(out_prefix='out')\n";
    out << "    \"\"\"\n";
    out << "    # Create layout\n";
    out << "    layout = SdkLayout(platform)\n";
    out << "\n";

    // Collect and emit all csl_rt operations in order
    emitLayoutOperations(module);

    out << "    # Compile layout\n";
    out << "    compile_artifacts = layout.compile(out_prefix='out')\n";
    out << "    return compile_artifacts\n";
    out << "\n";
    out << "\n";

    // Emit main function
    out << "def main():\n";
    out << "    # Parse command-line arguments\n";
    out << "    parser = argparse.ArgumentParser(description='Run WSE kernel')\n";
    out << "    parser.add_argument('--cmaddr', type=str, default=None,\n";
    out << "                        help='IP:port for CS system')\n";
    out << "    parser.add_argument('--arch', type=str, choices=['wse2', 'wse3'],\n";
    out << "                        default='wse3', help='Target WSE architecture')\n";
    out << "    args = parser.parse_args()\n";
    out << "\n";

    out << "    # Setup platform\n";
    out << "    config = SimfabConfig(dump_core=True)\n";
    out << "    target = SdkTarget.WSE3 if args.arch == 'wse3' else SdkTarget.WSE2\n";
    out << "    platform = get_platform(args.cmaddr, config, target)\n";
    out << "\n";

    out << "    # Build and compile layout\n";
    out << "    compile_artifacts = build_layout(platform)\n";
    out << "\n";

    out << "    # Create and run runtime\n";
    out << "    runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)\n";
    out << "\n";

    // Emit runtime operations
    emitRuntimeOperations(module);

    out << "\n";
    out << "    print('SUCCESS!')\n";
    out << "\n";
    out << "\n";
    out << "if __name__ == '__main__':\n";
    out << "    main()\n";
  }

  void emitLayoutOperations(ModuleOp module) {
    // Process operations at module level and inside functions
    emitLayoutOpsInFunction(module.getOperation());

    // Also walk through any func.func operations
    module.walk([&](func::FuncOp func) {
      emitLayoutOpsInFunction(func.getOperation());
    });
  }

  void emitLayoutOpsInFunction(Operation *op) {
    // Process direct children in order (preserves sequence)
    for (auto &region : op->getRegions()) {
      for (auto &block : region.getBlocks()) {
        for (auto &opChild : block.getOperations()) {
          auto opName = opChild.getName().getStringRef();

          if (opName == "csl_rt.create_code_region") {
            auto createCodeRegion = dyn_cast<CreateCodeRegionOp>(&opChild);
            if (createCodeRegion) {
              StringRef source = createCodeRegion.getSourceAttr().getValue();
              StringRef name = createCodeRegion.getNameAttr().getValue();
              int64_t width = createCodeRegion.getWidthAttr().getValue().getSExtValue();
              int64_t height =
                  createCodeRegion.getHeightAttr().getValue().getSExtValue();

              out << "    code_region = layout.create_code_region(\"" << source
                  << "\", \"" << name << "\", " << width << ", " << height << ")\n";
            }
          } else if (opName == "csl_rt.place") {
            auto place = dyn_cast<PlaceOp>(&opChild);
            if (place) {
              int64_t x = place.getXAttr().getValue().getSExtValue();
              int64_t y = place.getYAttr().getValue().getSExtValue();
              out << "    code_region.place(" << x << ", " << y << ")\n";
            }
          } else if (opName == "csl_rt.set_param_all") {
            auto setParam = dyn_cast<SetParamAllOp>(&opChild);
            if (setParam) {
              StringRef paramName = setParam.getParamNameAttr().getValue();
              int64_t paramValue = setParam.getParamValueAttr().getValue().getSExtValue();
              out << "    code_region.set_param_all(\"" << paramName << "\", "
                  << paramValue << ")\n";
            }
          } else if (opName == "csl_rt.export_name") {
            auto exportName = dyn_cast<ExportNameOp>(&opChild);
            if (exportName) {
              StringRef symName = exportName.getSymbolNameAttr().getValue();
              StringRef symType = exportName.getSymbolTypeAttr().getValue();
              out << "    layout.export_name(\"" << symName << "\", \"" << symType
                  << "\")\n";
            }
          }
        }
      }
    }
  }

  void emitRuntimeOperations(ModuleOp module) {
    // Process operations at module level and inside functions
    emitRuntimeOpsInFunction(module.getOperation());

    // Also walk through any func.func operations
    module.walk([&](func::FuncOp func) {
      emitRuntimeOpsInFunction(func.getOperation());
    });
  }

  void emitRuntimeOpsInFunction(Operation *op) {
    out << "    try:\n";
    out << "        # Load and run the program\n";
    out << "        runtime.load()\n";
    out << "\n";

    // Process direct children in order (preserves sequence)
    bool foundH2D = false;
    bool foundLaunch = false;
    bool foundD2H = false;

    for (auto &region : op->getRegions()) {
      for (auto &block : region.getBlocks()) {
        for (auto &opChild : block.getOperations()) {
          auto opName = opChild.getName().getStringRef();

          if (opName == "csl_rt.load") {
            // Already emitted above
          } else if (opName == "csl_rt.get_id") {
            auto getId = dyn_cast<GetIdOp>(&opChild);
            if (getId) {
              StringRef symbolName = getId.getSymbolNameAttr().getValue();
              out << "        id_" << symbolName << " = runtime.get_id(\""
                  << symbolName << "\")\n";
            }
          } else if (opName == "csl_rt.memcpy_h2d") {
            if (!foundH2D) {
              out << "\n";
              out << "        # Host-to-device transfers\n";
              foundH2D = true;
            }
            auto memcpyH2D = dyn_cast<MemcpyH2dOp>(&opChild);
            if (memcpyH2D) {
              int32_t destId = memcpyH2D.getDestIdAttr().getInt();
              StringRef srcName = memcpyH2D.getSrcNameAttr().getValue();
              int64_t px = memcpyH2D.getPxAttr().getValue().getSExtValue();
              int64_t py = memcpyH2D.getPyAttr().getValue().getSExtValue();
              int64_t w = memcpyH2D.getWAttr().getValue().getSExtValue();
              int64_t h = memcpyH2D.getHAttr().getValue().getSExtValue();
              int64_t elemPerPe =
                  memcpyH2D.getElemPerPeAttr().getValue().getSExtValue();

              out << "        runtime.memcpy_h2d(" << destId << ", \"" << srcName
                  << "\", " << px << ", " << py << ", " << w << ", " << h << ", "
                  << elemPerPe << ")\n";
            }
          } else if (opName == "csl_rt.launch") {
            if (!foundLaunch) {
              out << "\n";
              out << "        # Launch compute kernel\n";
              foundLaunch = true;
            }
            auto launch = dyn_cast<LaunchOp>(&opChild);
            if (launch) {
              StringRef symbolName = launch.getSymbolNameAttr().getValue();
              out << "        runtime.launch(\"" << symbolName << "\")\n";
            }
          } else if (opName == "csl_rt.memcpy_d2h") {
            if (!foundD2H) {
              out << "\n";
              out << "        # Device-to-host transfers\n";
              foundD2H = true;
            }
            auto memcpyD2H = dyn_cast<MemcpyD2hOp>(&opChild);
            if (memcpyD2H) {
              StringRef destName = memcpyD2H.getDestNameAttr().getValue();
              int64_t px = memcpyD2H.getPxAttr().getValue().getSExtValue();
              int64_t py = memcpyD2H.getPyAttr().getValue().getSExtValue();
              int64_t w = memcpyD2H.getWAttr().getValue().getSExtValue();
              int64_t h = memcpyD2H.getHAttr().getValue().getSExtValue();
              int64_t elemPerPe =
                  memcpyD2H.getElemPerPeAttr().getValue().getSExtValue();

              out << "        runtime.memcpy_d2h(\"" << destName << "\", " << px
                  << ", " << py << ", " << w << ", " << h << ", " << elemPerPe
                  << ")\n";
            }
          }
        }
      }
    }

    out << "\n";
    out << "    finally:\n";
    out << "        # Stop the program\n";
    out << "        runtime.stop()\n";
  }
};

/// Translation function for csl_rt → Python.
static LogicalResult translateCSLRuntimeToPy(ModuleOp module, llvm::raw_ostream &os) {
  CSLRuntimeToPyTranslator translator(os);
  return translator.translate(module);
}

} // namespace

void xilinx::csl_rt::registerCSLRuntimeToPyTranslation() {
  TranslateFromMLIRRegistration registration(
      "emit-csl-rt", "Emit CSL Runtime to Python",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return translateCSLRuntimeToPy(module, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect, xilinx::csl_rt::CSLRuntimeDialect,
                        mlir::func::FuncDialect, mlir::arith::ArithDialect,
                        mlir::memref::MemRefDialect>();
      });
}
