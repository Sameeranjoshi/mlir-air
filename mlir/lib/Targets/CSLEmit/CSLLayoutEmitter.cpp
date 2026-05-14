//===- CSLLayoutEmitter.cpp - Emit layout.csl ------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Registers --emit-csl-layout. Walks csl.wafer > csl.layout and emits the
// CSL layout wrapper (layout.csl) that cslc compiles: imports
// `<memcpy/get_params>`, calls `@set_rectangle` + `@set_tile_code(...)` per
// PE placement, and declares host-visible symbols via `@export_name`.
//
// Implementation: thin wrapper around makeLayoutCsl/probeLayout from
// CSLEmitAll.cpp — the same builder the unified --emit-csl uses for the
// per-wafer layout.csl file. Keeps single source of truth.
//
//===----------------------------------------------------------------------===//

#include "CSLEmitCommon.h"

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLHostOps.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace xilinx {
namespace csl {

// Forward-declared in CSLEmitAll.cpp.
std::string makeLayoutCsl(xilinx::csl::WaferOp wafer, int64_t width,
                          int64_t height, const std::string &progName);
void probeLayout(xilinx::csl::WaferOp wafer, std::string &progName,
                 int64_t &width, int64_t &height);

// Wafer-scoped entry point. Emits the layout.csl wrapper to `os`.
LogicalResult runLayoutEmitter(xilinx::csl::WaferOp wafer,
                               llvm::raw_ostream &os) {
  std::string progName;
  int64_t width, height;
  probeLayout(wafer, progName, width, height);
  os << makeLayoutCsl(wafer, width, height, progName);
  return success();
}

// Module-scoped entry point for --emit-csl-layout. Picks the first wafer.
LogicalResult runLayoutEmitter(ModuleOp module, llvm::raw_ostream &os) {
  namespace cslns = xilinx::csl;
  llvm::SmallVector<cslns::WaferOp, 4> wafers;
  module.walk([&](cslns::WaferOp w) { wafers.push_back(w); });
  if (wafers.empty()) {
    os << "// No csl.wafer found in module.\n";
    return success();
  }
  if (wafers.size() > 1)
    os << "// note: " << wafers.size()
       << " wafers in module; emitting " << wafers[0].getSymName() << "\n";
  return runLayoutEmitter(wafers[0], os);
}

void registerCSLLayoutTranslation() {
  static TranslateFromMLIRRegistration reg(
      "emit-csl-layout",
      "Emit CSL layout wrapper (layout.csl) from csl.layout",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        return runLayoutEmitter(module, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect,
                        xilinx::csl_layout::CSLLayoutDialect,
                        xilinx::csl_host::CSLHostDialect,
                        mlir::arith::ArithDialect,
                        mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect>();
      });
}

} // namespace csl
} // namespace xilinx
