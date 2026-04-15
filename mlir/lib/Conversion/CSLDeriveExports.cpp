//===- CSLDeriveExports.cpp - Infer csl.export directions ------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Implements the -csl-derive-exports pass.
//
// Algorithm (per csl.wafer):
//   1. Collect the single csl.program (first found) and csl.host.
//   2. Walk csl_host.memcpy_h2d ops → direction = "in"
//      Walk csl_host.memcpy_d2h ops → direction = "out"
//      The transfer sym is @layout::@alias.  Look up the alias in
//      csl_layout.export ops to find the PE variable symbol, then find
//      the matching csl.export in the program.
//   3. csl.export ops with no direction receive direction = "internal".
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/CSLDeriveExportsPass.h"
#include "air/Dialect/CSL/CSLHostOps.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

// Returns the LeafReference name of a SymbolRefAttr (the last nested ref,
// or the root if there are no nested refs).
static StringRef leafRef(SymbolRefAttr sym) {
  if (sym.getNestedReferences().empty())
    return sym.getRootReference().getValue();
  // FlatSymbolRefAttr has no getValue(); use getRootReference().getValue()
  return sym.getNestedReferences().back().getRootReference().getValue();
}

// Given a WaferOp, build a map from export alias → csl.export op for
// all exports in all csl.program children.
//
// Key:   alias string (e.g., "a")
// Value: pointer to csl.ExportOp
static llvm::DenseMap<StringAttr, xilinx::csl::ExportOp>
collectExportsByAlias(xilinx::csl::WaferOp wafer) {
  llvm::DenseMap<StringAttr, xilinx::csl::ExportOp> result;
  wafer.getBody().walk([&](xilinx::csl::ExportOp exp) {
    // Use the alias attribute if present, otherwise fall back to sym name.
    StringAttr key;
    if (auto alias = exp.getAliasAttr())
      key = alias;
    else
      key = StringAttr::get(exp.getContext(), exp.getSym());
    result[key] = exp;
  });
  return result;
}

// Annotate exports with direction derived from host transfer ops.
static void deriveExports(xilinx::csl::WaferOp wafer) {
  MLIRContext *ctx = wafer.getContext();
  OpBuilder builder(ctx);

  // Collect exports by alias.
  auto exportByAlias = collectExportsByAlias(wafer);

  // Helper: annotate an export op with the given direction string.
  auto annotate = [&](StringAttr alias, StringRef direction) {
    auto it = exportByAlias.find(alias);
    if (it == exportByAlias.end())
      return;
    xilinx::csl::ExportOp exp = it->second;
    // Skip if already annotated (first wins).
    if (exp.getDirection())
      return;
    exp->setAttr("direction", StringAttr::get(ctx, direction));
  };

  // Walk csl.host ops inside this wafer.
  wafer.getBody().walk([&](xilinx::csl::HostOp host) {
    host.getBody().walk([&](Operation *op) {
      if (auto h2d = dyn_cast<xilinx::csl_host::MemcpyH2DOp>(op)) {
        // sym = @layout::@alias — use the leaf (nested) ref as the alias.
        StringAttr alias = StringAttr::get(ctx, leafRef(h2d.getSym()));
        annotate(alias, "in");
      } else if (auto d2h = dyn_cast<xilinx::csl_host::MemcpyD2HOp>(op)) {
        StringAttr alias = StringAttr::get(ctx, leafRef(d2h.getSym()));
        annotate(alias, "out");
      }
    });
  });

  // Any export without a direction receives "internal".
  wafer.getBody().walk([&](xilinx::csl::ExportOp exp) {
    if (!exp.getDirection())
      exp->setAttr("direction", StringAttr::get(ctx, "internal"));
  });
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct CSLDeriveExportsPass
    : public PassWrapper<CSLDeriveExportsPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLDeriveExportsPass)

  StringRef getArgument() const override { return "csl-derive-exports"; }
  StringRef getDescription() const override {
    return "Annotate csl.export direction from csl_host transfer ops";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](xilinx::csl::WaferOp wafer) { deriveExports(wafer); });
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::air::createCSLDeriveExportsPass() {
  return std::make_unique<CSLDeriveExportsPass>();
}
