//===- CSLMaterializeStreamColors.cpp -------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Pass 1 of csl-streams-to-csl pipeline.
// Pre:  every csl_layout.stream has no `color` attr.
// Post: every csl_layout.stream has {color = @<sym>}; matching csl.color
//       @<sym> exists in same csl.layout body (no id yet).
//
// Naming: each stream @S gets a color symbol named "@S_color".
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLMaterializeStreamColorsPass.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

class CSLMaterializeStreamColorsPass
    : public PassWrapper<CSLMaterializeStreamColorsPass, OperationPass<>> {
public:
  StringRef getArgument() const final {
    return "csl-materialize-stream-colors";
  }
  StringRef getDescription() const final {
    return "Pass 1: synthesize csl.color symbols for csl_layout.stream ops";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::xilinx::csl::CSLDialect,
                    ::xilinx::csl_layout::CSLLayoutDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void CSLMaterializeStreamColorsPass::runOnOperation() {
  Operation *op = getOperation();
  op->walk([&](::xilinx::csl::LayoutOp layout) {
    Block &body = layout.getBody().front();

    // Snapshot the stream ops before mutating siblings.
    SmallVector<::xilinx::csl_layout::StreamOp> streams;
    for (Operation &nested : body) {
      if (auto s = dyn_cast<::xilinx::csl_layout::StreamOp>(&nested))
        streams.push_back(s);
    }

    OpBuilder builder(&body, body.begin());
    for (auto stream : streams) {
      if (stream.getColorAttr())
        continue; // idempotent

      // Build a fresh color name. If <stream>_color is already taken in the
      // layout's SymbolTable, suffix _0, _1, … until a free name is found.
      std::string baseName = (stream.getSymName() + "_color").str();
      std::string colorName = baseName;
      unsigned suffix = 0;
      while (mlir::SymbolTable::lookupSymbolIn(layout, colorName)) {
        colorName = baseName + "_" + std::to_string(suffix++);
      }

      // Insert csl.color at the top of the layout body.
      builder.setInsertionPointToStart(&body);
      builder.create<::xilinx::csl::ColorOp>(
          stream.getLoc(),
          /*sym_name=*/builder.getStringAttr(colorName),
          /*id=*/IntegerAttr());

      // Set {color = @<colorName>} on the stream.
      stream.setColorAttr(
          FlatSymbolRefAttr::get(builder.getContext(), colorName));
    }
  });
}

std::unique_ptr<Pass> xilinx::air::createCSLMaterializeStreamColorsPass() {
  return std::make_unique<CSLMaterializeStreamColorsPass>();
}
