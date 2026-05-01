//===- CSLAllocateColorIds.cpp ---------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Pass 2 of csl-streams-to-csl pipeline.
// Pre:  every csl.color exists (with or without `id`).
// Post: every csl.color has an `id` attribute.
//
// Algorithm (milestone — stub): monotonic, skip pinned. Future home for
// liveness-driven graph coloring over WSE-3's 24-color budget.
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLAllocateColorIdsPass.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallSet.h"

using namespace mlir;

namespace {

class CSLAllocateColorIdsPass
    : public PassWrapper<CSLAllocateColorIdsPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-allocate-color-ids"; }
  StringRef getDescription() const final {
    return "Pass 2: assign integer ids to virtual csl.color ops "
           "(monotonic stub; future graph-coloring home).";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::xilinx::csl::CSLDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void CSLAllocateColorIdsPass::runOnOperation() {
  // First gather already-pinned ids.
  llvm::SmallSet<int32_t, 32> usedIds;
  getOperation()->walk([&](::xilinx::csl::ColorOp c) {
    if (auto idAttr = c.getIdAttr())
      usedIds.insert(idAttr.getInt());
  });

  // Walk again, assigning fresh ids in declaration order.
  int32_t next = 0;
  getOperation()->walk([&](::xilinx::csl::ColorOp c) {
    if (c.getIdAttr())
      return;
    while (usedIds.count(next))
      ++next;
    c.setIdAttr(IntegerAttr::get(
        IntegerType::get(c.getContext(), 32), next));
    usedIds.insert(next);
    ++next;
  });
}

std::unique_ptr<Pass> xilinx::air::createCSLAllocateColorIdsPass() {
  return std::make_unique<CSLAllocateColorIdsPass>();
}
