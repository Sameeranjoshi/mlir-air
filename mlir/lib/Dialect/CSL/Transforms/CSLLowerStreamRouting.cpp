//===- CSLLowerStreamRouting.cpp ---------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Pass 3 of csl-streams-to-csl pipeline.
//
// Pre:  every csl_layout.stream has {color = @<x>}; color has an `id`.
// Post: each stream has two csl_layout.set_color_config siblings (one at
//       `from` coord, one at `to` coord); stream op is kept (still needed
//       by Pass 4 for put/get symbol resolution).
//
// Direction inference from coord delta:
//   (+1,0) -> tx=EAST/rx=WEST
//   (-1,0) -> tx=WEST/rx=EAST
//   (0,+1) -> tx=SOUTH/rx=NORTH
//   (0,-1) -> tx=NORTH/rx=SOUTH
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLLowerStreamRoutingPass.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {

struct InferredDirs {
  ::xilinx::csl::Direction src_tx;
  ::xilinx::csl::Direction dst_rx;
};

static InferredDirs inferDirs(int64_t dx, int64_t dy) {
  using ::xilinx::csl::Direction;
  if (dx == 1 && dy == 0)
    return {Direction::EAST, Direction::WEST};
  if (dx == -1 && dy == 0)
    return {Direction::WEST, Direction::EAST};
  if (dx == 0 && dy == 1)
    return {Direction::SOUTH, Direction::NORTH};
  if (dx == 0 && dy == -1)
    return {Direction::NORTH, Direction::SOUTH};
  llvm_unreachable("verifier on csl_layout.stream rejects non-cardinal deltas");
}

class CSLLowerStreamRoutingPass
    : public PassWrapper<CSLLowerStreamRoutingPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-lower-stream-routing"; }
  StringRef getDescription() const final {
    return "Pass 3: emit per-PE set_color_config from each csl_layout.stream";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::xilinx::csl::CSLDialect,
                    ::xilinx::csl_layout::CSLLayoutDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void CSLLowerStreamRoutingPass::runOnOperation() {
  getOperation()->walk([&](::xilinx::csl::LayoutOp layout) {
    // Snapshot the stream ops before mutating siblings.
    SmallVector<::xilinx::csl_layout::StreamOp> streams;
    for (Operation &nested : layout.getBody().front()) {
      if (auto s = dyn_cast<::xilinx::csl_layout::StreamOp>(&nested))
        streams.push_back(s);
    }

    OpBuilder b(layout.getContext());
    for (auto stream : streams) {
      auto colorAttr = stream.getColorAttr();
      if (!colorAttr) {
        stream.emitOpError("Pass 3 requires a {color = ...} attribute "
                           "(run --csl-materialize-stream-colors first)");
        signalPassFailure();
        return;
      }
      int64_t fx = static_cast<int64_t>(stream.getFromX());
      int64_t fy = static_cast<int64_t>(stream.getFromY());
      int64_t tx = static_cast<int64_t>(stream.getToX());
      int64_t ty = static_cast<int64_t>(stream.getToY());
      int64_t dx = tx - fx;
      int64_t dy = ty - fy;
      auto dirs = inferDirs(dx, dy);

      // Insert set_color_configs after the stream so they appear nearby.
      b.setInsertionPointAfter(stream);
      // Source endpoint: rx = RAMP, tx = inferred.
      b.create<::xilinx::csl_layout::SetColorConfigOp>(
          stream.getLoc(),
          /*color=*/colorAttr.getValue(),
          /*px=*/static_cast<uint64_t>(fx),
          /*py=*/static_cast<uint64_t>(fy),
          /*rx=*/::xilinx::csl::Direction::RAMP,
          /*tx=*/dirs.src_tx);
      // Destination endpoint: rx = inferred, tx = RAMP.
      b.create<::xilinx::csl_layout::SetColorConfigOp>(
          stream.getLoc(),
          /*color=*/colorAttr.getValue(),
          /*px=*/static_cast<uint64_t>(tx),
          /*py=*/static_cast<uint64_t>(ty),
          /*rx=*/dirs.dst_rx,
          /*tx=*/::xilinx::csl::Direction::RAMP);
      // Stream op NOT erased — Pass 4 needs it for symbol resolution.
    }
  });
}

std::unique_ptr<Pass> xilinx::air::createCSLLowerStreamRoutingPass() {
  return std::make_unique<CSLLowerStreamRoutingPass>();
}
