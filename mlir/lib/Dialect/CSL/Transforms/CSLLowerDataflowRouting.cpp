//===- CSLLowerDataflowRouting.cpp ---------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Pass 3 of csl-dataflow-to-csl pipeline.
//
// Pre:  every csl_layout.dataflow has {color = @<x>}; color has an `id`.
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

#include "air/Dialect/CSL/Transforms/CSLLowerDataflowRoutingPass.h"
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
  // Verifier guarantees the route is along a single axis with |delta| >= 1.
  // Normalize to step direction; multi-hop is handled by the caller.
  if (dy == 0) {
    if (dx > 0) return {Direction::EAST, Direction::WEST};
    return {Direction::WEST, Direction::EAST};
  }
  if (dy > 0) return {Direction::SOUTH, Direction::NORTH};
  return {Direction::NORTH, Direction::SOUTH};
}

class CSLLowerDataflowRoutingPass
    : public PassWrapper<CSLLowerDataflowRoutingPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-lower-dataflow-routing"; }
  StringRef getDescription() const final {
    return "Pass 3: emit per-PE set_color_config from each csl_layout.dataflow";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::xilinx::csl::CSLDialect,
                    ::xilinx::csl_layout::CSLLayoutDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void CSLLowerDataflowRoutingPass::runOnOperation() {
  getOperation()->walk([&](::xilinx::csl::LayoutOp layout) {
    // Snapshot the dataflow ops before mutating siblings.
    SmallVector<::xilinx::csl_layout::DataflowOp> dataflows;
    for (Operation &nested : layout.getBody().front()) {
      if (auto s = dyn_cast<::xilinx::csl_layout::DataflowOp>(&nested))
        dataflows.push_back(s);
    }

    OpBuilder b(layout.getContext());
    for (auto stream : dataflows) {
      auto colorAttr = stream.getColorAttr();
      if (!colorAttr) {
        stream.emitOpError("Pass 3 requires a {color = ...} attribute "
                           "(run --csl-materialize-dataflow-colors first)");
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
      // Intermediate PEs (multi-hop): walk one step at a time from src to
      // dst along the cardinal axis and emit a pass-through config at each
      // PE strictly between the endpoints. The intermediate's rx is the
      // opposite of the step direction (data arrives from there), its tx
      // is the step direction (data continues toward dst). Verifier on
      // csl_layout.dataflow guarantees the path is purely horizontal or
      // purely vertical.
      int64_t stepX = (dx > 0) ? 1 : (dx < 0) ? -1 : 0;
      int64_t stepY = (dy > 0) ? 1 : (dy < 0) ? -1 : 0;
      ::xilinx::csl::Direction interRx, interTx;
      if (stepX == 1)       { interRx = ::xilinx::csl::Direction::WEST;  interTx = ::xilinx::csl::Direction::EAST;  }
      else if (stepX == -1) { interRx = ::xilinx::csl::Direction::EAST;  interTx = ::xilinx::csl::Direction::WEST;  }
      else if (stepY == 1)  { interRx = ::xilinx::csl::Direction::NORTH; interTx = ::xilinx::csl::Direction::SOUTH; }
      else                  { interRx = ::xilinx::csl::Direction::SOUTH; interTx = ::xilinx::csl::Direction::NORTH; }
      int64_t ix = fx + stepX;
      int64_t iy = fy + stepY;
      while (ix != tx || iy != ty) {
        b.create<::xilinx::csl_layout::SetColorConfigOp>(
            stream.getLoc(),
            /*color=*/colorAttr.getValue(),
            /*px=*/static_cast<uint64_t>(ix),
            /*py=*/static_cast<uint64_t>(iy),
            /*rx=*/interRx,
            /*tx=*/interTx);
        ix += stepX;
        iy += stepY;
      }
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

std::unique_ptr<Pass> xilinx::air::createCSLLowerDataflowRoutingPass() {
  return std::make_unique<CSLLowerDataflowRoutingPass>();
}
