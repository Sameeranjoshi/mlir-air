//===- CSLLowerStreamData.cpp ---------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Pass 4 of csl-streams-to-csl pipeline.
//
// Pre:  Stage-3 form (set_color_configs in place; streams + put/get still
//       present).
// Post: no csl.stream.put/get/csl_layout.stream remain. Each program has
//       fabric DSDs + tasks + async builtin calls.
//
// Per-program task-id counter starts at 8 (tutorial idiom).
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLLowerStreamDataPass.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace {

// Find the csl.layout child of the enclosing csl.wafer.
static ::xilinx::csl::LayoutOp findEnclosingLayout(Operation *op) {
  auto wafer = op->getParentOfType<::xilinx::csl::WaferOp>();
  if (!wafer)
    return nullptr;
  for (Operation &child : wafer.getBody().front())
    if (auto l = dyn_cast<::xilinx::csl::LayoutOp>(&child))
      return l;
  return nullptr;
}

class CSLLowerStreamDataPass
    : public PassWrapper<CSLLowerStreamDataPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-lower-stream-data"; }
  StringRef getDescription() const final {
    return "Pass 4: expand csl.stream.put/get into fabric DSDs + tasks + "
           "async builtins; erase csl_layout.stream.";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::xilinx::csl::CSLDialect,
                    ::xilinx::csl_layout::CSLLayoutDialect>();
  }
  void runOnOperation() override;

private:
  LogicalResult expandPut(::xilinx::csl::StreamPutOp put, int32_t &nextTaskId);
  LogicalResult expandGet(::xilinx::csl::StreamGetOp get, int32_t &nextTaskId);
};

} // namespace

LogicalResult
CSLLowerStreamDataPass::expandPut(::xilinx::csl::StreamPutOp put,
                                  int32_t &nextTaskId) {
  auto layout = findEnclosingLayout(put);
  if (!layout) {
    put.emitOpError("could not find enclosing csl.layout");
    return failure();
  }
  auto stream = dyn_cast_or_null<::xilinx::csl_layout::StreamOp>(
      SymbolTable::lookupSymbolIn(layout, put.getStreamAttr().getAttr()));
  if (!stream) {
    put.emitOpError("could not resolve stream '@") << put.getStream() << "'";
    return failure();
  }
  auto colorAttr = stream.getColorAttr();
  if (!colorAttr) {
    put.emitOpError(
        "stream has no color (run --csl-materialize-stream-colors first)");
    return failure();
  }

  int32_t id = nextTaskId++;
  std::string taskName =
      (stream.getSymName() + "_put_done_" + Twine(id - 8)).str();

  // Build mem and fab DSDs immediately before the put.
  OpBuilder b(put);
  Location loc = put.getLoc();
  auto dsdTy = ::xilinx::csl::DsdType::get(b.getContext());
  auto srcDsd =
      b.create<::xilinx::csl::GetMemDsdOp>(loc, dsdTy, put.getSource());
  auto outDsd = b.create<::xilinx::csl::GetFabDsdOp>(
      loc, dsdTy,
      /*direction=*/::xilinx::csl::FabDsdDirection::fabout,
      /*color=*/colorAttr.getValue(),
      /*extent=*/put.getExtent());

  // Insert task at end of program body (sibling to csl.func). csl.program is
  // NoTerminator so end() is the right insertion point.
  auto program = put->getParentOfType<::xilinx::csl::ProgramOp>();
  Block &programBody = program.getBody().front();
  OpBuilder pb(&programBody, programBody.end());
  auto task = pb.create<::xilinx::csl::TaskOp>(
      loc,
      /*sym_name=*/pb.getStringAttr(taskName),
      /*trigger_kind=*/pb.getStringAttr("local_task_id"),
      /*id=*/pb.getI32IntegerAttr(id),
      /*color=*/FlatSymbolRefAttr());
  // Build the task body.
  Block &taskBody = task.getBody().emplaceBlock();
  OpBuilder bb(&taskBody, taskBody.begin());
  bb.create<::xilinx::csl::BuiltinCallOp>(
      loc,
      /*results=*/TypeRange{},
      /*callee=*/bb.getStringAttr("unblock_cmd_stream"),
      /*module=*/Value{},
      /*args=*/ValueRange{},
      /*async=*/UnitAttr{},
      /*activate=*/FlatSymbolRefAttr{});
  bb.create<::xilinx::csl::ReturnOp>(loc);

  // Replace the put with the async builtin call.
  auto activateRef = FlatSymbolRefAttr::get(b.getContext(), taskName);
  b.create<::xilinx::csl::BuiltinCallOp>(
      loc,
      /*results=*/TypeRange{},
      /*callee=*/b.getStringAttr("fmovs"),
      /*module=*/Value{},
      /*args=*/ValueRange{outDsd.getResult(), srcDsd.getResult()},
      /*async=*/UnitAttr::get(b.getContext()),
      /*activate=*/activateRef);

  put.erase();
  return success();
}

LogicalResult
CSLLowerStreamDataPass::expandGet(::xilinx::csl::StreamGetOp get,
                                  int32_t &nextTaskId) {
  auto layout = findEnclosingLayout(get);
  if (!layout) {
    get.emitOpError("could not find enclosing csl.layout");
    return failure();
  }
  auto stream = dyn_cast_or_null<::xilinx::csl_layout::StreamOp>(
      SymbolTable::lookupSymbolIn(layout, get.getStreamAttr().getAttr()));
  if (!stream) {
    get.emitOpError("could not resolve stream '@") << get.getStream() << "'";
    return failure();
  }
  auto colorAttr = stream.getColorAttr();
  if (!colorAttr) {
    get.emitOpError(
        "stream has no color (run --csl-materialize-stream-colors first)");
    return failure();
  }

  int32_t id = nextTaskId++;
  std::string taskName =
      (stream.getSymName() + "_get_done_" + Twine(id - 8)).str();

  OpBuilder b(get);
  Location loc = get.getLoc();
  auto dsdTy = ::xilinx::csl::DsdType::get(b.getContext());
  auto tgtDsd =
      b.create<::xilinx::csl::GetMemDsdOp>(loc, dsdTy, get.getTarget());
  auto inDsd = b.create<::xilinx::csl::GetFabDsdOp>(
      loc, dsdTy,
      /*direction=*/::xilinx::csl::FabDsdDirection::fabin,
      /*color=*/colorAttr.getValue(),
      /*extent=*/get.getExtent());

  auto program = get->getParentOfType<::xilinx::csl::ProgramOp>();
  Block &programBody = program.getBody().front();
  OpBuilder pb(&programBody, programBody.end());
  auto task = pb.create<::xilinx::csl::TaskOp>(
      loc,
      /*sym_name=*/pb.getStringAttr(taskName),
      /*trigger_kind=*/pb.getStringAttr("local_task_id"),
      /*id=*/pb.getI32IntegerAttr(id),
      /*color=*/FlatSymbolRefAttr());
  Block &taskBody = task.getBody().emplaceBlock();
  OpBuilder bb(&taskBody, taskBody.begin());
  bb.create<::xilinx::csl::BuiltinCallOp>(
      loc, TypeRange{}, bb.getStringAttr("unblock_cmd_stream"), Value{},
      ValueRange{}, UnitAttr{}, FlatSymbolRefAttr{});
  bb.create<::xilinx::csl::ReturnOp>(loc);

  // For get: target DSD is destination (first arg), in_dsd is source (second).
  auto activateRef = FlatSymbolRefAttr::get(b.getContext(), taskName);
  b.create<::xilinx::csl::BuiltinCallOp>(
      loc, TypeRange{}, b.getStringAttr("fmovs"), Value{},
      ValueRange{tgtDsd.getResult(), inDsd.getResult()},
      UnitAttr::get(b.getContext()), activateRef);

  get.erase();
  return success();
}

void CSLLowerStreamDataPass::runOnOperation() {
  bool failed = false;

  // Per-program task-id counter (restarts at 8 per program).
  getOperation()->walk([&](::xilinx::csl::ProgramOp program) {
    int32_t nextTaskId = 8;
    // Snapshot put/get ops; mutating during walk is unsafe.
    SmallVector<::xilinx::csl::StreamPutOp> puts;
    SmallVector<::xilinx::csl::StreamGetOp> gets;
    program->walk([&](Operation *op) {
      if (auto p = dyn_cast<::xilinx::csl::StreamPutOp>(op))
        puts.push_back(p);
      else if (auto g = dyn_cast<::xilinx::csl::StreamGetOp>(op))
        gets.push_back(g);
    });
    for (auto p : puts)
      if (::mlir::failed(expandPut(p, nextTaskId)))
        failed = true;
    for (auto g : gets)
      if (::mlir::failed(expandGet(g, nextTaskId)))
        failed = true;
  });

  if (failed) {
    signalPassFailure();
    return;
  }

  // Erase all csl_layout.stream ops (no longer needed).
  SmallVector<::xilinx::csl_layout::StreamOp> streams;
  getOperation()->walk(
      [&](::xilinx::csl_layout::StreamOp s) { streams.push_back(s); });
  for (auto s : streams)
    s.erase();
}

std::unique_ptr<Pass> xilinx::air::createCSLLowerStreamDataPass() {
  return std::make_unique<CSLLowerStreamDataPass>();
}
