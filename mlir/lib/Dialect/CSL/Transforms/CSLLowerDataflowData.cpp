//===- CSLLowerDataflowData.cpp ---------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Pass 4 of csl-dataflow-to-csl pipeline.
//
// Pre:  Stage-3 form (set_color_configs in place; streams + put/get still
//       present).
// Post: no csl.dataflow.put/get/csl_layout.dataflow remain. Each program has
//       fabric DSDs + tasks + async builtin calls.
//
// Per-program task-id counter starts at 8 (tutorial idiom).
//
// Multi-op barrier: when a program has N > 1 stream ops (puts + gets), each
// completion task gets a `barrier_total = N : i32` attribute and does NOT
// call unblock_cmd_stream directly. Instead the emitter emits a countdown
// counter that calls unblock_cmd_stream only when the last transfer
// completes. The counter itself is a `csl.var @_barrier_ctr : memref<1xi16>`
// inserted at program scope, and `csl.func @compute` resets it to N at the
// start of each invocation.
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLLowerDataflowDataPass.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

class CSLLowerDataflowDataPass
    : public PassWrapper<CSLLowerDataflowDataPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-lower-dataflow-data"; }
  StringRef getDescription() const final {
    return "Pass 4: expand csl.dataflow.put/get into fabric DSDs + tasks + "
           "async builtins; erase csl_layout.dataflow.";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::xilinx::csl::CSLDialect,
                    ::xilinx::csl_layout::CSLLayoutDialect,
                    ::mlir::arith::ArithDialect,
                    ::mlir::memref::MemRefDialect>();
  }
  void runOnOperation() override;

private:
  // barrierCtr is valid (non-null) only when barrierCount > 1.
  LogicalResult expandPut(::xilinx::csl::DataflowPutOp put,
                          int32_t &nextTaskId,
                          Value barrierCtr,
                          int32_t barrierCount);
  LogicalResult expandGet(::xilinx::csl::DataflowGetOp get,
                          int32_t &nextTaskId,
                          Value barrierCtr,
                          int32_t barrierCount);
};

} // namespace

LogicalResult
CSLLowerDataflowDataPass::expandPut(::xilinx::csl::DataflowPutOp put,
                                    int32_t &nextTaskId,
                                    Value barrierCtr,
                                    int32_t barrierCount) {
  auto layout = findEnclosingLayout(put);
  if (!layout) {
    put.emitOpError("could not find enclosing csl.layout");
    return failure();
  }
  auto stream = dyn_cast_or_null<::xilinx::csl_layout::DataflowOp>(
      SymbolTable::lookupSymbolIn(layout, put.getStreamAttr().getAttr()));
  if (!stream) {
    put.emitOpError("could not resolve stream '@") << put.getStream() << "'";
    return failure();
  }
  auto colorAttr = stream.getColorAttr();
  if (!colorAttr) {
    put.emitOpError(
        "stream has no color (run --csl-materialize-dataflow-colors first)");
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

  if (barrierCount <= 1) {
    // Single-op case: call unblock_cmd_stream directly.
    bb.create<::xilinx::csl::BuiltinCallOp>(
        loc,
        /*results=*/TypeRange{},
        /*callee=*/bb.getStringAttr("unblock_cmd_stream"),
        /*module=*/Value{},
        /*args=*/ValueRange{},
        /*async=*/UnitAttr{},
        /*activate=*/FlatSymbolRefAttr{});
  }
  // Multi-op case: task body is empty (just csl.return); emitter adds
  // the countdown logic via the barrier_total attribute.
  bb.create<::xilinx::csl::ReturnOp>(loc);

  if (barrierCount > 1) {
    task->setAttr("barrier_total",
                  pb.getI32IntegerAttr(barrierCount));
  }

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
CSLLowerDataflowDataPass::expandGet(::xilinx::csl::DataflowGetOp get,
                                    int32_t &nextTaskId,
                                    Value barrierCtr,
                                    int32_t barrierCount) {
  auto layout = findEnclosingLayout(get);
  if (!layout) {
    get.emitOpError("could not find enclosing csl.layout");
    return failure();
  }
  auto stream = dyn_cast_or_null<::xilinx::csl_layout::DataflowOp>(
      SymbolTable::lookupSymbolIn(layout, get.getStreamAttr().getAttr()));
  if (!stream) {
    get.emitOpError("could not resolve stream '@") << get.getStream() << "'";
    return failure();
  }
  auto colorAttr = stream.getColorAttr();
  if (!colorAttr) {
    get.emitOpError(
        "stream has no color (run --csl-materialize-dataflow-colors first)");
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

  if (barrierCount <= 1) {
    bb.create<::xilinx::csl::BuiltinCallOp>(
        loc, TypeRange{}, bb.getStringAttr("unblock_cmd_stream"), Value{},
        ValueRange{}, UnitAttr{}, FlatSymbolRefAttr{});
  }
  bb.create<::xilinx::csl::ReturnOp>(loc);

  if (barrierCount > 1) {
    task->setAttr("barrier_total",
                  pb.getI32IntegerAttr(barrierCount));
  }

  // For get: target DSD is destination (first arg), in_dsd is source (second).
  auto activateRef = FlatSymbolRefAttr::get(b.getContext(), taskName);
  b.create<::xilinx::csl::BuiltinCallOp>(
      loc, TypeRange{}, b.getStringAttr("fmovs"), Value{},
      ValueRange{tgtDsd.getResult(), inDsd.getResult()},
      UnitAttr::get(b.getContext()), activateRef);

  get.erase();
  return success();
}

void CSLLowerDataflowDataPass::runOnOperation() {
  bool failed = false;

  // Per-program task-id counter (restarts at 8 per program).
  getOperation()->walk([&](::xilinx::csl::ProgramOp program) {
    int32_t nextTaskId = 8;

    // Snapshot put/get ops; mutating during walk is unsafe.
    SmallVector<::xilinx::csl::DataflowPutOp> puts;
    SmallVector<::xilinx::csl::DataflowGetOp> gets;
    program->walk([&](Operation *op) {
      if (auto p = dyn_cast<::xilinx::csl::DataflowPutOp>(op))
        puts.push_back(p);
      else if (auto g = dyn_cast<::xilinx::csl::DataflowGetOp>(op))
        gets.push_back(g);
    });

    int32_t totalOps = static_cast<int32_t>(puts.size() + gets.size());

    // Multi-op barrier: insert _barrier_ctr var + reset in compute().
    Value barrierCtr;
    if (totalOps > 1) {
      Block &programBody = program.getBody().front();
      Location loc = program.getLoc();

      // Create csl.var @_barrier_ctr : memref<1xi16> at the START of the
      // program body, before any existing ops.
      OpBuilder pb(&programBody, programBody.begin());
      auto i16Ty = pb.getIntegerType(16);
      auto ctrTy = MemRefType::get({1}, i16Ty);
      auto ctrVar = pb.create<::xilinx::csl::VarOp>(
          loc, ctrTy, "_barrier_ctr");
      barrierCtr = ctrVar.getResult();

      // In csl.func @compute: insert counter reset at the VERY BEGINNING of
      // the entry block. csl.func bodies CAN reference program-body SSA
      // values (no IsolatedFromAbove), so %barrierCtr is in scope.
      program.walk([&](::xilinx::csl::FuncOp func) {
        if (func.getSymName() != "compute")
          return;
        Block &funcBody = func.getBody().front();
        OpBuilder fb(&funcBody, funcBody.begin());
        auto c0 = fb.create<arith::ConstantIndexOp>(loc, 0);
        auto cN = fb.create<arith::ConstantIntOp>(loc, totalOps, 16);
        fb.create<memref::StoreOp>(loc, cN.getResult(), barrierCtr,
                                   ValueRange{c0.getResult()});
      });
    }

    for (auto p : puts)
      if (::mlir::failed(expandPut(p, nextTaskId, barrierCtr, totalOps)))
        failed = true;
    for (auto g : gets)
      if (::mlir::failed(expandGet(g, nextTaskId, barrierCtr, totalOps)))
        failed = true;
  });

  if (failed) {
    signalPassFailure();
    return;
  }

  // Erase all csl_layout.dataflow ops (no longer needed).
  SmallVector<::xilinx::csl_layout::DataflowOp> dataflows;
  getOperation()->walk(
      [&](::xilinx::csl_layout::DataflowOp s) { dataflows.push_back(s); });
  for (auto s : dataflows)
    s.erase();
}

std::unique_ptr<Pass> xilinx::air::createCSLLowerDataflowDataPass() {
  return std::make_unique<CSLLowerDataflowDataPass>();
}
