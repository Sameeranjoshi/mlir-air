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
// Relay PE: when a program has exactly 1 get AND 1 put with middle ops
// between them (scf.for etc.), this is a relay pattern. The pass generates:
//   - In compute(): async GET fmovs that activates @_relay_N task
//   - @_relay_N task: runs middle transform ops, then async PUT fmovs
//     that activates @put_done_M task (no IsolatedFromAbove needed now)
//   - @put_done_M task: calls unblock_cmd_stream
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLLowerDataflowDataPass.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
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
  // Relay-specific expansion: handles the get→transform→put chain.
  // activateRef is the symbol that the GET fmovs will activate.
  // When activateRef is non-null, it overrides the default get_done task name.
  LogicalResult expandGetWithActivate(::xilinx::csl::DataflowGetOp get,
                                      FlatSymbolRefAttr activateRef);
  LogicalResult expandRelayProgram(::xilinx::csl::ProgramOp program,
                                   SmallVector<::xilinx::csl::DataflowGetOp> &gets,
                                   SmallVector<::xilinx::csl::DataflowPutOp> &puts,
                                   int32_t &nextTaskId);
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

// Expand a GET op with a custom activation symbol (for relay: activates the
// relay task instead of a get_done task).
LogicalResult
CSLLowerDataflowDataPass::expandGetWithActivate(
    ::xilinx::csl::DataflowGetOp get,
    FlatSymbolRefAttr activateRef) {
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

  b.create<::xilinx::csl::BuiltinCallOp>(
      loc, TypeRange{}, b.getStringAttr("fmovs"), Value{},
      ValueRange{tgtDsd.getResult(), inDsd.getResult()},
      UnitAttr::get(b.getContext()), activateRef);

  get.erase();
  return success();
}

// Relay expansion: exactly 1 get and 1 put with middle ops between them.
// Structure generated:
//   compute(): async GET fmovs → activates @_relay_N
//   csl.task @_relay_N: cloned middle ops + async PUT fmovs → activates @_put_done_M
//   csl.task @_put_done_M: unblock_cmd_stream
LogicalResult
CSLLowerDataflowDataPass::expandRelayProgram(
    ::xilinx::csl::ProgramOp program,
    SmallVector<::xilinx::csl::DataflowGetOp> &gets,
    SmallVector<::xilinx::csl::DataflowPutOp> &puts,
    int32_t &nextTaskId) {
  assert(gets.size() == 1 && puts.size() == 1 &&
         "expandRelayProgram: expected exactly 1 get and 1 put");

  auto getOp = gets[0];
  auto putOp = puts[0];
  Location loc = program.getLoc();
  MLIRContext *ctx = program.getContext();

  // Allocate task IDs.
  int32_t relayId = nextTaskId++;
  int32_t putDoneId = nextTaskId++;
  std::string relayName = "_relay_" + std::to_string(relayId - 8);

  // Look up PUT stream name for naming the put_done task.
  auto layout = findEnclosingLayout(putOp);
  if (!layout) {
    putOp.emitOpError("could not find enclosing csl.layout");
    return failure();
  }
  auto putStream = dyn_cast_or_null<::xilinx::csl_layout::DataflowOp>(
      SymbolTable::lookupSymbolIn(layout, putOp.getStreamAttr().getAttr()));
  if (!putStream) {
    putOp.emitOpError("could not resolve stream '@") << putOp.getStream() << "'";
    return failure();
  }
  auto putColorAttr = putStream.getColorAttr();
  if (!putColorAttr) {
    putOp.emitOpError(
        "stream has no color (run --csl-materialize-dataflow-colors first)");
    return failure();
  }
  std::string putDoneName =
      (putStream.getSymName() + "_put_done_" + Twine(putDoneId - 8)).str();

  // Find compute() function.
  ::xilinx::csl::FuncOp computeFunc;
  program.walk([&](::xilinx::csl::FuncOp f) {
    if (f.getSymName() == "compute")
      computeFunc = f;
  });
  if (!computeFunc) {
    program.emitOpError("relay expansion: could not find csl.func @compute");
    return failure();
  }

  Block &computeBody = computeFunc.getBody().front();

  // Collect middle ops (between GET and PUT, exclusive).
  // Must be done before erasing anything.
  SmallVector<Operation *> middleOps;
  bool inMiddle = false;
  for (Operation &op : computeBody) {
    if (&op == getOp.getOperation()) {
      inMiddle = true;
      continue;
    }
    if (&op == putOp.getOperation())
      break;
    if (inMiddle)
      middleOps.push_back(&op);
  }

  // Create the relay task at end of program body.
  Block &programBody = program.getBody().front();
  OpBuilder pb(&programBody, programBody.end());
  auto relayTask = pb.create<::xilinx::csl::TaskOp>(
      loc,
      /*sym_name=*/pb.getStringAttr(relayName),
      /*trigger_kind=*/pb.getStringAttr("local_task_id"),
      /*id=*/pb.getI32IntegerAttr(relayId),
      /*color=*/FlatSymbolRefAttr());
  Block &relayBody = relayTask.getBody().emplaceBlock();
  OpBuilder rb(&relayBody, relayBody.begin());

  // Build an IRMapping: clone arith.constant ops from compute() into relay
  // body so that middle ops' operands (which reference these constants) can
  // be substituted. Buffer refs (%buf etc.) are NOT in the mapping, so they
  // pass through as cross-region references (valid without IsolatedFromAbove).
  IRMapping mapping;
  for (Operation &op : computeBody) {
    if (auto cst = dyn_cast<arith::ConstantOp>(&op)) {
      auto *clone = rb.clone(op, mapping);
      mapping.map(op.getResult(0), clone->getResult(0));
    }
  }

  // Clone middle ops into relay body using the mapping.
  for (Operation *op : middleOps) {
    if (isa<arith::ConstantOp>(op))
      continue; // already cloned above
    rb.clone(*op, mapping);
  }

  // Expand the PUT inside relay body: create mem+fab DSDs then async fmovs.
  {
    auto dsdTy = ::xilinx::csl::DsdType::get(ctx);

    // Use extent from PUT op; if it's in the mapping use the mapped value,
    // otherwise use the original value (it may be a cross-region ref or a
    // constant that's already in the relay body via mapping).
    Value putExtent = putOp.getExtent();
    Value mappedExtent = mapping.lookupOrNull(putExtent);
    if (!mappedExtent)
      mappedExtent = putExtent;

    auto srcDsd = rb.create<::xilinx::csl::GetMemDsdOp>(
        loc, dsdTy, putOp.getSource());
    auto outDsd = rb.create<::xilinx::csl::GetFabDsdOp>(
        loc, dsdTy,
        /*direction=*/::xilinx::csl::FabDsdDirection::fabout,
        /*color=*/putColorAttr.getValue(),
        /*extent=*/mappedExtent);

    auto putDoneRef = FlatSymbolRefAttr::get(ctx, putDoneName);
    rb.create<::xilinx::csl::BuiltinCallOp>(
        loc, TypeRange{}, rb.getStringAttr("fmovs"), Value{},
        ValueRange{outDsd.getResult(), srcDsd.getResult()},
        UnitAttr::get(ctx), putDoneRef);
  }

  // Add csl.return to relay body.
  rb.create<::xilinx::csl::ReturnOp>(loc);

  // Create put_done task (unblock_cmd_stream + return).
  {
    auto putDoneTask = pb.create<::xilinx::csl::TaskOp>(
        loc,
        /*sym_name=*/pb.getStringAttr(putDoneName),
        /*trigger_kind=*/pb.getStringAttr("local_task_id"),
        /*id=*/pb.getI32IntegerAttr(putDoneId),
        /*color=*/FlatSymbolRefAttr());
    Block &pdBody = putDoneTask.getBody().emplaceBlock();
    OpBuilder pdb(&pdBody, pdBody.begin());
    pdb.create<::xilinx::csl::BuiltinCallOp>(
        loc, TypeRange{}, pdb.getStringAttr("unblock_cmd_stream"), Value{},
        ValueRange{}, UnitAttr{}, FlatSymbolRefAttr{});
    pdb.create<::xilinx::csl::ReturnOp>(loc);
  }

  // Erase middle ops from compute() (in reverse order to preserve def-use).
  for (auto it = middleOps.rbegin(); it != middleOps.rend(); ++it)
    (*it)->erase();

  // Erase PUT from compute().
  putOp.erase();

  // Expand GET in compute() with activate = relayName.
  auto relayRef = FlatSymbolRefAttr::get(ctx, relayName);
  if (failed(expandGetWithActivate(getOp, relayRef)))
    return failure();

  return success();
}

// Resolve a symbol reference to the actual color symbol name in the layout.
// The symRef may reference either:
//   - a csl.color directly (return sym_name as-is), or
//   - a csl_layout.dataflow whose `color` attr points to the actual color.
// Returns empty string on failure.
static std::string
resolveColorNameInLayout(StringRef symRef,
                         ::xilinx::csl::LayoutOp layout) {
  if (!layout)
    return "";
  auto *sym = mlir::SymbolTable::lookupSymbolIn(layout, symRef);
  if (!sym)
    return "";
  if (auto color = mlir::dyn_cast<::xilinx::csl::ColorOp>(sym))
    return color.getSymName().str();
  if (auto df = mlir::dyn_cast<::xilinx::csl_layout::DataflowOp>(sym)) {
    if (auto colorAttr = df.getColorAttr())
      return colorAttr.getValue().str();
  }
  return "";
}

// Update data task `color` attributes that reference a stream name to the
// resolved color name. Must be called BEFORE dataflow ops are erased.
static void updateDataTaskColors(Operation *root) {
  root->walk([&](::xilinx::csl::TaskOp taskOp) {
    if (taskOp.getTriggerKind() != "data_task")
      return;
    auto colorAttr = taskOp.getColorAttr();
    if (!colorAttr)
      return;
    auto layout = findEnclosingLayout(taskOp);
    if (!layout)
      return;
    std::string resolved =
        resolveColorNameInLayout(colorAttr.getValue(), layout);
    if (!resolved.empty() && resolved != colorAttr.getValue().str()) {
      taskOp.setColorAttr(mlir::FlatSymbolRefAttr::get(
          taskOp.getContext(), resolved));
    }
  });
}

// Expand csl.dataflow.send_wavelet to a 1-element fabout DSD + sync @fmovs.
// The color is resolved from the dataflow stream symbol in the enclosing layout.
// Unlike put/get expansions, there is NO completion task and NO async — the
// wavelet is sent synchronously from within the data task body.
static LogicalResult
expandSendWavelet(::xilinx::csl::DataflowSendWaveletOp sendOp) {
  auto layout = findEnclosingLayout(sendOp);
  if (!layout) {
    sendOp.emitOpError("could not find enclosing csl.layout");
    return failure();
  }
  auto stream = dyn_cast_or_null<::xilinx::csl_layout::DataflowOp>(
      SymbolTable::lookupSymbolIn(layout, sendOp.getStreamAttr().getAttr()));
  if (!stream) {
    sendOp.emitOpError("could not resolve stream '@")
        << sendOp.getStream() << "'";
    return failure();
  }
  auto colorAttr = stream.getColorAttr();
  if (!colorAttr) {
    sendOp.emitOpError(
        "stream has no color (run --csl-materialize-dataflow-colors first)");
    return failure();
  }

  OpBuilder b(sendOp);
  Location loc = sendOp.getLoc();
  auto dsdTy = ::xilinx::csl::DsdType::get(b.getContext());

  // 1-element extent constant.
  auto c1 = b.create<arith::ConstantIndexOp>(loc, 1);

  // fabout DSD with extent=1.
  auto outDsd = b.create<::xilinx::csl::GetFabDsdOp>(
      loc, dsdTy,
      /*direction=*/::xilinx::csl::FabDsdDirection::fabout,
      /*color=*/colorAttr.getValue(),
      /*extent=*/c1.getResult());

  // Synchronous @fmovs(outDsd, value) — no async, no activate.
  b.create<::xilinx::csl::BuiltinCallOp>(
      loc,
      /*results=*/TypeRange{},
      /*callee=*/b.getStringAttr("fmovs"),
      /*module=*/Value{},
      /*args=*/ValueRange{outDsd.getResult(), sendOp.getValue()},
      /*async=*/UnitAttr{},
      /*activate=*/FlatSymbolRefAttr{});

  sendOp.erase();
  return success();
}

void CSLLowerDataflowDataPass::runOnOperation() {
  bool failed = false;

  // Resolve data task `color` attributes that may reference a stream name
  // (e.g., @ch01) to the actual color name (e.g., @ch01_color). Must be done
  // before csl_layout.dataflow ops are erased at the end of this pass.
  updateDataTaskColors(getOperation());

  // Expand csl.dataflow.send_wavelet ops first (inside data task bodies).
  // These are independent of the put/get relay detection below.
  SmallVector<::xilinx::csl::DataflowSendWaveletOp> sendWavelets;
  getOperation()->walk([&](::xilinx::csl::DataflowSendWaveletOp op) {
    sendWavelets.push_back(op);
  });
  for (auto op : sendWavelets)
    if (::mlir::failed(expandSendWavelet(op)))
      failed = true;
  if (failed) {
    signalPassFailure();
    return;
  }

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

    int32_t numPuts = static_cast<int32_t>(puts.size());
    int32_t numGets = static_cast<int32_t>(gets.size());
    int32_t totalOps = numPuts + numGets;

    // Relay pattern: exactly 1 get AND 1 put.
    if (numGets == 1 && numPuts == 1) {
      if (::mlir::failed(expandRelayProgram(program, gets, puts, nextTaskId)))
        failed = true;
      return;
    }

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
