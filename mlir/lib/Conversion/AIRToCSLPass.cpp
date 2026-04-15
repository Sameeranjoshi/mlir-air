//===- AIRToCSLPass.cpp - Lower AIR to CSL v2 wafer IR --------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//
//
// Implements the -air-to-csl pass. Lowers a func.func containing
// air.launch > air.segment > air.herd (1x1 only) into a top-level
// csl.wafer with csl.program, csl.layout, csl.host siblings.
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToCSLPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLHostOps.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::func;

namespace {

/// Return true if `arg` is used as the memref operand of a memref.store.
static bool isStoreTarget(BlockArgument arg) {
  for (Operation *user : arg.getUsers())
    if (auto storeOp = dyn_cast<memref::StoreOp>(user))
      if (storeOp.getMemRef() == arg)
        return true;
  return false;
}

/// Lower a func.func containing air.launch > air.herd to a top-level
/// csl.wafer. The original func.func is erased on success.
static LogicalResult lowerFuncToWafer(FuncOp func) {
  SmallVector<xilinx::air::LaunchOp> launches;
  func.walk([&](xilinx::air::LaunchOp op) { launches.push_back(op); });
  if (launches.empty())
    return success();

  MLIRContext *ctx = func.getContext();
  OpBuilder moduleBuilder(ctx);
  moduleBuilder.setInsertionPointAfter(func);

  // V1: handle a single launch per func.
  auto launch = launches.front();

  // Find the (single) herd under this launch.
  xilinx::air::HerdOp herd;
  launch.walk([&](xilinx::air::HerdOp h) { herd = h; });
  if (!herd)
    return success();

  Location loc = herd.getLoc();
  ArrayRef<BlockArgument> kArgs = herd.getKernelArguments();
  unsigned nArgs = kArgs.size();

  // Classify each kernel arg: output if it is a store target.
  SmallVector<bool, 4> isOut(nArgs, false);
  for (unsigned i = 0; i < nArgs; ++i)
    isOut[i] = isStoreTarget(kArgs[i]);

  // Program name = herd's sym_name if present, else "pe".
  std::string progName =
      herd.getSymName().has_value() ? herd.getSymName()->str() : "pe";
  std::string layoutName = (func.getName() + "_layout").str();
  std::string funcName = func.getName().str();

  // ---- csl.wafer @<funcName> {arch = "wse3"} ----
  auto waferOp = xilinx::csl::WaferOp::create(
      moduleBuilder, loc, moduleBuilder.getStringAttr(funcName),
      moduleBuilder.getStringAttr("wse3"));
  if (waferOp.getBody().empty())
    waferOp.getBody().emplaceBlock();
  Block *waferBlock = &waferOp.getBody().front();
  OpBuilder wb(ctx);
  wb.setInsertionPointToEnd(waferBlock);

  // ---- csl.program @<progName> ----
  auto progOp = xilinx::csl::ProgramOp::create(wb, loc,
                                               wb.getStringAttr(progName));
  if (progOp.getBody().empty())
    progOp.getBody().emplaceBlock();
  Block *progBlock = &progOp.getBody().front();
  OpBuilder pb(ctx);
  pb.setInsertionPointToEnd(progBlock);

  // csl.var @argN : <type>
  SmallVector<Value, 4> varVals;
  SmallVector<std::string, 4> varNames;
  for (unsigned i = 0; i < nArgs; ++i) {
    std::string vname = "arg" + std::to_string(i);
    varNames.push_back(vname);
    auto varOp = xilinx::csl::VarOp::create(pb, loc, kArgs[i].getType(),
                                            pb.getStringAttr(vname));
    varVals.push_back(varOp.getResult());
  }

  // csl.func @compute { ... cloned herd body ... csl.return }
  auto computeOp =
      xilinx::csl::FuncOp::create(pb, loc, pb.getStringAttr("compute"));
  Block *funcBlock = &computeOp.getBody().emplaceBlock();
  OpBuilder fb(ctx);
  fb.setInsertionPointToEnd(funcBlock);

  IRMapping mapping;
  for (unsigned i = 0; i < nArgs; ++i)
    mapping.map(kArgs[i], varVals[i]);
  // Map tile id/size block args to safe constants.
  Value zeroIdx = arith::ConstantIndexOp::create(fb, loc, 0);
  Value oneIdx = arith::ConstantIndexOp::create(fb, loc, 1);
  for (BlockArgument id : herd.getIds())
    mapping.map(id, zeroIdx);
  for (BlockArgument sz : herd.getSize())
    mapping.map(sz, oneIdx);

  // Clone all ops except the herd terminator.
  for (Operation &op : herd.getBody().front()) {
    if (isa<xilinx::air::HerdTerminatorOp>(op))
      continue;
    fb.clone(op, mapping);
  }
  xilinx::csl::ReturnOp::create(fb, loc);

  // csl.export @argN {alias = "argN"} + csl.export @compute {kind = "func"}
  for (unsigned i = 0; i < nArgs; ++i) {
    xilinx::csl::ExportOp::create(
        pb, loc, FlatSymbolRefAttr::get(ctx, varNames[i]),
        /*alias=*/pb.getStringAttr(varNames[i]),
        /*kind=*/StringAttr{},
        /*direction=*/StringAttr{});
  }
  xilinx::csl::ExportOp::create(
      pb, loc, FlatSymbolRefAttr::get(ctx, "compute"),
      /*alias=*/StringAttr{},
      /*kind=*/pb.getStringAttr("func"),
      /*direction=*/StringAttr{});

  // ---- csl.layout {width = 1, height = 1} @<layoutName> ----
  auto layoutOp = xilinx::csl::LayoutOp::create(
      wb, loc, wb.getStringAttr(layoutName),
      /*width=*/wb.getI64IntegerAttr(1),
      /*height=*/wb.getI64IntegerAttr(1));
  if (layoutOp.getBody().empty())
    layoutOp.getBody().emplaceBlock();
  Block *layoutBlock = &layoutOp.getBody().front();
  OpBuilder lb(ctx);
  lb.setInsertionPointToEnd(layoutBlock);

  // csl_layout.place @<prog> at (0, 0)
  xilinx::csl_layout::PlaceOp::create(
      lb, loc, FlatSymbolRefAttr::get(ctx, progName),
      lb.getI64IntegerAttr(0), lb.getI64IntegerAttr(0));

  // csl_layout.export "argN" from @<prog>::@argN
  auto progSymAttr = StringAttr::get(ctx, progName);
  for (unsigned i = 0; i < nArgs; ++i) {
    SymbolRefAttr fromRef = SymbolRefAttr::get(
        progSymAttr, {FlatSymbolRefAttr::get(ctx, varNames[i])});
    xilinx::csl_layout::ExportOp::create(
        lb, loc, lb.getStringAttr(varNames[i]), fromRef,
        /*kind=*/StringAttr{});
  }
  // csl_layout.export "compute" from @<prog>::@compute {kind = "func"}
  {
    SymbolRefAttr fromRef = SymbolRefAttr::get(
        progSymAttr, {FlatSymbolRefAttr::get(ctx, "compute")});
    xilinx::csl_layout::ExportOp::create(
        lb, loc, lb.getStringAttr("compute"), fromRef,
        lb.getStringAttr("func"));
  }

  // ---- csl.host @<funcName>(%args...) {layout = @<layoutName>} ----
  auto hostOp = xilinx::csl::HostOp::create(
      wb, loc, wb.getStringAttr(funcName),
      FlatSymbolRefAttr::get(ctx, layoutName));
  if (hostOp.getBody().empty())
    hostOp.getBody().emplaceBlock();
  Block *hostBlock = &hostOp.getBody().front();
  for (unsigned i = 0; i < nArgs; ++i)
    hostBlock->addArgument(kArgs[i].getType(), loc);

  OpBuilder hb(ctx);
  hb.setInsertionPointToEnd(hostBlock);

  auto layoutSymAttr = StringAttr::get(ctx, layoutName);

  // csl_host.memcpy_h2d for input args
  for (unsigned i = 0; i < nArgs; ++i) {
    if (isOut[i])
      continue;
    SymbolRefAttr sym = SymbolRefAttr::get(
        layoutSymAttr, {FlatSymbolRefAttr::get(ctx, varNames[i])});
    xilinx::csl_host::MemcpyH2DOp::create(
        hb, loc, hostBlock->getArgument(i), sym,
        /*px=*/hb.getI64IntegerAttr(0), /*py=*/hb.getI64IntegerAttr(0),
        /*width=*/hb.getI64IntegerAttr(1),
        /*height=*/hb.getI64IntegerAttr(1));
  }
  // csl_host.launch @<layout>::@compute
  {
    SymbolRefAttr launchSym = SymbolRefAttr::get(
        layoutSymAttr, {FlatSymbolRefAttr::get(ctx, "compute")});
    xilinx::csl_host::LaunchOp::create(hb, loc, launchSym);
  }
  // csl_host.memcpy_d2h for output args
  for (unsigned i = 0; i < nArgs; ++i) {
    if (!isOut[i])
      continue;
    SymbolRefAttr sym = SymbolRefAttr::get(
        layoutSymAttr, {FlatSymbolRefAttr::get(ctx, varNames[i])});
    xilinx::csl_host::MemcpyD2HOp::create(
        hb, loc, sym, hostBlock->getArgument(i),
        /*px=*/hb.getI64IntegerAttr(0), /*py=*/hb.getI64IntegerAttr(0),
        /*width=*/hb.getI64IntegerAttr(1),
        /*height=*/hb.getI64IntegerAttr(1));
  }

  // Erase the original func.func (no longer needed).
  func.erase();
  return success();
}

struct AIRToCSLPass
    : public PassWrapper<AIRToCSLPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRToCSLPass)

  StringRef getArgument() const override { return "air-to-csl"; }
  StringRef getDescription() const override {
    return "Lower AIR dialect (1x1 herds) to CSL v2 wafer IR";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xilinx::csl::CSLDialect,
                    xilinx::csl_layout::CSLLayoutDialect,
                    xilinx::csl_host::CSLHostDialect,
                    arith::ArithDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<FuncOp> funcs;
    module.walk([&](FuncOp fn) { funcs.push_back(fn); });
    for (FuncOp fn : funcs) {
      if (failed(lowerFuncToWafer(fn))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::air::createAIRToCSLPass() {
  return std::make_unique<AIRToCSLPass>();
}
