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

// TODO(V1 limitation): does not follow memref.subview / memref.cast /
// memref.expand_shape. Args reached through view ops will be misclassified.

/// Check whether `arg` is used as the memref operand of a memref.load anywhere
/// in its use-list. V1: direct uses only; does not follow view-like ops.
static bool isLoadSource(BlockArgument arg) {
  for (Operation *user : arg.getUsers())
    if (auto loadOp = dyn_cast<memref::LoadOp>(user))
      if (loadOp.getMemRef() == arg)
        return true;
  return false;
}

/// Check whether `arg` is used as the memref operand of a memref.store anywhere
/// in its use-list. V1: direct uses only; does not follow view-like ops.
static bool isStoreTarget(BlockArgument arg) {
  for (Operation *user : arg.getUsers())
    if (auto storeOp = dyn_cast<memref::StoreOp>(user))
      if (storeOp.getMemRef() == arg)
        return true;
  return false;
}

/// Return true if the element type is supported by CSL (f32, f16, i32, i16).
static bool isSupportedElemType(Type ty) {
  return ty.isF32() || ty.isF16() || ty.isInteger(32) || ty.isInteger(16);
}

/// Validate that a herd's size operands are both constant 1.
static LogicalResult validateHerdSize(xilinx::air::HerdOp herd) {
  OperandRange sizes = herd.getSizeOperands();
  for (Value sz : sizes) {
    auto cstOp = sz.getDefiningOp<arith::ConstantIndexOp>();
    if (!cstOp || cstOp.value() != 1) {
      return herd->emitOpError("only 1x1 herds supported in this milestone");
    }
  }
  return success();
}

/// Validate all milestone-scope constraints for the herd body.
static LogicalResult validateHerdBody(xilinx::air::HerdOp herd) {
  // Reject dma_memcpy_nd inside herd
  auto walkResult = herd.walk([](xilinx::air::DmaMemcpyNdOp) {
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return herd->emitOpError("dma_memcpy_nd not yet supported in herd bodies");

  // Reject async operations (air.execute)
  walkResult = herd.walk([](xilinx::air::ExecuteOp) {
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return herd->emitOpError("async operations not supported in herd bodies");

  // Validate kernel arguments: static shapes, supported elem types
  for (BlockArgument arg : herd.getKernelArguments()) {
    auto memTy = dyn_cast<MemRefType>(arg.getType());
    if (!memTy)
      continue;
    for (int64_t dim : memTy.getShape()) {
      if (ShapedType::isDynamic(dim))
        return herd->emitOpError("kernel memrefs must be statically shaped");
    }
    if (!isSupportedElemType(memTy.getElementType()))
      return herd->emitOpError("unsupported element type");
  }

  return success();
}

/// Lower a func.func containing air.launch > air.herd to a top-level
/// csl.wafer. The original func.func is erased on success.
static LogicalResult lowerFuncToWafer(FuncOp func) {
  SmallVector<xilinx::air::LaunchOp> launches;
  func.walk([&](xilinx::air::LaunchOp op) { launches.push_back(op); });
  if (launches.empty())
    return success();

  if (launches.size() > 1) {
    func.emitError("-air-to-csl: multiple air.launch ops per func are not yet "
                   "supported");
    return failure();
  }

  // Reject channel ops anywhere in the func.
  auto channelWalk = func.walk([](xilinx::air::ChannelPutOp) {
    return WalkResult::interrupt();
  });
  if (channelWalk.wasInterrupted()) {
    func->emitOpError("inter-PE channels not yet supported");
    return failure();
  }
  channelWalk = func.walk([](xilinx::air::ChannelGetOp) {
    return WalkResult::interrupt();
  });
  if (channelWalk.wasInterrupted()) {
    func->emitOpError("inter-PE channels not yet supported");
    return failure();
  }

  MLIRContext *ctx = func.getContext();
  OpBuilder moduleBuilder(ctx);
  moduleBuilder.setInsertionPointAfter(func);

  // V1: handle a single launch per func.
  auto launch = launches.front();

  // Check for multiple herds under this launch.
  SmallVector<xilinx::air::HerdOp> herds;
  launch.walk([&](xilinx::air::HerdOp h) { herds.push_back(h); });
  if (herds.size() > 1) {
    launch->emitOpError("multiple herds not yet supported");
    return failure();
  }

  // Find the (single) herd under this launch.
  xilinx::air::HerdOp herd;
  if (!herds.empty())
    herd = herds.front();
  if (!herd)
    return success();

  // Validate herd size and body.
  if (failed(validateHerdSize(herd)))
    return failure();
  if (failed(validateHerdBody(herd)))
    return failure();

  Location loc = herd.getLoc();
  ArrayRef<BlockArgument> kArgs = herd.getKernelArguments();
  unsigned nArgs = kArgs.size();

  // Classify each kernel arg: an arg may be read, written, or both (RMW).
  SmallVector<bool, 4> isRead(nArgs, false);
  SmallVector<bool, 4> isWritten(nArgs, false);
  for (unsigned i = 0; i < nArgs; ++i) {
    isRead[i] = isLoadSource(kArgs[i]);
    isWritten[i] = isStoreTarget(kArgs[i]);
  }

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
  // Only emit constants if ids or sizes are actually used in the herd body.
  bool needZero = false, needOne = false;
  for (BlockArgument id : herd.getIds())
    if (!id.use_empty()) {
      needZero = true;
      break;
    }
  for (BlockArgument sz : herd.getSize())
    if (!sz.use_empty()) {
      needOne = true;
      break;
    }

  Value zeroIdx, oneIdx;
  if (needZero)
    zeroIdx = arith::ConstantIndexOp::create(fb, loc, 0);
  if (needOne)
    oneIdx = arith::ConstantIndexOp::create(fb, loc, 1);
  for (BlockArgument id : herd.getIds())
    if (!id.use_empty())
      mapping.map(id, zeroIdx);
  for (BlockArgument sz : herd.getSize())
    if (!sz.use_empty())
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

  // csl_host.memcpy_h2d for read args (before launch) — includes RMW args
  for (unsigned i = 0; i < nArgs; ++i) {
    if (!isRead[i])
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
  // csl_host.memcpy_d2h for written args (after launch) — includes RMW args
  for (unsigned i = 0; i < nArgs; ++i) {
    if (!isWritten[i])
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
