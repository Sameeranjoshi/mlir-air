//===- AIRToCSLDialect.cpp - AIR → csl.* lowering pass ----------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//
//
// Implements the -air-to-csl-dialect pass.
//
// For a 1×1 air.herd inside a func.func:
//   1. Validates milestone scope (1×1 only, no channels/DMA/async, static
//      memrefs, supported element types, single herd per launch).
//   2. Builds:
//        csl.spatial_placement {
//          %k = csl.kernel {
//            %a_buf = csl.var @a_buf : memref<256xf32>
//            %b_buf = csl.var @b_buf : memref<256xf32>
//            %c_buf = csl.var @c_buf : memref<256xf32>
//            csl.func @compute {
//              <herd body with herd args remapped to csl.var results>
//              csl.return
//            }
//            csl.export_symbol @a_buf alias("a")
//            csl.export_symbol @b_buf alias("b")
//            csl.export_symbol @c_buf alias("c")
//            csl.export_symbol @compute
//          } {source_file = "vecadd_pe.csl"} : !csl.kernel
//          %r = csl.code_region routes() colors() {
//          } {width = 1 : i64, height = 1 : i64} : !csl.code_region
//          csl.place %r %k {x = 0 : i64, y = 0 : i64}
//        }
//        csl.export_name "a" : memref<256xf32>{direction = "in"}
//        csl.export_name "b" : memref<256xf32>{direction = "in"}
//        csl.export_name "c" : memref<256xf32>{direction = "out"}
//        csl.export_name "compute" : () -> ()
//   3. Erases the original air.launch (and the entire AIR nest it contains).
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToCSLDialectPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "air-to-csl-dialect"

using namespace mlir;
using namespace mlir::func;

namespace {

//===----------------------------------------------------------------------===//
// Validation helpers
//===----------------------------------------------------------------------===//

/// Return true if the element type is one of {f32, f16, i32, i16}.
static bool isSupportedElemType(Type ty) {
  return ty.isF32() || ty.isF16() || ty.isInteger(32) || ty.isInteger(16);
}

/// Validate that a herd's size operands are both constant 1.
static LogicalResult validateHerdSize(xilinx::air::HerdOp herd) {
  OperandRange sizes = herd.getSizeOperands();
  for (Value sz : sizes) {
    auto cstOp = sz.getDefiningOp<arith::ConstantIndexOp>();
    if (!cstOp || cstOp.value() != 1) {
      return herd->emitOpError(
          "only 1x1 herds supported in this milestone");
    }
  }
  return success();
}

/// Validate all milestone scope constraints for the herd body.
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
    return herd->emitOpError(
        "async operations not supported in herd bodies");

  // Validate kernel arguments: static shapes, supported elem types
  for (BlockArgument arg : herd.getKernelArguments()) {
    auto memTy = dyn_cast<MemRefType>(arg.getType());
    if (!memTy)
      continue;
    // Check static shape
    for (int64_t dim : memTy.getShape()) {
      if (ShapedType::isDynamic(dim))
        return herd->emitOpError(
            "kernel memrefs must be statically shaped");
    }
    // Check element type
    if (!isSupportedElemType(memTy.getElementType()))
      return herd->emitOpError("unsupported element type");
  }

  return success();
}

/// Full validation for a func.func containing a single launch.
static LogicalResult validateFunc(FuncOp func) {
  // Collect launches
  SmallVector<xilinx::air::LaunchOp> launches;
  func.walk([&](xilinx::air::LaunchOp op) { launches.push_back(op); });

  if (launches.empty())
    return success(); // nothing to lower

  // Reject channel ops anywhere in the func
  auto walkResult = func.walk([](xilinx::air::ChannelPutOp) {
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return func->emitOpError(
        "inter-PE channels not yet supported");
  walkResult = func.walk([](xilinx::air::ChannelGetOp) {
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return func->emitOpError(
        "inter-PE channels not yet supported");

  for (auto launch : launches) {
    // Collect herds under this launch
    SmallVector<xilinx::air::HerdOp> herds;
    launch.walk([&](xilinx::air::HerdOp op) { herds.push_back(op); });

    if (herds.size() > 1)
      return launch->emitOpError("multiple herds not yet supported");

    for (auto herd : herds) {
      if (failed(validateHerdSize(herd)))
        return failure();
      if (failed(validateHerdBody(herd)))
        return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Lowering helpers
//===----------------------------------------------------------------------===//

/// Lower a single func.func containing air.launch → ... → air.herd
/// to the CSL spatial placement IR.
static LogicalResult lowerFunc(FuncOp func, OpBuilder &builder) {
  // Collect launches
  SmallVector<xilinx::air::LaunchOp> launches;
  func.walk([&](xilinx::air::LaunchOp op) { launches.push_back(op); });

  if (launches.empty())
    return success();

  for (auto launch : launches) {
    // Find the single herd (validated above)
    xilinx::air::HerdOp herd = nullptr;
    launch.walk([&](xilinx::air::HerdOp op) { herd = op; });
    if (!herd)
      continue;

    Location loc = launch.getLoc();

    // ---------- csl.spatial_placement { ----------
    builder.setInsertionPoint(launch);
    auto spOp = builder.create<xilinx::csl::SpatialPlacementOp>(loc);
    Block *spBlock = &spOp.getBody().emplaceBlock();
    OpBuilder spBuilder(spOp.getContext());
    spBuilder.setInsertionPointToEnd(spBlock);

    // ---------- %k = csl.kernel { ----------
    auto kernelOp = spBuilder.create<xilinx::csl::KernelOp>(
        loc,
        xilinx::csl::KernelType::get(builder.getContext()),
        /*source_file=*/spBuilder.getStringAttr("vecadd_pe.csl"),
        /*params=*/nullptr);
    Block *kernelBlock = &kernelOp.getBody().emplaceBlock();
    OpBuilder kBuilder(kernelOp.getContext());
    kBuilder.setInsertionPointToEnd(kernelBlock);

    // ---------- csl.var @a_buf / @b_buf / @c_buf ----------
    // The herd kernel arguments are the memref block args (skip tile-index args).
    ArrayRef<BlockArgument> kernelArgs = herd.getKernelArguments();
    unsigned numKernelArgs = kernelArgs.size();

    // Build export-name suffixes: use host func arg names if available.
    // Convention: first (N-1) args are "in", last arg is "out".
    // For symbol names, derive from position: a_buf, b_buf, c_buf for 3-arg case.
    // For the general case, use arg_0, arg_1, ..., arg_N patterns,
    // but the milestone always has exactly 3 memref args (a, b, c).
    // We use the position-based names matching the plan's convention.

    // Build position-based names.
    SmallVector<std::string> varNames;
    SmallVector<Value> varValues; // SSA results of csl.var ops
    for (unsigned i = 0; i < numKernelArgs; ++i) {
      // Derive symbol name from the herd argument; use "buf_N" as fallback.
      // For the 3-arg vecadd: positions 0→a_buf, 1→b_buf, 2→c_buf.
      // For the general case, use a_buf, b_buf, c_buf, d_buf, ...
      // We embed the naming convention here.
      std::string name;
      if (numKernelArgs == 3) {
        const char *names3[] = {"a_buf", "b_buf", "c_buf"};
        name = names3[i];
      } else {
        // Generic fallback: buf_0, buf_1, ...
        name = "buf_" + std::to_string(i);
      }
      varNames.push_back(name);
    }

    // Create csl.var ops for each kernel memref arg.
    for (unsigned i = 0; i < numKernelArgs; ++i) {
      Type argTy = kernelArgs[i].getType();
      auto varOp = kBuilder.create<xilinx::csl::VarOp>(
          loc, argTy, kBuilder.getStringAttr(varNames[i]));
      varValues.push_back(varOp.getResult());
    }

    // ---------- csl.func @compute { ----------
    auto funcOp = kBuilder.create<xilinx::csl::FuncOp>(
        loc, kBuilder.getStringAttr("compute"));
    Block *funcBlock = &funcOp.getBody().emplaceBlock();
    OpBuilder fBuilder(funcOp.getContext());
    fBuilder.setInsertionPointToEnd(funcBlock);

    // Clone the herd body into csl.func, remapping herd kernel args
    // to the csl.var SSA results. Since csl.func is not IsolatedFromAbove,
    // the cloned ops can reference the varValues from the enclosing kernel scope.
    IRMapping mapping;
    for (unsigned i = 0; i < numKernelArgs; ++i)
      mapping.map(kernelArgs[i], varValues[i]);

    // Map tile-index and size block arguments to safe constants (0 and 1
    // respectively) in case they are referenced in the body.
    {
      Value zeroIdx = fBuilder.create<arith::ConstantIndexOp>(loc, 0);
      for (BlockArgument id : herd.getIds())
        mapping.map(id, zeroIdx);
      Value oneIdx = fBuilder.create<arith::ConstantIndexOp>(loc, 1);
      for (BlockArgument sz : herd.getSize())
        mapping.map(sz, oneIdx);
    }

    // Clone all ops from the herd body except the herd_terminator.
    Block &herdBodyBlock = herd.getBody().front();
    for (Operation &op : herdBodyBlock) {
      if (isa<xilinx::air::HerdTerminatorOp>(op))
        continue;
      fBuilder.clone(op, mapping);
    }

    // Append csl.return
    fBuilder.create<xilinx::csl::ReturnOp>(loc);

    // ---------- csl.export_symbol inside kernel body ----------
    // Back to kernel scope: append export_symbol ops.
    kBuilder.setInsertionPointToEnd(kernelBlock);

    // Export each var with alias = its intended host name
    // Convention: "a_buf" → alias "a", "b_buf" → alias "b", "c_buf" → alias "c"
    // General: strip "_buf" suffix if present, else use full name.
    auto getAlias = [](StringRef name) -> std::string {
      if (name.ends_with("_buf"))
        return name.drop_back(4).str();
      return name.str();
    };

    for (unsigned i = 0; i < numKernelArgs; ++i) {
      std::string alias = getAlias(varNames[i]);
      kBuilder.create<xilinx::csl::ExportSymbolOp>(
          loc,
          FlatSymbolRefAttr::get(kBuilder.getContext(), varNames[i]),
          kBuilder.getStringAttr(alias));
    }
    // Export the compute function (no alias)
    kBuilder.create<xilinx::csl::ExportSymbolOp>(
        loc,
        FlatSymbolRefAttr::get(kBuilder.getContext(), "compute"),
        /*alias=*/StringAttr{});

    // ---------- csl.code_region ----------
    auto codeRegionOp = spBuilder.create<xilinx::csl::CodeRegionOp>(
        loc,
        xilinx::csl::CodeRegionType::get(builder.getContext()),
        /*region_name=*/StringAttr{},
        /*routes=*/ValueRange{},
        /*colors=*/ValueRange{},
        /*width=*/spBuilder.getI64IntegerAttr(1),
        /*height=*/spBuilder.getI64IntegerAttr(1));
    // Ensure the code_region body has a block.
    codeRegionOp.getBody().emplaceBlock();

    // ---------- csl.place ----------
    spBuilder.create<xilinx::csl::PlaceOp>(
        loc,
        codeRegionOp.getResult(),
        kernelOp.getResult(),
        spBuilder.getI64IntegerAttr(0),
        spBuilder.getI64IntegerAttr(0));

    // ---------- host-level csl.export_name ops ----------
    // Inserted after the spatial_placement, before the original return.
    builder.setInsertionPointAfter(spOp);

    // Determine direction for each func arg: first (N-1) → "in", last → "out".
    // We use the func.func's arg types (not the herd args) for the type.
    // The func's args at positions matching the launch's kernel operands give us the types.
    // For simplicity: walk the herd kernel args types, which equal the func arg types.
    for (unsigned i = 0; i < numKernelArgs; ++i) {
      Type argTy = kernelArgs[i].getType();
      std::string alias = getAlias(varNames[i]);
      bool isLast = (i == numKernelArgs - 1);
      StringAttr direction =
          isLast ? builder.getStringAttr("out") : builder.getStringAttr("in");
      builder.create<xilinx::csl::ExportNameOp>(
          loc,
          builder.getStringAttr(alias),
          TypeAttr::get(argTy),
          direction);
    }
    // Export "compute" as a function (no direction)
    builder.create<xilinx::csl::ExportNameOp>(
        loc,
        builder.getStringAttr("compute"),
        TypeAttr::get(FunctionType::get(builder.getContext(), {}, {})),
        /*direction=*/StringAttr{});

    // ---------- Erase the original air.launch ----------
    launch.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class AIRToCSLDialectPass
    : public PassWrapper<AIRToCSLDialectPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRToCSLDialectPass)

  StringRef getArgument() const final { return "air-to-csl-dialect"; }
  StringRef getDescription() const final {
    return "Lower AIR dialect (1x1 herds only) to CSL dialect ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xilinx::csl::CSLDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect funcs first to avoid walking during modification.
    SmallVector<FuncOp> funcs;
    module.walk([&](FuncOp fn) { funcs.push_back(fn); });

    for (FuncOp func : funcs) {
      // Validate first.
      if (failed(validateFunc(func))) {
        signalPassFailure();
        return;
      }
      // Lower.
      if (failed(lowerFunc(func, builder))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::air::createAIRToCSLDialectPass() {
  return std::make_unique<AIRToCSLDialectPass>();
}
