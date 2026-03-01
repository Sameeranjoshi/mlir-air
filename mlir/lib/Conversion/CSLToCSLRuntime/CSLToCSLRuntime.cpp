//===- CSLToCSLRuntime.cpp - Lower CSL to csl_rt dialect -------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Conversion pass: CSL dialect (spatial/semantic) → csl_rt (runtime) dialect.
// Implements minimal path lowering for one spatial_placement with one code_region.
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/Passes.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeDialect.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Conversion pattern: lower csl.spatial_placement → sequence of csl_rt ops.
/// Minimal path: one code_region, one place, multiple set_param_all and export_name.
struct SpatialPlacementConversionPattern
    : public OpConversionPattern<xilinx::csl::SpatialPlacementOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(xilinx::csl::SpatialPlacementOp op,
                                 OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto &body = op.getBodyRegion();

    if (body.empty()) {
      return failure();
    }

    // Scan the body to find:
    // - One code_region op (mandatory)
    // - One place op (mandatory)
    // - Zero or more set_param_all ops
    // - Zero or more export_name ops

    Block &block = body.front();
    xilinx::csl::CodeRegionOp codeRegionOp = nullptr;
    xilinx::csl::PlaceOp placeOp = nullptr;
    SmallVector<xilinx::csl::SetParamAllOp, 4> setParamOps;
    SmallVector<xilinx::csl::ExportNameOp, 4> exportNameOps;

    // TODO: Handle ports, streams, dataflow - for now only simple path
    for (auto &op : block) {
      if (auto crOp = dyn_cast<xilinx::csl::CodeRegionOp>(op)) {
        if (codeRegionOp)
          return failure(); // Multiple code regions not supported
        codeRegionOp = crOp;
      } else if (auto pOp = dyn_cast<xilinx::csl::PlaceOp>(op)) {
        if (placeOp)
          return failure(); // Multiple places not supported
        placeOp = pOp;
      } else if (auto spOp = dyn_cast<xilinx::csl::SetParamAllOp>(op)) {
        setParamOps.push_back(spOp);
      } else if (auto enOp = dyn_cast<xilinx::csl::ExportNameOp>(op)) {
        exportNameOps.push_back(enOp);
      } else if (!isa<xilinx::csl::ColorOp, xilinx::csl::RouteOp, xilinx::csl::KernelOp>(op)) {
        // Allow only these semantic ops; anything else is unsupported
        return failure();
      }
    }

    if (!codeRegionOp || !placeOp) {
      return failure();
    }

    // Extract kernel source file from place op (bound kernel).
    // For now, default to "pe.csl" if not found.
    StringAttr kernelSource = StringAttr::get(op.getContext(), "pe.csl");
    if (auto kernelOp = placeOp.getKernel().getDefiningOp<xilinx::csl::KernelOp>()) {
      kernelSource = kernelOp.getSourceFileAttr();
    }

    // Start building csl_rt ops.
    // 1. create_layout
    auto layoutOp = rewriter.create<xilinx::csl_rt::CreateLayoutOp>(
        loc, xilinx::csl_rt::LayoutType::get(op.getContext()));

    // 2. create_code_region
    auto regionOp = rewriter.create<xilinx::csl_rt::CreateCodeRegionOp>(
        loc, xilinx::csl_rt::CodeRegionType::get(op.getContext()), layoutOp.getLayout(),
        kernelSource,
        StringAttr::get(op.getContext(), "main"), // Default region name
        IntegerAttr::get(IndexType::get(op.getContext()), 16), // width
        IntegerAttr::get(IndexType::get(op.getContext()), 16)  // height
    );

    // 3. place
    auto placedOp = rewriter.create<xilinx::csl_rt::PlaceOp>(
        loc, xilinx::csl_rt::CodeRegionType::get(op.getContext()), regionOp.getCodeRegion(),
        IntegerAttr::get(IndexType::get(op.getContext()), 0), // x
        IntegerAttr::get(IndexType::get(op.getContext()), 0)  // y
    );

    Value lastValue = placedOp.getResult();

    // 4. set_param_all (if any)
    for (auto paramOp : setParamOps) {
      auto paramSetOp = rewriter.create<xilinx::csl_rt::SetParamAllOp>(
          loc, xilinx::csl_rt::CodeRegionType::get(op.getContext()), lastValue,
          paramOp.getParamNameAttr(),
          dyn_cast<mlir::IntegerAttr>(paramOp.getProperties().value));
      lastValue = paramSetOp.getResult();
    }

    // 5. export_name (if any)
    for (auto expOp : exportNameOps) {
      rewriter.create<xilinx::csl_rt::ExportNameOp>(
          loc, xilinx::csl_rt::LayoutType::get(op.getContext()), layoutOp.getLayout(),
          dyn_cast<mlir::StringAttr>(expOp.getProperties().sym_name),
          dyn_cast<mlir::StringAttr>(expOp.getProperties().type));
      // Use layout result from export for next export
    }

    // 6. compile
    rewriter.create<xilinx::csl_rt::CompileOp>(
        loc, xilinx::csl_rt::CompileArtifactsType::get(op.getContext()),
        layoutOp.getLayout());

    // Erase the original op
    rewriter.eraseOp(op);

    return success();
  }
};

/// Pass: CSL → csl_rt conversion.
class CSLToCSLRuntimePass
    : public PassWrapper<CSLToCSLRuntimePass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "csl-to-csl-rt"; }
  StringRef getDescription() const final {
    return "Convert CSL dialect to CSL Runtime dialect";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto &context = getContext();

    // Set up conversion target and patterns.
    ConversionTarget target(context);
    target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                           xilinx::csl_rt::CSLRuntimeDialect>();
    target.addIllegalOp<xilinx::csl::SpatialPlacementOp>();

    RewritePatternSet patterns(&context);
    patterns.add<SpatialPlacementConversionPattern>(&context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::air::createCSLToCSLRuntimePass() {
  return std::make_unique<CSLToCSLRuntimePass>();
}
