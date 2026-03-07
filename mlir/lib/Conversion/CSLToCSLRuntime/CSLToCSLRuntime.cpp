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

    Block &block = body.front();
    xilinx::csl::CodeRegionOp codeRegionOp = nullptr;
    xilinx::csl::PlaceOp placeOp = nullptr;

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
    auto layoutOp = xilinx::csl_rt::CreateLayoutOp::create(
        rewriter, loc, xilinx::csl_rt::LayoutType::get(op.getContext()));

    // 2. create_code_region
    auto regionOp = xilinx::csl_rt::CreateCodeRegionOp::create(
        rewriter, loc, xilinx::csl_rt::CodeRegionType::get(op.getContext()),
        layoutOp.getLayout(), kernelSource,
        StringAttr::get(op.getContext(), "main"), // Default region name
        rewriter.getIndexAttr(16), // width
        rewriter.getIndexAttr(16)   // height
    );

    // 3. place
    xilinx::csl_rt::PlaceOp::create(
        rewriter, loc, xilinx::csl_rt::CodeRegionType::get(op.getContext()),
        regionOp.getCodeRegion(),
        rewriter.getIndexAttr(0), // x
        rewriter.getIndexAttr(0)  // y
    );

    // 4. compile
    xilinx::csl_rt::CompileOp::create(
        rewriter, loc, xilinx::csl_rt::CompileArtifactsType::get(op.getContext()),
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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLToCSLRuntimePass)

  StringRef getArgument() const final { return "csl-to-csl-rt"; }
  StringRef getDescription() const final {
    return "Convert CSL dialect to CSL Runtime dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xilinx::csl_rt::CSLRuntimeDialect>();
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
