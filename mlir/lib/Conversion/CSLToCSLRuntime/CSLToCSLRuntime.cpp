//===- CSLToCSLRuntime.cpp - Lower CSL to csl_rt dialect -------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Synthesizes host-side csl_rt.* runtime ops from csl.export_name ops.
//
// The pass walks each func.func containing csl.export_name ops and appends:
//   %layout  = csl_rt.create_layout
//   %art     = csl_rt.compile %layout
//   %rt      = csl_rt.runtime_create %art
//   %h1..%hN = csl_rt.memcpy_h2d chained, one per "in" export
//   %lc      = csl_rt.launch for the fn-typed export
//   %d1..%dM = csl_rt.memcpy_d2h chained, one per "out" export
//
// csl.spatial_placement, csl.kernel, csl.export_name, csl.export_symbol are
// LEFT IN PLACE — the translator walks them directly to produce layout.csl
// and pe_program.csl.
//
// Runtime lifecycle calls (load/run/stop) are HostEmitter boilerplate; they
// are not synthesized here.
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/Passes.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeDialect.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace csl    = xilinx::csl;
namespace csl_rt = xilinx::csl_rt;

namespace {

class CSLToCSLRuntimePass
    : public PassWrapper<CSLToCSLRuntimePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLToCSLRuntimePass)

  StringRef getArgument() const final { return "csl-to-csl-rt"; }
  StringRef getDescription() const final {
    return "Synthesize host-side csl_rt.* runtime sequence from csl.export_name";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<csl_rt::CSLRuntimeDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp func) { lowerFunc(func); });
  }

private:
  void lowerFunc(func::FuncOp func) {
    // Collect host-level csl.export_name ops in declaration order.
    SmallVector<csl::ExportNameOp, 4> inExports, outExports;
    csl::ExportNameOp fnExport = nullptr;

    // Walk only the direct children of the func body (not nested regions).
    for (Operation &op : func.getBody().front()) {
      auto en = dyn_cast<csl::ExportNameOp>(op);
      if (!en)
        continue;
      auto dir = en.getDirection(); // std::optional<StringRef>
      if (dir && *dir == "in")
        inExports.push_back(en);
      else if (dir && *dir == "out")
        outExports.push_back(en);
      else if (isa<FunctionType>(en.getExportedType()))
        fnExport = en;
    }

    if (inExports.empty() && outExports.empty() && !fnExport)
      return; // nothing to do

    // Insert the synthesized sequence before the func's terminator.
    Operation *term = func.getBody().front().getTerminator();
    OpBuilder b(term);
    Location loc = func.getLoc();
    MLIRContext *ctx = func.getContext();

    // 1. create_layout (no operands, returns !csl_rt.layout)
    auto layoutTy = csl_rt::LayoutType::get(ctx);
    auto layout = csl_rt::CreateLayoutOp::create(b, loc, layoutTy);

    // 2. compile (takes layout, returns !csl_rt.compile_artifacts)
    auto artTy = csl_rt::CompileArtifactsType::get(ctx);
    auto art = csl_rt::CompileOp::create(b, loc, artTy,
                                          layout.getLayout(),
                                          /*out_prefix=*/StringAttr{});

    // 3. runtime_create (takes artifacts, returns !csl_rt.runtime)
    auto rtTy = csl_rt::RuntimeType::get(ctx);
    auto rt = csl_rt::RuntimeCreateOp::create(b, loc, rtTy, art.getArtifacts());

    Value cur = rt.getRuntime();

    // 4. memcpy_h2d for each "in" export
    for (csl::ExportNameOp en : inExports) {
      auto memTy = cast<MemRefType>(en.getExportedType());
      int64_t n = memTy.getNumElements();
      auto h2d = csl_rt::MemcpyH2dOp::create(
          b, loc, rtTy, cur,
          b.getI32IntegerAttr(0),           // dest_id: placeholder
          b.getStringAttr(en.getSymName()), // src_name: buffer name
          b.getIndexAttr(0),                // px
          b.getIndexAttr(0),                // py
          b.getIndexAttr(1),                // w
          b.getIndexAttr(1),                // h
          b.getIndexAttr(n));               // elem_per_pe
      cur = h2d.getResult();
    }

    // 5. launch for the function-typed export
    if (fnExport) {
      auto launch = csl_rt::LaunchOp::create(
          b, loc, rtTy, cur,
          b.getStringAttr(fnExport.getSymName()),
          /*nonblock=*/BoolAttr{});
      cur = launch.getResult();
    }

    // 6. memcpy_d2h for each "out" export
    for (csl::ExportNameOp en : outExports) {
      auto memTy = cast<MemRefType>(en.getExportedType());
      int64_t n = memTy.getNumElements();
      auto d2h = csl_rt::MemcpyD2hOp::create(
          b, loc, rtTy, cur,
          b.getStringAttr(en.getSymName()), // dest_name: buffer name
          b.getI32IntegerAttr(0),           // src_id: placeholder
          b.getIndexAttr(0),                // px
          b.getIndexAttr(0),                // py
          b.getIndexAttr(1),                // w
          b.getIndexAttr(1),                // h
          b.getIndexAttr(n));               // elem_per_pe
      cur = d2h.getResult();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::air::createCSLToCSLRuntimePass() {
  return std::make_unique<CSLToCSLRuntimePass>();
}
