//===- CSLVerifyParams.cpp - Validate layout.place vs program params ------===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Implements the -csl-verify-params pass.
//
// Algorithm (per csl.wafer):
//   For each csl_layout.place op in the wafer, look up the referenced
//   csl.program and collect its parameter bindings from two sources:
//     - top-level attributes on the place op (point-form bindings); and
//     - entries in the `params` dict attribute (range-form bindings).
//   Six reserved names are skipped when scanning top-level attrs:
//   `prog`, `px`, `py`, `x_range`, `y_range`, `iv_names`, `params`.
//   The collected binding names must match the program's declared
//   `param_names` exactly; any mismatch is reported as an op error.
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/CSLVerifyParamsPass.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace {

struct CSLVerifyParamsPass
    : public PassWrapper<CSLVerifyParamsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLVerifyParamsPass)

  StringRef getArgument() const override { return "csl-verify-params"; }
  StringRef getDescription() const override {
    return "Verify csl_layout.place attrs match csl.program block args";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xilinx::csl::CSLDialect,
                    xilinx::csl_layout::CSLLayoutDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool anyFailure = false;

    mod.walk([&](xilinx::csl_layout::PlaceOp place) {
      auto wafer = place->getParentOfType<xilinx::csl::WaferOp>();
      if (!wafer)
        return;

      StringRef progName = place.getProg();
      xilinx::csl::ProgramOp prog;
      wafer.getBody().walk([&](xilinx::csl::ProgramOp p) {
        if (p.getSymName() == progName)
          prog = p;
      });
      if (!prog) {
        place.emitOpError("references unknown program @") << progName;
        anyFailure = true;
        return;
      }

      // Collect declared parameter names from the referenced program.
      llvm::DenseSet<StringRef> declaredParams;
      if (auto names = prog.getParamNamesAttr()) {
        for (Attribute n : names) {
          declaredParams.insert(cast<StringAttr>(n).getValue());
        }
      }

      // Check every extra attribute on `place` is a declared parameter.
      // Range-form placements bind params via the `params` dictionary; the
      // point form binds them as top-level attrs.  Walk both.
      auto reportExtra = [&](StringRef name) {
        if (!declaredParams.contains(name)) {
          place.emitOpError("passes parameter '")
              << name << "' but @" << progName
              << " has no matching block argument";
          anyFailure = true;
        }
      };
      for (NamedAttribute attr : place->getAttrs()) {
        StringRef name = attr.getName();
        // Skip the operation's own positional arguments.
        if (name == "prog" || name == "px" || name == "py" ||
            name == "x_range" || name == "y_range" || name == "iv_names" ||
            name == "params")
          continue;
        reportExtra(name);
      }
      if (auto paramsDict = place.getParams()) {
        for (NamedAttribute entry : *paramsDict)
          reportExtra(entry.getName());
      }

      // Check every declared parameter is bound by the place op.
      auto hasParamBinding = [&](StringRef n) {
        if (place->hasAttr(n))
          return true;
        if (auto paramsDict = place.getParams())
          return paramsDict->contains(n);
        return false;
      };
      for (StringRef declared : declaredParams) {
        if (!hasParamBinding(declared)) {
          place.emitOpError("missing parameter '")
              << declared << "' required by @" << progName;
          anyFailure = true;
        }
      }
    });

    if (anyFailure)
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::air::createCSLVerifyParamsPass() {
  return std::make_unique<CSLVerifyParamsPass>();
}
