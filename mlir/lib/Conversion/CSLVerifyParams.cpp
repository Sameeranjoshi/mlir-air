//===- CSLVerifyParams.cpp - Validate layout.place vs program params ------===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Implements the -csl-verify-params pass.
//
// Checks performed (per csl.wafer):
//
//   (1) csl_layout.place bindings must match csl.program param names.
//       For each csl_layout.place op in the wafer, look up the referenced
//       csl.program and collect its parameter bindings from two sources:
//         - top-level attributes on the place op (point-form bindings); and
//         - entries in the `params` dict attribute (range-form bindings).
//       Seven reserved names are skipped when scanning top-level attrs:
//       `prog`, `px`, `py`, `x_range`, `y_range`, `iv_names`, `params`.
//       The collected binding names must match the program's declared
//       `param_names` exactly; any mismatch is reported as an op error.
//
//   (2) func.call callees must resolve to a func.func in the same program.
//
//   (3) Range-form placements' params names must match program block args
//       (this is a strict subset of (1), but gives a tailored error message
//       that names the offending param and the program).
//
//   (4) Equal sharding: for each csl_host.memcpy_h2d / memcpy_d2h, the total
//       host-buffer element count must be divisible by the PE placement
//       extent (w * h) of the program owning the referenced var.
//
//   (5) func.func ops sitting at csl.program scope must be `private` —
//       non-private helpers are rejected.
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/CSLVerifyParamsPass.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLHostOps.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

/// Lookup the PlaceOp that binds the program named `progSym` onto the grid
/// (duplicated from CSLHostEmitter.cpp to avoid crossing the Conversion ->
/// Targets layer; the logic is small and stable).
static xilinx::csl_layout::PlaceOp
findPlaceForProgram(xilinx::csl::WaferOp wafer, StringRef progSym) {
  xilinx::csl_layout::PlaceOp match;
  wafer.walk([&](xilinx::csl_layout::PlaceOp p) {
    if (match)
      return;
    if (progSym.empty() || p.getProg() == progSym)
      match = p;
  });
  return match;
}

/// Resolve the program that contains the var or func named `leaf` (e.g. `@a`).
/// Returns the empty StringRef if no match.
static StringRef findProgramForLeaf(xilinx::csl::WaferOp wafer, StringRef leaf) {
  StringRef found;
  wafer.walk([&](xilinx::csl::ProgramOp prog) {
    if (!found.empty())
      return;
    prog.getBody().walk([&](Operation *op) {
      if (!found.empty())
        return;
      if (auto v = dyn_cast<xilinx::csl::VarOp>(op)) {
        if (v.getSymName() == leaf)
          found = prog.getSymName();
      } else if (auto f = dyn_cast<xilinx::csl::FuncOp>(op)) {
        if (f.getSymName() == leaf)
          found = prog.getSymName();
      }
    });
  });
  return found;
}

/// Compute (w, h) PE extent from a PlaceOp. Point form is (1, 1); range form
/// uses (hi - lo) for each range.
static std::pair<int64_t, int64_t>
placeExtent(xilinx::csl_layout::PlaceOp place) {
  int64_t w = 1, h = 1;
  if (!place)
    return {w, h};
  if (place.getPx().has_value())
    return {w, h}; // Point form: 1x1.
  if (auto xr = place.getXRange()) {
    int64_t xlo = cast<IntegerAttr>((*xr)[0]).getInt();
    int64_t xhi = cast<IntegerAttr>((*xr)[1]).getInt();
    w = xhi - xlo;
  }
  if (auto yr = place.getYRange()) {
    int64_t ylo = cast<IntegerAttr>((*yr)[0]).getInt();
    int64_t yhi = cast<IntegerAttr>((*yr)[1]).getInt();
    h = yhi - ylo;
  }
  return {w, h};
}

/// Leaf reference name of a SymbolRefAttr (last nested ref, else root).
static StringRef leafRef(SymbolRefAttr sym) {
  if (sym.getNestedReferences().empty())
    return sym.getRootReference().getValue();
  return sym.getNestedReferences().back().getRootReference().getValue();
}

struct CSLVerifyParamsPass
    : public PassWrapper<CSLVerifyParamsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLVerifyParamsPass)

  StringRef getArgument() const override { return "csl-verify-params"; }
  StringRef getDescription() const override {
    return "Verify csl_layout.place, func.call, memcpy sharding, and "
           "private helpers in csl.wafer";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xilinx::csl::CSLDialect,
                    xilinx::csl_layout::CSLLayoutDialect,
                    xilinx::csl_host::CSLHostDialect,
                    mlir::func::FuncDialect>();
  }

  /// (1) + (3): Validate csl_layout.place attribute bindings against the
  /// referenced program's `param_names`.
  void checkPlaceBindings(xilinx::csl_layout::PlaceOp place,
                          xilinx::csl::WaferOp wafer, bool &anyFailure) {
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

    // Check point-form top-level attrs.
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
      if (name == "prog" || name == "px" || name == "py" ||
          name == "x_range" || name == "y_range" || name == "iv_names" ||
          name == "params")
        continue;
      reportExtra(name);
    }

    // (3) Check range-form params dict separately with a tailored error
    // message referencing the program by symbol.
    if (auto paramsDict = place.getParams()) {
      for (NamedAttribute entry : *paramsDict) {
        StringRef name = entry.getName();
        if (!declaredParams.contains(name)) {
          place.emitOpError("passes parameter '")
              << name << "' but @" << progName << " has no block arg '"
              << name << "'";
          anyFailure = true;
        }
      }
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
  }

  /// (2) Every func.call inside a csl.program must target a func.func that
  /// lives in the same program (csl.program is a SymbolTable, so we can use
  /// SymbolTable::lookupSymbolIn to scope the lookup).
  void checkCalleeResolution(xilinx::csl::ProgramOp program,
                             bool &anyFailure) {
    program.walk([&](func::CallOp c) {
      Operation *callee =
          SymbolTable::lookupSymbolIn(program, c.getCalleeAttr());
      if (!callee || !isa<func::FuncOp>(callee)) {
        c.emitError() << "func.call references unknown callee @"
                      << c.getCallee();
        anyFailure = true;
      }
    });
  }

  /// (4) Equal sharding for csl_host.memcpy_{h2d,d2h}: total buffer element
  /// count must divide evenly by (w * h) of the program placement.
  void checkEqualSharding(Operation *op, xilinx::csl::WaferOp wafer,
                          bool &anyFailure) {
    MemRefType memTy;
    SymbolRefAttr sym;
    if (auto h2d = dyn_cast<xilinx::csl_host::MemcpyH2DOp>(op)) {
      memTy = dyn_cast<MemRefType>(h2d.getSrc().getType());
      sym = h2d.getSym();
    } else if (auto d2h = dyn_cast<xilinx::csl_host::MemcpyD2HOp>(op)) {
      memTy = dyn_cast<MemRefType>(d2h.getDst().getType());
      sym = d2h.getSym();
    } else {
      return;
    }
    if (!memTy || !sym)
      return;

    int64_t total = 1;
    for (int64_t d : memTy.getShape())
      total *= d;

    StringRef leaf = leafRef(sym);
    StringRef progSym = findProgramForLeaf(wafer, leaf);
    xilinx::csl_layout::PlaceOp place = findPlaceForProgram(wafer, progSym);
    if (!place)
      return; // Reported elsewhere, or single-program with no placement.
    auto [w, h] = placeExtent(place);
    if (w * h <= 0)
      return;

    if (total % (w * h) != 0) {
      op->emitError() << op->getName().stripDialect() << " buffer has "
                      << total << " elements, not divisible by " << (w * h)
                      << "-PE placement";
      anyFailure = true;
    }
  }

  /// (5) func.func at csl.program scope must be `private`.
  void checkPrivateHelpers(xilinx::csl::ProgramOp program,
                           bool &anyFailure) {
    // Only walk the immediate children of the program body: func.func inside
    // a csl.func body would be nested deeper and is not a program-level
    // helper. In practice func.func can only appear at the program's top
    // level (it's module-symbol-ish) but we keep the scope tight to avoid
    // surprises.
    for (Operation &op : program.getBody().front()) {
      auto f = dyn_cast<func::FuncOp>(&op);
      if (!f)
        continue;
      if (!f.isPrivate()) {
        f.emitError() << "csl.program helpers must be 'private' func.func; @"
                      << f.getSymName() << " is not private";
        anyFailure = true;
      }
    }
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool anyFailure = false;

    mod.walk([&](xilinx::csl::WaferOp wafer) {
      // (1) + (3): place bindings vs program param_names.
      wafer.walk([&](xilinx::csl_layout::PlaceOp place) {
        checkPlaceBindings(place, wafer, anyFailure);
      });

      // (2) + (5): callee resolution and private-helper checks per program.
      wafer.walk([&](xilinx::csl::ProgramOp program) {
        checkCalleeResolution(program, anyFailure);
        checkPrivateHelpers(program, anyFailure);
      });

      // (4) Equal sharding for host memcpy ops.
      wafer.walk([&](Operation *op) {
        if (isa<xilinx::csl_host::MemcpyH2DOp,
                xilinx::csl_host::MemcpyD2HOp>(op))
          checkEqualSharding(op, wafer, anyFailure);
      });
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
