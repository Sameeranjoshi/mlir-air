//===- CSLLayoutOps.cpp - csl_layout op implementations ------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "air/Dialect/CSL/CSLLayoutOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CSL v4: csl_layout.PlaceOp — two assembly forms:
//
//   Point form:  csl_layout.place @prog at (px, py) attr-dict
//   Range form:  csl_layout.place @prog over [lo:hi[:stride], lo:hi[:stride]]
//                                  [vars (%i : i32[, %j : i32])]
//                                  [params { name = %iv : i16, ... }]
//                                  attr-dict
//
// The range form also accepts a bare integer for either dim, which expands
// internally to a unit-width range `[value, value+1, 1]`.
//===----------------------------------------------------------------------===//

// Parse one axis of an `over [...]` spec: either a bare integer (expands to
// [value, value+1, 1]) or a range `lo:hi[:stride]` (stride defaults to 1).
static ParseResult parsePlaceAxis(OpAsmParser &p,
                                  SmallVectorImpl<int64_t> &out) {
  int64_t lo;
  if (p.parseInteger(lo))
    return failure();
  // If no ':' follows, this is a bare integer.
  if (failed(p.parseOptionalColon())) {
    out.push_back(lo);
    out.push_back(lo + 1);
    out.push_back(1);
    return success();
  }
  int64_t hi;
  if (p.parseInteger(hi))
    return failure();
  int64_t stride = 1;
  if (succeeded(p.parseOptionalColon())) {
    if (p.parseInteger(stride))
      return failure();
  }
  out.push_back(lo);
  out.push_back(hi);
  out.push_back(stride);
  return success();
}

// Is this range a "bare integer" axis, i.e. hi - lo == 1 and stride == 1?
static bool isSingletonAxis(ArrayAttr r) {
  if (r.size() != 3)
    return false;
  int64_t lo = cast<IntegerAttr>(r[0]).getInt();
  int64_t hi = cast<IntegerAttr>(r[1]).getInt();
  int64_t st = cast<IntegerAttr>(r[2]).getInt();
  return (hi - lo == 1) && (st == 1);
}

// Print one axis as either a bare integer or `lo:hi[:stride]`.
static void printPlaceAxis(OpAsmPrinter &p, ArrayAttr r) {
  int64_t lo = cast<IntegerAttr>(r[0]).getInt();
  int64_t hi = cast<IntegerAttr>(r[1]).getInt();
  int64_t st = cast<IntegerAttr>(r[2]).getInt();
  if (isSingletonAxis(r)) {
    p << lo;
    return;
  }
  p << lo << ':' << hi;
  if (st != 1)
    p << ':' << st;
}

mlir::ParseResult xilinx::csl_layout::PlaceOp::parse(
    mlir::OpAsmParser &p, mlir::OperationState &result) {
  // @prog
  FlatSymbolRefAttr progAttr;
  if (p.parseAttribute(progAttr, "prog", result.attributes))
    return failure();

  Builder &b = p.getBuilder();

  if (succeeded(p.parseOptionalKeyword("at"))) {
    // at ( INT , INT )
    int64_t x, y;
    if (p.parseLParen() || p.parseInteger(x) || p.parseComma() ||
        p.parseInteger(y) || p.parseRParen())
      return failure();
    result.addAttribute("px", b.getI64IntegerAttr(x));
    result.addAttribute("py", b.getI64IntegerAttr(y));
  } else if (succeeded(p.parseOptionalKeyword("over"))) {
    // over [ axis (, axis)? ]
    SmallVector<int64_t, 3> xr, yr;
    if (p.parseLSquare())
      return failure();
    if (parsePlaceAxis(p, xr))
      return failure();
    bool hasY = false;
    if (succeeded(p.parseOptionalComma())) {
      if (parsePlaceAxis(p, yr))
        return failure();
      hasY = true;
    }
    if (p.parseRSquare())
      return failure();
    result.addAttribute("x_range", b.getI64ArrayAttr(xr));
    if (hasY)
      result.addAttribute("y_range", b.getI64ArrayAttr(yr));

    // Optional `vars (%i : i32[, %j : i32])`
    if (succeeded(p.parseOptionalKeyword("vars"))) {
      if (p.parseLParen())
        return failure();
      SmallVector<Attribute> names;
      do {
        OpAsmParser::UnresolvedOperand opnd;
        if (p.parseOperand(opnd))
          return failure();
        StringRef sref = opnd.name;
        bool consumed = sref.consume_front("%");
        assert(consumed &&
               "OpAsmParser::UnresolvedOperand::name must start with '%'");
        (void)consumed;
        // Require `: i32` — enforced for roundtrip fidelity.
        if (p.parseColon())
          return failure();
        llvm::SMLoc typeLoc = p.getCurrentLocation();
        Type ty;
        if (p.parseType(ty))
          return failure();
        if (!ty.isInteger(32))
          return p.emitError(typeLoc,
                             "expected 'i32' for induction variable type");
        names.push_back(b.getStringAttr(sref));
      } while (succeeded(p.parseOptionalComma()));
      if (p.parseRParen())
        return failure();
      result.addAttribute("iv_names", b.getArrayAttr(names));
    }

    // Optional `params { name = %iv : i16, ... }`
    if (succeeded(p.parseOptionalKeyword("params"))) {
      if (p.parseLBrace())
        return failure();
      SmallVector<NamedAttribute> entries;
      if (failed(p.parseOptionalRBrace())) {
        do {
          StringRef key;
          if (p.parseKeyword(&key))
            return failure();
          if (p.parseEqual())
            return failure();
          OpAsmParser::UnresolvedOperand opnd;
          if (p.parseOperand(opnd))
            return failure();
          StringRef ivRef = opnd.name;
          bool consumed = ivRef.consume_front("%");
          assert(consumed &&
                 "OpAsmParser::UnresolvedOperand::name must start with '%'");
          (void)consumed;
          if (p.parseColon())
            return failure();
          llvm::SMLoc typeLoc = p.getCurrentLocation();
          Type ty;
          if (p.parseType(ty))
            return failure();
          if (!ty.isInteger(16))
            return p.emitError(typeLoc,
                               "expected 'i16' for params value type");
          entries.push_back(
              b.getNamedAttr(key, b.getStringAttr(ivRef)));
        } while (succeeded(p.parseOptionalComma()));
        if (p.parseRBrace())
          return failure();
      }
      result.addAttribute("params", b.getDictionaryAttr(entries));
    }
  } else {
    return p.emitError(p.getCurrentLocation(),
                       "expected 'at' or 'over' after program symbol");
  }

  return p.parseOptionalAttrDict(result.attributes);
}

void xilinx::csl_layout::PlaceOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printAttributeWithoutType(getProgAttr());

  SmallVector<StringRef> elided = {"prog",    "px",       "py",
                                   "x_range", "y_range",  "iv_names",
                                   "params"};

  if (getPx().has_value() || getPy().has_value()) {
    // Point form.
    int64_t x = getPx().value_or(0);
    int64_t y = getPy().value_or(0);
    p << " at (" << x << ", " << y << ')';
  } else if (getXRange().has_value()) {
    // Range form.
    p << " over [";
    printPlaceAxis(p, *getXRange());
    if (auto yr = getYRange()) {
      p << ", ";
      printPlaceAxis(p, *yr);
    }
    p << ']';

    if (auto ivs = getIvNames()) {
      p << " vars (";
      llvm::interleaveComma(*ivs, p.getStream(), [&](Attribute a) {
        p << '%' << cast<StringAttr>(a).getValue() << " : i32";
      });
      p << ')';
    }

    if (auto params = getParams()) {
      p << " params {";
      llvm::interleaveComma(
          *params, p.getStream(), [&](NamedAttribute e) {
            p << e.getName().getValue() << " = %"
              << cast<StringAttr>(e.getValue()).getValue() << " : i16";
          });
      p << '}';
    }
  }

  p.printOptionalAttrDict((*this)->getAttrs(), elided);
}

mlir::LogicalResult xilinx::csl_layout::PlaceOp::verify() {
  bool hasPx = getPx().has_value();
  bool hasPy = getPy().has_value();
  bool isPoint = hasPx || hasPy;
  bool isRange = getXRange().has_value() || getYRange().has_value();

  if (isPoint && isRange)
    return emitOpError("cannot mix `at (x, y)` and `over [...]` forms");
  if (!isPoint && !isRange)
    return emitOpError("must use either `at (x, y)` or `over [...]`");

  if (isPoint) {
    if (!(hasPx && hasPy))
      return emitOpError("`at (x, y)` requires both px and py");
    if (getIvNames().has_value() || getParams().has_value() ||
        getXRange().has_value() || getYRange().has_value())
      return emitOpError(
          "vars/params are only allowed with `over [...]` form");
    return success();
  }

  // Range form.
  if (!getXRange().has_value())
    return emitOpError("`over [...]` requires an x range");

  auto checkAxis = [&](ArrayAttr r, StringRef name) -> LogicalResult {
    if (r.size() != 3)
      return emitOpError(name) << " must have shape [lo, hi, stride]";
    int64_t lo = cast<IntegerAttr>(r[0]).getInt();
    int64_t hi = cast<IntegerAttr>(r[1]).getInt();
    int64_t st = cast<IntegerAttr>(r[2]).getInt();
    if (hi <= lo)
      return emitOpError(name) << " must have hi > lo";
    if (st <= 0)
      return emitOpError(name) << " stride must be positive";
    return success();
  };
  if (failed(checkAxis(*getXRange(), "x_range")))
    return failure();
  if (auto yr = getYRange())
    if (failed(checkAxis(*yr, "y_range")))
      return failure();

  if (auto ivs = getIvNames()) {
    if (ivs->empty() || ivs->size() > 2)
      return emitOpError("vars must name 1 or 2 induction variables");
    for (Attribute a : *ivs)
      if (!dyn_cast<StringAttr>(a))
        return emitOpError("iv_names entries must be string attributes");
  }

  if (auto params = getParams()) {
    llvm::DenseSet<StringRef> names;
    if (auto ivs = getIvNames())
      for (Attribute a : *ivs)
        names.insert(cast<StringAttr>(a).getValue());
    for (NamedAttribute e : *params) {
      auto ref = dyn_cast<StringAttr>(e.getValue());
      if (!ref)
        return emitOpError("params value for '")
               << e.getName().getValue()
               << "' must be a string referencing a vars induction variable";
      if (!names.contains(ref.getValue()))
        return emitOpError("params value '%")
               << ref.getValue() << "' for '" << e.getName().getValue()
               << "' is not a declared vars induction variable";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// csl_layout.stream — verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult xilinx::csl_layout::StreamOp::verify() {
  int64_t dx = getToX() - getFromX();
  int64_t dy = getToY() - getFromY();
  // Single-hop, cardinal: exactly one of |dx|, |dy| is 1, the other is 0.
  bool xOne = (dx == 1 || dx == -1);
  bool yOne = (dy == 1 || dy == -1);
  bool xZero = (dx == 0);
  bool yZero = (dy == 0);
  if (!((xOne && yZero) || (xZero && yOne)))
    return emitOpError("requires single-hop cardinal route; got delta (")
           << dx << ", " << dy << ")";

  // Optional color must resolve to a csl.color in parent csl.layout.
  if (auto colorAttr = getColorAttr()) {
    auto layout = (*this)->getParentOfType<::xilinx::csl::LayoutOp>();
    if (!layout)
      return emitOpError("must be inside csl.layout body");
    auto *color =
        mlir::SymbolTable::lookupSymbolIn(layout, colorAttr.getAttr());
    if (!color)
      return emitOpError("references undefined color '@")
             << colorAttr.getValue() << "'";
    if (!mlir::isa<::xilinx::csl::ColorOp>(color))
      return emitOpError("'@") << colorAttr.getValue()
                                << "' is not a csl.color";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// csl_layout.set_color_config — verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult xilinx::csl_layout::SetColorConfigOp::verify() {
  // Resolve the @color symbol to a csl.color in the parent csl.layout body.
  auto layout = (*this)->getParentOfType<::xilinx::csl::LayoutOp>();
  if (!layout)
    return emitOpError("must be inside csl.layout body");
  auto *color = mlir::SymbolTable::lookupSymbolIn(
      layout, getColorAttr().getAttr());
  if (!color)
    return emitOpError("references undefined color symbol '@")
           << getColorAttr().getValue() << "'";
  if (!mlir::isa<::xilinx::csl::ColorOp>(color))
    return emitOpError("'@") << getColorAttr().getValue()
                              << "' is not a csl.color";
  return mlir::success();
}
