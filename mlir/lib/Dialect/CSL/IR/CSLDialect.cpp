//===- CSLDialect.cpp - CSL dialect implementation ------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "air/Dialect/CSL/CSLOpsDialect.cpp.inc"
using namespace mlir;

namespace xilinx::csl {

void CSLDialect::initialize() {
  addTypes<ColorType, DsdType, ImportedModuleType, RouteType,
           ComptimeType>();
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/CSL/CSLOps.cpp.inc"
      >();
}

Type CSLDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  if (keyword == "color")
    return ColorType::get(context);
  if (keyword == "dsd")
    return DsdType::get(context);
  if (keyword == "imported_module")
    return ImportedModuleType::get(context);
  if (keyword == "route")
    return RouteType::get(context);
  if (keyword == "comptime") {
    if (parser.parseLess())
      return Type();
    Type innerType;
    if (parser.parseType(innerType))
      return Type();
    if (parser.parseGreater())
      return Type();
    return ComptimeType::get(context, innerType);
  }

  parser.emitError(parser.getNameLoc(), "unknown csl type: " + keyword);
  return Type();
}

void CSLDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<ColorType>([&](Type) { os << "color"; })
      .Case<DsdType>([&](Type) { os << "dsd"; })
      .Case<ImportedModuleType>([&](Type) { os << "imported_module"; })
      .Case<RouteType>([&](Type) { os << "route"; })
      .Case<ComptimeType>([&](ComptimeType t) {
        os << "comptime<";
        os.printType(t.getInnerType());
        os << ">";
      })
      .Default([](Type) { llvm_unreachable("unexpected 'csl' type"); });
}

} // namespace xilinx::csl
