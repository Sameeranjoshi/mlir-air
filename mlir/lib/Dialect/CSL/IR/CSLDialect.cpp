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

using namespace mlir;

#include "air/Dialect/CSL/CSLOpsDialect.cpp.inc"

namespace xilinx::csl {

void CSLDialect::initialize() {
  addTypes<ColorType, DsdType, ImportedModuleType>();
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

  parser.emitError(parser.getNameLoc(), "unknown csl type: " + keyword);
  return Type();
}

void CSLDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<ColorType>([&](Type) { os << "color"; })
      .Case<DsdType>([&](Type) { os << "dsd"; })
      .Case<ImportedModuleType>([&](Type) { os << "imported_module"; })
      .Default([](Type) { llvm_unreachable("unexpected 'csl' type"); });
}

} // namespace xilinx::csl
