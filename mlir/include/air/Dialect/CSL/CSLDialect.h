//===- CSLDialect.h - CSL dialect declaration ------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_DIALECT_H
#define CSL_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx {
namespace csl {

class ColorType : public mlir::Type::TypeBase<ColorType, mlir::Type,
                                               mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.color";
};

class DsdType
    : public mlir::Type::TypeBase<DsdType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.dsd";
};

class ImportedModuleType
    : public mlir::Type::TypeBase<ImportedModuleType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.imported_module";
};

} // namespace csl
} // namespace xilinx

#include "air/Dialect/CSL/CSLOpsDialect.h.inc"

#endif // CSL_DIALECT_H
