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

// SdkLayout API handle types ---------------------------------------------------

/// Opaque handle for a `CodeRegion` object created by `csl.code_region`.
class CodeRegionType
    : public mlir::Type::TypeBase<CodeRegionType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.code_region";
};

/// Opaque handle for a port (input or output) on a `CodeRegion`.
class PortType
    : public mlir::Type::TypeBase<PortType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.port";
};

/// Opaque handle for an I/O stream (H2D or D2H).
class StreamType
    : public mlir::Type::TypeBase<StreamType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.stream";
};

/// Opaque handle for a kernel (CSL source + compile-time params).
class KernelType
    : public mlir::Type::TypeBase<KernelType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.kernel";
};

/// Register the --emit-csl translation in air-translate.
void registerCSLToTextTranslation();

} // namespace csl
} // namespace xilinx

#include "air/Dialect/CSL/CSLOpsDialect.h.inc"

#endif // CSL_DIALECT_H
