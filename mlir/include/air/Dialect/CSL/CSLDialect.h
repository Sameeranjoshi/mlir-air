//===- CSLDialect.h - CSL dialect declaration ------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_DIALECT_H
#define CSL_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
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

/// Opaque handle for a reusable access pattern (extent, stride, offset)
/// that applies to a memref to produce a strided DSD.
class ViewType
    : public mlir::Type::TypeBase<ViewType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.view";
};

/// Opaque handle for a routing configuration.
class RouteType
    : public mlir::Type::TypeBase<RouteType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.route";
};

// ---------------------------------------------------------------------------
// v2 types
// ---------------------------------------------------------------------------

/// Storage for !csl.comptime<T> — a compile-time-only value (CSL param).
struct ComptimeTypeStorage : mlir::TypeStorage {
  using KeyTy = mlir::Type;
  explicit ComptimeTypeStorage(mlir::Type t) : innerType(t) {}
  bool operator==(const KeyTy &key) const { return innerType == key; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return mlir::hash_value(key);
  }
  static ComptimeTypeStorage *construct(mlir::TypeStorageAllocator &alloc,
                                        const KeyTy &key) {
    return new (alloc.allocate<ComptimeTypeStorage>())
        ComptimeTypeStorage(key);
  }
  mlir::Type innerType;
};

/// !csl.comptime<T> — marks a block argument of csl.program as a CSL param.
/// Maps to "param x: T;" in emitted CSL source.
class ComptimeType
    : public mlir::Type::TypeBase<ComptimeType, mlir::Type,
                                  ComptimeTypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.comptime";
  static ComptimeType get(mlir::MLIRContext *ctx, mlir::Type innerType) {
    return Base::get(ctx, innerType);
  }
  mlir::Type getInnerType() const { return getImpl()->innerType; }
};

/// Individual translation registrations (one per output).
/// Defined in mlir/lib/Targets/CSLEmit/.
void registerCSLProgramTranslation();
void registerCSLLayoutTranslation();
void registerCSLHostTranslation();

/// Unified --emit-csl registration (writes all three files into --output-dir).
/// Defined in mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp.
void registerCSLEmitAllTranslation();

/// Register all four translations (program, layout, host, emit-all) at once.
/// Defined in mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp.
void registerCSLEmitTranslations();

} // namespace csl
} // namespace xilinx

#include "air/Dialect/CSL/CSLOpsDialect.h.inc"

#endif // CSL_DIALECT_H
