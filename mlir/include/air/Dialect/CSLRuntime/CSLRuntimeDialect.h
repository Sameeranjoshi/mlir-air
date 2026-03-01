//===- CSLRuntimeDialect.h - CSL Runtime dialect declaration ----*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSLRUNTIME_DIALECT_H
#define CSLRUNTIME_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx {
namespace csl_rt {

// Runtime dialect types (opaque handles for SdkLayout/SdkRuntime objects)

/// Opaque handle for an SdkLayout instance.
class LayoutType : public mlir::Type::TypeBase<LayoutType, mlir::Type,
                                               mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl_rt.layout";
};

/// Opaque handle for a CodeRegion object from SdkLayout.
class CodeRegionType
    : public mlir::Type::TypeBase<CodeRegionType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl_rt.code_region";
};

/// Opaque handle for compile artifacts from SdkLayout.
class CompileArtifactsType
    : public mlir::Type::TypeBase<CompileArtifactsType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl_rt.compile_artifacts";
};

/// Opaque handle for an SdkRuntime instance.
class RuntimeType
    : public mlir::Type::TypeBase<RuntimeType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl_rt.runtime";
};

/// Opaque handle for a Color in routing.
class ColorType : public mlir::Type::TypeBase<ColorType, mlir::Type,
                                              mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl_rt.color";
};

/// Opaque handle for a RoutingPosition.
class RoutingPositionType
    : public mlir::Type::TypeBase<RoutingPositionType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl_rt.routing_position";
};

/// Opaque handle for a port on a code region.
class PortType : public mlir::Type::TypeBase<PortType, mlir::Type,
                                             mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl_rt.port";
};

/// Opaque handle for an I/O stream.
class StreamType : public mlir::Type::TypeBase<StreamType, mlir::Type,
                                               mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl_rt.stream";
};

/// Register the csl_rt→Python translation in air-translate.
void registerCSLRuntimeToPyTranslation();

} // namespace csl_rt
} // namespace xilinx

#include "air/Dialect/CSLRuntime/CSLRuntimeOpsDialect.h.inc"

#endif // CSLRUNTIME_DIALECT_H
