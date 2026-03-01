//===- CSLRuntimeDialect.cpp - CSL Runtime dialect implementation *- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSLRuntime/CSLRuntimeDialect.h"
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define GET_DIALECT_CLASS_DEF
#include "air/Dialect/CSLRuntime/CSLRuntimeOpsDialect.cpp.inc"

namespace xilinx::csl_rt {

void CSLRuntimeDialect::initialize() {
  addTypes<LayoutType, CodeRegionType, CompileArtifactsType, RuntimeType,
           ColorType, RoutingPositionType, PortType, StreamType>();
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/CSLRuntime/CSLRuntimeOps.cpp.inc"
      >();
}

} // namespace xilinx::csl_rt

// Parse types: handles !csl_rt.layout, !csl_rt.code_region, etc.
mlir::Type xilinx::csl_rt::CSLRuntimeDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "layout")
    return LayoutType::get(getContext());
  if (keyword == "code_region")
    return CodeRegionType::get(getContext());
  if (keyword == "compile_artifacts")
    return CompileArtifactsType::get(getContext());
  if (keyword == "runtime")
    return RuntimeType::get(getContext());
  if (keyword == "color")
    return ColorType::get(getContext());
  if (keyword == "routing_position")
    return RoutingPositionType::get(getContext());
  if (keyword == "port")
    return PortType::get(getContext());
  if (keyword == "stream")
    return StreamType::get(getContext());

  parser.emitError(parser.getNameLoc())
      << "unknown csl_rt type '" << keyword << "'";
  return Type();
}

// Print types
void xilinx::csl_rt::CSLRuntimeDialect::printType(mlir::Type type,
                                                   mlir::DialectAsmPrinter &os) const {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<LayoutType>([&](auto) { os << "layout"; })
      .Case<CodeRegionType>([&](auto) { os << "code_region"; })
      .Case<CompileArtifactsType>([&](auto) { os << "compile_artifacts"; })
      .Case<RuntimeType>([&](auto) { os << "runtime"; })
      .Case<ColorType>([&](auto) { os << "color"; })
      .Case<RoutingPositionType>([&](auto) { os << "routing_position"; })
      .Case<PortType>([&](auto) { os << "port"; })
      .Case<StreamType>([&](auto) { os << "stream"; })
      .Default([this](mlir::Type) {
        llvm_unreachable("unknown csl_rt type");
      });
}
