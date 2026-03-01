//===- CSLRuntimeBase.td - CSL Runtime dialect base definitions -*-tablegen-*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Base definitions for the CSL Runtime (csl_rt) dialect.
// Defines the dialect, base op class, and types that map to SdkLayout/SdkRuntime.
//
//===----------------------------------------------------------------------===//

#ifndef CSLRUNTIME_BASE
#define CSLRUNTIME_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// CSL Runtime Dialect
//===----------------------------------------------------------------------===//

def CSLRuntime_Dialect : Dialect {
  let name = "csl_rt";
  let cppNamespace = "xilinx::csl_rt";
  let description = [{
    The `csl_rt` dialect provides a structured MLIR representation of runtime
    operations that directly correspond to the Cerebras SDK's SdkLayout and
    SdkRuntime Python APIs. It separates generic spatial layout (in the CSL
    dialect) from host-facing runtime operations (layout compilation, load,
    run, memcpy, etc.).

    The dialect includes two categories of operations:
    - **Layout**: `create_layout`, `create_code_region`, `place`, `set_param_all`,
      `export_name`, `compile`
    - **Runtime**: `runtime_create`, `load`, `run`, `stop`, `get_id`,
      `memcpy_h2d`, `memcpy_d2h`, `launch`
  }];

  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &os) const override;
  }];
}

//===----------------------------------------------------------------------===//
// CSL Runtime Op Base Class
//===----------------------------------------------------------------------===//

class CSLRuntime_Op<string mnemonic, list<Trait> traits = []>
    : Op<CSLRuntime_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CSL Runtime Types (declared in C++, referenced here for predicate constraints)
//===----------------------------------------------------------------------===//

def CSLRuntime_LayoutType : DialectType<
  CSLRuntime_Dialect,
  CPred<"llvm::isa<xilinx::csl_rt::LayoutType>($_self)">,
  "CSL Runtime layout type">,
  BuildableType<"xilinx::csl_rt::LayoutType::get($_builder.getContext())">;

def CSLRuntime_CodeRegionType : DialectType<
  CSLRuntime_Dialect,
  CPred<"llvm::isa<xilinx::csl_rt::CodeRegionType>($_self)">,
  "CSL Runtime code region type">,
  BuildableType<"xilinx::csl_rt::CodeRegionType::get($_builder.getContext())">;

def CSLRuntime_CompileArtifactsType : DialectType<
  CSLRuntime_Dialect,
  CPred<"llvm::isa<xilinx::csl_rt::CompileArtifactsType>($_self)">,
  "CSL Runtime compile artifacts type">,
  BuildableType<"xilinx::csl_rt::CompileArtifactsType::get($_builder.getContext())">;

def CSLRuntime_RuntimeType : DialectType<
  CSLRuntime_Dialect,
  CPred<"llvm::isa<xilinx::csl_rt::RuntimeType>($_self)">,
  "CSL Runtime instance type">,
  BuildableType<"xilinx::csl_rt::RuntimeType::get($_builder.getContext())">;

def CSLRuntime_ColorType : DialectType<
  CSLRuntime_Dialect,
  CPred<"llvm::isa<xilinx::csl_rt::ColorType>($_self)">,
  "CSL Runtime color type">,
  BuildableType<"xilinx::csl_rt::ColorType::get($_builder.getContext())">;

def CSLRuntime_RoutingPositionType : DialectType<
  CSLRuntime_Dialect,
  CPred<"llvm::isa<xilinx::csl_rt::RoutingPositionType>($_self)">,
  "CSL Runtime routing position type">,
  BuildableType<"xilinx::csl_rt::RoutingPositionType::get($_builder.getContext())">;

def CSLRuntime_PortType : DialectType<
  CSLRuntime_Dialect,
  CPred<"llvm::isa<xilinx::csl_rt::PortType>($_self)">,
  "CSL Runtime port type">,
  BuildableType<"xilinx::csl_rt::PortType::get($_builder.getContext())">;

def CSLRuntime_StreamType : DialectType<
  CSLRuntime_Dialect,
  CPred<"llvm::isa<xilinx::csl_rt::StreamType>($_self)">,
  "CSL Runtime stream type">,
  BuildableType<"xilinx::csl_rt::StreamType::get($_builder.getContext())">;

#endif // CSLRUNTIME_BASE
