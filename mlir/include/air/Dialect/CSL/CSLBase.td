//===- CSLBase.td - CSL dialect base definitions -----------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Base definitions for the CSL (Cerebras SDK Language) dialect family.
// Defines shared types, attributes, and enums used across all CSL sub-dialect
// categories: layout, placement, routing, kernel, data movement, and runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_BASE
#define CSL_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// CSL Dialect
//===----------------------------------------------------------------------===//

def CSL_Dialect : Dialect {
  let name = "csl";
  let cppNamespace = "xilinx::csl";
  let description = [{
    The `csl` dialect provides a structured MLIR representation of Cerebras
    Systems Language (CSL) programs targeting Wafer-Scale Engine (WSE)
    hardware. It mirrors key CSL constructs -- layout blocks, PE tile
    assignment, color-based routing, DSD-driven data movement, task-based
    kernels, and host runtime interfaces -- as first-class MLIR operations
    with verification and optimization opportunities.

    The dialect is organized into six conceptual sub-dialect categories,
    all sharing the `csl` namespace:

    - **Layout**: top-level spatial configuration (`csl.layout`,
      `csl.set_rectangle`).
    - **Routing**: color declaration and directional routing
      (`csl.color`, `csl.route`).
    - **Kernel**: PE-level compute (`csl.func`, `csl.task`, `csl.var`).
    - **Data Movement**: DSD creation and bulk moves (`csl.get_mem_dsd`,
      `csl.get_fab_dsd`, `csl.mov`).
    - **Runtime**: module system and host interface
      (`csl.import_module`, `csl.export_name`, `csl.export_symbol`,
       `csl.module`, `csl.comptime`).
  }];

  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &os) const override;
  }];
}

//===----------------------------------------------------------------------===//
// CSL Op Base Class
//===----------------------------------------------------------------------===//

class CSL_Op<string mnemonic, list<Trait> traits = []>
    : Op<CSL_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CSL Types (declared in C++, referenced here for predicate constraints)
//===----------------------------------------------------------------------===//

def CSL_ColorType : DialectType<
  CSL_Dialect,
  CPred<"llvm::isa<xilinx::csl::ColorType>($_self)">,
  "CSL color type">,
  BuildableType<"xilinx::csl::ColorType::get($_builder.getContext())">;

def CSL_DsdType : DialectType<
  CSL_Dialect,
  CPred<"llvm::isa<xilinx::csl::DsdType>($_self)">,
  "CSL DSD type">,
  BuildableType<"xilinx::csl::DsdType::get($_builder.getContext())">;

def CSL_ImportedModuleType : DialectType<
  CSL_Dialect,
  CPred<"llvm::isa<xilinx::csl::ImportedModuleType>($_self)">,
  "CSL imported module type">,
  BuildableType<"xilinx::csl::ImportedModuleType::get($_builder.getContext())">;

def CSL_KernelType : DialectType<
  CSL_Dialect,
  CPred<"llvm::isa<xilinx::csl::KernelType>($_self)">,
  "CSL kernel handle">,
  BuildableType<"xilinx::csl::KernelType::get($_builder.getContext())">;

def CSL_RouteType : DialectType<
  CSL_Dialect,
  CPred<"llvm::isa<xilinx::csl::RouteType>($_self)">,
  "CSL route handle">,
  BuildableType<"xilinx::csl::RouteType::get($_builder.getContext())">;

//===----------------------------------------------------------------------===//
// CSL Enums
//===----------------------------------------------------------------------===//

def CSL_DirNorth : I32EnumAttrCase<"NORTH", 0>;
def CSL_DirSouth : I32EnumAttrCase<"SOUTH", 1>;
def CSL_DirEast  : I32EnumAttrCase<"EAST",  2>;
def CSL_DirWest  : I32EnumAttrCase<"WEST",  3>;
def CSL_DirRamp  : I32EnumAttrCase<"RAMP",  4>;

def CSL_DirectionEnum : I32EnumAttr<"Direction",
    "CSL routing direction",
    [CSL_DirNorth, CSL_DirSouth, CSL_DirEast, CSL_DirWest, CSL_DirRamp]> {
  let cppNamespace = "xilinx::csl";
}

def CSL_DsdKindMem1D  : I32EnumAttrCase<"mem1d",  0>;
def CSL_DsdKindMem2D  : I32EnumAttrCase<"mem2d",  1>;
def CSL_DsdKindFabIn  : I32EnumAttrCase<"fabin",  2>;
def CSL_DsdKindFabOut : I32EnumAttrCase<"fabout", 3>;

def CSL_DsdKindEnum : I32EnumAttr<"DsdKind",
    "CSL DSD kind",
    [CSL_DsdKindMem1D, CSL_DsdKindMem2D, CSL_DsdKindFabIn,
     CSL_DsdKindFabOut]> {
  let cppNamespace = "xilinx::csl";
}

//===----------------------------------------------------------------------===//
// Edge enum — used by SdkLayout port creation ops
// Maps to cerebras.sdk.runtime.sdkruntimepybind.Edge
//===----------------------------------------------------------------------===//

def CSL_EdgeLeft   : I32EnumAttrCase<"LEFT",   0>;
def CSL_EdgeRight  : I32EnumAttrCase<"RIGHT",  1>;
def CSL_EdgeTop    : I32EnumAttrCase<"TOP",    2>;
def CSL_EdgeBottom : I32EnumAttrCase<"BOTTOM", 3>;

def CSL_EdgeEnum : I32EnumAttr<"Edge",
    "CSL port edge (SdkLayout)",
    [CSL_EdgeLeft, CSL_EdgeRight, CSL_EdgeTop, CSL_EdgeBottom]> {
  let cppNamespace = "xilinx::csl";
}


#endif // CSL_BASE
