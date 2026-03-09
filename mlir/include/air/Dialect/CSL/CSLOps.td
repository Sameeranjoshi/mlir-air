//===- CSLOps.td - All CSL dialect operations (consolidated) -*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Consolidated TableGen file containing all active CSL dialect operations.
// This includes routing ops, layout/spatial ops, PE program ops, and
// (commented) data movement ops.
//
// Final op count: 12 active ops
//   - Routing (2): csl.color, csl.route
//   - Layout (4): csl.spatial_placement, csl.code_region, csl.place, csl.dataflow
//   - PE Program (5): csl.kernel, csl.func, csl.task, csl.return, csl.var, csl.import_module
//   - Deferred (3): csl.get_mem_dsd, csl.get_fab_dsd, csl.mov
//
//===----------------------------------------------------------------------===//

#ifndef CSL_OPS
#define CSL_OPS

include "air/Dialect/CSL/CSLBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Type predicates (layout / spatial)
//===----------------------------------------------------------------------===//

def CSL_CodeRegionType : DialectType<
  CSL_Dialect,
  CPred<"llvm::isa<xilinx::csl::CodeRegionType>($_self)">,
  "CSL code region handle">,
  BuildableType<"xilinx::csl::CodeRegionType::get($_builder.getContext())">;

def CSL_PortType : DialectType<
  CSL_Dialect,
  CPred<"llvm::isa<xilinx::csl::PortType>($_self)">,
  "CSL port handle">,
  BuildableType<"xilinx::csl::PortType::get($_builder.getContext())">;

def CSL_StreamType : DialectType<
  CSL_Dialect,
  CPred<"llvm::isa<xilinx::csl::StreamType>($_self)">,
  "CSL stream handle">,
  BuildableType<"xilinx::csl::StreamType::get($_builder.getContext())">;

//===----------------------------------------------------------------------===//
// Routing ops — color and route declaration
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl.color — Declare a communication color
//===----------------------------------------------------------------------===//

def CSL_ColorOp : CSL_Op<"color", []> {
  let summary = "Declare a communication color";
  let description = [{
    Declares a CSL communication color. Colors are the fundamental routing
    primitive on the WSE fabric. When an integer ID is provided, the color
    is bound to that specific physical color. When no ID is given, the
    compiler assigns one automatically.

    Examples:
    ```mlir
    %red = csl.color : !csl.color
    %c0  = csl.color {id = 0 : i32} : !csl.color
    ```
  }];

  let arguments = (ins OptionalAttr<I32Attr>:$id);
  let results = (outs CSL_ColorType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// csl.route — Declare a routing configuration
//===----------------------------------------------------------------------===//

def CSL_RouteOp : CSL_Op<"route", []> {
  let summary = "Declare a routing configuration (input -> output directions)";
  let description = [{
    Declares a routing configuration with input and output directions.
    Direction values follow the Direction enum: NORTH=0, SOUTH=1, EAST=2,
    WEST=3, RAMP=4 (RAMP = PE compute unit).

    The result is a `!csl.route` handle used by code_region and other ops.

    Examples:
    ```mlir
    %r1 = csl.route in(RAMP) out(EAST) : !csl.route
    %r2 = csl.route in(WEST) out(RAMP) : !csl.route
    ```
  }];

  let arguments = (ins
    CSL_DirectionEnum:$input_dir,
    CSL_DirectionEnum:$output_dir
  );
  let results = (outs CSL_RouteType:$result);

  let assemblyFormat = "`in` `(` $input_dir `)` `out` `(` $output_dir `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Layout ops — spatial placement and region structure
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl.spatial_placement — Top-level spatial placement container
//===----------------------------------------------------------------------===//

def CSL_SpatialPlacementOp : CSL_Op<"spatial_placement", [
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Top-level spatial placement container";
  let description = [{
    Top-level container for the spatial placement programming model.
    The body holds color/route/kernel declarations, code region definitions
    with per-PE configuration, placement with kernel binding, and dataflow
    connections between ports.

    Example:
    ```mlir
    csl.spatial_placement {
      %red = csl.color : !csl.color
      %r1 = csl.route {input_dir = RAMP, output_dir = WEST} : !csl.route
      %k1 = csl.kernel "pe.csl" { } : !csl.kernel
      %regionA = csl.code_region routes(%r1) colors(%red) {
        csl.place %regionA %k1 {x = 0 : i64, y = 0 : i64}
      } : !csl.code_region
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.code_region — Define a code region with routes and colors
//===----------------------------------------------------------------------===//

def CSL_CodeRegionOp : CSL_Op<"code_region", [
    SingleBlock,
    NoTerminator,
    AttrSizedOperandSegments
  ]> {
  let summary = "Define a PE grid code region with routes and colors";
  let description = [{
    Defines a code region with routing configurations, colors, and PE grid shape.
    The body may contain ops to configure per-PE behavior.
    Returns a `!csl.code_region` handle used by place and dataflow ops.

    Example:
    ```mlir
    %r1 = csl.route {input_dir = RAMP, output_dir = WEST} : !csl.route
    %c = csl.color {id = 3 : i32} : !csl.color
    %regionA = csl.code_region routes(%r1) colors(%c) {
      ...
    } {width = 4 : i64, height = 4 : i64} : !csl.code_region
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$region_name,
    Variadic<CSL_RouteType>:$routes,
    Variadic<CSL_ColorType>:$colors,
    I64Attr:$width,
    I64Attr:$height
  );
  let results = (outs CSL_CodeRegionType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ($region_name^)? `routes` `(` $routes `)` `colors` `(` $colors `)`
    $body `{` `width` `=` $width `:` `i64` `,` `height` `=` $height `:` `i64` `}` attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// csl.place — Place a kernel at spatial coordinates
//===----------------------------------------------------------------------===//

def CSL_PlaceOp : CSL_Op<"place", []> {
  let summary = "Place a kernel at spatial coordinates";
  let description = [{
    Assigns a kernel to a (x, y) position in the spatial grid.
    Decouples kernel definition from region, allowing any kernel to be
    placed on any region.

    Example:
    ```mlir
    csl.place %region %kernel {x = 0 : i64, y = 0 : i64}
    ```
  }];

  let arguments = (ins
    CSL_CodeRegionType:$region,
    CSL_KernelType:$kernel,
    I64Attr:$x,
    I64Attr:$y
  );

  let assemblyFormat = "$region $kernel attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.dataflow — Connect two ports
//===----------------------------------------------------------------------===//

def CSL_DataflowOp : CSL_Op<"dataflow", []> {
  let summary = "Connect a source port to a destination port";
  let description = [{
    Creates a directed data connection between two I/O ports across regions.
    The `cslc` compiler automatically finds optimal routes between the
    connected ports.

    Example:
    ```mlir
    csl.dataflow %src_port -> %dst_port
    ```
  }];

  let arguments = (ins
    CSL_PortType:$src,
    CSL_PortType:$dst
  );

  let assemblyFormat = "$src `->` $dst attr-dict";
}

//===----------------------------------------------------------------------===//
// PE Program ops — kernel, functions, tasks, variables
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl.kernel — PE kernel program container
//===----------------------------------------------------------------------===//

def CSL_KernelOp : CSL_Op<"kernel", [
    SingleBlock,
    NoTerminator,
    SymbolTable
  ]> {
  let summary = "PE kernel program container";
  let description = [{
    A kernel container that groups PE-level compute operations (functions,
    tasks, variables) together with a CSL source file path and optional
    compile-time parameters. Returns a `!csl.kernel` handle that can be
    bound to a code region via `csl.place`.

    Example:
    ```mlir
    %k = csl.kernel {source_file = "pe.csl"} {
      csl.func @compute { csl.return }
    } : !csl.kernel

    %k2 = csl.kernel {source_file = "pe.csl", params = {width = 100 : i32}} {
      csl.var @buf : memref<1024xf32>
      csl.func @compute { csl.return }
    } : !csl.kernel
    ```
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$source_file,
    OptionalAttr<DictionaryAttr>:$params
  );
  let results = (outs CSL_KernelType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$body attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// csl.func — PE function declaration
//===----------------------------------------------------------------------===//

def CSL_FuncOp : CSL_Op<"func", [
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "PE function declaration";
  let description = [{
    Declares a function that runs on a single PE. The body contains standard
    MLIR operations representing the compute kernel. Corresponds to
    `fn name() void { ... }` in CSL.

    Example:
    ```mlir
    csl.func @compute {
      csl.return
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name $body attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.task — PE task bound to a color
//===----------------------------------------------------------------------===//

def CSL_TaskOp : CSL_Op<"task", [
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "PE task bound to a color";
  let description = [{
    Defines an event-driven task on a PE. Tasks are triggered when a
    wavelet arrives on the bound color. This is the fundamental
    execution mechanism on the WSE.

    Example:
    ```mlir
    %c = csl.color {id = 3 : i32} : !csl.color
    csl.task @recv color(%c) {
      csl.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    CSL_ColorType:$color
  );
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name `color` `(` $color `)` $body attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.return — Terminator for func/task
//===----------------------------------------------------------------------===//

def CSL_ReturnOp : CSL_Op<"return", [
    Terminator,
    ParentOneOf<["FuncOp", "TaskOp"]>
  ]> {
  let summary = "Return from a PE function or task";
  let description = [{
    Terminates a csl.func or csl.task region.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.var — PE variable declaration
//===----------------------------------------------------------------------===//

def CSL_VarOp : CSL_Op<"var", [Symbol]> {
  let summary = "PE variable declaration";
  let description = [{
    Declares a variable in the PE's local memory. Corresponds to
    `var name: [N]T` in CSL.

    Example:
    ```mlir
    csl.var @buf : memref<1024xf32>
    csl.var @matrix : memref<32x32xf16>
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type
  );

  let assemblyFormat = "$sym_name `:` $type attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.import_module — Import an external CSL library module
//===----------------------------------------------------------------------===//

def CSL_ImportModuleOp : CSL_Op<"import_module", []> {
  let summary = "Import an external CSL library module";
  let description = [{
    Imports a named CSL library for use in PE programs. Corresponds to
    `@import_module("name")` or `@import_module("name", params)` in CSL.

    Example:
    ```mlir
    %m = csl.import_module "<memcpy/memcpy>" : !csl.imported_module
    %m2 = csl.import_module "<memcpy/get_params>" {params = {width = 2 : i32}} : !csl.imported_module
    ```
  }];

  let arguments = (ins
    StrAttr:$module_name,
    OptionalAttr<DictionaryAttr>:$params
  );
  let results = (outs CSL_ImportedModuleType:$result);

  let assemblyFormat = "$module_name attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Data movement ops (deferred — commented out)
//===----------------------------------------------------------------------===//
// TODO: Data movement ops are deferred. Uncomment when CSL DSD lowering is ready.
//
// def CSL_GetMemDsdOp : CSL_Op<"get_mem_dsd", []> {
//   let summary = "Create a memory-backed Data Structure Descriptor";
//   let description = [{
//     Creates a DSD referencing a contiguous region of PE-local memory.
//     Corresponds to `@get_dsd(mem1d_dsd, ...)` in CSL.
//   }];
//   let arguments = (ins AnyMemRef:$buffer, Index:$length);
//   let results = (outs CSL_DsdType:$result);
//   let assemblyFormat = [{
//     $buffer `,` $length `:` type($buffer) `,` type($length) `->` type($result) attr-dict
//   }];
// }
//
// def CSL_GetFabDsdOp : CSL_Op<"get_fab_dsd", []> {
//   let summary = "Create a fabric-backed Data Structure Descriptor";
//   let description = [{
//     Creates a DSD backed by the on-chip fabric (network). Used for
//     streaming data to/from neighboring PEs via colors.
//   }];
//   let arguments = (ins CSL_DsdKindEnum:$kind, CSL_ColorType:$color, Index:$length);
//   let results = (outs CSL_DsdType:$result);
//   let assemblyFormat = [{
//     $kind $color `,` $length `:` type($color) `,` type($length) `->` type($result) attr-dict
//   }];
// }
//
// def CSL_MovOp : CSL_Op<"mov", []> {
//   let summary = "Bulk DSD-to-DSD data move";
//   let description = [{
//     Performs a hardware-accelerated bulk data move between two DSDs.
//     Corresponds to `@mov32` or `@mov16` builtins in CSL.
//   }];
//   let arguments = (ins CSL_DsdType:$dst, CSL_DsdType:$src);
//   let assemblyFormat = "$dst `,` $src `:` type($dst) `,` type($src) attr-dict";
// }

#endif // CSL_OPS
