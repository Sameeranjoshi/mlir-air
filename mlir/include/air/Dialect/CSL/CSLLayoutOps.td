//===- CSLLayoutOps.td - CSL layout and placement operations -*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Operations for spatial layout, placement, painting, port creation, and
// inter-region dataflow on the Cerebras Wafer-Scale Engine (WSE).
//
// Active ops match the programming model from placement_mlir_design.txt:
//   csl.spatial_placement, csl.code_region (with body), csl.paint,
//   csl.port, csl.place (with kernel), csl.dataflow
//
// SdkLayout-specific ops are guarded behind CSL_ENABLE_SDK_LAYOUT_OPS
// and currently inactive. To re-enable them, define that macro.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_LAYOUT_OPS
#define CSL_LAYOUT_OPS

include "air/Dialect/CSL/CSLBase.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// Type predicates (always active)
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
// Active ops — matching the spatial placement programming model
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl.spatial_placement
//===----------------------------------------------------------------------===//
//
// Top-level container for the spatial placement programming model.
// The body holds global color/route/kernel declarations, code region
// definitions with paint/port ops, placement with kernel binding,
// and dataflow connections between ports.
//
// Example:
//   ```mlir
//   csl.spatial_placement {
//     %red = csl.color : !csl.color
//     %r1 = csl.route in(RAMP) out(WEST) : i32
//     %k1 = csl.kernel "pe.csl" { ... } : !csl.kernel
//     %regionA = csl.code_region routes(%r1) colors(%red) shape(10, 10) {
//       csl.paint pe(0, 0) route(%r1) color(%red)
//     } : !csl.code_region
//     csl.place %regionA at(0, 0) kernel(%k1)
//   }
//   ```
//
def CSL_SpatialPlacementOp : CSL_Op<"spatial_placement", [
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Top-level spatial placement container";
  let description = [{
    Top-level container for the spatial placement programming model.
    The body holds color/route/kernel declarations, code region definitions
    with paint/port ops, placement with kernel binding, and dataflow
    connections between ports.

    Example:
    ```mlir
    csl.spatial_placement {
      %red = csl.color : !csl.color
      %r1 = csl.route in(RAMP) out(WEST) : i32
      %k1 = csl.kernel "pe.csl" { } : !csl.kernel
      %regionA = csl.code_region routes(%r1) colors(%red) shape(10, 10) {
        csl.paint pe(0, 0) route(%r1) color(%red)
      } : !csl.code_region
      csl.place %regionA at(0, 0) kernel(%k1)
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.code_region
//===----------------------------------------------------------------------===//
//
// Defines a code region with routes, colors, and a PE grid shape. The body
// contains paint ops that configure per-PE routing and coloring. Port ops
// are declared on the region from outside the body.
//
//   %regionA = csl.code_region routes(%r1, %r2)
//                              colors(%red, %green)
//                              shape(10, 10) {
//     csl.paint pe(0, 0) route(%r1) color(%red)
//   } : !csl.code_region
//
def CSL_CodeRegionOp : CSL_Op<"code_region", [
    SingleBlock,
    NoTerminator,
    AttrSizedOperandSegments
  ]> {
  let summary = "Define a code region with routes, colors, and PE shape";
  let description = [{
    Defines a code region with routing configurations, colors, and a PE
    grid shape. The body contains paint ops. Returns a `!csl.code_region`
    handle used by port, place, and dataflow ops.

    Example:
    ```mlir
    %regionA = csl.code_region routes(%r1, %r2) colors(%red, %green)
                               shape(10, 10) {
      csl.paint pe(0, 0) route(%r1) color(%red)
    } : !csl.code_region
    ```
  }];

  let arguments = (ins
    Variadic<I32>:$routes,
    Variadic<CSL_ColorType>:$colors,
    I64Attr:$width,
    I64Attr:$height
  );
  let results = (outs CSL_CodeRegionType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `routes` `(` $routes `)` `colors` `(` $colors `)`
    `shape` `(` $width `,` $height `)` $body attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// csl.paint
//===----------------------------------------------------------------------===//
//
// Paints a single PE within a code region body with a route and color.
// Used inside the body of a `csl.code_region` to configure per-PE
// routing and color assignments.
//
// Example:
//   ```mlir
//   csl.paint pe(0, 0) route(%r1) color(%red)
//   ```
//
def CSL_PaintOp : CSL_Op<"paint", []> {
  let summary = "Paint a PE with a route and color (inside code_region body)";
  let description = [{
    Paints a single PE with a routing configuration and color assignment.
    Used inside the body of a `csl.code_region`.

    Example:
    ```mlir
    csl.paint pe(0, 0) route(%r1) color(%red)
    ```
  }];

  let arguments = (ins
    I64Attr:$pe_x,
    I64Attr:$pe_y,
    I32:$route,
    CSL_ColorType:$color
  );

  let assemblyFormat = [{
    `pe` `(` $pe_x `,` $pe_y `)` `route` `(` $route `)` `color` `(` $color `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// csl.port
//===----------------------------------------------------------------------===//
//
// Declares a port on a code region. Ports represent I/O endpoints for
// inter-region communication. Each port has a type (input or output),
// a routing configuration, and a data size.
//
// Example:
//   ```mlir
//   %p1 = csl.port %regionA type("input") route(%r1) size(10) : !csl.port
//   %p2 = csl.port %regionA type("output") route(%r2) size(10) : !csl.port
//   ```
//
def CSL_PortOp : CSL_Op<"port", []> {
  let summary = "Declare a port on a code region";
  let description = [{
    Declares a port on a code region for inter-region communication.
    Each port has a type ("input" or "output"), a routing configuration,
    and a data size. Returns a `!csl.port` handle used by `csl.dataflow`.

    Example:
    ```mlir
    %p1 = csl.port %regionA type("input") route(%r1) size(10) : !csl.port
    %p2 = csl.port %regionA type("output") route(%r2) size(10) : !csl.port
    ```
  }];

  let arguments = (ins
    CSL_CodeRegionType:$region,
    StrAttr:$port_type,
    I32:$route,
    I64Attr:$size
  );
  let results = (outs CSL_PortType:$result);

  let assemblyFormat = [{
    $region `type` `(` $port_type `)` `route` `(` $route `)` `size` `(` $size `)`
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// csl.place
//===----------------------------------------------------------------------===//
//
// Places a code region at a global PE coordinate with a kernel binding.
// Decouples the kernel definition from the region, allowing any kernel
// to be placed on any region.
//
// Example:
//   ```mlir
//   csl.place %regionA at(0, 0) kernel(%k1)
//   ```
//
def CSL_PlaceOp : CSL_Op<"place", []> {
  let summary = "Place a code region at a coordinate with a kernel";
  let description = [{
    Places a code region at a global PE coordinate with a kernel binding.
    Decouples kernel definition from region, allowing any kernel to be
    placed on any region.

    Example:
    ```mlir
    csl.place %regionA at(0, 0) kernel(%k1)
    ```
  }];

  let arguments = (ins
    CSL_CodeRegionType:$region,
    I64Attr:$x,
    I64Attr:$y,
    CSL_KernelType:$kernel
  );

  let assemblyFormat = [{
    $region `at` `(` $x `,` $y `)` `kernel` `(` $kernel `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// csl.dataflow
//===----------------------------------------------------------------------===//
//
// Connects a source port to a destination port across regions. The `cslc`
// compiler automatically finds optimal routes between the connected ports.
// Inspired by `aie.flow()`.
//
// Example:
//   ```mlir
//   csl.dataflow %src_port -> %dst_port
//   ```
//
def CSL_DataflowOp : CSL_Op<"dataflow", []> {
  let summary = "Connect a source port to a destination port across regions";
  let description = [{
    Connects a source port to a destination port across regions.
    The `cslc` compiler automatically finds optimal routes between
    the connected ports. Inspired by the `aie.flow()` operation.

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
// SdkLayout-specific ops (currently inactive)
//
// These ops model the Python SdkLayout API one-to-one. They are preserved
// here for reference and future re-enablement. To activate them, define
// CSL_ENABLE_SDK_LAYOUT_OPS before including this file.
//===----------------------------------------------------------------------===//

// #ifdef CSL_ENABLE_SDK_LAYOUT_OPS

// def CSL_SdkLayoutOp : CSL_Op<"sdk_layout", [
//     SingleBlock,
//     NoTerminator
//   ]> {
//   let summary = "Top-level SdkLayout container (Python API model)";
//   let description = [{
//     Models the Python `SdkLayout` object.
//   }];
//   let regions = (region SizedRegion<1>:$body);
//   let assemblyFormat = "$body attr-dict";
// }

// def CSL_SymColorOp : CSL_Op<"sym_color", []> {
//   let summary = "Declare a free-standing symbolic color";
//   let arguments = (ins StrAttr:$sym_name);
//   let results = (outs CSL_ColorType:$result);
//   let assemblyFormat = "$sym_name attr-dict `:` type($result)";
// }

// def CSL_ScopedColorOp : CSL_Op<"scoped_color", []> {
//   let summary = "Declare a region-scoped symbolic color";
//   let arguments = (ins CSL_CodeRegionType:$region, StrAttr:$sym_name);
//   let results = (outs CSL_ColorType:$result);
//   let assemblyFormat = "$region $sym_name attr-dict `:` type($result)";
// }

// def CSL_RoutingPositionOp : CSL_Op<"routing_position", []> {
//   let summary = "Build a RoutingPosition (per-PE input/output direction set)";
//   let arguments = (ins I32ArrayAttr:$input_dirs, I32ArrayAttr:$output_dirs);
//   let results = (outs I32:$result);
//   let assemblyFormat = [{
//     `inputs` `(` $input_dirs `)` `outputs` `(` $output_dirs `)` attr-dict `:` type($result)
//   }];
// }

// def CSL_PaintAllOp : CSL_Op<"paint_all", []> {
//   let summary = "Paint every PE in a region with a color and routing positions";
//   let arguments = (ins CSL_CodeRegionType:$region, CSL_ColorType:$color, Variadic<I32>:$routing_positions);
//   let assemblyFormat = [{
//     $region `color` `(` $color `)` `routing` `(` $routing_positions `)` attr-dict
//   }];
// }

// def CSL_PaintRangeOp : CSL_Op<"paint_range", []> {
//   let summary = "Paint a rectangular sub-range of PEs with a color";
//   let arguments = (ins CSL_CodeRegionType:$region, I64Attr:$rect_ox, I64Attr:$rect_oy, I64Attr:$rect_w, I64Attr:$rect_h, CSL_ColorType:$color, Variadic<I32>:$routing_positions);
//   let assemblyFormat = [{
//     $region `rect` `(` $rect_ox `,` $rect_oy `,` $rect_w `,` $rect_h `)`
//     `color` `(` $color `)` `routing` `(` $routing_positions `)` attr-dict
//   }];
// }

// def CSL_SetParamAllOp : CSL_Op<"set_param_all", []> {
//   let summary = "Set a compile-time parameter on every PE in a region";
//   let arguments = (ins CSL_CodeRegionType:$region, OptionalAttr<StrAttr>:$param_name, OptionalAttr<I64Attr>:$value, Optional<CSL_ColorType>:$color);
//   let assemblyFormat = [{
//     $region (`param` `(` $param_name^ `)` `value` `(` $value `)`)? (`color` `(` $color^ `)`)? attr-dict
//   }];
// }

// def CSL_SetParamOp : CSL_Op<"set_param", []> {
//   let summary = "Set a compile-time parameter on a single PE";
//   let arguments = (ins CSL_CodeRegionType:$region, I64Attr:$pe_x, I64Attr:$pe_y, StrAttr:$param_name, I64Attr:$value);
//   let assemblyFormat = [{
//     $region `at` `(` $pe_x `,` $pe_y `)` `param` `(` $param_name `)` `value` `(` $value `)` attr-dict
//   }];
// }

// def CSL_CreateOutputPortOp : CSL_Op<"create_output_port", []> {
//   let summary = "Create an output port on a region edge";
//   let arguments = (ins CSL_CodeRegionType:$region, CSL_ColorType:$color, CSL_EdgeEnum:$edge, Variadic<I32>:$routing_positions, I64Attr:$data_size);
//   let results = (outs CSL_PortType:$result);
//   let assemblyFormat = [{
//     $region `color` `(` $color `)` `edge` `(` $edge `)` `routing` `(` $routing_positions `)`
//     `size` `(` $data_size `)` attr-dict `:` type($result)
//   }];
// }

// def CSL_CreateInputPortOp : CSL_Op<"create_input_port", []> {
//   let summary = "Create an input port on a region edge";
//   let arguments = (ins CSL_CodeRegionType:$region, CSL_ColorType:$color, CSL_EdgeEnum:$edge, Variadic<I32>:$routing_positions, I64Attr:$data_size);
//   let results = (outs CSL_PortType:$result);
//   let assemblyFormat = [{
//     $region `color` `(` $color `)` `edge` `(` $edge `)` `routing` `(` $routing_positions `)`
//     `size` `(` $data_size `)` attr-dict `:` type($result)
//   }];
// }

// def CSL_ConnectOp : CSL_Op<"connect", []> {
//   let summary = "Connect an output port to an input port";
//   let arguments = (ins CSL_PortType:$tx, CSL_PortType:$rx);
//   let assemblyFormat = "$tx `->` $rx attr-dict";
// }

// def CSL_HstackOp : CSL_Op<"hstack", []> {
//   let summary = "Stack code regions horizontally";
//   let arguments = (ins Variadic<CSL_CodeRegionType>:$code_regions);
//   let assemblyFormat = "`[` $code_regions `]` attr-dict";
// }

// def CSL_VstackOp : CSL_Op<"vstack", []> {
//   let summary = "Stack code regions vertically";
//   let arguments = (ins Variadic<CSL_CodeRegionType>:$code_regions, OptionalAttr<I64Attr>:$origin_x, OptionalAttr<I64Attr>:$origin_y);
//   let assemblyFormat = [{
//     `[` $code_regions `]` (`origin` `(` $origin_x^ `,` $origin_y `)`)? attr-dict
//   }];
// }

// def CSL_CreateInputStreamOp : CSL_Op<"create_input_stream", []> {
//   let summary = "Create a host-to-device (H2D) I/O stream";
//   let arguments = (ins CSL_PortType:$port, OptionalAttr<I64Attr>:$io_buffer_size);
//   let results = (outs CSL_StreamType:$result);
//   let assemblyFormat = [{
//     $port (`buf` `(` $io_buffer_size^ `)`)? attr-dict `:` type($result)
//   }];
// }

// def CSL_CreateOutputStreamOp : CSL_Op<"create_output_stream", []> {
//   let summary = "Create a device-to-host (D2H) I/O stream";
//   let arguments = (ins CSL_PortType:$port, OptionalAttr<I64Attr>:$io_buffer_size);
//   let results = (outs CSL_StreamType:$result);
//   let assemblyFormat = [{
//     $port (`buf` `(` $io_buffer_size^ `)`)? attr-dict `:` type($result)
//   }];
// }

// #endif // CSL_ENABLE_SDK_LAYOUT_OPS

#endif // CSL_LAYOUT_OPS
