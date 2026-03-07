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
    OptionalAttr<StrAttr>:$region_name,
    Variadic<I32>:$routes,
    Variadic<CSL_ColorType>:$colors,
    I64Attr:$width,
    I64Attr:$height
  );
  let results = (outs CSL_CodeRegionType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ($region_name^)? `routes` `(` $routes `)` `colors` `(` $colors `)`
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

def CSL_SetParamOp : CSL_Op<"set_param", []> {
  let summary = "Set a compile-time parameter on a single PE";
  let description = [{
    Sets a compile-time parameter value on a single PE identified by
    (pe_x, pe_y) within a code region.
    Corresponds to `code_region.set_param(IntVector(x, y), 'name', value)`
    in the Python SdkLayout API.

    Example:
    ```mlir
    csl.set_param %region at(0, 0) param("select") value(0 : i16)
    ```
  }];
  let arguments = (ins
    CSL_CodeRegionType:$region,
    I64Attr:$pe_x,
    I64Attr:$pe_y,
    StrAttr:$param_name,
    AnyAttr:$value
  );
  let assemblyFormat = [{
    $region `at` `(` $pe_x `,` $pe_y `)` `param` `(` $param_name `)` `value` `(` $value `)` attr-dict
  }];
}

def CSL_SymColorOp : CSL_Op<"sym_color", []> {
  let summary = "Declare a free-standing symbolic color (Python: Color('name'))";
  let description = [{
    Declares a free-standing symbolic color identified by name.
    Corresponds to `Color('name')` in the Python SdkLayout API.

    Example:
    ```mlir
    %c = csl.sym_color "tx" : !csl.color
    ```
  }];
  let arguments = (ins StrAttr:$sym_name);
  let results = (outs CSL_ColorType:$result);
  let assemblyFormat = "$sym_name attr-dict `:` type($result)";
}

def CSL_ScopedColorOp : CSL_Op<"scoped_color", []> {
  let summary = "Declare a region-scoped symbolic color (Python: region.color('name'))";
  let description = [{
    Declares a color scoped to a specific code region.
    Corresponds to `region.color('name')` in the Python SdkLayout API.

    Example:
    ```mlir
    %c = csl.scoped_color %region "tx" : !csl.color
    ```
  }];
  let arguments = (ins CSL_CodeRegionType:$region, StrAttr:$sym_name);
  let results = (outs CSL_ColorType:$result);
  let assemblyFormat = "$region $sym_name attr-dict `:` type($result)";
}

def CSL_SetParamColorOp : CSL_Op<"set_param_color", []> {
  let summary = "Set a color object as param on all PEs (Python: region.set_param_all(color))";
  let description = [{
    Sets a color object as a compile-time parameter on all PEs in a region.
    Corresponds to `code_region.set_param_all(color)` in the Python SdkLayout API.

    Example:
    ```mlir
    csl.set_param_color %region color(%c)
    ```
  }];
  let arguments = (ins
    CSL_CodeRegionType:$region,
    CSL_ColorType:$color
  );
  let assemblyFormat = "$region `color` `(` $color `)` attr-dict";
}

def CSL_InputPortOp : CSL_Op<"input_port", []> {
  let summary = "Create an input port on a region edge";
  let description = [{
    Creates an input port on a code region edge.
    Corresponds to `region.create_input_port(color, Edge.X, [routes], size)`
    in the Python SdkLayout API. Accepts one or more routing positions.

    Example:
    ```mlir
    %p = csl.input_port %region color(%c) edge(RIGHT) routes(%r) size(10) : !csl.port
    %p = csl.input_port %region color(%c) edge(RIGHT) routes(%r1, %r2) size(10) : !csl.port
    ```
  }];
  let arguments = (ins
    CSL_CodeRegionType:$region,
    CSL_ColorType:$color,
    CSL_EdgeEnum:$edge,
    Variadic<I32>:$routes,
    I64Attr:$size
  );
  let results = (outs CSL_PortType:$result);
  let assemblyFormat = [{
    $region `color` `(` $color `)` `edge` `(` $edge `)` `routes` `(` $routes `)`
    `size` `(` $size `)` attr-dict `:` type($result)
  }];
}

def CSL_OutputPortOp : CSL_Op<"output_port", []> {
  let summary = "Create an output port on a region edge";
  let description = [{
    Creates an output port on a code region edge.
    Corresponds to `region.create_output_port(color, Edge.X, [routes], size)`
    in the Python SdkLayout API. Accepts one or more routing positions.

    Example:
    ```mlir
    %p = csl.output_port %region color(%c) edge(LEFT) routes(%r) size(10) : !csl.port
    %p = csl.output_port %region color(%c) edge(LEFT) routes(%r1, %r2) size(10) : !csl.port
    ```
  }];
  let arguments = (ins
    CSL_CodeRegionType:$region,
    CSL_ColorType:$color,
    CSL_EdgeEnum:$edge,
    Variadic<I32>:$routes,
    I64Attr:$size
  );
  let results = (outs CSL_PortType:$result);
  let assemblyFormat = [{
    $region `color` `(` $color `)` `edge` `(` $edge `)` `routes` `(` $routes `)`
    `size` `(` $size `)` attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// csl.routing_position — multi-direction RoutingPosition (Tutorial 05+)
//===----------------------------------------------------------------------===//
//
// Builds a RoutingPosition with potentially multiple input and/or output
// directions. Direction names are comma-separated strings like "NORTH,RAMP".
//
// Example:
//   %r = csl.routing_position in("NORTH") out("RAMP,SOUTH") : i32
//   %r = csl.routing_position out("RAMP") : i32
//
def CSL_RoutingPositionOp : CSL_Op<"routing_position", []> {
  let summary = "Build a RoutingPosition with multiple input/output directions";
  let description = [{
    Builds a RoutingPosition with potentially multiple input/output directions.
    Direction names are comma-separated strings like "NORTH,RAMP".
    Corresponds to RoutingPosition().set_input([...]).set_output([...]).

    Example:
    ```mlir
    %r = csl.routing_position in("NORTH") out("RAMP,SOUTH") : i32
    %r2 = csl.routing_position out("RAMP") : i32
    ```
  }];
  let arguments = (ins
    OptionalAttr<StrAttr>:$in_dirs,
    OptionalAttr<StrAttr>:$out_dirs
  );
  let results = (outs I32:$result);
  let assemblyFormat = [{
    (`in` `(` $in_dirs^ `)`)? (`out` `(` $out_dirs^ `)`)? attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// csl.get_edge_routing — edge-specific routing override (Tutorial 05+)
//===----------------------------------------------------------------------===//
//
// Creates an edge-specific routing override for use with paint_all.
//
// Example:
//   %er = csl.get_edge_routing edge(BOTTOM) routes(%r1) : i32
//
def CSL_GetEdgeRoutingOp : CSL_Op<"get_edge_routing", []> {
  let summary = "Create an edge-specific routing position override";
  let description = [{
    Creates an edge-specific routing override for use with paint_all.
    Corresponds to get_edge_routing(Edge.X, [route_positions]).

    Example:
    ```mlir
    %er = csl.get_edge_routing edge(BOTTOM) routes(%r1) : i32
    ```
  }];
  let arguments = (ins CSL_EdgeEnum:$edge, Variadic<I32>:$routes);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    `edge` `(` $edge `)` `routes` `(` $routes `)` attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// csl.paint_all — paint all PEs in a region (Tutorial 05+)
//===----------------------------------------------------------------------===//
//
// Paints all PEs in a code region with a color and routing positions.
// Optional edge_routes override routing for specific edges.
//
// Example:
//   csl.paint_all %region color(%c) routes(%r1, %r2) edge_routes(%er)
//   csl.paint_all %region color(%c) routes(%r1) edge_routes()
//
def CSL_PaintAllOp : CSL_Op<"paint_all", [AttrSizedOperandSegments]> {
  let summary = "Paint all PEs in a region with core and edge-specific routes";
  let description = [{
    Paints all PEs in a code region with a color and routing positions.
    Optional edge_routes override routing for specific edges.
    Use edge_routes() for no edge routes.

    Example:
    ```mlir
    csl.paint_all %region color(%c) routes(%r1, %r2) edge_routes(%er)
    csl.paint_all %region color(%c) routes(%r1) edge_routes()
    ```
  }];
  let arguments = (ins
    CSL_CodeRegionType:$region,
    CSL_ColorType:$color,
    Variadic<I32>:$core_routes,
    Variadic<I32>:$edge_routes
  );
  let assemblyFormat = [{
    $region `color` `(` $color `)` `routes` `(` $core_routes `)`
    `edge_routes` `(` $edge_routes `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// csl.paint_range — paint a rectangular range of PEs (Tutorial 05+)
//===----------------------------------------------------------------------===//
//
// Paints a rectangular range of PEs (upper-left to lower-right inclusive)
// with a color and routing positions.
//
// Example:
//   csl.paint_range %region rect(0, 0, 1, 4) color(%c) routes(%r)
//
def CSL_PaintRangeOp : CSL_Op<"paint_range", []> {
  let summary = "Paint a rectangular range of PEs with a color and routes";
  let description = [{
    Paints a rectangular range of PEs (upper-left to lower-right) with a color.
    Corresponds to region.paint_range(IntRectangle(IntVector(x1,y1),
    IntVector(x2,y2)), color, [routes]).

    Example:
    ```mlir
    csl.paint_range %region rect(0, 0, 1, 4) color(%c) routes(%r)
    ```
  }];
  let arguments = (ins
    CSL_CodeRegionType:$region,
    I64Attr:$ul_x, I64Attr:$ul_y,
    I64Attr:$lr_x, I64Attr:$lr_y,
    CSL_ColorType:$color,
    Variadic<I32>:$routes
  );
  let assemblyFormat = [{
    $region `rect` `(` $ul_x `,` $ul_y `,` $lr_x `,` $lr_y `)`
    `color` `(` $color `)` `routes` `(` $routes `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// csl.set_param_range_color — set color param for a PE range (Tutorial 05+)
//===----------------------------------------------------------------------===//
//
// Sets a color parameter on all PEs in a rectangular range.
// Corresponds to region.set_param_range(IntRectangle(...), color).
//
// Example:
//   csl.set_param_range_color %region rect(0, 0, 1, 4) color(%c)
//
def CSL_SetParamRangeColorOp : CSL_Op<"set_param_range_color", []> {
  let summary = "Set a color parameter on all PEs in a rectangular range";
  let description = [{
    Sets a color parameter on all PEs in a rectangular range.
    Corresponds to region.set_param_range(IntRectangle(IntVector(x1,y1),
    IntVector(x2,y2)), color).

    Example:
    ```mlir
    csl.set_param_range_color %region rect(0, 0, 1, 4) color(%c)
    ```
  }];
  let arguments = (ins
    CSL_CodeRegionType:$region,
    I64Attr:$ul_x, I64Attr:$ul_y,
    I64Attr:$lr_x, I64Attr:$lr_y,
    CSL_ColorType:$color
  );
  let assemblyFormat = [{
    $region `rect` `(` $ul_x `,` $ul_y `,` $lr_x `,` $lr_y `)` `color` `(` $color `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// csl.set_param_range_named — set named param to color for a PE range (Tutorial 05+)
//===----------------------------------------------------------------------===//
//
// Sets a named parameter to a color value on all PEs in a rectangular range.
// Corresponds to region.set_param_range(IntRectangle(...), 'name', color).
//
// Example:
//   csl.set_param_range_named %region rect(1, 0, 2, 4) param("b_in") color(%c)
//
def CSL_SetParamRangeNamedOp : CSL_Op<"set_param_range_named", []> {
  let summary = "Set a named parameter to a color value on all PEs in a rectangular range";
  let description = [{
    Sets a named parameter to a color value on all PEs in a rectangular range.
    Corresponds to region.set_param_range(IntRectangle(IntVector(x1,y1),
    IntVector(x2,y2)), 'name', color).

    Example:
    ```mlir
    csl.set_param_range_named %region rect(1, 0, 2, 4) param("b_in") color(%c)
    ```
  }];
  let arguments = (ins
    CSL_CodeRegionType:$region,
    I64Attr:$ul_x, I64Attr:$ul_y,
    I64Attr:$lr_x, I64Attr:$lr_y,
    StrAttr:$param_name,
    CSL_ColorType:$color
  );
  let assemblyFormat = [{
    $region `rect` `(` $ul_x `,` $ul_y `,` $lr_x `,` $lr_y `)`
    `param` `(` $param_name `)` `color` `(` $color `)` attr-dict
  }];
}

def CSL_InputStreamOp : CSL_Op<"input_stream", []> {
  let summary = "Create a host-to-device (H2D) I/O stream";
  let description = [{
    Creates a host-to-device I/O stream from an input port.
    Corresponds to `layout.create_input_stream(port)` in the Python SdkLayout API.

    Example:
    ```mlir
    %s = csl.input_stream %port : !csl.stream
    ```
  }];
  let arguments = (ins CSL_PortType:$port);
  let results = (outs CSL_StreamType:$result);
  let assemblyFormat = "$port attr-dict `:` type($result)";
}

def CSL_OutputStreamOp : CSL_Op<"output_stream", []> {
  let summary = "Create a device-to-host (D2H) I/O stream";
  let description = [{
    Creates a device-to-host I/O stream from an output port.
    Corresponds to `layout.create_output_stream(port)` in the Python SdkLayout API.

    Example:
    ```mlir
    %s = csl.output_stream %port : !csl.stream
    ```
  }];
  let arguments = (ins CSL_PortType:$port);
  let results = (outs CSL_StreamType:$result);
  let assemblyFormat = "$port attr-dict `:` type($result)";
}

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
