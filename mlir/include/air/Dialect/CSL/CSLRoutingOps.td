//===- CSLRoutingOps.td - CSL routing operations -----------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Operations for CSL color and route declarations. Colors are the primary
// communication abstraction on the WSE: each color identifies a logical
// channel. Routes bind directions to color channels for wavelet routing.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_ROUTING_OPS
#define CSL_ROUTING_OPS

include "air/Dialect/CSL/CSLBase.td"

//===----------------------------------------------------------------------===//
// csl.color
//===----------------------------------------------------------------------===//
//
// Declares a CSL communication color. When an integer ID is provided, the
// color is bound to that physical color ID. When no ID is given, the
// compiler assigns a physical color ID at compile time.
//
// Examples:
//   ```mlir
//   %red   = csl.color : !csl.color
//   %green = csl.color : !csl.color
//   %c0    = csl.color 0 : !csl.color
//   ```
//
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
    %c0  = csl.color 0 : !csl.color
    ```
  }];

  let arguments = (ins OptionalAttr<I32Attr>:$id);
  let results = (outs CSL_ColorType:$result);

  let assemblyFormat = "($id^)? attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// csl.route
//===----------------------------------------------------------------------===//
//
// Declares a routing configuration with an input direction and an output
// direction. Direction values use the Direction enum:
//   NORTH=0, SOUTH=1, EAST=2, WEST=3, RAMP=4
// where RAMP represents the PE compute unit.
//
// The result is an i32 SSA handle used by code_region, paint, and port ops.
//
// Examples:
//   ```mlir
//   %r1 = csl.route in(RAMP) out(WEST) : i32
//   %r2 = csl.route in(RAMP) out(EAST) : i32
//   %r3 = csl.route in(WEST) out(RAMP) : i32
//   ```
//
def CSL_RouteOp : CSL_Op<"route", []> {
  let summary = "Declare a routing direction pair (input -> output)";
  let description = [{
    Declares a routing configuration with input and output directions.
    Direction values follow the Direction enum: NORTH=0, SOUTH=1, EAST=2,
    WEST=3, RAMP=4 (RAMP = PE compute unit).

    The result is an i32 SSA handle used by code_region, paint, and port ops.

    Examples:
    ```mlir
    %r1 = csl.route in(RAMP) out(WEST) : i32
    %r2 = csl.route in(WEST) out(RAMP) : i32
    ```
  }];

  let arguments = (ins
    OptionalAttr<CSL_DirectionEnum>:$input_dir,
    OptionalAttr<CSL_DirectionEnum>:$output_dir
  );
  let results = (outs I32:$result);

  let assemblyFormat = [{
    (`in` `(` $input_dir^ `)`)? (`out` `(` $output_dir^ `)`)? attr-dict `:` type($result)
  }];
}

#endif // CSL_ROUTING_OPS
