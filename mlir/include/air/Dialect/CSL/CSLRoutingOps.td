//===- CSLRoutingOps.td - CSL routing operations -----------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Operations for CSL color-based routing configuration. Colors are the
// primary communication abstraction on the WSE: each color identifies a
// logical channel, and routes bind colors to fabric directions.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_ROUTING_OPS
#define CSL_ROUTING_OPS

include "air/Dialect/CSL/CSLBase.td"

//===----------------------------------------------------------------------===//
// csl.color
//===----------------------------------------------------------------------===//

def CSL_ColorOp : CSL_Op<"color", []> {
  let summary = "Declare a communication color";
  let description = [{
    Declares a CSL communication color with the given integer ID.
    Colors are the fundamental routing primitive on the WSE fabric.

    Example:
    ```mlir
    %c0 = csl.color 0 : !csl.color
    %c1 = csl.color 1 : !csl.color
    ```
  }];

  let arguments = (ins I32Attr:$id);
  let results = (outs CSL_ColorType:$result);

  let assemblyFormat = "$id attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// csl.route
//===----------------------------------------------------------------------===//

def CSL_RouteOp : CSL_Op<"route", []> {
  let summary = "Configure routing direction for a color";
  let description = [{
    Binds a color to a fabric direction (NORTH, SOUTH, EAST, WEST, RAMP).
    This determines how wavelets tagged with this color propagate through
    the on-chip network.

    Example:
    ```mlir
    %c = csl.color 0 : !csl.color
    csl.route %c dir(EAST)
    ```
  }];

  let arguments = (ins
    CSL_ColorType:$color,
    CSL_DirectionEnum:$direction
  );

  let assemblyFormat = "$color `dir` `(` $direction `)` attr-dict";
}

#endif // CSL_ROUTING_OPS
