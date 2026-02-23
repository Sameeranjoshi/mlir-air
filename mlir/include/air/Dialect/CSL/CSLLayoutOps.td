//===- CSLLayoutOps.td - CSL layout operations -------------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_LAYOUT_OPS
#define CSL_LAYOUT_OPS

include "air/Dialect/CSL/CSLBase.td"

def CSL_LayoutOp : CSL_Op<"layout", [
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Top-level CSL layout container";
  let description = [{
    Represents the CSL `layout { }` block that contains spatial configuration:
    PE grid dimensions, tile-to-code assignments, and host-visible exports.

    Example:
    ```mlir
    csl.layout {
      csl.set_rectangle 2, 2
      csl.set_tile_code 0, 0 file("pe_program.csl")
      csl.export_name "data" : f32
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def CSL_SetRectangleOp : CSL_Op<"set_rectangle", []> {
  let summary = "Define PE grid dimensions";
  let description = [{
    Sets the rectangular PE grid dimensions for the CSL program.
    Corresponds to `@set_rectangle(width, height)` in CSL.

    Example:
    ```mlir
    csl.set_rectangle 4, 8
    ```
  }];

  let arguments = (ins
    I64Attr:$width,
    I64Attr:$height
  );

  let assemblyFormat = "$width `,` $height attr-dict";
}

#endif // CSL_LAYOUT_OPS
