//===- CSLPlacementOps.td - CSL placement operations -------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Operations for assigning CSL code modules to PE tiles on the WSE grid.
// Corresponds to `@set_tile_code(x, y, "file.csl", params)` in CSL.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_PLACEMENT_OPS
#define CSL_PLACEMENT_OPS

include "air/Dialect/CSL/CSLBase.td"

//===----------------------------------------------------------------------===//
// csl.set_tile_code
//===----------------------------------------------------------------------===//

def CSL_SetTileCodeOp : CSL_Op<"set_tile_code", []> {
  let summary = "Assign a CSL module to a PE at grid position (x, y)";
  let description = [{
    Maps a CSL source file to a specific PE tile in the grid. Corresponds
    to `@set_tile_code(x, y, "file.csl", .{ params })` in CSL.

    The optional `params` dictionary carries compile-time parameters
    forwarded to the PE module.

    Example:
    ```mlir
    csl.set_tile_code 0, 1 file("pe_program.csl")
    csl.set_tile_code 1, 1 file("pe_program.csl") params({col = 1 : i32})
    ```
  }];

  let arguments = (ins
    I64Attr:$x,
    I64Attr:$y,
    StrAttr:$file,
    OptionalAttr<DictionaryAttr>:$params
  );

  let assemblyFormat = [{
    $x `,` $y `file` `(` $file `)` (`params` `(` $params^ `)`)? attr-dict
  }];
}

#endif // CSL_PLACEMENT_OPS
