//===- CSLLayoutOps.td - csl_layout dialect ops ----------------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Ops for the csl_layout dialect — lives inside csl.layout{} regions and
// describes placement of csl.program templates onto PE grid rectangles.
// Emits to csl_layout.py via the sdkLayout Python API.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_LAYOUT_OPS
#define CSL_LAYOUT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// CSLLayout Dialect
//===----------------------------------------------------------------------===//

def CSLLayout_Dialect : Dialect {
  let name = "csl_layout";
  let cppNamespace = "xilinx::csl_layout";
  let description = [{
    The `csl_layout` dialect describes placement and routing of PE kernels.
    Ops in this dialect live inside `csl.layout {}` regions and are lowered
    to the sdkLayout Python API by `--emit-csl-layout`.
  }];
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

class CSLLayout_Op<string mnemonic, list<Trait> traits = []>
    : Op<CSLLayout_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// csl_layout.place — bind a program template to a single PE grid cell
//
// Format: csl_layout.place @prog at (px, py) attr-dict
//
// The attr-dict carries the concrete comptime parameter bindings
// that are passed to @set_tile_code / region.set_param().
//===----------------------------------------------------------------------===//

def CSLLayout_PlaceOp : CSLLayout_Op<"place"> {
  let summary = "Bind a csl.program template to a single PE at (px, py)";
  let description = [{
    Places a single instance of the referenced `csl.program` at PE grid
    coordinate (`px`, `py`).  Comptime parameter bindings are carried as
    extra attributes in the attr-dict.

    ```mlir
    csl_layout.place @vecadd_pe at (0, 0)
    csl_layout.place @gemv_pe at (1, 0) {col = 1 : i16, width = 4 : i16}
    ```

    Emits to:
    ```python
    region.place(px, py)
    region.set_param(IntVector(px, py), "col", 1)
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$prog,
    I64Attr:$px,
    I64Attr:$py
  );

  let assemblyFormat = "$prog `at` `(` $px `,` $py `)` attr-dict";
}

//===----------------------------------------------------------------------===//
// csl_layout.place_grid — bind a program template to a PE rectangle (SPMD)
//
// Format: csl_layout.place_grid @prog from (x0, y0) to (x1, y1) attr-dict
//
// Expanded to N `csl_layout.place` ops by -csl-elaborate-grid pass.
// Per-PE placeholder params use $col/$row syntax in attr-dict (future).
//===----------------------------------------------------------------------===//

def CSLLayout_PlaceGridOp : CSLLayout_Op<"place_grid"> {
  let summary = "Bind a csl.program template to a PE rectangle (SPMD)";
  let description = [{
    V1 stub — accepted by the parser but not yet lowered by any pass.
    Lowers to N `csl_layout.place` ops via `-csl-elaborate-grid`.

    ```mlir
    csl_layout.place_grid @gemv_pe from (0, 0) to (3, 0) {M = 4 : i16, N = 6 : i16}
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$prog,
    I64Attr:$x0, I64Attr:$y0,
    I64Attr:$x1, I64Attr:$y1
  );

  let assemblyFormat =
    "$prog `from` `(` $x0 `,` $y0 `)` `to` `(` $x1 `,` $y1 `)` attr-dict";
}

//===----------------------------------------------------------------------===//
// csl_layout.export — make a PE symbol host-visible through the layout
//
// Format: csl_layout.export $sym_name from $from attr-dict
//===----------------------------------------------------------------------===//

def CSLLayout_ExportOp : CSLLayout_Op<"export"> {
  let summary = "Re-export a PE symbol as host-visible through the layout";
  let description = [{
    Declares that a symbol inside a placed program is accessible from the
    host.  The `from` attribute is a nested symbol reference
    `@program::@var`.  The optional `type_attr` attribute records the memref
    type for host-side memcpy sizing; `mutable` indicates a writable tensor.

    ```mlir
    csl_layout.export "a" from @vecadd_pe::@a
    csl_layout.export "compute" from @vecadd_pe::@compute {kind = "func"}
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    SymbolRefAttr:$from,
    OptionalAttr<StrAttr>:$kind
  );

  let assemblyFormat = "$sym_name `from` $from attr-dict";
}

#endif // CSL_LAYOUT_OPS
