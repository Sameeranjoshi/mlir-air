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
// csl_layout.place — bind a program template to a PE or a subgrid
//
// Two forms:
//   Point form:  csl_layout.place @prog at (px, py) attr-dict
//   Range form:  csl_layout.place @prog over [lo:hi:stride, lo:hi:stride]
//                                  vars (%i : i32, %j : i32)
//                                  params { name = %i : i16, ... } attr-dict
//===----------------------------------------------------------------------===//

def CSLLayout_PlaceOp : CSLLayout_Op<"place"> {
  let summary = "Place a csl.program at a tile or over a subgrid";
  let description = [{
    Places a csl.program template at either a single PE (point form) or over
    a rectangular subgrid of PEs (range form).

    Point form:
    ```mlir
    csl_layout.place @vecadd_pe at (0, 0)
    csl_layout.place @gemv_pe at (1, 0) {col = 1 : i16, width = 4 : i16}
    ```

    Range form (SPADA-style half-open `lo:hi[:stride]`):
    ```mlir
    csl_layout.place @pe over [0:8, 0]
    csl_layout.place @pe over [0:4, 0:4]
    csl_layout.place @pe over [0:8, 0] vars (%i : i32) params {pid = %i : i16}
    ```

    Exactly one of `at (x, y)` or `over [...]` must be used.  The
    `vars (...)` and `params {...}` clauses are only legal with the range
    form.  The induction-variable names in `vars` are purely syntactic —
    stored as string attributes (no SSA values); each `params` entry binds a
    comptime-parameter name to one of those induction-variable names.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$prog,
    // Point form
    OptionalAttr<I64Attr>:$px,
    OptionalAttr<I64Attr>:$py,
    // Range form: each array is [lo, hi, stride]
    OptionalAttr<I64ArrayAttr>:$x_range,
    OptionalAttr<I64ArrayAttr>:$y_range,
    // Induction-variable names (strings), e.g. ["i", "j"]
    OptionalAttr<StrArrayAttr>:$iv_names,
    // Parameter bindings; each value is a StringAttr referencing an iv_name
    OptionalAttr<DictionaryAttr>:$params
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
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
