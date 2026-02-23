//===- CSLKernelOps.td - CSL kernel operations -------------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_KERNEL_OPS
#define CSL_KERNEL_OPS

include "air/Dialect/CSL/CSLBase.td"
include "mlir/IR/SymbolInterfaces.td"

def CSL_FuncOp : CSL_Op<"func", [
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "CSL function definition";
  let description = [{
    Defines a CSL function on a PE. The body contains standard MLIR
    operations (arith, scf, memref) representing the compute kernel.
    Corresponds to `fn name() void { ... }` in CSL.

    Example:
    ```mlir
    csl.func @compute() {
      csl.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<DictionaryAttr>:$arg_attrs,
    OptionalAttr<DictionaryAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
}

def CSL_TaskOp : CSL_Op<"task", [
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "CSL event-driven task bound to a color";
  let description = [{
    Defines an event-driven task on a PE. Tasks are triggered when a
    wavelet arrives on the bound color. This is the fundamental
    execution mechanism on the WSE.

    Example:
    ```mlir
    csl.task @recv_task() color(3) {
      csl.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I32Attr:$color_id
  );
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
}

def CSL_ReturnOp : CSL_Op<"return", [
    Terminator,
    ParentOneOf<["FuncOp", "TaskOp"]>
  ]> {
  let summary = "Return from a CSL function or task";
  let assemblyFormat = "attr-dict";
}

def CSL_VarOp : CSL_Op<"var", [Symbol]> {
  let summary = "Declare a PE-local variable";
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

#endif // CSL_KERNEL_OPS
