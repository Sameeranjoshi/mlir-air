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

//===----------------------------------------------------------------------===//
// csl.kernel
//===----------------------------------------------------------------------===//
//
// A kernel container that groups PE-level compute (functions, tasks,
// variables, and parameters) together with a CSL source file reference
// and optional compile-time parameters. Unlike `csl.module` (which is
// IsolatedFromAbove and uses SymbolTable), `csl.kernel` is designed for
// the spatial placement programming model where kernels are created as
// SSA values and bound to code regions via `csl.place`.
//
// The body can contain `csl.func`, `csl.task`, `csl.var`, `csl.param`,
// and `csl.comptime` ops — the same ops that live inside `csl.module`.
//
// Example:
//   ```mlir
//   %k = csl.kernel "pe_program.csl" params({width = 100 : i32}) {
//     csl.var @buf : memref<1024xf32>
//     csl.func @compute() {
//       csl.return
//     }
//     csl.comptime {
//       csl.export_symbol @compute
//     }
//   } : !csl.kernel
//   ```
//
def CSL_KernelOp : CSL_Op<"kernel", [
    SingleBlock,
    NoTerminator,
    SymbolTable
  ]> {
  let summary = "Kernel container grouping PE-level compute with a CSL source";
  let description = [{
    A kernel container that groups PE-level compute operations (functions,
    tasks, variables, and parameters) together with a CSL source file path
    and optional compile-time parameters. Returns a `!csl.kernel` handle
    that can be bound to a code region via `csl.place`.

    The body can contain `csl.func`, `csl.task`, `csl.var`, `csl.param`,
    and `csl.comptime` ops.

    Example:
    ```mlir
    %k = csl.kernel "pe_program.csl" params({width = 100 : i32}) {
      csl.var @buf : memref<1024xf32>
      csl.func @compute() {
        csl.return
      }
    } : !csl.kernel
    ```
  }];

  let arguments = (ins
    StrAttr:$source_file,
    OptionalAttr<DictionaryAttr>:$params
  );
  let results = (outs CSL_KernelType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $source_file (`params` `(` $params^ `)`)? $body attr-dict `:` type($result)
  }];
}

#endif // CSL_KERNEL_OPS
