//===- CSLRuntimeOps.td - CSL runtime operations ----------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_RUNTIME_OPS
#define CSL_RUNTIME_OPS

include "air/Dialect/CSL/CSLBase.td"
include "mlir/IR/SymbolInterfaces.td"

def CSL_ImportModuleOp : CSL_Op<"import_module", []> {
  let summary = "Import a CSL library module";
  let description = [{
    Imports a CSL module by name, optionally passing compile-time
    parameters. Corresponds to `@import_module("name", params)` in CSL.

    Example:
    ```mlir
    %mod = csl.import_module "<memcpy/memcpy>" : !csl.imported_module
    %mod2 = csl.import_module "<memcpy/get_params>" params({width = 2 : i32}) : !csl.imported_module
    ```
  }];

  let arguments = (ins
    StrAttr:$module_name,
    OptionalAttr<DictionaryAttr>:$params
  );
  let results = (outs CSL_ImportedModuleType:$result);

  let assemblyFormat = [{
    $module_name (`params` `(` $params^ `)`)? attr-dict `:` type($result)
  }];
}

def CSL_ExportNameOp : CSL_Op<"export_name", []> {
  let summary = "Declare a host-visible symbol in the layout";
  let description = [{
    Exports a named symbol to the host runtime, making it accessible
    via `SdkRuntime` API calls. Used inside `csl.layout` blocks.
    Corresponds to `@export_name("name", type, is_ptr)` in CSL.

    Example:
    ```mlir
    csl.export_name "data" : f32
    csl.export_name "compute" : () -> ()
    ```
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$type
  );

  let assemblyFormat = "$sym_name `:` $type attr-dict";
}

def CSL_ExportSymbolOp : CSL_Op<"export_symbol", []> {
  let summary = "Export a symbol from a PE module";
  let description = [{
    Exports a symbol from the PE module, making it accessible from the
    layout or host. Typically used inside `csl.comptime` blocks.
    Corresponds to `@export_symbol(sym, "name")` in CSL.

    Example:
    ```mlir
    csl.export_symbol @compute
    csl.export_symbol @buf_ptr alias("data")
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$sym,
    OptionalAttr<StrAttr>:$alias
  );

  let assemblyFormat = "$sym (`alias` `(` $alias^ `)`)? attr-dict";
}

def CSL_ComptimeOp : CSL_Op<"comptime", [
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Compile-time evaluation block";
  let description = [{
    A region evaluated at compile time by the CSL compiler. Used for
    binding tasks to colors, exporting symbols, and other static
    configuration. Corresponds to `comptime { ... }` in CSL.

    Example:
    ```mlir
    csl.comptime {
      csl.export_symbol @compute
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def CSL_ModuleOp : CSL_Op<"module", [
    IsolatedFromAbove,
    Symbol,
    SymbolTable,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "CSL module (PE program) definition";
  let description = [{
    Defines a CSL module that can be assigned to PE tiles. Contains
    parameter declarations, variable declarations, functions, tasks,
    and a comptime block.

    Example:
    ```mlir
    csl.module @pe_program {
      csl.param @memcpy_params : i64
      csl.var @buf : memref<1024xf32>
      csl.func @compute() { csl.return }
      csl.comptime { csl.export_symbol @compute }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$sym_name $body attr-dict";
}

def CSL_ParamOp : CSL_Op<"param", [Symbol]> {
  let summary = "Declare a compile-time module parameter";
  let description = [{
    Declares a compile-time parameter for a CSL module. Parameters
    are specialized per-tile when assigned via `csl.set_tile_code`.
    Corresponds to `param name: type;` in CSL.

    Example:
    ```mlir
    csl.param @memcpy_params : i64
    csl.param @tile_id : i32
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type
  );

  let assemblyFormat = "$sym_name `:` $type attr-dict";
}

#endif // CSL_RUNTIME_OPS
