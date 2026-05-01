//===- CSLOps.td - All CSL dialect operations (consolidated) -*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Consolidated TableGen file containing all active CSL dialect operations.
// This includes routing ops, PE program ops, V2 structural ops, and
// (commented) data movement ops.
//
// Active op count: 13
//   - Routing (2): csl.color, csl.route
//   - PE Program (4): csl.func, csl.task, csl.return, csl.var
//   - Runtime (1): csl.import_module
//   - V2 Structural (5): csl.wafer, csl.program, csl.export, csl.layout, csl.host
//   - Deferred (3): csl.get_mem_dsd, csl.get_fab_dsd, csl.mov
//
//===----------------------------------------------------------------------===//

#ifndef CSL_OPS
#define CSL_OPS

include "air/Dialect/CSL/CSLBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Routing ops — color and route declaration
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl.color — Declare a communication color
//===----------------------------------------------------------------------===//

def CSL_ColorOp : CSL_Op<"color", [
    Symbol,
    HasParent<"LayoutOp">
  ]> {
  let summary = "Declare a communication color (lives in csl.layout body)";
  let description = [{
    Declares a CSL communication color. Colors are the fundamental routing
    primitive on the WSE fabric. A color symbol is shared across all PE
    programs that reference it (via csl.get_fab_dsd / csl_layout.set_color_config).

    The op is **synthesized** by --csl-materialize-stream-colors from
    csl_layout.stream ops; users do not write it directly.

    When `id` is absent, the color is "virtual" and the
    --csl-allocate-color-ids pass assigns one. When present, the color is
    pinned to that physical id and the allocator skips it.

    Examples:
    ```mlir
    csl.color @red : !csl.color                 // virtual
    csl.color @blue {id = 5 : i32} : !csl.color // pinned
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<I32Attr>:$id);

  let assemblyFormat = "$sym_name attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.route — Declare a routing configuration
//===----------------------------------------------------------------------===//

def CSL_RouteOp : CSL_Op<"route", []> {
  let summary = "Declare a routing configuration (input -> output directions)";
  let description = [{
    Declares a routing configuration with input and output directions.
    Direction values follow the Direction enum: NORTH=0, SOUTH=1, EAST=2,
    WEST=3, RAMP=4 (RAMP = PE compute unit).

    The result is a `!csl.route` handle used by layout and other ops.

    Examples:
    ```mlir
    %r1 = csl.route in(RAMP) out(EAST) : !csl.route
    %r2 = csl.route in(WEST) out(RAMP) : !csl.route
    ```
  }];

  let arguments = (ins
    CSL_DirectionEnum:$input_dir,
    CSL_DirectionEnum:$output_dir
  );
  let results = (outs CSL_RouteType:$result);

  let assemblyFormat = "`in` `(` $input_dir `)` `out` `(` $output_dir `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// PE Program ops — functions, tasks, variables
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl.func — PE function declaration
//===----------------------------------------------------------------------===//

def CSL_FuncOp : CSL_Op<"func", [
    Symbol
  ]> {
  let summary = "PE function declaration";
  let description = [{
    Declares a function that runs on a single PE. The body contains standard
    MLIR operations representing the compute kernel. Corresponds to
    `fn name() void { ... }` in CSL.

    Example:
    ```mlir
    csl.func @compute {
      csl.return
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name $body attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.task — PE task bound to a color
//===----------------------------------------------------------------------===//

def CSL_TaskOp : CSL_Op<"task", [
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "PE task bound to a color";
  let description = [{
    Defines an event-driven task on a PE. Tasks are triggered when a
    wavelet arrives on the bound color. This is the fundamental
    execution mechanism on the WSE.

    Example:
    ```mlir
    %c = csl.color {id = 3 : i32} : !csl.color
    csl.task @recv color(%c) {
      csl.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    CSL_ColorType:$color
  );
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name `color` `(` $color `)` $body attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.return — Terminator for func/task
//===----------------------------------------------------------------------===//

def CSL_ReturnOp : CSL_Op<"return", [
    Terminator,
    ParentOneOf<["FuncOp", "TaskOp"]>
  ]> {
  let summary = "Return from a PE function or task";
  let description = [{
    Terminates a csl.func or csl.task region.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.var — PE variable declaration
//===----------------------------------------------------------------------===//

def CSL_VarOp : CSL_Op<"var", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "PE variable declaration";
  let description = [{
    Declares a variable in the PE's local memory and produces an SSA
    result for use by ops in enclosing regions such as csl.func.
    Corresponds to `var name: [N]T` in CSL.

    The result carries the same type as the declared memref. The printer
    names the result after the sym_name (via getAsmResultNames) so that
    `%a_buf = csl.var @a_buf : memref<256xf32>` appears in the textual IR.

    NOTE: csl.var does not carry the Symbol trait because MLIR symbols
    cannot have SSA results. The sym_name is used purely as a human-readable
    identifier and for csl.export string matching.

    Example:
    ```mlir
    %a_buf = csl.var @a_buf : memref<1024xf32>
    %mat   = csl.var @matrix : memref<32x32xf16>
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "$sym_name `:` type($result) attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.import_module — Import an external CSL library module
//===----------------------------------------------------------------------===//

def CSL_ImportModuleOp : CSL_Op<"import_module", []> {
  let summary = "Import an external CSL library module";
  let description = [{
    Imports a named CSL library for use in PE programs. Corresponds to
    `@import_module("name")` or `@import_module("name", params)` in CSL.

    Example:
    ```mlir
    %m = csl.import_module "<memcpy/memcpy>" : !csl.imported_module
    %m2 = csl.import_module "<memcpy/get_params>" {params = {width = 2 : i32}} : !csl.imported_module
    ```
  }];

  let arguments = (ins
    StrAttr:$module_name,
    OptionalAttr<DictionaryAttr>:$params
  );
  let results = (outs CSL_ImportedModuleType:$result);

  let assemblyFormat = "$module_name attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Data movement ops
//===----------------------------------------------------------------------===//
//
// v5 scope: csl.get_mem_dsd (mem1d only).
//
// Deferred: csl.get_fab_dsd, csl.mov — fabric DSDs require colors/routes/tasks
// which are intentionally excluded from SIMD-only execution. Will be added
// when inter-PE communication lands.
//

//===----------------------------------------------------------------------===//
// csl.get_mem_dsd — 1-D memory-backed Data Structure Descriptor
//===----------------------------------------------------------------------===//

def CSL_GetMemDsdOp : CSL_Op<"get_mem_dsd", []> {
  let summary = "Create a mem1d/mem4d DSD from a strided memref";
  let description = [{
    Builds a CSL memory DSD whose shape, stride, and offset are read from
    the operand memref's type. Rank 1 emits `mem1d_dsd`; rank 2..4 emits
    `mem4d_dsd`.

    The memref's `StridedLayoutAttr` (or identity layout for plain memrefs)
    supplies `.stride = ...` and `.offset = ...`; the memref's shape supplies
    `.extent = ...`. Build a strided view of a larger buffer with
    `memref.subview` first — MLIR's canonicalization and fold patterns on
    `subview`/`strided<>` apply uniformly.

    Contiguous:
    ```mlir
    %a = csl.var @a : memref<128xf32>
    %d = csl.get_mem_dsd %a : memref<128xf32> -> !csl.dsd
    ```

    Strided (every other element of a 128-elem buffer):
    ```mlir
    %v = memref.subview %a[0] [64] [2]
         : memref<128xf32> to memref<64xf32, strided<[2]>>
    %d = csl.get_mem_dsd %v : memref<64xf32, strided<[2]>> -> !csl.dsd
    ```

    2-D row-major 8x16 (→ `mem4d_dsd` with tuple `.stride` / `.extent`):
    ```mlir
    %v = memref.subview %M[0, 0] [8, 16] [16, 1]
         : memref<128xf32> to memref<8x16xf32, strided<[16, 1]>>
    %d = csl.get_mem_dsd %v : memref<8x16xf32, strided<[16, 1]>> -> !csl.dsd
    ```

    v5 supports mem1d + mem4d. Fabric DSDs (fabin/fabout) remain deferred
    with the rest of inter-PE communication.
  }];

  let arguments = (ins AnyMemRef:$buffer);
  let results   = (outs CSL_DsdType:$result);

  let assemblyFormat = [{
    $buffer `:` type($buffer) `->` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// csl.builtin_call — generic CSL builtin / imported-module member call
//===----------------------------------------------------------------------===//

def CSL_BuiltinCallOp : CSL_Op<"builtin_call", [AttrSizedOperandSegments]> {
  let summary = "Call a CSL language builtin or imported-module member";
  let description = [{
    Generic, untyped call to either:

    - A bare CSL language builtin (e.g. `@fmacs`, `@fadds`, `@fmovs`, `@fabs`,
      `@iadds`). Emitted as `@<callee>(arg0, arg1, ...);`.

    - A member of a module imported via `csl.import_module`. The module operand
      is the SSA value of type `!csl.imported_module`. Emitted as
      `<mod>.<callee>(arg0, arg1, ...);`.

    Operand and result typing is intentionally permissive — the CSL compiler
    owns semantic validation. One op covers every DSD builtin and every
    imported-module member call, keeping the dialect size constant.

    ```mlir
    // bare-builtin form
    csl.builtin_call "fmacs"(%y, %y, %A, %xi)
        : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()

    // module-member form (emits as `<mod>.<callee>(...)` in CSL)
    %r = csl.builtin_call "sqrt" in %math (%x) : (f32) -> f32
    ```
  }];

  let arguments = (ins
    StrAttr:$callee,
    Optional<CSL_ImportedModuleType>:$module,
    Variadic<AnyType>:$args);
  let results   = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee (`in` $module^)? `(` $args `)` attr-dict
      `:` functional-type($args, $results)
  }];
}


//===----------------------------------------------------------------------===//
// CSL Dialect v2 — Structural + Kernel Ops
//
// These ops form the v2 IR family:
//   csl.wafer  — top-level WSE program container
//   csl.program — PE kernel template (one .csl file)
//   csl.export  — mark a symbol as host-visible
//   csl.layout  — placement container (emits csl_layout.py)
//   csl.host    — host runtime container (emits run.py)
//===----------------------------------------------------------------------===//

include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// csl.wafer — WSE program container (analogous to aie.device)
//===----------------------------------------------------------------------===//

def CSL_WaferOp : CSL_Op<"wafer", [
    IsolatedFromAbove,
    SymbolTable,
    NoTerminator]> {
  let summary = "WSE program container";
  let description = [{
    Top-level container for a complete WSE program. Contains csl.program,
    csl.layout, and csl.host as sibling ops sharing a single symbol table.
    Analogous to aie.device — sits inside module {}, is the logical
    program boundary.

    ```mlir
    module {
      csl.wafer @wseprog {arch = "wse3"} {
        csl.program @vecadd_pe { ... }
        csl.layout {width=1, height=1} @layout { ... }
        csl.host @main(...) {layout = @layout} { ... }
      }
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$arch);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// csl.program — PE kernel template (one per pe_program.csl file)
//===----------------------------------------------------------------------===//

def CSL_ProgramOp : CSL_Op<"program", [
    Symbol,
    SymbolTable,
    NoTerminator,
    HasParent<"WaferOp">]> {
  let summary = "PE kernel template";
  let description = [{
    Models a single pe_program.csl source file. Has no coordinates —
    placement is handled by csl.layout. Block arguments typed with
    !csl.comptime<T> map to CSL 'param' declarations.

    ```mlir
    csl.program @vecadd_pe(%col: !csl.comptime<i16>) {
      %a = csl.var @a : memref<256xf32>
      csl.func @compute { ... }
      csl.export @a {alias = "a"}
    }
    ```

    The `param_names` attribute is populated by the custom parser from the
    SSA names of the block arguments (e.g., `["col"]`).  It lets later
    passes such as `-csl-verify-params` compare user-facing parameter names
    against the attributes carried on `csl_layout.place` ops, and it is
    used by the printer so round-trip preserves the original names.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrArrayAttr>:$param_names
  );
  let regions = (region SizedRegion<1>:$body);
  // Custom format handles optional (%arg: !csl.comptime<T>, ...) block args.
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// csl.export — mark a symbol as host-visible
//===----------------------------------------------------------------------===//

def CSL_ExportOp : CSL_Op<"export", [
    HasParent<"ProgramOp">]> {
  let summary = "Mark a PE symbol as host-visible";
  let description = [{
    Marks a csl.var or csl.func as accessible from the host runtime.
    The `direction` attribute (in/out) is added by -csl-infer-exports.

    ```mlir
    csl.export @a { alias = "a" }           // var with host-facing alias
    csl.export @compute { kind = "func" }   // function export
    // After -csl-infer-exports:
    csl.export @a { alias = "a", direction = "in" }
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$sym,
    OptionalAttr<StrAttr>:$alias,
    OptionalAttr<StrAttr>:$kind,
    OptionalAttr<StrAttr>:$direction
  );
  let assemblyFormat = "$sym attr-dict";
}

//===----------------------------------------------------------------------===//
// csl.layout — placement + routing container (emits csl_layout.py)
//===----------------------------------------------------------------------===//

def CSL_LayoutOp : CSL_Op<"layout", [
    Symbol,
    SymbolTable,
    NoTerminator,
    HasParent<"WaferOp">]> {
  let summary = "Layout container mapping programs to PE rectangles";
  let description = [{
    Models the csl_layout.py file generated via sdkLayout Python API.
    Contains csl_layout.* ops for placement and host-visible exports.

    ```mlir
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe {} at (0, 0)
      csl_layout.export @a from @vecadd_pe::@a
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$width,
    I64Attr:$height
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// csl.host — host runtime orchestration (emits run.py)
//===----------------------------------------------------------------------===//

def CSL_HostOp : CSL_Op<"host", [
    Symbol,
    NoTerminator,
    HasParent<"WaferOp">]> {
  let summary = "Host runtime orchestration container";
  let description = [{
    Models run.py. Contains csl_host.* ops for memcpy and launch.
    References the layout by symbol. Block args are the host-side
    function parameters (memrefs to transfer).

    ```mlir
    csl.host @main(%a: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_h2d %a to @main_layout::@a {...}
      csl_host.launch @main_layout::@compute
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$layout
  );
  let regions = (region SizedRegion<1>:$body);
  // Custom format handles (%arg: type, ...) {layout = @sym} {body}.
  let hasCustomAssemblyFormat = 1;
}

#endif // CSL_OPS
