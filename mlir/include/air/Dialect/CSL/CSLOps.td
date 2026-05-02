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

    The op is **synthesized** by --csl-materialize-dataflow-colors from
    csl_layout.dataflow ops; users do not write it directly.

    When `id` is absent, the color is "virtual" and the
    --csl-allocate-color-ids pass assigns one. When present, the color is
    pinned to that physical id and the allocator skips it.

    Examples:
    ```mlir
    csl.color @red                  // virtual
    csl.color @blue {id = 5 : i32}  // pinned
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
// csl.task — PE task with attribute-driven trigger binding
//===----------------------------------------------------------------------===//

def CSL_TaskOp : CSL_Op<"task", [
    Symbol
  ]> {
  let summary = "PE task with attribute-driven trigger binding";
  let description = [{
    Defines an event-driven task on a PE. The trigger binding is expressed
    via attributes:

    - `trigger_kind = "local_task_id"`: must also set `id` (i32). The
      task fires when `@get_local_task_id(N)` is activated. Used for
      send/recv completion.
    - `trigger_kind = "color"`: must also set `color` (FlatSymbolRefAttr
      to a csl.color). The task fires when a wavelet arrives on that color.
      Used for control tasks (no body arguments).

    - `trigger_kind = "data_task"`: must also set `color` (FlatSymbolRefAttr
      to a csl.color). The task fires when a wavelet arrives on that color
      and the wavelet payload is delivered as the task body's first block
      argument. The body must have at least one block argument (the f32
      payload). On WSE-3 the task ID comes from `@get_data_task_id(input_queue)`.
      Use `csl.dataflow.send_wavelet` inside the body to forward wavelets.

    The emitter generates the corresponding CSL triplet (id const, task
    body, comptime bind) automatically.

    Examples:
    ```mlir
    csl.task @exit_task attributes {trigger_kind = "local_task_id", id = 8 : i32} {
      csl.return
    }
    csl.task @recv attributes {trigger_kind = "color", color = @send_color} {
      csl.return
    }
    csl.task @relay attributes {trigger_kind = "data_task", color = @ch01_color} {
    ^bb0(%val: f32):
      csl.dataflow.send_wavelet @ch12 value(%val) : f32
      csl.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$trigger_kind,
    OptionalAttr<I32Attr>:$id,
    OptionalAttr<FlatSymbolRefAttr>:$color
  );
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
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
// csl.get_fab_dsd — Fabric DSD on a color (fabin or fabout)
//===----------------------------------------------------------------------===//

def CSL_GetFabDsdOp : CSL_Op<"get_fab_dsd", []> {
  let summary = "Build a fabin/fabout DSD on a color";
  let description = [{
    Builds a CSL fabric DSD that sends (`fabout`) to or receives (`fabin`)
    from the fabric on the given color, with a runtime extent in elements.

    The color is referenced by symbol name (FlatSymbolRefAttr) — it must
    resolve to a `csl.color` op in the enclosing `csl.layout` body.

    Synthesized by `--csl-lower-dataflow-data` from `csl.dataflow.put`/`get`.

    ```mlir
    %n = arith.constant 128 : index
    %out = csl.get_fab_dsd fabout @send_color extent(%n : index) : !csl.dsd
    %in  = csl.get_fab_dsd fabin  @send_color extent(%n : index) : !csl.dsd
    ```

    Emits to CSL:
    ```csl
    const out_dsd = @get_dsd(fabout_dsd,
        .{ .extent = 128, .fabric_color = send_color });
    ```
  }];

  let arguments = (ins
    CSL_FabDsdDirectionAttr:$direction,
    FlatSymbolRefAttr:$color,
    Index:$extent
  );
  let results = (outs CSL_DsdType:$result);

  let assemblyFormat =
      "$direction $color `extent` `(` $extent `:` type($extent) `)` "
      "attr-dict `:` type($result)";
  let hasVerifier = 1;
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
    Variadic<AnyType>:$args,
    UnitAttr:$async,
    OptionalAttr<FlatSymbolRefAttr>:$activate);
  let results   = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee (`in` $module^)? `(` $args `)` attr-dict
      `:` functional-type($args, $results)
  }];

  let hasVerifier = 1;
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

//===----------------------------------------------------------------------===//
// csl.dataflow.put — Send a buffer through a declared stream
//===----------------------------------------------------------------------===//

def CSL_DataflowPutOp : CSL_Op<"dataflow.put", []> {
  let summary = "Send a memref buffer through an inter-PE stream";
  let description = [{
    Sends `extent` elements from `source` through the named stream. The
    stream symbol must resolve to a `csl_layout.dataflow` whose `from` coord
    matches the parent program's placement.

    Sync-looking at the user level; `--csl-lower-dataflow-data` expands this
    to fabric DSD + async builtin + completion task.

    ```mlir
    %n = arith.constant 128 : index
    csl.dataflow.put @send_ch source(%buf) extent(%n : index)
                   : memref<128xf32>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$stream,
    AnyMemRef:$source,
    Index:$extent
  );

  let assemblyFormat =
      "$stream `source` `(` $source `)` `extent` `(` $extent `:` "
      "type($extent) `)` `:` type($source) attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// csl.dataflow.get — Receive a buffer through a declared stream
//===----------------------------------------------------------------------===//

def CSL_DataflowGetOp : CSL_Op<"dataflow.get", []> {
  let summary = "Receive into a memref buffer through an inter-PE stream";
  let description = [{
    Receives `extent` elements through the named stream into `target`. The
    stream symbol must resolve to a `csl_layout.dataflow` whose `to` coord
    matches the parent program's placement.

    Sync-looking at the user level; `--csl-lower-dataflow-data` expands this
    to fabric DSD + async builtin + completion task.

    ```mlir
    %n = arith.constant 128 : index
    csl.dataflow.get @send_ch target(%buf) extent(%n : index)
                   : memref<128xf32>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$stream,
    AnyMemRef:$target,
    Index:$extent
  );

  let assemblyFormat =
      "$stream `target` `(` $target `)` `extent` `(` $extent `:` "
      "type($extent) `)` `:` type($target) attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// csl.dataflow.send_wavelet — Forward a single scalar value per-wavelet
//===----------------------------------------------------------------------===//

def CSL_DataflowSendWaveletOp : CSL_Op<"dataflow.send_wavelet", []> {
  let summary = "Send a single wavelet on a dataflow channel (used inside data tasks)";
  let description = [{
    Sends a single scalar value as one wavelet on the named dataflow channel.
    Must be used inside a `csl.task` with `trigger_kind = "data_task"`.
    The stream symbol must resolve to a `csl_layout.dataflow` op.

    Lowered by `--csl-lower-dataflow-data` to a 1-element fabout DSD and a
    synchronous `@fmovs` call (no async, no activate task).

    ```mlir
    csl.task @relay attributes {trigger_kind = "data_task", color = @ch01_color} {
    ^bb0(%val: f32):
      %delta = arith.constant -100.0 : f32
      %nv = arith.addf %val, %delta : f32
      csl.dataflow.send_wavelet @ch12 value(%nv) : f32
      csl.return
    }
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$stream,
    AnyType:$value
  );
  let assemblyFormat =
      "$stream `value` `(` $value `)` `:` type($value) attr-dict";
}

#endif // CSL_OPS
