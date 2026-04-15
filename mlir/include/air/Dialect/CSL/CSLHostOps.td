//===- CSLHostOps.td - csl_host dialect ops --------------------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Ops for the csl_host dialect — lives inside csl.host{} regions and
// describes host-side data movement and kernel launch.
// Emits to run.py via the SdkRuntime Python API.
//
//===----------------------------------------------------------------------===//

#ifndef CSL_HOST_OPS
#define CSL_HOST_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// CSLHost Dialect
//===----------------------------------------------------------------------===//

def CSLHost_Dialect : Dialect {
  let name = "csl_host";
  let cppNamespace = "xilinx::csl_host";
  let description = [{
    The `csl_host` dialect describes host-side data movement and program
    launch for the Cerebras WSE.  Ops live inside `csl.host {}` regions
    and emit to `run.py` (SdkRuntime Python API).
  }];
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

class CSLHost_Op<string mnemonic, list<Trait> traits = []>
    : Op<CSLHost_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// csl_host.memcpy_h2d — copy a host buffer into a PE-local variable
//
// Format: csl_host.memcpy_h2d %src to $sym { px, py, width, height } : memref
//===----------------------------------------------------------------------===//

def CSLHost_MemcpyH2DOp : CSLHost_Op<"memcpy_h2d"> {
  let summary = "Copy host buffer to a PE-local variable";
  let description = [{
    Transfers data from host buffer `%src` into the PE-local export symbol
    named by `sym` (a nested symbol `@layout::@export_sym`).  `px`, `py`,
    `width`, `height` describe the target PE rectangle.

    ```mlir
    csl_host.memcpy_h2d %a_in to @main_layout::@a
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<256xf32>
    ```

    Emits to:
    ```python
    a_id = runner.get_id("a")
    runner.memcpy_h2d(a_id, a_in, 0, 0, 1, 1, N)
    ```
  }];

  let arguments = (ins
    AnyMemRef:$src,
    SymbolRefAttr:$sym,
    I64Attr:$px,
    I64Attr:$py,
    I64Attr:$width,
    I64Attr:$height
  );

  let assemblyFormat = "$src `to` $sym attr-dict `:` type($src)";
}

//===----------------------------------------------------------------------===//
// csl_host.memcpy_d2h — copy a PE-local variable back to a host buffer
//
// Format: csl_host.memcpy_d2h $sym to %dst { px, py, width, height } : memref
//===----------------------------------------------------------------------===//

def CSLHost_MemcpyD2HOp : CSLHost_Op<"memcpy_d2h"> {
  let summary = "Copy a PE-local variable back to a host buffer";
  let description = [{
    Transfers data from the PE-local export symbol named by `sym` into the
    host buffer `%dst`.  `px`, `py`, `width`, `height` describe the source
    PE rectangle.

    ```mlir
    csl_host.memcpy_d2h @main_layout::@c to %c_out
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<256xf32>
    ```

    Emits to:
    ```python
    c_id = runner.get_id("c")
    runner.memcpy_d2h(c_out, c_id, 0, 0, 1, 1, N)
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$sym,
    AnyMemRef:$dst,
    I64Attr:$px,
    I64Attr:$py,
    I64Attr:$width,
    I64Attr:$height
  );

  let assemblyFormat = "$sym `to` $dst attr-dict `:` type($dst)";
}

//===----------------------------------------------------------------------===//
// csl_host.launch — RPC call into a PE function
//
// Format: csl_host.launch $sym attr-dict
//===----------------------------------------------------------------------===//

def CSLHost_LaunchOp : CSLHost_Op<"launch"> {
  let summary = "RPC call into a PE function";
  let description = [{
    Issues a blocking RPC call to the PE function or task identified by
    the nested symbol reference `sym` (`@layout::@func_export`).

    ```mlir
    csl_host.launch @main_layout::@compute
    ```

    Emits to:
    ```python
    runner.launch("compute", nonblock=False)
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$sym
  );

  let assemblyFormat = "$sym attr-dict";
}

#endif // CSL_HOST_OPS
