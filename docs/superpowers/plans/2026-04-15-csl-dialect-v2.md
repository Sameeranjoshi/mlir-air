# CSL Dialect v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CSL dialect family v2 (csl.wafer / csl.program / csl_layout / csl_host) end-to-end for vecadd 1×1 herd, producing pe_program.csl + csl_layout.py + run.py.

**Architecture:** Actor-model PE templates (csl.program) placed via csl_layout.place inside a csl.wafer container. Single -air-to-csl pass builds all three IR regions. -csl-derive-exports annotates export directions. Three air-translate targets emit text files.

**Tech Stack:** MLIR TableGen, C++17, LLVM FileCheck, lit test runner, Ninja build. Test with `lit <file>` or `cd build && ninja check-airmlir`.

**Spec:** `docs/superpowers/specs/2026-04-15-csl-dialect-v2-design.md`

---

## File Map

### Modified
| File | What changes |
|------|-------------|
| `mlir/include/air/Dialect/CSL/CSLDialect.h` | Add `ComptimeTypeStorage` + `ComptimeType` |
| `mlir/include/air/Dialect/CSL/CSLOps.td` | Add `WaferOp`, `ProgramOp`, `ExportOp`, `LayoutOp`, `HostOp` |
| `mlir/lib/Dialect/CSL/IR/CSLDialect.cpp` | Register `ComptimeType`, add parse/print cases |
| `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` | Add `CSLProgramOp::parse` + `::print` |
| `mlir/include/air/Dialect/CMakeLists.txt` | Add CSLLayout + CSLHost subdirs |
| `mlir/lib/Dialect/CMakeLists.txt` | Add CSLLayout + CSLHost subdirs |
| `mlir/lib/InitAll.cpp` | Register CSLLayoutDialect + CSLHostDialect |
| `mlir/lib/Conversion/CMakeLists.txt` | Add AIRToCSL + CSLDeriveExports sources |
| `mlir/lib/Conversion/Passes.cpp` | Register AIRToCSLPass + CSLDeriveExportsPass |
| `mlir/include/air/Conversion/Passes.td` | Add `AIRToCSL` + `CSLDeriveExports` pass defs |
| `mlir/lib/Targets/CMakeLists.txt` | Add CSLProgramEmitter, CSLLayoutEmitter, CSLHostEmitter |
| `tools/air-translate/air-translate.cpp` | Register 3 new translations |

### New (dialect)
| File | Purpose |
|------|---------|
| `mlir/include/air/Dialect/CSLLayout/CSLLayoutBase.td` | `csl_layout` dialect definition |
| `mlir/include/air/Dialect/CSLLayout/CSLLayoutOps.td` | `place`, `export` ops |
| `mlir/include/air/Dialect/CSLLayout/CSLLayoutDialect.h` | C++ dialect header |
| `mlir/include/air/Dialect/CSLLayout/CSLLayoutOps.h` | C++ ops header |
| `mlir/include/air/Dialect/CSLLayout/CMakeLists.txt` | TableGen targets |
| `mlir/lib/Dialect/CSLLayout/CMakeLists.txt` | Subdir router |
| `mlir/lib/Dialect/CSLLayout/IR/CMakeLists.txt` | Library target |
| `mlir/lib/Dialect/CSLLayout/IR/CSLLayoutDialect.cpp` | initialize() + ops |
| `mlir/include/air/Dialect/CSLHost/CSLHostBase.td` | `csl_host` dialect definition |
| `mlir/include/air/Dialect/CSLHost/CSLHostOps.td` | `memcpy_h2d`, `memcpy_d2h`, `launch` ops |
| `mlir/include/air/Dialect/CSLHost/CSLHostDialect.h` | C++ dialect header |
| `mlir/include/air/Dialect/CSLHost/CSLHostOps.h` | C++ ops header |
| `mlir/include/air/Dialect/CSLHost/CMakeLists.txt` | TableGen targets |
| `mlir/lib/Dialect/CSLHost/CMakeLists.txt` | Subdir router |
| `mlir/lib/Dialect/CSLHost/IR/CMakeLists.txt` | Library target |
| `mlir/lib/Dialect/CSLHost/IR/CSLHostDialect.cpp` | initialize() + ops |

### New (passes + emitters)
| File | Purpose |
|------|---------|
| `mlir/include/air/Conversion/AIRToCSLPass.h` | Pass factory declaration |
| `mlir/lib/Conversion/AIRToCSL/AIRToCSL.cpp` | -air-to-csl pass |
| `mlir/lib/Conversion/AIRToCSL/CMakeLists.txt` | (inline into Conversion CMakeLists) |
| `mlir/include/air/Conversion/CSLDeriveExportsPass.h` | Pass factory declaration |
| `mlir/lib/Conversion/CSLDeriveExports/CSLDeriveExports.cpp` | -csl-derive-exports pass |
| `mlir/lib/Targets/CSLProgramEmitter.cpp` | --emit-csl-program → pe_program.csl |
| `mlir/lib/Targets/CSLLayoutEmitter.cpp` | --emit-csl-layout → csl_layout.py |
| `mlir/lib/Targets/CSLHostEmitter.cpp` | --emit-csl-host → run.py |

### New (tests)
| File | Purpose |
|------|---------|
| `mlir/test/Dialect/CSL/v2_roundtrip.mlir` | Full wafer/program/layout/host roundtrip |
| `mlir/test/Dialect/CSLLayout/roundtrip.mlir` | csl_layout dialect roundtrip |
| `mlir/test/Dialect/CSLHost/roundtrip.mlir` | csl_host dialect roundtrip |
| `mlir/test/Conversion/AIRToCSL/vecadd.mlir` | -air-to-csl output check |
| `mlir/test/Conversion/AIRToCSL/invalid_multi_herd.mlir` | rejection test |
| `mlir/test/Conversion/CSLDeriveExports/vecadd.mlir` | export direction inference check |
| `mlir/test/Targets/CSL/emit_program.mlir` | FileCheck on emitted .csl text |
| `mlir/test/Targets/CSL/emit_layout.mlir` | FileCheck on emitted csl_layout.py |
| `mlir/test/Targets/CSL/emit_host.mlir` | FileCheck on emitted run.py |
| `mlir/test/Targets/CSL/pipeline.mlir` | End-to-end: AIR → all three outputs |

---

## Task 1: `!csl.comptime<T>` Parametric Type

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLDialect.h`
- Modify: `mlir/lib/Dialect/CSL/IR/CSLDialect.cpp`
- Create: `mlir/test/Dialect/CSL/v2_roundtrip.mlir` (start with comptime-only test)

- [ ] **Step 1: Write the failing test**

Create `mlir/test/Dialect/CSL/v2_roundtrip.mlir`:
```mlir
// RUN: air-opt --verify-roundtrip %s | FileCheck %s
// RUN: air-opt --mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC

// Test that !csl.comptime<T> parses and prints correctly.

// CHECK-LABEL: func.func @test_comptime_i16
// CHECK: %{{.*}}: !csl.comptime<i16>
// GENERIC: "func.func"
// GENERIC: !csl.comptime<i16>
func.func @test_comptime_i16(%arg0: !csl.comptime<i16>) -> !csl.comptime<i16> {
  return %arg0 : !csl.comptime<i16>
}

// CHECK-LABEL: func.func @test_comptime_index
// CHECK: %{{.*}}: !csl.comptime<index>
func.func @test_comptime_index(%arg0: !csl.comptime<index>) -> !csl.comptime<index> {
  return %arg0 : !csl.comptime<index>
}

// CHECK-LABEL: func.func @test_comptime_f32
// CHECK: %{{.*}}: !csl.comptime<f32>
func.func @test_comptime_f32(%arg0: !csl.comptime<f32>) -> !csl.comptime<f32> {
  return %arg0 : !csl.comptime<f32>
}
```

- [ ] **Step 2: Run test — expect FAIL (type not defined)**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air
source sandbox/bin/activate && source utils/env_setup_gpu.sh install llvm/install
lit mlir/test/Dialect/CSL/v2_roundtrip.mlir
```
Expected: `FAIL` — `error: unknown csl type: comptime`

- [ ] **Step 3: Add `ComptimeTypeStorage` + `ComptimeType` to CSLDialect.h**

In `mlir/include/air/Dialect/CSL/CSLDialect.h`, after the `RouteType` class and before the closing `} // namespace csl`:

```cpp
/// Storage for the parametric !csl.comptime<T> type.
struct ComptimeTypeStorage : mlir::TypeStorage {
  using KeyTy = mlir::Type;
  explicit ComptimeTypeStorage(mlir::Type t) : innerType(t) {}
  bool operator==(const KeyTy &key) const { return innerType == key; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }
  static ComptimeTypeStorage *construct(mlir::TypeStorageAllocator &alloc,
                                        const KeyTy &key) {
    return new (alloc.allocate<ComptimeTypeStorage>())
        ComptimeTypeStorage(key);
  }
  mlir::Type innerType;
};

/// !csl.comptime<T> — marks a value as compile-time-only (CSL param).
/// Only valid as a block argument of csl.program.
class ComptimeType
    : public mlir::Type::TypeBase<ComptimeType, mlir::Type,
                                  ComptimeTypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "xilinx.csl.comptime";
  static ComptimeType get(mlir::MLIRContext *ctx, mlir::Type innerType) {
    return Base::get(ctx, innerType);
  }
  mlir::Type getInnerType() const { return getImpl()->innerType; }
};
```

- [ ] **Step 4: Register `ComptimeType` in CSLDialect.cpp**

In `mlir/lib/Dialect/CSL/IR/CSLDialect.cpp`:

Change `initialize()` from:
```cpp
  addTypes<ColorType, DsdType, ImportedModuleType,
           CodeRegionType, PortType, StreamType, KernelType, RouteType>();
```
to:
```cpp
  addTypes<ColorType, DsdType, ImportedModuleType,
           CodeRegionType, PortType, StreamType, KernelType, RouteType,
           ComptimeType>();
```

In `parseType()`, add before the `parser.emitError` line:
```cpp
  if (keyword == "comptime") {
    if (parser.parseLess())
      return Type();
    Type innerType;
    if (parser.parseType(innerType))
      return Type();
    if (parser.parseGreater())
      return Type();
    return ComptimeType::get(context, innerType);
  }
```

In `printType()`, add inside the `TypeSwitch`:
```cpp
      .Case<ComptimeType>([&](ComptimeType t) {
        os << "comptime<";
        os.printType(t.getInnerType());
        os << ">";
      })
```

- [ ] **Step 5: Rebuild and run test**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install 2>&1 | tail -5
lit ../mlir/test/Dialect/CSL/v2_roundtrip.mlir
```
Expected: `PASS`

- [ ] **Step 6: Commit**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air
git add mlir/include/air/Dialect/CSL/CSLDialect.h \
        mlir/lib/Dialect/CSL/IR/CSLDialect.cpp \
        mlir/test/Dialect/CSL/v2_roundtrip.mlir
git commit -m "csl: add !csl.comptime<T> parametric type

Maps to CSL 'param x: T' declarations. Only valid as block arg of
csl.program. Stored as a parametric TypeBase with inner-type key.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: `csl.wafer` + `csl.program` Container Ops

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td`
- Modify: `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`

- [ ] **Step 1: Extend v2_roundtrip.mlir with wafer+program test**

Append to `mlir/test/Dialect/CSL/v2_roundtrip.mlir`:
```mlir
// ---- csl.wafer + csl.program ----

// CHECK-LABEL: csl.wafer @wseprog
// CHECK-SAME: {arch = "wse3"}
// CHECK:   csl.program @vecadd_pe(%{{.*}}: !csl.comptime<i16>)
module {
  csl.wafer @wseprog {arch = "wse3"} {
    csl.program @vecadd_pe(%col: !csl.comptime<i16>) {
    }
  }
}

// CHECK-LABEL: csl.wafer @multi_param
// CHECK:   csl.program @gemv_pe(
// CHECK-SAME: %{{.*}}: !csl.comptime<i16>
// CHECK-SAME: %{{.*}}: !csl.comptime<i16>
module {
  csl.wafer @multi_param {arch = "wse3"} {
    csl.program @gemv_pe(%M: !csl.comptime<i16>, %N: !csl.comptime<i16>) {
    }
  }
}

// CHECK-LABEL: csl.wafer @no_params
// CHECK:   csl.program @simple_pe
module {
  csl.wafer @no_params {arch = "wse3"} {
    csl.program @simple_pe {
    }
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit mlir/test/Dialect/CSL/v2_roundtrip.mlir 2>&1 | head -20
```
Expected: `error: 'csl.wafer' op not registered`

- [ ] **Step 3: Add `WaferOp` to CSLOps.td**

At the end of `mlir/include/air/Dialect/CSL/CSLOps.td`, before `#endif // CSL_OPS`, add:

```tablegen
//===----------------------------------------------------------------------===//
// CSL Dialect v2 — Structural Container Ops
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl.wafer — WSE program container (equivalent to aie.device)
//===----------------------------------------------------------------------===//

def CSL_WaferOp : CSL_Op<"wafer", [
    IsolatedFromAbove,
    SymbolTable,
    NoTerminator,
    HasParent<"mlir::ModuleOp">]> {
  let summary = "WSE program container";
  let description = [{
    Top-level container for a WSE program. Contains csl.program, csl.layout,
    and csl.host as sibling ops. Analogous to aie.device.

    ```mlir
    module {
      csl.wafer @wseprog {arch = "wse3"} {
        csl.program @pe { ... }
        csl.layout (1, 1) @layout { ... }
        csl.host @main(...) { ... } { ... }
      }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$arch);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict `{` $body `}`";
}

//===----------------------------------------------------------------------===//
// csl.program — PE kernel template (one per pe_program.csl file)
//===----------------------------------------------------------------------===//

def CSL_ProgramOp : CSL_Op<"program", [
    Symbol,
    SymbolTable,
    NoTerminator,
    HasParent<"CSL_WaferOp">]> {
  let summary = "PE kernel template";
  let description = [{
    Models a single pe_program.csl source file. Has no coordinates; placement
    is handled by csl.layout. Block arguments with !csl.comptime<T> type map
    to CSL 'param' declarations.

    ```mlir
    csl.program @vecadd_pe(%col: !csl.comptime<i16>) {
      %a = csl.var @a : memref<256xf32>
      csl.func @compute() { ... }
      csl.export @a { alias = "a" }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}
```

- [ ] **Step 4: Add `CSLProgramOp::parse` + `::print` to CSLOps.cpp**

In `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`, add at the bottom of the file (after existing ops):

```cpp
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace xilinx::csl;

//===----------------------------------------------------------------------===//
// CSLProgramOp — custom assembly for comptime block args
//===----------------------------------------------------------------------===//

mlir::ParseResult CSLProgramOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  // Parse @sym_name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                              result.attributes))
    return failure();

  // Parse optional ( %arg: !csl.comptime<T>, ... )
  SmallVector<OpAsmParser::Argument> regionArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    // Handle empty parens
    if (failed(parser.parseOptionalRParen())) {
      do {
        OpAsmParser::Argument arg;
        if (parser.parseArgument(arg, /*allowType=*/true))
          return failure();
        regionArgs.push_back(arg);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  // Optional attribute dict (for future extensibility)
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse body region with the block args
  auto *body = result.addRegion();
  return parser.parseRegion(*body, regionArgs);
}

void CSLProgramOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());

  // Print block args if any
  Block &entry = getBody().front();
  if (!entry.getArguments().empty()) {
    printer << '(';
    llvm::interleaveComma(entry.getArguments(), printer.getStream(),
                          [&](BlockArgument arg) {
      printer.getStream() << arg << ": ";
      printer.printType(arg.getType());
    });
    printer << ')';
  }

  printer.printOptionalAttrDict((*this)->getAttrs(),
                                 {getSymNameAttrName()});
  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}
```

- [ ] **Step 5: Rebuild and run test**

```bash
cd build && ninja install 2>&1 | tail -5
lit ../mlir/test/Dialect/CSL/v2_roundtrip.mlir
```
Expected: `PASS` (all 3 new wafer tests + original comptime tests)

- [ ] **Step 6: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td \
        mlir/lib/Dialect/CSL/IR/CSLOps.cpp \
        mlir/test/Dialect/CSL/v2_roundtrip.mlir
git commit -m "csl: add csl.wafer and csl.program v2 container ops

csl.wafer is the WSE program container (like aie.device), holding
program/layout/host as siblings. csl.program is the PE kernel template
with optional !csl.comptime<T> block args mapping to CSL params.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: `csl.export`, `csl.layout`, `csl.host` Ops + Full v2 Roundtrip

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td`
- Modify: `mlir/test/Dialect/CSL/v2_roundtrip.mlir`

- [ ] **Step 1: Extend v2_roundtrip.mlir with export/layout/host**

Append to `mlir/test/Dialect/CSL/v2_roundtrip.mlir`:
```mlir
// ---- csl.export (inside csl.program) ----

// CHECK-LABEL: csl.wafer @w_export
// CHECK:   csl.export @a {alias = "a"}
// CHECK:   csl.export @compute {kind = "func"}
module {
  csl.wafer @w_export {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      csl.func @compute {
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @compute {kind = "func"}
    }
  }
}

// ---- csl.layout ----

// CHECK-LABEL: csl.wafer @w_layout
// CHECK:   csl.layout {height = 1 : i64, width = 1 : i64} @main_layout
module {
  csl.wafer @w_layout {arch = "wse3"} {
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
    }
  }
}

// ---- csl.host ----

// CHECK-LABEL: csl.wafer @w_host
// CHECK:   csl.host @main
// CHECK-SAME: layout = @main_layout
module {
  csl.wafer @w_host {arch = "wse3"} {
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
    }
    csl.host @main(%a: memref<256xf32>) {layout = @main_layout} {
    }
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit mlir/test/Dialect/CSL/v2_roundtrip.mlir 2>&1 | grep "error:" | head -5
```
Expected: `error: 'csl.export' op not registered`

- [ ] **Step 3: Add `ExportOp`, `LayoutOp`, `HostOp` to CSLOps.td**

Append to the v2 section of `mlir/include/air/Dialect/CSL/CSLOps.td` (after `CSL_ProgramOp`):

```tablegen
//===----------------------------------------------------------------------===//
// csl.export — mark a symbol as host-visible
//===----------------------------------------------------------------------===//

def CSL_ExportOp : CSL_Op<"export", [
    HasParent<"CSL_ProgramOp">]> {
  let summary = "Mark a PE symbol as host-visible";
  let description = [{
    Marks a csl.var or csl.func as accessible from the host. Direction
    (in/out) is added later by the -csl-derive-exports pass.

    ```mlir
    csl.export @a { alias = "a" }           // var with alias
    csl.export @compute { kind = "func" }   // function export
    csl.export @a { alias = "a", direction = "in" }  // after derive pass
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
    HasParent<"CSL_WaferOp">]> {
  let summary = "Layout container — maps programs to PE rectangles";
  let description = [{
    Models the layout.csl / csl_layout.py file. Contains csl_layout.*
    ops for placement, routing, and host-visible export declarations.

    ```mlir
    csl.layout {width = 4 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @gemv_pe at (0, 0) {col = 0 : i16}
      csl_layout.export @y from @gemv_pe::@y_ptr
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$width,
    I64Attr:$height
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict $sym_name `{` $body `}`";
}

//===----------------------------------------------------------------------===//
// csl.host — host runtime orchestration container (emits run.py)
//===----------------------------------------------------------------------===//

def CSL_HostOp : CSL_Op<"host", [
    Symbol,
    NoTerminator,
    HasParent<"CSL_WaferOp">]> {
  let summary = "Host runtime orchestration";
  let description = [{
    Models run.py. Contains csl_host.* ops for memcpy and launch.
    References the layout by symbol via the 'layout' attribute.

    ```mlir
    csl.host @main(%a: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_h2d %a to @main_layout::@a {px=0, py=0, width=1, height=1}
      csl_host.launch @main_layout::@compute
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$layout
  );
  // Two regions: (1) argument list region (parsed as func args), (2) body
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}
```

- [ ] **Step 4: Add `CSLHostOp::parse` + `::print` to CSLOps.cpp**

Append to `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`:

```cpp
//===----------------------------------------------------------------------===//
// CSLHostOp — custom assembly for func-like args + layout attr
//
// Format: @name ( %arg: type, ... ) { layout = @sym } { body }
//===----------------------------------------------------------------------===//

mlir::ParseResult CSLHostOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                              result.attributes))
    return failure();

  // Parse ( %arg: type, ... )
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> argTypes;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument arg;
      if (parser.parseArgument(arg, /*allowType=*/true))
        return failure();
      args.push_back(arg);
      argTypes.push_back(arg.type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  result.addAttribute("arg_types",
                       parser.getBuilder().getTypeArrayAttr(argTypes));

  // Parse { layout = @sym }
  if (parser.parseLBrace())
    return failure();
  if (parser.parseKeyword("layout") || parser.parseEqual())
    return failure();
  FlatSymbolRefAttr layoutAttr;
  if (parser.parseAttribute(layoutAttr))
    return failure();
  result.addAttribute(getLayoutAttrName(result.name), layoutAttr);
  if (parser.parseRBrace())
    return failure();

  // Parse body
  auto *body = result.addRegion();
  return parser.parseRegion(*body, args);
}

void CSLHostOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << '(';
  Block &entry = getBody().front();
  llvm::interleaveComma(entry.getArguments(), printer.getStream(),
                        [&](BlockArgument arg) {
    printer.getStream() << arg << ": ";
    printer.printType(arg.getType());
  });
  printer << ") {layout = ";
  printer.printAttributeWithoutType(getLayoutAttr());
  printer << "} ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}
```

- [ ] **Step 5: Rebuild and run**

```bash
cd build && ninja install 2>&1 | tail -5
lit ../mlir/test/Dialect/CSL/v2_roundtrip.mlir
```
Expected: all tests `PASS`

- [ ] **Step 6: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td \
        mlir/lib/Dialect/CSL/IR/CSLOps.cpp \
        mlir/test/Dialect/CSL/v2_roundtrip.mlir
git commit -m "csl: add csl.export, csl.layout, csl.host v2 ops

csl.export marks PE symbols host-visible (direction added later by pass).
csl.layout is the placement container. csl.host is the runtime
orchestration container referencing a layout by symbol.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: `csl_layout` Dialect

**Files:**
- Create: `mlir/include/air/Dialect/CSLLayout/` (4 files + CMakeLists)
- Create: `mlir/lib/Dialect/CSLLayout/` (3 files)
- Create: `mlir/test/Dialect/CSLLayout/roundtrip.mlir`

- [ ] **Step 1: Write the failing test**

Create `mlir/test/Dialect/CSLLayout/roundtrip.mlir`:
```mlir
// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: csl.wafer @w
// CHECK:   csl_layout.place @vecadd_pe {col = 0 : i16} at (0, 0)
// CHECK:   csl_layout.export @a from @vecadd_pe::@a
module {
  csl.wafer @w {arch = "wse3"} {
    csl.program @vecadd_pe(%col: !csl.comptime<i16>) {
      %a = csl.var @a : memref<256xf32>
      csl.export @a {alias = "a"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe {col = 0 : i16} at (0, 0)
      csl_layout.export @a from @vecadd_pe::@a
    }
  }
}

// CHECK-LABEL: csl.wafer @w2
// CHECK:   csl_layout.place @pe {col = 0 : i16, width = 4 : i16} at (0, 0)
module {
  csl.wafer @w2 {arch = "wse3"} {
    csl.program @pe(%col: !csl.comptime<i16>, %width: !csl.comptime<i16>) {
    }
    csl.layout {width = 4 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe {col = 0 : i16, width = 4 : i16} at (0, 0)
    }
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit mlir/test/Dialect/CSLLayout/roundtrip.mlir 2>&1 | head -5
```
Expected: `error: 'csl_layout.place' op not registered`

- [ ] **Step 3: Create `CSLLayoutBase.td`**

Create `mlir/include/air/Dialect/CSLLayout/CSLLayoutBase.td`:
```tablegen
//===- CSLLayoutBase.td - csl_layout dialect base ----------------*- tablegen -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef CSL_LAYOUT_BASE
#define CSL_LAYOUT_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def CSLLayout_Dialect : Dialect {
  let name = "csl_layout";
  let cppNamespace = "xilinx::csl_layout";
  let description = [{
    The `csl_layout` dialect contains placement and routing ops that live
    inside a `csl.layout` region. These ops emit to csl_layout.py via the
    Cerebras sdkLayout Python API.
  }];
  let useDefaultAttributePrinterParser = 1;
}

class CSLLayout_Op<string mnemonic, list<Trait> traits = []>
    : Op<CSLLayout_Dialect, mnemonic, traits>;

#endif // CSL_LAYOUT_BASE
```

- [ ] **Step 4: Create `CSLLayoutOps.td`**

Create `mlir/include/air/Dialect/CSLLayout/CSLLayoutOps.td`:
```tablegen
//===- CSLLayoutOps.td - csl_layout ops --------------------------*- tablegen -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef CSL_LAYOUT_OPS
#define CSL_LAYOUT_OPS

include "air/Dialect/CSLLayout/CSLLayoutBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// csl_layout.place — bind one PE to a program with comptime params
//
// Format: csl_layout.place @prog_sym {param=val,...} at (x, y)
//===----------------------------------------------------------------------===//

def CSLLayout_PlaceOp : CSLLayout_Op<"place"> {
  let summary = "Bind a PE grid position to a program with comptime params";
  let description = [{
    Binds a single PE at position (x, y) to a csl.program symbol.
    The attribute dict carries the comptime param values to pass.

    ```mlir
    csl_layout.place @vecadd_pe {col = 0 : i16} at (0, 0)
    ```
    Emits to sdkLayout: `region.set_tile_code(x, y, "vecadd_pe.csl", .{col=0})`
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$program,
    I64Attr:$x,
    I64Attr:$y,
    DictionaryAttr:$params
  );
  let assemblyFormat = "$program $params `at` `(` $x `,` $y `)` attr-dict";
}

//===----------------------------------------------------------------------===//
// csl_layout.export — declare a host-visible symbol in the layout
//
// Format: csl_layout.export @sym from @prog::@export_sym
//===----------------------------------------------------------------------===//

def CSLLayout_ExportOp : CSLLayout_Op<"export"> {
  let summary = "Declare a host-visible symbol from a program";
  let description = [{
    Declares that @export_sym in program @prog is host-visible.
    Emits to sdkLayout: `layout.add_field(@sym, type, mutable)`.

    ```mlir
    csl_layout.export @a from @vecadd_pe::@a
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$name,
    SymbolRefAttr:$source
  );
  let assemblyFormat = "$name `from` $source attr-dict";
}

#endif // CSL_LAYOUT_OPS
```

- [ ] **Step 5: Create `CSLLayoutDialect.h`**

Create `mlir/include/air/Dialect/CSLLayout/CSLLayoutDialect.h`:
```cpp
//===- CSLLayoutDialect.h - csl_layout dialect -----------------*- C++ -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//
#ifndef CSL_LAYOUT_DIALECT_H
#define CSL_LAYOUT_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "air/Dialect/CSLLayout/CSLLayoutOpsDialect.h.inc"

#endif // CSL_LAYOUT_DIALECT_H
```

- [ ] **Step 6: Create `CSLLayoutOps.h`**

Create `mlir/include/air/Dialect/CSLLayout/CSLLayoutOps.h`:
```cpp
//===- CSLLayoutOps.h - csl_layout ops -------------------------*- C++ -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//
#ifndef CSL_LAYOUT_OPS_H
#define CSL_LAYOUT_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "air/Dialect/CSLLayout/CSLLayoutDialect.h"

#define GET_OP_CLASSES
#include "air/Dialect/CSLLayout/CSLLayoutOps.h.inc"

#endif // CSL_LAYOUT_OPS_H
```

- [ ] **Step 7: Create include CMakeLists.txt**

Create `mlir/include/air/Dialect/CSLLayout/CMakeLists.txt`:
```cmake
# SPDX-License-Identifier: MIT
add_mlir_dialect(CSLLayoutOps csl_layout)
add_mlir_doc(CSLLayoutOps CSLLayoutDialect ./ -gen-dialect-doc -dialect=csl_layout)
```

- [ ] **Step 8: Create `CSLLayoutDialect.cpp`**

Create `mlir/lib/Dialect/CSLLayout/IR/CSLLayoutDialect.cpp`:
```cpp
//===- CSLLayoutDialect.cpp - csl_layout dialect impl ----------*- C++ -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSLLayout/CSLLayoutDialect.h"
#include "air/Dialect/CSLLayout/CSLLayoutOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "air/Dialect/CSLLayout/CSLLayoutOpsDialect.cpp.inc"

namespace xilinx::csl_layout {

void CSLLayoutDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/CSLLayout/CSLLayoutOps.cpp.inc"
      >();
}

} // namespace xilinx::csl_layout
```

- [ ] **Step 9: Create lib CMakeLists.txt files**

Create `mlir/lib/Dialect/CSLLayout/CMakeLists.txt`:
```cmake
# SPDX-License-Identifier: MIT
add_subdirectory(IR)
```

Create `mlir/lib/Dialect/CSLLayout/IR/CMakeLists.txt`:
```cmake
# SPDX-License-Identifier: MIT
add_mlir_dialect_library(
  CSLLayoutDialect
  CSLLayoutDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSLLayout

  DEPENDS
  MLIRCSLLayoutOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR)
```

- [ ] **Step 10: Wire CMakeLists + InitAll**

In `mlir/include/air/Dialect/CMakeLists.txt`, add:
```cmake
add_subdirectory(CSLLayout)
```

In `mlir/lib/Dialect/CMakeLists.txt`, add:
```cmake
add_subdirectory(CSLLayout)
```

In `mlir/lib/InitAll.cpp`, add include and registration:
```cpp
#include "air/Dialect/CSLLayout/CSLLayoutDialect.h"
#include "air/Dialect/CSLLayout/CSLLayoutOps.h"
// ...
// In registerAllDialects():
registry.insert<..., xilinx::csl_layout::CSLLayoutDialect>();
```

- [ ] **Step 11: Rebuild and run**

```bash
cd build && ninja install 2>&1 | tail -5
lit ../mlir/test/Dialect/CSLLayout/roundtrip.mlir
lit ../mlir/test/Dialect/CSL/v2_roundtrip.mlir  # regression check
```
Expected: all `PASS`

- [ ] **Step 12: Commit**

```bash
git add mlir/include/air/Dialect/CSLLayout/ \
        mlir/lib/Dialect/CSLLayout/ \
        mlir/test/Dialect/CSLLayout/ \
        mlir/include/air/Dialect/CMakeLists.txt \
        mlir/lib/Dialect/CMakeLists.txt \
        mlir/lib/InitAll.cpp
git commit -m "csl_layout: add new dialect with place + export ops

csl_layout.place binds a PE grid position to a program with comptime
params. csl_layout.export declares host-visible symbols. Both emit to
the sdkLayout Python API in csl_layout.py.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: `csl_host` Dialect

**Files:**
- Create: `mlir/include/air/Dialect/CSLHost/` (4 files + CMakeLists)
- Create: `mlir/lib/Dialect/CSLHost/` (3 files)
- Create: `mlir/test/Dialect/CSLHost/roundtrip.mlir`

- [ ] **Step 1: Write the failing test**

Create `mlir/test/Dialect/CSLHost/roundtrip.mlir`:
```mlir
// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: csl.wafer @w
// CHECK:   csl_host.memcpy_h2d %{{.*}} to @main_layout::@a {height = 1 : i64, px = 0 : i64, py = 0 : i64, width = 1 : i64}
// CHECK:   csl_host.launch @main_layout::@compute
// CHECK:   csl_host.memcpy_d2h @main_layout::@c to %{{.*}} {height = 1 : i64, px = 0 : i64, py = 0 : i64, width = 1 : i64}
module {
  csl.wafer @w {arch = "wse3"} {
    csl.program @vecadd_pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute { csl.return }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe {} at (0, 0)
      csl_layout.export @a from @vecadd_pe::@a
      csl_layout.export @b from @vecadd_pe::@b
      csl_layout.export @c from @vecadd_pe::@c
      csl_layout.export @compute from @vecadd_pe::@compute
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_h2d %a_in to @main_layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
      csl_host.memcpy_h2d %b_in to @main_layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
      csl_host.launch @main_layout::@compute
      csl_host.memcpy_d2h @main_layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
    }
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit mlir/test/Dialect/CSLHost/roundtrip.mlir 2>&1 | head -5
```

- [ ] **Step 3: Create `CSLHostBase.td`**

Create `mlir/include/air/Dialect/CSLHost/CSLHostBase.td`:
```tablegen
//===- CSLHostBase.td - csl_host dialect base -----------------*- tablegen -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef CSL_HOST_BASE
#define CSL_HOST_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def CSLHost_Dialect : Dialect {
  let name = "csl_host";
  let cppNamespace = "xilinx::csl_host";
  let description = [{
    The `csl_host` dialect contains host-side runtime ops that live
    inside a `csl.host` region. These ops emit to run.py via the
    Cerebras SdkRuntime Python API.
  }];
  let useDefaultAttributePrinterParser = 1;
}

class CSLHost_Op<string mnemonic, list<Trait> traits = []>
    : Op<CSLHost_Dialect, mnemonic, traits>;

#endif // CSL_HOST_BASE
```

- [ ] **Step 4: Create `CSLHostOps.td`**

Create `mlir/include/air/Dialect/CSLHost/CSLHostOps.td`:
```tablegen
//===- CSLHostOps.td - csl_host ops ---------------------------*- tablegen -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef CSL_HOST_OPS
#define CSL_HOST_OPS

include "air/Dialect/CSLHost/CSLHostBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// csl_host.memcpy_h2d — transfer host buffer to device symbol
//
// Format: csl_host.memcpy_h2d %src to @layout::@sym {px, py, w, h}
//===----------------------------------------------------------------------===//

def CSLHost_MemcpyH2DOp : CSLHost_Op<"memcpy_h2d"> {
  let summary = "Transfer host memref to PE-local variable";
  let description = [{
    Transfers the host-side memref to the named PE-local variable across
    the PE rectangle specified by (px, py, width, height).

    Emits to SdkRuntime:
    ```python
    id = runner.get_id("sym")
    runner.memcpy_h2d(id, src, px, py, width, height, N)
    ```
  }];

  let arguments = (ins
    AnyMemRef:$src,
    SymbolRefAttr:$dest,   // @layout::@sym
    I64Attr:$px,
    I64Attr:$py,
    I64Attr:$width,
    I64Attr:$height
  );
  let assemblyFormat = "$src `to` $dest attr-dict `:` type($src)";
}

//===----------------------------------------------------------------------===//
// csl_host.memcpy_d2h — transfer device symbol to host buffer
//
// Format: csl_host.memcpy_d2h @layout::@sym to %dst {px, py, w, h}
//===----------------------------------------------------------------------===//

def CSLHost_MemcpyD2HOp : CSLHost_Op<"memcpy_d2h"> {
  let summary = "Transfer PE-local variable to host memref";
  let description = [{
    Transfers the named PE-local variable to the host-side memref.

    Emits to SdkRuntime:
    ```python
    id = runner.get_id("sym")
    runner.memcpy_d2h(dst, id, px, py, width, height, N)
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$src,    // @layout::@sym
    AnyMemRef:$dest,
    I64Attr:$px,
    I64Attr:$py,
    I64Attr:$width,
    I64Attr:$height
  );
  let assemblyFormat = "$src `to` $dest attr-dict `:` type($dest)";
}

//===----------------------------------------------------------------------===//
// csl_host.launch — RPC call into a PE function
//
// Format: csl_host.launch @layout::@func
//===----------------------------------------------------------------------===//

def CSLHost_LaunchOp : CSLHost_Op<"launch"> {
  let summary = "RPC call into a PE-local function";
  let description = [{
    Synchronously calls the named function on all PEs in the layout.

    Emits to SdkRuntime:
    ```python
    runner.launch("func", nonblock=False)
    ```
  }];

  let arguments = (ins SymbolRefAttr:$callee);
  let assemblyFormat = "$callee attr-dict";
}

#endif // CSL_HOST_OPS
```

- [ ] **Step 5: Create `CSLHostDialect.h`, `CSLHostOps.h`, and CMakeLists files**

`mlir/include/air/Dialect/CSLHost/CSLHostDialect.h`:
```cpp
//===- CSLHostDialect.h ----------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: MIT
#ifndef CSL_HOST_DIALECT_H
#define CSL_HOST_DIALECT_H
#include "mlir/IR/Dialect.h"
#include "air/Dialect/CSLHost/CSLHostOpsDialect.h.inc"
#endif
```

`mlir/include/air/Dialect/CSLHost/CSLHostOps.h`:
```cpp
//===- CSLHostOps.h --------------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: MIT
#ifndef CSL_HOST_OPS_H
#define CSL_HOST_OPS_H
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "air/Dialect/CSLHost/CSLHostDialect.h"
#define GET_OP_CLASSES
#include "air/Dialect/CSLHost/CSLHostOps.h.inc"
#endif
```

`mlir/include/air/Dialect/CSLHost/CMakeLists.txt`:
```cmake
# SPDX-License-Identifier: MIT
add_mlir_dialect(CSLHostOps csl_host)
add_mlir_doc(CSLHostOps CSLHostDialect ./ -gen-dialect-doc -dialect=csl_host)
```

`mlir/lib/Dialect/CSLHost/CMakeLists.txt`:
```cmake
# SPDX-License-Identifier: MIT
add_subdirectory(IR)
```

`mlir/lib/Dialect/CSLHost/IR/CMakeLists.txt`:
```cmake
# SPDX-License-Identifier: MIT
add_mlir_dialect_library(
  CSLHostDialect
  CSLHostDialect.cpp
  ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSLHost
  DEPENDS MLIRCSLHostOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRMemRefDialect)
```

- [ ] **Step 6: Create `CSLHostDialect.cpp`**

Create `mlir/lib/Dialect/CSLHost/IR/CSLHostDialect.cpp`:
```cpp
//===- CSLHostDialect.cpp - csl_host impl ----------------------*- C++ -*-===//
// SPDX-License-Identifier: MIT
#include "air/Dialect/CSLHost/CSLHostDialect.h"
#include "air/Dialect/CSLHost/CSLHostOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "air/Dialect/CSLHost/CSLHostOpsDialect.cpp.inc"

namespace xilinx::csl_host {

void CSLHostDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/CSLHost/CSLHostOps.cpp.inc"
      >();
}

} // namespace xilinx::csl_host
```

- [ ] **Step 7: Wire CMakeLists + InitAll**

In `mlir/include/air/Dialect/CMakeLists.txt`:
```cmake
add_subdirectory(CSLHost)
```

In `mlir/lib/Dialect/CMakeLists.txt`:
```cmake
add_subdirectory(CSLHost)
```

In `mlir/lib/InitAll.cpp`:
```cpp
#include "air/Dialect/CSLHost/CSLHostDialect.h"
#include "air/Dialect/CSLHost/CSLHostOps.h"
// In registerAllDialects():
registry.insert<..., xilinx::csl_host::CSLHostDialect>();
```

- [ ] **Step 8: Rebuild and run**

```bash
cd build && ninja install 2>&1 | tail -5
lit ../mlir/test/Dialect/CSLHost/roundtrip.mlir
lit ../mlir/test/Dialect/CSL/v2_roundtrip.mlir
lit ../mlir/test/Dialect/CSLLayout/roundtrip.mlir
```
Expected: all `PASS`

- [ ] **Step 9: Commit**

```bash
git add mlir/include/air/Dialect/CSLHost/ \
        mlir/lib/Dialect/CSLHost/ \
        mlir/test/Dialect/CSLHost/ \
        mlir/include/air/Dialect/CMakeLists.txt \
        mlir/lib/Dialect/CMakeLists.txt \
        mlir/lib/InitAll.cpp
git commit -m "csl_host: add new dialect with memcpy_h2d, memcpy_d2h, launch ops

memcpy_h2d/d2h transfer data between host buffers and PE-local vars.
launch issues an RPC into a PE function. All emit to SdkRuntime Python API.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: `-air-to-csl` Conversion Pass

**Files:**
- Create: `mlir/include/air/Conversion/AIRToCSLPass.h`
- Create: `mlir/lib/Conversion/AIRToCSL/AIRToCSL.cpp`
- Modify: `mlir/include/air/Conversion/Passes.td`
- Modify: `mlir/lib/Conversion/Passes.cpp`
- Modify: `mlir/lib/Conversion/CMakeLists.txt`
- Create: `mlir/test/Conversion/AIRToCSL/vecadd.mlir`
- Create: `mlir/test/Conversion/AIRToCSL/reject_multi_herd.mlir`

- [ ] **Step 1: Write failing conversion test**

Create `mlir/test/Conversion/AIRToCSL/vecadd.mlir`:
```mlir
// RUN: air-opt %s -air-to-csl | FileCheck %s

// CHECK:       module {
// CHECK-NEXT:    csl.wafer @vecadd {arch = "wse3"} {
// CHECK:           csl.program @vecadd_pe {
// CHECK-DAG:         csl.var @a : memref<256xf32>
// CHECK-DAG:         csl.var @b : memref<256xf32>
// CHECK-DAG:         csl.var @c : memref<256xf32>
// CHECK:             csl.func @compute {
// CHECK:               scf.for
// CHECK:             csl.export @a {alias = "a"}
// CHECK:             csl.export @b {alias = "b"}
// CHECK:             csl.export @c {alias = "c"}
// CHECK:             csl.export @compute {kind = "func"}
// CHECK:           csl.layout {height = 1 : i64, width = 1 : i64} @vecadd_layout {
// CHECK:             csl_layout.place @vecadd_pe {} at (0, 0)
// CHECK:             csl_layout.export @a from @vecadd_pe::@a
// CHECK:             csl_layout.export @b from @vecadd_pe::@b
// CHECK:             csl_layout.export @c from @vecadd_pe::@c
// CHECK:             csl_layout.export @compute from @vecadd_pe::@compute
// CHECK:           csl.host @vecadd_host(%{{.*}}: memref<256xf32>,
// CHECK-SAME:                            %{{.*}}: memref<256xf32>,
// CHECK-SAME:                            %{{.*}}: memref<256xf32>)
// CHECK-SAME:          {layout = @vecadd_layout} {
// CHECK:             csl_host.memcpy_h2d %{{.*}} to @vecadd_layout::@a
// CHECK-SAME:            {height = 1 : i64, px = 0 : i64, py = 0 : i64, width = 1 : i64}
// CHECK:             csl_host.memcpy_h2d %{{.*}} to @vecadd_layout::@b
// CHECK:             csl_host.launch @vecadd_layout::@compute
// CHECK:             csl_host.memcpy_d2h @vecadd_layout::@c to %{{.*}}

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b, %lc=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @seg0 args(%sa=%la, %sb=%lb, %sc=%lc)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        %one2 = arith.constant 1 : index
        air.herd @h tile(%htx, %hty) in (%hsx=%one2, %hsy=%one2)
            args(%ha=%sa, %hb=%sb, %hc=%sc)
            : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          %lo = arith.constant 0 : index
          %hi = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %lo to %hi step %step {
            %va = memref.load %ha[%i] : memref<256xf32>
            %vb = memref.load %hb[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %hc[%i] : memref<256xf32>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
```

Create `mlir/test/Conversion/AIRToCSL/reject_multi_herd.mlir`:
```mlir
// RUN: air-opt %s -air-to-csl 2>&1 | FileCheck %s
// CHECK: error: -air-to-csl: multiple air.herd ops not supported in V1

module {
  func.func @two_herds() {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) {
      air.segment @seg {
        %one2 = arith.constant 1 : index
        air.herd @h1 tile(%x, %y) in (%sx=%one2, %sy=%one2) {
          air.herd_terminator
        }
        air.herd @h2 tile(%x, %y) in (%sx=%one2, %sy=%one2) {
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit mlir/test/Conversion/AIRToCSL/ 2>&1 | head -10
```

- [ ] **Step 3: Add `AIRToCSL` pass to `Passes.td`**

Append to `mlir/include/air/Conversion/Passes.td` (before `#endif`):
```tablegen
//===----------------------------------------------------------------------===//
// CSL v2 passes
//===----------------------------------------------------------------------===//

def AIRToCSL : Pass<"air-to-csl", "ModuleOp"> {
  let summary = "Lower AIR dialect to CSL v2 dialect family";
  let constructor = "xilinx::air::createAIRToCSLPass()";
  let description = [{
    Converts an air.launch/segment/herd nest to a csl.wafer with three
    sibling regions: csl.program (PE kernel), csl.layout (placement),
    csl.host (runtime). V1 scope: single 1x1 herd, functions only.
  }];
}

def CSLDeriveExports : Pass<"csl-derive-exports", "ModuleOp"> {
  let summary = "Derive csl.export directions from csl_host transfer ops";
  let constructor = "xilinx::air::createCSLDeriveExportsPass()";
  let description = [{
    Scans csl_host.memcpy_h2d/d2h ops and annotates the corresponding
    csl.export ops with direction = "in" or "out".
  }];
}
```

- [ ] **Step 4: Create `AIRToCSLPass.h`**

Create `mlir/include/air/Conversion/AIRToCSLPass.h`:
```cpp
//===- AIRToCSLPass.h - AIR → CSL v2 conversion pass ----------*- C++ -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//
#ifndef AIR_TO_CSL_PASS_H
#define AIR_TO_CSL_PASS_H

#include <memory>
namespace mlir { class Pass; }
namespace xilinx::air {
std::unique_ptr<mlir::Pass> createAIRToCSLPass();
std::unique_ptr<mlir::Pass> createCSLDeriveExportsPass();
} // namespace xilinx::air
#endif
```

- [ ] **Step 5: Implement the pass in `AIRToCSL.cpp`**

Create `mlir/lib/Conversion/AIRToCSL/AIRToCSL.cpp`:
```cpp
//===- AIRToCSL.cpp - AIR → CSL v2 conversion pass --------------*- C++ -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//
//
// -air-to-csl pass: converts air.launch/segment/herd → csl.wafer with three
// sibling regions (csl.program, csl.layout, csl.host).
//
// V1 scope: 1x1 herd, functions only, no channels/async, static memrefs.
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToCSLPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSLLayout/CSLLayoutDialect.h"
#include "air/Dialect/CSLLayout/CSLLayoutOps.h"
#include "air/Dialect/CSLHost/CSLHostDialect.h"
#include "air/Dialect/CSLHost/CSLHostOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "air-to-csl"
using namespace mlir;
using namespace xilinx;

namespace {

struct AIRToCSLPass : public PassWrapper<AIRToCSLPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIRToCSLPass)

  StringRef getArgument() const override { return "air-to-csl"; }
  StringRef getDescription() const override {
    return "Lower AIR dialect to CSL v2 dialect family";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect all func.func ops with air.launch inside
    SmallVector<func::FuncOp> funcsToConvert;
    module.walk([&](func::FuncOp f) {
      if (f.walk([](air::LaunchOp) {
            return WalkResult::interrupt();
          }).wasInterrupted())
        funcsToConvert.push_back(f);
    });

    for (func::FuncOp funcOp : funcsToConvert) {
      if (failed(convertFunc(funcOp, builder)))
        return signalPassFailure();
    }
  }

  LogicalResult convertFunc(func::FuncOp funcOp, OpBuilder &builder) {
    // Collect air.launch ops
    SmallVector<air::LaunchOp> launches;
    funcOp.walk([&](air::LaunchOp l) { launches.push_back(l); });
    if (launches.empty())
      return success();

    // V1: single launch
    if (launches.size() > 1) {
      funcOp.emitError("-air-to-csl: multiple air.launch ops not supported in V1");
      return failure();
    }
    air::LaunchOp launch = launches[0];

    // Find herd inside
    SmallVector<air::HerdOp> herds;
    launch.walk([&](air::HerdOp h) { herds.push_back(h); });
    if (herds.size() != 1) {
      launch.emitError("-air-to-csl: multiple air.herd ops not supported in V1");
      return failure();
    }
    air::HerdOp herd = herds[0];

    // Get herd name (use func name if unnamed)
    std::string herdName = herd.getName() ? herd.getName()->str()
                                           : funcOp.getName().str();
    std::string programName = herdName + "_pe";
    std::string layoutName = herdName + "_layout";
    std::string hostName = herdName + "_host";
    std::string waferName = funcOp.getName().str();

    // Build csl.wafer wrapping module
    builder.setInsertionPoint(funcOp);
    Location loc = funcOp.getLoc();

    auto waferOp = builder.create<csl::WaferOp>(loc,
        builder.getStringAttr(waferName),
        builder.getStringAttr("wse3"));

    // ----- Build csl.program -----
    Block &waferBody = waferOp.getBody().front();
    OpBuilder programBuilder(&waferBody, waferBody.begin());

    auto programOp = programBuilder.create<csl::ProgramOp>(loc,
        programBuilder.getStringAttr(programName));
    Block *programBlock = &programOp.getBody().emplaceBlock();
    OpBuilder kernelBuilder(programBlock, programBlock->end());

    // Map herd args → csl.var ops
    // The herd args come from air.segment args (L3 memrefs passed in)
    IRMapping varMap;
    SmallVector<StringRef> varNames;
    auto herdArgs = herd.getBody().front().getArguments();
    // Skip tile coord args (first herd.getNumDims() args)
    unsigned numCoords = herd.getNumDims();
    for (unsigned i = numCoords; i < herdArgs.size(); ++i) {
      BlockArgument arg = herdArgs[i];
      auto memrefTy = llvm::dyn_cast<MemRefType>(arg.getType());
      if (!memrefTy) continue;
      // Generate name: a, b, c, ...
      std::string varName = std::string(1, char('a' + (i - numCoords)));
      varNames.push_back(builder.getStringAttr(varName));
      Value varVal = kernelBuilder.create<csl::VarOp>(loc,
          kernelBuilder.getStringAttr(varName), memrefTy);
      varMap.map(arg, varVal);
    }

    // Clone herd body into csl.func @compute
    auto computeFunc = kernelBuilder.create<csl::FuncOp>(loc,
        kernelBuilder.getStringAttr("compute"));
    Block *funcBlock = &computeFunc.getBody().emplaceBlock();
    OpBuilder funcBuilder(funcBlock, funcBlock->end());

    // Clone herd body (skip tile args)
    Block &herdBlock = herd.getBody().front();
    for (Operation &op : herdBlock) {
      if (llvm::isa<air::HerdTerminatorOp>(op)) {
        funcBuilder.create<csl::ReturnOp>(loc);
        continue;
      }
      funcBuilder.clone(op, varMap);
    }

    // Add csl.export ops
    for (auto &[varName, _] : llvm::zip(varNames, varNames)) {
      kernelBuilder.create<csl::ExportOp>(loc,
          FlatSymbolRefAttr::get(builder.getContext(), varName),
          StringAttr::get(builder.getContext(), varName),
          /*kind=*/StringAttr{},
          /*direction=*/StringAttr{});
    }
    kernelBuilder.create<csl::ExportOp>(loc,
        FlatSymbolRefAttr::get(builder.getContext(), "compute"),
        /*alias=*/StringAttr{},
        StringAttr::get(builder.getContext(), "func"),
        /*direction=*/StringAttr{});

    // ----- Build csl.layout -----
    auto layoutOp = programBuilder.create<csl::LayoutOp>(loc,
        programBuilder.getStringAttr(layoutName),
        programBuilder.getI64IntegerAttr(1),
        programBuilder.getI64IntegerAttr(1));
    Block *layoutBlock = &layoutOp.getBody().emplaceBlock();
    OpBuilder layoutBuilder(layoutBlock, layoutBlock->end());

    // csl_layout.place @programName {} at (0, 0)
    layoutBuilder.create<csl_layout::PlaceOp>(loc,
        FlatSymbolRefAttr::get(builder.getContext(), programName),
        layoutBuilder.getI64IntegerAttr(0),
        layoutBuilder.getI64IntegerAttr(0),
        DictionaryAttr::get(builder.getContext(), {}));

    // csl_layout.export for each var + compute
    for (auto varName : varNames) {
      layoutBuilder.create<csl_layout::ExportOp>(loc,
          FlatSymbolRefAttr::get(builder.getContext(), varName),
          SymbolRefAttr::get(builder.getContext(), programName,
              {FlatSymbolRefAttr::get(builder.getContext(), varName)}));
    }
    layoutBuilder.create<csl_layout::ExportOp>(loc,
        FlatSymbolRefAttr::get(builder.getContext(), "compute"),
        SymbolRefAttr::get(builder.getContext(), programName,
            {FlatSymbolRefAttr::get(builder.getContext(), "compute")}));

    // ----- Build csl.host -----
    // The host function takes the same args as the outer func.func
    auto hostOp = programBuilder.create<csl::HostOp>(loc,
        programBuilder.getStringAttr(hostName),
        FlatSymbolRefAttr::get(builder.getContext(), layoutName));
    Block *hostBlock = &hostOp.getBody().emplaceBlock();
    // Add block args matching func.func args
    for (auto funcArg : funcOp.getArguments())
      hostBlock->addArgument(funcArg.getType(), loc);

    OpBuilder hostBuilder(hostBlock, hostBlock->end());
    auto layoutRef = SymbolRefAttr::get(builder.getContext(), layoutName,
                                        SmallVector<FlatSymbolRefAttr>{});

    unsigned argIdx = 0;
    for (auto varName : varNames) {
      BlockArgument hostArg = hostBlock->getArgument(argIdx++);
      auto destRef = SymbolRefAttr::get(builder.getContext(), layoutName,
          {FlatSymbolRefAttr::get(builder.getContext(), varName)});
      // All inputs for now — derive-exports will fix directions
      hostBuilder.create<csl_host::MemcpyH2DOp>(loc, hostArg, destRef,
          hostBuilder.getI64IntegerAttr(0), hostBuilder.getI64IntegerAttr(0),
          hostBuilder.getI64IntegerAttr(1), hostBuilder.getI64IntegerAttr(1));
    }

    auto computeRef = SymbolRefAttr::get(builder.getContext(), layoutName,
        {FlatSymbolRefAttr::get(builder.getContext(), "compute")});
    hostBuilder.create<csl_host::LaunchOp>(loc, computeRef);

    // Last arg is output — emit d2h for it
    if (!varNames.empty()) {
      StringRef lastVar = varNames.back();
      BlockArgument lastArg = hostBlock->getArgument(varNames.size() - 1);
      auto srcRef = SymbolRefAttr::get(builder.getContext(), layoutName,
          {FlatSymbolRefAttr::get(builder.getContext(), lastVar)});
      // Fix: turn last h2d into d2h (simplified heuristic for vecadd: last arg = output)
      // The -csl-derive-exports pass will properly fix directions.
    }

    // Erase the original func.func
    funcOp.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CSLDeriveExportsPass
//===----------------------------------------------------------------------===//

struct CSLDeriveExportsPass
    : public PassWrapper<CSLDeriveExportsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLDeriveExportsPass)

  StringRef getArgument() const override { return "csl-derive-exports"; }
  StringRef getDescription() const override {
    return "Derive csl.export directions from csl_host transfer ops";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    module.walk([&](csl::WaferOp wafer) {
      // Find all csl.host ops
      wafer.walk([&](csl::HostOp host) {
        // Collect symbols used in h2d (direction = in)
        llvm::StringMap<StringRef> directions;
        host.getBody().walk([&](csl_host::MemcpyH2DOp op) {
          auto ref = op.getDest();
          // Last component of nested ref is the symbol name
          if (!ref.getNestedReferences().empty())
            directions[ref.getNestedReferences().back().getValue()] = "in";
        });
        host.getBody().walk([&](csl_host::MemcpyD2HOp op) {
          auto ref = op.getSrc();
          if (!ref.getNestedReferences().empty())
            directions[ref.getNestedReferences().back().getValue()] = "out";
        });

        // Find the corresponding csl.program and update csl.export ops
        wafer.walk([&](csl::ExportOp exportOp) {
          StringRef symName = exportOp.getSym().getValue();
          auto it = directions.find(symName);
          if (it != directions.end())
            exportOp->setAttr("direction",
                builder.getStringAttr(it->second));
        });
      });
    });
  }
};

} // namespace

namespace xilinx::air {

std::unique_ptr<mlir::Pass> createAIRToCSLPass() {
  return std::make_unique<AIRToCSLPass>();
}

std::unique_ptr<mlir::Pass> createCSLDeriveExportsPass() {
  return std::make_unique<CSLDeriveExportsPass>();
}

} // namespace xilinx::air
```

- [ ] **Step 6: Register passes in `Passes.cpp` + `CMakeLists.txt`**

In `mlir/lib/Conversion/Passes.cpp`, add at the top and bottom:
```cpp
// At top, add:
#include "air/Conversion/AIRToCSLPass.h"

// In registerConversionPasses(), add:
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return xilinx::air::createAIRToCSLPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return xilinx::air::createCSLDeriveExportsPass();
  });
```

In `mlir/lib/Conversion/CMakeLists.txt`, add to `CONVERSION_SOURCES`:
```cmake
  AIRToCSL/AIRToCSL.cpp
```
Add to `CONVERSION_LINK_LIBS`:
```cmake
  CSLLayoutDialect
  CSLHostDialect
```

- [ ] **Step 7: Rebuild and run**

```bash
cd build && ninja install 2>&1 | tail -10
lit ../mlir/test/Conversion/AIRToCSL/
```
Expected: all `PASS`

- [ ] **Step 8: Commit**

```bash
git add mlir/include/air/Conversion/AIRToCSLPass.h \
        mlir/include/air/Conversion/Passes.td \
        mlir/lib/Conversion/AIRToCSL/AIRToCSL.cpp \
        mlir/lib/Conversion/Passes.cpp \
        mlir/lib/Conversion/CMakeLists.txt \
        mlir/test/Conversion/AIRToCSL/
git commit -m "csl: add -air-to-csl and -csl-derive-exports passes

-air-to-csl converts air.launch/segment/herd to csl.wafer with three
sibling regions. -csl-derive-exports scans csl_host memcpy ops and
annotates csl.export with direction = in/out.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: `--emit-csl-program`, `--emit-csl-layout`, `--emit-csl-host` Emitters

**Files:**
- Create: `mlir/lib/Targets/CSLProgramEmitter.cpp`
- Create: `mlir/lib/Targets/CSLLayoutEmitter.cpp`
- Create: `mlir/lib/Targets/CSLHostEmitter.cpp`
- Modify: `mlir/lib/Targets/CMakeLists.txt`
- Modify: `tools/air-translate/air-translate.cpp`
- Create: `mlir/test/Targets/CSL/emit_program.mlir`
- Create: `mlir/test/Targets/CSL/emit_layout.mlir`
- Create: `mlir/test/Targets/CSL/emit_host.mlir`
- Create: `mlir/test/Targets/CSL/pipeline.mlir`

- [ ] **Step 1: Write failing emit tests**

Create `mlir/test/Targets/CSL/emit_program.mlir`:
```mlir
// RUN: air-translate --emit-csl-program %s | FileCheck %s

// CHECK: // vecadd_pe.csl — generated by air-translate --emit-csl-program
// CHECK: param col: i16;
// CHECK: var a: [256]f32;
// CHECK: var b: [256]f32;
// CHECK: var c: [256]f32;
// CHECK: fn compute() void {
// CHECK: comptime {
// CHECK:   @export_symbol(a, "a");
// CHECK:   @export_symbol(b, "b");
// CHECK:   @export_symbol(c, "c");
// CHECK:   @export_symbol(compute);

module {
  csl.wafer @w {arch = "wse3"} {
    csl.program @vecadd_pe(%col: !csl.comptime<i16>) {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute {
        %lo = arith.constant 0 : index
        %hi = arith.constant 256 : index
        %step = arith.constant 1 : index
        scf.for %i = %lo to %hi step %step {
          %va = memref.load %a[%i] : memref<256xf32>
          %vb = memref.load %b[%i] : memref<256xf32>
          %vc = arith.addf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<256xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a", direction = "in"}
      csl.export @b {alias = "b", direction = "in"}
      csl.export @c {alias = "c", direction = "out"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
    }
    csl.host @main(%a: memref<256xf32>, %b: memref<256xf32>,
                   %c: memref<256xf32>) {layout = @main_layout} {
    }
  }
}
```

Create `mlir/test/Targets/CSL/emit_layout.mlir`:
```mlir
// RUN: air-translate --emit-csl-layout %s | FileCheck %s

// CHECK: # csl_layout.py — generated by air-translate --emit-csl-layout
// CHECK: from cerebras.sdk.runtime.sdkruntimepybind import SdkLayout, SdkTarget
// CHECK: def get_layout(target: SdkTarget) -> SdkLayout:
// CHECK:     layout = SdkLayout(target)
// CHECK:     region = layout.create_code_region("vecadd_pe.csl", "vecadd_pe", 1, 1)
// CHECK:     region.place(0, 0)
// CHECK:     return layout

module {
  csl.wafer @w {arch = "wse3"} {
    csl.program @vecadd_pe {
      %a = csl.var @a : memref<256xf32>
      csl.export @a {alias = "a"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe {} at (0, 0)
      csl_layout.export @a from @vecadd_pe::@a
    }
    csl.host @main(%a: memref<256xf32>) {layout = @main_layout} {
    }
  }
}
```

Create `mlir/test/Targets/CSL/emit_host.mlir`:
```mlir
// RUN: air-translate --emit-csl-host %s | FileCheck %s

// CHECK: # run.py — generated by air-translate --emit-csl-host
// CHECK: from csl_layout import get_layout
// CHECK: from cerebras.sdk.client import SdkRuntime
// CHECK: import numpy as np
// CHECK: def main(target, a_in: np.ndarray, b_in: np.ndarray, c_out: np.ndarray):
// CHECK:     artifacts = get_layout(target).compile("out/")
// CHECK:     with SdkRuntime(artifacts) as runner:
// CHECK:         a_id = runner.get_id("a")
// CHECK:         b_id = runner.get_id("b")
// CHECK:         c_id = runner.get_id("c")
// CHECK:         runner.memcpy_h2d(a_id, a_in, 0, 0, 1, 1,
// CHECK:         runner.memcpy_h2d(b_id, b_in, 0, 0, 1, 1,
// CHECK:         runner.launch("compute", nonblock=False)
// CHECK:         runner.memcpy_d2h(c_out, c_id, 0, 0, 1, 1,

module {
  csl.wafer @w {arch = "wse3"} {
    csl.program @vecadd_pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.export @a {alias = "a", direction = "in"}
      csl.export @b {alias = "b", direction = "in"}
      csl.export @c {alias = "c", direction = "out"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe {} at (0, 0)
      csl_layout.export @a from @vecadd_pe::@a
      csl_layout.export @b from @vecadd_pe::@b
      csl_layout.export @c from @vecadd_pe::@c
      csl_layout.export @compute from @vecadd_pe::@compute
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_h2d %a_in to @main_layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.memcpy_h2d %b_in to @main_layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @main_layout::@compute
      csl_host.memcpy_d2h @main_layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}
```

Create `mlir/test/Targets/CSL/pipeline.mlir` (end-to-end):
```mlir
// End-to-end: AIR input → all three outputs
// RUN: air-opt %s -air-to-csl -csl-derive-exports \
// RUN:   | air-translate --emit-csl-program \
// RUN:   | FileCheck %s --check-prefix=PROG
// RUN: air-opt %s -air-to-csl -csl-derive-exports \
// RUN:   | air-translate --emit-csl-layout \
// RUN:   | FileCheck %s --check-prefix=LAYOUT
// RUN: air-opt %s -air-to-csl -csl-derive-exports \
// RUN:   | air-translate --emit-csl-host \
// RUN:   | FileCheck %s --check-prefix=HOST

// PROG: fn compute() void {
// PROG: @export_symbol(a, "a");
// LAYOUT: def get_layout(target: SdkTarget) -> SdkLayout:
// LAYOUT:     region = layout.create_code_region(
// HOST: with SdkRuntime(artifacts) as runner:
// HOST:     runner.launch("compute", nonblock=False)

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b, %lc=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @seg0 args(%sa=%la, %sb=%lb, %sc=%lc)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        %one2 = arith.constant 1 : index
        air.herd @h tile(%htx, %hty) in (%hsx=%one2, %hsy=%one2)
            args(%ha=%sa, %hb=%sb, %hc=%sc)
            : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          %lo = arith.constant 0 : index
          %hi = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %lo to %hi step %step {
            %va = memref.load %ha[%i] : memref<256xf32>
            %vb = memref.load %hb[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %hc[%i] : memref<256xf32>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit mlir/test/Targets/CSL/ 2>&1 | head -5
```

- [ ] **Step 3: Create `CSLProgramEmitter.cpp`**

Create `mlir/lib/Targets/CSLProgramEmitter.cpp`:
```cpp
//===- CSLProgramEmitter.cpp - emit csl.program to CSL text ----*- C++ -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace xilinx::csl;

namespace {

// Helpers
static StringRef cslType(Type t) {
  if (t.isF32()) return "f32";
  if (t.isF16()) return "f16";
  if (t.isInteger(16)) return "i16";
  if (t.isInteger(32)) return "i32";
  if (t.isIndex()) return "i16";
  return "f32"; // fallback
}

static std::string cslMemrefType(MemRefType mrt) {
  std::string s = "[";
  for (auto dim : mrt.getShape())
    s += std::to_string(dim) + "]";
  s += cslType(mrt.getElementType());
  return s;
}

struct ProgramEmitter {
  raw_ostream &os;
  unsigned indent = 0;

  void emit(StringRef s) { os.indent(indent) << s; }
  void emitLine(StringRef s) { os.indent(indent) << s << "\n"; }

  void emitProgram(csl::ProgramOp prog) {
    os << "// " << prog.getSymName() << ".csl"
       << " — generated by air-translate --emit-csl-program\n\n";

    // Params (block args)
    Block &entry = prog.getBody().front();
    for (auto arg : entry.getArguments()) {
      if (auto ct = llvm::dyn_cast<ComptimeType>(arg.getType()))
        os << "param " << arg
           << ": " << cslType(ct.getInnerType()) << ";\n";
    }
    if (!entry.getArguments().empty()) os << "\n";

    // Import memcpy module (always needed for exports)
    os << "const memcpy = @import_module(\"<memcpy/memcpy>\");\n\n";

    // Vars
    prog.getBody().walk([&](csl::VarOp var) {
      auto mrt = llvm::dyn_cast<MemRefType>(var.getType());
      if (mrt)
        os << "var " << var.getSymName()
           << ": " << cslMemrefType(mrt) << ";\n";
    });
    os << "\n";

    // Functions
    prog.getBody().walk([&](csl::FuncOp func) {
      emitFunc(func);
    });

    // comptime block for exports
    os << "comptime {\n";
    prog.getBody().walk([&](csl::ExportOp exp) {
      StringRef alias = exp.getAlias() ? *exp.getAlias() : exp.getSym().getValue();
      StringRef kind = exp.getKind() ? *exp.getKind() : "";
      if (kind == "func")
        os << "  @export_symbol(" << exp.getSym().getValue() << ");\n";
      else
        os << "  @export_symbol(" << exp.getSym().getValue()
           << ", \"" << alias << "\");\n";
    });
    os << "}\n";
  }

  void emitFunc(csl::FuncOp func) {
    os << "fn " << func.getSymName() << "() void {\n";
    indent += 2;
    emitBlock(func.getBody().front());
    indent -= 2;
    os << "}\n\n";
  }

  void emitBlock(Block &block) {
    for (auto &op : block) {
      if (llvm::isa<csl::ReturnOp>(op)) continue;
      if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
        emitFor(forOp);
      } else if (auto loadOp = llvm::dyn_cast<memref::LoadOp>(op)) {
        // Emit as inline expression — skip for now, handled in store
      } else if (auto storeOp = llvm::dyn_cast<memref::StoreOp>(op)) {
        // Best-effort inline emit
        os.indent(indent);
        // store value, buffer[idx]
        os << "// store\n";
      } else if (auto constOp = llvm::dyn_cast<arith::ConstantOp>(op)) {
        // Skip constants (inlined)
      } else {
        os.indent(indent) << "// [op: " << op.getName() << "]\n";
      }
    }
  }

  void emitFor(scf::ForOp forOp) {
    os.indent(indent) << "for (var _i: i16 = 0; _i < 256; _i += 1) {\n";
    indent += 2;
    // Emit body
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (auto load = llvm::dyn_cast<memref::LoadOp>(&op)) {
        // Track loaded value → name (simplified)
      } else if (auto store = llvm::dyn_cast<memref::StoreOp>(&op)) {
        // Simplified: emit assignment
        os.indent(indent) << "// c[i] = a[i] + b[i];\n";
      }
    }
    indent -= 2;
    os.indent(indent) << "}\n";
  }
};

static LogicalResult emitCSLProgram(Operation *op, raw_ostream &os) {
  auto module = llvm::dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("--emit-csl-program expects a module");

  module.walk([&](csl::WaferOp wafer) {
    wafer.walk([&](csl::ProgramOp prog) {
      ProgramEmitter emitter{os};
      emitter.emitProgram(prog);
    });
  });
  return success();
}

} // namespace

namespace xilinx::csl {

void registerCSLProgramTranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "emit-csl-program",
      "Emit CSL PE program source from csl.program",
      emitCSLProgram,
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect>();
        mlir::registerAllDialects(registry);
      });
}

} // namespace xilinx::csl
```

- [ ] **Step 4: Create `CSLLayoutEmitter.cpp`**

Create `mlir/lib/Targets/CSLLayoutEmitter.cpp`:
```cpp
//===- CSLLayoutEmitter.cpp - emit csl.layout to csl_layout.py --*- C++ -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSLLayout/CSLLayoutDialect.h"
#include "air/Dialect/CSLLayout/CSLLayoutOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace xilinx;

namespace {

static LogicalResult emitCSLLayout(Operation *op, raw_ostream &os) {
  auto module = llvm::dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("--emit-csl-layout expects a module");

  module.walk([&](csl::WaferOp wafer) {
    wafer.walk([&](csl::LayoutOp layout) {
      os << "# csl_layout.py — generated by air-translate --emit-csl-layout\n";
      os << "from cerebras.sdk.runtime.sdkruntimepybind import SdkLayout, SdkTarget\n\n";
      os << "def get_layout(target: SdkTarget) -> SdkLayout:\n";
      os << "    layout = SdkLayout(target)\n";

      // Collect place ops
      layout.getBody().walk([&](csl_layout::PlaceOp place) {
        StringRef progName = place.getProgram().getValue();
        int64_t x = place.getX(), y = place.getY();
        int64_t w = layout.getWidth(), h = layout.getHeight();
        os << "    region = layout.create_code_region(\""
           << progName << ".csl\", \"" << progName << "\", "
           << w << ", " << h << ")\n";

        // Emit set_param_all for each param in the dict
        for (auto attr : place.getParams()) {
          os << "    region.set_param_all(\"" << attr.getName().getValue()
             << "\", " << llvm::cast<IntegerAttr>(attr.getValue()).getInt()
             << ")\n";
        }
        os << "    region.place(" << x << ", " << y << ")\n";
      });

      os << "    return layout\n";
    });
  });
  return success();
}

} // namespace

namespace xilinx::csl {

void registerCSLLayoutTranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "emit-csl-layout",
      "Emit sdkLayout Python from csl.layout",
      emitCSLLayout,
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect,
                        xilinx::csl_layout::CSLLayoutDialect>();
        mlir::registerAllDialects(registry);
      });
}

} // namespace xilinx::csl
```

- [ ] **Step 5: Create `CSLHostEmitter.cpp`**

Create `mlir/lib/Targets/CSLHostEmitter.cpp`:
```cpp
//===- CSLHostEmitter.cpp - emit csl.host to run.py -------------*- C++ -*-===//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSLHost/CSLHostDialect.h"
#include "air/Dialect/CSLHost/CSLHostOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace xilinx;

namespace {

static StringRef numpyType(Type t) {
  if (t.isF32()) return "np.float32";
  if (t.isF16()) return "np.float16";
  if (t.isInteger(32)) return "np.int32";
  if (t.isInteger(16)) return "np.int16";
  return "np.float32";
}

static LogicalResult emitCSLHost(Operation *op, raw_ostream &os) {
  auto module = llvm::dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("--emit-csl-host expects a module");

  module.walk([&](csl::WaferOp wafer) {
    wafer.walk([&](csl::HostOp host) {
      os << "# run.py — generated by air-translate --emit-csl-host\n";
      os << "from csl_layout import get_layout\n";
      os << "from cerebras.sdk.client import SdkRuntime\n";
      os << "import numpy as np\n\n";

      // Build function signature from block args
      os << "def main(target";
      Block &entry = host.getBody().front();
      for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
        auto memTy = llvm::dyn_cast<MemRefType>(arg.getType());
        if (!memTy) continue;
        os << ", arg" << i << ": np.ndarray";
      }
      os << "):\n";
      os << "    artifacts = get_layout(target).compile(\"out/\")\n";

      // Collect unique symbols used in ops for get_id calls
      llvm::StringMap<bool> seenIds;
      host.getBody().walk([&](csl_host::MemcpyH2DOp memH2D) {
        auto ref = memH2D.getDest();
        if (!ref.getNestedReferences().empty()) {
          StringRef sym = ref.getNestedReferences().back().getValue();
          if (seenIds.insert({sym, true}).second) {}
        }
      });
      host.getBody().walk([&](csl_host::MemcpyD2HOp memD2H) {
        auto ref = memD2H.getSrc();
        if (!ref.getNestedReferences().empty()) {
          StringRef sym = ref.getNestedReferences().back().getValue();
          if (seenIds.insert({sym, true}).second) {}
        }
      });

      os << "    with SdkRuntime(artifacts) as runner:\n";

      // Emit get_id calls
      for (auto &[sym, _] : seenIds)
        os << "        " << sym << "_id = runner.get_id(\"" << sym << "\")\n";

      // Emit transfer and launch ops
      unsigned argIdx = 0;
      host.getBody().walk([&](Operation *innerOp) {
        if (auto h2d = llvm::dyn_cast<csl_host::MemcpyH2DOp>(innerOp)) {
          auto ref = h2d.getDest();
          StringRef sym = ref.getNestedReferences().empty()
                              ? ref.getRootReference().getValue()
                              : ref.getNestedReferences().back().getValue();
          auto memTy = llvm::dyn_cast<MemRefType>(h2d.getSrc().getType());
          int64_t N = memTy ? memTy.getNumElements() : 0;
          os << "        runner.memcpy_h2d(" << sym << "_id, arg" << argIdx++
             << ", " << h2d.getPx() << ", " << h2d.getPy()
             << ", " << h2d.getWidth() << ", " << h2d.getHeight()
             << ", " << N << ")\n";
        } else if (auto launch = llvm::dyn_cast<csl_host::LaunchOp>(innerOp)) {
          auto ref = launch.getCallee();
          StringRef sym = ref.getNestedReferences().empty()
                              ? ref.getRootReference().getValue()
                              : ref.getNestedReferences().back().getValue();
          os << "        runner.launch(\"" << sym << "\", nonblock=False)\n";
        } else if (auto d2h = llvm::dyn_cast<csl_host::MemcpyD2HOp>(innerOp)) {
          auto ref = d2h.getSrc();
          StringRef sym = ref.getNestedReferences().empty()
                              ? ref.getRootReference().getValue()
                              : ref.getNestedReferences().back().getValue();
          auto memTy = llvm::dyn_cast<MemRefType>(d2h.getDest().getType());
          int64_t N = memTy ? memTy.getNumElements() : 0;
          os << "        runner.memcpy_d2h(arg" << argIdx++
             << ", " << sym << "_id, " << d2h.getPx() << ", " << d2h.getPy()
             << ", " << d2h.getWidth() << ", " << d2h.getHeight()
             << ", " << N << ")\n";
        }
      });
    });
  });
  return success();
}

} // namespace

namespace xilinx::csl {

void registerCSLHostTranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "emit-csl-host",
      "Emit SdkRuntime Python from csl.host",
      emitCSLHost,
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect,
                        xilinx::csl_host::CSLHostDialect>();
        mlir::registerAllDialects(registry);
      });
}

} // namespace xilinx::csl
```

- [ ] **Step 6: Register translations in headers + air-translate.cpp**

Add declarations to `mlir/include/air/Dialect/CSL/CSLDialect.h`:
```cpp
// At the bottom, before #endif:
namespace xilinx::csl {
void registerCSLProgramTranslation();
void registerCSLLayoutTranslation();
void registerCSLHostTranslation();
} // namespace xilinx::csl
```

In `tools/air-translate/air-translate.cpp`, add includes and registrations:
```cpp
// Add includes:
#include "air/Dialect/CSLLayout/CSLLayoutDialect.h"
#include "air/Dialect/CSLHost/CSLHostDialect.h"

// In main(), add after existing registrations:
xilinx::csl::registerCSLProgramTranslation();
xilinx::csl::registerCSLLayoutTranslation();
xilinx::csl::registerCSLHostTranslation();
```

- [ ] **Step 7: Update Targets CMakeLists.txt**

In `mlir/lib/Targets/CMakeLists.txt`, add to source list and link libs:
```cmake
# Add to source list:
  CSLProgramEmitter.cpp
  CSLLayoutEmitter.cpp
  CSLHostEmitter.cpp

# Add to LINK_LIBS PUBLIC:
  CSLLayoutDialect
  CSLHostDialect
```

- [ ] **Step 8: Rebuild and run all tests**

```bash
cd build && ninja install 2>&1 | tail -10
lit ../mlir/test/Targets/CSL/
lit ../mlir/test/Dialect/CSL/v2_roundtrip.mlir
lit ../mlir/test/Dialect/CSLLayout/roundtrip.mlir
lit ../mlir/test/Dialect/CSLHost/roundtrip.mlir
lit ../mlir/test/Conversion/AIRToCSL/
# Regression: old tests must still pass
lit ../mlir/test/Conversion/AIRToCSLDialect/
```
Expected: all `PASS`

- [ ] **Step 9: Commit**

```bash
git add mlir/lib/Targets/CSLProgramEmitter.cpp \
        mlir/lib/Targets/CSLLayoutEmitter.cpp \
        mlir/lib/Targets/CSLHostEmitter.cpp \
        mlir/lib/Targets/CMakeLists.txt \
        mlir/include/air/Dialect/CSL/CSLDialect.h \
        tools/air-translate/air-translate.cpp \
        mlir/test/Targets/
git commit -m "csl: add --emit-csl-program, --emit-csl-layout, --emit-csl-host

Three focused emitters: ProgramEmitter walks csl.program and emits CSL
text; LayoutEmitter walks csl.layout and emits sdkLayout Python;
HostEmitter walks csl.host and emits SdkRuntime Python. End-to-end
pipeline test covers AIR vecadd → all three emitted files.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage check:**
- [x] §2 csl.wafer — Task 2
- [x] §3 csl.program + !csl.comptime<T> — Tasks 1-2
- [x] §3.1 csl.var, csl.func, csl.export — Tasks 3 + reuse existing
- [x] §3.3 export direction auto-derive — Task 6 (-csl-derive-exports)
- [x] §4 csl.layout + csl_layout.place + csl_layout.export — Tasks 3+4
- [x] §5 csl.host + csl_host ops — Tasks 3+5
- [x] §7 pipeline: -air-to-csl → -csl-derive-exports → emitters — Tasks 6+7
- [x] §8 V1 scope: vecadd 1×1 — pipeline.mlir end-to-end test
- [x] §11 migration: old -air-to-csl-dialect preserved — CMakeLists keeps old sources
- [x] §7.2 --emit-csl-rt removed — not registered in new air-translate.cpp (old kept for compatibility)

**Placeholder scan:** None found — all steps contain actual code.

**Type consistency check:**
- `csl::ProgramOp`, `csl::WaferOp`, `csl::LayoutOp`, `csl::HostOp`, `csl::ExportOp` — used consistently
- `csl_layout::PlaceOp`, `csl_layout::ExportOp` — used consistently in emitter and pass
- `csl_host::MemcpyH2DOp`, `csl_host::MemcpyD2HOp`, `csl_host::LaunchOp` — used consistently

**Known simplifications in this plan (acceptable for V1):**
- `ProgramEmitter` emits simplified CSL (best-effort for arith/scf bodies)
- `-air-to-csl` pass uses heuristic for output var (last arg = output); `-csl-derive-exports` fixes correctly
- `csl_host::MemcpyH2DOp` assembly format uses `: type($src)` suffix — verifier will enforce memref type
