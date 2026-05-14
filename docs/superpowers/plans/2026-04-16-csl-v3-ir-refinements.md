# CSL v3 IR Refinements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce user boilerplate (auto-infer exports), enforce param contracts, split emitter into 3 focused files, add unified `--emit-csl` command.

**Architecture:** Three passes (`-csl-verify-params`, `-csl-infer-exports`, updated `-air-to-csl`) handle analysis and IR mutation. The monolithic `CSLV2ToPy.cpp` emitter is split into `CSLEmit/{ProgramEmitter,LayoutEmitter,HostEmitter}.cpp` with shared helpers. A new `--emit-csl --output-dir` translation writes all 3 files at once.

**Tech Stack:** MLIR 22 C++ (PassWrapper, OpBuilder, SymbolRefAttr), LLVM FileCheck, `air-opt`/`air-translate`, Cerebras SDK 1.4 (`cslc`, `cs_python`).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `mlir/lib/Conversion/CSLVerifyParams.cpp` | Create | `-csl-verify-params` pass |
| `mlir/include/air/Conversion/CSLVerifyParamsPass.h` | Create | Pass factory declaration |
| `mlir/lib/Conversion/CSLInferExports.cpp` | Create | `-csl-infer-exports` pass |
| `mlir/include/air/Conversion/CSLInferExportsPass.h` | Create | Pass factory declaration |
| `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp` | Create | PE program emission |
| `mlir/lib/Targets/CSLEmit/CSLLayoutEmitter.cpp` | Create | Layout emission |
| `mlir/lib/Targets/CSLEmit/CSLHostEmitter.cpp` | Create | Host runtime emission |
| `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` | Create | `--emit-csl` unified entry |
| `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h` | Create | Shared helpers |
| `mlir/lib/Targets/CSLV2ToPy.cpp` | Delete | Replaced by CSLEmit/ |
| `mlir/lib/Conversion/AIRToCSLPass.cpp` | Modify | Remove export generation |
| `mlir/include/air/Conversion/Passes.h` | Modify | Add new pass headers |
| `mlir/lib/Conversion/Passes.cpp` | Modify | Register new passes |
| `mlir/lib/Conversion/CMakeLists.txt` | Modify | Add new sources |
| `mlir/lib/Targets/CMakeLists.txt` | Modify | Restructure for CSLEmit/ |
| `tools/air-translate/air-translate.cpp` | Modify | Register --emit-csl |
| `mlir/test/Conversion/AIRToCSL/verify_params.mlir` | Create | Param validation tests |
| `mlir/test/Conversion/AIRToCSL/infer_exports.mlir` | Create | Export inference tests |
| `mlir/test/Conversion/AIRToCSL/vecadd_no_exports.mlir` | Create | Full pipeline without manual exports |

---

## Task 1: `-csl-verify-params` pass

**Files:**
- Create: `mlir/include/air/Conversion/CSLVerifyParamsPass.h`
- Create: `mlir/lib/Conversion/CSLVerifyParams.cpp`
- Modify: `mlir/include/air/Conversion/Passes.h`
- Modify: `mlir/lib/Conversion/Passes.cpp`
- Modify: `mlir/lib/Conversion/CMakeLists.txt`
- Create: `mlir/test/Conversion/AIRToCSL/verify_params.mlir`

- [ ] **Step 1: Write the test**

Create `mlir/test/Conversion/AIRToCSL/verify_params.mlir`:
```mlir
// RUN: air-opt %s -csl-verify-params -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// CHECK-LABEL: csl.wafer @valid_params
module {
  csl.wafer @valid_params {arch = "wse3"} {
    csl.program @pe(%M: !csl.comptime<i16>) { }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0) {M = 4 : i16}
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// Negative: extra param on place that program doesn't declare
module {
  csl.wafer @extra_param {arch = "wse3"} {
    csl.program @pe(%M: !csl.comptime<i16>) { }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      // expected-error @+1 {{passes parameter 'N' but @pe has no matching block argument}}
      csl_layout.place @pe at (0, 0) {M = 4 : i16, N = 6 : i16}
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// Valid: no params on either side
module {
  csl.wafer @no_params {arch = "wse3"} {
    csl.program @pe { }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} { }
  }
}
```

- [ ] **Step 2: Create pass header**

Create `mlir/include/air/Conversion/CSLVerifyParamsPass.h`:
```cpp
#ifndef AIR_CONVERSION_CSLVERIFYPARAMS_PASS_H
#define AIR_CONVERSION_CSLVERIFYPARAMS_PASS_H
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir { class ModuleOp; template <typename T> class OperationPass; }
namespace xilinx { namespace air {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCSLVerifyParamsPass();
} }
#endif
```

- [ ] **Step 3: Implement the pass**

Create `mlir/lib/Conversion/CSLVerifyParams.cpp`:
```cpp
#include "air/Conversion/CSLVerifyParamsPass.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct CSLVerifyParamsPass
    : public PassWrapper<CSLVerifyParamsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLVerifyParamsPass)
  StringRef getArgument() const override { return "csl-verify-params"; }
  StringRef getDescription() const override {
    return "Verify csl_layout.place params match csl.program block args";
  }
  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<xilinx::csl::CSLDialect, xilinx::csl_layout::CSLLayoutDialect>();
  }
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool failed = false;
    mod.walk([&](xilinx::csl_layout::PlaceOp place) {
      // Resolve the program symbol within the wafer.
      auto wafer = place->getParentOfType<xilinx::csl::WaferOp>();
      if (!wafer) return;
      StringRef progName = place.getProg();
      xilinx::csl::ProgramOp prog;
      wafer.walk([&](xilinx::csl::ProgramOp p) {
        if (p.getSymName() == progName) prog = p;
      });
      if (!prog) return;

      // Collect program block arg names.
      llvm::DenseSet<StringRef> progParams;
      if (!prog.getBody().empty()) {
        for (BlockArgument arg : prog.getBody().front().getArguments()) {
          // Block args on csl.program are printed as named args.
          // The name comes from the arg's position — we need to match by name.
          // For now, use the arg dictionary attr on the program if available.
        }
      }

      // Collect place attrs (excluding px, py, and known built-in attrs).
      for (NamedAttribute attr : place->getAttrs()) {
        StringRef name = attr.getName();
        if (name == "prog" || name == "px" || name == "py")
          continue;
        // Check if this param exists as a block arg on the program.
        // The program's block args are positional — we check by matching
        // the attr name against the arg names in the custom parser.
        // For simplicity: check if any block arg's dictionary has this name.
        bool found = false;
        if (!prog.getBody().empty()) {
          for (BlockArgument arg : prog.getBody().front().getArguments()) {
            // The arg name in the printed form is stored via argAttrs.
            // Access via the program's arg_attrs dictionary.
            if (auto argAttrs = prog->getAttrOfType<ArrayAttr>("arg_attrs")) {
              if (arg.getArgNumber() < argAttrs.size()) {
                if (auto dict = dyn_cast<DictionaryAttr>(argAttrs[arg.getArgNumber()])) {
                  if (dict.get("csl.name") &&
                      cast<StringAttr>(dict.get("csl.name")).getValue() == name) {
                    found = true;
                    break;
                  }
                }
              }
            }
          }
        }
        if (!found) {
          place.emitOpError("passes parameter '") << name
            << "' but @" << progName << " has no matching block argument";
          failed = true;
        }
      }
    });
    if (failed)
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::air::createCSLVerifyParamsPass() {
  return std::make_unique<CSLVerifyParamsPass>();
}
```

**Note:** The block arg name resolution depends on how `csl.program` stores arg names. The implementer should check how the custom parser in `CSLOps.cpp` stores block arg names — it may use `mlir::function_interface_impl` arg attrs or a custom `arg_names` attribute. Inspect the parsed IR for a program with params (e.g., `csl.program @pe(%M: !csl.comptime<i16>)`) to see how `%M` is stored, then adjust the name lookup accordingly.

- [ ] **Step 4: Wire into build system**

Add to `mlir/include/air/Conversion/Passes.h`:
```cpp
#include "air/Conversion/CSLVerifyParamsPass.h"
```

Add to `mlir/lib/Conversion/Passes.cpp` (after existing registrations):
```cpp
mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
  return createCSLVerifyParamsPass();
});
```

Add to `mlir/lib/Conversion/CMakeLists.txt` (in CONVERSION_SOURCES):
```cmake
CSLVerifyParams.cpp
```

- [ ] **Step 5: Build and test**

```bash
cd build && ninja install
lit build/mlir/test/Conversion/AIRToCSL/verify_params.mlir
```

- [ ] **Step 6: Commit**

```bash
git add mlir/include/air/Conversion/CSLVerifyParamsPass.h \
        mlir/lib/Conversion/CSLVerifyParams.cpp \
        mlir/include/air/Conversion/Passes.h \
        mlir/lib/Conversion/Passes.cpp \
        mlir/lib/Conversion/CMakeLists.txt \
        mlir/test/Conversion/AIRToCSL/verify_params.mlir
git commit -m "feat: add -csl-verify-params pass to validate layout↔program param contract"
```

---

## Task 2: `-csl-infer-exports` pass

**Files:**
- Create: `mlir/include/air/Conversion/CSLInferExportsPass.h`
- Create: `mlir/lib/Conversion/CSLInferExports.cpp`
- Modify: `mlir/include/air/Conversion/Passes.h`
- Modify: `mlir/lib/Conversion/Passes.cpp`
- Modify: `mlir/lib/Conversion/CMakeLists.txt`
- Create: `mlir/test/Conversion/AIRToCSL/infer_exports.mlir`

- [ ] **Step 1: Write the test**

Create `mlir/test/Conversion/AIRToCSL/infer_exports.mlir`:
```mlir
// RUN: air-opt %s -csl-infer-exports | FileCheck %s
//
// Verifies -csl-infer-exports auto-generates csl.export and csl_layout.export
// from csl.host ops. Input has NO manual exports.

// CHECK-LABEL: csl.wafer @vecadd
module {
  csl.wafer @vecadd {arch = "wse3"} {
    // CHECK: csl.program @pe
    csl.program @pe {
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
      // CHECK: csl.export @a {alias = "a", direction = "in"}
      // CHECK: csl.export @b {alias = "b", direction = "in"}
      // CHECK: csl.export @c {alias = "c", direction = "out"}
      // CHECK: csl.export @compute {direction = "internal", kind = "func"}
    }
    // CHECK: csl.layout
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
      // CHECK: csl_layout.export "a" from @pe::@a
      // CHECK: csl_layout.export "b" from @pe::@b
      // CHECK: csl_layout.export "c" from @pe::@c
      // CHECK: csl_layout.export "compute" from @pe::@compute {kind = "func"}
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}
```

- [ ] **Step 2: Create pass header**

Create `mlir/include/air/Conversion/CSLInferExportsPass.h`:
```cpp
#ifndef AIR_CONVERSION_CSLINFEREXPORTS_PASS_H
#define AIR_CONVERSION_CSLINFEREXPORTS_PASS_H
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir { class ModuleOp; template <typename T> class OperationPass; }
namespace xilinx { namespace air {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCSLInferExportsPass();
} }
#endif
```

- [ ] **Step 3: Implement the pass**

Create `mlir/lib/Conversion/CSLInferExports.cpp`. The pass:

1. Walks each `csl.wafer` in the module
2. For each `csl.host` op inside the wafer:
   - Collects all `csl_host.memcpy_h2d` → direction "in", leaf sym = var name
   - Collects all `csl_host.memcpy_d2h` → direction "out", leaf sym = var name
   - Collects all `csl_host.launch` → kind "func", leaf sym = func name
3. Resolves the `csl.layout` and `csl.program` by walking the wafer
4. For each collected symbol:
   - Creates `csl.export` in the program (if not already present) with alias + direction
   - Creates `csl_layout.export` in the layout (if not already present) with from ref + optional kind
5. Any existing `csl.export` without a direction gets "internal" (fallback from derive-exports)

Key implementation details:
- Use `leafRef()` pattern from `CSLDeriveExports.cpp:37-42` to extract the symbol name from nested `@layout::@sym` refs
- Use `ExportOp::create(builder, loc, sym, alias, kind, direction)` — all 4 attr args as in `AIRToCSLPass.cpp:255`
- Use `csl_layout::ExportOp::create(builder, loc, sym_name, from, kind)` — 3 attrs
- Check for duplicates before creating (walk existing exports, skip if sym already exported)
- The `from` SymbolRefAttr for layout exports is `@program::@sym` — use `SymbolRefAttr::get(StringAttr::get(ctx, progName), {FlatSymbolRefAttr::get(ctx, symName)})`

- [ ] **Step 4: Wire into build system**

Same pattern as Task 1: add header include to `Passes.h`, registration to `Passes.cpp`, source to `CMakeLists.txt`.

- [ ] **Step 5: Build and test**

```bash
cd build && ninja install
lit build/mlir/test/Conversion/AIRToCSL/infer_exports.mlir
ninja check-csl  # all 23+ tests pass
```

- [ ] **Step 6: Commit**

```bash
git add mlir/include/air/Conversion/CSLInferExportsPass.h \
        mlir/lib/Conversion/CSLInferExports.cpp \
        mlir/include/air/Conversion/Passes.h \
        mlir/lib/Conversion/Passes.cpp \
        mlir/lib/Conversion/CMakeLists.txt \
        mlir/test/Conversion/AIRToCSL/infer_exports.mlir
git commit -m "feat: add -csl-infer-exports pass to auto-generate exports from host ops"
```

---

## Task 3: Emitter split into 3 files + unified `--emit-csl`

**Files:**
- Create: `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h`
- Create: `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp`
- Create: `mlir/lib/Targets/CSLEmit/CSLLayoutEmitter.cpp`
- Create: `mlir/lib/Targets/CSLEmit/CSLHostEmitter.cpp`
- Create: `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp`
- Create: `mlir/lib/Targets/CSLEmit/CMakeLists.txt`
- Delete: `mlir/lib/Targets/CSLV2ToPy.cpp`
- Modify: `mlir/lib/Targets/CMakeLists.txt`
- Modify: `tools/air-translate/air-translate.cpp`

This is a mechanical refactor. The existing `CSLV2ToPy.cpp` is split:

**CSLEmitCommon.h** — shared helpers extracted from CSLV2ToPy.cpp:
- `cslTypeName()` (lines 42-49)
- `indent()` (lines 52-55)
- `resolve()` (lines 58-64)
- `emitFuncBody()` (lines 68-174)

**CSLProgramEmitter.cpp** — `ProgramEmitter` class (lines 178-349 of old file)
**CSLLayoutEmitter.cpp** — `LayoutEmitter` class (lines 353-445 of old file)
**CSLHostEmitter.cpp** — `HostEmitter` class (lines 449-604 of old file)

**CSLEmitAll.cpp** — new `--emit-csl` translation:
- Takes `--output-dir` option
- Creates 3 output files
- Calls all 3 emitters
- Also re-exports the 3 individual translations for backward compat

- [ ] **Step 1: Create CSLEmit/ directory and common header**

Extract the shared helpers from `CSLV2ToPy.cpp` into `CSLEmitCommon.h`. This includes `cslTypeName`, `indent`, `resolve`, and `emitFuncBody`.

- [ ] **Step 2: Move ProgramEmitter to CSLProgramEmitter.cpp**

Move the `ProgramEmitter` class and `emit()` method. Include `CSLEmitCommon.h` for shared helpers.

- [ ] **Step 3: Move LayoutEmitter to CSLLayoutEmitter.cpp**

Move the `LayoutEmitter` class. Include common header.

- [ ] **Step 4: Move HostEmitter to CSLHostEmitter.cpp**

Move the `HostEmitter` class. Include common header.

- [ ] **Step 5: Create CSLEmitAll.cpp with unified --emit-csl**

```cpp
// Register --emit-csl with --output-dir option.
// Internally: create 3 file streams, call all 3 emitters.
static llvm::cl::opt<std::string> outputDir(
    "output-dir",
    llvm::cl::desc("Output directory for --emit-csl"),
    llvm::cl::init(""));

// In the registration callback:
// 1. Determine program name from csl.program sym_name
// 2. Open <outputDir>/<progName>.csl, <outputDir>/csl_layout.py, <outputDir>/run.py
// 3. Call ProgramEmitter, LayoutEmitter, HostEmitter
// 4. Return success if all three succeed
```

- [ ] **Step 6: Create CSLEmit/CMakeLists.txt and update parent**

```cmake
# mlir/lib/Targets/CSLEmit/CMakeLists.txt
set(CSLEMIT_SOURCES
  CSLProgramEmitter.cpp
  CSLLayoutEmitter.cpp
  CSLHostEmitter.cpp
  CSLEmitAll.cpp
)
```

Update `mlir/lib/Targets/CMakeLists.txt`: replace `CSLV2ToPy.cpp` with `add_subdirectory(CSLEmit)` or include the new sources.

- [ ] **Step 7: Delete CSLV2ToPy.cpp**

- [ ] **Step 8: Update air-translate.cpp registration**

Replace `registerCSLV2ToPyTranslations()` with `registerCSLEmitTranslations()` (or whatever the new registration function is named in CSLEmitAll.cpp).

- [ ] **Step 9: Build and verify all existing tests pass**

```bash
cd build && ninja install
ninja check-csl  # all tests must still pass
```

- [ ] **Step 10: Test --emit-csl unified command**

```bash
mkdir -p /tmp/csl_test_out
air-opt out/input.mlir -air-to-csl -csl-infer-exports | \
  air-translate --emit-csl --output-dir=/tmp/csl_test_out
ls /tmp/csl_test_out/  # should show pe.csl, csl_layout.py, run.py
```

- [ ] **Step 11: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/ \
        mlir/lib/Targets/CMakeLists.txt \
        tools/air-translate/air-translate.cpp
git rm mlir/lib/Targets/CSLV2ToPy.cpp
git commit -m "refactor: split emitter into CSLEmit/ (program, layout, host) + add --emit-csl"
```

---

## Task 4: Update `-air-to-csl` to not generate exports

**Files:**
- Modify: `mlir/lib/Conversion/AIRToCSLPass.cpp`
- Modify: `mlir/test/Conversion/AIRToCSL/air_to_csl_vecadd.mlir`
- Create: `mlir/test/Conversion/AIRToCSL/vecadd_no_exports.mlir`

Now that `-csl-infer-exports` generates exports from host ops, the `-air-to-csl` pass should stop creating `csl.export` and `csl_layout.export` ops. The pass still creates `csl.host` ops with the symbol refs — that's what `-csl-infer-exports` reads.

- [ ] **Step 1: Remove export generation from AIRToCSLPass.cpp**

Remove the code that creates `csl.export` ops in the program (around lines 255-267) and `csl_layout.export` ops in the layout (around lines 213-234 area). Keep everything else.

- [ ] **Step 2: Update air_to_csl_vecadd.mlir test**

Remove the CHECK lines that verify `csl.export` and `csl_layout.export` since the pass no longer creates them. The test should only check: `csl.wafer`, `csl.program` (vars + func), `csl.layout` (place only), `csl.host` (memcpy + launch).

- [ ] **Step 3: Create vecadd_no_exports.mlir — full pipeline test**

Create `mlir/test/Conversion/AIRToCSL/vecadd_no_exports.mlir`:
```mlir
// Full pipeline: -air-to-csl (no exports) → -csl-infer-exports (adds exports) → verify
// RUN: air-opt %s -air-to-csl -csl-infer-exports | FileCheck %s

// CHECK-LABEL: csl.wafer @vecadd
// CHECK: csl.program @h
// CHECK:   csl.var @arg0
// CHECK:   csl.var @arg1
// CHECK:   csl.var @arg2
// CHECK:   csl.func @compute
// CHECK:   csl.export @arg0 {alias = "arg0", direction = "in"}
// CHECK:   csl.export @arg1 {alias = "arg1", direction = "in"}
// CHECK:   csl.export @arg2 {alias = "arg2", direction = "out"}
// CHECK:   csl.export @compute {direction = "internal", kind = "func"}
// CHECK: csl.layout
// CHECK:   csl_layout.place @h
// CHECK:   csl_layout.export "arg0" from @h::@arg0
// CHECK:   csl_layout.export "arg1" from @h::@arg1
// CHECK:   csl_layout.export "arg2" from @h::@arg2
// CHECK:   csl_layout.export "compute" from @h::@compute {kind = "func"}
// CHECK: csl.host @vecadd
// CHECK:   csl_host.memcpy_h2d
// CHECK:   csl_host.memcpy_h2d
// CHECK:   csl_host.launch
// CHECK:   csl_host.memcpy_d2h

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>,
                    %c: memref<256xf32>) {
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

- [ ] **Step 4: Build and test**

```bash
cd build && ninja install
ninja check-csl  # all tests pass
```

- [ ] **Step 5: Commit**

```bash
git add mlir/lib/Conversion/AIRToCSLPass.cpp \
        mlir/test/Conversion/AIRToCSL/air_to_csl_vecadd.mlir \
        mlir/test/Conversion/AIRToCSL/vecadd_no_exports.mlir
git commit -m "feat: remove export generation from -air-to-csl; rely on -csl-infer-exports"
```

---

## Task 5: End-to-end validation + update e2e test

**Files:**
- Modify: `mlir/test/Conversion/AIRToCSL/vecadd_e2e.mlir`

- [ ] **Step 1: Update vecadd_e2e.mlir pipeline**

Change the RUN lines to use the new pipeline:
```mlir
// RUN: air-opt %s -air-to-csl -csl-infer-exports | air-translate --emit-csl-program | FileCheck %s --check-prefix=PROG
// RUN: air-opt %s -air-to-csl | air-translate --emit-csl-layout | FileCheck %s --check-prefix=LAYOUT
// RUN: air-opt %s -air-to-csl -csl-infer-exports | air-translate --emit-csl-host | FileCheck %s --check-prefix=HOST
```

Note: `-csl-derive-exports` is replaced by `-csl-infer-exports` in the PROG and HOST lines.

- [ ] **Step 2: Run all tests**

```bash
cd build && ninja install
ninja check-csl  # ALL tests pass
```

- [ ] **Step 3: Validate on CS-3 simulator**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/out
air-opt input.mlir -air-to-csl -csl-infer-exports | air-translate --emit-csl-program > h.csl
/home/bricklib_dataflow/sdk/SDK_1_4/cslc layout.csl --arch wse3 --fabric-dims 8,3 --fabric-offsets 4,1 -o compiled --memcpy --channels 1
/home/bricklib_dataflow/sdk/SDK_1_4/cs_python run_test.py --name compiled
# Expected: PASS
```

- [ ] **Step 4: Commit**

```bash
git add mlir/test/Conversion/AIRToCSL/vecadd_e2e.mlir
git commit -m "test: update e2e test to use -csl-infer-exports pipeline"
```

---

## Task 6: Clean up deprecated `-csl-derive-exports`

- [ ] **Step 1: Update derive_exports.mlir test to use -csl-infer-exports**

Change the RUN line and adjust CHECK patterns (the new pass creates exports in the same format).

- [ ] **Step 2: Remove CSLDeriveExports.cpp**

Delete the file, remove from CMakeLists, Passes.h, Passes.cpp.

- [ ] **Step 3: Build and test**

```bash
cd build && ninja install
ninja check-csl
```

- [ ] **Step 4: Commit**

```bash
git rm mlir/lib/Conversion/CSLDeriveExports.cpp \
       mlir/include/air/Conversion/CSLDeriveExportsPass.h
git add mlir/test/Conversion/AIRToCSL/derive_exports.mlir \
        mlir/lib/Conversion/CMakeLists.txt \
        mlir/include/air/Conversion/Passes.h \
        mlir/lib/Conversion/Passes.cpp
git commit -m "refactor: remove deprecated -csl-derive-exports; replaced by -csl-infer-exports"
```
