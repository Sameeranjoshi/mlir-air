# AIR → CSL Vecadd Milestone — Plan v2

> **Supersedes:** `docs/superpowers/plans/2026-04-13-air-to-csl-vecadd.md` (commit `3453fa6f`) for Phases 1–7.
> **Unchanged from v1:** goal, architecture commitments, file-layout conventions, execution-handoff section. Refer to v1 for those.
> **Drives this rewrite:** Phase 0 landed (commit `d7691801`) and surfaced that the memcpy workflow on SDK 1.4 needs **two** CSL files + an external `cslc --memcpy` step + a simpler `run.py`. Spec amendment in commit `a7d0a7de` §12 has the full discovery.

## Why v2

The v1 plan assumed `run.py` would build a layout inline via `SdkLayout` + `layout.compile()` and that `csl_rt.create_layout`/`create_code_region`/`place`/`compile`/`export_name` ops would lower to those Python calls. The actual memcpy workflow is:

1. Compiler emits `layout.csl` (a CSL file with a `layout { ... }` block)
2. Compiler emits `pe_program.csl` (the PE program with pointer aliases + memcpy import)
3. Compiler emits a simpler `run.py` that operates on a **pre-compiled** directory
4. External step: `cslc --arch=wse3 layout.csl --memcpy --channels 1 --fabric-dims=8,3 --fabric-offsets=4,1 -o out`
5. Run step: `cs_python run.py --name out --check`

This collapses the `csl_rt.create_layout / create_code_region / place / compile / set_param_all / export_name` lowering (it's not needed — those Python-side calls don't happen) and adds a new **LayoutEmitter** translator sub-component that writes `layout.csl` from the `csl.*` spatial ops directly.

## What v1 got wrong (concretely)

| Claim | Reality | Impact on v2 |
|---|---|---|
| "`csl.comptime` wraps `csl.export_symbol`" | `csl.comptime` was removed in the March 2026 consolidation. `csl.export_symbol` goes **directly** inside `csl.kernel`. The translator wraps them in a `comptime { }` block when emitting. | IR contract in §2 drops the `csl.comptime` nesting. |
| "Single `run.py` builds layout + runs" | Two CSL files + external cslc + simpler run.py | Phase 5 splits into LayoutEmitter + KernelEmitter + HostEmitter. |
| "`csl-to-csl-rt` lowers layout ops to `csl_rt.*` layout ops" | Those Python calls don't exist in the memcpy path. | Phase 3 only synthesizes the runtime sequence (`runtime_create` → `memcpy_h2d` → `launch` → `memcpy_d2h` → `stop`), leaving `csl.*` spatial ops in place. |
| "`csl.var` can carry the pointer alias" | `csl.var` has only `sym_name` and `type`, no initializer. | Pointer aliases (`var a_ptr: [*]f32 = &a_buf`) are emitted by the KernelEmitter, not represented in IR. Driven by which buffers the host exports. |
| "csl.export_name and csl.export_symbol need to be added" | **Already in the working tree** (uncommitted at the time of this writing). | Phase 1 shrinks to "verify + commit what's already there." |

## Phase structure (v2)

| Phase | What lands | Build-dependent? |
|---|---|---|
| **1 (revised)** | Verify the uncommitted `csl.export_name` / `csl.export_symbol` ops + `export_ops.mlir` test, commit them | Yes (needs `ninja install` + `lit`) |
| **2** | `air-to-csl-dialect` pass scaffold + lowering + 7 reject tests | Yes |
| **3** | `csl-to-csl-rt` extension: runtime sequence synthesis only (no layout lowering) | Yes |
| **4** | LayoutEmitter (new sub-component) — walks `csl.*` spatial ops → `layout.csl` | Yes |
| **5** | KernelEmitter rework — walks `csl.kernel` → `pe_program.csl` with memcpy patterns | Yes |
| **6** | HostEmitter rework — walks `csl_rt.*` → simpler `run.py` (opens pre-compiled dir) | Yes |
| **7** | Hardware test (e2e on CS-3 via `air-opt | air-translate → cslc → cs_python`) | Yes |
| **8** | Quarantine `AIRToCSLPass.cpp` (Phase-1 emitter) | Yes |

Phase 4 and Phase 5 swap responsibilities compared to v1's KernelEmitter/HostEmitter split. The new three-way split is:

- **LayoutEmitter** → `layout.csl` (was not in v1; new component)
- **KernelEmitter** → `pe_program.csl` (was in v1; now needs memcpy-specific additions)
- **HostEmitter** → `run.py` (was in v1; now simpler — no `build_layout()` Python)

## The updated IR contract (Phase 2 output)

This is what `air-opt -air-to-csl-dialect vecadd.mlir` produces. It's the single most important artifact in the plan — every downstream phase consumes it.

```mlir
module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    csl.spatial_placement {
      %k = csl.kernel {
        csl.var @a_buf : memref<256xf32>
        csl.var @b_buf : memref<256xf32>
        csl.var @c_buf : memref<256xf32>

        csl.func @compute {
          %c0   = arith.constant 0   : index
          %c256 = arith.constant 256 : index
          %c1   = arith.constant 1   : index
          scf.for %i = %c0 to %c256 step %c1 {
            %va = memref.load %a_buf[%i] : memref<256xf32>
            %vb = memref.load %b_buf[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb     : f32
            memref.store %vc, %c_buf[%i]  : memref<256xf32>
          }
          csl.return
        }

        // No csl.comptime wrapper. These ops live directly in the kernel
        // body; the translator wraps them in `comptime { }` during emission.
        csl.export_symbol @a_buf alias("a")
        csl.export_symbol @b_buf alias("b")
        csl.export_symbol @c_buf alias("c")
        csl.export_symbol @compute
      } {source_file = "vecadd_pe.csl"} : !csl.kernel

      %r = csl.code_region routes() colors() {
      } {width = 1 : i64, height = 1 : i64} : !csl.code_region

      csl.place %r %k {x = 0 : i64, y = 0 : i64}
    }

    csl.export_name "a" : memref<256xf32>{direction = "in"}
    csl.export_name "b" : memref<256xf32>{direction = "in"}
    csl.export_name "c" : memref<256xf32>{direction = "out"}
    csl.export_name "compute" : () -> ()

    return
  }
}
```

**Key differences from v1 contract:**
- No `csl.comptime` wrapper around `csl.export_symbol`.
- No space before `{direction = ...}` — matches the printer in the existing test file.
- No pointer-alias ops in IR — the KernelEmitter synthesizes `var a_ptr: [*]f32 = &a_buf;` from the combination of `csl.var` + matching `csl.export_symbol`.
- No `csl.import_module` for memcpy in IR — the KernelEmitter emits `const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);` unconditionally when any host export exists.
- No `param memcpy_params: comptime_struct;` op — same, the KernelEmitter emits this header unconditionally.

The IR stays clean; the emission logic has the target-specific details.

---

## Phase 1 — Verify Phase 1 ops are committable

**State on 2026-04-14:** `csl.export_name` and `csl.export_symbol` already exist in the working tree at `mlir/include/air/Dialect/CSL/CSLOps.td` lines 415–475, and `mlir/test/Dialect/CSL/export_ops.mlir` is an untracked round-trip test. Both match the shape I would have written (modulo the `{direction = ...}` printer format). **Phase 1 collapses to: build + run the test + commit.**

### Task 1.1 — Verify the test passes and commit

- [ ] **Step 1:** Confirm the build is current with these changes included:
  ```bash
  cd build && ninja install 2>&1 | tail -10
  ```
  Expected: clean build. TableGen regenerates `CSLOps.h.inc` with the two new op classes.

- [ ] **Step 2:** Run the round-trip test:
  ```bash
  cd build && lit -v ../mlir/test/Dialect/CSL/export_ops.mlir
  ```
  Expected: PASS.

- [ ] **Step 3:** Run the full CSL dialect test suite to check for regressions:
  ```bash
  cd build && lit -v ../mlir/test/Dialect/CSL/
  ```
  Expected: all tests pass.

- [ ] **Step 4:** Commit only these two files. **Do not** touch `tools/air-runner/CMakeLists.txt` or `utils/build-*.sh` (separate user work).
  ```bash
  git add mlir/include/air/Dialect/CSL/CSLOps.td mlir/test/Dialect/CSL/export_ops.mlir
  git commit -m "feat(csl): add csl.export_name and csl.export_symbol ops

  These ops are required by the AIR → CSL vecadd milestone. They were
  referenced by the design spec but never defined; the implementation
  was added separately to CSLOps.td (lines 415-475) and a round-trip
  test at mlir/test/Dialect/CSL/export_ops.mlir.

  - csl.export_name: name + type + optional direction (in/out) for
    host-visible buffers or functions. Consumed by csl-to-csl-rt and
    by the LayoutEmitter translator.
  - csl.export_symbol: lives directly inside csl.kernel (no comptime
    wrapper — csl.comptime was removed in the March 2026 consolidation,
    the translator synthesizes the comptime block).

  See docs/superpowers/specs/2026-04-13-air-to-csl-vecadd-design.md §5.3.1
  and docs/superpowers/plans/2026-04-14-air-to-csl-vecadd-v2.md §Phase 1."
  ```

---

## Phase 2 — `air-to-csl-dialect` pass

### Task 2.1 — Scaffold the pass

**Files:**
- Create: `mlir/include/air/Conversion/AIRToCSLDialectPass.h`
- Create: `mlir/lib/Conversion/AIRToCSLDialect/CMakeLists.txt`
- Create: `mlir/lib/Conversion/AIRToCSLDialect/AIRToCSLDialect.cpp`
- Modify: `mlir/include/air/Conversion/Passes.td` (add `AIRToCSLDialect` pass def)
- Modify: `mlir/lib/Conversion/CMakeLists.txt` (add `add_subdirectory(AIRToCSLDialect)`)

Follow the pattern of the existing `CSLToCSLRuntime` pass — look at its header, CMakeLists, and .cpp for the exact shape. The pass is initially an empty `runOnOperation()` that does nothing; Task 2.3 fills it in.

The header:
```cpp
#ifndef AIR_CONVERSION_AIRTOCSLDIALECT_PASS_H
#define AIR_CONVERSION_AIRTOCSLDIALECT_PASS_H
#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir { class ModuleOp; template <typename T> class OperationPass; }
namespace xilinx::air {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAIRToCSLDialectPass();
}
#endif
```

The `Passes.td` addition (insert directly above the existing `def AIRToCSL`):
```tablegen
def AIRToCSLDialect : Pass<"air-to-csl-dialect", "ModuleOp"> {
  let summary = "Lower AIR dialect (1x1 herds only) to CSL dialect ops";
  let constructor = "xilinx::air::createAIRToCSLDialectPass()";
  let description = [{
    Converts a func.func containing air.launch / air.segment / air.herd
    (1x1 grid only, milestone scope) into csl.spatial_placement with
    csl.kernel, csl.code_region, csl.place, and host-level csl.export_name
    ops. The herd compute body (standard arith/scf/memref) is moved
    verbatim into a csl.func @compute inside csl.kernel. csl.export_symbol
    ops live directly in the kernel body (no csl.comptime wrapper).

    See docs/superpowers/plans/2026-04-14-air-to-csl-vecadd-v2.md.
  }];
}
```

- [ ] **Steps 1–6:** Follow the v1 plan Task 2.1 for the CMakeLists, empty .cpp body, and build-verification. The only change is: for `-air-to-csl-dialect` to appear in `air-opt --help`, the pass library must be linked into `air-opt`. Check `tools/air-opt/CMakeLists.txt` after the build — if there's an explicit LIBS list, add `AIRToCSLDialectPass` to it.

- [ ] **Step 7: Commit**
  ```bash
  git add mlir/include/air/Conversion/AIRToCSLDialectPass.h \
          mlir/include/air/Conversion/Passes.td \
          mlir/lib/Conversion/AIRToCSLDialect/ \
          mlir/lib/Conversion/CMakeLists.txt
  git commit -m "feat(air-to-csl-dialect): scaffold empty pass"
  ```

### Task 2.2 — Failing FileCheck test

**File:** `mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir`

Content: The AIR vecadd input from the end of this document's §2, with CHECK lines pinning the IR contract above. The CHECK lines need to account for the no-comptime structure:

```mlir
// RUN: air-opt %s -air-to-csl-dialect | FileCheck %s

// CHECK-LABEL: func.func @vecadd
// CHECK:   csl.spatial_placement {
// CHECK:     %[[K:.*]] = csl.kernel {
// CHECK-DAG:    csl.var @a_buf : memref<256xf32>
// CHECK-DAG:    csl.var @b_buf : memref<256xf32>
// CHECK-DAG:    csl.var @c_buf : memref<256xf32>
// CHECK:        csl.func @compute {
// CHECK:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:            %{{.*}} = memref.load %a_buf[%{{.*}}] : memref<256xf32>
// CHECK:            %{{.*}} = memref.load %b_buf[%{{.*}}] : memref<256xf32>
// CHECK:            %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK:            memref.store %{{.*}}, %c_buf[%{{.*}}] : memref<256xf32>
// CHECK:          }
// CHECK:          csl.return
// CHECK:        }
// CHECK-DAG:    csl.export_symbol @a_buf alias("a")
// CHECK-DAG:    csl.export_symbol @b_buf alias("b")
// CHECK-DAG:    csl.export_symbol @c_buf alias("c")
// CHECK-DAG:    csl.export_symbol @compute
// CHECK:     } {source_file = "vecadd_pe.csl"} : !csl.kernel
// CHECK:     %[[R:.*]] = csl.code_region routes() colors() {
// CHECK:     } {width = 1 : i64, height = 1 : i64} : !csl.code_region
// CHECK:     csl.place %[[R]] %[[K]] {x = 0 : i64, y = 0 : i64}
// CHECK:   }
// CHECK-DAG: csl.export_name "a" : memref<256xf32>{direction = "in"}
// CHECK-DAG: csl.export_name "b" : memref<256xf32>{direction = "in"}
// CHECK-DAG: csl.export_name "c" : memref<256xf32>{direction = "out"}
// CHECK-DAG: csl.export_name "compute" : () -> ()

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%a, %b0=%b, %c0=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @seg0 args(%a1=%a0, %b1=%b0, %c1_=%c0)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile(%htx, %hty) in (%hsx=%c1_0, %hsy=%c1_0)
            args(%a2=%a1, %b2=%b1, %c2=%c1_)
            : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1_1 = arith.constant 1 : index
          scf.for %i = %c0 to %c256 step %c1_1 {
            %va = memref.load %a2[%i] : memref<256xf32>
            %vb = memref.load %b2[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %c2[%i] : memref<256xf32>
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

- [ ] **Step 1:** Save the file. - [ ] **Step 2:** Run `lit -v`, expect FAIL (pass body is empty). - [ ] **Step 3:** Commit.

### Task 2.3 — Implement the lowering

Same as v1 Task 2.3 with three adjustments:

1. Do **not** emit a `csl.comptime` op — the `csl.export_symbol` ops are appended directly to the `csl.kernel`'s body block, alongside `csl.var` and `csl.func`.
2. Map the host `func.func` memref args to direction via position: first (N-1) args are `"in"`, last is `"out"`. For vecadd that's `%a, %b` → in, `%c` → out. For future non-3-arg kernels, add a TODO to take an explicit `direction` attribute on func args.
3. Skip the pointer-alias generation — that's the KernelEmitter's job, not this pass's.

The implementation walks once, validates (rejecting all milestone-out-of-scope patterns with `emitOpError`), then builds the csl.* IR using `OpBuilder`. See v1 Task 2.3 for the full pseudo-code — the only structural change is that the `csl.comptime` creation block is deleted; `csl.export_symbol` ops are created with `builder.setInsertionPointToEnd(kernelBlock)` instead.

Rejection conditions (from spec §7.2): unchanged from v1. 1×1 only, no channels, no dma, no async, no dynamic memref, no multiple herds, no non-{f32,f16,i32,i16} eltype. Each rejection is a single `op->emitOpError("...")` call.

- [ ] **Step 1:** Implement `lowerFunc(func::FuncOp)` per the sketch in v1 Task 2.3, minus comptime.
- [ ] **Step 2:** Build. - [ ] **Step 3:** Run `lit -v` on `vecadd.mlir`, expect PASS. - [ ] **Step 4:** Commit.

### Task 2.4 — Rejection tests

Unchanged from v1 Task 2.4–2.10. Seven `reject_*.mlir` files, each with `// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s` and an `expected-error` substring. File contents identical to v1.

- [ ] Write all 7 files, run `lit -v mlir/test/Conversion/AIRToCSLDialect/`, commit as one batch.

---

## Phase 3 — `csl-to-csl-rt` extension (runtime sequence only)

**Major change from v1:** this pass **no longer lowers the layout-building ops**. It only synthesizes the host-side runtime sequence. The `csl.spatial_placement` op stays in the module for the translator to consume directly.

### Task 3.1 — Remove the SpatialPlacementConversionPattern legalization

The existing `CSLToCSLRuntime.cpp` (mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp, 194 LOC) currently has:
- `SpatialPlacementConversionPattern` (lines 31–152) that creates `csl_rt.create_layout`/`create_code_region`/`place`/`compile` ops and erases `csl.spatial_placement`.
- `target.addIllegalOp<SpatialPlacementOp>()` (line 177) forces the pattern to match.

**New behavior:**
- Delete `SpatialPlacementConversionPattern` entirely (the 122 lines from 31–152).
- Remove `target.addIllegalOp<SpatialPlacementOp>()`.
- `csl.spatial_placement` becomes a legal, non-lowered op in the output IR.
- Add a new walk-based pass body (not `applyPartialConversion` — just iterate the host func.func once) that appends runtime-sequence ops.

### Task 3.2 — Synthesize runtime sequence

For each `func.func` in the module that contains `csl.export_name` ops at the top level:

1. Collect all `csl.export_name` ops in declaration order: partition into inputs (`direction = "in"`), outputs (`direction = "out"`), and host-callable functions (`() -> ()` type, no direction).
2. At the end of the func body (before `return`), create:
   - `%rt = csl_rt.runtime_create` (no operands — the runtime is constructed from the pre-compiled dir at Python side, not from in-memory artifacts; `csl_rt.runtime_create` produces a placeholder SSA value that the HostEmitter recognizes but doesn't translate to a Python call)
   - `%loaded = csl_rt.load %rt`
   - For each input, a `csl_rt.memcpy_h2d` op carrying the buffer name and element count
   - `csl_rt.launch` for the host-callable function name
   - For each output, a `csl_rt.memcpy_d2h`
   - `csl_rt.stop`
3. Propagate the `direction` attribute onto `csl_rt.memcpy_h2d`/`memcpy_d2h` as a discardable attr if HostEmitter needs it (probably not — the op type already determines direction).

**Important:** `csl_rt.runtime_create` currently takes a `compile_artifacts` operand (see `CSLRuntimeOps.td` line ~180). We have no compile_artifacts SSA value in the new world. Options:

- **Option A (easiest):** Synthesize a placeholder `csl_rt.compile_artifacts` constant (e.g., `%art = csl_rt.dummy_artifacts`) and feed it to `runtime_create`. The HostEmitter ignores the artifacts SSA and emits `runtime = SdkRuntime(args.name, ...)` from the command-line arg.
- **Option B:** Make `csl_rt.runtime_create`'s operand optional (TableGen tweak). Cleaner IR, more code to change.

Pick A for this milestone. Add a comment on the synthesized op: `// pre-compiled path; HostEmitter reads args.name`. Option B is a follow-up refinement.

Actually — simpler still: **don't generate `csl_rt.runtime_create` / `csl_rt.load` / `csl_rt.stop` at all**. The HostEmitter already knows it needs to emit `runtime = SdkRuntime(args.name, cmaddr=args.cmaddr)`, `runtime.load()`, `runtime.run()`, `runtime.stop()` as boilerplate around the memcpy sequence. The pass only needs to synthesize the **data-movement** ops (`memcpy_h2d`, `launch`, `memcpy_d2h`), which are the only ones that carry per-program information.

**Revised task:** the pass synthesizes only `csl_rt.memcpy_h2d`, `csl_rt.launch`, `csl_rt.memcpy_d2h` in the correct order. The HostEmitter wraps them in the runtime lifecycle boilerplate.

- [ ] **Step 1:** Delete the `SpatialPlacementConversionPattern` and the partial-conversion target setup.
- [ ] **Step 2:** Write a walk-based runOnOperation:

```cpp
void runOnOperation() override {
  ModuleOp module = getOperation();
  module.walk([&](func::FuncOp func) {
    SmallVector<csl::ExportNameOp> inExports, outExports;
    csl::ExportNameOp fnExport = nullptr;
    func.walk([&](csl::ExportNameOp en) {
      auto ty = en.getExportedType();
      auto dir = en.getDirection();  // Optional<StringRef>
      if (dir && *dir == "in")  inExports.push_back(en);
      else if (dir && *dir == "out") outExports.push_back(en);
      else if (isa<FunctionType>(ty)) fnExport = en;
    });
    if (inExports.empty() && outExports.empty() && !fnExport) return;

    OpBuilder builder(func.getBody());
    builder.setInsertionPoint(func.getBody().front().getTerminator());
    Location loc = func.getLoc();

    for (auto en : inExports) {
      auto memTy = cast<MemRefType>(en.getExportedType());
      int64_t n = memTy.getNumElements();
      builder.create<csl_rt::MemcpyH2dOp>(loc, /* ...args matching op def... */);
    }
    if (fnExport)
      builder.create<csl_rt::LaunchOp>(loc, /* name = fnExport.getSymName() */);
    for (auto en : outExports) {
      auto memTy = cast<MemRefType>(en.getExportedType());
      int64_t n = memTy.getNumElements();
      builder.create<csl_rt::MemcpyD2hOp>(loc, /* ... */);
    }
  });
}
```

Read `CSLRuntimeOps.td` to get the exact argument order for `MemcpyH2dOp` / `LaunchOp` / `MemcpyD2hOp` constructors. The current pass already uses these op types (see CSLRuntimeToPy.cpp lines 300–344 for how they're consumed), so the interface is stable.

- [ ] **Step 3:** Test file `mlir/test/Conversion/CSLToCSLRuntime/vecadd_with_kernel.mlir` — the same hand-written input as v1 Task 3.1, with CHECK lines updated to match the new output (only memcpy+launch+memcpy, no create_layout/place/compile).
- [ ] **Step 4:** Build, run lit, commit.

### Task 3.3 — Migrate existing trivial test

The pre-existing `layout_to_runtime.mlir` test exercises the old `SpatialPlacementConversionPattern` behavior. Since we deleted that pattern, the test will fail. Options:

1. Delete the test (it was only exercising behavior we removed).
2. Rewrite it to match the new pass behavior.

Do (1). The test was covering a single-region spatial_placement → create_layout/place/compile scenario that no longer exists. It has no replacement value.

- [ ] Delete `mlir/test/Conversion/CSLToCSLRuntime/layout_to_runtime.mlir`. - [ ] Commit in the same commit as Task 3.2.

---

## Phase 4 — LayoutEmitter (new sub-component)

**Goal:** Emit `layout.csl` from `csl.spatial_placement` + `csl.code_region` + `csl.place` + host-level `csl.export_name` ops.

### Target output

Given the IR contract above, the LayoutEmitter produces (matching the golden at `mlir/test/Conversion/AIRToCSL/golden/vecadd_layout.csl.golden`):

```csl
const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });

layout {
  @set_rectangle(1, 1);
  @set_tile_code(0, 0, "vecadd_pe.csl", .{ .memcpy_params = memcpy.get_params(0) });

  @export_name("a", [*]f32, true);
  @export_name("b", [*]f32, true);
  @export_name("c", [*]f32, false);
  @export_name("compute", fn()void);
}
```

### Task 4.1 — Failing lit test

**File:** `mlir/test/Targets/CSLRuntimeToCSL/vecadd_layout_emit.mlir`

```mlir
// RUN: air-opt %s | air-translate --emit-csl-rt -o %t/
// RUN: cat %t/vecadd_layout.csl | FileCheck %s

// CHECK: const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });
// CHECK: layout {
// CHECK:   @set_rectangle(1, 1);
// CHECK:   @set_tile_code(0, 0, "vecadd_pe.csl"
// CHECK-DAG: @export_name("a", [*]f32, true);
// CHECK-DAG: @export_name("b", [*]f32, true);
// CHECK-DAG: @export_name("c", [*]f32, false);
// CHECK-DAG: @export_name("compute", fn()void);
// CHECK: }

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    csl.spatial_placement {
      %k = csl.kernel {
        csl.var @a_buf : memref<256xf32>
      } {source_file = "vecadd_pe.csl"} : !csl.kernel
      %r = csl.code_region routes() colors() {
      } {width = 1 : i64, height = 1 : i64} : !csl.code_region
      csl.place %r %k {x = 0 : i64, y = 0 : i64}
    }
    csl.export_name "a" : memref<256xf32>{direction = "in"}
    csl.export_name "b" : memref<256xf32>{direction = "in"}
    csl.export_name "c" : memref<256xf32>{direction = "out"}
    csl.export_name "compute" : () -> ()
    return
  }
}
```

### Task 4.2 — Implement LayoutEmitter

**File:** `mlir/lib/Targets/CSLRuntimeToPy.cpp` — add a new class `LayoutEmitter` above the existing translator class.

Responsibilities:
1. Find the `csl.spatial_placement` op and its `csl.code_region` child. Extract `width` and `height` attrs.
2. Find the `csl.place` op. Extract `x`, `y` attrs and the referenced `csl.kernel`'s `source_file` attr.
3. Walk the host func.func for `csl.export_name` ops. For each:
   - `direction = "in"` → `@export_name("<name>", [*]f32, true);`
   - `direction = "out"` → `@export_name("<name>", [*]f32, false);`
   - no direction, FunctionType → `@export_name("<name>", fn()void);`
   - The `[*]f32` part is derived from the memref element type (extend to `[*]i32`, `[*]f16`, `[*]i16` as the eltype varies).
4. Emit the `const memcpy = @import_module("<memcpy/get_params>", ...)` header. The `.width` and `.height` come from `csl.code_region`'s `width`/`height` attrs.
5. Emit the `layout { ... }` block with `@set_rectangle(W, H)`, `@set_tile_code(x, y, "<source_file>", .{ .memcpy_params = memcpy.get_params(0) })`, and the export_name lines.
6. Write the output to `<outDir>/layout.csl` (name hardcoded; future milestones can make it configurable from `csl.spatial_placement` attrs).

Sketch:

```cpp
class LayoutEmitter {
public:
  LayoutEmitter(StringRef outDir) : outDir(outDir.str()) {}
  LogicalResult emit(ModuleOp module);
private:
  std::string outDir;
  StringRef cslPtrType(Type elt);  // "[*]f32" / "[*]f16" / "[*]i32" / "[*]i16"
};

LogicalResult LayoutEmitter::emit(ModuleOp module) {
  // 1. Find the single spatial_placement
  csl::SpatialPlacementOp sp = nullptr;
  module.walk([&](csl::SpatialPlacementOp op) { sp = op; });
  if (!sp) return success();  // no-op for modules without spatial placement

  // 2. Find code_region, place
  csl::CodeRegionOp region = nullptr;
  csl::PlaceOp place = nullptr;
  sp.walk([&](Operation *op) {
    if (auto r = dyn_cast<csl::CodeRegionOp>(op)) region = r;
    else if (auto p = dyn_cast<csl::PlaceOp>(op)) place = p;
  });
  if (!region || !place) {
    sp.emitError("LayoutEmitter: expected exactly one code_region and one place");
    return failure();
  }

  // 3. Resolve kernel source_file
  auto kernel = cast<csl::KernelOp>(place.getKernel().getDefiningOp());
  StringRef sourceFile = kernel.getSourceFile().value_or("pe.csl");
  int64_t W = region.getWidth();
  int64_t H = region.getHeight();
  int64_t px = place.getX();
  int64_t py = place.getY();

  // 4. Collect host-level exports
  func::FuncOp host = nullptr;
  module.walk([&](func::FuncOp f) {
    if (!host) host = f;  // milestone: single function
  });

  // 5. Open output file
  SmallString<256> path(outDir);
  llvm::sys::path::append(path, "layout.csl");
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) return failure();

  // 6. Emit
  os << "const memcpy = @import_module(\"<memcpy/get_params>\", .{ .width = "
     << W << ", .height = " << H << " });\n\n";
  os << "layout {\n";
  os << "  @set_rectangle(" << W << ", " << H << ");\n";
  os << "  @set_tile_code(" << px << ", " << py << ", \"" << sourceFile
     << "\", .{ .memcpy_params = memcpy.get_params(0) });\n\n";

  for (auto en : host.getOps<csl::ExportNameOp>()) {
    StringRef name = en.getSymName();
    Type t = en.getExportedType();
    auto dir = en.getDirection();
    if (auto memTy = dyn_cast<MemRefType>(t)) {
      StringRef ptr = cslPtrType(memTy.getElementType());
      bool writable = dir && *dir == "in";
      os << "  @export_name(\"" << name << "\", " << ptr << ", "
         << (writable ? "true" : "false") << ");\n";
    } else if (isa<FunctionType>(t)) {
      os << "  @export_name(\"" << name << "\", fn()void);\n";
    }
  }
  os << "}\n";
  return success();
}

StringRef LayoutEmitter::cslPtrType(Type elt) {
  if (elt.isF32()) return "[*]f32";
  if (elt.isF16()) return "[*]f16";
  if (elt.isInteger(32)) return "[*]i32";
  if (elt.isInteger(16)) return "[*]i16";
  return "[*]f32";  // fallback
}
```

The translator's top-level `translate()` method gains a LayoutEmitter pass before the KernelEmitter pass.

- [ ] Implement, build, run lit, commit.

---

## Phase 5 — KernelEmitter (rework)

**Goal:** Emit `pe_program.csl` from `csl.kernel` with the memcpy-specific patterns.

### Target output

Matching `mlir/test/Conversion/AIRToCSL/golden/vecadd_pe.csl.golden`:

```csl
param memcpy_params: comptime_struct;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

const N: i32 = 256;

var a_buf: [N]f32;
var b_buf: [N]f32;
var c_buf: [N]f32;

var a_ptr: [*]f32 = &a_buf;
var b_ptr: [*]f32 = &b_buf;
const c_ptr: [*]f32 = &c_buf;

fn compute() void {
  for (@range(i32, N)) |i| {
    c_buf[i] = a_buf[i] + b_buf[i];
  }
  sys_mod.unblock_cmd_stream();
}

comptime {
  @export_symbol(a_ptr, "a");
  @export_symbol(b_ptr, "b");
  @export_symbol(c_ptr, "c");
  @export_symbol(compute);
}
```

### Responsibilities added from v1

1. Always emit the `param memcpy_params: comptime_struct;` header.
2. Always emit `const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);`.
3. For each `csl.var` that's referenced by a `csl.export_symbol`, emit a pointer alias:
   - If the corresponding host-level `csl.export_name` has `direction = "in"` → `var <name>_ptr: [*]<T> = &<name>;` (writable from host)
   - If `direction = "out"` → `const <name>_ptr: [*]<T> = &<name>;` (read-only from host)
4. In each `csl.func` whose name matches a function-typed `csl.export_name`, append `sys_mod.unblock_cmd_stream();` before `csl.return`.
5. The `@export_symbol(...)` calls target the **pointer alias** (e.g. `a_ptr`), not the buffer. Look up the mapping by the `csl.export_symbol`'s alias attr (e.g. alias "a" → `a_ptr`) or synthesize the name (`<sym>_ptr`).
6. Wrap the `@export_symbol` calls in a `comptime { }` block at the end of the file.

### Task 5.1 — Failing lit test

**File:** `mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir`

Similar to v1 Task 4.1 but with CHECK lines covering the memcpy header, pointer aliases, and `sys_mod.unblock_cmd_stream()`:

```mlir
// RUN: air-opt %s | air-translate --emit-csl-rt -o %t/
// RUN: cat %t/vecadd_pe.csl | FileCheck %s

// CHECK: param memcpy_params: comptime_struct
// CHECK: const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params)
// CHECK: const N: i32 = 256
// CHECK: var a_buf: [N]f32
// CHECK: var b_buf: [N]f32
// CHECK: var c_buf: [N]f32
// CHECK: var a_ptr: [*]f32 = &a_buf
// CHECK: var b_ptr: [*]f32 = &b_buf
// CHECK: const c_ptr: [*]f32 = &c_buf
// CHECK: fn compute() void {
// CHECK:   for (@range(i32, 256)) |{{.*}}| {
// CHECK:     c_buf[{{.*}}] = a_buf[{{.*}}] + b_buf[{{.*}}];
// CHECK:   }
// CHECK:   sys_mod.unblock_cmd_stream();
// CHECK: }
// CHECK: comptime {
// CHECK-DAG: @export_symbol(a_ptr, "a");
// CHECK-DAG: @export_symbol(b_ptr, "b");
// CHECK-DAG: @export_symbol(c_ptr, "c");
// CHECK-DAG: @export_symbol(compute);
// CHECK: }

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    csl.spatial_placement {
      %k = csl.kernel {
        csl.var @a_buf : memref<256xf32>
        csl.var @b_buf : memref<256xf32>
        csl.var @c_buf : memref<256xf32>
        csl.func @compute {
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c1 = arith.constant 1 : index
          scf.for %i = %c0 to %c256 step %c1 {
            %va = memref.load %a_buf[%i] : memref<256xf32>
            %vb = memref.load %b_buf[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %c_buf[%i] : memref<256xf32>
          }
          csl.return
        }
        csl.export_symbol @a_buf alias("a")
        csl.export_symbol @b_buf alias("b")
        csl.export_symbol @c_buf alias("c")
        csl.export_symbol @compute
      } {source_file = "vecadd_pe.csl"} : !csl.kernel
      %r = csl.code_region routes() colors() {
      } {width = 1 : i64, height = 1 : i64} : !csl.code_region
      csl.place %r %k {x = 0 : i64, y = 0 : i64}
    }
    csl.export_name "a" : memref<256xf32>{direction = "in"}
    csl.export_name "b" : memref<256xf32>{direction = "in"}
    csl.export_name "c" : memref<256xf32>{direction = "out"}
    csl.export_name "compute" : () -> ()
    return
  }
}
```

### Task 5.2 — Implement KernelEmitter

Same 7-op dispatch table as v1 Task 4.2 (`csl.var`, `csl.func`, `csl.return`, `scf.for`, `memref.load`, `memref.store`, `arith.addf`, `arith.constant`), plus the five memcpy-specific emissions listed above.

The pointer-alias logic: walk the kernel body once and build a map `varName → (ptrName, isConst, cslType)` driven by which vars appear in a `csl.export_symbol`. The `isConst` bit comes from looking up the matching host-level `csl.export_name`'s direction — which means KernelEmitter needs access to the enclosing module, not just the `csl.kernel` op. Pass `ModuleOp` through the emitter's constructor.

The `sys_mod.unblock_cmd_stream()` logic: for each `csl.func`, check whether its `sym_name` matches any function-typed host-level `csl.export_name`. If yes, append the unblock call before `csl.return`.

- [ ] Implement (a substantial chunk of C++ — ~400 LOC), build, run lit, commit. Diff the output against `vecadd_pe.csl.golden` to confirm match.

### Task 5.3 — KernelEmitter precondition rejection test

Unchanged from v1 Task 4.3. One file pinning the "scf.for requires lo=0, step=1, constant hi" precondition violation.

---

## Phase 6 — HostEmitter (rework)

**Goal:** Emit a simpler `run.py` that opens a pre-compiled directory via `SdkRuntime(args.name, cmaddr=...)` and walks the `csl_rt.memcpy_h2d`/`launch`/`memcpy_d2h` ops to produce the runtime sequence.

### Target output

Matching `mlir/test/Conversion/AIRToCSL/golden/run.py.golden` — the full script is ~75 lines, already committed and verified on CS-3. Key structural pieces:

```python
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

parser = argparse.ArgumentParser()
parser.add_argument("--name")
parser.add_argument("--cmaddr", default=None)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()

a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32) * 2.0
c = np.zeros(N, dtype=np.float32)
expected = a + b

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
id_a = runner.get_id("a")
id_b = runner.get_id("b")
id_c = runner.get_id("c")
runner.load()
runner.run()

runner.memcpy_h2d(id_a, a, 0, 0, 1, 1, N, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(id_b, b, 0, 0, 1, 1, N, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.launch("compute", nonblock=False)
runner.memcpy_d2h(c, id_c, 0, 0, 1, 1, N, streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.stop()

if args.check:
    ...  # np.array_equal, print PASS/FAIL
```

### Task 6.1 — Failing lit test

**File:** `mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir`

Similar to v1 Task 5.1 but CHECK lines updated to match the golden run.py (no `build_layout`, no `SdkLayout` import, `SdkRuntime(args.name, cmaddr=args.cmaddr)` call, `memcpy_h2d`/`memcpy_d2h` with the full kwargs).

### Task 6.2 — Implement HostEmitter

The existing `CSLRuntimeToPy.cpp` has a working `emitLayoutOperations` (lines 201–257) that walks `csl_rt.create_code_region` / `csl_rt.place` / `csl_rt.set_param_all` / `csl_rt.export_name`. **Delete this** — none of those ops are generated by the new csl-to-csl-rt pass.

The existing `emitRuntimeOperations` (lines 259–355) walks `csl_rt.memcpy_h2d` / `csl_rt.launch` / `csl_rt.memcpy_d2h` / `csl_rt.load` / `csl_rt.get_id`. **Keep and rework**: the walk logic stays the same, but the output format changes. Replace the current `build_layout(platform)` wrapper with direct `SdkRuntime(args.name, cmaddr=args.cmaddr)` construction + explicit `get_id` calls derived from the seen memcpy/launch ops.

Add buffer materialization by inspecting `csl_rt.memcpy_h2d` srcName and elemPerPe attrs:
- First input (index 0): `name = np.arange(N, dtype=np.float32)`
- Second input: `name = np.arange(N, dtype=np.float32) * 2.0`
- Any output: `name = np.zeros(N, dtype=np.float32)`
- `expected = <first_in> + <second_in>`

The `--check` block is hardcoded boilerplate: `np.array_equal(<out>, expected)`, print PASS/FAIL, exit 0/1.

- [ ] Implement, build, run lit, commit. Diff output against `run.py.golden`.

---

## Phase 7 — Hardware integration test

### Task 7.1 — Write the e2e test

**File:** `test/csl/test_vecadd_e2e.py`

```python
"""End-to-end test: AIR → CSL → CS-3. The milestone's definition of done.

Runs the full pipeline and asserts the validator prints PASS on CS-3.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir"


def test_vecadd_end_to_end(tmp_path):
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir()

    # 1. air-opt | air-translate → layout.csl + pe_program.csl + run.py
    subprocess.run(
        f"air-opt {SRC} -air-to-csl-dialect -csl-to-csl-rt | "
        f"air-translate --emit-csl-rt -o {out_dir}",
        shell=True, check=True,
    )
    assert (out_dir / "vecadd_layout.csl").exists() or (out_dir / "layout.csl").exists()
    assert (out_dir / "vecadd_pe.csl").exists()
    assert (out_dir / "run.py").exists()

    # 2. cslc --memcpy → out/
    compiled = out_dir / "out"
    subprocess.run(
        ["cslc", "--arch=wse3",
         str(out_dir / "layout.csl"),  # or vecadd_layout.csl depending on LayoutEmitter filename
         "--fabric-dims=8,3", "--fabric-offsets=4,1",
         "-o", str(compiled),
         "--memcpy", "--channels", "1"],
        check=True, capture_output=True, text=True,
    )
    assert compiled.exists()

    # 3. cs_python run.py → PASS
    result = subprocess.run(
        ["cs_python", str(out_dir / "run.py"),
         "--name", str(compiled),
         "--check"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"run.py failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout
```

- [ ] Write, run (`pytest test/csl/test_vecadd_e2e.py -v`), iterate until PASS. **Do not commit** until this passes. It's the gate.

---

## Phase 8 — Quarantine Phase-1 emitter

Unchanged from v1 Phase 7. Remove `-air-to-csl=...` pass registration from `Passes.td`, remove the .cpp source from `mlir/lib/Conversion/CMakeLists.txt`, delete the old `mlir/test/Conversion/AIRToCSL/{basic,gemv}.mlir` fixtures, move the .cpp to `archived_code/`. Run full test suite + hardware test to confirm no regression. Commit.

---

## What didn't change from v1

- The `file structure` (what files exist, their roles) — still a new `air-to-csl-dialect` pass + extended `csl-to-csl-rt` + reworked `CSLRuntimeToPy.cpp`.
- The architectural commitments (§5.2 of the spec): every arrow is a real MLIR/translation step, csl.kernel body holds standard MLIR ops, fail loud no half-outputs.
- The rejection test strategy (§8.4): one reject_*.mlir per failure mode in §7.
- The "definition of done": single test runs the full pipeline on CS-3 and prints PASS.
- The execution handoff: subagent-driven development, one fresh subagent per task.

## Open questions surfaced during v2

1. **LayoutEmitter output filename**: `layout.csl` or `vecadd_layout.csl`? The golden uses `vecadd_layout.csl.golden` but the SDK doesn't care about the name (it's referenced by path in `cslc`). Recommend `layout.csl` (simple, matches convention) and update the Phase 7 test to match.
2. **`csl.code_region` fabric dims**: the `cslc --fabric-dims=8,3 --fabric-offsets=4,1` values are hardcoded in the golden build script. For the milestone we can hardcode them in the translator or the integration test. Future milestones need them as attributes on `csl.spatial_placement`.
3. **`runtime.run()` vs `runtime.load()`**: the golden run.py calls both. The HostEmitter should emit both unconditionally (they're not driven by any csl_rt op).
4. **First input is `arange`, second is `arange * 2.0`**: this is deterministic test data. Future milestones with different kernels need different seeds, but for vecadd it's fine and guaranteed to produce a non-trivial output.

These are notes for the implementer; none block starting Phase 1.
