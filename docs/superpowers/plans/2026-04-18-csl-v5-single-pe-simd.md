# CSL v5 — Single-PE Complete Coverage + N-PE SIMD Implementation Plan

> **For agentic workers:** Execute via **`superpowers:executing-plans`** — step-by-step, TDD, one commit per step. Checkbox (`- [ ]`) syntax tracks progress.

**Goal:** Extend the shipped v4 subgrid stack so a single PE can express the compute surface used by SIMD-friendly scientific kernels (mem DSDs, vector builtins via a generic op, conditionals, generic module imports), and N-PE placements work **SIMD-only** — same program on every PE, sharded data, **no inter-PE communication**.

**Architecture:** Pure additions inside the existing `csl` dialect + `CSLEmit/` family. We uncomment the one DSD op the dialect has been reserving (`csl.get_mem_dsd`, `mem1d` only), add **one generic `csl.builtin_call` op** that covers every CSL language builtin and imported-module member call in a single form, and teach the emitter to emit it along with `scf.if` and user-authored `csl.import_module`. The N-PE SIMD story is already unlocked by v4 `csl_layout.place over [lo:hi, lo:hi]` + equal-sharding memcpy — v5 just proves it composes with per-PE DSD compute (tests only, no emitter code).

**Tech stack:** C++17, MLIR 18, TableGen, LLVM `lit` + `FileCheck`, Cerebras SDK (CS-3 simulator only).

**Scope boundary (locked in after review 2026-04-18):**

| In scope | Out of scope |
|---|---|
| Memory DSDs: `mem1d_dsd` | Fabric DSDs (`fabin_dsd`, `fabout_dsd`) |
| **One** generic `csl.builtin_call` op | Per-builtin ops (`csl.fmacs`/`fadds`/`fmovs`/…) — CSL has hundreds; one op covers all |
| Generic user `csl.import_module` (any path) | Hardcoded module list in emitter |
| `scf.if` + `arith.cmpf` / `cmpi` in kernel scope | `scf.while` (no example needs it) |
| `f32`, `i32` end-to-end | `f16` — deferred to later one-shot patch |
| N-PE SIMD: same program, sharded data | Any inter-PE communication (colors, routes, tasks, wavelets, `<collectives_2d>`) |

**Explicitly deferred to a later patch (do NOT implement in v5):**

- **`f16` / `i16` end-to-end.** The compute pipeline changes per type width; handled as a single follow-up patch over all ops at once.
- **`math.sqrt` / `math.exp` / `math.log`.** Would require an imported-module member call. The generic `csl.builtin_call` op added in Task 2 is **designed to already support this form** (`%mod::"sqrt"(%x)`) so the future patch is tests + emitter tweak only, no op changes. For v5's corpus, `norm` computes `sum-of-squares` (not `sqrt`).
- **`func.func {csl.extern}` attribute-based external member calls.** Not needed — the generic `csl.builtin_call` replaces it.

**Why this shape:** Per the SDK-docs graph queries, SIMD-friendly scientific kernels (saxpy, dot, 1-D stencil-no-halo, elementwise transforms, mandelbrot-per-pixel) are reachable with the scope above. Colors/tasks/fabric unlock halo stencils/GEMM-collectives and are ~3 weeks of work the user has explicitly excluded for this phase.

**SDK tutorials reachable with v5 scope (cross-check emitted CSL against these):**

| Tutorial | Purpose | Maps to v5 task |
|---|---|---|
| GEMV 02 (Memory DSDs) | Canonical `@get_dsd(mem1d_dsd, …)` + `@fmacs` + `@fadds` shape | Tasks 3, 4, 7 (saxpy/dot) |
| GEMV 03 (Memcpy) | H2D/D2H memcpy flow | Already covered by v4 |
| GEMV 04 (Params) | compile-time params | Already covered by v4 |
| GEMV 05 (Multiple PEs) | SIMD-style shard, **no routes** | Task 8 (N-PE SIMD) |
| Benchmark: Mandelbrot (per-pixel variant) | Per-PE independent compute | Future Task 7 addition if desired |

**Not reachable (fabric/routes/collectives):** GEMV 06-09, 25-pt stencil (time-marching), 7pt-stencil-spmv, power-method, PCG, gemv-collectives_2d, gemv-checkerboard.

**Versioning note:** v4 shipped on branch `air-to-fire` (11 tasks, subgrid grammar). This is **v5**, which appends to v4 — no v4 behavior changes.

---

## File Structure

**Modified:**

| File | What changes |
|---|---|
| `mlir/include/air/Dialect/CSL/CSLOps.td` | Uncomment `CSL_GetMemDsdOp` (mem1d only). Add **one** new op: `CSL_BuiltinCallOp`. Remove the "Deferred" comment. |
| `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` | Only if a trivial verifier is needed (ordinarily tablegen-driven). |
| `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` | Extend **`emitTileCodeBody()`** (the kernel-body per-op dispatch) to handle `csl.get_mem_dsd`, `csl.builtin_call`, `scf.if`, `arith.cmpf`, `arith.cmpi`. `CSLEmitCommon.h` is just a namespace helper — no changes there. |
| `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp` | In `ProgramEmitter::emit()`, walk user `csl.import_module` ops at program scope and emit each as `const <ssa-name> = @import_module("<path>"[, <params>]);` — in addition to the two hardcoded imports (memcpy, layout). |
| `docs/more_docs/csl_dialect_reference.md` | DSD ops no longer "deferred" (mem1d only); document `csl.builtin_call`; document scope boundary. |
| `utils/run_csl_sdk.sh` | Include the new scientific + SIMD e2e test subdirs in the batch run. |

**Created:**

| File | Purpose |
|---|---|
| `mlir/test/Dialect/CSL/dsd_roundtrip.mlir` | Round-trip: `csl.get_mem_dsd` parse → print → parse. |
| `mlir/test/Dialect/CSL/builtin_call_roundtrip.mlir` | Round-trip: `csl.builtin_call` bare form and `%mod::"name"` form. |
| `mlir/test/Targets/CSLEmit/e2e/dsds.mlir` | End-to-end: saxpy via `@get_dsd` + `@fmacs` + `@fadds`. |
| `mlir/test/Targets/CSLEmit/e2e/control_flow_if.mlir` | `scf.if` + `arith.cmpf` → CSL `if/else`. |
| `mlir/test/Targets/CSLEmit/e2e/imports.mlir` | User-level `csl.import_module` → `@import_module(...)` at program top. |
| `mlir/test/Targets/CSLEmit/e2e/scientific/saxpy.mlir` | Saxpy (`y = a*x + y`) via DSDs. |
| `mlir/test/Targets/CSLEmit/e2e/scientific/dot.mlir` | Dot product via DSDs + scalar reduction. |
| `mlir/test/Targets/CSLEmit/e2e/scientific/stencil_1d.mlir` | 1-D 3-point stencil with boundary `scf.if` (no halo). |
| `mlir/test/Targets/CSLEmit/e2e/scientific/norm_sq.mlir` | Norm-squared (dot-with-self, no sqrt). |
| `mlir/test/Targets/CSLEmit/e2e/simd_dsd.mlir` | N-PE SIMD subgrid (4×2) with per-PE DSD saxpy — proves v4 subgrid + v5 DSDs compose. |

**Not touched:** `CSLLayoutOps.td`, `CSLHostOps.td`, `CSLVerifyParams.cpp`, `CSLInferExports.cpp`, `AIRToCSLPass.cpp`, layout/host emitters. All v4 guarantees hold.

---

## How to build and test

From the repo root after activating the sandbox (per `CLAUDE.md`):

```bash
cd build && ninja install                        # rebuild after any .td/.cpp change
lit mlir/test/Dialect/CSL/dsd_roundtrip.mlir -v  # single-file dev loop
```

**Correct build/check targets (verified via `ninja -t targets all`):**

| Target | Scope | Use when |
|---|---|---|
| `ninja check-airmlir-dialect-csl` | CSL dialect round-trip tests | fastest inner loop while editing ops |
| `ninja check-airmlir-conversion-airtocsl` | `-air-to-csl` conversion | AIR→CSL changes |
| `ninja check-airmlir-targets-cslemit` | CSL emitter + e2e | emitter changes |
| `ninja check-airmlir-targets-cslemit-e2e` | Only the e2e corpus | final SIMD/scientific validation |
| `ninja check-air-mlir` | **Full** MLIR test suite | end of a task, before committing |

> **Note:** `mlir-air/CLAUDE.md` currently says `ninja check-airmlir` in the testing section; that target does not exist — the real alias is **`check-air-mlir`** (with the dash). Flag in Task 10 docs update.

**Commit style:** small, conventional, one step at a time. Example: `feat(csl): add csl.builtin_call op (v5 task 2)`. Each numbered step below maps 1:1 to a commit.

---

## Task 1 — Uncomment & refine `csl.get_mem_dsd` (mem1d only)

**Context:** `CSLOps.td` has `CSL_GetMemDsdOp` in a commented "deferred" block. `!csl.dsd` and `CSL_DsdKindEnum` (mem1d/mem2d/fabin/fabout) already exist in `CSLBase.td`. Uncomment, restrict to `mem1d` for v5, expose `length` as an SSA `index`.

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td` (replace the commented block)
- Create: `mlir/test/Dialect/CSL/dsd_roundtrip.mlir`

- [ ] **Step 1.1 — Write the round-trip test first**

`mlir/test/Dialect/CSL/dsd_roundtrip.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @dsd_basic
module {
  csl.wafer @dsd_basic {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      csl.func @compute {
        %len = arith.constant 128 : index
        // CHECK: csl.get_mem_dsd
        // CHECK-SAME: memref<128xf32>
        // CHECK-SAME: !csl.dsd
        %d = csl.get_mem_dsd %a, %len : memref<128xf32>, index -> !csl.dsd
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
      csl_host.launch @layout::@compute
    }
  }
}
```

- [ ] **Step 1.2 — Confirm failure** `lit mlir/test/Dialect/CSL/dsd_roundtrip.mlir -v` — expected: `error: custom op 'csl.get_mem_dsd' is unknown`.

- [ ] **Step 1.3 — Uncomment and restrict to mem1d** in `CSLOps.td`:

```tablegen
//===----------------------------------------------------------------------===//
// csl.get_mem_dsd — 1-D memory-backed Data Structure Descriptor
//===----------------------------------------------------------------------===//

def CSL_GetMemDsdOp : CSL_Op<"get_mem_dsd", [Pure]> {
  let summary = "Create a mem1d DSD over a contiguous PE-local buffer";
  let description = [{
    Creates a DSD referencing a contiguous 1-D slice of PE-local memory.
    Maps to `@get_dsd(mem1d_dsd, .{ .base_address = &a, .extent = N })` in CSL.

    v5 supports only `mem1d_dsd`. `mem2d_dsd`, `fabin_dsd`, `fabout_dsd` are
    deferred — the CSL fabric DSDs require colors/routes/tasks which are
    intentionally excluded from SIMD-only execution.

    ```mlir
    %a = csl.var @a : memref<128xf32>
    %n = arith.constant 128 : index
    %d = csl.get_mem_dsd %a, %n : memref<128xf32>, index -> !csl.dsd
    ```
  }];

  let arguments = (ins AnyMemRef:$buffer, Index:$length);
  let results   = (outs CSL_DsdType:$result);

  let assemblyFormat = [{
    $buffer `,` $length `:` type($buffer) `,` type($length) `->` type($result) attr-dict
  }];
}
```

Remove the `// TODO: Deferred` comment block around the op.

- [ ] **Step 1.4 — Rebuild** `cd build && ninja install` — expect clean build.

- [ ] **Step 1.5 — Re-run test** `lit mlir/test/Dialect/CSL/dsd_roundtrip.mlir -v` — expect PASS.

- [ ] **Step 1.6 — Run narrow suite** `cd build && ninja check-airmlir-dialect-csl` — all green.

- [ ] **Step 1.7 — Commit** `feat(csl): uncomment csl.get_mem_dsd for mem1d (v5 task 1)`.

---

## Task 2 — Add one generic `csl.builtin_call` op

**Context:** CSL has hundreds of DSD/language builtins (`@fmacs`, `@fadds`, `@fmovs`, `@fsubs`, `@fmaxs`, `@fmins`, `@fnegs`, `@iadds`, `@imuls`, `@fabs`, `@sqrts`, …) plus every member of every imported module (`math.sqrt`, `math.exp`, `layout.get_x_coord`, …). A per-builtin op would explode the dialect. Instead: **one generic op** whose mnemonic string picks the callee; optional `!csl.imported_module` operand selects module-member form.

**Op form (both supported):**

```mlir
// Bare language builtin — emitted as `@fmacs(y, y, A, xi);`
csl.builtin_call "fmacs"(%y_dsd, %y_dsd, %A_dsd, %xi) : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()

// Imported-module member — emitted as `math.sqrt(x);`
// (Tested later when math-lib patch lands; op supports it today.)
%r = csl.builtin_call %math::"sqrt"(%x) : (f32) -> f32
```

**Why one op:** The emitter for this op is a one-liner (prepend `@` for bare form, prepend `<mod>.` for module form, comma-separate operands, done). The CSL compiler validates argument types; MLIR's job is to carry the callee name and operands through untyped. Zero per-builtin TableGen entries.

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td`
- Create: `mlir/test/Dialect/CSL/builtin_call_roundtrip.mlir`

- [ ] **Step 2.1 — Write round-trip test**

`mlir/test/Dialect/CSL/builtin_call_roundtrip.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @bc
module {
  csl.wafer @bc {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %n   = arith.constant 128 : index
        %i   = arith.constant 0   : index
        %A_d = csl.get_mem_dsd %A, %n : memref<128xf32>, index -> !csl.dsd
        %y_d = csl.get_mem_dsd %y, %n : memref<128xf32>, index -> !csl.dsd
        %xi  = memref.load %x[%i] : memref<128xf32>
        // CHECK: csl.builtin_call "fmacs"
        // CHECK-SAME: (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        csl.builtin_call "fmacs"(%y_d, %y_d, %A_d, %xi)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        // Module form — round-trips today even without emitter support
        %m = csl.import_module "<math>" : !csl.imported_module
        // CHECK: csl.builtin_call %{{.*}}::"sqrt"
        %r = csl.builtin_call %m::"sqrt"(%xi) : (f32) -> f32
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
      csl_host.launch @layout::@compute
    }
  }
}
```

- [ ] **Step 2.2 — Confirm failure** — expected: `error: custom op 'csl.builtin_call' is unknown`.

- [ ] **Step 2.3 — Add the op** in `CSLOps.td` (add after `CSL_GetMemDsdOp`):

```tablegen
//===----------------------------------------------------------------------===//
// csl.builtin_call — generic CSL builtin / imported-module member call
//===----------------------------------------------------------------------===//

def CSL_BuiltinCallOp : CSL_Op<"builtin_call"> {
  let summary = "Call a CSL language builtin or imported-module member";
  let description = [{
    Generic, untyped call to either:

    - A bare CSL language builtin (e.g. `@fmacs`, `@fadds`, `@fmovs`, `@fabs`,
      `@iadds`). Emitted as `@<callee>(arg0, arg1, …);`.

    - A member of a module imported via `csl.import_module`. The module operand
      is the SSA value of type `!csl.imported_module`. Emitted as
      `<mod>.<callee>(arg0, arg1, …);`.

    Operand and result typing is intentionally permissive — the CSL compiler
    owns semantic validation. One op covers every DSD builtin and every
    imported-module member call, keeping the dialect size constant.

    ```mlir
    // bare-builtin form
    csl.builtin_call "fmacs"(%y, %y, %A, %xi)
        : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()

    // module-member form
    %r = csl.builtin_call %math::"sqrt"(%x) : (f32) -> f32
    ```
  }];

  let arguments = (ins
    StrAttr:$callee,
    Optional<CSL_ImportedModuleType>:$module,
    Variadic<AnyType>:$operands);
  let results   = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    ($module^ `::`)? $callee `(` $operands `)` attr-dict
      `:` functional-type($operands, $results)
  }];
}
```

- [ ] **Step 2.4 — Rebuild** `cd build && ninja install`.

- [ ] **Step 2.5 — Re-run round-trip** — expect PASS on both bare and `%mod::"name"` forms.

- [ ] **Step 2.6 — Run narrow suite** `ninja check-airmlir-dialect-csl`.

- [ ] **Step 2.7 — Commit** `feat(csl): add generic csl.builtin_call op (v5 task 2)`.

---

## Task 3 — Emit `csl.get_mem_dsd` → `@get_dsd(mem1d_dsd, …)`

**Context:** `emitTileCodeBody()` in `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` is the op-by-op dispatch inside kernel (`csl.func` / `func.func`) bodies. It already handles `arith.*`, `memref.load/store`, `scf.for`, `func.call/return`. Add a case for `csl.get_mem_dsd` that emits `const <ssa> = @get_dsd(mem1d_dsd, .{ .base_address = &<memref>, .extent = <len> });`.

**Buffer naming:** `csl.var @a : memref<...>` is already lowered to `var a : ...` at program scope. Inside kernel bodies, a memref SSA value referring to `@a` emits as bare name `a`. Reuse whatever name-resolver `memref.load` / `memref.store` already uses in the existing `emitTileCodeBody()` cases — grep for how those cases look up the buffer name and reuse the same helper.

**Length:** the `index` operand is either a constant (emit as integer literal) or an SSA index (emit as the SSA's CSL name, e.g. `len_0`). Reuse the existing integer-SSA formatter.

**Files:** `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` — add new cases next to the existing arith/memref/scf.for dispatch inside `emitTileCodeBody()`. Create `mlir/test/Targets/CSLEmit/e2e/dsds.mlir`.

- [ ] **Step 3.1 — Write the e2e FileCheck test first**

`mlir/test/Targets/CSLEmit/e2e/dsds.mlir`:

```mlir
// RUN: air-translate --emit-csl --output-dir=%t %s && FileCheck %s --input-file=%t/saxpy_dsd/pe_program.csl

module {
  csl.wafer @saxpy_dsd {arch = "wse3"} {
    csl.program @pe {
      csl.var @A : memref<128xf32>
      csl.var @y : memref<128xf32>
      // CHECK: var A : [128]f32;
      // CHECK: var y : [128]f32;
      csl.func @compute {
        %n = arith.constant 128 : index
        // CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &A, .extent = 128 });
        %Ad = csl.get_mem_dsd %A, %n : memref<128xf32>, index -> !csl.dsd
        // CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 128 });
        %yd = csl.get_mem_dsd %y, %n : memref<128xf32>, index -> !csl.dsd
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
      csl_host.launch @layout::@compute
    }
  }
}
```

- [ ] **Step 3.2 — Confirm failure** — expect either translate-crash or missing CHECK line.

- [ ] **Step 3.3 — Implement the emitter case** in `emitTileCodeBody`:

```cpp
// csl.get_mem_dsd %buf, %len : ... -> !csl.dsd
if (auto dsd = dyn_cast<csl::GetMemDsdOp>(&op)) {
  StringRef bufName  = getValueName(dsd.getBuffer()); // existing helper
  std::string lenStr = formatIndex(dsd.getLength());  // literal or ssa name
  os << "const " << getValueName(dsd.getResult())
     << " = @get_dsd(mem1d_dsd, .{ .base_address = &" << bufName
     << ", .extent = " << lenStr << " });\n";
  continue;
}
```

(Exact helper names may differ — reuse what `memref.load` already uses for the base-address reference and what `scf.for` uses for index operands.)

- [ ] **Step 3.4 — Rebuild** `cd build && ninja install`.

- [ ] **Step 3.5 — Re-run test** — expect PASS.

- [ ] **Step 3.6 — Run CSL emit suite** `ninja check-airmlir-targets-cslemit`.

- [ ] **Step 3.7 — Commit** `feat(csl-emit): emit @get_dsd(mem1d_dsd, …) (v5 task 3)`.

---

## Task 4 — Emit `csl.builtin_call` → `@name(…)` / `mod.name(…)`

**Context:** One emitter case covers every bare builtin (`@fmacs`, `@fadds`, `@fmovs`, `@fabs`, …) and every module-member call (`math.sqrt`, etc.). Logic is trivial: prefix with `@` or `<mod>.`, comma-separate operands, handle the zero-result vs one-result case for result binding.

**Files:** `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h`. Extend `mlir/test/Targets/CSLEmit/e2e/dsds.mlir` (created in Task 3) with `fmacs`/`fadds` check lines, OR add a small dedicated test — either is fine; the scientific corpus in Task 7 exercises the real paths.

- [ ] **Step 4.1 — Extend the e2e test** — append to `dsds.mlir`:

```mlir
// (inside @compute, after %Ad and %yd)
// CHECK: @fmacs({{.*}}, {{.*}}, {{.*}}, {{.*}});
%i  = arith.constant 0 : index
%xi = arith.constant 2.0 : f32
csl.builtin_call "fmacs"(%yd, %yd, %Ad, %xi)
    : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()

// CHECK: @fadds({{.*}}, {{.*}}, {{.*}});
csl.builtin_call "fadds"(%yd, %yd, %Ad)
    : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
```

- [ ] **Step 4.2 — Confirm failure** — run the test, expect missing `@fmacs` line.

- [ ] **Step 4.3 — Implement the emitter case** in `emitTileCodeBody`:

```cpp
// csl.builtin_call [%mod::] "name"(args…) : (T…) -> (R…)
if (auto bc = dyn_cast<csl::BuiltinCallOp>(&op)) {
  // Result binding: 0 results → bare stmt; 1 result → `const r = …;`.
  if (bc.getNumResults() == 1) {
    os << "const " << getValueName(bc.getResult(0)) << " = ";
  } else if (bc.getNumResults() > 1) {
    op.emitError("csl.builtin_call: multi-result not supported in v5");
    return failure();
  }

  if (Value mod = bc.getModule()) {
    os << getValueName(mod) << "." << bc.getCallee();
  } else {
    os << "@" << bc.getCallee();
  }

  os << "(";
  llvm::interleaveComma(bc.getOperands(), os,
                        [&](Value v) { os << getValueName(v); });
  os << ");\n";
  continue;
}
```

- [ ] **Step 4.4 — Rebuild** `cd build && ninja install`.

- [ ] **Step 4.5 — Re-run test** — expect PASS.

- [ ] **Step 4.6 — Run narrow suite** `ninja check-airmlir-targets-cslemit`.

- [ ] **Step 4.7 — Commit** `feat(csl-emit): emit csl.builtin_call bare + module forms (v5 task 4)`.

---

## Task 5 — Emit `scf.if` + `arith.cmpf` / `arith.cmpi` in kernel scope

**Context:** Scientific kernels need boundary checks (`if (i == 0 || i == N-1) …`) and conditional updates. Current emitter has `scf.for` but no `scf.if`, and no comparison ops. Restrict v5 to `scf.if` **without yielded values** (no `scf.yield %x, %y`) — the CSL target doesn't need the SSA-merge form for the scientific corpus, and supporting it later is additive.

**Emission pattern:**

```
if (<cond>) {
  <then-body>
} else {
  <else-body>
}
```

`arith.cmpf OEQ/OLT/OLE/OGT/OGE/ONE` → `==/</<=/>/>=/!=` on floats; `arith.cmpi EQ/SLT/SLE/SGT/SGE/NE/ULT/ULE/UGT/UGE` → same on ints. Unordered predicates (`UNO`, `UEQ`, etc.) reject with a clear diagnostic — they're not needed for the corpus.

**Files:** `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h`. Create `mlir/test/Targets/CSLEmit/e2e/control_flow_if.mlir`.

- [ ] **Step 5.1 — Write failing e2e test** `mlir/test/Targets/CSLEmit/e2e/control_flow_if.mlir`:

```mlir
// RUN: air-translate --emit-csl --output-dir=%t %s && FileCheck %s --input-file=%t/cond/pe_program.csl

module {
  csl.wafer @cond {arch = "wse3"} {
    csl.program @pe {
      csl.var @x : memref<8xf32>
      csl.func @compute {
        %c0  = arith.constant 0   : index
        %c1  = arith.constant 1   : index
        %c8  = arith.constant 8   : index
        %zero = arith.constant 0.0 : f32
        scf.for %i = %c0 to %c8 step %c1 {
          %v = memref.load %x[%i] : memref<8xf32>
          // CHECK: if (
          // CHECK-SAME: <
          %lt = arith.cmpf olt, %v, %zero : f32
          scf.if %lt {
            // CHECK: x[{{.*}}] = 0.0;
            memref.store %zero, %x[%i] : memref<8xf32>
          }
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
      csl_host.launch @layout::@compute
    }
  }
}
```

- [ ] **Step 5.2 — Confirm failure**.

- [ ] **Step 5.3 — Implement cmpf/cmpi → boolean SSA**

In `emitTileCodeBody`, bind the comparison result as a named CSL `bool`/`i1`:

```cpp
if (auto cmp = dyn_cast<arith::CmpFOp>(&op)) {
  StringRef opTxt = translateCmpFPred(cmp.getPredicate()); // "==", "<", etc.
  // Reject unordered predicates with op.emitError.
  os << "const " << getValueName(cmp.getResult()) << " = "
     << getValueName(cmp.getLhs()) << " " << opTxt << " "
     << getValueName(cmp.getRhs()) << ";\n";
  continue;
}
// analogous arith::CmpIOp case
```

- [ ] **Step 5.4 — Implement `scf.if` (no-yield form)**

```cpp
if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
  if (ifOp.getNumResults() != 0) {
    return op.emitError("scf.if with yielded values is unsupported in v5");
  }
  os << "if (" << getValueName(ifOp.getCondition()) << ") {\n";
  if (failed(emitTileCodeBody(ifOp.getThenRegion().front(), os))) return failure();
  os << "}";
  if (!ifOp.getElseRegion().empty()) {
    os << " else {\n";
    if (failed(emitTileCodeBody(ifOp.getElseRegion().front(), os))) return failure();
    os << "}";
  }
  os << "\n";
  continue;
}
```

(If `emitTileCodeBody` isn't currently recursive over nested regions, lift the block-walk into a small helper — keep it minimal.)

- [ ] **Step 5.5 — Rebuild + re-run test** — expect PASS.

- [ ] **Step 5.6 — Run CSL emit suite** `ninja check-airmlir-targets-cslemit`.

- [ ] **Step 5.7 — Commit** `feat(csl-emit): scf.if + arith.cmpf/cmpi in kernel scope (v5 task 5)`.

---

## Task 6 — Emit user-level `csl.import_module` generically

**Context:** `CSLProgramEmitter.cpp:66-70` currently hardcodes exactly two imports in every `program.csl`:

```
const sys_mod    = @import_module("<memcpy/memcpy>", memcpy_params);
const layout_mod = @import_module("<layout>");
```

The dialect has `csl.import_module "<path>" [{params = {...}}] : !csl.imported_module` that is valid inside `csl.program`/kernel scope today but is **never walked**. After this task, any user-written `csl.import_module` at program scope emits as `const <ssa-name> = @import_module("<path>"[, .{…params…}]);` in the program prolog — right after the two hardcoded lines, in IR order. `<math>`, `<memcpy/get_params>`, `<collectives_2d>` (when later unlocked), etc. all work without further dialect changes.

**Rule:** imports emit **once per `!csl.imported_module` SSA** at program top. Uses of the module (via `csl.builtin_call %mod::"member"`) reference the emitted name.

**Files:** `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp`. Create `mlir/test/Targets/CSLEmit/e2e/imports.mlir`.

- [ ] **Step 6.1 — Write failing test** `mlir/test/Targets/CSLEmit/e2e/imports.mlir`:

```mlir
// RUN: air-translate --emit-csl --output-dir=%t %s && FileCheck %s --input-file=%t/w/pe_program.csl

module {
  csl.wafer @w {arch = "wse3"} {
    csl.program @pe {
      // Existing hardcoded imports stay at the top.
      // CHECK: @import_module("<memcpy/memcpy>"
      // CHECK: @import_module("<layout>")
      // User-authored imports follow, in IR order.
      // CHECK: const {{.*}} = @import_module("<math>");
      %math = csl.import_module "<math>" : !csl.imported_module
      // CHECK: const {{.*}} = @import_module("<memcpy/get_params>", .{ .width = 4 : i32 });
      %gp = csl.import_module "<memcpy/get_params>" {params = {width = 4 : i32}}
                : !csl.imported_module
      csl.func @compute { csl.return }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
      csl_host.launch @layout::@compute
    }
  }
}
```

- [ ] **Step 6.2 — Confirm failure**.

- [ ] **Step 6.3 — Walk user imports in `CSLProgramEmitter.cpp`**

After the two hardcoded `@import_module` lines, add:

```cpp
// Emit user-authored `csl.import_module` ops at program scope, in IR order.
programOp.walk<WalkOrder::PreOrder>([&](csl::ImportModuleOp imp) {
  // Skip if nested inside a function/task body — those should stay inline
  // (or we can choose to hoist all; for v5, hoist only program-scope ones).
  if (imp->getParentOp() != programOp.getOperation()) return WalkResult::advance();
  os << "const " << getValueName(imp.getResult())
     << " = @import_module(\"" << imp.getPath() << "\"";
  if (auto params = imp.getParamsAttr()) {
    os << ", ";
    emitDictAttrAsCslStruct(params, os); // reuse whatever emitter uses for struct literals
  }
  os << ");\n";
  return WalkResult::advance();
});
os << "\n";
```

If no struct-literal helper exists yet, add a small one — this same formatter is reused by Task 4's `@get_dsd`.

- [ ] **Step 6.4 — Rebuild + re-run test** — expect PASS.

- [ ] **Step 6.5 — Run CSL emit suite** `ninja check-airmlir-targets-cslemit`.

- [ ] **Step 6.6 — Commit** `feat(csl-emit): emit user-level csl.import_module (v5 task 6)`.

---

## Task 7 — Scientific 1-PE corpus

**Context:** Prove the v5 stack on realistic-shape kernels. Four lit tests, each `.mlir` + expected CSL. All run through `air-translate --emit-csl` and FileCheck the produced `pe_program.csl`. No SDK runs in CI — those happen via `utils/run_csl_sdk.sh` (Task 9) outside CI on hardware-bearing hosts.

**Kernels:**

1. **saxpy** — `y = a*x + y` via `@fmacs` over mem1d DSDs.
2. **dot** — `d = sum_i x[i] * y[i]` via scalar `scf.for` + `arith.mulf`/`addf` (no DSD reduction builtin needed); write result to a `memref<1xf32>` that the host pulls via memcpy.
3. **stencil_1d** — `y[i] = 0.5*x[i-1] + x[i] + 0.5*x[i+1]` for `i` in `[1, N-1)`; boundary `scf.if` writes `x[i]` verbatim for endpoints. No halo, no inter-PE.
4. **norm_sq** — `s = sum_i x[i] * x[i]` (sum of squares). **No sqrt** — sqrt lands with the future math-lib patch; v5's corpus stops at `sum_sq`. Rename is intentional.

**Files:** `mlir/test/Targets/CSLEmit/e2e/scientific/{saxpy,dot,stencil_1d,norm_sq}.mlir`.

- [ ] **Step 7.1 — Write saxpy.mlir** (DSD form):

Key IR skeleton:
```mlir
csl.func @compute {
  %n  = arith.constant 128 : index
  %A  = csl.get_mem_dsd %A_buf, %n : memref<128xf32>, index -> !csl.dsd
  %Y  = csl.get_mem_dsd %y_buf, %n : memref<128xf32>, index -> !csl.dsd
  %a  = memref.load %alpha[%c0] : memref<1xf32>
  csl.builtin_call "fmacs"(%Y, %Y, %A, %a)
      : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
  csl.return
}
```
FileCheck: `@get_dsd(mem1d_dsd, …)` ×2, `@fmacs(y, y, A, alpha_0);`.

- [ ] **Step 7.2 — Write dot.mlir** — scalar `scf.for` accumulator, write result `memref<1xf32>` for host pull.

- [ ] **Step 7.3 — Write stencil_1d.mlir** — scalar loop with boundary `scf.if` (exercises Task 5).

- [ ] **Step 7.4 — Write norm_sq.mlir** — scalar reduction over `x[i]*x[i]` (no sqrt).

- [ ] **Step 7.5 — Run all four** `lit mlir/test/Targets/CSLEmit/e2e/scientific/ -v` — all PASS.

- [ ] **Step 7.6 — Commit** `test(csl-emit): scientific 1-PE corpus — saxpy, dot, stencil_1d, norm_sq (v5 task 7)`.

---

## Task 8 — N-PE SIMD e2e (test-only, proves composition)

**Context:** v4 already ships `csl_layout.place @pe over [0:W, 0:H]` with equal-sharding memcpy. v5 adds DSD compute inside `@pe`. Task 8 is a **pure test** demonstrating that a 4×2 subgrid with per-PE DSD saxpy works end-to-end through the emitter. **No emitter or dialect code change** — if this test passes cleanly on the first try, we've validated compositionality.

Scope reminder: same program on every PE, sharded input, no inter-PE comm. Subgrid is parameterized via the already-shipping `csl_layout.place over [lo:hi, lo:hi]` grammar.

**Files:** `mlir/test/Targets/CSLEmit/e2e/simd_dsd.mlir`.

- [ ] **Step 8.1 — Write simd_dsd.mlir**

```mlir
// RUN: air-translate --emit-csl --output-dir=%t %s && \
// RUN:   FileCheck %s --check-prefix=LAYOUT --input-file=%t/simd/layout.csl && \
// RUN:   FileCheck %s --check-prefix=PE     --input-file=%t/simd/pe_program.csl

module {
  csl.wafer @simd {arch = "wse3"} {
    csl.program @pe {
      csl.var @A : memref<32xf32>    // each PE owns its 32-elem shard
      csl.var @y : memref<32xf32>
      csl.var @alpha : memref<1xf32>
      csl.func @compute {
        %n  = arith.constant 32 : index
        %c0 = arith.constant  0 : index
        %Ad = csl.get_mem_dsd %A, %n : memref<32xf32>, index -> !csl.dsd
        %yd = csl.get_mem_dsd %y, %n : memref<32xf32>, index -> !csl.dsd
        %a  = memref.load %alpha[%c0] : memref<1xf32>
        // PE: @fmacs(y, y, A, alpha_0);
        csl.builtin_call "fmacs"(%yd, %yd, %Ad, %a)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        csl.return
      }
    }
    // LAYOUT: @set_tile_code
    // LAYOUT: for
    // LAYOUT-SAME: 0
    // LAYOUT-SAME: 4
    // LAYOUT-SAME: 2
    csl.layout {width = 4 : i64, height = 2 : i64} @layout {
      csl_layout.place @pe over [0:4, 0:2]
    }
    csl.host @main() {layout = @layout} {
      csl_host.launch @layout::@compute
    }
  }
}
```

- [ ] **Step 8.2 — Run it** `lit mlir/test/Targets/CSLEmit/e2e/simd_dsd.mlir -v`. If it fails, **stop and diagnose** — the failure is the bug (v4 subgrid and v5 DSD should compose without code change).

- [ ] **Step 8.3 — Commit** `test(csl-emit): N-PE SIMD DSD saxpy on 4x2 subgrid (v5 task 8)`.

---

## Task 9 — Run full suite green + update SDK smoke script

- [ ] **Step 9.1 — Run full MLIR suite** `cd build && ninja check-air-mlir`.
  - Expected: the 30 pre-existing CSL tests from v4 + the new v5 tests (~7–9) all PASS.
  - The 91 pre-existing AIE/airrt/Lowering failures (documented in `project_csl_v4_status.md`) remain — unrelated, `AIR_ENABLE_AIE=OFF` build.

- [ ] **Step 9.2 — Update `utils/run_csl_sdk.sh`** — append the new subdirs to the glob list:

```bash
for d in \
  mlir/test/Targets/CSLEmit/e2e/elementwise \
  mlir/test/Targets/CSLEmit/e2e/sizes \
  mlir/test/Targets/CSLEmit/e2e/kernels \
  mlir/test/Targets/CSLEmit/e2e/multifunc \
  mlir/test/Targets/CSLEmit/e2e/layouts \
  mlir/test/Targets/CSLEmit/e2e/control_flow \
  mlir/test/Targets/CSLEmit/e2e/roundtrip \
  mlir/test/Targets/CSLEmit/e2e/sharding \
  mlir/test/Targets/CSLEmit/e2e/multi_wafer \
  mlir/test/Targets/CSLEmit/e2e/scientific \
  ; do
  # existing emit + run logic
done

# Plus the top-level files (dsds.mlir, control_flow_if.mlir, imports.mlir, simd_dsd.mlir)
```

- [ ] **Step 9.3 — Dry-run the script** on one wafer without SDK (should emit cleanly, skip the SDK run with a warning) — sanity check the glob.

- [ ] **Step 9.4 — Commit** `chore(csl): run_csl_sdk.sh covers v5 scientific + simd corpus (v5 task 9)`.

---

## Task 10 — Docs + memory update

- [ ] **Step 10.1 — Update `docs/more_docs/csl_dialect_reference.md`**:
  - DSD section: `csl.get_mem_dsd` (mem1d only) is supported; fabin/fabout remain deferred.
  - New section: `csl.builtin_call` with both forms (bare, module-member).
  - Scope paragraph: single-PE full / N-PE SIMD / no inter-PE; deferred list (f16, math lib, fabric DSDs).

- [ ] **Step 10.2 — Fix the `CLAUDE.md` typo** — `ninja check-airmlir` → `ninja check-air-mlir` in the testing section. (One-line edit.)

- [ ] **Step 10.3 — Create `.claude/projects/.../memory/project_csl_v5_status.md`**:

```markdown
---
name: CSL v5 implementation status
description: CSL v5 (single-PE complete + N-PE SIMD) shipped on branch air-to-fire; N tasks complete
type: project
---
CSL v5 spec + plan + implementation, branch `air-to-fire`:
- Plan: `docs/superpowers/plans/2026-04-18-csl-v5-single-pe-simd.md` (10 tasks)

**Scope anchors (locked in):**
- Mem1d DSDs only (`csl.get_mem_dsd`); no fabric DSDs.
- One generic `csl.builtin_call` op covers all CSL builtins + module-member calls.
- Generic user-level `csl.import_module` emission (any path, any params).
- `scf.if` + `arith.cmpf` / `cmpi` in kernel scope (no yielded values).
- N-PE SIMD is a pure test-composition — no emitter code.
- f32 / i32 only. f16 deferred to one-shot follow-up patch.
- Math library (`math.sqrt` etc.) deferred — builtin_call op supports module-member form today.

**Execution target:** `ninja check-air-mlir` (note: `check-airmlir` does NOT exist — was a typo in CLAUDE.md).
Narrow loop: `ninja check-airmlir-dialect-csl` / `check-airmlir-targets-cslemit`.

**Next up (not v5):**
1. f16/i16 one-shot patch.
2. Math library (tests + maybe emitter tweaks on module-member form).
3. Colors/routes/tasks → unlocks halo stencils, GEMM with collectives.
```

- [ ] **Step 10.4 — Commit** `docs(csl): v5 reference + fix check target typo + v5 status memory (v5 task 10)`.

---

## Execution

Run via **`superpowers:executing-plans`** from the repo root on branch `air-to-fire`. Each numbered task becomes one TaskCreate todo; each `Step N.M` becomes one commit on a clean working tree.

**Preflight** (before Task 1):

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air
source sandbox/bin/activate
source utils/env_setup_gpu.sh install llvm/install
cd build && ninja install && ninja check-airmlir-dialect-csl
# baseline: should be 30/30 green (v4 state)
```

If the baseline is not green, fix that before starting v5.

**Task count:** 10 tasks, ~45 commits total. Each task ends with its own narrow-suite green + a single semantic commit.
