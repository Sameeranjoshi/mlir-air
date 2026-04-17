# CSL Dialect v4 — Unified Placement, `func` Dialect, E2E Test Corpus

**Date:** 2026-04-17
**Status:** Design (ready for plan)
**Branch:** `air-to-fire`
**Extends:** `2026-04-16-csl-v3-ir-refinements-design.md`
**Target:** **WSE-3 only** for v4. WSE-2 support is not a scope item; emitter drops `commands_wse2.sh` and `--arch=wse2`.
**References:**
- MLIR Stencils paper (arXiv:2601.17754 — `csl-wrapper`, `csl-ir`)
- SPADA paper (`paper/SPADA.pdf` — `place` / `dataflow` / `compute` subgrid blocks)
- MLIR `func` dialect (https://mlir.llvm.org/docs/Dialects/Func/)
- CSL language index (https://sdk.cerebras.net/csl/language_index) — authoritative CSL syntax / type system reference
- CSL `<layout>` library (https://sdk.cerebras.net/csl/language/libraries#layout) — `layout_module.get_x_coord()` / `get_y_coord()` for runtime PE coord lookup
- SDK Runtime API (https://sdk.cerebras.net/api-docs/sdkruntime-api) — authoritative host `memcpy_h2d` / `memcpy_d2h` semantics

**Design rule:** follow upstream MLIR conventions. When upstream already models a concept (functions, calls, loops, arithmetic, buffers), reuse that dialect instead of duplicating it in CSL.

---

## 1. Goal

Unblock a credible paper demo by widening coverage in four dimensions, reusing upstream MLIR where possible:

1. **Operator coverage** — from `addf` only to the full basic-arith set the CSL emitter can trivially produce.
2. **Multi-function programs** — use `func.func` / `func.call` / `func.return` instead of inventing `csl.call`.
3. **N-PE SIMD placement** — extend `csl_layout.place` with a subgrid range, not a new op.
4. **Grouped E2E `.mlir` corpus** — one file per theme, per-wafer sub-directories on emission, one FileCheck per variant.

Explicitly **not** in scope for v4:
- Removing the `direction` attribute on `csl.export` (keep explicit; user decision).
- Adding CSL `for` emission (stay on `while` only).
- `math.tanh` via `<math>` import (defer to v4.1).
- Routing / colors / streams / DSDs (v5).
- A separate `csl_layout.place_range` op (fold into `place`).

---

## 2. State at v3 (baseline)

| Area | Status |
|---|---|
| `csl.wafer / program / layout / host`, `csl_layout.*`, `csl_host.*`, `!csl.comptime<T>` | ✅ |
| `-air-to-csl`, `-csl-verify-params`, `-csl-infer-exports` | ✅ |
| Emitter split (`CSLEmit/Program,Layout,Host,All`) + unified `--emit-csl --output-dir` | ✅ |
| SDK-runnable 1-PE vecadd (`SUCCESS!` on CS-3 simulator) | ✅ |
| Emitted op coverage in `emitFuncBody` (`CSLEmitCommon.h`) | `arith.constant`, `arith.addf`, `arith.addi`, `memref.load/store`, `scf.for` |
| Multi-function programs | ❌ |
| N-PE SIMD | ❌ |
| Test corpus | 27 lit tests, all `addf` at size 256 f32 — thin |

---

## 3. Design principles

1. **Reuse upstream dialects.** `func.func` / `func.call` / `func.return` for helpers and calls. `arith.*` for arithmetic. `scf.for` for loops. No `csl.call`, no CSL-private duplicates.
2. **One op per concept.** Subgrid placement extends the existing `csl_layout.place`; no `place_range`. 1-PE case (`at (x, y)`) is shorthand for a 1×1 range.
3. **Module-level iteration.** The emitter walks every `csl.wafer` in the module and writes one output sub-directory per wafer. No `--wafer=` CLI selector.
4. **Grouped, not sprawling, tests.** Related variants share one `.mlir` file with several `csl.wafer @foo_v1 / @foo_v2 / …` blocks and one `CHECK-*` prefix per variant.
5. **Pure MLIR tests.** No `.sh` inside the MLIR test tree. Manual SDK runs use a single utility script that iterates the emitter output dirs.
6. **Keep the dialect lean.** If the boilerplate is SDK-side, it belongs in the emitter, not the IR.

---

## 4. Change A — operator coverage

### 4.1 Ops added to `emitFuncBody` (in `CSLEmitCommon.h`)

Same pattern as today's `arith::AddFOp` case block. Each adds ~5 lines.

| MLIR op | CSL emission | Type | Needed for |
|---|---|---|---|
| `arith.subf` | `var t: f32 = a - b;` | f32/f16 | elementwise |
| `arith.mulf` | `var t: f32 = a * b;` | f32/f16 | dot, saxpy, vecmul |
| `arith.divf` | `var t: f32 = a / b;` | f32/f16 | vecdiv |
| `arith.maxf` / `maximumf` | `var t: f32 = max(a, b);` | f32/f16 | relu |
| `arith.minf` / `minimumf` | `var t: f32 = min(a, b);` | f32/f16 | clamp |
| `arith.subi` | `var t: i32 = a - b;` | i32/i16 | ints |
| `arith.muli` | `var t: i32 = a * b;` | i32/i16 | ints |
| `arith.negf` | `var t: f32 = -a;` | f32/f16 | sign flip |

Element type is derived from the result type (same helper `cslTypeName()` we already use for `memref`).

### 4.2 What we deliberately do not add in v4
- `math.*` — needs `@import_module("<math>")` in the program preamble. One extra line, but defer to keep v4 scope crisp.
- Integer division / remainder — skip until a benchmark wants them.
- Comparison ops (`arith.cmpf/cmpi`) + `scf.if` — add together in v4.1 to unblock branching kernels.

---

## 5. Change B — multi-function programs via `func` dialect

### 5.1 IR shape

```mlir
csl.program @pe {
  %a = csl.var @a : memref<256xf32>
  %b = csl.var @b : memref<256xf32>
  %c = csl.var @c : memref<256xf32>

  func.func private @scaled_add(%x: f32, %y: f32, %s: f32) -> f32 {
    %m = arith.mulf %x, %s : f32
    %r = arith.addf %m, %y : f32
    func.return %r : f32
  }

  csl.func @compute {
    // iterate, call helper, store
    %c2 = arith.constant 2.0 : f32
    // … loop body …
    %r = func.call @scaled_add(%va, %vb, %c2) : (f32, f32, f32) -> f32
    // memref.store %r, %c[%i]
    csl.return
  }

  csl.export @a {alias = "a", direction = "in"}
  csl.export @b {alias = "b", direction = "in"}
  csl.export @c {alias = "c", direction = "out"}
  csl.export @compute {kind = "func"}
}
```

Rules:
- `csl.func` is still the **host-launched** PE entry. Its body terminates with the implicit `sys_mod.unblock_cmd_stream();` (unchanged).
- Helpers are `func.func private @name(args) -> ret`. `private` means they are not exported to the host (no `csl.export` generated for them).
- Calls are standard `func.call @helper(...)`.
- Returns are `func.return`.

### 5.2 CSL emission

- `func.func private @helper(%x: f32, %y: f32) -> f32 { ... func.return %r }`
  ↳ `fn helper(x: f32, y: f32) f32 { ... return r; }`
- `%r = func.call @helper(%a, %b) : (f32, f32) -> f32`
  ↳ `var t3: f32 = helper(a, b);`
- `func.return %x : f32`
  ↳ `return x;`
- A `func.func` *without* `private` visibility is still allowed but has no effect on exports. (For v4, all helpers are `private`; top-level host-facing entries remain `csl.func`.)

### 5.3 Order of emission
The program emitter emits helpers **before** `csl.func @compute` so forward references resolve. Walk order: `csl.var` decls → `csl.var` ptrs → `func.func private @*` → `csl.func` → `@export_symbol` comptime block.

### 5.4 Validation
`-csl-verify-params` gains one check: every `func.call` callee must resolve to a `func.func` symbol inside the same `csl.program`. Missing callee → error.

---

## 6. Change C — unified `csl_layout.place` with subgrid range

### 6.1 Op extension (no new op)

The existing `csl_layout.place` accepts either:
- `at (x, y)` — single point, current form (1-PE, degenerate case).
- `over [lo:hi:stride, lo:hi:stride]` — SPADA-style subgrid, new form. **Both 1-D and 2-D** are in scope for v4 (user decision).

```mlir
// 1-PE (unchanged)
csl_layout.place @pe at (0, 0)

// 8-PE row (1-D)
csl_layout.place @pe over [0:8, 0]

// 2-D subgrid (4×4)
csl_layout.place @pe over [0:4, 0:4]

// Per-PE comptime param derived from coordinate (only when a kernel needs
// something other than the plain coord — the coord itself should be read at
// runtime via <layout>, see §6.3.1).
csl_layout.place @pe over [0:8, 0] vars (%i : i32, %j : i32)
    params { shard_offset = %i : i16 }
```

Subgrid grammar (borrowed verbatim from SPADA §III):
```
subgrid ::= `[` range (`,` range)? `]`
range   ::= integer | (expr `:` expr (`:` expr)?)
```

### 6.2 Verifier
- `at (x, y)` and `over […]` are mutually exclusive; one must be present.
- `vars` and `params` are optional and only legal with `over […]`.
- Each `params` attribute name must match a block argument of the referenced `csl.program` (existing `-csl-verify-params` rule, just extended).

### 6.3 Layout emitter (`layout.csl`)

- `at (x, y)` → `@set_tile_code(x, y, "pe.csl", .{ .memcpy_params = memcpy.get_params(x) });` (unchanged).
- `over [0:W, 0]` → a `@set_rectangle(W, 1);` followed by a loop:
  ```csl
  for (i: i16, 0..W) {
    @set_tile_code(i, 0, "pe.csl", .{
      .memcpy_params = memcpy.get_params(i),
      .shard_offset = i,                    // only if `params { shard_offset = %i }` was given
    });
  }
  ```
- `over [0:W, 0:H]` → nested `for` pair (outer `j: 0..H`, inner `i: 0..W`). `@set_rectangle(W, H);` above the loops.
- `vars` / `params` on the op are **optional** — omit them entirely for plain N-PE SIMD kernels.

#### 6.3.1 Reading PE coords from the kernel (preferred)

CSL kernels that need their own coordinates should use the `<layout>` library at runtime rather than receiving them as comptime params:

```csl
const layout_mod = @import_module("<layout>");
// ...
const x = layout_mod.get_x_coord();    // u16, 0-indexed
const y = layout_mod.get_y_coord();    // u16, 0-indexed
```

The emitter injects this `@import_module("<layout>")` into every `program.csl` automatically so kernels can call `get_x_coord()` / `get_y_coord()` without any IR-level plumbing. Reserve `vars` / `params` for values that **cannot** be derived from the PE coord (e.g. a per-wafer offset passed from the host, or a shard-specific constant).

### 6.4 Host emitter (`run.py`) — equal-sharding semantics

CSL's `memcpy_h2d` / `memcpy_d2h` **require equal sharding**: the total buffer size must be an integer multiple of the PE count (or PE grid area in 2-D). Each PE holds exactly one shard of `l` elements.

`SdkRuntime.memcpy_h2d(ptr, host_buf, w, h, l, elem_size, …)` (ref: [SDK Runtime API](https://sdk.cerebras.net/api-docs/sdkruntime-api)) means:
- `(w, h)` — the **PE grid extent** of the placement.
- `l` — the **per-PE element count** along the major axis of the shard.
- Total host buffer size = `w * h * l * elem_size`.

Derivation rules in the host emitter:

| `place` form | `(w, h)` | `l` |
|---|---|---|
| `at (x, y)` | `(1, 1)` | full element count of the `memref` |
| `over [0:W, 0]` | `(W, 1)` | `total_elems / W`, must divide exactly |
| `over [0:W, 0:H]` | `(W, H)` | `total_elems / (W * H)`, must divide exactly |

If the host op's `memref` element count does **not** divide evenly by the PE count, the emitter emits a verifier error (not runtime — caught at `-csl-verify-params`). Example: `memref<256xf32>` on `over [0:8, 0]` → `l = 32` ✓. `memref<255xf32>` on `over [0:8, 0]` → error.

Worked examples (from the Definition of Done):
- `memref<256xf32>` + `over [0:8, 0]` → `memcpy_h2d(…, 8, 1, 32, 4, …)`.
- `memref<16x16xf32>` + `over [0:4, 0:4]` → `memcpy_h2d(…, 4, 4, 16, 4, …)` (each PE gets a 4×4 tile, 16 elements).
- `memref<4x4xf32>` + `over [0:4, 0:4]` → `memcpy_h2d(…, 4, 4, 1, 4, …)` (each PE gets 1 element — your worked example).

### 6.5 `-air-to-csl` change
`air.herd` with `size = [N, 1]` or `[N, M]` lowers to `csl_layout.place @pe over [0:N, 0:M]` instead of being rejected. The current 1×1 path stays, lowering to `at (0, 0)`.

---

## 7. Change D — per-wafer emission subdirs

### 7.1 New emitter semantics

```
air-translate --emit-csl --output-dir=<DIR>  input.mlir
```

- Walks every `csl.wafer` symbol in the input module.
- For each wafer `@<name>`, creates `<DIR>/<name>/` and writes:
  - `program.csl`, `layout.csl`, `csl_layout.py`, `run.py`, `commands_wse3.sh`
- Single-wafer module → single subdir. Multi-wafer module → one subdir each. No special case.

### 7.2 Why (upstream conformance)
Matches how `mlir-translate`-family tools handle modules — iterate the symbols, don't add CLI selectors. Keeps the command stable and the output layout predictable.

### 7.3 Manual SDK run

```
utils/run_csl_sdk.sh <DIR>
# iterates <DIR>/*/commands_wse3.sh, runs each, greps SUCCESS
```

---

## 8. E2E test corpus — grouped `.mlir` files

### 8.1 Layout

```
mlir/test/Targets/CSLEmit/e2e/
├── elementwise.mlir    # add/sub/mul/div/max/min × {f32,f16} and add/sub/mul × {i32,i16}
├── sizes.mlir          # vecadd @ 64 / 256 / 1024; rank-2 16×16
├── kernels.mlir        # dot, reduce, saxpy, relu (one csl.wafer each)
├── multifunc.mlir      # func.func helpers: single call, chained call, nested
├── layouts.mlir        # at (0,0); at (4,4); over [0:8,0]; over [0,0:8]; over [0:2,0:2]
├── control_flow.mlir   # scf.for (simple + nested); multi-statement bodies
└── roundtrip.mlir      # sanity — parse/print every new form
```

Seven files. Everything in each file shares a theme; variants are separate `csl.wafer` blocks.

### 8.2 Test mechanics

```mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=VECADD-F32 %s < %t/vecadd_f32/program.csl
// RUN: FileCheck --check-prefix=VECMUL-F32 %s < %t/vecmul_f32/program.csl
// … one per wafer …

csl.wafer @vecadd_f32 {arch = "wse3"} { /* … */ }
// VECADD-F32: t{{[0-9]+}} = {{.*}} + {{.*}}

csl.wafer @vecmul_f32 {arch = "wse3"} { /* … */ }
// VECMUL-F32: t{{[0-9]+}} = {{.*}} * {{.*}}
```

One `air-translate` invocation per file — writes all wafer subdirs at once. One `FileCheck` per wafer. Temp dir `%t` is lit-managed; artefacts remain on disk for manual SDK runs.

### 8.3 Coverage target
≥ 25 distinct `csl.wafer` blocks across the seven files. Every operator added in §4.1 exercised at least once. Every new placement form in §6 exercised at least once. Every emitter path (helper, nested loop, multi-statement) exercised at least once.

### 8.4 SDK e2e runs
Not part of `ninja check-csl` (lit). After a successful lit pass, developer runs `utils/run_csl_sdk.sh build/…/%t_output/` to confirm `SUCCESS!` per wafer on a machine with `cslc` and `cs_python` on PATH. Captured manually as part of paper evaluation.

---

## 9. Files changed

### New
- `mlir/test/Targets/CSLEmit/e2e/elementwise.mlir`
- `mlir/test/Targets/CSLEmit/e2e/sizes.mlir`
- `mlir/test/Targets/CSLEmit/e2e/kernels.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multifunc.mlir`
- `mlir/test/Targets/CSLEmit/e2e/layouts.mlir`
- `mlir/test/Targets/CSLEmit/e2e/control_flow.mlir`
- `mlir/test/Targets/CSLEmit/e2e/roundtrip.mlir`
- `utils/run_csl_sdk.sh`

### Modified
- `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h` — add `arith::{SubF,MulF,DivF,MaxF,MinF,NegF,SubI,MulI}Op` cases; add `func::FuncOp` / `func::CallOp` / `func::ReturnOp` cases.
- `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp` — walk `func.func private @*` before `csl.func`; pass their signatures to `emitFuncBody`.
- `mlir/lib/Targets/CSLEmit/CSLLayoutEmitter.cpp` — emit `@set_tile_code` loop when `place` is `over […]`; propagate `vars` / `params`.
- `mlir/lib/Targets/CSLEmit/CSLHostEmitter.cpp` — derive `(w, h, l)` from subgrid extent.
- `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` — walk every `csl.wafer` in module; emit to per-wafer subdir; fix help text (drop `wse2`).
- `mlir/include/air/Dialect/CSL/CSLLayoutOps.td` — extend `csl_layout.place` with optional `over` region + `vars` / `params` attrs (keep `at (x, y)` as default).
- `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` — custom parser/printer for the new form.
- `mlir/lib/Conversion/CSLVerifyParams.cpp` — support range placement + `func.call` callee validation.
- `mlir/lib/Conversion/AIRToCSLPass.cpp` — lower `air.herd size=[N, M]` to `place` in `over` form.
- `mlir/test/CMakeLists.txt` (or equivalent) — register `CSLEmit/e2e/*.mlir` with lit.
- Rename: `mlir/test/Targets/CSLV2ToPy/` → `mlir/test/Targets/CSLEmit/` (move existing three emitter tests too).

### Deleted
- `mlir/test/Conversion/AIRToCSL/reject_2x2_herd.mlir` — now supported.

---

## 10. Resolved design decisions

1. **2-D subgrid is in scope for v4.** Both `over [0:W, 0]` and `over [0:W, 0:H]` emit and verify. 2-D is not deferred.
2. **Helper visibility.** Only `private` helpers are in scope for v4. Non-private `func.func` at the program level is reserved for future features and is rejected with a helpful diagnostic (`-csl-verify-params` diagnoses it).
3. **Equal-sharding is enforced.** Host `memcpy` size must divide evenly by PE count (§6.4). Unequal sharding is a verifier error, not a runtime surprise. Two worked examples land as tests in `layouts.mlir`.
4. **PE coords via `<layout>` library.** Kernels use `layout_module.get_x_coord()` / `get_y_coord()` (SDK runtime). The emitter auto-injects `@import_module("<layout>")` in every `program.csl`. `vars` / `params` on `csl_layout.place` stays for values that cannot be derived from the coord alone.
5. **WSE-3 only.** Emitter drops any `wse2` / `commands_wse2.sh` artefacts. If a user passes `--arch=wse2` the tool errors.
6. **Stale-code cleanup is an explicit task.** Task 1 in the plan sweeps test dir renames, help text, and any remaining `wse2` / `place_range` / `CSLV2ToPy` references that fell out of earlier drafts.

---

## 11. Definition of done

- Every `.mlir` file under `mlir/test/Targets/CSLEmit/e2e/` passes `ninja check-csl`.
- `air-translate --emit-csl --output-dir=%t` on a multi-wafer file creates one subdir per wafer.
- `utils/run_csl_sdk.sh` on the per-wafer output dirs prints `SUCCESS!` for every 1-PE wafer and for the N-PE SIMD vecadd wafer on the CS-3 simulator.
- `direction = "in"/"out"` on `csl.export` is preserved (kept explicit).
- `func.func private @helper` is emitted as a CSL `fn helper(...)` before the `csl.func @compute` entry.
- `csl_layout.place @pe over [0:8, 0]` round-trips and the generated `layout.csl` contains a `for (i: i16, 0..8) { @set_tile_code(i, 0, ...) }` loop.
- `mlir/test/Targets/CSLV2ToPy/` is renamed to `mlir/test/Targets/CSLEmit/`.
- Help text for `--emit-csl` no longer mentions `wse2`.
- Every emitted `program.csl` contains `const layout_mod = @import_module("<layout>");` so kernels can read their own coords at runtime.
- `memcpy_h2d` / `memcpy_d2h` `(w, h, l)` derivation from subgrid extent is covered by the three worked examples in §6.4, each with a test in `layouts.mlir`.
- 2-D `over [0:W, 0:H]` placement emits a nested-`for` `@set_tile_code` and round-trips.
- Unequal sharding (e.g. `memref<255xf32>` on 8 PEs) is rejected by `-csl-verify-params` with a clear diagnostic.
