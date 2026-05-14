# CSL Dialect v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen CSL dialect coverage so we can emit multi-operator, multi-function, N-PE SIMD programs from a grouped `.mlir` E2E corpus.

**Architecture:** Reuse upstream MLIR (`func.func`, `arith.*`) wherever possible. Extend the existing `csl_layout.place` op with a subgrid range form (no new op). Emitter iterates all `csl.wafer` symbols in the module, writing one subdir per wafer. Test corpus is pure `.mlir` under `mlir/test/Targets/CSLEmit/e2e/`.

**Tech Stack:** MLIR 22 C++, TableGen, FileCheck + lit, `air-opt` / `air-translate`, Cerebras SDK 1.4 (`cslc`, `cs_python`).

**Spec:** `docs/superpowers/specs/2026-04-17-csl-v4-subgrid-design.md`

**Design rule throughout:** follow upstream MLIR conventions. If an approach here diverges from standard practice, prefer the upstream way.

**Scope anchors (from the spec, do not re-litigate in tasks):**
- **WSE-3 only.** Drop any `commands_wse2.sh` / `wse2` mention; unknown `--arch` values error.
- **Both 1-D and 2-D subgrid** in `csl_layout.place over […]` — 2-D is **not** deferred.
- **Equal sharding is enforced.** Host `memref` element count must divide evenly by the subgrid PE count; unequal is a verifier error (see Task 9).
- **PE coords come from the `<layout>` library at runtime.** The program emitter auto-injects `@import_module("<layout>")`; kernels call `layout_module.get_x_coord()` / `get_y_coord()`. `vars` / `params` on `csl_layout.place` is reserved for values that can't be derived from the coord.
- **CSL language reference:** https://sdk.cerebras.net/csl/language_index for syntax/type rules when the emitter needs a new construct.
- **SDK Runtime API reference:** https://sdk.cerebras.net/api-docs/sdkruntime-api for `memcpy_h2d` / `memcpy_d2h` argument semantics.

---

## Task 1: Cosmetics + stale-code sweep (test dir rename + help text + wse2/CSLV2ToPy leftovers)

Warmup task: isolate every purely textual / tree-level cleanup from behavioural changes so later tasks touch only real logic.

**Files:**
- Rename: `mlir/test/Targets/CSLV2ToPy/` → `mlir/test/Targets/CSLEmit/`
- Modify: `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` (help text — currently mentions `commands_wse{2,3}.sh`)
- Modify: `mlir/test/Targets/CMakeLists.txt` (if it lists the subdir)
- Sweep (grep-and-fix, not blind rename) every remaining mention of: `CSLV2ToPy`, `wse2`, `commands_wse2`, `place_range`, `--wafer=` across `mlir/`, `tools/`, `docs/`, `python/`, `test/gpu/`. Each hit: decide in-place whether it's live (delete/fix) or historical (leave in `docs/superpowers/` design notes).

- [ ] **Step 1: Rename the test directory**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air
git mv mlir/test/Targets/CSLV2ToPy mlir/test/Targets/CSLEmit
```

- [ ] **Step 2: Update `CMakeLists.txt` if it references the old path**

Check `mlir/test/Targets/CMakeLists.txt` and `mlir/test/Targets/CSLEmit/CMakeLists.txt` (the moved one). Replace any `CSLV2ToPy` with `CSLEmit`.

- [ ] **Step 3: Fix `--emit-csl` help text**

Edit `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` line ~317:

Old:
```cpp
"Emit all CSL files (program, layout, host, commands_wse{2,3}.sh) "
"into --output-dir",
```

New:
```cpp
"Emit all CSL files (program, layout, host, commands_wse3.sh) "
"into --output-dir. Writes one subdirectory per csl.wafer.",
```

- [ ] **Step 4: Sweep stale references**

Run these greps from the repo root and fix every hit (keep historical context inside `docs/superpowers/specs/` and `docs/superpowers/plans/` older than 2026-04-17 — do not touch those):

```bash
# Use the Grep tool (not shell) — each should return few if any non-doc hits.
# Targets: mlir/, tools/, python/, utils/, test/gpu/, CMakeLists.txt
```

Patterns to audit:
- `CSLV2ToPy` — anywhere. Expect: the renamed dir + its CMakeLists (handled in steps 1–2). Anything else is stale.
- `commands_wse2` / `wse2` — expect: references removed from help text, code, and fixture scripts. Anything in `docs/more_docs/` is historical; leave alone unless it's user-facing install/build instructions.
- `place_range` — expect: zero hits outside pre-2026-04-17 design notes.
- `--wafer=` / `WaferSelector` — expect: zero hits. Any remaining CLI selector is stale from the earlier draft.

For each live hit, apply the minimal fix (delete or rename). For each clearly-historical hit, leave it but note the path in the commit message.

- [ ] **Step 5: Build + run tests**

```bash
cd build && ninja install && ninja check-csl
```

Expected: all 27 existing tests pass, paths printed as `Targets/CSLEmit/*` not `Targets/CSLV2ToPy/*`.

- [ ] **Step 6: Commit**

```bash
git add mlir/test/Targets/CSLEmit mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp <any other sweep hits>
git commit -m "refactor(csl-emit): rename CSLV2ToPy to CSLEmit; drop wse2; sweep stale refs"
```

---

## Task 2: Expand arith op coverage in emitter

Add `subf / mulf / divf / maxf / minf / negf / subi / muli` cases to `emitFuncBody`. Same structure as the existing `addf` / `addi` cases.

**Files:**
- Modify: `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h` (the `emitFuncBody` function starting at line 67)
- Test: `mlir/test/Targets/CSLEmit/e2e/elementwise.mlir` (new)

- [ ] **Step 1: Write the failing FileCheck test first**

Create `mlir/test/Targets/CSLEmit/e2e/elementwise.mlir`:

```mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=VECMUL %s < %t/vecmul_f32/program.csl
// RUN: FileCheck --check-prefix=VECSUB %s < %t/vecsub_f32/program.csl
// RUN: FileCheck --check-prefix=VECDIV %s < %t/vecdiv_f32/program.csl
// RUN: FileCheck --check-prefix=VECMAX %s < %t/vecmax_f32/program.csl

csl.wafer @vecmul_f32 {arch = "wse3"} {
  csl.program @pe {
    %a = csl.var @a : memref<256xf32>
    %b = csl.var @b : memref<256xf32>
    %c = csl.var @c : memref<256xf32>
    csl.func @compute {
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c256 step %c1 {
        %va = memref.load %a[%i] : memref<256xf32>
        %vb = memref.load %b[%i] : memref<256xf32>
        %r  = arith.mulf %va, %vb : f32
        memref.store %r, %c[%i] : memref<256xf32>
      }
      csl.return
    }
    csl.export @a {alias = "a", direction = "in"}
    csl.export @b {alias = "b", direction = "in"}
    csl.export @c {alias = "c", direction = "out"}
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 1, height = 1} @layout {
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>)
      {layout = @layout} {
    csl_host.memcpy_h2d %a to @layout::@a {px=0, py=0, width=1, height=1}
    csl_host.memcpy_h2d %b to @layout::@b {px=0, py=0, width=1, height=1}
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@c to %c {px=0, py=0, width=1, height=1}
  }
}
// VECMUL: var t{{[0-9]+}}: f32 = {{.*}} * {{.*}};

csl.wafer @vecsub_f32 {arch = "wse3"} { /* same shape, arith.subf */ }
// VECSUB: var t{{[0-9]+}}: f32 = {{.*}} - {{.*}};

csl.wafer @vecdiv_f32 {arch = "wse3"} { /* same shape, arith.divf */ }
// VECDIV: var t{{[0-9]+}}: f32 = {{.*}} / {{.*}};

csl.wafer @vecmax_f32 {arch = "wse3"} { /* same shape, arith.maximumf */ }
// VECMAX: var t{{[0-9]+}}: f32 = max({{.*}}, {{.*}});
```

Expand the truncated wafers by copy-pasting the `@vecmul_f32` body and changing only the `arith` op. Keep layouts/hosts identical.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd build && ninja install
lit ../mlir/test/Targets/CSLEmit/e2e/elementwise.mlir -v
```

Expected: fails because (a) the emitter doesn't iterate multiple wafers yet (Task 4 fixes that — test will still fail on every wafer but `vecmul_f32` even after Step 3; add a `REQUIRES` guard or narrow to one wafer here and add more after Task 4), and (b) `mulf` / `subf` / `divf` / `maxf` aren't recognized.

**NOTE:** For now narrow this file to a single `csl.wafer @vecmul_f32` and a single RUN line. We'll widen it once Task 4 lands per-wafer subdirs.

- [ ] **Step 3: Add the arith op cases to `emitFuncBody`**

In `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h`, after the existing `arith::AddFOp` case (line 95), add a helper macro to DRY the binary-op pattern:

```cpp
// Emit a binary CSL expression: `var tN: <csl-ty> = lhs OP rhs;`
auto emitBinary = [&](mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                      llvm::StringRef cslOp) {
  std::string l = resolve(nameMap, lhs);
  std::string r = resolve(nameMap, rhs);
  std::string tname = "t" + std::to_string(tempCount++);
  indent(os, indentLevel);
  os << "var " << tname << ": " << cslTypeName(result.getType()) << " = "
     << l << " " << cslOp << " " << r << ";\n";
  nameMap[result] = tname;
};
```

Replace the existing `AddFOp` / `AddIOp` blocks with:

```cpp
if (auto o = dyn_cast<arith::AddFOp>(&op))  { emitBinary(o.getLhs(), o.getRhs(), o.getResult(), "+"); continue; }
if (auto o = dyn_cast<arith::SubFOp>(&op))  { emitBinary(o.getLhs(), o.getRhs(), o.getResult(), "-"); continue; }
if (auto o = dyn_cast<arith::MulFOp>(&op))  { emitBinary(o.getLhs(), o.getRhs(), o.getResult(), "*"); continue; }
if (auto o = dyn_cast<arith::DivFOp>(&op))  { emitBinary(o.getLhs(), o.getRhs(), o.getResult(), "/"); continue; }
if (auto o = dyn_cast<arith::AddIOp>(&op))  { emitBinary(o.getLhs(), o.getRhs(), o.getResult(), "+"); continue; }
if (auto o = dyn_cast<arith::SubIOp>(&op))  { emitBinary(o.getLhs(), o.getRhs(), o.getResult(), "-"); continue; }
if (auto o = dyn_cast<arith::MulIOp>(&op))  { emitBinary(o.getLhs(), o.getRhs(), o.getResult(), "*"); continue; }
```

For `max`/`min` use function-call form instead of infix:

```cpp
auto emitBuiltin = [&](mlir::Value lhs, mlir::Value rhs, mlir::Value result,
                       llvm::StringRef fn) {
  std::string l = resolve(nameMap, lhs);
  std::string r = resolve(nameMap, rhs);
  std::string tname = "t" + std::to_string(tempCount++);
  indent(os, indentLevel);
  os << "var " << tname << ": " << cslTypeName(result.getType()) << " = "
     << fn << "(" << l << ", " << r << ");\n";
  nameMap[result] = tname;
};

if (auto o = dyn_cast<arith::MaximumFOp>(&op))  { emitBuiltin(o.getLhs(), o.getRhs(), o.getResult(), "max"); continue; }
if (auto o = dyn_cast<arith::MinimumFOp>(&op))  { emitBuiltin(o.getLhs(), o.getRhs(), o.getResult(), "min"); continue; }
```

For `negf` (unary):

```cpp
if (auto o = dyn_cast<arith::NegFOp>(&op)) {
  std::string v = resolve(nameMap, o.getOperand());
  std::string tname = "t" + std::to_string(tempCount++);
  indent(os, indentLevel);
  os << "var " << tname << ": " << cslTypeName(o.getResult().getType())
     << " = -" << v << ";\n";
  nameMap[o.getResult()] = tname;
  continue;
}
```

- [ ] **Step 4: Run the narrowed test**

```bash
cd build && ninja install
lit ../mlir/test/Targets/CSLEmit/e2e/elementwise.mlir -v
```

Expected: PASS for the single `@vecmul_f32` wafer. (Widening to all five wafers happens after Task 4.)

- [ ] **Step 5: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/CSLEmitCommon.h mlir/test/Targets/CSLEmit/e2e/elementwise.mlir
git commit -m "feat(csl-emit): support sub/mul/div/max/min/neg arith ops"
```

---

## Task 3: Support `func.func` / `func.call` / `func.return` in emitter

Let users declare internal helpers via upstream `func` dialect; call and return via `func.call` / `func.return`. No new CSL op.

**Files:**
- Modify: `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h` (`emitFuncBody`)
- Modify: `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp` (walk `func.func` before `csl.func`)
- Test: `mlir/test/Targets/CSLEmit/e2e/multifunc.mlir` (new)

- [ ] **Step 1: Write the failing test**

Create `mlir/test/Targets/CSLEmit/e2e/multifunc.mlir`:

```mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=PROG %s < %t/helper_add/program.csl

csl.wafer @helper_add {arch = "wse3"} {
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
      %c0 = arith.constant 0 : index
      %n = arith.constant 256 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 2.0 : f32
      scf.for %i = %c0 to %n step %c1 {
        %va = memref.load %a[%i] : memref<256xf32>
        %vb = memref.load %b[%i] : memref<256xf32>
        %r  = func.call @scaled_add(%va, %vb, %cst) : (f32, f32, f32) -> f32
        memref.store %r, %c[%i] : memref<256xf32>
      }
      csl.return
    }
    csl.export @a {alias = "a", direction = "in"}
    csl.export @b {alias = "b", direction = "in"}
    csl.export @c {alias = "c", direction = "out"}
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 1, height = 1} @layout {
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>)
      {layout = @layout} {
    csl_host.memcpy_h2d %a to @layout::@a {px=0, py=0, width=1, height=1}
    csl_host.memcpy_h2d %b to @layout::@b {px=0, py=0, width=1, height=1}
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@c to %c {px=0, py=0, width=1, height=1}
  }
}
// PROG: fn scaled_add(x: f32, y: f32, s: f32) f32 {
// PROG:   var t{{[0-9]+}}: f32 = x * s;
// PROG:   var t{{[0-9]+}}: f32 = t{{[0-9]+}} + y;
// PROG:   return t{{[0-9]+}};
// PROG: }
// PROG: fn compute() void {
// PROG:   var t{{[0-9]+}}: f32 = scaled_add({{.*}}, {{.*}}, {{.*}});
// PROG: }
```

- [ ] **Step 2: Run, verify failure**

```bash
cd build && ninja install
lit ../mlir/test/Targets/CSLEmit/e2e/multifunc.mlir -v
```

Expected: fails at `func.func` / `func.call` / `func.return` — "unsupported op in function body".

- [ ] **Step 3: Emit `func.func` helpers in `CSLProgramEmitter`**

In `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp`, between the variable-declarations section and the `csl.func @compute` emission, add:

```cpp
// Emit func.func private helpers before csl.func @compute.
progOp.walk([&](func::FuncOp f) {
  if (!f.isPrivate()) return WalkResult::advance();
  os << "fn " << f.getSymName() << "(";
  auto args = f.getArguments();
  for (auto it : llvm::enumerate(args)) {
    if (it.index()) os << ", ";
    std::string argName = f.getArgAttrOfType<StringAttr>(
                             it.index(), "llvm.name")
                             ? f.getArgAttrOfType<StringAttr>(
                                    it.index(), "llvm.name").str()
                             : ("a" + std::to_string(it.index()));
    nameMap[it.value()] = argName;
    os << argName << ": " << cslTypeName(it.value().getType());
  }
  os << ") ";
  if (f.getNumResults() > 0)
    os << cslTypeName(f.getResultTypes()[0]) << " ";
  else
    os << "void ";
  os << "{\n";
  unsigned tempCount = 0;
  llvm::DenseMap<Value, std::string> innerMap = nameMap;
  if (failed(emitFuncBody(f.getBody(), os, /*indentLevel=*/1, outerMap, innerMap, tempCount)))
    return WalkResult::interrupt();
  os << "}\n\n";
  return WalkResult::advance();
});
```

(If `nameMap` / `outerMap` are scoped differently in this file, adapt accordingly — see how the existing `csl.func @compute` emission passes them.)

- [ ] **Step 4: Recognise `func.call` + `func.return` in `emitFuncBody`**

In `CSLEmitCommon.h`, add before the "Unknown op" fallthrough:

```cpp
if (auto callOp = dyn_cast<func::CallOp>(&op)) {
  std::string tname;
  indent(os, indentLevel);
  if (callOp.getNumResults() > 0) {
    tname = "t" + std::to_string(tempCount++);
    os << "var " << tname << ": "
       << cslTypeName(callOp.getResult(0).getType()) << " = ";
    nameMap[callOp.getResult(0)] = tname;
  }
  os << callOp.getCallee() << "(";
  for (auto it : llvm::enumerate(callOp.getOperands())) {
    if (it.index()) os << ", ";
    os << resolve(nameMap, it.value());
  }
  os << ");\n";
  continue;
}
if (auto retOp = dyn_cast<func::ReturnOp>(&op)) {
  indent(os, indentLevel);
  if (retOp.getNumOperands() > 0) {
    os << "return " << resolve(nameMap, retOp.getOperand(0)) << ";\n";
  } else {
    os << "return;\n";
  }
  continue;
}
```

- [ ] **Step 5: Rebuild, rerun**

```bash
cd build && ninja install
lit ../mlir/test/Targets/CSLEmit/e2e/multifunc.mlir -v
```

Expected: PASS.

- [ ] **Step 6: Auto-inject `@import_module("<layout>")` into every `program.csl`**

CSL kernels that need their own PE coordinates should call `layout_mod.get_x_coord()` / `get_y_coord()` (see [CSL `<layout>` library](https://sdk.cerebras.net/csl/language/libraries#layout)). The emitter makes this always available without IR-level plumbing.

In `CSLProgramEmitter.cpp`, at the top of the preamble (right after the existing `@import_module("<memcpy/memcpy>", ...)`), add:

```cpp
os << "const layout_mod = @import_module(\"<layout>\");\n";
```

Add a test wafer to `multifunc.mlir` that uses `layout_mod.get_x_coord()` directly in its kernel:

```mlir
csl.wafer @coord_aware {arch = "wse3"} {
  csl.program @pe {
    %c = csl.var @c : memref<1xi32>
    csl.func @compute {
      // Kernel reads its own x coord. The emitter produces this as an
      // inline CSL snippet via a dedicated csl.comptime block (or
      // equivalent) — for Step 6's scope, just assert the import appears
      // in every program.csl.
      csl.return
    }
    csl.export @c {alias = "c", direction = "out"}
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 1, height = 1} @layout {
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main(%c: memref<1xi32>) {layout = @layout} {
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@c to %c {px=0, py=0, width=1, height=1}
  }
}

// COORD: const layout_mod = @import_module("<layout>");
```

Extend the top RUN block with one more FileCheck:

```mlir
// RUN: FileCheck --check-prefix=COORD %s < %t/coord_aware/program.csl
```

- [ ] **Step 7: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/CSLEmitCommon.h \
        mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp \
        mlir/test/Targets/CSLEmit/e2e/multifunc.mlir
git commit -m "feat(csl-emit): support func.func/call/return + auto-import <layout>"
```

---

## Task 4: Per-wafer emission subdirs

Emitter walks every `csl.wafer` in the module and writes one subdir per wafer under `--output-dir`.

**Files:**
- Modify: `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` (`emitAll`)
- Modify: `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp` (accept a wafer arg so it emits one program)
- Modify: `mlir/lib/Targets/CSLEmit/CSLLayoutEmitter.cpp` (same)
- Modify: `mlir/lib/Targets/CSLEmit/CSLHostEmitter.cpp` (same)
- Test: `mlir/test/Targets/CSLEmit/e2e/elementwise.mlir` (widen to multi-wafer)

- [ ] **Step 1: Write the failing multi-wafer test**

Widen `mlir/test/Targets/CSLEmit/e2e/elementwise.mlir` to contain 4 wafers (`vecmul`, `vecsub`, `vecdiv`, `vecmax` — all f32, size 256), plus 4 `FileCheck --check-prefix` RUN lines reading per-wafer subdirs (`%t/vecmul_f32/program.csl`, etc.).

- [ ] **Step 2: Run, verify failure**

Expected: only the first wafer's files exist in `%t/` (since current emitter picks the first program). The per-wafer `FileCheck` reads fail.

- [ ] **Step 3: Refactor emitter entry points to take a wafer**

Change the three emitter functions from `(ModuleOp, raw_ostream&)` to `(csl::WaferOp, raw_ostream&)`:

In `CSLProgramEmitter.cpp` rename `runProgramEmitter(ModuleOp, ...)` to a helper that takes `WaferOp`. Update the three per-wafer calls. Keep the old `ModuleOp`-based functions as thin wrappers that pick the first wafer (for the existing `--emit-csl-program` etc. stdout translations).

- [ ] **Step 4: Walk all wafers in `emitAll`**

Rewrite `emitAll` in `CSLEmitAll.cpp`:

```cpp
static LogicalResult emitAll(ModuleOp module, llvm::raw_ostream &os) {
  if (EmitCslOutputDir.empty()) {
    module.emitError() << "--emit-csl requires --output-dir=<path>";
    return failure();
  }
  if (auto ec = llvm::sys::fs::create_directories(EmitCslOutputDir.getValue())) {
    module.emitError() << "cannot create output-dir: " << ec.message();
    return failure();
  }

  bool any = false;
  auto result = module.walk([&](csl::WaferOp wafer) -> WalkResult {
    any = true;
    std::string waferName = wafer.getSymName().str();
    llvm::SmallString<128> waferDir(EmitCslOutputDir.getValue());
    llvm::sys::path::append(waferDir, waferName);
    if (auto ec = llvm::sys::fs::create_directories(waferDir)) {
      wafer.emitError() << "cannot create " << waferDir.c_str()
                        << ": " << ec.message();
      return WalkResult::interrupt();
    }

    std::string progName;
    int64_t layoutW, layoutH;
    probeLayoutForWafer(wafer, progName, layoutW, layoutH);

    auto open = [&](StringRef filename,
                    std::unique_ptr<llvm::raw_fd_ostream> &outPtr) {
      llvm::SmallString<128> p(waferDir);
      llvm::sys::path::append(p, filename);
      std::error_code ec;
      outPtr = std::make_unique<llvm::raw_fd_ostream>(
          StringRef(p.data(), p.size()), ec, llvm::sys::fs::OF_Text);
      return ec ? failure() : success();
    };

    std::unique_ptr<llvm::raw_fd_ostream> progOs, layoutPyOs, hostOs;
    if (failed(open(progName + ".csl", progOs)) ||
        failed(open("csl_layout.py", layoutPyOs)) ||
        failed(open("run.py", hostOs)))
      return WalkResult::interrupt();

    if (failed(runProgramEmitterForWafer(wafer, *progOs)) ||
        failed(runLayoutEmitterForWafer(wafer, *layoutPyOs)) ||
        failed(runHostEmitterForWafer(wafer, *hostOs)))
      return WalkResult::interrupt();
    progOs.reset(); layoutPyOs.reset(); hostOs.reset();

    // layout.csl + commands_wse3.sh — same as current single-wafer logic,
    // but parameterised on waferDir/progName/layoutW/layoutH.
    // Factor out into a helper if the body grows.

    os << "emitted: " << waferDir.c_str() << "/\n";
    return WalkResult::advance();
  });
  if (!any) {
    module.emitError() << "no csl.wafer found in module";
    return failure();
  }
  return result.wasInterrupted() ? failure() : success();
}
```

`probeLayoutForWafer(WaferOp, ...)` is a small helper adapted from the existing `probeLayout`, scoped to a single wafer.

- [ ] **Step 5: Make the existing single-file translations still work**

The stdout translations (`--emit-csl-program`, `--emit-csl-layout`, `--emit-csl-host`) stay on the old `ModuleOp` interface — they now pick the first `csl.wafer`. This matches upstream `mlir-translate` convention where a translation picks one interpretation when multiple are possible.

- [ ] **Step 6: Rebuild, run full test suite**

```bash
cd build && ninja install && ninja check-csl
```

Expected: every `elementwise.mlir` check passes (4 wafer subdirs). Every existing v3 test still passes (single-wafer files still work — just produce one subdir now).

- [ ] **Step 7: Update any v3 tests that assumed flat emission**

Find any test with `%t/program.csl` and change to `%t/<wafer_name>/program.csl`. Candidates:

```bash
grep -rln "%t/program.csl\|%t/layout.csl\|%t/run.py" mlir/test/
```

- [ ] **Step 8: Commit**

```bash
git add mlir/lib/Targets/CSLEmit mlir/test/CSLEmit mlir/test/Targets/CSLEmit
git commit -m "refactor(csl-emit): emit one subdir per csl.wafer"
```

---

## Task 5: Extend `csl_layout.place` with subgrid range form

Unified op. `at (x, y)` stays for the 1-PE case; `over [lo:hi:stride]` is the new N-PE form. Also add optional `vars` and `params` clauses.

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLLayoutOps.td` (extend `CSL_Layout_PlaceOp`)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` (custom parser/printer)
- Test: `mlir/test/Targets/CSLEmit/e2e/layouts.mlir` (new)

- [ ] **Step 1: Write the failing roundtrip test**

Create `mlir/test/Targets/CSLEmit/e2e/layouts.mlir` with several wafers:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @place_point
csl.wafer @place_point {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 1, height = 1} @layout {
    // CHECK: csl_layout.place @pe at (0, 0)
    csl_layout.place @pe at (0, 0)
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @place_row
csl.wafer @place_row {arch = "wse3"} {
  csl.program @pe { csl.func @compute { csl.return } }
  csl.layout {width = 8, height = 1} @layout {
    // CHECK: csl_layout.place @pe over [0:8, 0]
    csl_layout.place @pe over [0:8, 0]
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}

// CHECK-LABEL: csl.wafer @place_row_vars
csl.wafer @place_row_vars {arch = "wse3"} {
  csl.program @pe(%pid: !csl.comptime<i16>) { csl.func @compute { csl.return } }
  csl.layout {width = 8, height = 1} @layout {
    // CHECK: csl_layout.place @pe over [0:8, 0] vars (%{{.*}} : i32, %{{.*}} : i32) params {pid = %{{.*}} : i16}
    csl_layout.place @pe over [0:8, 0] vars (%i : i32, %j : i32) params {pid = %i : i16}
  }
  csl.host @main() {layout = @layout} {
    csl_host.launch @layout::@compute
  }
}
```

- [ ] **Step 2: Run, verify failure**

Expected: parse error on `over […]` — the op doesn't support it yet.

- [ ] **Step 3: Extend the TableGen op**

In `mlir/include/air/Dialect/CSL/CSLLayoutOps.td`, extend `CSL_Layout_PlaceOp`:

```td
def CSL_Layout_PlaceOp : CSL_Layout_Op<"place", [Symbol]> {
  let summary = "Place a csl.program at a tile or over a subgrid";
  let description = [{
    Two forms:
      csl_layout.place @pe at (x, y)                              // 1-PE
      csl_layout.place @pe over [lo:hi:stride, lo:hi:stride]      // subgrid
                       vars (%i : i32, %j : i32)                  // optional iv names
                       params { name = %i : i16, ... }            // optional per-PE params
    The 1-PE form is shorthand for `over [x:x+1, y:y+1]`.
  }];

  let arguments = (ins
    SymbolRefAttr:$program,
    // Point form:
    OptionalAttr<I64Attr>:$px,
    OptionalAttr<I64Attr>:$py,
    // Range form:
    OptionalAttr<I64ArrayAttr>:$x_range,   // [lo, hi, stride]
    OptionalAttr<I64ArrayAttr>:$y_range,   // [lo, hi, stride]
    OptionalAttr<StrArrayAttr>:$iv_names,  // ["i", "j"]
    OptionalAttr<DictionaryAttr>:$params   // e.g. {pid = i_var}
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
```

- [ ] **Step 4: Add custom parse/print**

In `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` (or the corresponding `CSLLayoutOps.cpp` if split), add:

- `parseAt((parser, result))`: accept `at (` INT `,` INT `)` → set `px` / `py`
- `parseOver((parser, result))`: accept `over [` range (`,` range)? `]` `vars`? `params`? → set `x_range` / optional `y_range` / optional `iv_names` / optional `params`
- Print the same shape

Exact code goes in the custom parser/printer function; follow the pattern of existing ops in the same file.

- [ ] **Step 5: Add verifier**

```cpp
LogicalResult PlaceOp::verify() {
  bool isPoint = getPx().has_value() || getPy().has_value();
  bool isRange = getXRange().has_value();
  if (isPoint && isRange)
    return emitOpError("cannot use both `at (x, y)` and `over [...]`");
  if (!isPoint && !isRange)
    return emitOpError("must use either `at (x, y)` or `over [...]`");
  if (isPoint && (!getPx() || !getPy()))
    return emitOpError("`at (x, y)` requires both px and py");
  // vars / params only legal with over form:
  if (isPoint && (getIvNames().has_value() || getParams().has_value()))
    return emitOpError("vars/params only allowed with `over [...]` form");
  return success();
}
```

- [ ] **Step 6: Run test, confirm PASS**

```bash
cd build && ninja install
lit ../mlir/test/Targets/CSLEmit/e2e/layouts.mlir -v
```

- [ ] **Step 7: Commit**

```bash
git add mlir/include/air/Dialect/CSL mlir/lib/Dialect/CSL mlir/test/Targets/CSLEmit/e2e/layouts.mlir
git commit -m "feat(csl-layout): extend place op with subgrid range form"
```

---

## Task 6: Layout emitter — subgrid-aware `@set_tile_code`

When `csl_layout.place` is in `over` form, emit a `for` loop over the range instead of one-off `@set_tile_code` calls. Propagate `params` into the tile code struct.

**Files:**
- Modify: `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp` (`makeLayoutCsl` — currently only iterates a flat `width × height`)

- [ ] **Step 1: Widen the test — add CHECK lines for `layout.csl`**

In `mlir/test/Targets/CSLEmit/e2e/layouts.mlir`, after the existing roundtrip RUN line, add four variants: point, 1-D row, 2-D subgrid, and a `params`-bearing row (the `params` example uses a non-coord name like `shard_offset` because coords themselves come from the `<layout>` library — see Task 3b):

```mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=POINT-LAYOUT  %s < %t/place_point/layout.csl
// RUN: FileCheck --check-prefix=ROW-LAYOUT    %s < %t/place_row/layout.csl
// RUN: FileCheck --check-prefix=GRID-LAYOUT   %s < %t/place_grid/layout.csl
// RUN: FileCheck --check-prefix=PARAMS-LAYOUT %s < %t/place_row_vars/layout.csl

// POINT-LAYOUT: @set_rectangle(1, 1);
// POINT-LAYOUT: @set_tile_code(0, 0, "pe.csl"

// ROW-LAYOUT: @set_rectangle(8, 1);
// ROW-LAYOUT: for (i: i16, 0..8) {
// ROW-LAYOUT:   @set_tile_code(i, 0, "pe.csl"

// GRID-LAYOUT: @set_rectangle(4, 4);
// GRID-LAYOUT: for (j: i16, 0..4) {
// GRID-LAYOUT:   for (i: i16, 0..4) {
// GRID-LAYOUT:     @set_tile_code(i, j, "pe.csl"

// PARAMS-LAYOUT: for (i: i16, 0..8) {
// PARAMS-LAYOUT:   @set_tile_code(i, 0, "pe.csl", .{
// PARAMS-LAYOUT:     .memcpy_params = memcpy.get_params(i),
// PARAMS-LAYOUT:     .shard_offset = i,
```

- [ ] **Step 2: Run, verify failure**

Expected: no `for (i: i16, ...)` loop appears; today's emitter unrolls.

- [ ] **Step 3: Implement in `makeLayoutCsl`**

Rewrite the tile-code block. Instead of nested `for` loops over `width × height` hard-coding each `(x, y)`, walk the wafer's `csl.layout` region and find every `csl_layout.place`. Handle three cases (point, 1-D, 2-D):

```cpp
wafer.walk([&](csl_layout::PlaceOp p) {
  if (p.getPx().has_value()) {
    // point form — unchanged
    int64_t x = *p.getPx(), y = *p.getPy();
    os << "  @set_rectangle(1, 1);\n";
    os << "  @set_tile_code(" << x << ", " << y << ", \""
       << progName << ".csl\", .{ .memcpy_params = memcpy.get_params("
       << x << ") });\n";
    return;
  }

  // range form: extract x range (+ optional y range)
  auto xr = *p.getXRange();   // [lo, hi, stride] as IntegerAttr[3]
  int64_t xlo = cast<IntegerAttr>(xr[0]).getInt();
  int64_t xhi = cast<IntegerAttr>(xr[1]).getInt();
  StringRef iName = "i";
  StringRef jName = "j";
  if (auto ivs = p.getIvNames()) {
    iName = cast<StringAttr>((*ivs)[0]).getValue();
    if (ivs->size() > 1) jName = cast<StringAttr>((*ivs)[1]).getValue();
  }

  bool has2D = p.getYRange().has_value();
  if (has2D) {
    auto yr = *p.getYRange();
    int64_t ylo = cast<IntegerAttr>(yr[0]).getInt();
    int64_t yhi = cast<IntegerAttr>(yr[1]).getInt();
    os << "  @set_rectangle(" << (xhi - xlo) << ", " << (yhi - ylo) << ");\n";
    os << "  for (" << jName << ": i16, " << ylo << ".." << yhi << ") {\n";
    os << "    for (" << iName << ": i16, " << xlo << ".." << xhi << ") {\n";
    emitTileCodeBody(os, progName, iName, jName, p, /*indent=*/6);
    os << "    }\n";
    os << "  }\n";
  } else {
    os << "  @set_rectangle(" << (xhi - xlo) << ", 1);\n";
    os << "  for (" << iName << ": i16, " << xlo << ".." << xhi << ") {\n";
    emitTileCodeBody(os, progName, iName, /*jName=*/"0", p, /*indent=*/4);
    os << "  }\n";
  }
});
```

Factor the tile-code struct emission into a small helper so point/1-D/2-D paths stay DRY:

```cpp
static void emitTileCodeBody(raw_ostream &os, StringRef progName,
                             StringRef xExpr, StringRef yExpr,
                             csl_layout::PlaceOp p, int indent) {
  std::string pad(indent, ' ');
  os << pad << "@set_tile_code(" << xExpr << ", " << yExpr << ", \""
     << progName << ".csl\", .{\n";
  os << pad << "  .memcpy_params = memcpy.get_params(" << xExpr << "),\n";
  if (auto ps = p.getParams()) {
    for (auto namedAttr : *ps) {
      os << pad << "  ." << namedAttr.getName().strref() << " = "
         << cslExprOfAttr(namedAttr.getValue(), xExpr, yExpr) << ",\n";
    }
  }
  os << pad << "});\n";
}
```

`cslExprOfAttr` maps the attr back to its CSL source spelling — if it's a ref to `%i`, print `xExpr`; `%j` → `yExpr`; else print the literal.

- [ ] **Step 4: Run, confirm PASS**

```bash
cd build && ninja install && lit ../mlir/test/Targets/CSLEmit/e2e/layouts.mlir -v
```

- [ ] **Step 5: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp
git commit -m "feat(csl-emit): emit @set_tile_code loop for subgrid placement"
```

---

## Task 7: Host emitter — subgrid-aware `(w, h, l)` with equal-sharding

Host `memcpy_h2d` / `memcpy_d2h` arguments must reflect the subgrid extent so one memcpy covers all PEs in the placement. CSL requires **equal sharding**: total host buffer length must divide evenly by the PE count.

**Background (SDK Runtime API — https://sdk.cerebras.net/api-docs/sdkruntime-api):**
`memcpy_h2d(tensor_id, host_buf, 0, 0, w, h, l, elem_size, …)` means:
- `(w, h)` = **PE grid extent** of the placement (not the array shape).
- `l` = **per-PE element count** (each PE receives exactly `l` elements of `elem_size` bytes).
- Host buffer length must equal `w * h * l`.

**Derivation table** (ported verbatim from spec §6.4):

| `place` form | `(w, h)` | `l` |
|---|---|---|
| `at (x, y)` | `(1, 1)` | full element count of the `memref` |
| `over [0:W, 0]` | `(W, 1)` | `total_elems / W` (must divide) |
| `over [0:W, 0:H]` | `(W, H)` | `total_elems / (W * H)` (must divide) |

Worked examples:
- `memref<256xf32>` + `over [0:8, 0]` → `w=8, h=1, l=32, elem_size=4`.
- `memref<16x16xf32>` + `over [0:4, 0:4]` → `w=4, h=4, l=16, elem_size=4` (each PE gets a 16-elem tile).
- `memref<4x4xf32>` + `over [0:4, 0:4]` → `w=4, h=4, l=1, elem_size=4` (each PE gets 1 element — user's worked example).

Unequal sharding (e.g. `memref<255xf32>` on 8 PEs) is **not** emitted — it must have been rejected by `-csl-verify-params` (Task 9) before reaching the host emitter. The host emitter should `assert` divisibility as a defensive internal check.

**Files:**
- Modify: `mlir/lib/Targets/CSLEmit/CSLHostEmitter.cpp`

- [ ] **Step 1: Extend `layouts.mlir` with host-side CHECK lines for all three cases**

```mlir
// RUN: FileCheck --check-prefix=POINT-HOST  %s < %t/place_point/run.py
// RUN: FileCheck --check-prefix=ROW-HOST    %s < %t/place_row/run.py
// RUN: FileCheck --check-prefix=GRID-HOST   %s < %t/place_grid/run.py

// 256-elem array on 1 PE
// POINT-HOST: runner.memcpy_h2d(runner.get_id("a"), a, 0, 0, 1, 1, 256

// 256-elem array on 8 PEs (row)
// ROW-HOST:   runner.memcpy_h2d(runner.get_id("a"), a, 0, 0, 8, 1, 32

// 16x16 array on 4x4 PE grid (16 elems per PE)
// GRID-HOST:  runner.memcpy_h2d(runner.get_id("a"), a, 0, 0, 4, 4, 16
```

- [ ] **Step 2: Run, verify failure**

Today's host emitter uses the `csl.layout {width, height}` from the wafer and assumes `l = total_elems`. The 8-PE row test will produce `1, 1, 256` — wrong.

- [ ] **Step 3: Rewrite the `(w, h, l)` derivation**

In `CSLHostEmitter.cpp`, find the `memcpy_h2d` / `memcpy_d2h` emission. Replace the current `{width, height, total_elems}` lookup with:

```cpp
// For each host op (csl_host.memcpy_h2d / memcpy_d2h):
//   1. Resolve the target var symbol → csl.var.
//   2. Find the csl_layout.place that references the var's enclosing csl.program.
//   3. Compute (w, h) from the place form and l from memref element count.

auto place = findPlaceForProgram(wafer, progSym);
int64_t total = memrefElemCount(op.getBuffer().getType());
int64_t w, h, l;
if (place.getPx().has_value()) {
  w = 1; h = 1; l = total;
} else {
  auto xr = *place.getXRange();
  w = cast<IntegerAttr>(xr[1]).getInt() - cast<IntegerAttr>(xr[0]).getInt();
  if (place.getYRange().has_value()) {
    auto yr = *place.getYRange();
    h = cast<IntegerAttr>(yr[1]).getInt() - cast<IntegerAttr>(yr[0]).getInt();
  } else {
    h = 1;
  }
  assert(total % (w * h) == 0 && "-csl-verify-params should have rejected unequal sharding");
  l = total / (w * h);
}
os << "  runner.memcpy_h2d(runner.get_id(\"" << alias << "\"), "
   << buf << ", 0, 0, " << w << ", " << h << ", " << l
   << ", " << elemBytes << ", ...);\n";
```

- [ ] **Step 4: Run, confirm PASS** for all three cases.

- [ ] **Step 5: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/CSLHostEmitter.cpp mlir/test/Targets/CSLEmit/e2e/layouts.mlir
git commit -m "feat(csl-emit): derive host memcpy (w,h,l) from subgrid extent + equal sharding"
```

---

## Task 8: `-air-to-csl` lowers N-PE herd to `place over […]` (1-D and 2-D)

`air.herd` with `size = [N, 1]` → `csl_layout.place @pe over [0:N, 0]`. `size = [N, M]` with both >1 → `csl_layout.place @pe over [0:N, 0:M]`. `[1, 1]` stays as `at (0, 0)`. Do **not** skip the 2-D case — 2-D subgrid is in scope for v4.

**Files:**
- Modify: `mlir/lib/Conversion/AIRToCSLPass.cpp`
- Delete: `mlir/test/Conversion/AIRToCSL/reject_2x2_herd.mlir`
- Test: `mlir/test/Conversion/AIRToCSL/simd_herd.mlir` (new)

- [ ] **Step 1: Write the new FileCheck**

Two wafers in one file — 1-D and 2-D:

```mlir
// RUN: air-opt %s -air-to-csl | FileCheck %s
// CHECK-LABEL: csl.wafer @simd_1d
// CHECK:       csl_layout.place @pe over [0:8, 0]
// CHECK-LABEL: csl.wafer @simd_2d
// CHECK:       csl_layout.place @pe over [0:4, 0:4]

module {
  func.func @vecadd_simd_1d(...) {
    air.herd tile (%tx, %ty) in (%sx = 8, %sy = 1)
        args(%a = ..., %b = ..., %c = ...) : ... {
      // body: loads, arith.addf, store
      air.herd_terminator
    }
    return
  }

  func.func @vecadd_simd_2d(...) {
    air.herd tile (%tx, %ty) in (%sx = 4, %sy = 4)
        args(%a = ..., %b = ..., %c = ...) : ... {
      air.herd_terminator
    }
    return
  }
}
```

- [ ] **Step 2: Run, verify failure**

Today's pass rejects `size != [1, 1]`.

- [ ] **Step 3: Relax the rejection + emit range form**

In `AIRToCSLPass.cpp`, find the herd-size check (currently the source of the `reject_2x2_herd.mlir` diagnostic). Replace:

```cpp
// Old:
if (sizeX != 1 || sizeY != 1)
  return herd.emitError("unsupported herd size");

// New:
if (sizeX < 1 || sizeY < 1)
  return herd.emitError("herd size must be positive");
```

Then change the `csl_layout.place` construction:

```cpp
OpBuilder builder(...);
if (sizeX == 1 && sizeY == 1) {
  // Point form.
  builder.create<csl_layout::PlaceOp>(loc, progSymbolRef,
      /*px=*/0, /*py=*/0, /*x_range=*/nullptr, /*y_range=*/nullptr,
      /*iv_names=*/nullptr, /*params=*/nullptr);
} else {
  // Range form. 1-D when sizeY == 1; 2-D otherwise.
  auto xRange = builder.getI64ArrayAttr({0, sizeX, 1});
  ArrayAttr yRange = (sizeY > 1)
      ? builder.getI64ArrayAttr({0, sizeY, 1})
      : nullptr;
  builder.create<csl_layout::PlaceOp>(loc, progSymbolRef,
      /*px=*/nullptr, /*py=*/nullptr, xRange, yRange,
      /*iv_names=*/nullptr, /*params=*/nullptr);
}
```

- [ ] **Step 4: Delete the stale reject test**

```bash
rm mlir/test/Conversion/AIRToCSL/reject_2x2_herd.mlir
```

- [ ] **Step 5: Build, run all conversion tests**

```bash
cd build && ninja install && ninja check-csl
```

- [ ] **Step 6: Commit**

```bash
git add mlir/lib/Conversion/AIRToCSLPass.cpp mlir/test/Conversion/AIRToCSL
git commit -m "feat(air-to-csl): lower N-PE air.herd to csl_layout.place over [0:N, 0]"
```

---

## Task 9: Update `-csl-verify-params`

Extend existing pass to validate:
(a) `func.call` callees resolve to a `func.func` in the same `csl.program`.
(b) Range-form `params` names match program block args.
(c) **Equal sharding** — every `csl_host.memcpy_*` buffer's element count divides evenly by the `csl_layout.place` PE count (see Task 7 for semantics).
(d) `func.func` at program level **must be `private`** (non-private rejected with diagnostic).

**Files:**
- Modify: `mlir/lib/Conversion/CSLVerifyParams.cpp`
- Test: `mlir/test/Conversion/AIRToCSL/verify_params.mlir` (extend)

- [ ] **Step 1: Add failing-case tests**

Extend `verify_params.mlir` with four new cases:

```mlir
// Case: unknown callee
// CHECK: error: func.call references unknown callee @missing

// Case: range params name mismatch
// CHECK: error: csl_layout.place passes parameter 'foo' but @pe has no block arg 'foo'

// Case: unequal sharding
// CHECK: error: csl_host.memcpy_h2d buffer has 255 elements, not divisible by 8-PE placement

// Case: non-private helper
// CHECK: error: csl.program helpers must be 'private' func.func; @helper is not private
```

- [ ] **Step 2: Run, verify failure**

Today's pass ignores both cases.

- [ ] **Step 3: Implement checks**

In `CSLVerifyParams.cpp`:

```cpp
// Check 1: func.call callees
program.walk([&](func::CallOp c) {
  auto callee = SymbolTable::lookupSymbolIn(program, c.getCalleeAttr());
  if (!callee || !isa<func::FuncOp>(callee)) {
    c.emitError() << "func.call references unknown callee "
                  << c.getCallee();
    signalPassFailure();
  }
});

// Check 2: range-form params
wafer.walk([&](csl_layout::PlaceOp p) {
  auto params = p.getParams();
  if (!params) return;
  auto prog = SymbolTable::lookupSymbolIn(wafer, p.getProgram());
  auto progOp = cast<csl::ProgramOp>(prog);
  llvm::StringSet<> validArgs;
  if (auto names = progOp.getParamNames()) {
    for (auto attr : *names)
      validArgs.insert(cast<StringAttr>(attr).getValue());
  }
  for (auto nattr : *params) {
    if (!validArgs.contains(nattr.getName().strref())) {
      p.emitError() << "csl_layout.place passes parameter '"
                    << nattr.getName().strref()
                    << "' but @" << progOp.getSymName()
                    << " has no block arg '" << nattr.getName().strref() << "'";
      signalPassFailure();
      return;
    }
  }
});

// Check 3: equal sharding. For each memcpy, find the matching place and
// require total_elems % (w * h) == 0.
wafer.walk([&](Operation *op) {
  if (!isa<csl_host::MemcpyH2DOp, csl_host::MemcpyD2HOp>(op)) return;
  auto buf = op->getOperand(0);             // the host-side buffer
  auto mrT = cast<MemRefType>(buf.getType());
  int64_t total = 1;
  for (auto d : mrT.getShape()) total *= d;
  auto place = findPlaceForHostOp(wafer, op);
  int64_t w = 1, h = 1;
  if (!place.getPx().has_value()) {
    auto xr = *place.getXRange();
    w = cast<IntegerAttr>(xr[1]).getInt() - cast<IntegerAttr>(xr[0]).getInt();
    if (place.getYRange().has_value()) {
      auto yr = *place.getYRange();
      h = cast<IntegerAttr>(yr[1]).getInt() - cast<IntegerAttr>(yr[0]).getInt();
    }
  }
  if (total % (w * h) != 0) {
    op->emitError() << "csl_host.memcpy_* buffer has " << total
                    << " elements, not divisible by " << (w * h) << "-PE placement";
    signalPassFailure();
  }
});

// Check 4: helpers must be private.
program.walk([&](func::FuncOp f) {
  if (!f.isPrivate()) {
    f.emitError() << "csl.program helpers must be 'private' func.func; @"
                  << f.getSymName() << " is not private";
    signalPassFailure();
  }
});
```

- [ ] **Step 4: Run, confirm PASS.**

- [ ] **Step 5: Commit**

```bash
git add mlir/lib/Conversion/CSLVerifyParams.cpp mlir/test/Conversion/AIRToCSL/verify_params.mlir
git commit -m "feat(csl-verify-params): validate func.call callees + range-form params"
```

---

## Task 10: Build out the rest of the E2E corpus

Fill in the remaining files under `mlir/test/Targets/CSLEmit/e2e/`:

- [ ] **Step 1: `sizes.mlir`** — `vecadd` wafers at sizes 64, 256, 1024; plus one rank-2 `memref<16x16xf32>` wafer with a nested loop.
- [ ] **Step 2: `kernels.mlir`** — `dot`, `reduce`, `saxpy`, `relu` wafers (each a csl.wafer with appropriate `csl.func @compute`). Use ops added in Task 2.
- [ ] **Step 3: `control_flow.mlir`** — wafers covering: simple scf.for; nested scf.for; 3-arith-ops body; mixed loads/stores with temps.
- [ ] **Step 4: `roundtrip.mlir`** — parse-print sanity for every new form introduced in Tasks 3 / 5.

Each file: `// RUN:` the full pipeline, `FileCheck --check-prefix=<NAME>` each wafer's `program.csl` (and `layout.csl` / `run.py` if variant-specific).

- [ ] **Step 5: Run the whole corpus**

```bash
cd build && ninja install && ninja check-csl
```

- [ ] **Step 6: Commit**

```bash
git add mlir/test/Targets/CSLEmit/e2e/
git commit -m "test(csl-emit): add grouped E2E corpus (sizes/kernels/control_flow/roundtrip)"
```

---

## Task 11: SDK run helper

Pure-MLIR tests only FileCheck the emitted text; use this script to take a per-wafer emission dir to the simulator.

**Files:**
- Create: `utils/run_csl_sdk.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Usage: run_csl_sdk.sh <emission-root>
# Iterates every subdirectory with a commands_wse3.sh and runs it.
# Requires: cslc, cs_python on PATH.
set -e
ROOT="${1:?usage: run_csl_sdk.sh <dir>}"
any_fail=0
for dir in "$ROOT"/*/; do
  name="$(basename "$dir")"
  if [[ -x "$dir/commands_wse3.sh" ]]; then
    echo "=== $name ==="
    if ( cd "$dir" && ./commands_wse3.sh 2>&1 | tee run.log | grep -q "SUCCESS!" ); then
      echo "  PASS"
    else
      echo "  FAIL (see $dir/run.log)"
      any_fail=1
    fi
  fi
done
exit $any_fail
```

- [ ] **Step 2: chmod + commit**

```bash
chmod +x utils/run_csl_sdk.sh
git add utils/run_csl_sdk.sh
git commit -m "feat(utils): add run_csl_sdk.sh to batch-run emitted wafers on simulator"
```

- [ ] **Step 3: Manual SDK verification (not a lit test)**

On a machine with Cerebras SDK on PATH:

```bash
# Re-run the corpus so %t is populated. Easiest: one pass through lit.
cd build && ninja check-csl
# Then find the lit temp dir for elementwise.mlir (printed on -v output)
# and run the sdk helper against it.
utils/run_csl_sdk.sh /tmp/lit-tmp-XXXXXX/elementwise.mlir.tmp/
```

Expected: `PASS` for every 1-PE wafer; `PASS` for at least the 8-PE row wafer.

---

## Rollout order (why this sequence)

- Tasks 1 → 4 are pure infrastructure; they land first so Tasks 5+ build on the per-wafer scaffold.
- Tasks 5 → 8 deliver the subgrid feature end-to-end (dialect → verifier → conversion → emitter → host).
- Task 9 tightens diagnostics once the new shapes exist.
- Task 10 is the corpus build-out; it relies on Tasks 2/3/4 being in place so the tests are meaningful.
- Task 11 is the SDK runner, usable any time after Task 4.

Estimated effort: 1–2 days each for Tasks 1/2/3/11; 1–2 days each for Tasks 6/7/8/9; 3–5 days for Tasks 4/5/10 (most complex).

---

## Definition of done (from spec §11)

- Every `.mlir` file under `mlir/test/Targets/CSLEmit/e2e/` passes `ninja check-csl`.
- `air-translate --emit-csl --output-dir=%t` on a multi-wafer file creates one subdir per wafer.
- `utils/run_csl_sdk.sh` prints `SUCCESS!` for every 1-PE wafer and for the N-PE SIMD vecadd wafer on the CS-3 simulator.
- `func.func private @helper` emits as a CSL `fn helper(...)` before the `csl.func @compute` entry.
- `csl_layout.place @pe over [0:8, 0]` round-trips and `layout.csl` contains `for (i: i16, 0..8) { @set_tile_code(i, 0, ...) }`.
- `mlir/test/Targets/CSLV2ToPy/` renamed to `mlir/test/Targets/CSLEmit/`.
- Help text for `--emit-csl` no longer mentions `wse2`.
