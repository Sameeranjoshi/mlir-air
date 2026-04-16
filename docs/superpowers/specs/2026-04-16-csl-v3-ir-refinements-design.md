# CSL Dialect v3 — IR Refinements & Emitter Architecture

## Goal

Reduce user boilerplate, enforce compile-time contracts, and make the emitter modular and maintainable. Three new passes + emitter restructure. No new dialect ops for SDK plumbing — the dialect stays lean, the emitter generates SDK boilerplate from existing IR context.

## Architecture

```
User writes:  csl.wafer { csl.program, csl.layout, csl.host }
                                    ↓
air-opt -air-to-csl                 # AIR → clean csl.wafer IR
air-opt -csl-verify-params          # validate layout↔program param contract
air-opt -csl-infer-exports          # auto-generate exports from host ops
                                    ↓
air-translate --emit-csl --output-dir=./out   # serialize to 3 files
  ├── CSLProgramEmitter  → pe.csl          (SDK preamble + vars + pointers + funcs + comptime)
  ├── CSLLayoutEmitter   → csl_layout.py   (Python SDK layout API)
  └── CSLHostEmitter     → run.py          (Python SDK runtime)
```

**Principle:** Ops for concepts users write (var, func, export, dsd, color). Emitter stages for SDK plumbing (memcpy params, pointer indirection, unblock calls).

## Tech Stack

MLIR 22 C++ (pass framework, OpBuilder), LLVM FileCheck, `air-opt` / `air-translate`, Cerebras SDK 1.4 (`cslc`, `cs_python`).

---

## 1. `-csl-verify-params` Pass

**What:** Validates that attributes on `csl_layout.place` match block arguments on the referenced `csl.program`. Emits diagnostic error on mismatch.

**Checks:**
- Every attr on `place` (excluding `px`, `py` position attrs) must have a corresponding block arg on the program with the same name
- Types must be compatible: attr type must match the `!csl.comptime<T>` inner type on the block arg
- Extra attrs on `place` that don't exist on the program → error
- Missing attrs (program expects a param not provided by place) → error

**Example error:**
```
error: csl_layout.place passes parameter 'N' to @gemv_pe but @gemv_pe has no such block argument
  csl_layout.place @gemv_pe at (0, 0) {M = 4 : i16, N = 6 : i16}
                                                      ^
```

**Implementation:** ~50 lines. Walk all `csl_layout::PlaceOp` ops, resolve the program symbol, compare attrs vs block args.

**File:** `mlir/lib/Conversion/CSLVerifyParams.cpp`
**Header:** `mlir/include/air/Conversion/CSLVerifyParamsPass.h`
**Test:** `mlir/test/Conversion/AIRToCSL/verify_params.mlir` (positive + negative cases)

---

## 2. `-csl-infer-exports` Pass

**What:** Walks `csl.host` ops and auto-generates `csl.export` ops in `csl.program` and `csl_layout.export` ops in `csl.layout`. Replaces `-csl-derive-exports`.

**Inference rules:**

| Host op | Direction | Mutability (for emitter) | Program export | Layout export |
|---|---|---|---|---|
| `csl_host.memcpy_h2d %arg to @layout::@sym` | `"in"` | mutable (`true`) | `csl.export @sym {alias = "<sym>", direction = "in"}` | `csl_layout.export "<sym>" from @prog::@sym` |
| `csl_host.memcpy_d2h @layout::@sym to %arg` | `"out"` | not mutable (`false`) | `csl.export @sym {alias = "<sym>", direction = "out"}` | `csl_layout.export "<sym>" from @prog::@sym` |
| `csl_host.launch @layout::@sym` | func | N/A | `csl.export @sym {kind = "func"}` | `csl_layout.export "<sym>" from @prog::@sym {kind = "func"}` |

**Symbol resolution:** `@layout::@sym` is a nested `SymbolRefAttr`. The pass:
1. Resolves `@layout` → finds the `csl.layout` op
2. From the layout, finds which `csl.program` the symbol belongs to (via existing `csl_layout.place` or by walking the wafer's programs)
3. Creates exports in both program and layout

**Mutability derivation:** The emitter derives mutability from direction — no new IR attribute needed:
- `direction = "in"` → `isMutable = true` (host writes to device)
- `direction = "out"` → `isMutable = false` (host reads from device)
- `kind = "func"` → no mutability concept

**Type derivation for `@export_name`:** The emitter reads the `csl.var` op's memref type to derive the pointer element type: `memref<256xf32>` → `[*]f32`.

**Idempotency:** If `csl.export` ops already exist (user wrote them manually or pass ran twice), skip duplicates.

**File:** `mlir/lib/Conversion/CSLInferExports.cpp`
**Header:** `mlir/include/air/Conversion/CSLInferExportsPass.h`
**Test:** `mlir/test/Conversion/AIRToCSL/infer_exports.mlir`

**Migration:** The current `-csl-derive-exports` pass (`CSLDeriveExports.cpp`) is superseded. Keep it temporarily for backward compat but mark deprecated. New tests use `-csl-infer-exports`.

---

## 3. Unified `--emit-csl` Translation

**What:** Single `air-translate --emit-csl --output-dir=./out` command that writes all three files.

**File naming:**
- PE program: `<program_sym_name>.csl` (e.g., `pe.csl` from `csl.program @pe`)
- Layout: `csl_layout.py`
- Host: `run.py`

**Implementation:** Register a new translation `--emit-csl` that:
1. Takes `--output-dir` string option (required)
2. Internally creates three `llvm::raw_fd_ostream` objects for the three files
3. Calls `ProgramEmitter(programOs).emit(module)`, `LayoutEmitter(layoutOs).emit(module)`, `HostEmitter(hostOs).emit(module)`

**Existing translations preserved:** `--emit-csl-program`, `--emit-csl-layout`, `--emit-csl-host` continue to work (write to stdout). Useful for debugging and FileCheck tests.

---

## 4. Emitter Restructure: 3-File Split

**Current state:** `mlir/lib/Targets/CSLV2ToPy.cpp` (600+ lines, monolithic).

**New structure:**
```
mlir/lib/Targets/CSLEmit/
  CSLProgramEmitter.cpp   — PE program emission
  CSLLayoutEmitter.cpp    — layout emission (Python SDK API)
  CSLHostEmitter.cpp      — host runtime emission
  CSLEmitAll.cpp           — --emit-csl unified entry point
  CSLEmitCommon.h         — shared helpers (cslTypeName, indent, etc.)
```

### CSLProgramEmitter

Emits `<program>.csl`. Reads `csl.program` and generates, in order:

1. **SDK preamble** — `param memcpy_params: comptime_struct;` + `const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);` (emitted if any `csl.export` with non-func kind exists, indicating host data movement)
2. **Comptime params** — from `csl.program` block args with `!csl.comptime<T>` type
3. **Variable declarations** — from `csl.var` ops → `var name: [N]T;`
4. **Pointer variables** — for each `csl.export` with `direction`: `var/const name_ptr: [*]T = &name;` (mutable from direction)
5. **Function bodies** — from `csl.func` ops, with `sys_mod.unblock_cmd_stream();` appended
6. **Comptime export block** — from `csl.export` ops:
   - Buffer exports → `@export_symbol(name_ptr, "alias");`
   - Function exports → `@export_symbol(name);`

Each section is a helper method: `emitPreamble()`, `emitParams()`, `emitVars()`, `emitPointers()`, `emitFunctions()`, `emitComptimeBlock()`.

### CSLLayoutEmitter

Emits `csl_layout.py`. Reads `csl.layout` and generates Python SDK layout API:
- `SdkLayout(target)`, `create_code_region(file, name, w, h)`, `set_param_all(...)`, `place(x, y)`, `return layout`

No changes to current behavior. Just moved to its own file.

### CSLHostEmitter

Emits `run.py`. Reads `csl.host` and generates Python SDK runtime:
- Imports, `def main(target, args...)`, `compile`, `SdkRuntime`, `memcpy_h2d/d2h`, `launch`

No changes to current behavior. Just moved to its own file.

---

## 5. User-Facing IR (Before and After)

### Before (current — user must write exports manually):
```mlir
csl.wafer @vecadd {arch = "wse3"} {
  csl.program @pe {
    %a = csl.var @a : memref<256xf32>
    %c = csl.var @c : memref<256xf32>
    csl.func @compute { /* ... */ }
    csl.export @a {alias = "a"}                          // ← boilerplate
    csl.export @c {alias = "c"}                          // ← boilerplate
    csl.export @compute {kind = "func"}                  // ← boilerplate
  }
  csl.layout {width = 1, height = 1} @layout {
    csl_layout.place @pe at (0, 0)
    csl_layout.export "a" from @pe::@a                   // ← boilerplate
    csl_layout.export "c" from @pe::@c                   // ← boilerplate
    csl_layout.export "compute" from @pe::@compute {kind = "func"}  // ← boilerplate
  }
  csl.host @main(%a_in: memref<256xf32>, %c_out: memref<256xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %a_in to @layout::@a {px=0, py=0, width=1, height=1}
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@c to %c_out {px=0, py=0, width=1, height=1}
  }
}
```

### After (user writes only the essential):
```mlir
csl.wafer @vecadd {arch = "wse3"} {
  csl.program @pe {
    %a = csl.var @a : memref<256xf32>
    %c = csl.var @c : memref<256xf32>
    csl.func @compute { /* ... */ }
    // exports auto-generated by -csl-infer-exports
  }
  csl.layout {width = 1, height = 1} @layout {
    csl_layout.place @pe at (0, 0)
    // layout exports auto-generated by -csl-infer-exports
  }
  csl.host @main(%a_in: memref<256xf32>, %c_out: memref<256xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %a_in to @layout::@a {px=0, py=0, width=1, height=1}
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@c to %c_out {px=0, py=0, width=1, height=1}
  }
}
```

---

## 6. Testing Strategy

### Pass tests (FileCheck on IR)
- `verify_params.mlir` — positive (valid params) + negative (mismatch, missing, extra)
- `infer_exports.mlir` — verify exports auto-generated from host ops, idempotency

### Emitter tests (FileCheck on text output)
- Update existing `emit_csl_program.mlir` — check new modular output matches
- Update existing `emit_csl_layout.mlir` — no change expected
- Update existing `emit_csl_host.mlir` — no change expected
- New: `emit_csl_all.mlir` — test `--emit-csl --output-dir` writes all 3 files

### End-to-end test
- Update `vecadd_e2e.mlir` — use `-csl-infer-exports` instead of `-csl-derive-exports`
- New: `vecadd_no_exports.mlir` — input with NO manual exports, full pipeline through emitter

### SDK validation test
- Compile emitted CSL with `cslc` (already verified working)
- Run on simulator with `cs_python` (already verified working)
- Script: `test/sdk/vecadd_sdk_test.sh` (not a lit test — requires SDK)

---

## 7. Implementation Order

1. **`-csl-verify-params`** — standalone, no dependencies
2. **`-csl-infer-exports`** — replaces derive-exports, updates existing tests
3. **Emitter split** — move existing code into 3 files + common header
4. **`--emit-csl` unified command** — wire up the 3 emitters with --output-dir
5. **Update `-air-to-csl` pipeline** — remove manual exports from generated IR
6. **End-to-end validation** — full pipeline → cslc → cs_python → PASS

---

## 8. Files Changed

### New files
- `mlir/lib/Conversion/CSLVerifyParams.cpp`
- `mlir/include/air/Conversion/CSLVerifyParamsPass.h`
- `mlir/lib/Conversion/CSLInferExports.cpp`
- `mlir/include/air/Conversion/CSLInferExportsPass.h`
- `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp`
- `mlir/lib/Targets/CSLEmit/CSLLayoutEmitter.cpp`
- `mlir/lib/Targets/CSLEmit/CSLHostEmitter.cpp`
- `mlir/lib/Targets/CSLEmit/CSLEmitAll.cpp`
- `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h`
- `mlir/test/Conversion/AIRToCSL/verify_params.mlir`
- `mlir/test/Conversion/AIRToCSL/infer_exports.mlir`
- `mlir/test/Conversion/AIRToCSL/vecadd_no_exports.mlir`

### Modified files
- `mlir/include/air/Conversion/Passes.h` — add new pass headers
- `mlir/lib/Conversion/Passes.cpp` — register new passes
- `mlir/lib/Conversion/CMakeLists.txt` — add new sources
- `mlir/lib/Targets/CMakeLists.txt` — restructure for CSLEmit/ subdirectory
- `mlir/test/Targets/CSLV2ToPy/emit_csl_program.mlir` — update CHECK patterns
- `mlir/test/Conversion/AIRToCSL/vecadd_e2e.mlir` — use new pass pipeline
- `tools/air-translate/air-translate.cpp` — register --emit-csl translation

### Deleted files
- `mlir/lib/Targets/CSLV2ToPy.cpp` — replaced by CSLEmit/ directory

---

## References

- CSL Builtins Spec (RPC section): `docs/superpowers/raw/csl_spec_builtins.md` lines 3438-3555
- `@export_name(name, type, isMutable)` — layout context, declares host-visible name
- `@export_symbol(symbol, ?name)` — PE comptime, advertises device symbol
- Mutability: `isMutable=true` for h2d (host writes), `false` for d2h (host reads)
- SDK tutorials: `../sdk-examples/tutorials/gemv-01` through `gemv-05` (param + export patterns)
- Stencils paper: `paper/mlir_Stencils_csl.pdf` (arXiv:2601.17754)
