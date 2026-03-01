# CSL Runtime Dialect — Implementation Plan

This document is the **authoritative implementation plan** for introducing the **runtime dialect** (Plan B from [CSL_RUNTIME_DIALECT_OPTIONS.md](CSL_RUNTIME_DIALECT_OPTIONS.md)), separating host-API ops from the CSL dialect, and emitting **layout.py** (SdkLayout API) and **run.py** (SdkRuntime API) from the runtime dialect. It is written so that an implementor (e.g. Claude) can follow it step-by-step with precise file names and deliverables.

**API references (stable subset):**
- **SdkRuntime API:** https://sdk.cerebras.net/api-docs/sdkruntime-api  
- **SdkLayout API:** https://sdk.cerebras.net/api-docs/sdklayout-api  

**Target example:** [GEMV 5: Multiple PEs](https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-05-multiple-pes) — multiple PEs, memcpy H2D/D2H, launch, no streaming. run.py should be generatable as a **template** from runtime dialect IR; layout should use **Python SdkLayout API** (layout.py), not layout.csl.

---

## 1. Goals and Constraints

| # | Goal / constraint | Implication |
|---|-------------------|-------------|
| 1 | **Separate generic CSL from runtime** | Ops that map 1:1 to SdkLayout/SdkRuntime go into the **runtime dialect** (`csl_rt`). CSL dialect keeps only **semantic** layout/kernel ops (regions, colors, placement, ports, etc.). |
| 2 | **Canonical MLIR format** | All ops use **declarative `assemblyFormat`** only. No custom parsers/printers. Standard form: `%result = dialect.op %arg0, %arg1 {attr = value} : (type0, type1) -> result_type`. Apply to **both** CSL and runtime dialect. |
| 3 | **CSL → runtime lowering** | 1-to-N mapping. Implement when trivial; otherwise add a **TODO** and a placeholder lowering. |
| 4 | **Stable, minimal backend** | Emit only a **small subset** of the SDK needed for a simple multi-PE + memcpy + launch program. Other ops can be stubbed or commented. |
| 5 | **GEMV-05 as end-to-end example** | Final artifact: a program that corresponds to GEMV-05 (layout.py + run.py template + kernel .csl), using SdkLayout/SdkRuntime Python APIs. |

---

## 2. Naming and Scope

- **Runtime dialect name:** `csl_rt` (namespace `xilinx::csl_rt` in C++).
- **CSL dialect:** Stays `csl`; no name change for now. Optionally consider more generic names later (e.g. `spatial.placement`) in a follow-up.
- **Ops that move to `csl_rt`:** Those that directly correspond to a single SdkLayout or SdkRuntime API call (see Section 5).
- **Ops that stay in CSL:** Spatial/semantic structure (e.g. `spatial_placement`, `code_region`, `place`, `paint`, `color`, `route`, ports, streams, `export_name`, kernel/module/func/task, DSDs). These remain the “generic” program representation; they **lower** to `csl_rt` (1-to-N where trivial, else TODO).

---

## 3. Directory and File Layout

All paths relative to repo root (e.g. `mlir-air-gpu/`).

### 3.1 Runtime dialect (new)

```
mlir/include/air/Dialect/CSLRuntime/
  CMakeLists.txt
  CSLRuntimeBase.td          # Dialect, base op class, types, enums
  CSLRuntimeOps.td           # All csl_rt.* ops (layout + runtime)
  CSLRuntimeDialect.h
  CSLRuntimeOps.h

mlir/lib/Dialect/CSLRuntime/IR/
  CMakeLists.txt
  CSLRuntimeDialect.cpp
  CSLRuntimeOps.cpp          # Only if needed for non-TableGen logic; else empty/minimal
```

No custom asm parsers/printers: all ops use TableGen `assemblyFormat`.

### 3.2 CSL dialect (existing — modify)

```
mlir/include/air/Dialect/CSL/
  CMakeLists.txt             # Unchanged or add dependency on CSLRuntime if needed later
  CSLBase.td                 # Keep; remove any runtime-only types if moved
  CSLLayoutOps.td            # Keep only semantic layout ops; ensure declarative assembly
  CSLRoutingOps.td           # Keep; canonical assembly
  CSLKernelOps.td            # Keep; replace custom asm with assemblyFormat
  CSLDataMovementOps.td      # Keep; canonical assembly
  CSLRuntimeOps.td           # Keep only PE/module/export semantics (export_name, etc.); not host API
  CSLOps.td                  # Includes; do not include CSLRuntime
  CSLDialect.h
  CSLOps.h

mlir/lib/Dialect/CSL/IR/
  CSLDialect.cpp
  CSLOps.cpp                 # Remove custom parsers/printers for func/task; use TableGen
```

### 3.3 Conversion (new)

```
mlir/include/air/Conversion/CSLToCSLRuntime/
  (none required if pass is in same dir as impl)

mlir/lib/Conversion/CSLToCSLRuntime/
  CMakeLists.txt
  CSLToCSLRuntime.cpp        # Conversion pass: CSL layout ops → csl_rt ops
```

### 3.4 Translation (backend emission)

```
mlir/lib/Targets/
  CSLToTextTranslation.cpp   # Existing: keep for csl.kernel → .csl emission
  CSLRuntimeToPy.cpp         # NEW: csl_rt → layout.py + run.py (template)
```

### 3.5 Registration and driver

- **Dialect registration:** `mlir/lib/InitAll.cpp` and `mlir/include/air/InitAll.h` — register `CSLRuntimeDialect`.
- **Pass registration:** Register `csl-to-csl-rt` in the appropriate Passes.h/Passes.td and add to air-opt.
- **Translation:** Register `--emit-csl-rt` (or similar) in `tools/air-translate/air-translate.cpp` to run `CSLRuntimeToPy` and optionally kernel emission.

### 3.6 Tests and example

```
mlir/test/Conversion/CSLToCSLRuntime/
  layout_to_runtime.mlir      # CSL spatial_placement → csl_rt (if lowering implemented)
  (or TODO_lowering.mlir with TODO)

mlir/test/Dialect/CSLRuntime/
  layout_ops.mlir            # Parsing/printing csl_rt layout ops
  runtime_ops.mlir           # Parsing/printing csl_rt runtime ops (load, run, get_id, etc.)
  gemv05_example.mlir        # Minimal GEMV-05-style csl_rt program

mlir/test/Target/
  emit_csl_rt_layout_py.mlir # FileCheck for emitted layout.py
  emit_csl_rt_run_py.mlir    # FileCheck for emitted run.py (template)
```

### 3.7 Documentation

```
docs/more_docs/
  CSL_RUNTIME_DIALECT_OPTIONS.md   # Existing
  CSL_RUNTIME_DIALECT_IMPLEMENTATION_PLAN.md  # This file
  CSL_RUNTIME_DIALECT_SPEC.md      # NEW: Op list, types, canonical form, mapping to API
```

---

## 4. Canonical MLIR Format (Mandatory)

Every op in both CSL and `csl_rt` must use **declarative** `assemblyFormat` in TableGen. No `hasCustomAssemblyFormat = 1` and no custom `parse()`/`print()` in C++ for these ops.

**Standard form:**

```mlir
%result = csl_rt.create_code_region %layout %file, %name, %w, %h : (!csl_rt.layout, !none, !none, index, index) -> !csl_rt.code_region
```

For ops with only attributes (no SSA operands):

```mlir
%layout = csl_rt.create_layout {target = #csl_rt.target<wse3>} : () -> !csl_rt.layout
```

**CSL dialect:** Remove custom assembly for `csl.func` and `csl.task`; define equivalent `assemblyFormat` so they print in the same style (e.g. `%result = csl.func @name () : () -> () { ... }`).

---

## 5. Runtime Dialect: Minimal Op Set

Only include ops needed for **GEMV-05-style** program: multi-PE rectangle, one code region, place, set_param_all, export_name, compile, then SdkRuntime: load, run, get_id, memcpy_h2d, memcpy_d2h, launch, stop. Other API methods can be commented or omitted.

### 5.1 Types (`CSLRuntimeBase.td`)

| Type            | MLIR syntax        | Corresponds to        |
|-----------------|--------------------|------------------------|
| LayoutType      | `!csl_rt.layout`   | SdkLayout instance     |
| CodeRegionType  | `!csl_rt.code_region` | CodeRegion object   |
| ColorType       | `!csl_rt.color`    | Color (optional for minimal) |
| RoutingPositionType | `!csl_rt.routing_position` | RoutingPosition |
| PortHandleType  | `!csl_rt.port`     | PortHandle (optional for minimal) |
| StreamType      | `!csl_rt.stream`   | Stream name/handle (optional) |
| CompileArtifactsType | `!csl_rt.compile_artifacts` | SdkCompileArtifacts |
| RuntimeType     | `!csl_rt.runtime`   | SdkRuntime instance    |

For **minimal** first version: `!csl_rt.layout`, `!csl_rt.code_region`, `!csl_rt.compile_artifacts`, `!csl_rt.runtime`. Add Color/Port/Stream when adding streaming or ports.

### 5.2 Layout ops (SdkLayout API — [sdklayout-api](https://sdk.cerebras.net/api-docs/sdklayout-api))

| csl_rt op | SDK call | Notes |
|-----------|----------|--------|
| `csl_rt.create_layout` | `SdkLayout(platform)` or `SdkLayout()` | Creates layout. Optional `platform` operand for run.py path. |
| `csl_rt.create_code_region` | `layout.create_code_region(source, name, width, height)` | source=file path, name=string, width/height=dimensions. |
| `csl_rt.place` | `code_region.place(x, y)` | Single place call. |
| `csl_rt.set_param_all` | `code_region.set_param_all(name, value)` | Name and value (int/float) as attributes. |
| `csl_rt.export_name` | `layout.export_name("name", "type_spec")` | Export symbol for host. |
| `csl_rt.compile` | `layout.compile(out_prefix='out')` | Returns compile artifacts. |

Optional (can add later or comment): alloc_color, routing_position, paint, create_input_port, create_output_port, create_input_stream, create_output_stream, connect.

### 5.3 Runtime ops (SdkRuntime API — [sdkruntime-api](https://sdk.cerebras.net/api-docs/sdkruntime-api))

| csl_rt op | SDK call | Notes |
|-----------|----------|--------|
| `csl_rt.runtime_create` | `SdkRuntime(compile_artifacts, platform, ...)` | Build runtime from artifacts. |
| `csl_rt.load` | `runtime.load()` | No operands; consumes runtime symbol. |
| `csl_rt.run` | `runtime.run()` | No operands. |
| `csl_rt.stop` | `runtime.stop()` | No operands. |
| `csl_rt.get_id` | `runtime.get_id(symbol_name)` | Returns symbol id (int). |
| `csl_rt.memcpy_h2d` | `runtime.memcpy_h2d(dest, src, px, py, w, h, elem_per_pe, ...)` | dest=id, src=array (template placeholder), ROI (px,py,w,h), elem_per_pe. |
| `csl_rt.memcpy_d2h` | `runtime.memcpy_d2h(dest, src, px, py, w, h, elem_per_pe, ...)` | dest=array (template), src=id, ROI, elem_per_pe. |
| `csl_rt.launch` | `runtime.launch(symbol_name, nonblock=False)` | Launch host-callable function. |

All ops use **assemblyFormat**; types and attributes as in TableGen. For `memcpy_*` and `launch`, use attributes for ROI and options; the emitter can generate Python with placeholders (e.g. `# TODO: fill data`) where host arrays go.

---

## 6. CSL Dialect: What Stays, What Lowers

- **Stay in CSL (semantic):**  
  `spatial_placement`, `code_region`, `paint`, `place`, `dataflow`, `color`, `route`, `input_port`, `output_port`, `input_stream`, `output_stream`, `export_name`, `set_param_all`, `set_param`, `set_param_color`, `kernel`, `module`, `func`, `task`, `var`, `param`, `comptime`, `get_mem_dsd`, `get_fab_dsd`, `mov`, etc.

- **Lower to csl_rt (1-to-1 or 1-to-N):**  
  - One `csl.spatial_placement` body → sequence of `csl_rt.create_layout`, `csl_rt.create_code_region`, `csl_rt.place`, `csl_rt.set_param_all`, `csl_rt.export_name`, `csl_rt.compile`.  
  - If trivial (linear scan of body, no complex control flow), implement in `CSLToCSLRuntime.cpp`.  
  - If non-trivial (e.g. multiple regions, ports, streams), implement **one** simple path (e.g. single region, no ports) and add **TODO** for the rest.

- **Remove from CSL:**  
  Any op that is **purely** a duplicate of an SDK call with no extra semantics should live only in `csl_rt`. Today’s CSL “layout” ops are semantic; we do **not** delete them—we **lower** them to `csl_rt`.

---

## 7. Implementation Phases (Step-by-Step)

### Phase 1: Runtime dialect skeleton (no emission)

1. **Add** `mlir/include/air/Dialect/CSLRuntime/CMakeLists.txt` — TableGen for dialect and ops.
2. **Add** `CSLRuntimeBase.td`: dialect `csl_rt`, base op class, types `LayoutType`, `CodeRegionType`, `CompileArtifactsType`, `RuntimeType` (C++ declared, TableGen predicate).
3. **Add** `CSLRuntimeOps.td`: minimal ops — `create_layout`, `create_code_region`, `place`, `set_param_all`, `export_name`, `compile`, `runtime_create`, `load`, `run`, `stop`, `get_id`, `memcpy_h2d`, `memcpy_d2h`, `launch`. All with **declarative** `assemblyFormat`.
4. **Add** `CSLRuntimeDialect.h` / `CSLRuntimeOps.h` and `CSLRuntimeDialect.cpp` / `CSLRuntimeOps.cpp` (minimal; no custom asm).
5. **Wire** CMake and `InitAll.cpp` / `InitAll.h` so `csl_rt` is registered.
6. **Add** `mlir/test/Dialect/CSLRuntime/layout_ops.mlir` and `runtime_ops.mlir` — parse/print tests in **canonical** form.

**Deliverable:** `air-opt` can parse and print a module containing `csl_rt.*` ops; no emission yet.

---

### Phase 2: Canonical form for CSL dialect

1. **Change** `csl.func` and `csl.task` from custom assembly to **declarative** `assemblyFormat` in `CSLKernelOps.td` (so they look like `%r = csl.func @name () : () -> () { ... }` and `%r = csl.task @name color(3) { ... }`).
2. **Remove** custom `parse`/`print` for func/task from `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`.
3. **Audit** all other CSL ops: ensure every op uses only `assemblyFormat` (no `hasCustomAssemblyFormat`). Fix any that do.
4. **Run** existing CSL tests; fix any parse/print regressions.

**Deliverable:** CSL dialect uses only canonical SSA form and declarative assembly; tests pass.

---

### Phase 3: CSL → csl_rt lowering (minimal path)

1. **Add** `mlir/lib/Conversion/CSLToCSLRuntime/CMakeLists.txt` and `CSLToCSLRuntime.cpp`.
2. **Implement** lowering for **one** simple path:  
   - One `csl.spatial_placement` with one `csl.code_region`, one `csl.place`, and a few `csl.set_param_all` and `csl.export_name`.  
   - Pattern: spatial_placement → create_layout; code_region → create_code_region; place → place; set_param_all → set_param_all; export_name → export_name.  
   - Kernel file emission stays as today (csl.kernel → .csl); no change.
3. **If** multiple regions, ports, or streams are present: add **TODO** in code and a test `TODO_lowering.mlir` that documents the unsupported case.
4. **Register** pass `csl-to-csl-rt` and add test `mlir/test/Conversion/CSLToCSLRuntime/layout_to_runtime.mlir` (or TODO test).

**Deliverable:** For the minimal layout, `air-opt --csl-to-csl-rt` produces valid `csl_rt` IR.

---

### Phase 4: Emit layout.py and run.py from csl_rt

1. **Add** `CSLRuntimeToPy.cpp` in `mlir/lib/Targets/`: walk `csl_rt` ops and emit Python.
2. **Layout emission:** For each `csl_rt.create_layout`, `create_code_region`, `place`, `set_param_all`, `export_name`, `compile` emit the corresponding SdkLayout API calls. Use a **single** layout variable (e.g. `layout`) and region variables keyed by SSA value or name. Match [SdkLayout API](https://sdk.cerebras.net/api-docs/sdklayout-api).
3. **Run emission:** For `runtime_create`, emit `SdkRuntime(compile_artifacts, platform, ...)`. For `load`, `run`, `stop`, emit `runtime.load()`, `runtime.run()`, `runtime.stop()`. For `get_id`, emit `runtime.get_id("name")`. For `memcpy_h2d`/`memcpy_d2h`, emit the call with **placeholders** for host arrays (e.g. `# data = ... ; runtime.memcpy_h2d(...)`). For `launch`, emit `runtime.launch("name", nonblock=False)`. Match [SdkRuntime API](https://sdk.cerebras.net/api-docs/sdkruntime-api).
4. **run.py as template:** Emit argument parsing, platform creation, and a clear section for “fill in data” so the output is a runnable template consistent with GEMV-05.
5. **Register** translation `--emit-csl-rt` (or `--csl-rt-emit`) in `air-translate`; output directory option `--csl-output-dir` or similar.
6. **Add** tests: `emit_csl_rt_layout_py.mlir` and `emit_csl_rt_run_py.mlir` with FileCheck on emitted layout.py and run.py.

**Deliverable:** From a csl_rt module (or from CSL after lowering), `air-translate --emit-csl-rt` produces layout.py and run.py that match the minimal API subset.

---

### Phase 5: GEMV-05 end-to-end example

1. **Add** `mlir/test/Dialect/CSLRuntime/gemv05_example.mlir`: a single module that encodes the **structure** of GEMV-05 (multiple PEs, one code region, place, set_param_all for width/M/N, export_name for A, x, b, y, compute; then runtime load/run, get_id, memcpy_h2d for A/x/b, launch compute, memcpy_d2h for y, stop).
2. **Emit** layout.py and run.py from this module; add FileCheck tests.
3. **Document** in `CSL_RUNTIME_DIALECT_SPEC.md`: op list, type list, and mapping to [SdkRuntime API](https://sdk.cerebras.net/api-docs/sdkruntime-api) and [SdkLayout API](https://sdk.cerebras.net/api-docs/sdklayout-api). Note that run.py is a **template** (user fills data) and layout uses **Python SdkLayout API**, not layout.csl.

**Deliverable:** One MLIR file (gemv05_example.mlir) → layout.py + run.py template + kernel .csl; docs updated.

---

## 8. File Names Quick Reference

| Purpose | Path |
|--------|------|
| Runtime dialect base | `mlir/include/air/Dialect/CSLRuntime/CSLRuntimeBase.td` |
| Runtime dialect ops | `mlir/include/air/Dialect/CSLRuntime/CSLRuntimeOps.td` |
| Runtime dialect C++ | `mlir/lib/Dialect/CSLRuntime/IR/CSLRuntimeDialect.cpp`, `CSLRuntimeOps.cpp` |
| CSL → csl_rt pass | `mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp` |
| csl_rt → Python | `mlir/lib/Targets/CSLRuntimeToPy.cpp` |
| CSL kernel → .csl | `mlir/lib/Targets/CSLToTextTranslation.cpp` (existing) |
| Runtime spec doc | `docs/more_docs/CSL_RUNTIME_DIALECT_SPEC.md` |
| GEMV-05 example | `mlir/test/Dialect/CSLRuntime/gemv05_example.mlir` |

---

## 9. Out of Scope (for later)

- C++ SDK backend for csl_rt.
- Full SdkLayout surface (paint_all, paint_range, ports, streams, connect, edge routing). Implement minimal set; comment or stub the rest.
- Renaming CSL to a more “generic” name (e.g. spatial.*); can be a follow-up refactor.
- Optimizations on csl_rt (reordering, batching); leave for future.
