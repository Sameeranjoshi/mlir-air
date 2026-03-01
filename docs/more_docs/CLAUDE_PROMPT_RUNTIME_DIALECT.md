# Claude Prompt: CSL Runtime Dialect Implementation

Use this document to instruct Claude to implement the CSL Runtime Dialect in **step-by-step** order. Each section is a **prompt** you can paste; do one phase at a time and verify before moving to the next.

**Context to give Claude once (e.g. at the start):**

- Repo: mlir-air-gpu (AIR/CSL MLIR dialects and air-translate).
- **Implementation plan:** `docs/more_docs/CSL_RUNTIME_DIALECT_IMPLEMENTATION_PLAN.md` — follow it exactly (file names, phases, canonical MLIR format).
- **API references:** SdkRuntime https://sdk.cerebras.net/api-docs/sdkruntime-api , SdkLayout https://sdk.cerebras.net/api-docs/sdklayout-api .
- **Target example:** GEMV 5 Multiple PEs https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-05-multiple-pes — run.py as template, layout via Python SdkLayout API (not layout.csl).
- **Rules:** (1) All new and modified ops use **declarative `assemblyFormat`** only; no custom parsers/printers. (2) Runtime dialect is `csl_rt`; CSL dialect keeps only semantic ops. (3) If CSL→csl_rt lowering is non-trivial for a case, add TODO and a test that documents it.

---

## Prompt 1: Phase 1 — Runtime dialect skeleton

Implement **Phase 1** from `docs/more_docs/CSL_RUNTIME_DIALECT_IMPLEMENTATION_PLAN.md`.

Do the following:

1. Create the directory `mlir/include/air/Dialect/CSLRuntime/` and add `CMakeLists.txt` for TableGen (dialect + ops), following the pattern of `mlir/include/air/Dialect/CSL/CMakeLists.txt` but for the `csl_rt` dialect.
2. Create `CSLRuntimeBase.td`: define the dialect `csl_rt` (cppNamespace `xilinx::csl_rt`), a base op class `CSLRuntime_Op`, and types: `LayoutType`, `CodeRegionType`, `CompileArtifactsType`, `RuntimeType`. Declare these types in C++ (in a header) and reference them in TableGen via DialectType predicates, same pattern as CSL dialect types.
3. Create `CSLRuntimeOps.td`: define the minimal layout and runtime ops with **declarative** `assemblyFormat` only (no custom asm):
   - **Layout:** `csl_rt.create_layout`, `csl_rt.create_code_region`, `csl_rt.place`, `csl_rt.set_param_all`, `csl_rt.export_name`, `csl_rt.compile`
   - **Runtime:** `csl_rt.runtime_create`, `csl_rt.load`, `csl_rt.run`, `csl_rt.stop`, `csl_rt.get_id`, `csl_rt.memcpy_h2d`, `csl_rt.memcpy_d2h`, `csl_rt.launch`
   Use the exact file and op names from the plan. Every op must follow the canonical form `%result = csl_rt.op_name %arg0, ... {attrs} : (types) -> result_type` (or equivalent with only attributes where applicable).
4. Add `CSLRuntimeDialect.h` and `CSLRuntimeOps.h`, and `mlir/lib/Dialect/CSLRuntime/IR/CSLRuntimeDialect.cpp` and `CSLRuntimeOps.cpp` (minimal; no custom parse/print). Add `CMakeLists.txt` under `mlir/lib/Dialect/CSLRuntime/IR/`.
5. Register the dialect in `mlir/lib/InitAll.cpp` and `mlir/include/air/InitAll.h`. Add `add_subdirectory(CSLRuntime)` in the parent CMakeLists for include and lib.
6. Add two tests under `mlir/test/Dialect/CSLRuntime/`: `layout_ops.mlir` and `runtime_ops.mlir`, that exercise parsing and printing of the new ops in canonical SSA form. Use `// RUN: air-opt %s | FileCheck %s`.

After implementation, run the new tests and fix any failures. Do not implement emission or lowering yet.

---

## Prompt 2: Phase 2 — Canonical form for CSL dialect

Implement **Phase 2** from `docs/more_docs/CSL_RUNTIME_DIALECT_IMPLEMENTATION_PLAN.md`.

Do the following:

1. In `mlir/include/air/Dialect/CSL/CSLKernelOps.td`, change `csl.func` and `csl.task` so they **do not** use `hasCustomAssemblyFormat`. Define a declarative `assemblyFormat` that produces the same logical syntax (e.g. `%r = csl.func @name () : () -> () { ... }` and `%r = csl.task @name color(3) { ... }`). Ensure operands/results and the `color` attribute are in the format.
2. Remove the custom `parse` and `print` implementations for `FuncOp` and `TaskOp` from `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`. Keep only the include for the generated op definitions.
3. Search the CSL dialect TableGen files for any other op with `hasCustomAssemblyFormat = 1`; if found, replace with an equivalent `assemblyFormat` and remove any corresponding custom parse/print in C++.
4. Run all existing CSL dialect tests (e.g. under `mlir/test/Dialect/CSL/`) and fix any parse/print or verification regressions.

Deliverable: CSL dialect uses only canonical SSA form and declarative assembly; no custom parsers/printers for dialect ops.

---

## Prompt 3: Phase 3 — CSL to csl_rt lowering (minimal path)

Implement **Phase 3** from `docs/more_docs/CSL_RUNTIME_DIALECT_IMPLEMENTATION_PLAN.md`.

Do the following:

1. Create `mlir/lib/Conversion/CSLToCSLRuntime/CMakeLists.txt` and `CSLToCSLRuntime.cpp`.
2. Implement a conversion pass that rewrites **one** `csl.spatial_placement` into a sequence of `csl_rt` ops for the **minimal** case:
   - One `csl.code_region` in the body → `csl_rt.create_code_region` (use kernel source file from the corresponding `csl.place`/kernel if available, else a default like `"pe_program.csl"`).
   - One `csl.place` → `csl_rt.place` with the same (x, y).
   - Each `csl.set_param_all` → `csl_rt.set_param_all` with the same param name and value.
   - Each `csl.export_name` → `csl_rt.export_name` with the same name and type spec.
   - Before these, emit `csl_rt.create_layout`. After export_name, emit `csl_rt.compile`.
   If the body has multiple code regions, or ports/streams/dataflow that require more csl_rt ops, add a **TODO** in the code and do not implement that path (leave the op unconverted or emit a clear error).
3. Register the pass as `csl-to-csl-rt` (or equivalent name in your pass registry). Add it to air-opt.
4. Add a test in `mlir/test/Conversion/CSLToCSLRuntime/`: either `layout_to_runtime.mlir` that runs `air-opt --csl-to-csl-rt` and FileChecks the csl_rt output, or `TODO_lowering.mlir` that documents an unsupported case with a TODO.

Do not implement Python emission in this step.

---

## Prompt 4: Phase 4 — Emit layout.py and run.py from csl_rt

Implement **Phase 4** from `docs/more_docs/CSL_RUNTIME_DIALECT_IMPLEMENTATION_PLAN.md`.

Do the following:

1. Create `mlir/lib/Targets/CSLRuntimeToPy.cpp`. Implement a translation that walks the MLIR module and emits Python code for each `csl_rt.*` op.
2. **Layout emission:** For `csl_rt.create_layout`, emit `layout = SdkLayout(platform)` (or `SdkLayout()` for standalone layout.py). For `create_code_region`, emit `layout.create_code_region(source, name, width, height)`. For `place`, emit `region.place(x, y)`. For `set_param_all`, emit `region.set_param_all(name, value)`. For `export_name`, emit `layout.export_name("name", "type_spec")`. For `compile`, emit `compile_artifacts = layout.compile(out_prefix='out')`. Match the [SdkLayout API](https://sdk.cerebras.net/api-docs/sdklayout-api).
3. **Run emission:** For `csl_rt.runtime_create`, emit construction of `SdkRuntime(compile_artifacts, platform, ...)`. For `load`, `run`, `stop`, emit `runtime.load()`, `runtime.run()`, `runtime.stop()`. For `get_id`, emit `runtime.get_id("symbol_name")`. For `memcpy_h2d` and `memcpy_d2h`, emit the corresponding Python calls with **placeholders** for host arrays (e.g. comment or variable name like `# data = ...`). For `launch`, emit `runtime.launch("name", nonblock=False)`. Match the [SdkRuntime API](https://sdk.cerebras.net/api-docs/sdkruntime-api).
4. Emit a **run.py template**: include argparse, platform creation, and a clear section where user fills in data for memcpy and where to read results. Structure should be compatible with GEMV-05 style (get_id, memcpy_h2d, launch, memcpy_d2h, stop).
5. Register the translation in `tools/air-translate/air-translate.cpp` (e.g. `--emit-csl-rt` or `--csl-rt-emit`) with an output directory option.
6. Add FileCheck tests: `mlir/test/Target/emit_csl_rt_layout_py.mlir` and `emit_csl_rt_run_py.mlir` that run the translator and check the emitted layout.py and run.py snippets.

Use the exact file names from the implementation plan. Keep the emitted API subset small and stable (only the ops listed in the plan).

---

## Prompt 5: Phase 5 — GEMV-05 end-to-end and spec doc

Implement **Phase 5** from `docs/more_docs/CSL_RUNTIME_DIALECT_IMPLEMENTATION_PLAN.md`.

Do the following:

1. Add `mlir/test/Dialect/CSLRuntime/gemv05_example.mlir`: a single MLIR module that encodes the **structure** of [GEMV 5: Multiple PEs](https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-05-multiple-pes): one layout with one code region, dimensions and placement, set_param_all for width/M/N (or equivalent), export_name for A, x, b, y, compute; then runtime load, run, get_id for A/x/b/y, memcpy_h2d for A/x/b, launch compute, memcpy_d2h for y, stop. Use only `csl_rt` ops (no CSL layout ops in this file). Kernel .csl can be referenced by the code region source path.
2. Run the translator on this file and ensure it produces layout.py and run.py that are consistent with the SdkLayout and SdkRuntime APIs. Add or extend FileCheck tests so this example is regression-tested.
3. Create `docs/more_docs/CSL_RUNTIME_DIALECT_SPEC.md` with: (a) list of all `csl_rt` types and ops, (b) canonical assembly form for each op, (c) mapping of each op to the corresponding SdkLayout or SdkRuntime API call (with links to https://sdk.cerebras.net/api-docs/sdkruntime-api and https://sdk.cerebras.net/api-docs/sdklayout-api). Note that run.py is intended as a **template** (user fills data) and that layout is generated using the **Python SdkLayout API**, not layout.csl.

---

## One-shot prompt (all phases)

If you prefer to give a single prompt that covers the full plan:

---

Implement the **CSL Runtime Dialect** according to `docs/more_docs/CSL_RUNTIME_DIALECT_IMPLEMENTATION_PLAN.md` in order:

1. **Phase 1:** Create the `csl_rt` dialect (CSLRuntime) with types and minimal ops in `mlir/include/air/Dialect/CSLRuntime/` and `mlir/lib/Dialect/CSLRuntime/IR/`. All ops must use declarative `assemblyFormat` only. Add parse/print tests under `mlir/test/Dialect/CSLRuntime/`.
2. **Phase 2:** Convert CSL `csl.func` and `csl.task` (and any other CSL ops with custom asm) to use only declarative `assemblyFormat`; remove custom parse/print from `CSLOps.cpp`. Ensure all CSL tests still pass.
3. **Phase 3:** Add the conversion pass CSL → csl_rt in `mlir/lib/Conversion/CSLToCSLRuntime/` for the minimal path (one spatial_placement, one code_region, place, set_param_all, export_name → create_layout, create_code_region, place, set_param_all, export_name, compile). Add TODO for non-trivial cases. Register pass and add test.
4. **Phase 4:** Add `CSLRuntimeToPy.cpp` in `mlir/lib/Targets/` to emit layout.py (SdkLayout API) and run.py (SdkRuntime API) from csl_rt ops. run.py must be a template with placeholders for host data. Register `--emit-csl-rt` in air-translate. Add FileCheck tests for emitted output.
5. **Phase 5:** Add `gemv05_example.mlir` under `mlir/test/Dialect/CSLRuntime/` encoding GEMV-05 structure with csl_rt ops; verify emission. Write `docs/more_docs/CSL_RUNTIME_DIALECT_SPEC.md` with op list, canonical form, and API mapping.

Use the exact file names and op names from the implementation plan. Reference SdkRuntime API (https://sdk.cerebras.net/api-docs/sdkruntime-api) and SdkLayout API (https://sdk.cerebras.net/api-docs/sdklayout-api) for the minimal subset. No custom parsers/printers in either dialect.

---

End of prompts.
