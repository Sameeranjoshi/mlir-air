# AIR → CSL Backend, Vecadd Milestone — Design Spec

**Date:** 2026-04-13
**Status:** Approved during brainstorming; ready for implementation planning.
**Branch context:** `air-to-fire`. Five existing CSL design docs in `docs/more_docs/` provide background; this spec supersedes their phased plan for the *next milestone only*.

---

## 1. Goal

Compile a single-PE 256-element f32 vecadd from an AIR program through the CSL dialect to a runnable Cerebras program, and execute it on the CS-3 attached to this machine, returning the correct array to the host.

This is the **smallest end-to-end** that exercises every layer of the intended pipeline. It is the "definition of done" milestone for the `AIR → csl.* → csl_rt.* → CSL + Python → CS-3` path the project has been documenting but not yet wiring together.

## 2. Definition of Done

A single test passes:

`test/csl/test_vecadd_e2e.py::test_vecadd_end_to_end`

The test invokes:

```bash
air-opt mlir/test/Conversion/AIRToCSL/vecadd.mlir \
        -air-to-csl-dialect -csl-to-csl-rt | \
  air-translate --emit-csl-rt -o $OUT_DIR
cs_python $OUT_DIR/run.py --arch wse3 --check
```

…on CS-3, and asserts:

1. Both `$OUT_DIR/vecadd_pe.csl` and `$OUT_DIR/run.py` exist.
2. `cs_python run.py` exits with code 0.
3. The script's `np.allclose(c, a + b, atol=0, rtol=0)` validator prints `PASS`.

The test runs unconditionally on this machine. It does not run in CI environments without CS-3 (concerns about CI gating are out of scope for this milestone).

## 3. Scope

### In scope

- Single PE (1×1 herd, 1×1 code region).
- One AIR kernel: 256-element vecadd via `scf.for` over `memref<256xf32>` + `arith.addf`.
- Three host-visible buffers: `a`, `b` (inputs), `c` (output).
- One host-callable kernel function (`compute`).
- Hand-written CSL fixtures used as **golden files** for comparison, then matched textually by the compiler-generated output.

### Explicitly out of scope (deferred to follow-up milestones)

- Multi-PE grids and the spatial mapping logic that comes with them.
- Inter-PE communication: `air.channel.put`/`get`, colors, routes, `csl.dataflow`.
- DSDs (Data Structure Descriptors) and bulk memory operations.
- `air.dma_memcpy_nd` lowering.
- More kernels (mul, sub, reductions, etc.). Adding them is one row in a dispatch table per kernel; not in this milestone.
- Performance. The milestone is correctness only.
- The "TDF / spatial common dialect" multi-backend abstraction discussed in `docs/more_docs/design_air_to_csl.md` §4.

The scope wall is enforced by the compiler itself: each layer rejects out-of-scope inputs with a precise error (see §7).

## 4. Disposition of Existing Code

Three pre-existing pieces of code interact with this milestone. Each gets a clear disposition.

| Piece | Path | Disposition |
|---|---|---|
| **Phase-1 direct text emitter** | `mlir/lib/Conversion/AIRToCSLPass.cpp` (748 LOC) | Two steps. **Step A (inside this milestone):** stop registering `-air-to-csl=...` from `air-opt` so the option name is freed for the new pass. The .cpp file stays in tree, unbuilt, until step B. **Step B (immediately after the milestone hardware test passes):** move the file to `archived_code/`. The file's own header comment already labels it "kept for reference and later use." |
| **CSL → CSL Runtime lowering pass** | `mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp` (193 LOC, 1 trivial test) | Extend incrementally. The existing patterns (`spatial_placement → create_layout`, `code_region → create_code_region`, `place → place`, `compile`) are correct; we add the missing patterns (`set_param_all`, `export_name`, runtime sequence). No rewrite. |
| **csl_rt → Python translator** | `mlir/lib/Targets/CSLRuntimeToPy.cpp` (377 LOC) | Substantial rework. The host-side `build_layout()` skeleton stays; the `emitKernelPrograms()` stub is replaced by a real `KernelEmitter` (see §5.3); the `__main__` skeleton gains argparse + numpy buffer materialization + validation. |

The fourth piece — the `air-to-csl-dialect` pass — does not exist and is the central new artifact of this milestone (§5.1).

## 5. Architecture & Components

### 5.1 Pipeline

```
                vecadd.mlir  (AIR + standard MLIR)
                       │
                       │  air-opt -air-to-csl-dialect      ← NEW PASS
                       ▼
                csl.* IR  (csl.spatial_placement +
                           csl.code_region +
                           csl.kernel containing csl.func
                           with arith / scf / memref ops)
                       │
                       │  air-opt -csl-to-csl-rt            ← EXTEND existing
                       ▼
                csl_rt.* IR  (create_layout, create_code_region,
                              place, set_param_all, export_name,
                              compile, runtime_*, memcpy_*, launch)
                       │
                       │  air-translate --emit-csl-rt       ← REWORK existing
                       ▼
        ┌──────────────┴──────────────┐
        ▼                              ▼
   pe_program.csl              run.py
        │                              │
        └──────────┬───────────────────┘
                   ▼
            cs_python run.py  →  CS-3  →  validates and exits 0
```

### 5.2 Architectural commitments

1. **Every arrow is a real MLIR/translation step.** No string templates anywhere except where CSL syntax is finally written to disk. The Phase-1 pattern of "string-builder masquerading as a pass" is explicitly rejected.
2. **`csl.kernel` body holds standard MLIR ops, not csl-specific compute ops.** The herd compute body (with `arith`, `scf`, `memref`) is preserved verbatim into a `csl.func @compute()`. We do not introduce `csl.add`, `csl.for`, `csl.constant`, etc. This means MLIR canonicalization and CSE work for free, and adding new arithmetic ops is one entry in a dispatch table — not a new IR op.
3. **The compile-time → runtime handoff is two files in a directory.** `vecadd_pe.csl` and `run.py`, with no other state. Either file can be hand-edited or hand-written to test the other half independently.
4. **The Kernel Emitter is the single place where MLIR semantics meet CSL syntax.** All MLIR-op-to-CSL-text translation logic lives in one component, with a small explicit dispatch table.
5. **Fail loud, no half-outputs, no silent fallbacks.** Each layer rejects unsupported inputs with a precise error citing the source location. Translators do not write partial files.

### 5.3 Components

#### 5.3.1 `air-to-csl-dialect` pass (NEW)

- **Path:** `mlir/lib/Conversion/AIRToCSLDialect/AIRToCSLDialect.cpp` (+ header in `mlir/include/air/Conversion/`)
- **Pass option:** `-air-to-csl-dialect`
- **Input:** A module containing `func.func` enclosing `air.launch` / `air.segment` / `air.herd` nests, plus standard MLIR.
- **Output:** Same module with the AIR nests replaced by `csl.spatial_placement` containing `csl.code_region`, `csl.kernel`, `csl.place`, and surrounding `csl.export_name` ops at the host level. The herd compute body is moved verbatim into a `csl.func @compute()` inside the kernel.

##### IR contract (the single most important artifact in this design)

For the milestone vecadd, the post-pass IR has the structure below. Op syntax fragments are **illustrative**; the exact custom assembly forms come from the existing `CSLOps.td` / `CSLRuntimeOps.td` definitions, which the implementation must conform to. The structural commitments — what ops appear, their nesting, the data they carry — are authoritative.

```mlir
module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    csl.spatial_placement {
      %k = csl.kernel {
        csl.var @a_buf : memref<256xf32>
        csl.var @b_buf : memref<256xf32>
        csl.var @c_buf : memref<256xf32>

        csl.func @compute() : () -> () {
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

        csl.comptime {
          csl.export_symbol @a_buf alias("a")
          csl.export_symbol @b_buf alias("b")
          csl.export_symbol @c_buf alias("c")
          csl.export_symbol @compute
        }
      } {source_file = "vecadd_pe.csl"} : !csl.kernel

      %r = csl.code_region routes() colors() {
      } {width = 1 : i64, height = 1 : i64} : !csl.code_region

      csl.place %r %k {x = 0 : i64, y = 0 : i64}
    }

    csl.export_name "a" : memref<256xf32> {direction = "in"}
    csl.export_name "b" : memref<256xf32> {direction = "in"}
    csl.export_name "c" : memref<256xf32> {direction = "out"}
    csl.export_name "compute" : () -> ()

    return
  }
}
```

The `direction` attribute on `csl.export_name` is **new in this milestone**: the existing op declares a host-visible name with a type, but does not distinguish input from output buffers. The lowering needs that information to decide whether to emit `memcpy_h2d` or `memcpy_d2h`. Adding the attribute to `CSLRuntimeOps.td` is part of this milestone's work.

Each downstream layer's correctness is defined by what it produces *given this exact input*. Each downstream layer can be tested independently by hand-writing this IR.

##### What this pass does NOT do

- Does not optimize the herd body. (Canonicalization runs separately upstream if needed.)
- Does not handle `air.channel`, `air.dma_memcpy_nd`, multi-PE layouts, async tokens.
- Does not generate any text. Output is parseable, verifiable MLIR.

#### 5.3.2 `csl-to-csl-rt` pass extension

- **Path:** `mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp`
- **Input:** the IR shape above.
- **Output:** the same module with `csl.spatial_placement` replaced by a sequence of `csl_rt.*` ops, plus the host-side `func.func @vecadd` body filled with the runtime sequence.

##### New patterns added

1. `csl.var` inside a kernel → recorded in a SymbolTable so `export_symbol` references resolve; nothing emitted at the runtime level.
2. `csl.export_symbol` inside `csl.comptime` → contributes to a per-region exported-names map; no `csl_rt` op (the symbols are exported by the .csl file itself).
3. `csl.export_name` at the host level → `csl_rt.export_name` on the layout, encoding both the buffer name and a CSL type-spec string derived from the memref type. The exact string format (e.g. `<f32>[256]`, or `f32` plus a separate shape attribute) is determined during implementation by inspecting how the existing `csl_rt.export_name` op is consumed by the SDK Python layer; the spec assumes the op gains a shape attribute or its existing string-type attribute is extended.
4. **Runtime sequence synthesis:** driven by the `direction` attribute on each `csl.export_name`. The synthesized sequence in the host `func.func` body is:
   - `csl_rt.runtime_create`
   - `csl_rt.load`
   - `csl_rt.memcpy_h2d` for each `direction = "in"` buffer (in declaration order)
   - `csl_rt.launch` for the host-callable function (the `() -> ()` export)
   - `csl_rt.memcpy_d2h` for each `direction = "out"` buffer
   - `csl_rt.stop`

`csl_rt.set_param_all` is **not** used in this milestone. The vecadd kernel hardcodes `N = 256` as a literal in the generated CSL, so no host-to-kernel parameter passing is needed. The `set_param_all` op stays in the dialect for future milestones; the lowering pass simply doesn't generate it.

##### Out of scope for this extension

Ports, streams, color/route lowering. Stay as TODOs in the existing pattern set with clear errors if encountered.

#### 5.3.3 `--emit-csl-rt` translator (rework)

- **Path:** `mlir/lib/Targets/CSLRuntimeToPy.cpp`
- **Input:** the post-`csl-to-csl-rt` IR.
- **Output:** two files in the directory passed via `-o`:
  - `<source_file>` (CSL kernel source, name from `csl.kernel { source_file = ... }`)
  - `run.py` (host-side Python script)

##### KernelEmitter sub-component

A struct that walks one `csl.kernel` op and writes one `.csl` file:

```cpp
LogicalResult emitKernel(csl::KernelOp kernel, llvm::raw_ostream &os);
```

Internally:

- **Name table** (`llvm::DenseMap<Value, std::string>`): binds MLIR SSA values to CSL identifiers. Block arguments and `csl.var`s get stable names from their op attributes; intermediate `arith` results get auto-generated names (`t0`, `t1`, …).
- **Op dispatch table**, indexed by op name. Each entry is a small function that emits CSL syntax for one op kind.

##### Op dispatch table for the milestone

| MLIR op | CSL syntax produced |
|---|---|
| `csl.var @x : memref<NxT>` | `var x: [N]T;` |
| `csl.func @f() { ... }` | `fn f() void { <body> }` |
| `csl.return` | (function-end marker; no text) |
| `csl.comptime { csl.export_symbol @x alias("y") }` | `comptime { @export_symbol(x, "y"); }` (or 2-arg form for fn) |
| `arith.constant N : index` | bound to name; emitted inline at use site as the literal |
| `arith.constant V : f32` | bound to name; emitted inline |
| `scf.for %i = %lo to %hi step %s { ... }` | `for (@range(i32, <hi>)) \|<i>\| { <body> }` (when lo=0, step=1) |
| `memref.load %buf[%i]` | `<buf>[<i>]` (RHS in an assignment) |
| `arith.addf %a, %b` | `<a> + <b>` |
| `memref.store %v, %buf[%i]` | `<buf>[<i>] = <v>;` |

Seven patterns. Each precondition (e.g. `scf.for` requires `lo == 0`, `step == 1`, constant `hi`) is checked before emitting.

##### HostEmitter sub-component

Walks `csl_rt.*` ops in the host-side `func.func` and writes `run.py`. Starts from the existing 377 LOC's `build_layout()` scaffolding where it carries weight; the implementation should read it carefully early and rewrite the parts that don't (the kernel-emission stub is replaced wholesale by the KernelEmitter above). Required functional changes:

- Removes the stub `emitKernelPrograms()` (the KernelEmitter writes the `.csl` file separately).
- Adds buffer materialization for `memcpy_h2d/d2h`: numpy arrays sized from the `csl_rt.export_name` type annotations, populated with deterministic test data (`np.arange` for inputs, zeros for outputs).
- Adds a `__main__` guard with `argparse` for `--cmaddr`, `--arch`, `--check`.
- After `memcpy_d2h`, emits a `np.allclose(c, a + b, atol=0, rtol=0)` check that prints `PASS` and exits 0 on success, prints first 8 mismatches and exits 1 on failure.

## 6. Data Flow

### 6.1 Compile time

One module, three passes, each strictly extending or replacing what the previous produced. No back-references, no cross-pass state.

```
vecadd.mlir        air-to-csl-dialect    csl-to-csl-rt    --emit-csl-rt
─────────────────  ──────────────────    ─────────────    ─────────────
func.func @vecadd  csl.spatial_placement csl_rt.create_*  vecadd_pe.csl
  air.launch         csl.kernel            csl_rt.compile run.py
    air.segment       (csl.func, csl.var,  csl_rt.runtime_*
      air.herd         csl.export_symbol)  csl_rt.memcpy_*
        scf.for      csl.code_region       csl_rt.launch
          arith        csl.place           csl_rt.stop
          memref       csl.export_name
```

### 6.2 Run time

```
cs_python run.py
   1. parse args (--cmaddr, --arch, --check)
   2. build numpy host buffers (a, b deterministic; c zeros)
   3. build_layout(platform):
        layout = SdkLayout(platform)
        region = layout.create_code_region('vecadd_pe.csl', 'vecadd', 1, 1)
        region.place(0, 0)
        layout.export_name('a', '<f32>[256]')   # ×3 buffers + 1 fn
        return layout.compile(out_prefix='out')
   4. runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=True)
      runtime.load()
   5. memcpy_h2d for a and b
   6. runtime.launch('compute')                 (blocks)
   7. memcpy_d2h for c
      runtime.stop()
   8. assert np.allclose(c, a + b, atol=0, rtol=0); print 'PASS'; exit 0
```

The compile-time → run-time handoff is **two files in a directory** with no other state.

## 7. Error Handling

### 7.1 Principle

Fail loud at the earliest layer that can detect the problem, with a source location. Translators never write half-files. Compilers never silently skip ops they don't understand.

### 7.2 `air-to-csl-dialect` rejections

Emit `op->emitOpError(...)` and return `failure()` for:

| Detected | Error message |
|---|---|
| `air.herd` with `hsx > 1` or `hsy > 1` | `air-to-csl-dialect: only 1×1 herds supported in this milestone (found %dx%d)` |
| `air.channel.put` / `air.channel.get` | `air-to-csl-dialect: inter-PE channels not yet supported` |
| `air.dma_memcpy_nd` inside the herd body | `air-to-csl-dialect: dma_memcpy_nd not yet supported in herd bodies; use scalar load/store` |
| Memref with dynamic dims or unranked | `air-to-csl-dialect: kernel memrefs must be statically shaped (got %T)` |
| Memref element type not in {f32, f16, i32, i16} | `air-to-csl-dialect: unsupported element type %T (supported: f32 f16 i32 i16)` |
| `air.execute` / async tokens in the herd body | `air-to-csl-dialect: async operations not supported in herd bodies for CSL backend` |
| Multiple `air.herd`s in one launch | `air-to-csl-dialect: multiple herds not yet supported (split into separate functions)` |

### 7.3 `csl-to-csl-rt` rejections

- `csl.code_region` with non-empty `routes()` or `colors()` → `routing not supported in milestone`
- `csl.spatial_placement` with more than one `csl.code_region` → `single-region only`
- `csl.place` referencing a kernel not in the same `spatial_placement` → `place must reference local kernel`

The pre-existing pass has at least one place that silently skips unhandled ops. That gets fixed as part of this milestone.

### 7.4 `--emit-csl-rt` translator rejections

- KernelEmitter walks the kernel body **before** opening the output file. If any op isn't in the dispatch table, it aborts with `unsupported MLIR op for CSL kernel emission: <op name> at <loc>` and does not create a partial `.csl` file.
- Each dispatch-table entry has a precondition check (e.g. `scf.for` requires `lo == 0`, `step == 1`, constant `hi`); precondition violations are precise errors.
- HostEmitter has the same policy for `csl_rt.*` ops it doesn't recognize.
- On any abort, the translator returns `failure()` and exits nonzero. No half-written outputs.

### 7.5 Runtime errors (in generated `run.py`)

- `argparse` validates `--arch` ∈ {wse2, wse3}.
- `SdkLayout.compile()` exceptions propagate (the SDK's diagnostics are clear).
- `runtime.get_id('name')` raising means the kernel's `@export_symbol` doesn't match the host's `export_name`; the script catches it and prints both sides.
- `np.allclose` failure prints first 8 mismatching indices with both sides and exits 1.
- Exit code reflects pass/fail.

### 7.6 What we don't wrap

- **CSL compiler errors** (`cslc` rejecting our generated `.csl`) — surface them via the SDK's own diagnostic stream. If `cslc` says "syntax error at line 42," the KernelEmitter has a bug; look at the kernel file by hand. The unit test suite's golden-file comparison should catch these in CI before they hit hardware.
- **CS-3 runtime errors** (`SdkRuntime.run()` failing) — propagate the SDK exception. Most are environmental.

## 8. Testing

### 8.1 Test pyramid

```
                     ┌──────────────────────────────┐
                     │  test_vecadd_e2e.py          │   1 test, runs on CS-3
                     │  (full pipeline, real hw)    │   "definition of done"
                     └──────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
 vecadd_kernel_emit.mlir  vecadd_with_kernel.mlir  vecadd_dialect.mlir
 (csl.* → CSL text)       (csl.* → csl_rt.*)       (AIR → csl.*)
 FileCheck on .csl        FileCheck on csl_rt IR   FileCheck on csl.* IR
                                    │
                                    ▼
                    layer-rejection unit tests
                    (one .mlir per error message in §7)
```

### 8.2 Bootstrap golden files (Day 1)

**Before any compiler code is written**, hand-write `vecadd_pe.csl` and `run.py` that match what the compiler will eventually produce. Verify they run on CS-3 with the SDK directly. Save as:

- `mlir/test/Conversion/AIRToCSL/golden/vecadd_pe.csl.golden`
- `mlir/test/Conversion/AIRToCSL/golden/run.py.golden`

Two purposes:
1. Removes ambiguity about what valid CSL for this exact program looks like *before* writing any compiler code.
2. Removes "is the bug in my code or the SDK?" from every subsequent debug session. If the golden files run, the SDK is fine; any later failure is in the compiler.

The unit tests in §8.3 FileCheck against patterns extracted from the golden files rather than typing them by hand, keeping tests maintainable.

### 8.3 Per-layer unit tests (lit + FileCheck)

| Test | Input | Asserts |
|---|---|---|
| `mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir` | A 1×1 vecadd AIR program | The post-pass module matches the IR shape from §5.3.1 |
| `mlir/test/Conversion/CSLToCSLRuntime/vecadd_with_kernel.mlir` | Hand-written copy of §5.3.1 IR | `create_layout`, `create_code_region`, `place`, three `export_name`s, `compile`, `runtime_create`, `load`, two `memcpy_h2d`s, `launch "compute"`, one `memcpy_d2h`, `stop` |
| `mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir` | Hand-written `csl.kernel` op | FileCheck patterns (drawn from the golden file) verify each line of the generated `vecadd_pe.csl` |
| `mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir` | Hand-written `csl_rt.*` sequence | Generated `run.py` has imports, `build_layout()`, `runtime_create`, `memcpy` calls with right region args, `launch('compute')`, `stop()`, validator block |

The hand-written inputs in tests 2-4 mean each layer's test passes independently — a broken upstream layer cannot mask a working downstream one or vice versa.

### 8.4 Layer-rejection tests

`mlir/test/Conversion/AIRToCSLDialect/reject_*.mlir` — one tiny file per failure mode in §7.2, each with `// expected-error @below` annotations. Examples:

- `reject_2x2_herd.mlir` → expects `"only 1×1 herds supported"`
- `reject_channel_op.mlir` → expects `"inter-PE channels not yet supported"`
- `reject_dynamic_memref.mlir` → expects `"kernel memrefs must be statically shaped"`
- (one per row in §7.2 table)

Same pattern for `csl-to-csl-rt` rejections (§7.3) and KernelEmitter rejections (§7.4). These are cheap (~5 lines each), they're the project's safety net against accidental scope expansion.

### 8.5 Hardware integration test

`test/csl/test_vecadd_e2e.py`:

```python
import os, subprocess, pytest
from pathlib import Path

def test_vecadd_end_to_end(tmp_path):
    src = "mlir/test/Conversion/AIRToCSL/vecadd.mlir"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # 1. compile
    subprocess.run(
        f"air-opt {src} -air-to-csl-dialect -csl-to-csl-rt | "
        f"air-translate --emit-csl-rt -o {out_dir}",
        shell=True, check=True,
    )
    assert (out_dir / "vecadd_pe.csl").exists()
    assert (out_dir / "run.py").exists()

    # 2. run on CS-3
    result = subprocess.run(
        ["cs_python", str(out_dir / "run.py"), "--arch", "wse3", "--check"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"vecadd run.py failed on CS-3:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # 3. internal "PASS" sentinel from run.py's validator
    assert "PASS" in result.stdout
```

This runs unconditionally on this machine. CS-3 is here; the SDK is installed; there's no gate.

### 8.6 What is not tested

- Performance.
- Numerical edge cases (NaN, denormals, overflow). The kernel is `a + b` with `a, b ∈ np.arange(256)`; no edge cases reachable.
- Robustness to bad input flags or environment misconfiguration. The Python script crashes with the SDK's native error.
- Anything past `vecadd`. New kernels are follow-up commits, not this milestone.

## 9. Milestone Boundaries

### 9.1 What this milestone ships

- A new `direction` attribute on `csl.export_name` (TableGen + parser/printer).
- `air-to-csl-dialect` pass that produces the §5.3.1 IR shape from a 1×1 vecadd AIR program.
- Extended `csl-to-csl-rt` lowering covering `csl.export_name` (with `direction`) → `csl_rt.export_name`, plus full host-side runtime sequence synthesis (`runtime_create`, `load`, `memcpy_h2d` per input, `launch`, `memcpy_d2h` per output, `stop`).
- Reworked `--emit-csl-rt` translator with a real KernelEmitter (7-entry dispatch table) and a runnable `run.py` HostEmitter.
- Bootstrap golden files (`vecadd_pe.csl.golden`, `run.py.golden`) verified to run on CS-3 directly.
- Layer unit tests + layer-rejection unit tests + one hardware integration test.
- Step A of Phase-1 emitter quarantine: `-air-to-csl=...` no longer registered. (Step B — moving the file to `archived_code/` — happens immediately after the hardware test passes, as a final commit.)

### 9.2 What this milestone does NOT ship (deferred to follow-ups, in this rough order)

1. More elementwise kernels (mul, sub, fma, sqrt) — adds dispatch-table entries.
2. Multi-PE 1D layouts (e.g. 4×1) with embarrassingly parallel input partitioning.
3. Reductions on a single PE.
4. Inter-PE communication via `csl.dataflow` (colors, routes, send/recv).
5. DSDs and bulk memory operations.
6. Real GEMV (multi-PE, broadcast input vector, partial-sum reductions).
7. Performance tuning.
8. The TDF/spatial common dialect for multi-backend reuse.

Each follow-up has a clear entry path through this milestone's architecture: most of them only touch the dispatch table and the rejection list.

## 10. Open Questions (minor, not blocking)

- **`cs_python` PATH sourcing:** the test invocation assumes `cs_python` is on PATH. Confirmed via `which cs_python` → `/home/bricklib_dataflow/sdk/SDK_1_4/cs_python`. If a future machine doesn't have it, the test will fail loudly with a clear `command not found`, which is the right behavior.
- **Default `--arch`:** the test passes `--arch wse3` because that matches CS-3. If CS-2 access is added later, the test should accept either via env var.
- **Where the source lit test lives:** `mlir/test/Conversion/AIRToCSL/vecadd.mlir` is reused by both the lit unit test and the Python integration test. Acceptable given it's the canonical input. If they diverge, split into separate fixtures.

## 11. Out of Scope for This Spec

- Implementation plan. This spec is the *what* and *why*. The *how* and *in what order* is the job of the writing-plans skill, which is invoked next.
- Estimates. Implementation plan will produce them.
- CI integration. The test runs locally; CI is a separate concern.

## 12. Approval

All five sections (Architecture, Components, Data Flow, Error Handling, Testing) were presented to the user and approved during the brainstorming session on 2026-04-13.

Notable decisions made by the user (not by the brainstorming agent's recommendation alone):

- **Approach 2 over Approach 1**: generate CSL from the IR rather than use a hand-written kernel template. The brainstorming agent had recommended Approach 1 (template + structural pipeline first, generation as a follow-up). The user explicitly chose Approach 2, citing the prior project's bad experience with string-based emission and a desire to do the kernel emission "properly" from the start. The spec reflects Approach 2.
- **Hardware test runs unconditionally**: no `CS3_AVAILABLE` gate. The brainstorming agent had proposed an environment-variable gate. The user said "I mean run on hardware definitely" — the test runs every time, full stop.
