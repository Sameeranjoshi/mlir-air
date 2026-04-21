# `-csl-auto-vectorize` — Loop-Idiom Recognition for CSL DSD Operations

**Date:** 2026-04-21
**Status:** Design (ready for plan)
**Branch:** `air-to-fire`
**Extends:** CSL dialect v5 (see `2026-04-18-csl-v5-single-pe-simd.md`) — consumes the CSL IR that `-air-to-csl` produces, emits more of the same CSL IR.
**References:**
- SPADA paper, §VI.D Automatic Vectorization (`docs/superpowers/raw/papers/SPADA.pdf`) — tiered fallback strategy, purity predicate.
- LLVM `LoopIdiomRecognize` (`llvm/lib/Transforms/Scalar/LoopIdiomRecognize.cpp`) — the idiom-per-pattern structural model.
- MLIR Affine SuperVectorize (`mlir/lib/Dialect/Affine/Transforms/SuperVectorize.cpp`) — reference affine-access analysis (not used directly; we do hand-rolled analysis for MVP).
- CSL language — DSD builtins: https://sdk.cerebras.net/csl/language/builtins and https://sdk.cerebras.net/csl/language/dsds
- CSL language appendix — SIMD mode: https://sdk.cerebras.net/csl/language/appendix#language-appendix-simd
- Existing CSL dialect — `mlir/include/air/Dialect/CSL/CSLOps.td` (`csl.get_mem_dsd`, `csl.builtin_call`), hand-written DSD example in `mlir/test/Targets/CSLEmit/e2e/dsds.mlir`.

**Design rule:** this is a CSL-dialect optimization pass. It consumes CSL IR, produces CSL IR, and must be *non-destructive* — any loop that fails the legality predicate is left untouched (SPADA Tier-4 fallback).

---

## 1. Goal

Automatically convert scalar `scf.for`-based element-wise loops inside `csl.func` bodies into DSD-based CSL vector-builtin calls, wherever doing so is provably semantics-preserving. Users write ordinary MLIR scalar loops; the pass produces `csl.get_mem_dsd` + `csl.builtin_call` form where the hardware accelerates it.

### In scope (first cut — "S4 scope" per brainstorm)

- **Rank:** `memref<NxT>` (rank-1, `mem1d_dsd`) and `memref<MxNxT>` (rank-2, `mem4d_dsd`).
- **Extent:** static (constant loop bounds and constant memref shape).
- **Stride:** unit stride directly, plus constant non-unit stride via `memref.subview` with `strided<[c], offset: k>` layout.
- **Index:** affine functions of the induction var — `%i`, `%i + k` (k any signed constant including negative), `%i * c + k`. Access-in-bounds checked; OOB → reject.
- **Loops:** single-block body, no `iter_args`, no `scf.if` / nested `scf.for` / function calls inside body (rank-2 is the exception: perfect nest of two `scf.for`s).
- **Element types (T1):** `f32` only. Single-precision float is the universal DSD arithmetic path on WSE hardware. Integer widths are explicitly deferred — see §9.2. Rationale: CSL DSDs are 16-bit-native on the integer side (`@add16`, `@sub16`, `@mov16`, `@mov32`, no `@...32` arith and no `@mul16`), so any honest integer support is an i16 story, not an i32 story. The typical MLIR input uses i32 arithmetic which **cannot** be vectorized to DSDs regardless of scope — it would always fall through to scalar. Shipping "integer support" before the workload is known (i16 kernels vs i32 kernels) risks committing to the wrong path.
- **Idiom set (tier-1, scope D):** elementwise binary (add/sub/mul) + FMA fusion + copy/unary (mov/neg) + scalar-broadcast operand.

### Explicitly out of scope

- Tier-2 `@map` lowering for pure-but-non-idiom loops. Requires a new CSL dialect op and closure-lowering machinery. Follow-up pass.
- Tier-1 scope E — reductions with loop-carried `iter_args` (`norm_sq`, `dot`). Different legality shape; separate pass.
- Fabric DSDs (`fabin_dsd` / `fabout_dsd`) with color triggers. Per the existing Phase-1 boundary, inter-PE communication is out.
- Half-precision (`f16`) and narrow-int (`i16`) element types.
- Non-unit-step loops (`scf.for %i = 0 to N step 2`). Need SDK-semantic confirmation for DSD stride under non-unit IV step.
- Non-perfect rank-2 nests (code between outer and inner loop).

---

## 2. Motivation and prior art

`-air-to-csl` today emits CSL IR in which `scf.for` / `memref.load` / `memref.store` / `arith.*` flow straight through to the text emitter (`mlir/lib/Targets/CSLEmit/CSLEmitCommon.h:326-347`), which prints them as CSL `while` loops. The pipeline never lowers SCF. This is a correctness-preserving but performance-losing default: the Cerebras WSE hardware has dedicated vector-descriptor (DSD) operations (`@fadds`, `@fmacs`, `@fmuls`, …) that execute a whole memory range per instruction, and the current pipeline never invokes them automatically. The only way to get DSD form today is to hand-write `csl.get_mem_dsd` / `csl.builtin_call` in the MLIR input (as `mlir/test/Targets/CSLEmit/e2e/dsds.mlir` does).

**Prior art consulted:**
- **LLVM `LoopIdiomRecognize`** matches loops against a curated table of idioms (memset, memcpy, popcount, …) and replaces them with intrinsic calls. We adopt the structure verbatim: one pattern class per idiom, shared legality analysis up front.
- **SPADA §VI.D** describes exactly this problem on exactly this target. Their pattern-match predicate is *pure body, indexing-only iterator usage, no control flow* — we adopt this predicate verbatim. SPADA names a tiered fallback strategy; we implement Tiers 1 and 4 (DSD-match; scalar-fall-through) in this pass, deferring Tier 2 (`@map`) to a follow-up.
- **MLIR Affine SuperVectorize** is the closest MLIR analogue but targets the `vector` dialect's SIMD-register semantics. DSDs are strided memory descriptors, not SIMD registers, so we bypass `vector` (see brainstorm). Affine-access analysis utilities from upstream MLIR are still useful as an *optional* legality cross-check; MVP uses a hand-rolled matcher.

---

## 3. Architecture

### 3.1 Pass identity

```
Name:        -csl-auto-vectorize
Registers:   createCSLAutoVectorizePass()
Operates on: ModuleOp
Dialects:    scf, memref, arith, csl (declared in getDependentDialects)
Runs:        between -air-to-csl (producer of CSL IR with scalar loops) and
             -csl-infer-exports (annotation pass) and air-translate --emit-csl.
```

Canonical RUN line:
```
air-opt %s -csl-auto-vectorize -csl-infer-exports | air-translate --emit-csl --output-dir=%t
```

### 3.2 Pipeline ordering

```
air-opt:
  -air-to-csl           (already exists; scalar loops survive)
  -csl-auto-vectorize   (NEW — rewrites legalized loops to DSDs)
  -csl-infer-exports    (already exists; pure annotation)
air-translate:
  --emit-csl            (already exists; handles both DSD and scalar forms)
```

`-csl-auto-vectorize` is **opt-in per invocation**. `aircc.py` is updated to include it in its default CSL pipeline (same commit as the pass); there is no `--no-auto-vectorize` flag — users who want the pre-pass IR run `air-opt` without the pass.

### 3.3 Non-destruction guarantee

- Every pattern's `matchAndRewrite` returns `failure()` unless it can perform the full legal rewrite. No partial rewrites.
- Any `scf.for` the pass does not recognize is preserved unchanged.
- Running the pass twice is a no-op on the second run (first run's output contains no `scf.for` for the greedy driver to match on).
- Mixed input — hand-written `csl.get_mem_dsd` alongside unmatched `scf.for` — is preserved cleanly in both respects.

### 3.4 Placement — prep commit

`CSLInferExports.cpp` is a CSL→CSL transform currently misfiled in `mlir/lib/Conversion/`. The first commit of this work relocates it to the upstream MLIR convention location and establishes the directory for future CSL transforms:

| Before | After |
|---|---|
| `mlir/lib/Conversion/CSLInferExports.cpp` | `mlir/lib/Dialect/CSL/Transforms/CSLInferExports.cpp` |
| `mlir/include/air/Conversion/CSLInferExportsPass.h` | `mlir/include/air/Dialect/CSL/Transforms/Passes.h` |
| entry in `mlir/include/air/Conversion/Passes.td` | new `mlir/include/air/Dialect/CSL/Transforms/Passes.td` |
| Build wiring in `mlir/lib/Conversion/CMakeLists.txt` | new `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt` + parent-dir add_subdirectory |
| Per-pass registration in `air-opt` main | `registerCSLTransformPasses()` helper (groups all CSL transforms) |

Pass flag name `-csl-infer-exports` is unchanged. All existing test RUN lines continue to work unmodified. `ninja check-air-mlir` must pass on the prep commit alone.

### 3.5 Placement — new-pass commit

The `-csl-auto-vectorize` pass is added into the directory structure the prep commit created:

```
mlir/include/air/Dialect/CSL/Transforms/
  Passes.h                  ← extended: add createCSLAutoVectorizePass()
  Passes.td                 ← extended: add CSLAutoVectorize pass def
  LoopIdiomAnalysis.h       ← NEW: analyzeForLoop() + LoopIdiom struct

mlir/lib/Dialect/CSL/Transforms/
  CMakeLists.txt            ← extended
  CSLInferExports.cpp       (from prep commit)
  LoopIdiomAnalysis.cpp     ← NEW: shared legality implementation
  CSLAutoVectorize.cpp      ← NEW: pass registration + pattern-set driver
  Patterns/
    CMakeLists.txt
    PatternsCommon.h        ← NEW: shared IR-construction helpers
    ElementwisePatterns.cpp ← NEW: Fadds/Fsubs/Fmuls (f32)
    FmaPattern.cpp          ← NEW: Fmacs (f32)
    MovePatterns.cpp        ← NEW: Fmovs/Fnegs (f32)
    ScalarBroadcastPatterns.cpp ← NEW: FmulsScalar, FmacsScalar (f32)
```

Each `Patterns/*.cpp` contains 1–4 sibling `OpRewritePattern<scf::ForOp>` subclasses. Pattern files stay ~100–150 lines; each maps 1:1 to a published CSL builtin name.

---

## 4. Shared legality analysis — `analyzeForLoop`

Lives in `LoopIdiomAnalysis.{h,cpp}`. Called first by every pattern's `matchAndRewrite`. Returns `FailureOr<LoopIdiom>`; on failure logs a one-line reject reason through `LLVM_DEBUG(DBG_TYPE("csl-auto-vectorize"))`.

### 4.1 `LoopIdiom` struct

```cpp
struct LoopIdiom {
  // Loop shape
  int64_t lb;             // constant lower bound
  int64_t ub;             // constant upper bound
  int64_t step;           // constant step (MVP: must be 1)
  int64_t extent;         // ub - lb
  Value   inductionVar;

  // Body classification
  SmallVector<memref::LoadOp,  4> loads;
  SmallVector<memref::StoreOp, 1> stores;   // exactly 1 after purity check
  SmallVector<Operation *, 8>     bodyOps;  // arith.* in program order
  SmallVector<Value, 2>           loopInvariants;  // scalar-broadcast candidates

  // Per-access descriptor — one per distinct memref operand
  struct AccessPattern {
    Value                 buffer;         // root memref (pre-subview)
    SmallVector<int64_t,2> strides;       // size() == rank
    SmallVector<int64_t,2> offsets;       // constant, signed; per-rank
    int64_t               subviewExtent;  // effective post-clip length (= LoopIdiom.extent for MVP)
  };
  SmallVector<AccessPattern, 4> accesses;

  // Rank info
  unsigned rank;          // 1 or 2
  // For rank-2, the inner scf.for's IV
  Value   innerInductionVar;
  int64_t innerLb, innerUb, innerStep;
};
```

### 4.2 Legality predicate (MVP)

Applied in order; first failure logged and returned.

**Loop-shape rules:**
1. `getConstantIntValue` succeeds on lower, upper, step.
2. `step == 1`.
3. `extent > 0` and `extent <= 65535` (DSD `mem1d_dsd.extent` is `u16`; `mem4d_dsd.extent` is a tuple of `u16`. Source: `docs/superpowers/raw/cerebras_sdk_docs/csl/Language/DSDs.md:36`, `:223-224`).
4. Single-block body; terminator `scf.yield` with no operands (no loop-carried `iter_args` — rejects reductions).
5. No `scf.if` / nested `scf.for` / `scf.while` / `func.call` in body. **Exception:** rank-2 nest detection, §4.3.
6. Induction variable uses are restricted to memref index positions, either directly or via `arith.addi %iv, %constant` / `arith.muli %iv, %constant` / `affine.apply` with affine map in `%iv` (SPADA "indexing-only iterator usage").

**Purity rules (SPADA verbatim):**

7. Exactly one `memref.store` in body (single output).
8. Every non-terminator op is one of `arith.*`, `memref.load`, `memref.store`, `arith.constant`. No side-effecting ops.
9. SSA values consumed by body but defined outside the loop classify as: the IV (see rule 6), a loop-invariant memref (→ `AccessPattern.buffer`), or a loop-invariant scalar of type `f32` (→ `loopInvariants`, the scalar-broadcast candidate set). Anything else rejects.

**Access-pattern rules (per load/store):**
10. Index is a constant-coefficient affine function of the IV: MVP accepts `%i`, `%i + k`, `%i * c + k` for constants `c, k ∈ ℤ` (including negative).
11. **Access-in-bounds** (per negative-offset support): for index expression `coeff*%i + k` on buffer extent `N`, let `a_min = min(lb*coeff + k, (ub-1)*coeff + k)` and `a_max = max(…)`. Reject unless `0 ≤ a_min` and `a_max < N`. Accepts stencil patterns where the loop range has been clipped to keep accesses in bounds (e.g., `for i in [1, N-1)` with `a[i-1]`).
12. **DSD field-width fits** (from `DSDs.md:36-38` for mem1d, `:218-224` for mem4d):
    - rank-1: `coeff ∈ [-128, 127]` (mem1d stride is `i8`), `offset = k + lb*coeff` for the effective subview must fit in `i16` word units.
    - rank-2: per-dim `stride ∈ [-32768, 32767]` (mem4d stride tuple is `i16` per element), `offset ∈ [-32768, 32767]`. Also: rank-2 stride and extent must be compile-time known, matching our "static extent" scope.

### 4.3 Rank-2 detection

`analyzeForLoop` first tries the rank-1 interpretation. If the outer loop body contains **exactly one** operation and that operation is a nested `scf.for`, the analyzer recurses into the inner loop, applies the same shape predicate, then combines: every access index must be affine in `(outer_iv, inner_iv)`. The composite `AccessPattern` has rank-2 strides/offsets. Access-in-bounds becomes the product-range check.

### 4.4 Loop-invariant scalar detection

A value `%α` is loop-invariant iff its defining op (if any) is outside the `scf.for`'s body region, OR it is a block argument of the enclosing `csl.func`. `analyzeForLoop` collects these into `LoopIdiom.loopInvariants`; the `*ScalarPattern` classes consult this set.

---

## 5. Idiom match table

One `OpRewritePattern<scf::ForOp>` subclass per row. Each pattern:
1. Calls `analyzeForLoop`; on `failure()`, returns `failure()`.
2. Checks if `bodyOps` matches its specific body signature.
3. If matched, rewrites; returns `success()`. Else returns `failure()`.

The following table reflects the **actual SDK float-DSD builtin set** as enumerated from `docs/superpowers/raw/cerebras_sdk_docs/csl/Language/Builtins.md`. The `…s` suffix = single precision = 32-bit float. (Half-precision `@…h` variants exist but are T3 follow-up scope.)

| Pattern class | Body signature | CSL builtin | Covered examples |
|---|---|---|---|
| `FaddsPattern` | `%0 = load a[i]; %1 = load b[i]; %2 = arith.addf %0, %1 : f32; store %2, c[i]` | `@fadds` | `vecadd_*`, `plain_vecadd`, `helper_add` |
| `FsubsPattern` | same with `arith.subf : f32` | `@fsubs` | — |
| `FmulsPattern` | same with `arith.mulf : f32` | `@fmuls` | `vecmul_f32` |
| `FmacsPattern` | `%m = mulf a[i], b[i] : f32; %s = addf %m, c[i] : f32; store %s, c[i]` | `@fmacs` | FMA forms (3-buffer) |
| `FmovsPattern` | `%0 = load a[i]; store %0, c[i]` on `f32` | `@fmovs` | simple f32 copies |
| `FnegsPattern` | `%0 = load a[i] : f32; %1 = arith.negf %0; store %1, c[i]` | `@fnegs` | `neg_signflip` |
| `FmulsScalarPattern` | `%0 = load a[i] : f32; %1 = mulf %0, %α; store %1, c[i]` (α loop-invariant f32) | `@fmuls(dc, da, α)` | saxpy-scale leg |
| `FmacsScalarPattern` | `%m = mulf a[i], %α : f32; %s = addf %m, c[i] : f32; store %s, c[i]` | `@fmacs(dc, dc, da, α)` | `saxpy`, `even_saxpy` |

Every pattern's `matchAndRewrite` bails (`return failure()`) if `load.getType() != f32`. Non-f32 loops fall through unchanged to the emitter's scalar path — users of i32/i16/f16 kernels see no regression, they simply don't get DSD speedup in this release.

### 5.1 Element-type gate

Each pattern class begins with a type guard: if `loads[0].getType() != f32` (or the relevant op's result type isn't f32), the pattern returns `failure()` immediately. The pattern count is one per algebraic idiom; widening to f16 / i16 later is a matter of adding sibling classes (`FaddhPattern`, `Add16Pattern`, …) inside the same `Patterns/*.cpp` files, not restructuring.

---

## 6. Rewrite shapes

### 6.1 Canonical rank-1 elementwise

**Input:**
```mlir
csl.func @compute {
  %c0 = arith.constant 0 : index
  %n  = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %va = memref.load %a[%i] : memref<1024xf32>
    %vb = memref.load %b[%i] : memref<1024xf32>
    %vc = arith.addf %va, %vb : f32
    memref.store %vc, %c[%i] : memref<1024xf32>
  }
  csl.return
}
```

**Output:**
```mlir
csl.func @compute {
  %da = csl.get_mem_dsd %a : memref<1024xf32> -> !csl.dsd
  %db = csl.get_mem_dsd %b : memref<1024xf32> -> !csl.dsd
  %dc = csl.get_mem_dsd %c : memref<1024xf32> -> !csl.dsd
  csl.builtin_call "fadds"(%dc, %da, %db) : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
  csl.return
}
```

The `scf.for` and its `arith.constant` index definitions are erased. Any `arith.constant` left dead after erasure is cleaned up by the greedy driver's built-in DCE.

### 6.2 Stencil with negative offset

**Input (loop range `[1, N-1)` keeps accesses in bounds):**
```mlir
scf.for %i = %c1 to %Nm1 step %c1 {
  %vl = memref.load %a[%i - 1] : memref<128xf32>
  %vc = memref.load %a[%i]     : memref<128xf32>
  %sum = arith.addf %vl, %vc   : f32
  memref.store %sum, %c[%i]    : memref<128xf32>
}
```

**Output:**
```mlir
%da_sh  = memref.subview %a[0] [126] [1] : memref<128xf32> to memref<126xf32, strided<[1], offset: 0>>
%da_ctr = memref.subview %a[1] [126] [1] : memref<128xf32> to memref<126xf32, strided<[1], offset: 1>>
%dc_out = memref.subview %c[1] [126] [1] : memref<128xf32> to memref<126xf32, strided<[1], offset: 1>>
%d_sh  = csl.get_mem_dsd %da_sh  : memref<126xf32, strided<[1], offset: 0>> -> !csl.dsd
%d_ctr = csl.get_mem_dsd %da_ctr : memref<126xf32, strided<[1], offset: 1>> -> !csl.dsd
%d_out = csl.get_mem_dsd %dc_out : memref<126xf32, strided<[1], offset: 1>> -> !csl.dsd
csl.builtin_call "fadds"(%d_out, %d_ctr, %d_sh) : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
```

### 6.3 Saxpy with scalar broadcast

**Input:**
```mlir
csl.func @compute {
  %α = memref.load %alpha[%c0] : memref<1xf32>
  scf.for %i = %c0 to %n step %c1 {
    %vx = memref.load %x[%i] : memref<128xf32>
    %vy = memref.load %y[%i] : memref<128xf32>
    %m  = arith.mulf %vx, %α : f32
    %s  = arith.addf %m, %vy : f32
    memref.store %s, %y[%i] : memref<128xf32>
  }
  csl.return
}
```

**Output:**
```mlir
csl.func @compute {
  %α = memref.load %alpha[%c0] : memref<1xf32>
  %dx = csl.get_mem_dsd %x : memref<128xf32> -> !csl.dsd
  %dy = csl.get_mem_dsd %y : memref<128xf32> -> !csl.dsd
  csl.builtin_call "fmacs"(%dy, %dy, %dx, %α) : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
  csl.return
}
```

Matches `mlir/test/Targets/CSLEmit/e2e/dsds.mlir:27` verbatim.

### 6.4 Rank-2

**Input:**
```mlir
scf.for %i = %c0 to %M step %c1 {
  scf.for %j = %c0 to %N step %c1 {
    %va = memref.load %A[%i, %j] : memref<8x16xf32>
    %vb = memref.load %B[%i, %j] : memref<8x16xf32>
    %vc = arith.addf %va, %vb : f32
    memref.store %vc, %C[%i, %j] : memref<8x16xf32>
  }
}
```

**Output:**
```mlir
%va = memref.subview %A[0, 0] [8, 16] [1, 1] : memref<8x16xf32> to memref<8x16xf32, strided<[16, 1]>>
%vb = memref.subview %B[0, 0] [8, 16] [1, 1] : memref<8x16xf32> to memref<8x16xf32, strided<[16, 1]>>
%vc = memref.subview %C[0, 0] [8, 16] [1, 1] : memref<8x16xf32> to memref<8x16xf32, strided<[16, 1]>>
%dA = csl.get_mem_dsd %va : memref<8x16xf32, strided<[16, 1]>> -> !csl.dsd
%dB = csl.get_mem_dsd %vb : memref<8x16xf32, strided<[16, 1]>> -> !csl.dsd
%dC = csl.get_mem_dsd %vc : memref<8x16xf32, strided<[16, 1]>> -> !csl.dsd
csl.builtin_call "fadds"(%dC, %dA, %dB) : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()
```

`csl.get_mem_dsd` with rank-2 memref emits `mem4d_dsd` per its existing documented behavior (`mlir/include/air/Dialect/CSL/CSLOps.td:270-275`). Identity subviews are kept deliberately in the output IR to make the stride/extent explicit for downstream readers; MLIR's subview-fold patterns may collapse them under normalisation passes, which is fine.

---

## 7. Testing strategy

Tests land under a new directory `mlir/test/Dialect/CSL/Transforms/auto-vectorize/`.

### 7.1 Positive dialect-level tests

One `.mlir` file per pattern class — minimal `csl.wafer` / `csl.program` / `csl.func` skeleton, RUN line `air-opt %s -csl-auto-vectorize`, no emitter. CHECK patterns verify:
- `CHECK-NOT: scf.for`
- `CHECK: csl.get_mem_dsd` (one per distinct buffer)
- `CHECK: csl.builtin_call "<callee>"(...) : (!csl.dsd, ...) -> ()`

File list:
```
fadds.mlir   iadds.mlir
fsubs.mlir   fmuls.mlir   imuls.mlir
fmacs.mlir
fmovs.mlir   fnegs.mlir   imovs.mlir   inegs.mlir
fmuls_scalar.mlir
fmacs_scalar.mlir
stencil_fadds.mlir          # negative-offset subview
rank2_fadds.mlir            # 2D nest → mem4d_dsd
rank2_fmacs.mlir
```

### 7.2 Negative / fall-through tests

Critical for proving the non-destruction guarantee. One file per rejection class; each asserts `CHECK: scf.for` (the scalar loop survives) *and* `CHECK-NOT: csl.get_mem_dsd` (pass did not introduce DSDs on this input). Each file also runs with `-debug-only=csl-auto-vectorize 2>&1` so lit can FileCheck the exact reject-reason string under a second prefix `TRACE:`.

```
mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/
  multi_store.mlir          # 2 memref.store → reject (not pure)
  inner_if.mlir             # scf.if in body → reject (control flow)
  non_constant_bound.mlir   # ub = block arg → reject (dynamic extent)
  non_affine_index.mlir     # memref.load %a[%mystery] → reject
  oob_access.mlir           # a[i-1] with i ∈ [0, N) → reject (OOB)
  iter_args_reduction.mlir  # scf.for with iter_args → reject (out of scope)
  unsupported_body_op.mlir  # func.call in body → reject
  mixed_preserved.mlir      # mixed hand-written DSD + unmatched scf.for → both kept
```

### 7.3 End-to-end through emitter

Under `mlir/test/Targets/CSLEmit/e2e/auto-vectorize/`. Full pipeline RUN line: `air-opt %s -csl-auto-vectorize -csl-infer-exports | air-translate --emit-csl --output-dir=%t`. CHECK patterns verify generated `.csl` text contains `@get_dsd(` + `@fadds(` etc. Minimal set at MVP: `vecadd.mlir`, `saxpy_fma.mlir`, `rank2_matmul.mlir`.

### 7.4 Existing-test preservation

- The prep commit alone must pass `ninja check-air-mlir` unchanged (no test text touched; only file locations moved).
- The new-pass commit must keep existing RUN lines (those not using `-csl-auto-vectorize`) producing identical output. A new lit target `check-airmlir-dialect-csl-transforms` groups all categories in §7.1–7.3.

---

## 8. Debugging and diagnostics

- No user-facing diagnostics. Failing to vectorize is not a user-visible event.
- `LLVM_DEBUG(DBG_TYPE("csl-auto-vectorize"))` emits:
  - `reject: <reason> @<loc>` — one per failed `analyzeForLoop` call. Reasons are short strings drawn from the rule list in §4.2 (e.g., `"step != 1"`, `"body has scf.if"`, `"access OOB on buffer <name> at i=-1"`).
  - `match: <PatternName> @<loc>` — one per successful pattern rewrite.
- Run with `-debug-only=csl-auto-vectorize` to enable.

---

## 9. Rollout and follow-ups

### 9.1 Commit sequence

1. **Prep commit:** relocate `CSLInferExports` to `mlir/lib/Dialect/CSL/Transforms/`. Zero behavior change. `ninja check-air-mlir` green.
2. **Pass commit:** add `-csl-auto-vectorize` with S4-scope + D-tier + T1-types (f32-only) coverage. Full test suite per §7. `aircc.py` updated to include the pass in its default CSL pipeline.

### 9.2 Follow-ups (not in this spec)

- **i16 integer support — one follow-up patch.** Adds `Add16Pattern`, `Sub16Pattern`, `Mov16Pattern` as sibling classes inside the existing `Patterns/` files. No structural change to the pass — just sibling rewrite patterns with a type gate of `i16`. Does **not** include multiply (`@mul16` doesn't exist), direct negation (`@neg16` doesn't exist — goes through `arith.subi 0, x → @sub16`), or FMA (no integer FMA in SDK).
- **i32 bulk copy (`@mov32`).** Single pattern (`Mov32Pattern`), likely bundled with the i16 patch. i32 arithmetic (`add`/`sub`/`mul`) has no DSD builtin; those loops stay scalar forever. Ship when there's a workload that benefits.
- **f16 (half-precision) support.** Sibling classes (`FaddhPattern`, `FmachPattern`, …) using `@faddh` / `@fmach` / `@fmulh` / `@fmovh` / `@fnegh`. Orthogonal to i16.
- **Logical / shift integer DSD ops** (`@and16`, `@or16`, `@xor16`, `@sll16`, `@slr16`, `@sar16`, `@popcnt`, `@clz`, `@ctz`). Straightforward pattern additions once i16 lands.
- **Tier-2 `@map` lowering.** Pure-body non-idiom loops → CSL `@map` with closure. Requires new CSL dialect op (`csl.map`) and closure-lowering.
- **Reductions (`@fadds` with DSR accumulator).** `scf.for` with `iter_args` of scalar type, commutative associative body. Likely a sibling pass `-csl-recognize-reductions`.
- **Non-unit loop step.** Confirm SDK semantics under DSD stride vs IV step composition.
- **Memref canonicalisation interactions.** If upstream memref-subview folding runs before `-csl-auto-vectorize`, some rewrites may see already-folded input. Expected to be a no-op, but plan stage should verify with a mixed test.
