# `-csl-auto-vectorize` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CSL-dialect MLIR pass `-csl-auto-vectorize` that recognizes scalar `scf.for` loops over `memref<Nxf32>` / `memref<MxNxf32>` buffers and rewrites them into `csl.get_mem_dsd` + `csl.builtin_call` form (SPADA Tier-1 DSD vectorization); any loop that fails the legality predicate is preserved unchanged (Tier-4 scalar fall-through).

**Architecture:** Pattern-set-per-idiom (Approach B from spec §3). A shared `analyzeForLoop` helper runs first to validate loop shape + purity + access pattern; then each of eight `OpRewritePattern<scf::ForOp>` subclasses (one per f32 idiom — `@fadds`, `@fsubs`, `@fmuls`, `@fmacs`, `@fmovs`, `@fnegs`, `@fmuls`-scalar-broadcast, `@fmacs`-scalar-broadcast) checks its specific body-op signature and rewrites iff it matches. Rank-2 is a sibling of rank-1 that traverses the inner `scf.for`. The pass is non-destructive: any unmatched loop stays scalar.

**Tech Stack:** MLIR C++ (LLVM 19 toolchain via `llvm/install`), CMake (`add_mlir_library`), `scf` / `memref` / `arith` dialects, `xilinx::csl` dialect, `PatternApplicator` / `applyPatternsGreedily` driver, LLVM `lit` + `FileCheck` for tests, `LLVM_DEBUG` for opt-trace output.

**Spec:** `docs/superpowers/specs/2026-04-21-csl-auto-vectorize-design.md`
**SDK reference:** `docs/superpowers/raw/cerebras_sdk_docs/csl/Language/Builtins.md` (`@fadds` L3054, `@fmacs` L3100, `@fmovs` L3140, `@fmuls` L3161, `@fnegs` L3181, `@fsubs` L3252), `docs/superpowers/raw/cerebras_sdk_docs/csl/Language/DSDs.md` (mem1d extent L36, mem4d L212-228).
**Existing hand-written DSD example to match:** `mlir/test/Targets/CSLEmit/e2e/dsds.mlir` (shows canonical `csl.get_mem_dsd` + `csl.builtin_call "fmacs"` shape).

**Environment prerequisite for every Bash command below:**

```bash
module load python/3.10.3 cmake/3.26.0 rocm/6.2.1 singularity
source /home/bricklib_dataflow/air-csl/mlir-air/sandbox/bin/activate
source /home/bricklib_dataflow/air-csl/mlir-air/utils/env_setup_gpu.sh install /home/bricklib_dataflow/air-csl/mlir-air/llvm/install
```
After that, `air-opt`, `lit`, `ninja`, `FileCheck` are on `PATH` and the build directory is `/home/bricklib_dataflow/air-csl/mlir-air/build`.

---

## File Structure Overview

**Prep commit moves (Task 1):**

| From | To |
|---|---|
| `mlir/lib/Conversion/CSLInferExports.cpp` | `mlir/lib/Dialect/CSL/Transforms/CSLInferExports.cpp` |
| `mlir/include/air/Conversion/CSLInferExportsPass.h` | `mlir/include/air/Dialect/CSL/Transforms/CSLInferExportsPass.h` |

**New files created by pass commits (Tasks 2-22):**

| Path | Purpose | Tasks |
|---|---|---|
| `mlir/include/air/Dialect/CSL/Transforms/Passes.h` | aggregates CSL transform pass headers + `registerCSLTransformPasses()` | 1, 2 |
| `mlir/include/air/Dialect/CSL/Transforms/CSLAutoVectorizePass.h` | declares `createCSLAutoVectorizePass()` | 2 |
| `mlir/include/air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h` | `LoopIdiom` struct + `analyzeForLoop` signature | 3, 4, 5, 6, 7 |
| `mlir/lib/Dialect/CSL/CMakeLists.txt` | add `Transforms` subdir | 1 |
| `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt` | builds `CSLTransforms` library | 1, 2, 3, 9 |
| `mlir/lib/Dialect/CSL/Transforms/CSLInferExports.cpp` | relocated (unchanged logic) | 1 |
| `mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp` | `analyzeForLoop` implementation | 3, 4, 5, 6, 7 |
| `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp` | pass registration + pattern-set driver | 2, 8, 9, 18 |
| `mlir/lib/Dialect/CSL/Transforms/Patterns/PatternsCommon.h` | shared IR-construction helpers (build subview, get_mem_dsd, builtin_call) | 8 |
| `mlir/lib/Dialect/CSL/Transforms/Patterns/ElementwisePatterns.cpp` | Fadds / Fsubs / Fmuls patterns | 9, 10, 11 |
| `mlir/lib/Dialect/CSL/Transforms/Patterns/FmaPattern.cpp` | Fmacs pattern | 12 |
| `mlir/lib/Dialect/CSL/Transforms/Patterns/MovePatterns.cpp` | Fmovs / Fnegs patterns | 13, 14 |
| `mlir/lib/Dialect/CSL/Transforms/Patterns/ScalarBroadcastPatterns.cpp` | FmulsScalar / FmacsScalar patterns | 15, 16 |
| `mlir/lib/Dialect/CSL/Transforms/Patterns/Rank2Patterns.cpp` | rank-2 variants | 18, 19 |

**New test directories:**

| Path | Contents | Tasks |
|---|---|---|
| `mlir/test/Dialect/CSL/Transforms/auto-vectorize/` | one positive lit file per pattern | 9-19 |
| `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/` | negative/preservation tests | 20 |
| `mlir/test/Targets/CSLEmit/e2e/auto-vectorize/` | end-to-end through emitter | 21 |

**Existing files modified:**

| Path | Change | Task |
|---|---|---|
| `mlir/lib/Conversion/CMakeLists.txt` | remove `CSLInferExports.cpp` from list | 1 |
| `mlir/include/air/Conversion/Passes.h` | remove `#include "air/Conversion/CSLInferExportsPass.h"` | 1 |
| `mlir/lib/Conversion/Passes.cpp` | drop hand-registration of CSLInferExports (moved to new dir's registration) | 1 |
| `tools/air-opt/air-opt.cpp` (or wherever `registerConversionPasses()` is called) | add `registerCSLTransformPasses()` call | 1 |
| `python/air/compiler/aircc/main.py` | include `-csl-auto-vectorize` in CSL pipeline | 22 |

---

## Task 1: Prep commit — relocate CSLInferExports

**Files:**
- Create: `mlir/include/air/Dialect/CSL/Transforms/Passes.h`
- Create: `mlir/include/air/Dialect/CSL/Transforms/CSLInferExportsPass.h` (content copied from old path)
- Create: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`
- Create: `mlir/lib/Dialect/CSL/Transforms/CSLInferExports.cpp` (content copied from old path, include-path updated)
- Modify: `mlir/lib/Dialect/CSL/CMakeLists.txt:4` (add `add_subdirectory(Transforms)`)
- Modify: `mlir/lib/Conversion/CMakeLists.txt:9` (remove `CSLInferExports.cpp`)
- Modify: `mlir/include/air/Conversion/Passes.h` (remove `#include "air/Conversion/CSLInferExportsPass.h"`)
- Modify: `mlir/lib/Conversion/Passes.cpp` (remove hand-registration of `createCSLInferExportsPass`)
- Modify: `tools/air-opt/air-opt.cpp` (add `registerCSLTransformPasses()` after the existing `registerConversionPasses()` call)
- Delete: `mlir/include/air/Conversion/CSLInferExportsPass.h`
- Delete: `mlir/lib/Conversion/CSLInferExports.cpp`

- [ ] **Step 1.1: Confirm baseline build + tests are green**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install && ninja check-air-mlir
```
Expected: clean build, all tests pass. If this fails, stop and investigate before proceeding — the relocation commit must be a strict no-op.

- [ ] **Step 1.2: Verify the file we're moving**

```bash
wc -l /home/bricklib_dataflow/air-csl/mlir-air/mlir/include/air/Conversion/CSLInferExportsPass.h \
      /home/bricklib_dataflow/air-csl/mlir-air/mlir/lib/Conversion/CSLInferExports.cpp
```
Expected: 37 lines for the header, 256 lines for the .cpp. If different, read both files before proceeding.

- [ ] **Step 1.3: Find where `registerConversionPasses` is called from air-opt**

```bash
grep -Rn "registerConversionPasses" /home/bricklib_dataflow/air-csl/mlir-air/tools/
```
Expected: one hit, probably in `tools/air-opt/air-opt.cpp` or similar. **Record this path** — every "modify air-opt registration" step uses it. Call it `$AIR_OPT_CPP` below.

- [ ] **Step 1.4: Create directory skeletons**

```bash
mkdir -p /home/bricklib_dataflow/air-csl/mlir-air/mlir/include/air/Dialect/CSL/Transforms
mkdir -p /home/bricklib_dataflow/air-csl/mlir-air/mlir/lib/Dialect/CSL/Transforms/Patterns
mkdir -p /home/bricklib_dataflow/air-csl/mlir-air/mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough
mkdir -p /home/bricklib_dataflow/air-csl/mlir-air/mlir/test/Targets/CSLEmit/e2e/auto-vectorize
```

- [ ] **Step 1.5: Create `mlir/include/air/Dialect/CSL/Transforms/CSLInferExportsPass.h`**

File content (identical to old header except the include guard):

```cpp
//===- CSLInferExportsPass.h ------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-infer-exports pass factory.
//
// The pass walks csl.host regions for csl_host.memcpy_h2d / memcpy_d2h /
// launch ops and auto-generates the corresponding csl.export ops in
// csl.program and csl_layout.export ops in csl.layout.  Pre-existing exports
// not referenced by any host op receive direction = "internal".
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_INFER_EXPORTS_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_INFER_EXPORTS_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCSLInferExportsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_CSL_INFER_EXPORTS_PASS_H
```

- [ ] **Step 1.6: Create `mlir/include/air/Dialect/CSL/Transforms/Passes.h`**

```cpp
//===- Passes.h ---------------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Aggregate header + registration entry point for every pass that operates
// intra-CSL-dialect (CSL -> CSL rewrites).  Air-opt calls
// registerCSLTransformPasses() in addition to registerConversionPasses().
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_PASSES_H
#define AIR_DIALECT_CSL_TRANSFORMS_PASSES_H

#include "air/Dialect/CSL/Transforms/CSLInferExportsPass.h"

namespace xilinx {
namespace air {

void registerCSLTransformPasses();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_PASSES_H
```

- [ ] **Step 1.7: Move the .cpp file and update its include**

```bash
git mv mlir/lib/Conversion/CSLInferExports.cpp \
       mlir/lib/Dialect/CSL/Transforms/CSLInferExports.cpp
```

Then edit the include at the top of the moved file:

```cpp
// OLD:
// #include "air/Conversion/CSLInferExportsPass.h"
// NEW:
#include "air/Dialect/CSL/Transforms/CSLInferExportsPass.h"
```

- [ ] **Step 1.8: Delete the old header + create registration .cpp**

```bash
git rm mlir/include/air/Conversion/CSLInferExportsPass.h
```

Create `mlir/lib/Dialect/CSL/Transforms/Passes.cpp`:

```cpp
//===- Passes.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

void xilinx::air::registerCSLTransformPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> {
        return createCSLInferExportsPass();
      });
}
```

- [ ] **Step 1.9: Create `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`**

```cmake
# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_mlir_library(
  CSLTransforms
  CSLInferExports.cpp
  Passes.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSL/Transforms

  LINK_LIBS PUBLIC
  CSLDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms)
```

- [ ] **Step 1.10: Wire the new library into parent CMakeLists**

Edit `mlir/lib/Dialect/CSL/CMakeLists.txt`:

```cmake
# Part of the air-to-csl project.
# SPDX-License-Identifier: MIT

add_subdirectory(IR)
add_subdirectory(Transforms)
```

- [ ] **Step 1.11: Remove CSLInferExports from the Conversion library**

Edit `mlir/lib/Conversion/CMakeLists.txt`:

Remove line 9 (`CSLInferExports.cpp`) from `set(CONVERSION_SOURCES ...)`.

- [ ] **Step 1.12: Remove the stale include and registration**

Edit `mlir/include/air/Conversion/Passes.h` — remove the line:

```cpp
#include "air/Conversion/CSLInferExportsPass.h"
```

Edit `mlir/lib/Conversion/Passes.cpp` — remove the registration block for CSLInferExports (lines 73-76 in the original file):

```cpp
// DELETE THESE LINES:
mlir::registerPass(
    []() -> std::unique_ptr<mlir::Pass> {
      return createCSLInferExportsPass();
    });
```

- [ ] **Step 1.13: Link CSLTransforms from the air-opt tool**

Find the `target_link_libraries` block for the air-opt executable (typically in `tools/air-opt/CMakeLists.txt`):

```bash
grep -n "AIRConversionPasses" /home/bricklib_dataflow/air-csl/mlir-air/tools/air-opt/CMakeLists.txt
```

In the same `target_link_libraries` block, add `CSLTransforms` alongside `AIRConversionPasses`.

- [ ] **Step 1.14: Call the new registration in air-opt main**

Edit `$AIR_OPT_CPP` (from Step 1.3). Find the body of `main` or a registration setup function — it already calls `xilinx::air::registerConversionPasses()`. Add the include and companion call:

```cpp
// Near the top of the file, alongside existing air/Conversion/Passes.h include:
#include "air/Dialect/CSL/Transforms/Passes.h"
```

```cpp
// Inside main or the registration setup, right after the existing call:
xilinx::air::registerConversionPasses();
xilinx::air::registerCSLTransformPasses();   // NEW
```

- [ ] **Step 1.15: Build**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install
```
Expected: clean build. If there are include-path errors, double-check the include in the moved .cpp (Step 1.7) and the header's guard macro.

- [ ] **Step 1.16: Run the full dialect test suite**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja check-air-mlir
```
Expected: every existing test passes, same count as Step 1.1. The relocation is a strict no-op — if any test fails, the most likely cause is a missing registration (pass name `-csl-infer-exports` isn't found by air-opt) or a missing link dep in `tools/air-opt/CMakeLists.txt`.

- [ ] **Step 1.17: Commit**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air
git add mlir/include/air/Dialect/CSL/Transforms/ \
        mlir/lib/Dialect/CSL/Transforms/ \
        mlir/lib/Dialect/CSL/CMakeLists.txt \
        mlir/include/air/Conversion/Passes.h \
        mlir/include/air/Conversion/CSLInferExportsPass.h \
        mlir/lib/Conversion/CMakeLists.txt \
        mlir/lib/Conversion/Passes.cpp \
        mlir/lib/Conversion/CSLInferExports.cpp \
        tools/air-opt/
git commit -m "$(cat <<'EOF'
refactor(csl): move CSLInferExports to Dialect/CSL/Transforms

CSL-to-CSL transforms belong under mlir/lib/Dialect/CSL/Transforms/
(upstream MLIR convention) rather than mlir/lib/Conversion/.  This prep
commit establishes the directory and registration hook
(registerCSLTransformPasses) ahead of -csl-auto-vectorize.

Pass flag name -csl-infer-exports unchanged; no test updates.
EOF
)"
```

---

## Task 2: Empty `-csl-auto-vectorize` pass skeleton

**Files:**
- Create: `mlir/include/air/Dialect/CSL/Transforms/CSLAutoVectorizePass.h`
- Create: `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/noop.mlir`
- Modify: `mlir/include/air/Dialect/CSL/Transforms/Passes.h` (add new include)
- Modify: `mlir/lib/Dialect/CSL/Transforms/Passes.cpp` (register new pass)
- Modify: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt` (add new source)

- [ ] **Step 2.1: Create `mlir/test/Dialect/CSL/Transforms/auto-vectorize/noop.mlir`**

Write the failing test first:

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// Smoke test: the pass registers and is a no-op on a CSL program that
// contains no scf.for loop.  We only verify it round-trips cleanly.

// CHECK-LABEL: csl.wafer @noop
module {
  csl.wafer @noop {arch = "wse3"} {
    csl.program @pe {
      csl.func @compute {
        csl.return
      }
      csl.export @compute {kind = "func"}
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

- [ ] **Step 2.2: Run test to verify pass doesn't exist yet**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && \
  lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/noop.mlir -v
```
Expected: `error: '-csl-auto-vectorize': Unknown command line argument`.

- [ ] **Step 2.3: Create `mlir/include/air/Dialect/CSL/Transforms/CSLAutoVectorizePass.h`**

```cpp
//===- CSLAutoVectorizePass.h -----------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Declares the -csl-auto-vectorize pass factory.
//
// The pass walks csl.program bodies for scf.for loops whose body matches a
// known CSL DSD idiom (elementwise f32 add/sub/mul, FMA, mov, neg, plus
// scalar-broadcast variants) and rewrites the loop into
// csl.get_mem_dsd + csl.builtin_call.  Loops that fail the legality
// predicate are preserved unchanged (Tier-4 scalar fall-through).
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_AUTO_VECTORIZE_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_AUTO_VECTORIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCSLAutoVectorizePass();

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_CSL_AUTO_VECTORIZE_PASS_H
```

- [ ] **Step 2.4: Create `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp` (skeleton only)**

```cpp
//===- CSLAutoVectorize.cpp -------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// -csl-auto-vectorize pass: rewrites scalar scf.for loops over memref buffers
// inside csl.func bodies into csl.get_mem_dsd + csl.builtin_call form where
// the body matches a known DSD idiom.
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLAutoVectorizePass.h"

#include "air/Dialect/CSL/CSLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;

namespace {

struct CSLAutoVectorizePass
    : public PassWrapper<CSLAutoVectorizePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLAutoVectorizePass)

  StringRef getArgument() const override { return "csl-auto-vectorize"; }
  StringRef getDescription() const override {
    return "Recognize scalar scf.for loops that match CSL DSD idioms and "
           "rewrite them into csl.get_mem_dsd + csl.builtin_call form";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect,
                    arith::ArithDialect, xilinx::csl::CSLDialect>();
  }

  void runOnOperation() override {
    // No-op for now — patterns are added in later tasks.
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::air::createCSLAutoVectorizePass() {
  return std::make_unique<CSLAutoVectorizePass>();
}
```

- [ ] **Step 2.5: Register the pass**

Edit `mlir/include/air/Dialect/CSL/Transforms/Passes.h` — add the include:

```cpp
#include "air/Dialect/CSL/Transforms/CSLAutoVectorizePass.h"
```

Edit `mlir/lib/Dialect/CSL/Transforms/Passes.cpp`:

```cpp
void xilinx::air::registerCSLTransformPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> {
        return createCSLInferExportsPass();
      });
  mlir::registerPass(                                     // ADD THIS BLOCK
      []() -> std::unique_ptr<mlir::Pass> {               //
        return createCSLAutoVectorizePass();              //
      });                                                 //
}
```

- [ ] **Step 2.6: Update Transforms CMakeLists**

Edit `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`:

```cmake
add_mlir_library(
  CSLTransforms
  CSLInferExports.cpp
  CSLAutoVectorize.cpp          # ADD
  Passes.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSL/Transforms

  LINK_LIBS PUBLIC
  CSLDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect              # ADD
  MLIRMemRefDialect           # ADD
  MLIRArithDialect)           # ADD
```

- [ ] **Step 2.7: Build**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install
```
Expected: builds cleanly.

- [ ] **Step 2.8: Run test to verify it now passes**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && \
  lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/noop.mlir -v
```
Expected: `PASS`.

- [ ] **Step 2.9: Commit**

```bash
git add mlir/include/air/Dialect/CSL/Transforms/CSLAutoVectorizePass.h \
        mlir/include/air/Dialect/CSL/Transforms/Passes.h \
        mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp \
        mlir/lib/Dialect/CSL/Transforms/Passes.cpp \
        mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt \
        mlir/test/Dialect/CSL/Transforms/auto-vectorize/noop.mlir
git commit -m "feat(csl): add -csl-auto-vectorize pass skeleton (no-op)"
```

---

## Task 3: `LoopIdiom` struct + `analyzeForLoop` skeleton returning failure

**Files:**
- Create: `mlir/include/air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h`
- Create: `mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt` (add new source)

- [ ] **Step 3.1: Create `mlir/include/air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h`**

```cpp
//===- LoopIdiomAnalysis.h --------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Shared legality helper for -csl-auto-vectorize pattern rewrites.  Every
// pattern calls analyzeForLoop() first; on success it returns a filled-in
// LoopIdiom struct describing loop shape, body classification, and per-access
// stride/offset data.  Body-content matching (is this loop an @fadds? an
// @fmacs?) is the pattern's responsibility, not the analyzer's.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_LOOP_IDIOM_ANALYSIS_H
#define AIR_DIALECT_CSL_TRANSFORMS_LOOP_IDIOM_ANALYSIS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx {
namespace air {

/// One memref access (load or store) classified by its affine pattern.
/// For rank-R memrefs, strides and offsets each have R entries.
/// Indexes the IV as:  access = buffer[stride[d] * iv_d + offset[d]]
struct DsdAccessPattern {
  mlir::Value buffer;                              // root memref SSA value
  llvm::SmallVector<int64_t, 2> strides;           // per-rank coefficient
  llvm::SmallVector<int64_t, 2> offsets;           // per-rank constant term
  unsigned rank = 1;                               // 1 or 2 in MVP
};

/// Outcome of analyzing one scf.for against the DSD legality predicate.
/// Valid only if analyzeForLoop() returned success.
struct LoopIdiom {
  // Loop shape
  int64_t lb = 0;
  int64_t ub = 0;
  int64_t step = 1;
  int64_t extent = 0;         // ub - lb  (always > 0)
  mlir::Value inductionVar;

  // For rank-2 nests, the inner loop's shape is recorded here.
  bool isRank2 = false;
  int64_t innerLb = 0;
  int64_t innerUb = 0;
  int64_t innerStep = 1;
  int64_t innerExtent = 0;
  mlir::Value innerInductionVar;

  // Body classification — all loads and stores inside body, plus the
  // arith-op chain (program order).  Terminators (scf.yield) are excluded.
  llvm::SmallVector<mlir::memref::LoadOp, 4> loads;
  llvm::SmallVector<mlir::memref::StoreOp, 1> stores;     // exactly 1
  llvm::SmallVector<mlir::Operation *, 8> bodyOps;        // arith.* only

  // SSA values defined outside the loop, used inside, classified:
  //   - memref buffers are captured per-access in `accesses` below
  //   - scalar loop-invariants (potential @fmuls/@fmacs scalar-broadcast
  //     operands) are collected here
  llvm::SmallVector<mlir::Value, 2> loopInvariantScalars;

  // One pattern entry per distinct memref-operand of a load/store inside
  // the body.  Populated in load+store order; a pattern can look them up
  // by pointer-equality against `loads[i].getMemRef()` etc.
  llvm::SmallVector<DsdAccessPattern, 4> accesses;
};

/// Run the full MVP legality predicate on `op`.  On failure returns
/// failure() and emits an LLVM_DEBUG(DBG_TYPE("csl-auto-vectorize"))
/// trace with the reject reason.  On success returns a LoopIdiom with
/// every field populated.
mlir::FailureOr<LoopIdiom> analyzeForLoop(mlir::scf::ForOp op);

/// DSD field-width limits straight from the SDK docs
/// (DSDs.md:36 for mem1d, :218-224 for mem4d).
constexpr int64_t kMaxDsdExtent = 65535;            // u16
constexpr int64_t kMaxMem1dStride = 127;            // i8
constexpr int64_t kMinMem1dStride = -128;
constexpr int64_t kMaxDsdOffset = 32767;            // i16
constexpr int64_t kMinDsdOffset = -32768;
constexpr int64_t kMaxMem4dStride = 32767;          // i16
constexpr int64_t kMinMem4dStride = -32768;

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_LOOP_IDIOM_ANALYSIS_H
```

- [ ] **Step 3.2: Create `mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp` with stub body**

```cpp
//===- LoopIdiomAnalysis.cpp ------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;

namespace xilinx {
namespace air {

FailureOr<LoopIdiom> analyzeForLoop(scf::ForOp op) {
  LLVM_DEBUG(llvm::dbgs() << "reject: analyzer not yet implemented @"
                          << op.getLoc() << "\n");
  return failure();
}

} // namespace air
} // namespace xilinx
```

- [ ] **Step 3.3: Add source to CMakeLists**

Edit `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`:

```cmake
add_mlir_library(
  CSLTransforms
  CSLInferExports.cpp
  CSLAutoVectorize.cpp
  LoopIdiomAnalysis.cpp          # ADD
  Passes.cpp

  PARTIAL_SOURCES_INTENDED
  ...
```

- [ ] **Step 3.4: Build (sanity check: the stub compiles)**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install
```
Expected: builds cleanly.

- [ ] **Step 3.5: Commit**

```bash
git add mlir/include/air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h \
        mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp \
        mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt
git commit -m "feat(csl-auto-vectorize): add LoopIdiom struct and analyzeForLoop stub"
```

---

## Task 4: Loop-shape predicate rules (1-5 from spec §4.2)

**Goal:** Implement rules 1-5 of `analyzeForLoop`. These check constant bounds, step=1, extent cap, single block / no iter_args, no nested control flow (rank-2 exception is Task 7).

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp`

- [ ] **Step 4.1: Implement loop-shape rules**

Replace the `analyzeForLoop` body in `LoopIdiomAnalysis.cpp`:

```cpp
FailureOr<LoopIdiom> analyzeForLoop(scf::ForOp op) {
  LoopIdiom info;

  // Rule 1 — constant bounds and step.
  std::optional<int64_t> lb = getConstantIntValue(op.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(op.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(op.getStep());
  if (!lb || !ub || !step) {
    LLVM_DEBUG(llvm::dbgs() << "reject: non-constant bounds or step @"
                            << op.getLoc() << "\n");
    return failure();
  }
  info.lb = *lb;
  info.ub = *ub;
  info.step = *step;
  info.extent = info.ub - info.lb;
  info.inductionVar = op.getInductionVar();

  // Rule 2 — step == 1.
  if (info.step != 1) {
    LLVM_DEBUG(llvm::dbgs() << "reject: step != 1 @" << op.getLoc() << "\n");
    return failure();
  }

  // Rule 3 — extent in DSD u16 range.
  if (info.extent <= 0 || info.extent > kMaxDsdExtent) {
    LLVM_DEBUG(llvm::dbgs() << "reject: extent " << info.extent
                            << " outside [1," << kMaxDsdExtent << "] @"
                            << op.getLoc() << "\n");
    return failure();
  }

  // Rule 4 — single-block body, scf.yield with no yielded values.
  if (!op.getRegion().hasOneBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "reject: body has multiple blocks @"
                            << op.getLoc() << "\n");
    return failure();
  }
  if (op.getNumRegionIterArgs() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "reject: loop has iter_args (reduction?) @"
                            << op.getLoc() << "\n");
    return failure();
  }

  // Rule 5 — no nested control flow or calls in body.  (Rank-2 will relax
  // this in Task 7; for now any nested op of these kinds rejects.)
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(&inner)) {
      LLVM_DEBUG(llvm::dbgs() << "reject: nested control flow ("
                              << inner.getName() << ") @"
                              << inner.getLoc() << "\n");
      return failure();
    }
    if (inner.hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
        inner.mightHaveTrait<OpTrait::IsTerminator>())
      continue;
    if (!isa<arith::ArithDialect>(inner.getDialect()) &&
        !isa<memref::MemRefDialect>(inner.getDialect())) {
      LLVM_DEBUG(llvm::dbgs() << "reject: disallowed dialect in body ("
                              << inner.getName() << ") @"
                              << inner.getLoc() << "\n");
      return failure();
    }
  }

  // Subsequent rules 6-12 land in later tasks.
  LLVM_DEBUG(llvm::dbgs() << "reject: rules 6-12 not yet implemented @"
                          << op.getLoc() << "\n");
  return failure();
}
```

- [ ] **Step 4.2: Build (sanity check)**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install
```
Expected: builds cleanly. (No test yet — patterns are the unit that tests analyzer behavior. We'll exercise these rules through fall-through tests in Task 20.)

- [ ] **Step 4.3: Commit**

```bash
git add mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp
git commit -m "feat(csl-auto-vectorize): analyzeForLoop — loop-shape predicate (rules 1-5)"
```

---

## Task 5: Purity predicate rules (6-9 from spec §4.2)

**Goal:** Add purity classification — single store, allowed body-op dialects only, IV-use restrictions, loop-invariant-scalar collection.

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp`

- [ ] **Step 5.1: Implement rules 6-9 — replace the trailing "rules 6-12 not yet implemented" block**

Append to the end of `analyzeForLoop` before the final `return failure()`:

```cpp
  // Rules 7 & 8 — collect loads / stores / arith ops; enforce single store
  // and allowed body-op set (arith.* + memref.load/store only; constants
  // that are index values are index-typed arith.constant which is already
  // dialect=arith).
  for (Operation &inner : op.getBody()->without_terminator()) {
    if (auto ld = dyn_cast<memref::LoadOp>(&inner)) {
      info.loads.push_back(ld);
      continue;
    }
    if (auto st = dyn_cast<memref::StoreOp>(&inner)) {
      info.stores.push_back(st);
      continue;
    }
    // Anything that passed Rule 5 and isn't load/store must be arith.*.
    // Record it for idiom-signature matching by patterns.
    info.bodyOps.push_back(&inner);
  }
  if (info.stores.size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "reject: body has "
                            << info.stores.size() << " stores (need 1) @"
                            << op.getLoc() << "\n");
    return failure();
  }
  if (info.loads.empty()) {
    // Trivial move from a scalar constant doesn't match any MVP idiom;
    // patterns always need at least one load to build a DSD from.
    LLVM_DEBUG(llvm::dbgs() << "reject: body has zero loads @"
                            << op.getLoc() << "\n");
    return failure();
  }

  // Rule 6 — induction variable uses restricted to memref index positions,
  // either directly or via arith.addi/arith.muli %iv, %const / affine.apply.
  for (Operation *user : info.inductionVar.getUsers()) {
    if (isa<memref::LoadOp, memref::StoreOp>(user)) {
      // Ok — check the IV is used as an index (not as the memref operand).
      for (auto [idx, operand] : llvm::enumerate(user->getOperands())) {
        if (operand != info.inductionVar) continue;
        if (auto ld = dyn_cast<memref::LoadOp>(user)) {
          if (idx == 0) {
            LLVM_DEBUG(llvm::dbgs() << "reject: IV used as memref operand @"
                                    << user->getLoc() << "\n");
            return failure();
          }
        } else if (auto st = dyn_cast<memref::StoreOp>(user)) {
          if (idx <= 1) {
            LLVM_DEBUG(llvm::dbgs() << "reject: IV used as value/memref @"
                                    << user->getLoc() << "\n");
            return failure();
          }
        }
      }
      continue;
    }
    if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp>(user)) {
      // OK — the other operand must be a constant for the index to be
      // affine in %iv with constant coefficients (enforced in Task 6).
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "reject: IV has non-index user: "
                            << user->getName() << " @" << user->getLoc()
                            << "\n");
    return failure();
  }

  // Rule 9 — collect loop-invariant scalar f32 values consumed by the body.
  // (Loop-invariant memrefs are captured per-access in Task 6.)
  llvm::DenseSet<Value> seen;
  for (Operation *bodyOp : info.bodyOps) {
    for (Value v : bodyOp->getOperands()) {
      if (seen.contains(v)) continue;
      seen.insert(v);
      // Block arg of the scf.for's body = the IV (already handled).
      if (auto bArg = dyn_cast<BlockArgument>(v)) {
        if (bArg == info.inductionVar) continue;
      }
      // Defined inside the loop? Skip.
      if (Operation *def = v.getDefiningOp()) {
        if (op->isAncestor(def)) continue;
      }
      // Loop-invariant scalar of supported type.
      if (v.getType().isF32()) {
        info.loopInvariantScalars.push_back(v);
      }
      // Non-f32 scalar invariants are allowed through analysis (they may
      // still be valid — e.g. a scalar constant that participates in
      // index math outside our IV).  Patterns that need the scalar will
      // re-check its type.
    }
  }

  // Rules 10-12 land in Task 6.
  LLVM_DEBUG(llvm::dbgs() << "reject: access-pattern check not implemented @"
                          << op.getLoc() << "\n");
  return failure();
```

- [ ] **Step 5.2: Build**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install
```
Expected: clean build. (`DenseSet` needs `#include "llvm/ADT/DenseSet.h"` — add it at the top of the .cpp if the build fails with undefined symbol.)

- [ ] **Step 5.3: Commit**

```bash
git add mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp
git commit -m "feat(csl-auto-vectorize): analyzeForLoop — purity predicate (rules 6-9)"
```

---

## Task 6: Access-pattern predicate rules (10-12 from spec §4.2)

**Goal:** For each load/store in the body, match the index against `coeff * %iv + k` with constant `coeff`, `k`; check in-bounds for signed offsets; check DSD field-width limits. Produces `DsdAccessPattern` entries.

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp`

- [ ] **Step 6.1: Add the index-matcher helper at file scope (above `analyzeForLoop`)**

```cpp
/// Match index expression `expr` against (coeff * %iv + k) with constant
/// integer `coeff`, `k`.  Returns {coeff, k} on match, failure otherwise.
static FailureOr<std::pair<int64_t, int64_t>>
matchAffineIndexInIV(Value expr, Value iv) {
  // Base case: the IV itself.
  if (expr == iv) return std::make_pair(int64_t(1), int64_t(0));

  // A constant — expressible as (0 * iv + c).
  if (auto c = getConstantIntValue(expr))
    return std::make_pair(int64_t(0), *c);

  Operation *def = expr.getDefiningOp();
  if (!def) return failure();

  // arith.addi a, b : i_or_index  →  coeff(a) + coeff(b), k(a) + k(b)
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto l = matchAffineIndexInIV(add.getLhs(), iv);
    auto r = matchAffineIndexInIV(add.getRhs(), iv);
    if (failed(l) || failed(r)) return failure();
    return std::make_pair(l->first + r->first, l->second + r->second);
  }

  // arith.subi a, b → coeff(a) - coeff(b), k(a) - k(b)
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    auto l = matchAffineIndexInIV(sub.getLhs(), iv);
    auto r = matchAffineIndexInIV(sub.getRhs(), iv);
    if (failed(l) || failed(r)) return failure();
    return std::make_pair(l->first - r->first, l->second - r->second);
  }

  // arith.muli iv, c  or  arith.muli c, iv  →  (coeff*c, k*c) IF the OTHER
  // side is a compile-time constant (otherwise reject — we can't multiply
  // two affine expressions and stay affine).
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto l = matchAffineIndexInIV(mul.getLhs(), iv);
    auto r = matchAffineIndexInIV(mul.getRhs(), iv);
    if (failed(l) || failed(r)) return failure();
    // One side must have coeff == 0 (be a pure constant).
    if (l->first == 0)
      return std::make_pair(l->second * r->first, l->second * r->second);
    if (r->first == 0)
      return std::make_pair(l->first * r->second, l->second * r->second);
    return failure();
  }

  return failure();
}
```

- [ ] **Step 6.2: Replace the trailing "not implemented" block with full rules 10-12**

Delete the final debug trace + `return failure()` and insert:

```cpp
  // Rules 10, 11, 12 — per-access analysis.
  auto analyzeAccess = [&](Value memRef, ValueRange indices,
                           Operation *accessOp) -> LogicalResult {
    auto ty = dyn_cast<MemRefType>(memRef.getType());
    if (!ty) {
      LLVM_DEBUG(llvm::dbgs() << "reject: non-memref access operand @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    if ((unsigned)ty.getRank() != indices.size()) {
      LLVM_DEBUG(llvm::dbgs() << "reject: rank/indices mismatch @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    if (ty.getRank() != 1) {
      // Rank-2 handling arrives in Task 7.
      LLVM_DEBUG(llvm::dbgs() << "reject: non-rank-1 access (rank "
                              << ty.getRank() << ") @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    if (!ty.hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "reject: dynamic memref shape @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }

    // Rule 10 — index is affine in IV.
    auto aff = matchAffineIndexInIV(indices[0], info.inductionVar);
    if (failed(aff)) {
      LLVM_DEBUG(llvm::dbgs() << "reject: non-affine index @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    int64_t coeff = aff->first;
    int64_t k = aff->second;

    // Rule 11 — access-in-bounds.
    // Iteration range is [lb, ub).
    int64_t ax0 = coeff * info.lb + k;
    int64_t ax1 = coeff * (info.ub - 1) + k;
    int64_t amin = std::min(ax0, ax1);
    int64_t amax = std::max(ax0, ax1);
    int64_t bufExtent = ty.getShape()[0];
    if (amin < 0 || amax >= bufExtent) {
      LLVM_DEBUG(llvm::dbgs() << "reject: access OOB [" << amin << ","
                              << amax << "] on buffer extent " << bufExtent
                              << " @" << accessOp->getLoc() << "\n");
      return failure();
    }

    // Rule 12 — DSD field widths (mem1d: stride i8, offset i16).
    if (coeff < kMinMem1dStride || coeff > kMaxMem1dStride) {
      LLVM_DEBUG(llvm::dbgs() << "reject: stride " << coeff
                              << " outside mem1d i8 range @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }
    if (amin < kMinDsdOffset || amin > kMaxDsdOffset) {
      LLVM_DEBUG(llvm::dbgs() << "reject: effective offset " << amin
                              << " outside i16 range @"
                              << accessOp->getLoc() << "\n");
      return failure();
    }

    DsdAccessPattern ap;
    ap.buffer = memRef;
    ap.strides = {coeff};
    ap.offsets = {amin};                 // effective post-clip start offset
    ap.rank = 1;
    info.accesses.push_back(ap);
    return success();
  };

  for (auto &ld : info.loads)
    if (failed(analyzeAccess(ld.getMemRef(), ld.getIndices(), ld)))
      return failure();
  for (auto &st : info.stores)
    if (failed(analyzeAccess(st.getMemRef(), st.getIndices(), st)))
      return failure();

  LLVM_DEBUG(llvm::dbgs() << "accept: LoopIdiom extent=" << info.extent
                          << " loads=" << info.loads.size()
                          << " stores=" << info.stores.size() << " @"
                          << op.getLoc() << "\n");
  return info;
```

- [ ] **Step 6.3: Build**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install
```
Expected: clean build.

- [ ] **Step 6.4: Commit**

```bash
git add mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp
git commit -m "feat(csl-auto-vectorize): analyzeForLoop — access-pattern predicate (rules 10-12)"
```

---

## Task 7: Rank-2 detection

**Goal:** When the outer loop's body is exactly one nested `scf.for`, recurse into it and combine into a rank-2 `LoopIdiom`. Everything else unchanged.

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp`

- [ ] **Step 7.1: Detect perfect rank-2 nest upfront in `analyzeForLoop`**

After Rule 5's nested-control-flow check, insert a new branch that intercepts the rank-2 case *before* it is rejected. Restructure Rule 5 like so (replace the existing Rule 5 block):

```cpp
  // Rule 5 — no nested control flow or calls in body. Exception: exactly
  // one nested scf.for whose body is the rank-1 shape (perfect 2-deep nest).
  scf::ForOp innerFor;
  {
    unsigned nonTermCount = 0;
    for (Operation &inner : op.getBody()->without_terminator()) {
      nonTermCount++;
      if (auto nested = dyn_cast<scf::ForOp>(&inner)) {
        if (innerFor) {
          LLVM_DEBUG(llvm::dbgs() << "reject: multiple nested loops @"
                                  << inner.getLoc() << "\n");
          return failure();
        }
        innerFor = nested;
      }
    }
    if (innerFor && nonTermCount != 1) {
      LLVM_DEBUG(llvm::dbgs() << "reject: non-perfect rank-2 nest @"
                              << innerFor.getLoc() << "\n");
      return failure();
    }
  }

  if (innerFor) {
    // Outer-loop body is the one inner scf.for — no body ops to classify
    // here.  Perform shape + extent checks on the inner loop, then treat
    // the inner loop's body as the "real" body for rules 6-12.
    std::optional<int64_t> ilb = getConstantIntValue(innerFor.getLowerBound());
    std::optional<int64_t> iub = getConstantIntValue(innerFor.getUpperBound());
    std::optional<int64_t> istep = getConstantIntValue(innerFor.getStep());
    if (!ilb || !iub || !istep || *istep != 1) {
      LLVM_DEBUG(llvm::dbgs() << "reject: inner loop non-constant or step!=1 @"
                              << innerFor.getLoc() << "\n");
      return failure();
    }
    info.isRank2 = true;
    info.innerLb = *ilb;
    info.innerUb = *iub;
    info.innerStep = 1;
    info.innerExtent = info.innerUb - info.innerLb;
    info.innerInductionVar = innerFor.getInductionVar();
    if (info.innerExtent <= 0 || info.innerExtent > kMaxDsdExtent ||
        info.innerLb < 0) {
      LLVM_DEBUG(llvm::dbgs() << "reject: inner extent out of range @"
                              << innerFor.getLoc() << "\n");
      return failure();
    }
    if (!innerFor.getRegion().hasOneBlock() ||
        innerFor.getNumRegionIterArgs() != 0) {
      LLVM_DEBUG(llvm::dbgs() << "reject: inner loop iter_args/non-single-block @"
                              << innerFor.getLoc() << "\n");
      return failure();
    }
  }

  // Loops from here on — iterate over innerFor's body if rank-2, else
  // the outer body.  Bind a helper.
  Block *bodyBlock = innerFor ? innerFor.getBody() : op.getBody();
  Value primaryIV = info.inductionVar;    // outer IV in rank-2
  Value secondaryIV = innerFor ? innerFor.getInductionVar() : Value{};
```

Then in Rule 5's tail (the "not load/store → body op" loop from Task 5), switch the iterator to `bodyBlock->without_terminator()` — same change elsewhere in the function.

- [ ] **Step 7.2: Update Rule 6 (IV uses) to include both IVs when rank-2**

Replace the Rule 6 loop with:

```cpp
  auto checkIVUses = [&](Value iv, const char *which) -> LogicalResult {
    for (Operation *user : iv.getUsers()) {
      if (isa<memref::LoadOp, memref::StoreOp>(user)) {
        for (auto [idx, operand] : llvm::enumerate(user->getOperands())) {
          if (operand != iv) continue;
          if (auto ld = dyn_cast<memref::LoadOp>(user)) {
            if (idx == 0) {
              LLVM_DEBUG(llvm::dbgs() << "reject: " << which
                                      << " IV used as memref operand @"
                                      << user->getLoc() << "\n");
              return failure();
            }
          } else if (auto st = dyn_cast<memref::StoreOp>(user)) {
            if (idx <= 1) {
              LLVM_DEBUG(llvm::dbgs() << "reject: " << which
                                      << " IV used as value/memref @"
                                      << user->getLoc() << "\n");
              return failure();
            }
          }
        }
        continue;
      }
      if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp>(user)) continue;
      // Outer IV used by inner scf.for (as nothing directly — the inner
      // uses ITS OWN iv).  So non-memref, non-arith users should be empty.
      if (isa<scf::ForOp>(user)) continue;
      LLVM_DEBUG(llvm::dbgs() << "reject: " << which
                              << " IV has non-index user: "
                              << user->getName() << "\n");
      return failure();
    }
    return success();
  };
  if (failed(checkIVUses(primaryIV, "outer"))) return failure();
  if (info.isRank2 && failed(checkIVUses(secondaryIV, "inner")))
    return failure();
```

- [ ] **Step 7.3: Update the access-pattern analyzer to match in two IVs for rank-2**

Extend `matchAffineIndexInIV` to a two-IV variant, or write a thin wrapper:

```cpp
/// Like matchAffineIndexInIV but handles a pair of IVs; returns
/// (coeff_primary, coeff_secondary, k).  Either coeff may be 0.
static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
matchAffineIndexInIVPair(Value expr, Value ivA, Value ivB) {
  auto a = matchAffineIndexInIV(expr, ivA);
  if (succeeded(a) && ivB == Value{}) return std::make_tuple(a->first, int64_t(0), a->second);
  // Try treating expr as affine in ivB with ivA contributing via a constant.
  // Simpler: call matchAffineIndexInIV with one IV at a time after splitting.
  // For MVP: require index expression to decompose as (cA*ivA + cB*ivB + k)
  // with additive structure.  We handle it recursively:
  if (expr == ivA) return std::make_tuple(int64_t(1), int64_t(0), int64_t(0));
  if (expr == ivB) return std::make_tuple(int64_t(0), int64_t(1), int64_t(0));
  if (auto c = getConstantIntValue(expr))
    return std::make_tuple(int64_t(0), int64_t(0), *c);

  Operation *def = expr.getDefiningOp();
  if (!def) return failure();
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto l = matchAffineIndexInIVPair(add.getLhs(), ivA, ivB);
    auto r = matchAffineIndexInIVPair(add.getRhs(), ivA, ivB);
    if (failed(l) || failed(r)) return failure();
    return std::make_tuple(std::get<0>(*l) + std::get<0>(*r),
                           std::get<1>(*l) + std::get<1>(*r),
                           std::get<2>(*l) + std::get<2>(*r));
  }
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    auto l = matchAffineIndexInIVPair(sub.getLhs(), ivA, ivB);
    auto r = matchAffineIndexInIVPair(sub.getRhs(), ivA, ivB);
    if (failed(l) || failed(r)) return failure();
    return std::make_tuple(std::get<0>(*l) - std::get<0>(*r),
                           std::get<1>(*l) - std::get<1>(*r),
                           std::get<2>(*l) - std::get<2>(*r));
  }
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto l = matchAffineIndexInIVPair(mul.getLhs(), ivA, ivB);
    auto r = matchAffineIndexInIVPair(mul.getRhs(), ivA, ivB);
    if (failed(l) || failed(r)) return failure();
    bool lIsConst = std::get<0>(*l) == 0 && std::get<1>(*l) == 0;
    bool rIsConst = std::get<0>(*r) == 0 && std::get<1>(*r) == 0;
    if (lIsConst)
      return std::make_tuple(std::get<2>(*l) * std::get<0>(*r),
                             std::get<2>(*l) * std::get<1>(*r),
                             std::get<2>(*l) * std::get<2>(*r));
    if (rIsConst)
      return std::make_tuple(std::get<0>(*l) * std::get<2>(*r),
                             std::get<1>(*l) * std::get<2>(*r),
                             std::get<2>(*l) * std::get<2>(*r));
    return failure();
  }
  return failure();
}
```

- [ ] **Step 7.4: Refactor `analyzeAccess` to use the pair-matcher for rank-2**

Replace the single-rank body of `analyzeAccess` with rank-aware code:

```cpp
    if (ty.getRank() == 1) {
      if (info.isRank2) {
        // Outer rank-2 nest over rank-1 accesses: index must be affine only
        // in the inner IV, outer coefficient must be zero.
        auto aff = matchAffineIndexInIVPair(indices[0], primaryIV, secondaryIV);
        if (failed(aff) || std::get<0>(*aff) != 0) {
          LLVM_DEBUG(llvm::dbgs() << "reject: inner-only-indexed rank-1 access"
                                     " but outer-coeff non-zero @"
                                  << accessOp->getLoc() << "\n");
          return failure();
        }
        int64_t coeff = std::get<1>(*aff);
        int64_t kk = std::get<2>(*aff);
        // Range over [innerLb, innerUb)
        int64_t x0 = coeff * info.innerLb + kk;
        int64_t x1 = coeff * (info.innerUb - 1) + kk;
        int64_t amin = std::min(x0, x1);
        int64_t amax = std::max(x0, x1);
        int64_t bufExtent = ty.getShape()[0];
        if (amin < 0 || amax >= bufExtent) {
          LLVM_DEBUG(llvm::dbgs() << "reject: rank-1 (inside rank-2) OOB @"
                                  << accessOp->getLoc() << "\n");
          return failure();
        }
        DsdAccessPattern ap{memRef, {coeff}, {amin}, 1};
        info.accesses.push_back(ap);
        return success();
      }
      // Pure rank-1 (existing logic from Task 6 — keep it).
      auto aff = matchAffineIndexInIV(indices[0], primaryIV);
      if (failed(aff)) { /* reject */ return failure(); }
      int64_t coeff = aff->first, k = aff->second;
      int64_t x0 = coeff * info.lb + k, x1 = coeff * (info.ub - 1) + k;
      int64_t amin = std::min(x0, x1), amax = std::max(x0, x1);
      int64_t bufExtent = ty.getShape()[0];
      if (amin < 0 || amax >= bufExtent) return failure();
      if (coeff < kMinMem1dStride || coeff > kMaxMem1dStride) return failure();
      if (amin < kMinDsdOffset || amin > kMaxDsdOffset) return failure();
      DsdAccessPattern ap{memRef, {coeff}, {amin}, 1};
      info.accesses.push_back(ap);
      return success();
    }
    if (ty.getRank() == 2 && info.isRank2) {
      // Rank-2 memref inside rank-2 nest — each dim gets its own (coeff, k).
      int64_t strides[2] = {0, 0};
      int64_t offsets[2] = {0, 0};
      // Dim 0 uses outer IV only; dim 1 uses inner IV only.
      auto affOuter =
          matchAffineIndexInIVPair(indices[0], primaryIV, secondaryIV);
      auto affInner =
          matchAffineIndexInIVPair(indices[1], primaryIV, secondaryIV);
      if (failed(affOuter) || failed(affInner)) return failure();
      if (std::get<1>(*affOuter) != 0 || std::get<0>(*affInner) != 0) {
        LLVM_DEBUG(llvm::dbgs() << "reject: rank-2 access mixes IVs @"
                                << accessOp->getLoc() << "\n");
        return failure();
      }
      strides[0] = std::get<0>(*affOuter);
      offsets[0] = strides[0] * info.lb + std::get<2>(*affOuter);
      strides[1] = std::get<1>(*affInner);
      offsets[1] = strides[1] * info.innerLb + std::get<2>(*affInner);
      int64_t M = ty.getShape()[0], N = ty.getShape()[1];
      if (offsets[0] < 0 ||
          offsets[0] + (info.extent - 1) * strides[0] >= M ||
          offsets[1] < 0 ||
          offsets[1] + (info.innerExtent - 1) * strides[1] >= N) {
        LLVM_DEBUG(llvm::dbgs() << "reject: rank-2 OOB @"
                                << accessOp->getLoc() << "\n");
        return failure();
      }
      for (int64_t s : strides)
        if (s < kMinMem4dStride || s > kMaxMem4dStride) return failure();
      for (int64_t o : offsets)
        if (o < kMinDsdOffset || o > kMaxDsdOffset) return failure();
      DsdAccessPattern ap;
      ap.buffer = memRef;
      ap.strides = {strides[0], strides[1]};
      ap.offsets = {offsets[0], offsets[1]};
      ap.rank = 2;
      info.accesses.push_back(ap);
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "reject: rank mismatch @"
                            << accessOp->getLoc() << "\n");
    return failure();
```

- [ ] **Step 7.5: Build**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install
```
Expected: clean build.

- [ ] **Step 7.6: Commit**

```bash
git add mlir/lib/Dialect/CSL/Transforms/LoopIdiomAnalysis.cpp
git commit -m "feat(csl-auto-vectorize): analyzeForLoop — rank-2 nested-loop detection"
```

---

## Task 8: `PatternsCommon.h` — shared IR-construction helpers

**Files:**
- Create: `mlir/lib/Dialect/CSL/Transforms/Patterns/PatternsCommon.h`

- [ ] **Step 8.1: Write the helper header**

```cpp
//===- PatternsCommon.h - DSD IR-construction helpers -----------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Shared IR-construction helpers for -csl-auto-vectorize pattern rewrites.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_PATTERNS_COMMON_H
#define AIR_DIALECT_CSL_TRANSFORMS_PATTERNS_COMMON_H

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSL/Transforms/LoopIdiomAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

namespace xilinx {
namespace air {

/// Build the memref SSA value that the DSD will wrap, applying
/// memref.subview where offsets/strides differ from the root buffer's
/// natural layout.  Returns an SSA value of memref type suitable as
/// csl.get_mem_dsd operand.
inline mlir::Value buildSubviewForAccess(mlir::PatternRewriter &rewriter,
                                         mlir::Location loc,
                                         const LoopIdiom &info,
                                         const DsdAccessPattern &ap) {
  using namespace mlir;
  auto rootTy = cast<MemRefType>(ap.buffer.getType());
  bool needsSubview = false;
  for (int64_t o : ap.offsets) if (o != 0) needsSubview = true;
  for (int64_t s : ap.strides) if (s != 1) needsSubview = true;
  // Rank-1: if the effective extent equals the buffer extent AND offset=0
  // AND stride=1, pass the raw memref.
  if (!needsSubview) return ap.buffer;

  // Compose subview offsets/sizes/strides.
  SmallVector<OpFoldResult> offsets, sizes, strides;
  if (ap.rank == 1) {
    offsets.push_back(rewriter.getIndexAttr(ap.offsets[0]));
    sizes.push_back(rewriter.getIndexAttr(info.extent));
    strides.push_back(rewriter.getIndexAttr(ap.strides[0]));
  } else {
    assert(ap.rank == 2);
    offsets.push_back(rewriter.getIndexAttr(ap.offsets[0]));
    offsets.push_back(rewriter.getIndexAttr(ap.offsets[1]));
    sizes.push_back(rewriter.getIndexAttr(info.extent));
    sizes.push_back(rewriter.getIndexAttr(info.innerExtent));
    strides.push_back(rewriter.getIndexAttr(ap.strides[0]));
    strides.push_back(rewriter.getIndexAttr(ap.strides[1]));
  }
  auto resultTy = cast<MemRefType>(
      memref::SubViewOp::inferResultType(rootTy, offsets, sizes, strides));
  return rewriter.create<memref::SubViewOp>(loc, resultTy, ap.buffer,
                                            offsets, sizes, strides);
}

/// Build `csl.get_mem_dsd` on the given memref.
inline mlir::Value buildGetMemDsd(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value memRef) {
  using namespace mlir;
  auto dsdTy = xilinx::csl::DsdType::get(rewriter.getContext());
  return rewriter.create<xilinx::csl::GetMemDsdOp>(loc, dsdTy, memRef);
}

/// Build a `csl.builtin_call "<callee>"(args...)` with no results.
inline void buildBuiltinCall(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, llvm::StringRef callee,
                             mlir::ValueRange args) {
  using namespace mlir;
  rewriter.create<xilinx::csl::BuiltinCallOp>(
      loc,
      /*results=*/TypeRange{},
      /*callee=*/rewriter.getStringAttr(callee),
      /*module=*/Value{},
      /*operands=*/args);
}

} // namespace air
} // namespace xilinx

#endif // AIR_DIALECT_CSL_TRANSFORMS_PATTERNS_COMMON_H
```

- [ ] **Step 8.2: Commit (header-only; no build change yet)**

```bash
git add mlir/lib/Dialect/CSL/Transforms/Patterns/PatternsCommon.h
git commit -m "feat(csl-auto-vectorize): add PatternsCommon helpers (subview, get_mem_dsd, builtin_call)"
```

---

## Task 9: FaddsPattern (first end-to-end idiom)

**Files:**
- Create: `mlir/lib/Dialect/CSL/Transforms/Patterns/ElementwisePatterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fadds.mlir`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp` (wire up the pattern)
- Modify: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt` (add new source)

- [ ] **Step 9.1: Write the failing test — `fadds.mlir`**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = a[i] + b[i]  →  @fadds(dc, da, db)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: %[[DA:.+]] = csl.get_mem_dsd %a : memref<1024xf32> -> !csl.dsd
// CHECK: %[[DB:.+]] = csl.get_mem_dsd %b : memref<1024xf32> -> !csl.dsd
// CHECK: %[[DC:.+]] = csl.get_mem_dsd %c : memref<1024xf32> -> !csl.dsd
// CHECK: csl.builtin_call "fadds"(%[[DC]], %[[DA]], %[[DB]]) : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @vecadd {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<1024xf32>
      %b = csl.var @b : memref<1024xf32>
      %c = csl.var @c : memref<1024xf32>
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
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
```

- [ ] **Step 9.2: Run test to verify it fails**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && \
  lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/fadds.mlir -v
```
Expected: FAIL, `scf.for` still present in output (pass hasn't implemented any patterns yet).

- [ ] **Step 9.3: Create `mlir/lib/Dialect/CSL/Transforms/Patterns/ElementwisePatterns.cpp`**

```cpp
//===- ElementwisePatterns.cpp -----------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Elementwise binary DSD idiom patterns for -csl-auto-vectorize:
//   @fadds  — arith.addf c = a + b
//   @fsubs  — arith.subf c = a - b   (added in Task 10)
//   @fmuls  — arith.mulf c = a * b   (added in Task 11)
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/Patterns/PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// Match body signature:
///   %0 = memref.load a[idx] : memref<_xf32>
///   %1 = memref.load b[idx] : memref<_xf32>
///   %2 = arith.addf %0, %1 : f32
///   memref.store %2, c[idx]
struct FaddsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;

    // Shape: 2 loads, 1 store, 1 body op (arith.addf), all f32.
    if (info.loads.size() != 2 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1)
      return failure();
    auto add = dyn_cast<arith::AddFOp>(info.bodyOps[0]);
    if (!add) return failure();
    if (!add.getType().isF32()) return failure();

    // The store's value must be the addf's result.
    if (info.stores[0].getValue() != add.getResult()) return failure();
    // The addf's operands must be the two loads (order-independent).
    Value l0 = info.loads[0].getResult();
    Value l1 = info.loads[1].getResult();
    Value ra = add.getLhs(), rb = add.getRhs();
    if (!((ra == l0 && rb == l1) || (ra == l1 && rb == l0)))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FaddsPattern @" << op.getLoc() << "\n");

    // accesses[0..1] correspond to loads[0..1]; accesses[2] to stores[0].
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrB = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[2]);
    Value dA  = buildGetMemDsd(rewriter, loc, mrA);
    Value dB  = buildGetMemDsd(rewriter, loc, mrB);
    Value dC  = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fadds", {dC, dA, dB});

    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx {
namespace air {

void populateElementwisePatterns(RewritePatternSet &patterns) {
  patterns.add<FaddsPattern>(patterns.getContext());
}

} // namespace air
} // namespace xilinx
```

- [ ] **Step 9.4: Wire the pattern into the pass driver**

Edit `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp`. Add at the top with other includes:

```cpp
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
```

Declare the population hook near the anonymous namespace (before the pass class):

```cpp
namespace xilinx { namespace air {
  void populateElementwisePatterns(mlir::RewritePatternSet &patterns);
}}
```

Replace the empty `runOnOperation()` body with:

```cpp
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    xilinx::air::populateElementwisePatterns(patterns);
    // More populators land in later tasks.

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
```

- [ ] **Step 9.5: Add the new source to CMakeLists**

Edit `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`:

```cmake
add_mlir_library(
  CSLTransforms
  CSLInferExports.cpp
  CSLAutoVectorize.cpp
  LoopIdiomAnalysis.cpp
  Passes.cpp
  Patterns/ElementwisePatterns.cpp    # ADD

  PARTIAL_SOURCES_INTENDED
  ...)
```

- [ ] **Step 9.6: Build**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install
```
Expected: clean build.

- [ ] **Step 9.7: Run the test and verify it passes**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && \
  lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/fadds.mlir -v
```
Expected: PASS.

- [ ] **Step 9.8: Commit**

```bash
git add mlir/lib/Dialect/CSL/Transforms/Patterns/ \
        mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp \
        mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt \
        mlir/test/Dialect/CSL/Transforms/auto-vectorize/fadds.mlir
git commit -m "feat(csl-auto-vectorize): FaddsPattern — recognize c[i]=a[i]+b[i] as @fadds"
```

---

## Task 10: FsubsPattern

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/Patterns/ElementwisePatterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fsubs.mlir`

- [ ] **Step 10.1: Write the failing test — `fsubs.mlir`**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = a[i] - b[i]  →  @fsubs(dc, da, db)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fsubs"(%{{.*}}, %{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @vecsub {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<512xf32>
      %b = csl.var @b : memref<512xf32>
      %c = csl.var @c : memref<512xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 512 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<512xf32>
          %vb = memref.load %b[%i] : memref<512xf32>
          %vc = arith.subf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<512xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
```

- [ ] **Step 10.2: Run to verify failure**

```bash
lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/fsubs.mlir -v
```
Expected: FAIL (no FsubsPattern yet).

- [ ] **Step 10.3: Add the pattern to `ElementwisePatterns.cpp`**

Copy the FaddsPattern class, rename to `FsubsPattern`, change:
- `dyn_cast<arith::AddFOp>` → `dyn_cast<arith::SubFOp>`
- `"fadds"` → `"fsubs"`
- **Remove the operand-order symmetry** — subtraction is *not* commutative. Only accept `ra == l0 && rb == l1`.

```cpp
struct FsubsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 2 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1)
      return failure();
    auto sub = dyn_cast<arith::SubFOp>(info.bodyOps[0]);
    if (!sub || !sub.getType().isF32()) return failure();
    if (info.stores[0].getValue() != sub.getResult()) return failure();
    Value l0 = info.loads[0].getResult();
    Value l1 = info.loads[1].getResult();
    if (!(sub.getLhs() == l0 && sub.getRhs() == l1)) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FsubsPattern @" << op.getLoc() << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrB = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[2]);
    Value dA  = buildGetMemDsd(rewriter, loc, mrA);
    Value dB  = buildGetMemDsd(rewriter, loc, mrB);
    Value dC  = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fsubs", {dC, dA, dB});
    rewriter.eraseOp(op);
    return success();
  }
};
```

Update `populateElementwisePatterns`:

```cpp
void populateElementwisePatterns(RewritePatternSet &patterns) {
  patterns.add<FaddsPattern, FsubsPattern>(patterns.getContext());
}
```

- [ ] **Step 10.4: Build and re-run**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install && \
  lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/fsubs.mlir -v
```
Expected: PASS.

- [ ] **Step 10.5: Commit**

```bash
git add mlir/lib/Dialect/CSL/Transforms/Patterns/ElementwisePatterns.cpp \
        mlir/test/Dialect/CSL/Transforms/auto-vectorize/fsubs.mlir
git commit -m "feat(csl-auto-vectorize): FsubsPattern"
```

---

## Task 11: FmulsPattern

Identical structure to `FsubsPattern` — a new pattern class + a new test file. Follow the same five steps.

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/Patterns/ElementwisePatterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fmuls.mlir`

**Deltas from Task 10:**
- In the test, `arith.subf` → `arith.mulf`, `"fsubs"` → `"fmuls"`, rename module/wafer to `vecmul`.
- In the pattern: `dyn_cast<arith::SubFOp>` → `dyn_cast<arith::MulFOp>`, `"fsubs"` → `"fmuls"`. Multiplication **is commutative** so accept both operand orders like `FaddsPattern`.
- Register `FmulsPattern` in `populateElementwisePatterns`.

Commit: `feat(csl-auto-vectorize): FmulsPattern`

---

## Task 12: FmacsPattern (FMA fusion — two-op body)

**Files:**
- Create: `mlir/lib/Dialect/CSL/Transforms/Patterns/FmaPattern.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fmacs.mlir`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp` (add new populator call)
- Modify: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt` (add new source)

- [ ] **Step 12.1: Write the failing test — `fmacs.mlir`**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = a[i] * b[i] + c[i]  →  @fmacs(dc, dc, da, db)
// (The expected SDK form; b is placed after da per 4-operand @fmacs signature.)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fmacs"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd, !csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @vecmac {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<1024xf32>
      %b = csl.var @b : memref<1024xf32>
      %c = csl.var @c : memref<1024xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<1024xf32>
          %vb = memref.load %b[%i] : memref<1024xf32>
          %vc = memref.load %c[%i] : memref<1024xf32>
          %m  = arith.mulf %va, %vb : f32
          %s  = arith.addf %m, %vc : f32
          memref.store %s, %c[%i] : memref<1024xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
```

- [ ] **Step 12.2: Run to verify failure**

```bash
lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/fmacs.mlir -v
```
Expected: FAIL.

- [ ] **Step 12.3: Create `mlir/lib/Dialect/CSL/Transforms/Patterns/FmaPattern.cpp`**

```cpp
//===- FmaPattern.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/Patterns/PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// Match body signature:
///   %va = load a[i]; %vb = load b[i]; %vc = load c[i];
///   %m  = mulf %va, %vb : f32
///   %s  = addf %m,  %vc : f32     (or addf %vc, %m — commutative)
///   store %s, c[i]
/// The store target must match one of the loaded buffers (accumulator).
struct FmacsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;

    // 3 loads (a, b, accumulator), 1 store, 2 body ops (mulf, addf).
    if (info.loads.size() != 3 || info.stores.size() != 1 ||
        info.bodyOps.size() != 2)
      return failure();

    // Identify the two ops (program order).
    auto mul = dyn_cast<arith::MulFOp>(info.bodyOps[0]);
    auto add = dyn_cast<arith::AddFOp>(info.bodyOps[1]);
    if (!mul || !add) return failure();
    if (!mul.getType().isF32() || !add.getType().isF32()) return failure();

    // addf's operands are (mul.result, X) or (X, mul.result); X must be a
    // load result.
    Value mulRes = mul.getResult();
    Value accLoadVal;
    if (add.getLhs() == mulRes)
      accLoadVal = add.getRhs();
    else if (add.getRhs() == mulRes)
      accLoadVal = add.getLhs();
    else
      return failure();

    // mul's operands must both be load results.
    Value mA = mul.getLhs(), mB = mul.getRhs();
    auto isLoadResult = [&](Value v) {
      return llvm::any_of(info.loads, [&](memref::LoadOp ld) {
        return ld.getResult() == v;
      });
    };
    if (!isLoadResult(mA) || !isLoadResult(mB) || !isLoadResult(accLoadVal))
      return failure();

    // The store target buffer must be the one accLoadVal was loaded from,
    // at the same index (analyzer ensures index = iv).
    memref::LoadOp accLoad;
    for (auto &ld : info.loads)
      if (ld.getResult() == accLoadVal) { accLoad = ld; break; }
    if (!accLoad) return failure();
    if (info.stores[0].getMemRef() != accLoad.getMemRef()) return failure();
    if (info.stores[0].getValue() != add.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FmacsPattern @" << op.getLoc() << "\n");

    // Identify access-pattern indices for a, b, c_acc, c_store.
    // info.accesses is in load-then-store order; we need to look up by buffer.
    auto findAp = [&](Value buf) -> const DsdAccessPattern * {
      for (const auto &ap : info.accesses)
        if (ap.buffer == buf) return &ap;
      return nullptr;
    };
    const auto *apA = findAp(
        cast<memref::LoadOp>(mA.getDefiningOp()).getMemRef());
    const auto *apB = findAp(
        cast<memref::LoadOp>(mB.getDefiningOp()).getMemRef());
    const auto *apC = findAp(accLoad.getMemRef());
    if (!apA || !apB || !apC) return failure();

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, *apA);
    Value mrB = buildSubviewForAccess(rewriter, loc, info, *apB);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, *apC);
    Value dA  = buildGetMemDsd(rewriter, loc, mrA);
    Value dB  = buildGetMemDsd(rewriter, loc, mrB);
    Value dC  = buildGetMemDsd(rewriter, loc, mrC);
    // @fmacs(dest, src_acc, src_a, src_b) — acc is read *and* written.
    buildBuiltinCall(rewriter, loc, "fmacs", {dC, dC, dA, dB});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx {
namespace air {

void populateFmaPattern(RewritePatternSet &patterns) {
  patterns.add<FmacsPattern>(patterns.getContext());
}

} // namespace air
} // namespace xilinx
```

- [ ] **Step 12.4: Wire in the pass driver**

Edit `CSLAutoVectorize.cpp` — add the forward decl and call:

```cpp
namespace xilinx { namespace air {
  void populateElementwisePatterns(mlir::RewritePatternSet &);
  void populateFmaPattern(mlir::RewritePatternSet &);        // NEW
}}
```

```cpp
void runOnOperation() override {
  RewritePatternSet patterns(&getContext());
  xilinx::air::populateElementwisePatterns(patterns);
  xilinx::air::populateFmaPattern(patterns);                 // NEW
  ...
}
```

Edit `CMakeLists.txt` — add `Patterns/FmaPattern.cpp`.

- [ ] **Step 12.5: Build + run test**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja install && \
  lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/fmacs.mlir -v
```
Expected: PASS.

**Pattern-ordering note.** `FmacsPattern` should take priority over the sum of `FaddsPattern` + `FmulsPattern` on the same IR. Because all patterns match on `scf::ForOp`, the greedy driver picks the one with the higher benefit. Set `FmacsPattern`'s benefit in its constructor: change `OpRewritePattern::OpRewritePattern;` to:

```cpp
FmacsPattern(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}
```

(Fadds / Fsubs / Fmuls use default benefit=1.) Without this, whether the FMA or the plain add wins is an order-of-registration accident — we want the FMA because it matches the full 2-op body.

- [ ] **Step 12.6: Run the full positive-test directory to confirm no regressions**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && \
  lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/ -v
```
Expected: all 3 tests (fadds, fsubs, fmuls — fmacs if Step 12.5 already passed) PASS.

- [ ] **Step 12.7: Commit**

```bash
git add mlir/lib/Dialect/CSL/Transforms/Patterns/FmaPattern.cpp \
        mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp \
        mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt \
        mlir/test/Dialect/CSL/Transforms/auto-vectorize/fmacs.mlir
git commit -m "feat(csl-auto-vectorize): FmacsPattern with benefit=2 (FMA fusion)"
```

---

## Task 13: FmovsPattern

**Files:**
- Create: `mlir/lib/Dialect/CSL/Transforms/Patterns/MovePatterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fmovs.mlir`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`

- [ ] **Step 13.1: Write the failing test**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = a[i]  →  @fmovs(dc, da)
// (pure buffer copy, no arith op in body)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fmovs"(%{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @veccopy {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 256 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %v = memref.load %a[%i] : memref<256xf32>
          memref.store %v, %c[%i] : memref<256xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
```

- [ ] **Step 13.2: Run to verify failure**

- [ ] **Step 13.3: Create `mlir/lib/Dialect/CSL/Transforms/Patterns/MovePatterns.cpp`**

```cpp
//===- MovePatterns.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/Patterns/PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// c[i] = a[i] on f32 — one load, one store, zero body ops.
struct FmovsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 1 || info.stores.size() != 1 ||
        !info.bodyOps.empty())
      return failure();
    if (!info.loads[0].getResult().getType().isF32()) return failure();
    if (info.stores[0].getValue() != info.loads[0].getResult())
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "match: FmovsPattern @" << op.getLoc() << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value dA = buildGetMemDsd(rewriter, loc, mrA);
    Value dC = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fmovs", {dC, dA});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx { namespace air {
void populateMovePatterns(RewritePatternSet &patterns) {
  patterns.add<FmovsPattern>(patterns.getContext());
}
}}
```

- [ ] **Step 13.4: Wire it in** (same pattern as Task 12 — forward decl, call `populateMovePatterns`, add source to CMakeLists). Build, run test.

- [ ] **Step 13.5: Commit**

```bash
git commit -m "feat(csl-auto-vectorize): FmovsPattern — c[i]=a[i] as @fmovs"
```

---

## Task 14: FnegsPattern

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/Patterns/MovePatterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fnegs.mlir`

**Deltas from Task 13:**
- Test body: `%0 = load; %1 = arith.negf %0; store %1, c[i]`.
- Expected check: `csl.builtin_call "fnegs"(...) : (!csl.dsd, !csl.dsd) -> ()`.
- Pattern: 1 load, 1 store, 1 body op = `arith::NegFOp`. Store value = neg result; neg operand = load result.

Add the new pattern class (copy FmovsPattern, check `info.bodyOps.size() == 1` + `auto neg = dyn_cast<arith::NegFOp>(info.bodyOps[0])`, callee `"fnegs"`). Register in `populateMovePatterns`. Build, test, commit.

---

## Task 15: FmulsScalarPattern (scalar broadcast)

**Files:**
- Create: `mlir/lib/Dialect/CSL/Transforms/Patterns/ScalarBroadcastPatterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fmuls_scalar.mlir`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`

- [ ] **Step 15.1: Write the failing test**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = alpha * a[i]  (alpha is loop-invariant scalar)  →  @fmuls(dc, da, alpha)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: csl.builtin_call "fmuls"(%{{.*}}, %{{.*}}, %{{.*}}) : (!csl.dsd, !csl.dsd, f32) -> ()

module {
  csl.wafer @scale {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      %c = csl.var @c : memref<128xf32>
      csl.func @compute {
        %alpha = arith.constant 2.0 : f32
        %c0 = arith.constant 0 : index
        %n  = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<128xf32>
          %v  = arith.mulf %va, %alpha : f32
          memref.store %v, %c[%i] : memref<128xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
```

- [ ] **Step 15.2: Run to verify failure.**

- [ ] **Step 15.3: Create `ScalarBroadcastPatterns.cpp`**

```cpp
//===- ScalarBroadcastPatterns.cpp ------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/Patterns/PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// c[i] = alpha * a[i], alpha loop-invariant f32.  1 load, 1 store, 1 mulf;
/// one operand of mulf is the load, the other is in loopInvariantScalars.
struct FmulsScalarPattern : public OpRewritePattern<scf::ForOp> {
  FmulsScalarPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (info.loads.size() != 1 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1)
      return failure();
    auto mul = dyn_cast<arith::MulFOp>(info.bodyOps[0]);
    if (!mul || !mul.getType().isF32()) return failure();
    Value loadVal = info.loads[0].getResult();
    Value scalar;
    if (mul.getLhs() == loadVal) scalar = mul.getRhs();
    else if (mul.getRhs() == loadVal) scalar = mul.getLhs();
    else return failure();
    if (!llvm::is_contained(info.loopInvariantScalars, scalar))
      return failure();
    if (info.stores[0].getValue() != mul.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: FmulsScalarPattern @" << op.getLoc()
                            << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value dA = buildGetMemDsd(rewriter, loc, mrA);
    Value dC = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fmuls", {dC, dA, scalar});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx { namespace air {
void populateScalarBroadcastPatterns(RewritePatternSet &patterns) {
  patterns.add<FmulsScalarPattern>(patterns.getContext());
}
}}
```

- [ ] **Step 15.4: Wire in + CMakeLists. Build. Run. Commit.**

**Benefit note.** `FmulsScalarPattern` must have benefit=2 to beat `FmulsPattern` (benefit=1) — otherwise a scalar-broadcast loop could match `FmulsPattern` first and fail (no second load) and *never retry* `FmulsScalarPattern`. Actually — the greedy driver re-tries all patterns after every rewrite, so technically both patterns get a shot. The benefit-2 ordering makes the trace deterministic.

```bash
git commit -m "feat(csl-auto-vectorize): FmulsScalarPattern — c[i] = α * a[i]"
```

---

## Task 16: FmacsScalarPattern

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/Patterns/ScalarBroadcastPatterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fmacs_scalar.mlir`

**Deltas from Task 15:**
- Test body mirrors `dsds.mlir` saxpy: `%m = arith.mulf %va, %alpha : f32`, `%s = arith.addf %m, %vy : f32`, `store %s, y[i]` (the accumulator is loaded from `%y`).
- Pattern: 2 loads, 1 store, 2 body ops (mulf + addf); scalar = the non-load-result operand of mulf. Store target buffer = accumulator load's buffer. Emit `csl.builtin_call "fmacs"(%dy, %dy, %dx, %alpha) : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()`.
- Benefit=3 (beats both FmacsPattern at benefit=2 and FmulsScalarPattern at benefit=2).

Register in `populateScalarBroadcastPatterns`. Build, test, commit.

---

## Task 17: Stencil test (negative offset, existing FaddsPattern)

**Files:**
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/stencil_fadds.mlir`

This task is test-only — it validates the analyzer's negative-offset + subview logic through an already-implemented pattern. No new source code.

- [ ] **Step 17.1: Write the test**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// 2-point stencil: c[i] = a[i-1] + a[i]  for i ∈ [1, N-1)
// The loop is clipped to keep a[i-1] in bounds.  The pass must emit two
// subviews of `a` (offsets 0 and 1) and one of `c` (offset 1).

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: memref.subview %a[0]
// CHECK: memref.subview %a[1]
// CHECK: memref.subview %c[1]
// CHECK: csl.builtin_call "fadds"

module {
  csl.wafer @stencil2 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      %c = csl.var @c : memref<128xf32>
      csl.func @compute {
        %c1 = arith.constant 1 : index
        %Nm1 = arith.constant 127 : index
        scf.for %i = %c1 to %Nm1 step %c1 {
          %im1 = arith.subi %i, %c1 : index
          %vl = memref.load %a[%im1] : memref<128xf32>
          %vc = memref.load %a[%i]   : memref<128xf32>
          %s  = arith.addf %vl, %vc  : f32
          memref.store %s, %c[%i]    : memref<128xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
```

- [ ] **Step 17.2: Run**

```bash
lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/stencil_fadds.mlir -v
```
Expected: PASS. (If the analyzer mishandles the `arith.subi %i, %c1` → coefficient=1, offset=-1, this will fail. Fix in `matchAffineIndexInIV` if needed — the Task 6 implementation already handles this.)

- [ ] **Step 17.3: Commit**

```bash
git add mlir/test/Dialect/CSL/Transforms/auto-vectorize/stencil_fadds.mlir
git commit -m "test(csl-auto-vectorize): stencil pattern with negative offset + subview"
```

---

## Task 18: Rank-2 FaddsPattern

**Files:**
- Create: `mlir/lib/Dialect/CSL/Transforms/Patterns/Rank2Patterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/rank2_fadds.mlir`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp`
- Modify: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`

- [ ] **Step 18.1: Write the failing test**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// 2D matrix add: C[i,j] = A[i,j] + B[i,j]  → @fadds on mem4d_dsd (rank-2).

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: %[[SA:.+]] = memref.subview %A[0, 0] [8, 16] [1, 1]
// CHECK: %[[SB:.+]] = memref.subview %B[0, 0] [8, 16] [1, 1]
// CHECK: %[[SC:.+]] = memref.subview %C[0, 0] [8, 16] [1, 1]
// CHECK: %[[DA:.+]] = csl.get_mem_dsd %[[SA]]
// CHECK: %[[DB:.+]] = csl.get_mem_dsd %[[SB]]
// CHECK: %[[DC:.+]] = csl.get_mem_dsd %[[SC]]
// CHECK: csl.builtin_call "fadds"(%[[DC]], %[[DA]], %[[DB]])

module {
  csl.wafer @mat2 {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<8x16xf32>
      %B = csl.var @B : memref<8x16xf32>
      %C = csl.var @C : memref<8x16xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %M  = arith.constant 8 : index
        %N  = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %M step %c1 {
          scf.for %j = %c0 to %N step %c1 {
            %va = memref.load %A[%i, %j] : memref<8x16xf32>
            %vb = memref.load %B[%i, %j] : memref<8x16xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %C[%i, %j] : memref<8x16xf32>
          }
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
```

- [ ] **Step 18.2: Create `Rank2Patterns.cpp`**

```cpp
//===- Rank2Patterns.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/Patterns/PatternsCommon.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csl-auto-vectorize"

using namespace mlir;
using namespace xilinx::air;

namespace {

/// Rank-2 body signature identical to Fadds but `info.isRank2 == true`
/// and memref rank is 2.
struct Rank2FaddsPattern : public OpRewritePattern<scf::ForOp> {
  Rank2FaddsPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto maybe = analyzeForLoop(op);
    if (failed(maybe)) return failure();
    const LoopIdiom &info = *maybe;
    if (!info.isRank2) return failure();
    if (info.loads.size() != 2 || info.stores.size() != 1 ||
        info.bodyOps.size() != 1) return failure();
    auto add = dyn_cast<arith::AddFOp>(info.bodyOps[0]);
    if (!add || !add.getType().isF32()) return failure();
    Value l0 = info.loads[0].getResult();
    Value l1 = info.loads[1].getResult();
    if (!((add.getLhs() == l0 && add.getRhs() == l1) ||
          (add.getLhs() == l1 && add.getRhs() == l0))) return failure();
    if (info.stores[0].getValue() != add.getResult()) return failure();

    LLVM_DEBUG(llvm::dbgs() << "match: Rank2FaddsPattern @" << op.getLoc()
                            << "\n");

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    Value mrA = buildSubviewForAccess(rewriter, loc, info, info.accesses[0]);
    Value mrB = buildSubviewForAccess(rewriter, loc, info, info.accesses[1]);
    Value mrC = buildSubviewForAccess(rewriter, loc, info, info.accesses[2]);
    Value dA = buildGetMemDsd(rewriter, loc, mrA);
    Value dB = buildGetMemDsd(rewriter, loc, mrB);
    Value dC = buildGetMemDsd(rewriter, loc, mrC);
    buildBuiltinCall(rewriter, loc, "fadds", {dC, dA, dB});
    rewriter.eraseOp(op);
    return success();
  }
};

} // anonymous namespace

namespace xilinx { namespace air {
void populateRank2Patterns(RewritePatternSet &patterns) {
  patterns.add<Rank2FaddsPattern>(patterns.getContext());
}
}}
```

- [ ] **Step 18.3: Wire it into the driver + CMakeLists. Build. Run. Commit.**

```bash
git commit -m "feat(csl-auto-vectorize): Rank2FaddsPattern — mem4d_dsd fadds"
```

---

## Task 19: Rank-2 FmacsPattern

**Files:**
- Modify: `mlir/lib/Dialect/CSL/Transforms/Patterns/Rank2Patterns.cpp`
- Create: `mlir/test/Dialect/CSL/Transforms/auto-vectorize/rank2_fmacs.mlir`

**Deltas from Task 18:**
- Test body: 3 loads (A[i,j], B[i,j], C[i,j]), mulf + addf, store to C.
- Pattern: mirror `FmacsPattern` but gated on `info.isRank2`, benefit=3.

Build, test, commit as `feat(csl-auto-vectorize): Rank2FmacsPattern`.

---

## Task 20: Fall-through tests (non-destruction guarantee)

**Goal:** Eight negative-test files that assert the pass preserves scalar `scf.for` when the legality predicate fails. One commit, one FileCheck RUN per file.

**Files (all new):**
- `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/multi_store.mlir`
- `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/inner_if.mlir`
- `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/non_constant_bound.mlir`
- `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/non_affine_index.mlir`
- `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/oob_access.mlir`
- `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/iter_args_reduction.mlir`
- `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/unsupported_body_op.mlir`
- `mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/mixed_preserved.mlir`

Every file uses this skeleton — RUN line `air-opt %s -csl-auto-vectorize | FileCheck %s` and the two assertions `CHECK: scf.for` + `CHECK-NOT: csl.get_mem_dsd`:

- [ ] **Step 20.1: `multi_store.mlir` — body has 2 stores (not pure)**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @multi_store {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %b = csl.var @b : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<64xf32>
          memref.store %va, %c[%i] : memref<64xf32>     // first store
          memref.store %va, %b[%i] : memref<64xf32>     // second store — rejects
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
```

- [ ] **Step 20.2: `inner_if.mlir` — body contains scf.if**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @inner_if {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<64xf32>
          %p  = arith.cmpf ogt, %va, %zero : f32
          scf.if %p {
            memref.store %va, %c[%i] : memref<64xf32>
          }
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
```

- [ ] **Step 20.3: `non_constant_bound.mlir` — upper bound is a block arg**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @nonconst {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute(%dyn_ub: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %dyn_ub step %c1 {
          %v = memref.load %a[%i] : memref<64xf32>
          memref.store %v, %c[%i] : memref<64xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
```

- [ ] **Step 20.4: `non_affine_index.mlir` — index is an unrelated memref load**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @nonaffine {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %idx = csl.var @idx : memref<64xindex>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %gather = memref.load %idx[%i] : memref<64xindex>
          %v = memref.load %a[%gather] : memref<64xf32>     // non-IV index
          memref.store %v, %c[%i] : memref<64xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
```

- [ ] **Step 20.5: `oob_access.mlir` — a[i-1] with i ∈ [0, N)**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @oob {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %im1 = arith.subi %i, %c1 : index
          %v = memref.load %a[%im1] : memref<64xf32>  // OOB at i=0
          memref.store %v, %c[%i] : memref<64xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
```

- [ ] **Step 20.6: `iter_args_reduction.mlir` — scf.for with iter_args**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @reduce {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      csl.func @compute() -> f32 {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        %r = scf.for %i = %c0 to %n step %c1 iter_args(%s = %zero) -> f32 {
          %v = memref.load %a[%i] : memref<64xf32>
          %s2 = arith.addf %s, %v : f32
          scf.yield %s2 : f32
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
```

- [ ] **Step 20.7: `unsupported_body_op.mlir` — body contains math.sqrt (not arith.*)**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd
module {
  csl.wafer @sqrt {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %v = memref.load %a[%i] : memref<64xf32>
          %r = math.sqrt %v : f32
          memref.store %r, %c[%i] : memref<64xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
```

- [ ] **Step 20.8: `mixed_preserved.mlir` — hand-written DSD + unmatched scalar loop, both must survive**

```mlir
// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// Input mixes: (a) a hand-written csl.get_mem_dsd + csl.builtin_call "fmacs"
// that must be preserved verbatim; (b) an unmatched scalar scf.for (extent
// too large: 100_000 > kMaxDsdExtent) that must also survive.  Both pieces
// share the csl.func — pattern must not tangle them.

// CHECK-LABEL: csl.func @compute
// CHECK: csl.get_mem_dsd
// CHECK: csl.builtin_call "fmacs"
// CHECK: scf.for
// CHECK-NOT: csl.get_mem_dsd %a :

module {
  csl.wafer @mixed {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      %a = csl.var @a : memref<100000xf32>
      %c = csl.var @c : memref<100000xf32>
      csl.func @compute {
        %scal = arith.constant 2.0 : f32
        %Ad = csl.get_mem_dsd %A : memref<128xf32> -> !csl.dsd
        %yd = csl.get_mem_dsd %y : memref<128xf32> -> !csl.dsd
        csl.builtin_call "fmacs"(%yd, %yd, %Ad, %scal)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()

        %c0 = arith.constant 0 : index
        %n  = arith.constant 100000 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %v = memref.load %a[%i] : memref<100000xf32>
          memref.store %v, %c[%i] : memref<100000xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {}
  }
}
```

- [ ] **Step 20.9: Run all eight fall-through tests**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && \
  lit ../mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/ -v
```
Expected: all 8 PASS. If any fails, inspect the analyzer's debug output:

```bash
air-opt path/to/failing.mlir -csl-auto-vectorize -debug-only=csl-auto-vectorize 2>&1 | head
```

The expected reject reason for each file:
- `multi_store.mlir` → `reject: body has 2 stores (need 1)`
- `inner_if.mlir` → `reject: nested control flow (scf.if)`
- `non_constant_bound.mlir` → `reject: non-constant bounds or step`
- `non_affine_index.mlir` → `reject: non-affine index` or `reject: IV has non-index user: memref.load`
- `oob_access.mlir` → `reject: access OOB [-1,62]...`
- `iter_args_reduction.mlir` → `reject: loop has iter_args`
- `unsupported_body_op.mlir` → `reject: disallowed dialect in body (math.sqrt)`
- `mixed_preserved.mlir` → `reject: extent 100000 outside [1,65535]`

- [ ] **Step 20.10: Commit**

```bash
git add mlir/test/Dialect/CSL/Transforms/auto-vectorize/fallthrough/
git commit -m "test(csl-auto-vectorize): fall-through tests — pass must not touch non-matching loops"
```

---

## Task 21: End-to-end tests through the emitter

**Files:**
- Create: `mlir/test/Targets/CSLEmit/e2e/auto-vectorize/vecadd.mlir`
- Create: `mlir/test/Targets/CSLEmit/e2e/auto-vectorize/saxpy_fma.mlir`
- Create: `mlir/test/Targets/CSLEmit/e2e/auto-vectorize/rank2.mlir`

Each test runs the full pipeline `air-opt %s -csl-auto-vectorize -csl-infer-exports | air-translate --emit-csl --output-dir=%t` and FileCheck-s the generated `.csl` text for `@get_dsd(` + `@fadds(` / `@fmacs(` lines. This proves the downstream emitter handles the output IR.

- [ ] **Step 21.1: `vecadd.mlir`**

```mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/vecadd_e2e/pe.csl
//
// CHECK-LABEL: fn compute() void
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &a, .extent = 1024 });
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &b, .extent = 1024 });
// CHECK: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &c, .extent = 1024 });
// CHECK: @fadds(

module {
  csl.wafer @vecadd_e2e {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<1024xf32>
      %b = csl.var @b : memref<1024xf32>
      %c = csl.var @c : memref<1024xf32>
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
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<1024xf32>, %b_in: memref<1024xf32>,
                   %c_out: memref<1024xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
    }
  }
}
```

- [ ] **Step 21.2: `saxpy_fma.mlir`** — mirror `dsds.mlir` but with a scalar scf.for input (post-pass output should match the hand-written DSD form):

```mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/saxpy_auto/pe.csl
//
// CHECK-LABEL: fn compute() void
// CHECK: @fmacs(

module {
  csl.wafer @saxpy_auto {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %alpha = arith.constant 2.0 : f32
        %c0 = arith.constant 0 : index
        %n  = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %A[%i] : memref<128xf32>
          %vy = memref.load %y[%i] : memref<128xf32>
          %m  = arith.mulf %va, %alpha : f32
          %s  = arith.addf %m, %vy : f32
          memref.store %s, %y[%i] : memref<128xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%A_in: memref<128xf32>, %y_io: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %A_in to @layout::@A
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %y_io to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_io
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }
}
```

- [ ] **Step 21.3: `rank2.mlir`** — rank-2 vecadd through emitter:

```mlir
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/mat2_auto/pe.csl
//
// CHECK: @get_dsd(mem4d_dsd,
// CHECK: @fadds(

module {
  csl.wafer @mat2_auto {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<8x16xf32>
      %B = csl.var @B : memref<8x16xf32>
      %C = csl.var @C : memref<8x16xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %M  = arith.constant 8 : index
        %N  = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %M step %c1 {
          scf.for %j = %c0 to %N step %c1 {
            %va = memref.load %A[%i, %j] : memref<8x16xf32>
            %vb = memref.load %B[%i, %j] : memref<8x16xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %C[%i, %j] : memref<8x16xf32>
          }
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%Ai: memref<8x16xf32>, %Bi: memref<8x16xf32>,
                   %Co: memref<8x16xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %Ai to @layout::@A
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8x16xf32>
      csl_host.memcpy_h2d %Bi to @layout::@B
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8x16xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@C to %Co
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8x16xf32>
    }
  }
}
```

- [ ] **Step 21.4: Run all three**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && \
  lit ../mlir/test/Targets/CSLEmit/e2e/auto-vectorize/ -v
```
Expected: 3 PASS.

- [ ] **Step 21.5: Commit**

```bash
git add mlir/test/Targets/CSLEmit/e2e/auto-vectorize/
git commit -m "test(csl-auto-vectorize): e2e tests through air-translate --emit-csl"
```

---

## Task 22: Pipeline integration + new lit target

**Files:**
- Modify: `python/air/compiler/aircc/main.py` (add `-csl-auto-vectorize` to CSL pipeline)
- Modify: `mlir/test/CMakeLists.txt` or equivalent (add `check-airmlir-dialect-csl-transforms` target)

- [ ] **Step 22.1: Find the CSL pipeline in aircc.py**

```bash
grep -n "csl-infer-exports\|air-to-csl" /home/bricklib_dataflow/air-csl/mlir-air/python/air/compiler/aircc/main.py
```
Expected: one or two lines showing the existing CSL-path invocation of `air-opt`.

- [ ] **Step 22.2: Add the pass to the pipeline**

Edit the relevant `air-opt` command list in `python/air/compiler/aircc/main.py`. Example delta (exact location depends on existing code — insert the new arg immediately after `-air-to-csl` and before `-csl-infer-exports`):

```python
# OLD:
#     "air-opt", "-air-to-csl=...", "-csl-infer-exports", input_mlir

# NEW:
      "air-opt", "-air-to-csl=...", "-csl-auto-vectorize", "-csl-infer-exports", input_mlir
```

- [ ] **Step 22.3: Find where existing lit check targets are defined**

```bash
grep -Rn "check-airmlir-dialect-csl\|check-airmlir-conversion" \
  /home/bricklib_dataflow/air-csl/mlir-air/mlir/CMakeLists.txt \
  /home/bricklib_dataflow/air-csl/mlir-air/mlir/test/
```
Expected: one or more `add_lit_testsuite` or `add_lit_target` entries for the existing CSL dialect / conversion check targets.

- [ ] **Step 22.4: Add the new lit target alongside**

Edit the CMakeLists in question, copying the existing CSL entries and adding:

```cmake
add_lit_testsuite(check-airmlir-dialect-csl-transforms
  "Running the CSL Transforms MLIR tests"
  ${CMAKE_CURRENT_BINARY_DIR}/Dialect/CSL/Transforms
  DEPENDS air-opt air-translate FileCheck not)
set_target_properties(check-airmlir-dialect-csl-transforms
  PROPERTIES FOLDER "Tests")
```

- [ ] **Step 22.5: Build and run the new target**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && \
  ninja install check-airmlir-dialect-csl-transforms
```
Expected: all tests from Tasks 9-20 pass under the new target.

- [ ] **Step 22.6: Run full test suite for final regression check**

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air/build && ninja check-air-mlir
```
Expected: no regressions. All previously-green tests still green, plus the new ones.

- [ ] **Step 22.7: Commit**

```bash
git add python/air/compiler/aircc/main.py \
        mlir/test/CMakeLists.txt
git commit -m "chore(csl-auto-vectorize): wire pass into aircc.py + add check-airmlir-dialect-csl-transforms"
```

---

## Self-Review Checklist

Run these mentally against the plan above before handing off:

1. **Spec coverage.** Every one of spec §§1 (scope), 3 (architecture), 4 (analyzer), 5 (idiom table), 6 (rewrite shapes), 7 (tests), 8 (debugging), 9.1 (commit sequence) is implemented by a task. §9.2 (follow-ups) is explicitly out of scope per spec.
2. **Placeholder scan.** No `TBD`, `TODO`, `FIXME`, or vague "similar to X" phrasings without code. Task 11 delta-from-10 and Task 14 delta-from-13 are spelled out with exact changes.
3. **Type consistency.** `LoopIdiom`, `DsdAccessPattern`, `analyzeForLoop`, `populate*Patterns`, and every pattern class name appears identically across tasks. `kMaxDsdExtent` used once, consistently.
4. **Naming of populators.** `populateElementwisePatterns`, `populateFmaPattern`, `populateMovePatterns`, `populateScalarBroadcastPatterns`, `populateRank2Patterns` are all declared in-file and forward-declared in `CSLAutoVectorize.cpp`. Every call site matches.
5. **Benefit ordering.** Plain-binary patterns = 1. `FmacsPattern` = 2, `Rank2FaddsPattern` = 2, `FmulsScalarPattern` = 2, `Rank2FmacsPattern` = 3, `FmacsScalarPattern` = 3. No benefit=1 pattern can hide a benefit≥2 pattern on the same IR.
6. **Test dir conventions.** Positive tests in `mlir/test/Dialect/CSL/Transforms/auto-vectorize/`, fall-through in `.../fallthrough/`, e2e in `mlir/test/Targets/CSLEmit/e2e/auto-vectorize/`. Matches spec §7.
