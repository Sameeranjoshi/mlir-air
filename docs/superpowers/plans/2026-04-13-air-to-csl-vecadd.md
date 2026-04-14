# AIR → CSL Vecadd Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile a 1×1 256-element f32 vecadd from AIR all the way through `csl.*` → `csl_rt.*` → CSL source + `run.py` and execute it on the CS-3 attached to this machine, returning the correct array to the host.

**Architecture:** New `air-to-csl-dialect` MLIR pass produces `csl.*` IR; the existing `csl-to-csl-rt` pass is extended to lower the new ops plus synthesize the host-side runtime sequence; `CSLRuntimeToPy.cpp` is reworked into a real KernelEmitter (walks `csl.kernel` body and writes valid CSL source) plus a real HostEmitter (emits a runnable Python script using `SdkLayout` / `SdkRuntime`). Two new CSL ops (`csl.export_name` and `csl.export_symbol`) are added because the spec referenced them but they were missing. Hand-written golden files anchor every layer of testing.

**Tech Stack:** MLIR (LLVM 18+), TableGen, C++17, Cerebras SDK 1.4, CSL language, Python 3.10 (cs_python), pytest, lit, FileCheck.

**Reference spec:** `docs/superpowers/specs/2026-04-13-air-to-csl-vecadd-design.md`. Read it first.

**SDK location on this machine:** `/home/bricklib_dataflow/sdk/SDK_1_4/` (`cs_python` and `cslc` are on PATH).

**CSL reference syntax:** `/home/bricklib_dataflow/temp/csl-examples/tutorials/gemv-00-basic-syntax/code.csl` — read this before writing the golden CSL file.

---

## File Structure

### New files
| Path | Responsibility |
|---|---|
| `mlir/test/Conversion/AIRToCSL/golden/vecadd_pe.csl.golden` | Hand-written reference CSL kernel that the compiler will eventually match (Phase 0). |
| `mlir/test/Conversion/AIRToCSL/golden/run.py.golden` | Hand-written reference host script that the compiler will eventually match (Phase 0). |
| `mlir/test/Conversion/AIRToCSL/golden/build_and_run.sh` | Two-line shell driver to invoke the golden run.py for bootstrap validation (Phase 0). |
| `mlir/include/air/Conversion/AIRToCSLDialectPass.h` | Pass declaration header for the new AIR→csl.* pass. |
| `mlir/lib/Conversion/AIRToCSLDialect/CMakeLists.txt` | CMake glue for the new pass. |
| `mlir/lib/Conversion/AIRToCSLDialect/AIRToCSLDialect.cpp` | The new AIR→csl.* pass implementation. |
| `mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir` | Lit test: AIR vecadd input, FileCheck of produced csl.* IR. |
| `mlir/test/Conversion/AIRToCSLDialect/reject_2x2_herd.mlir` | Lit test: rejects 2×2 herd with a precise error. |
| `mlir/test/Conversion/AIRToCSLDialect/reject_channel_op.mlir` | Lit test: rejects `air.channel.put`. |
| `mlir/test/Conversion/AIRToCSLDialect/reject_dynamic_memref.mlir` | Lit test: rejects dynamic memref shape. |
| `mlir/test/Conversion/AIRToCSLDialect/reject_unsupported_eltype.mlir` | Lit test: rejects f64 element type. |
| `mlir/test/Conversion/AIRToCSLDialect/reject_async_token.mlir` | Lit test: rejects `air.execute` async token in herd body. |
| `mlir/test/Conversion/AIRToCSLDialect/reject_dma_in_body.mlir` | Lit test: rejects `air.dma_memcpy_nd` in herd body. |
| `mlir/test/Conversion/AIRToCSLDialect/reject_multiple_herds.mlir` | Lit test: rejects two herds in one launch. |
| `mlir/test/Conversion/CSLToCSLRuntime/vecadd_with_kernel.mlir` | Lit test: hand-written csl.* IR, FileCheck of produced csl_rt.* IR. |
| `mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir` | Lit test: hand-written `csl.kernel`, FileCheck of produced `vecadd_pe.csl`. |
| `mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir` | Lit test: hand-written `csl_rt.*` sequence, FileCheck of produced `run.py`. |
| `test/csl/test_vecadd_e2e.py` | The hardware integration test. The "definition of done." |

### Modified files
| Path | Change |
|---|---|
| `mlir/include/air/Dialect/CSL/CSLOps.td` | Add `CSL_ExportNameOp` and `CSL_ExportSymbolOp` op definitions. |
| `mlir/include/air/Conversion/Passes.td` | Add `AIRToCSLDialect` pass def; remove `AIRToCSL` (Phase-1) pass def at the very end of the milestone (Phase 7). |
| `mlir/include/air/Conversion/CMakeLists.txt` | Already auto-generates from Passes.td; no change needed. |
| `mlir/lib/Conversion/CMakeLists.txt` | Add `add_subdirectory(AIRToCSLDialect)`. |
| `mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp` | Add patterns for `csl.export_name` → `csl_rt.export_name` plus host-side runtime sequence synthesis. |
| `mlir/lib/Targets/CSLRuntimeToPy.cpp` | Replace the `emitKernelPrograms()` stub with a real `KernelEmitter`; rework `emitRunPy()` to produce a runnable script with argparse, numpy buffers, validator. |
| `mlir/lib/Conversion/CMakeLists.txt` (and tools/air-opt build) | Wire the new pass into `air-opt`. |
| `tools/air-opt/CMakeLists.txt` (if it lists pass libraries explicitly) | Add the new pass library. |

### Files deleted/moved (Phase 7 only)
| Path | Action |
|---|---|
| `mlir/lib/Conversion/AIRToCSLPass.cpp` | Move to `archived_code/AIRToCSLPass.cpp` after the hardware test passes. |
| `mlir/lib/Conversion/CMakeLists.txt` | Remove the `AIRToCSLPass.cpp` source entry. |

---

## Phase 0 — Bootstrap golden files (no compiler code)

**Why first:** Until we have a known-good CSL kernel + run.py that we've personally seen execute on CS-3, every later debug session will be ambiguous about whether the bug is in the compiler or in our understanding of CSL/SDK. The golden files remove that ambiguity once and for all.

### Task 0.1: Hand-write the golden CSL kernel

**Files:**
- Create: `mlir/test/Conversion/AIRToCSL/golden/vecadd_pe.csl.golden`

- [ ] **Step 1: Read the reference syntax**

Run: `cat /home/bricklib_dataflow/temp/csl-examples/tutorials/gemv-00-basic-syntax/code.csl`

Expected: working CSL with `var`, `fn`, `for (@range(...)) |i| { ... }` syntax.

- [ ] **Step 2: Write the golden file**

Create `mlir/test/Conversion/AIRToCSL/golden/vecadd_pe.csl.golden` with this exact content:

```csl
// Golden reference CSL kernel for the AIR → CSL vecadd milestone.
// This file is hand-written and verified to compile and run on CS-3.
// The KernelEmitter in CSLRuntimeToPy.cpp must produce output equivalent to this.

const N: i32 = 256;

var a_buf: [N]f32;
var b_buf: [N]f32;
var c_buf: [N]f32;

fn compute() void {
  for (@range(i32, N)) |i| {
    c_buf[i] = a_buf[i] + b_buf[i];
  }
}

comptime {
  @export_symbol(a_buf, "a");
  @export_symbol(b_buf, "b");
  @export_symbol(c_buf, "c");
  @export_symbol(compute);
}
```

- [ ] **Step 3: Verify it compiles standalone**

Run from a temp dir:
```bash
cd /tmp && mkdir -p vecadd_bootstrap && cd vecadd_bootstrap && \
  cp /home/bricklib_dataflow/air-csl/mlir-air/mlir/test/Conversion/AIRToCSL/golden/vecadd_pe.csl.golden vecadd_pe.csl && \
  cslc --arch=wse3 vecadd_pe.csl --fabric-dims=8,3 --fabric-offsets=4,1 -o out
```

Expected: `cslc` exits 0, produces an `out/` directory. If it fails: read the error, fix the golden file, repeat. Do not proceed until this succeeds.

(Note: `cslc` standalone needs a layout file in real use, but for syntax validation we are checking `cslc` accepts the kernel source. If `cslc` requires a layout-only invocation, prefix with `--layout=...` or use the SDK's recommended single-file build mode. If neither works, defer full standalone validation to Task 0.3.)

### Task 0.2: Hand-write the golden run.py

**Files:**
- Create: `mlir/test/Conversion/AIRToCSL/golden/run.py.golden`

- [ ] **Step 1: Write the golden run.py**

Create `mlir/test/Conversion/AIRToCSL/golden/run.py.golden` with this exact content:

```python
#!/usr/bin/env cs_python
"""Golden reference run.py for the AIR → CSL vecadd milestone.

Hand-written and verified to run on CS-3. The HostEmitter in
CSLRuntimeToPy.cpp must produce output equivalent to this.
"""

import argparse
import sys
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    SdkLayout,
    SdkTarget,
    SimfabConfig,
    get_platform,
)


def build_layout(platform):
    layout = SdkLayout(platform)
    region = layout.create_code_region("vecadd_pe.csl", "vecadd", 1, 1)
    region.place(0, 0)
    layout.export_name("a", "<f32>[256]")
    layout.export_name("b", "<f32>[256]")
    layout.export_name("c", "<f32>[256]")
    layout.export_name("compute", "fn()void")
    return layout.compile(out_prefix="out")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmaddr", default=None)
    parser.add_argument("--arch", default="wse3", choices=("wse2", "wse3"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config = SimfabConfig()
    target = SdkTarget.WSE3 if args.arch == "wse3" else SdkTarget.WSE2
    platform = get_platform(args.cmaddr, config, target)

    artifacts = build_layout(platform)
    runtime = SdkRuntime(artifacts, platform, memcpy_required=True)
    runtime.load()

    a = np.arange(256, dtype=np.float32)
    b = np.arange(256, dtype=np.float32) * 2.0
    c = np.zeros(256, dtype=np.float32)
    expected = a + b

    id_a = runtime.get_id("a")
    id_b = runtime.get_id("b")
    id_c = runtime.get_id("c")

    runtime.memcpy_h2d(id_a, a, 0, 0, 1, 1, 256)
    runtime.memcpy_h2d(id_b, b, 0, 0, 1, 1, 256)
    runtime.launch("compute")
    runtime.memcpy_d2h(c, id_c, 0, 0, 1, 1, 256)
    runtime.stop()

    if args.check:
        if not np.array_equal(c, expected):
            mismatches = np.where(c != expected)[0]
            print(f"FAIL: {len(mismatches)} mismatches", file=sys.stderr)
            for i in mismatches[:8]:
                print(f"  c[{i}] = {c[i]}  expected {expected[i]}", file=sys.stderr)
            sys.exit(1)
        print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

### Task 0.3: Verify the golden pair runs on CS-3

**Files:**
- Create: `mlir/test/Conversion/AIRToCSL/golden/build_and_run.sh`

- [ ] **Step 1: Write the driver script**

```bash
cat > mlir/test/Conversion/AIRToCSL/golden/build_and_run.sh <<'EOF'
#!/usr/bin/env bash
# Bootstrap validation: copy the golden files to a temp dir and run them on CS-3.
# Used during Phase 0 to prove the SDK + reference are working before any compiler code.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
cp "$HERE/vecadd_pe.csl.golden" "$TMP/vecadd_pe.csl"
cp "$HERE/run.py.golden"        "$TMP/run.py"
cd "$TMP"
cs_python run.py --arch wse3 --check
EOF
chmod +x mlir/test/Conversion/AIRToCSL/golden/build_and_run.sh
```

- [ ] **Step 2: Run it**

Run: `bash mlir/test/Conversion/AIRToCSL/golden/build_and_run.sh`

Expected: prints `PASS`, exits 0.

If it fails, debug the golden files **before any other code is written**. Common issues:
- Wrong import path for `cerebras.sdk.runtime.sdkruntimepybind` — check what `cs_python -c "import cerebras.sdk.runtime.sdkruntimepybind"` says.
- `<f32>[256]` type-spec syntax may be wrong for this SDK version — try `f32` plain, or check SDK docs at `/home/bricklib_dataflow/sdk/SDK_1_4/` or `https://sdk.cerebras.net/api-docs/sdklayout-api`.
- `region.place(0, 0)` may need additional args.
- `memcpy_h2d` signature may be `(dest_id, src, px, py, w, h, elem_per_pe)` or have additional kwargs (`order=MemcpyOrder.ROW_MAJOR`, etc.).

**Iterate on the golden files until this script prints PASS.** This is the most important Phase 0 deliverable. Everything downstream consumes the result.

- [ ] **Step 3: Commit**

```bash
git add mlir/test/Conversion/AIRToCSL/golden/
git commit -m "test(csl): add bootstrap golden vecadd kernel and run.py

Hand-written CSL kernel and host script verified to run on CS-3.
The KernelEmitter and HostEmitter in CSLRuntimeToPy.cpp will be
written to match this output. See docs/superpowers/specs/2026-04-13-
air-to-csl-vecadd-design.md §8.2 for rationale."
```

---

## Phase 1 — Add `csl.export_name` and `csl.export_symbol` ops

**Why:** The spec assumed these ops existed. They don't. They're trivial — each is a declarative op with one or two attributes — but everything downstream depends on them.

### Task 1.1: Add CSL_ExportNameOp TableGen

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td` (location: end of file, before any closing `#endif`)

- [ ] **Step 1: Read the existing op definitions**

Run: `head -250 mlir/include/air/Dialect/CSL/CSLOps.td`

Expected: see existing op defs like `CSL_KernelOp`, `CSL_VarOp`, `CSL_FuncOp`. Note their TableGen pattern (which class they extend, how `assemblyFormat` is written).

- [ ] **Step 2: Write a parsing test that fails**

Create `mlir/test/Dialect/CSL/export_ops.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: func.func @export_name_basic
func.func @export_name_basic() {
  // CHECK: csl.export_name "a" : memref<256xf32> {direction = "in"}
  csl.export_name "a" : memref<256xf32> {direction = "in"}
  // CHECK: csl.export_name "c" : memref<256xf32> {direction = "out"}
  csl.export_name "c" : memref<256xf32> {direction = "out"}
  // CHECK: csl.export_name "compute" : () -> ()
  csl.export_name "compute" : () -> ()
  return
}

// CHECK-LABEL: func.func @export_symbol_basic
func.func @export_symbol_basic() {
  csl.kernel {
    csl.var @x : memref<8xf32>
    csl.comptime {
      // CHECK: csl.export_symbol @x alias("a")
      csl.export_symbol @x alias("a")
      // CHECK: csl.export_symbol @y
      csl.export_symbol @y
    }
  } {source_file = "k.csl"} : !csl.kernel
  return
}
```

- [ ] **Step 3: Run the test, verify it fails**

Run: `cd build && lit -v mlir/test/Dialect/CSL/export_ops.mlir`

Expected: FAIL with a parser error about `csl.export_name`.

- [ ] **Step 4: Add the TableGen op definitions**

Open `mlir/include/air/Dialect/CSL/CSLOps.td` and find the section where Runtime ops live (after the kernel ops; look for `CSL_ImportModuleOp` or similar). Add these op definitions before the closing of the file's includes/endif:

```tablegen
//===----------------------------------------------------------------------===//
// csl.export_name — Host-visible name and type for a buffer or function
//===----------------------------------------------------------------------===//

def CSL_ExportNameOp : CSL_Op<"export_name"> {
  let summary = "Declare a host-visible exported name with a type";
  let description = [{
    Declares a name that the host runtime (csl_rt) can address. The
    optional `direction` attribute distinguishes input buffers (`"in"`)
    from output buffers (`"out"`). Functions have no direction.

    Lowering: `csl-to-csl-rt` converts this op to `csl_rt.export_name`
    on the layout, and uses `direction` to decide whether to emit
    `memcpy_h2d` or `memcpy_d2h` in the host runtime sequence.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$exported_type,
    OptionalAttr<StrAttr>:$direction
  );

  let assemblyFormat = [{
    $sym_name `:` $exported_type (`{` `direction` `=` $direction^ `}`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// csl.export_symbol — Inside csl.comptime: export a kernel-local symbol
//===----------------------------------------------------------------------===//

def CSL_ExportSymbolOp : CSL_Op<"export_symbol"> {
  let summary = "Export a kernel-local symbol via @export_symbol";
  let description = [{
    Inside a `csl.comptime` block, declares that a kernel-local symbol
    (variable or function) should be made callable / readable from the
    host. The optional `alias` attribute provides an alternative name
    for the host side; when omitted, the symbol's own name is used.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$symbol,
    OptionalAttr<StrAttr>:$alias
  );

  let assemblyFormat = [{
    $symbol (`alias` `(` $alias^ `)`)? attr-dict
  }];
}
```

- [ ] **Step 5: Rebuild and re-run the test**

Run: `cd build && ninja AIRCSLOpsIncGen && ninja AIRCSLDialect && lit -v mlir/test/Dialect/CSL/export_ops.mlir`

Expected: PASS.

If you get TableGen errors about `StrAttr` / `FlatSymbolRefAttr` / `TypeAttr` not being defined, ensure the file's existing includes pull in `mlir/IR/AttrTypeBase.td` and `mlir/IR/SymbolInterfaces.td`. Most CSL ops files already include these — verify before adding new includes.

- [ ] **Step 6: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td mlir/test/Dialect/CSL/export_ops.mlir
git commit -m "feat(csl): add csl.export_name and csl.export_symbol ops

These ops were referenced by the design but never defined. Adding
them as a pre-requisite for the air-to-csl-dialect milestone.

- csl.export_name carries a name, type, and optional direction
  attribute (in/out) used by csl-to-csl-rt to drive memcpy emission.
- csl.export_symbol lives inside csl.comptime and marks a
  kernel-local symbol as host-visible, with an optional alias.

See docs/superpowers/specs/2026-04-13-air-to-csl-vecadd-design.md §5.3.1."
```

---

## Phase 2 — `air-to-csl-dialect` pass

**Goal:** A pass that turns `air.launch`/`air.segment`/`air.herd` (1×1 only) into the IR shape from spec §5.3.1. This is the largest single component in the milestone.

### Task 2.1: Skeleton pass + registration

**Files:**
- Create: `mlir/include/air/Conversion/AIRToCSLDialectPass.h`
- Modify: `mlir/include/air/Conversion/Passes.td`
- Create: `mlir/lib/Conversion/AIRToCSLDialect/CMakeLists.txt`
- Create: `mlir/lib/Conversion/AIRToCSLDialect/AIRToCSLDialect.cpp`
- Modify: `mlir/lib/Conversion/CMakeLists.txt`

- [ ] **Step 1: Write the header**

Create `mlir/include/air/Conversion/AIRToCSLDialectPass.h`:

```cpp
//===- AIRToCSLDialectPass.h ------------------------------------*- C++ -*-===//
//
// Lowers AIR dialect ops (1×1 herds only, milestone scope) to the CSL
// dialect (csl.spatial_placement, csl.kernel, csl.code_region, csl.place,
// csl.export_name).
//
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_AIRTOCSLDIALECT_PASS_H
#define AIR_CONVERSION_AIRTOCSLDIALECT_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx::air {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIRToCSLDialectPass();

} // namespace xilinx::air

#endif // AIR_CONVERSION_AIRTOCSLDIALECT_PASS_H
```

- [ ] **Step 2: Add the pass def to Passes.td**

Open `mlir/include/air/Conversion/Passes.td`. Find the existing `def AIRToCSL` block (around line 479). Add this **above** it (we'll remove `AIRToCSL` in Phase 7):

```tablegen
def AIRToCSLDialect : Pass<"air-to-csl-dialect", "ModuleOp"> {
  let summary = "Lower AIR dialect (1x1 herds) to CSL dialect ops";
  let constructor = "xilinx::air::createAIRToCSLDialectPass()";
  let description = [{
    This pass converts AIR launch/segment/herd nests into csl.* MLIR
    ops (csl.spatial_placement, csl.code_region, csl.kernel, csl.place,
    csl.export_name, csl.export_symbol). Only 1×1 herds are supported in
    this milestone. The herd compute body is moved verbatim into a
    csl.func @compute() inside csl.kernel.

    See docs/superpowers/specs/2026-04-13-air-to-csl-vecadd-design.md.
  }];
}
```

- [ ] **Step 3: Write the empty .cpp**

Create `mlir/lib/Conversion/AIRToCSLDialect/AIRToCSLDialect.cpp`:

```cpp
//===- AIRToCSLDialect.cpp - AIR → csl.* lowering pass ----------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRToCSLDialectPass.h"
#include "air/Conversion/PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "air-to-csl-dialect"

using namespace mlir;

namespace xilinx::air {

#define GEN_PASS_DEF_AIRTOCSLDIALECT
#include "air/Conversion/Passes.h.inc"

namespace {

class AIRToCSLDialectPass
    : public impl::AIRToCSLDialectBase<AIRToCSLDialectPass> {
public:
  void runOnOperation() override;
};

void AIRToCSLDialectPass::runOnOperation() {
  // Phase 2 will add the real lowering here. For now, the pass is a no-op
  // so it can be registered and called from air-opt.
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAIRToCSLDialectPass() {
  return std::make_unique<AIRToCSLDialectPass>();
}

} // namespace xilinx::air
```

- [ ] **Step 4: Write the CMakeLists**

Create `mlir/lib/Conversion/AIRToCSLDialect/CMakeLists.txt`:

```cmake
add_mlir_library(AIRToCSLDialectPass
  AIRToCSLDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_AIR_INCLUDE_DIRS}/air/Conversion

  DEPENDS
  AIRConversionPassIncGen
  AIRDialect
  AIRCSLDialect
  AIRCSLOpsIncGen

  LINK_LIBS PUBLIC
  AIRDialect
  AIRCSLDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRSupport
)
```

(Adjust `AIRCSLOpsIncGen` if the existing CSL dialect uses a different gen target name. Check `mlir/lib/Dialect/CSL/IR/CMakeLists.txt`.)

- [ ] **Step 5: Wire into parent CMakeLists**

Open `mlir/lib/Conversion/CMakeLists.txt`, find the existing `add_subdirectory(CSLToCSLRuntime)` line, add directly below it:

```cmake
add_subdirectory(AIRToCSLDialect)
```

- [ ] **Step 6: Build and verify the pass is registered**

Run:
```bash
cd build && ninja install
air-opt --help 2>&1 | grep air-to-csl-dialect
```

Expected: prints `--air-to-csl-dialect`. If not, the pass library isn't being linked into `air-opt`. Check `tools/air-opt/CMakeLists.txt` — most pipelines auto-link via `MLIR_AIR_PASSES` or similar variable; otherwise add `AIRToCSLDialectPass` to the `target_link_libraries(air-opt ...)` list.

- [ ] **Step 7: Commit**

```bash
git add mlir/include/air/Conversion/AIRToCSLDialectPass.h \
        mlir/include/air/Conversion/Passes.td \
        mlir/lib/Conversion/AIRToCSLDialect/ \
        mlir/lib/Conversion/CMakeLists.txt
git commit -m "feat(air-to-csl-dialect): scaffold empty pass

Pass is registered with air-opt as -air-to-csl-dialect but does
nothing yet. Phase 2 will add the actual lowering."
```

### Task 2.2: Failing FileCheck test

**Files:**
- Create: `mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir`

- [ ] **Step 1: Write the test file**

```mlir
// RUN: air-opt %s -air-to-csl-dialect | FileCheck %s

// 1×1 vecadd AIR program — the canonical milestone input.

// CHECK-LABEL: func.func @vecadd
// CHECK-SAME:    %[[A:.*]]: memref<256xf32>
// CHECK-SAME:    %[[B:.*]]: memref<256xf32>
// CHECK-SAME:    %[[C:.*]]: memref<256xf32>

// CHECK:   csl.spatial_placement {
// CHECK:     %[[K:.*]] = csl.kernel {
// CHECK-DAG:    csl.var @a_buf : memref<256xf32>
// CHECK-DAG:    csl.var @b_buf : memref<256xf32>
// CHECK-DAG:    csl.var @c_buf : memref<256xf32>
// CHECK:        csl.func @compute() : () -> () {
// CHECK:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:            %{{.*}} = memref.load %a_buf[%{{.*}}] : memref<256xf32>
// CHECK:            %{{.*}} = memref.load %b_buf[%{{.*}}] : memref<256xf32>
// CHECK:            %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK:            memref.store %{{.*}}, %c_buf[%{{.*}}] : memref<256xf32>
// CHECK:          }
// CHECK:          csl.return
// CHECK:        }
// CHECK:        csl.comptime {
// CHECK-DAG:      csl.export_symbol @a_buf alias("a")
// CHECK-DAG:      csl.export_symbol @b_buf alias("b")
// CHECK-DAG:      csl.export_symbol @c_buf alias("c")
// CHECK-DAG:      csl.export_symbol @compute
// CHECK:        }
// CHECK:     } {source_file = "vecadd_pe.csl"} : !csl.kernel
// CHECK:     %[[R:.*]] = csl.code_region routes() colors() {
// CHECK:     } {height = 1 : i64, width = 1 : i64} : !csl.code_region
// CHECK:     csl.place %[[R]] %[[K]] {x = 0 : i64, y = 0 : i64}
// CHECK:   }

// CHECK-DAG: csl.export_name "a" : memref<256xf32> {direction = "in"}
// CHECK-DAG: csl.export_name "b" : memref<256xf32> {direction = "in"}
// CHECK-DAG: csl.export_name "c" : memref<256xf32> {direction = "out"}
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

- [ ] **Step 2: Run it, verify it fails**

Run: `cd build && lit -v ../mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir`

Expected: FAIL because the pass currently does nothing — the AIR ops survive into the output and FileCheck doesn't find any csl.* ops. Note: depending on lit's CHECK-DAG strictness this may report "expected string not found in input."

- [ ] **Step 3: Commit the test**

```bash
git add mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir
git commit -m "test(air-to-csl-dialect): add failing vecadd FileCheck test

Pinned IR contract from the spec. Currently red because the pass
is still a no-op."
```

### Task 2.3: Implement the lowering — input validation

**Files:**
- Modify: `mlir/lib/Conversion/AIRToCSLDialect/AIRToCSLDialect.cpp`

- [ ] **Step 1: Add the input-validation walk**

Replace the body of `AIRToCSLDialectPass::runOnOperation()` with this implementation. Read each section before pasting; you'll add patterns later in this task.

```cpp
void AIRToCSLDialectPass::runOnOperation() {
  ModuleOp module = getOperation();

  // 1. Validate. Reject any AIR shape outside the milestone scope, with
  //    a precise error citing the offending op. Fail fast: do not attempt
  //    partial lowering.
  WalkResult validation = module.walk([](Operation *op) -> WalkResult {
    if (auto herd = dyn_cast<air::HerdOp>(op)) {
      // Only 1×1 herds are supported.
      auto sizes = herd.getSizeOperands();
      if (sizes.size() != 2)
        return herd.emitOpError(
            "air-to-csl-dialect: expected 2-D herd size (got ")
               << sizes.size() << ")";
      auto isOne = [](Value v) {
        if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
          return cst.value() == 1;
        return false;
      };
      if (!isOne(sizes[0]) || !isOne(sizes[1]))
        return herd.emitOpError(
            "air-to-csl-dialect: only 1x1 herds supported in this milestone");

      // Reject async tokens in herd body.
      WalkResult bodyWalk = herd.getBody()->walk([](Operation *inner) -> WalkResult {
        if (isa<air::ExecuteOp>(inner))
          return inner->emitOpError(
              "air-to-csl-dialect: async operations not supported in herd "
              "bodies for CSL backend");
        if (isa<air::DmaMemcpyNdOp>(inner))
          return inner->emitOpError(
              "air-to-csl-dialect: dma_memcpy_nd not yet supported in herd "
              "bodies; use scalar load/store");
        return WalkResult::advance();
      });
      if (bodyWalk.wasInterrupted())
        return WalkResult::interrupt();
    }
    if (auto chPut = dyn_cast<air::ChannelPutOp>(op))
      return chPut.emitOpError(
          "air-to-csl-dialect: inter-PE channels not yet supported");
    if (auto chGet = dyn_cast<air::ChannelGetOp>(op))
      return chGet.emitOpError(
          "air-to-csl-dialect: inter-PE channels not yet supported");
    return WalkResult::advance();
  });
  if (validation.wasInterrupted())
    return signalPassFailure();

  // 2. Validate every herd memref arg: ranked, static, supported eltype.
  WalkResult typeCheck = module.walk([](air::HerdOp herd) -> WalkResult {
    for (Value arg : herd.getKernelOperands()) {
      auto memTy = dyn_cast<MemRefType>(arg.getType());
      if (!memTy || !memTy.hasStaticShape())
        return herd.emitOpError(
            "air-to-csl-dialect: kernel memrefs must be statically shaped");
      Type elt = memTy.getElementType();
      bool ok = elt.isF32() || elt.isF16() || elt.isInteger(32) || elt.isInteger(16);
      if (!ok)
        return herd.emitOpError(
            "air-to-csl-dialect: unsupported element type ") << elt;
    }
    return WalkResult::advance();
  });
  if (typeCheck.wasInterrupted())
    return signalPassFailure();

  // 3. Reject multiple herds in one launch.
  WalkResult herdCount = module.walk([](air::LaunchOp launch) -> WalkResult {
    int n = 0;
    launch.walk([&](air::HerdOp) { ++n; });
    if (n > 1)
      return launch.emitOpError(
          "air-to-csl-dialect: multiple herds not yet supported");
    return WalkResult::advance();
  });
  if (herdCount.wasInterrupted())
    return signalPassFailure();

  // 4. Lower each func.func that contains an air.launch.
  SmallVector<func::FuncOp, 4> targets;
  module.walk([&](func::FuncOp f) {
    bool hasLaunch = false;
    f.walk([&](air::LaunchOp) { hasLaunch = true; });
    if (hasLaunch) targets.push_back(f);
  });

  for (func::FuncOp f : targets)
    if (failed(lowerFunc(f)))
      return signalPassFailure();
}
```

You'll note `lowerFunc` is undefined — that's the next step.

- [ ] **Step 2: Sketch the lowerFunc helper**

Add this above `runOnOperation` (still inside the anonymous namespace):

```cpp
// Forward declaration so runOnOperation can call it.
LogicalResult lowerFunc(func::FuncOp func);

// Implementation: walk the func body, find the single air.launch, replace
// it with a csl.spatial_placement and host-side csl.export_name ops.
LogicalResult lowerFunc(func::FuncOp func) {
  // Find the single air.launch (validated to exist by the caller).
  air::LaunchOp launch = nullptr;
  func.walk([&](air::LaunchOp op) { launch = op; });
  if (!launch) return success();  // shouldn't happen given caller filter

  // Find the single segment + herd inside it.
  air::SegmentOp segment = nullptr;
  launch.walk([&](air::SegmentOp op) { segment = op; });
  air::HerdOp herd = nullptr;
  if (segment) segment.walk([&](air::HerdOp op) { herd = op; });
  if (!herd) {
    return launch.emitOpError(
        "air-to-csl-dialect: launch must contain a segment with a herd");
  }

  OpBuilder builder(launch);
  Location loc = launch.getLoc();

  // Collect the host-side memref args from the func signature. The first
  // (N-1) are inputs, the last 1 is the output. Convention: this is the
  // milestone's heuristic; future milestones can take an explicit
  // direction attribute on func args.
  SmallVector<BlockArgument, 4> hostArgs(func.getArguments().begin(),
                                         func.getArguments().end());
  if (hostArgs.size() != 3) {
    return func.emitOpError(
        "air-to-csl-dialect: milestone expects exactly 3 memref args (a, b, c)");
  }
  StringRef inNames[2] = {"a", "b"};
  StringRef outName = "c";

  // Build the csl.spatial_placement op.
  auto sp = builder.create<csl::SpatialPlacementOp>(loc);
  builder.setInsertionPointToStart(&sp.getBody().emplaceBlock());

  // Build the csl.kernel containing csl.var x3, csl.func @compute, csl.comptime.
  auto kernelTy = csl::KernelType::get(builder.getContext());
  auto kernel = builder.create<csl::KernelOp>(loc, kernelTy);
  kernel->setAttr("source_file", builder.getStringAttr("vecadd_pe.csl"));
  Block *kBlock = &kernel.getBody().emplaceBlock();
  builder.setInsertionPointToStart(kBlock);

  // Three csl.var declarations: a_buf, b_buf, c_buf.
  StringRef varNames[3] = {"a_buf", "b_buf", "c_buf"};
  SmallVector<Value, 3> varSyms;  // populated by csl.var creators
  for (int i = 0; i < 3; ++i) {
    auto memTy = cast<MemRefType>(hostArgs[i].getType());
    auto v = builder.create<csl::VarOp>(loc, builder.getStringAttr(varNames[i]),
                                        TypeAttr::get(memTy));
    varSyms.push_back(v.getResult());
  }

  // Build csl.func @compute() with the herd body moved in. Map original
  // herd-arg memref values to the corresponding csl.var symbols.
  auto computeFn = builder.create<csl::FuncOp>(
      loc, builder.getStringAttr("compute"),
      TypeAttr::get(builder.getFunctionType({}, {})));
  Block *fnBlock = &computeFn.getBody().emplaceBlock();
  builder.setInsertionPointToStart(fnBlock);

  // Move the herd body's ops into the func body, mapping the herd-arg
  // memref values to the csl.var symbol references.
  IRMapping mapping;
  // herd.getKernelOperands() returns the SSA values passed into the herd;
  // these correspond positionally to the launch args, which themselves
  // correspond to the func args. So mapping[i] = varSyms[i].
  auto herdBlockArgs = herd.getKernelArguments();
  for (auto [orig, sym] : llvm::zip(herdBlockArgs, varSyms))
    mapping.map(orig, sym);

  // Walk the herd body and clone each op into the func body, except for
  // the herd terminator (we'll add a csl.return).
  for (Operation &op : herd.getBody()->without_terminator())
    builder.clone(op, mapping);

  builder.create<csl::ReturnOp>(loc);

  // Build the csl.comptime block with four csl.export_symbol ops.
  builder.setInsertionPointToEnd(kBlock);
  auto comptime = builder.create<csl::ComptimeOp>(loc);
  Block *ctBlock = &comptime.getBody().emplaceBlock();
  builder.setInsertionPointToStart(ctBlock);
  for (auto [varName, alias] :
       llvm::zip(ArrayRef<StringRef>(varNames), ArrayRef<StringRef>{"a","b","c"})) {
    builder.create<csl::ExportSymbolOp>(loc,
        FlatSymbolRefAttr::get(builder.getContext(), varName),
        builder.getStringAttr(alias));
  }
  builder.create<csl::ExportSymbolOp>(loc,
      FlatSymbolRefAttr::get(builder.getContext(), "compute"),
      /*alias=*/StringAttr());

  // Build the csl.code_region (1×1, no routes/colors) and csl.place.
  builder.setInsertionPointAfter(kernel);
  auto regionTy = csl::CodeRegionType::get(builder.getContext());
  auto region = builder.create<csl::CodeRegionOp>(loc, regionTy,
                                                  /*routes=*/ValueRange{},
                                                  /*colors=*/ValueRange{});
  region->setAttr("width",  builder.getI64IntegerAttr(1));
  region->setAttr("height", builder.getI64IntegerAttr(1));
  // Empty body: code_region requires a region but for 1×1 we have nothing
  // to put in it.
  region.getBody().emplaceBlock();

  auto place = builder.create<csl::PlaceOp>(loc, region.getResult(), kernel.getResult());
  place->setAttr("x", builder.getI64IntegerAttr(0));
  place->setAttr("y", builder.getI64IntegerAttr(0));

  // Done with the spatial_placement body.
  builder.setInsertionPointAfter(sp);

  // Add host-level csl.export_name ops with direction attributes.
  for (int i = 0; i < 2; ++i) {
    auto memTy = cast<MemRefType>(hostArgs[i].getType());
    builder.create<csl::ExportNameOp>(loc,
        builder.getStringAttr(inNames[i]),
        TypeAttr::get(memTy),
        builder.getStringAttr("in"));
  }
  {
    auto memTy = cast<MemRefType>(hostArgs[2].getType());
    builder.create<csl::ExportNameOp>(loc,
        builder.getStringAttr(outName),
        TypeAttr::get(memTy),
        builder.getStringAttr("out"));
  }
  // The host-callable function name (compute), no direction.
  builder.create<csl::ExportNameOp>(loc,
      builder.getStringAttr("compute"),
      TypeAttr::get(builder.getFunctionType({}, {})),
      /*direction=*/StringAttr());

  // Erase the original air.launch (and everything inside it).
  launch.erase();
  return success();
}
```

**Note:** the constructor calls (`builder.create<csl::SpatialPlacementOp>(loc)`, etc.) assume the existing CSL op constructors. The exact signatures may differ from these — read `mlir/include/air/Dialect/CSL/CSLOps.h` and the corresponding `.h.inc` to see the actual constructor signatures, and adjust the calls. This is mechanical: the *structure* of what we're building is correct; only the constructor argument list may need tweaks.

- [ ] **Step 3: Build**

Run: `cd build && ninja install 2>&1 | tail -40`

Expected: clean build. Compile errors will tell you which constructor signatures need adjusting.

- [ ] **Step 4: Run the FileCheck test**

Run: `cd build && lit -v ../mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir`

Expected: PASS.

If FileCheck fails, the IR is structurally close but the printed form differs from what the CHECK lines expect. Two responses:
1. If your output is genuinely correct CSL but printed slightly differently (whitespace, attribute ordering), update the CHECK lines to match (CHECK-DAG is your friend).
2. If your output has a structural problem (missing op, wrong nesting), fix the pass implementation.

- [ ] **Step 5: Commit**

```bash
git add mlir/lib/Conversion/AIRToCSLDialect/AIRToCSLDialect.cpp
git commit -m "feat(air-to-csl-dialect): implement 1x1 vecadd lowering

Walks func.func containing air.launch/segment/herd, validates
milestone scope (1x1 only, no async, no channels, static memrefs,
supported eltypes), then builds the csl.spatial_placement IR shape
from the spec: csl.kernel with three csl.var, csl.func @compute()
with the herd body moved in via IRMapping, csl.comptime with
csl.export_symbol exports, csl.code_region (1x1), csl.place,
host-level csl.export_name ops with direction attributes.

Test: mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir"
```

### Task 2.4-2.10: Layer-rejection tests

Repeat for each rejection case from spec §7.2. Each test is ~15 lines and exists to lock in a specific error message so future scope expansion can't accidentally pass through the gate.

For each rejection case, the steps are identical:

- [ ] **Step a: Write the test file**
- [ ] **Step b: Run it, verify the expected error appears**
- [ ] **Step c: Commit**

The test files are listed below with their content. Add them all in a single phase of work.

#### Task 2.4: reject_2x2_herd.mlir

```mlir
// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s

// CHECK: only 1x1 herds supported

module {
  func.func @bad(%a: memref<256xf32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%a) : memref<256xf32> {
      air.segment @s args(%a1=%a0) : memref<256xf32> {
        air.herd @h tile(%htx,%hty) in (%hsx=%c2,%hsy=%c2) args(%a2=%a1) : memref<256xf32> {
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

#### Task 2.5: reject_channel_op.mlir

```mlir
// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s

// CHECK: inter-PE channels not yet supported

module {
  air.channel @ch [1, 1]
  func.func @bad(%a: memref<256xf32>) {
    %c0 = arith.constant 0 : index
    air.channel.put @ch[%c0, %c0] (%a[] [] []) : (memref<256xf32>)
    return
  }
}
```

#### Task 2.6: reject_dynamic_memref.mlir

```mlir
// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s

// CHECK: kernel memrefs must be statically shaped

module {
  func.func @bad(%a: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%a) : memref<?xf32> {
      air.segment @s args(%a1=%a0) : memref<?xf32> {
        air.herd @h tile(%htx,%hty) in (%hsx=%c1,%hsy=%c1) args(%a2=%a1) : memref<?xf32> {
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

#### Task 2.7: reject_unsupported_eltype.mlir

```mlir
// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s

// CHECK: unsupported element type

module {
  func.func @bad(%a: memref<256xf64>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%a) : memref<256xf64> {
      air.segment @s args(%a1=%a0) : memref<256xf64> {
        air.herd @h tile(%htx,%hty) in (%hsx=%c1,%hsy=%c1) args(%a2=%a1) : memref<256xf64> {
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

#### Task 2.8: reject_async_token.mlir

```mlir
// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s

// CHECK: async operations not supported

module {
  func.func @bad(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%a, %b0=%b, %c0=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @s args(%a1=%a0, %b1=%b0, %c1_=%c0)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        air.herd @h tile(%htx,%hty) in (%hsx=%c1,%hsy=%c1)
            args(%a2=%a1, %b2=%b1, %c2=%c1_) : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          %t = air.execute -> (!air.async.token) {
            air.execute_terminator
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

#### Task 2.9: reject_dma_in_body.mlir

```mlir
// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s

// CHECK: dma_memcpy_nd not yet supported

module {
  func.func @bad(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%a, %b0=%b, %c0=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @s args(%a1=%a0, %b1=%b0, %c1_=%c0)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        air.herd @h tile(%htx,%hty) in (%hsx=%c1,%hsy=%c1)
            args(%a2=%a1, %b2=%b1, %c2=%c1_) : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          air.dma_memcpy_nd (%c2[][][], %a2[][][]) : (memref<256xf32>, memref<256xf32>)
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

#### Task 2.10: reject_multiple_herds.mlir

```mlir
// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s

// CHECK: multiple herds not yet supported

module {
  func.func @bad(%a: memref<256xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%a) : memref<256xf32> {
      air.segment @s args(%a1=%a0) : memref<256xf32> {
        air.herd @h1 tile(%htx,%hty) in (%hsx=%c1,%hsy=%c1) args(%a2=%a1) : memref<256xf32> {
          air.herd_terminator
        }
        air.herd @h2 tile(%htx2,%hty2) in (%hsx2=%c1,%hsy2=%c1) args(%a3=%a1) : memref<256xf32> {
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

For each of Tasks 2.4-2.10:

- [ ] **Step a:** Save the file at `mlir/test/Conversion/AIRToCSLDialect/<filename>.mlir`.
- [ ] **Step b:** Run `cd build && lit -v ../mlir/test/Conversion/AIRToCSLDialect/<filename>.mlir`. Expected: PASS (the test asserts the error appears, so a passing rejection test is the success case).
- [ ] **Step c:** After all 7 rejection tests pass, commit them as one commit:

```bash
git add mlir/test/Conversion/AIRToCSLDialect/reject_*.mlir
git commit -m "test(air-to-csl-dialect): pin all milestone-scope rejections

One test per failure mode from spec §7.2. These guard against
accidental scope expansion."
```

---

## Phase 3 — Extend `csl-to-csl-rt`

**Goal:** Convert the IR shape from Phase 2 into `csl_rt.*` ops including the full host-side runtime sequence.

### Task 3.1: Failing FileCheck test

**Files:**
- Create: `mlir/test/Conversion/CSLToCSLRuntime/vecadd_with_kernel.mlir`

- [ ] **Step 1: Write the test**

```mlir
// RUN: air-opt --csl-to-csl-rt %s | FileCheck %s

// Hand-written copy of the post-air-to-csl-dialect IR for vecadd.
// This test is independent of the air-to-csl-dialect pass.

// CHECK-LABEL: func.func @vecadd
// CHECK: %[[L:.*]] = csl_rt.create_layout
// CHECK: %[[R:.*]] = csl_rt.create_code_region %[[L]]
// CHECK: csl_rt.place %[[R]]
// CHECK-DAG: csl_rt.export_name %[[L]] "a"
// CHECK-DAG: csl_rt.export_name %[[L]] "b"
// CHECK-DAG: csl_rt.export_name %[[L]] "c"
// CHECK-DAG: csl_rt.export_name %[[L]] "compute"
// CHECK: %[[ART:.*]] = csl_rt.compile %[[L]]
// CHECK: %[[RT:.*]] = csl_rt.runtime_create %[[ART]]
// CHECK: %{{.*}} = csl_rt.load %[[RT]]
// CHECK: %{{.*}} = csl_rt.memcpy_h2d %{{.*}} {{.*}} "a"
// CHECK: %{{.*}} = csl_rt.memcpy_h2d %{{.*}} {{.*}} "b"
// CHECK: %{{.*}} = csl_rt.launch %{{.*}} "compute"
// CHECK: %{{.*}} = csl_rt.memcpy_d2h %{{.*}} "c"
// CHECK: csl_rt.stop %{{.*}}

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    csl.spatial_placement {
      %k = csl.kernel {
        csl.var @a_buf : memref<256xf32>
        csl.var @b_buf : memref<256xf32>
        csl.var @c_buf : memref<256xf32>
        csl.func @compute() : () -> () {
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

- [ ] **Step 2: Run, verify it fails**

Run: `cd build && lit -v ../mlir/test/Conversion/CSLToCSLRuntime/vecadd_with_kernel.mlir`

Expected: FAIL because the existing pass doesn't yet emit the runtime sequence.

### Task 3.2: Add export_name + runtime sequence patterns

**Files:**
- Modify: `mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp`

- [ ] **Step 1: Read the existing pass to find where to add**

Run: `cat mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp`

Look for the `runOnOperation()` method and the existing pattern application. New patterns will be appended to that pattern list.

- [ ] **Step 2: Add a helper that synthesizes the runtime sequence**

Add this function above the pass class:

```cpp
// Helper: given the host func.func that contains a csl.spatial_placement
// (already lowered to csl_rt.create_layout / .compile) and a list of
// csl.export_name ops, append the host-side runtime sequence to the func
// body: runtime_create, load, memcpy_h2d for each "in", launch, memcpy_d2h
// for each "out", stop.
//
// The `compileResult` is the SSA value produced by csl_rt.compile.
// The `exports` list is in declaration order; "in" exports become h2d,
// "out" exports become d2h, exports with no direction whose type is a
// FunctionType become the launch target.
LogicalResult synthesizeRuntimeSequence(OpBuilder &builder, Location loc,
                                        Value compileResult,
                                        ArrayRef<csl::ExportNameOp> exports) {
  // 1. runtime_create
  auto runtimeTy = csl_rt::RuntimeType::get(builder.getContext());
  auto rt = builder.create<csl_rt::RuntimeCreateOp>(loc, runtimeTy, compileResult);

  // 2. load
  auto loaded = builder.create<csl_rt::LoadOp>(loc, runtimeTy, rt.getResult());
  Value cur = loaded.getResult();

  // 3. memcpy_h2d for each "in" export
  StringRef launchName;
  SmallVector<csl::ExportNameOp, 4> outExports;
  for (csl::ExportNameOp e : exports) {
    StringAttr dir = e.getDirectionAttr();
    Type t = e.getExportedType();
    if (dir && dir.getValue() == "in") {
      auto memTy = cast<MemRefType>(t);
      int64_t elems = memTy.getNumElements();
      auto h2d = builder.create<csl_rt::MemcpyH2dOp>(
          loc, runtimeTy, cur,
          /*dest_id=*/builder.getI32IntegerAttr(0),  // resolved at runtime via get_id
          /*src=*/builder.getStringAttr(e.getSymName()),
          /*px=*/builder.getIndexAttr(0),
          /*py=*/builder.getIndexAttr(0),
          /*w=*/builder.getIndexAttr(1),
          /*h=*/builder.getIndexAttr(1),
          /*elem_per_pe=*/builder.getIndexAttr(elems));
      cur = h2d.getResult();
    } else if (dir && dir.getValue() == "out") {
      outExports.push_back(e);
    } else if (!dir && isa<FunctionType>(t)) {
      launchName = e.getSymName();
    }
  }

  // 4. launch
  if (launchName.empty())
    return failure();  // no host-callable function found
  auto launch = builder.create<csl_rt::LaunchOp>(
      loc, runtimeTy, cur, builder.getStringAttr(launchName));
  cur = launch.getResult();

  // 5. memcpy_d2h for each "out" export
  for (csl::ExportNameOp e : outExports) {
    auto memTy = cast<MemRefType>(e.getExportedType());
    int64_t elems = memTy.getNumElements();
    auto d2h = builder.create<csl_rt::MemcpyD2hOp>(
        loc, runtimeTy, cur,
        builder.getStringAttr(e.getSymName()),
        builder.getI32IntegerAttr(0),
        builder.getIndexAttr(0),
        builder.getIndexAttr(0),
        builder.getIndexAttr(1),
        builder.getIndexAttr(1),
        builder.getIndexAttr(elems));
    cur = d2h.getResult();
  }

  // 6. stop
  builder.create<csl_rt::StopOp>(loc, cur);
  return success();
}
```

**Note:** the constructor argument order/types for `MemcpyH2dOp`, `MemcpyD2hOp`, `LaunchOp`, etc. above are educated guesses. Read `mlir/include/air/Dialect/CSLRuntime/CSLRuntimeOps.td` to see the exact `arguments` ordering and adjust each `builder.create<...>` call to match. The builder error messages will tell you which positional args are wrong.

- [ ] **Step 3: Wire export_name and synthesizeRuntimeSequence into the pass**

In the existing `runOnOperation()` method (or equivalent pattern-application code), after the existing patterns produce the `csl_rt.compile` op, add a step that:
1. Walks the host func.func to collect all `csl.export_name` ops in declaration order (before erasing them).
2. For each, creates the corresponding `csl_rt.export_name` on the layout (the layout SSA value should be available from the create_layout op).
3. Calls `synthesizeRuntimeSequence` after the compile op, passing in the collected exports.
4. Erases the original `csl.export_name` ops.

The exact integration depends on the existing pass structure; aim for minimal change. If the existing pass uses dialect conversion patterns, add a new `OpConversionPattern<csl::ExportNameOp>`. If it uses a manual walk, append to it.

- [ ] **Step 4: Build and run the test**

Run: `cd build && ninja install && lit -v ../mlir/test/Conversion/CSLToCSLRuntime/vecadd_with_kernel.mlir`

Expected: PASS.

- [ ] **Step 5: Make sure existing layout_to_runtime.mlir still passes**

Run: `cd build && lit -v ../mlir/test/Conversion/CSLToCSLRuntime/`

Expected: ALL pass. We must not regress the existing trivial test.

- [ ] **Step 6: Commit**

```bash
git add mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp \
        mlir/test/Conversion/CSLToCSLRuntime/vecadd_with_kernel.mlir
git commit -m "feat(csl-to-csl-rt): lower csl.export_name + synthesize runtime sequence

Adds host-side runtime sequence synthesis driven by the direction
attribute on csl.export_name: runtime_create, load, memcpy_h2d for
each 'in' buffer, launch for the function-typed export, memcpy_d2h
for each 'out' buffer, stop. Plus per-buffer csl_rt.export_name on
the layout.

Test: mlir/test/Conversion/CSLToCSLRuntime/vecadd_with_kernel.mlir"
```

### Task 3.3: Add rejection tests for csl-to-csl-rt

Add three more rejection tests for §7.3:

- `mlir/test/Conversion/CSLToCSLRuntime/reject_routes.mlir` — `csl.code_region routes(%c) colors()`, expects "routing not supported in milestone".
- `mlir/test/Conversion/CSLToCSLRuntime/reject_multi_region.mlir` — two `csl.code_region` ops, expects "single-region only".
- `mlir/test/Conversion/CSLToCSLRuntime/reject_extern_kernel.mlir` — `csl.place` referencing a kernel from a different `csl.spatial_placement`, expects "place must reference local kernel".

For each:
- [ ] **Step a:** Implement the corresponding rejection check in `CSLToCSLRuntime.cpp` (a quick walk over `csl.code_region` ops at the start of the pass).
- [ ] **Step b:** Add the test file with `// RUN: not air-opt -csl-to-csl-rt %s 2>&1 | FileCheck %s`.
- [ ] **Step c:** After all three, commit:

```bash
git add mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp \
        mlir/test/Conversion/CSLToCSLRuntime/reject_*.mlir
git commit -m "test(csl-to-csl-rt): pin milestone-scope rejections"
```

---

## Phase 4 — KernelEmitter rework in `CSLRuntimeToPy.cpp`

**Goal:** Replace the `emitKernelPrograms()` stub with a real walker that produces valid CSL source matching the golden file.

### Task 4.1: Failing FileCheck test

**Files:**
- Create: `mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir`

- [ ] **Step 1: Write the test**

```mlir
// RUN: air-opt %s | air-translate --emit-csl-rt -o %t/
// RUN: cat %t/vecadd_pe.csl | FileCheck %s

// Hand-written csl.kernel — independent of any earlier pass.

// CHECK-LABEL: const N: i32 = 256
// CHECK: var a_buf: [256]f32
// CHECK: var b_buf: [256]f32
// CHECK: var c_buf: [256]f32
// CHECK: fn compute() void {
// CHECK:   for (@range(i32, 256)) |i| {
// CHECK:     c_buf[i] = a_buf[i] + b_buf[i];
// CHECK:   }
// CHECK: }
// CHECK: comptime {
// CHECK-DAG: @export_symbol(a_buf, "a");
// CHECK-DAG: @export_symbol(b_buf, "b");
// CHECK-DAG: @export_symbol(c_buf, "c");
// CHECK-DAG: @export_symbol(compute);
// CHECK: }

module {
  csl.spatial_placement {
    %k = csl.kernel {
      csl.var @a_buf : memref<256xf32>
      csl.var @b_buf : memref<256xf32>
      csl.var @c_buf : memref<256xf32>
      csl.func @compute() : () -> () {
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
}
```

- [ ] **Step 2: Run, verify it fails**

Run: `cd build && lit -v ../mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir`

Expected: FAIL because the current `emitKernelPrograms` is a stub that emits comments, not real CSL.

### Task 4.2: Implement the KernelEmitter

**Files:**
- Modify: `mlir/lib/Targets/CSLRuntimeToPy.cpp`

- [ ] **Step 1: Replace the stub**

Open `mlir/lib/Targets/CSLRuntimeToPy.cpp`. Find the existing `emitKernelPrograms()` method (around line 91). Delete it entirely, and the `kernels` walk in `translate()`. Replace with the architecture below.

Above the existing classes, add:

```cpp
//===----------------------------------------------------------------------===//
// KernelEmitter — walks csl.kernel ops and writes valid CSL source files
//===----------------------------------------------------------------------===//

namespace {

class KernelEmitter {
public:
  KernelEmitter(StringRef outDir) : outDir(outDir.str()) {}

  // Walk a module and write one .csl file per csl.kernel found.
  // Returns failure if any unsupported op is encountered (no half-files).
  LogicalResult emitAll(ModuleOp module) {
    LogicalResult result = success();
    module.walk([&](csl::KernelOp kernel) {
      if (failed(emitOne(kernel))) {
        result = failure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

private:
  std::string outDir;
  // Per-kernel state:
  llvm::DenseMap<Value, std::string> nameTable;
  unsigned tmpCounter = 0;

  // Validate the entire kernel body before opening any output file.
  // If anything is unsupported, return failure with a diagnostic.
  LogicalResult validateKernel(csl::KernelOp kernel) {
    LogicalResult ok = success();
    kernel.walk([&](Operation *op) {
      if (isa<csl::KernelOp, csl::VarOp, csl::FuncOp, csl::ReturnOp,
              csl::ComptimeOp, csl::ExportSymbolOp, scf::ForOp, scf::YieldOp,
              memref::LoadOp, memref::StoreOp, arith::AddFOp,
              arith::ConstantOp, func::FuncOp>(op))
        return;
      // scf.for precondition: lo == 0, step == 1, constant hi
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        auto lo = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
        auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
        auto hi = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
        if (!lo || lo.value() != 0 || !step || step.value() != 1 || !hi) {
          op->emitOpError("KernelEmitter: scf.for requires lo=0, step=1, "
                          "constant hi");
          ok = failure();
        }
        return;
      }
      op->emitOpError("KernelEmitter: unsupported MLIR op for CSL kernel "
                      "emission: ") << op->getName();
      ok = failure();
    });
    return ok;
  }

  LogicalResult emitOne(csl::KernelOp kernel) {
    if (failed(validateKernel(kernel))) return failure();

    StringRef sourceFile = kernel->getAttrOfType<StringAttr>("source_file").getValue();
    SmallString<256> path(outDir);
    llvm::sys::path::append(path, sourceFile);

    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      kernel->emitError("KernelEmitter: cannot open ") << path << ": " << ec.message();
      return failure();
    }

    nameTable.clear();
    tmpCounter = 0;

    os << "// Generated CSL kernel from MLIR csl.kernel @" << kernel->getName() << ".\n\n";

    // Pass 1: emit csl.var declarations as `var name: [N]T;`
    int64_t commonN = -1;  // detect a shared dimension to emit as `const N`
    for (auto var : kernel.getOps<csl::VarOp>()) {
      auto memTy = cast<MemRefType>(var.getExportedType());
      int64_t n = memTy.getNumElements();
      if (commonN == -1) commonN = n;
      else if (commonN != n) commonN = -2;  // not all the same
    }
    if (commonN > 0) {
      os << "const N: i32 = " << commonN << ";\n\n";
    }

    for (auto var : kernel.getOps<csl::VarOp>()) {
      auto memTy = cast<MemRefType>(var.getExportedType());
      int64_t n = memTy.getNumElements();
      Type elt = memTy.getElementType();
      std::string sizeExpr =
          (commonN > 0 && n == commonN) ? "N" : std::to_string(n);
      os << "var " << var.getSymName() << ": [" << sizeExpr << "]"
         << cslElementTypeName(elt) << ";\n";
      // Bind the var symbol's SSA value to its CSL identifier (we'll need this
      // when memref.load/store reference the var).
      nameTable[var.getResult()] = var.getSymName().str();
    }
    os << "\n";

    // Pass 2: emit csl.func bodies.
    for (auto fn : kernel.getOps<csl::FuncOp>()) {
      if (failed(emitFunc(fn, os))) return failure();
      os << "\n";
    }

    // Pass 3: emit csl.comptime block.
    for (auto ct : kernel.getOps<csl::ComptimeOp>()) {
      os << "comptime {\n";
      for (auto exp : ct.getOps<csl::ExportSymbolOp>()) {
        os << "  @export_symbol(" << exp.getSymbol().getValue();
        if (auto alias = exp.getAliasAttr())
          os << ", \"" << alias.getValue() << "\"";
        os << ");\n";
      }
      os << "}\n";
    }

    return success();
  }

  static StringRef cslElementTypeName(Type t) {
    if (t.isF32()) return "f32";
    if (t.isF16()) return "f16";
    if (t.isInteger(32)) return "i32";
    if (t.isInteger(16)) return "i16";
    return "f32";  // fallback (caller should have rejected this earlier)
  }

  std::string mintTmpName() {
    return "t" + std::to_string(tmpCounter++);
  }

  LogicalResult emitFunc(csl::FuncOp fn, llvm::raw_ostream &os) {
    os << "fn " << fn.getSymName() << "() void {\n";
    if (failed(emitRegion(fn.getBody(), os, /*indent=*/2))) return failure();
    os << "}\n";
    return success();
  }

  LogicalResult emitRegion(Region &region, llvm::raw_ostream &os, int indent) {
    for (Operation &op : region.front()) {
      if (failed(emitOp(&op, os, indent))) return failure();
    }
    return success();
  }

  LogicalResult emitOp(Operation *op, llvm::raw_ostream &os, int indent) {
    std::string ind(indent, ' ');

    // Constants: bind name only (used inline at use sites).
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      auto v = cst.getResult();
      if (auto idx = dyn_cast<IntegerAttr>(cst.getValue()))
        nameTable[v] = std::to_string(idx.getInt());
      else if (auto f = dyn_cast<FloatAttr>(cst.getValue()))
        nameTable[v] = std::to_string(f.getValueAsDouble());
      return success();
    }

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Already validated: lo=0, step=1, constant hi.
      auto hiCst = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
      int64_t hi = hiCst.value();
      std::string iName = "i";
      // Bind the induction variable.
      nameTable[forOp.getInductionVar()] = iName;
      os << ind << "for (@range(i32, " << hi << ")) |" << iName << "| {\n";
      if (failed(emitRegion(forOp.getRegion(), os, indent + 2))) return failure();
      os << ind << "}\n";
      return success();
    }

    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      // Bind: tN = buf[i]
      std::string tmp = mintTmpName();
      nameTable[load.getResult()] = tmp;
      os << ind << "var " << tmp << " = " << nameTable[load.getMemRef()]
         << "[" << nameTable[load.getIndices().front()] << "];\n";
      return success();
    }

    if (auto add = dyn_cast<arith::AddFOp>(op)) {
      std::string tmp = mintTmpName();
      nameTable[add.getResult()] = tmp;
      os << ind << "var " << tmp << " = " << nameTable[add.getLhs()]
         << " + " << nameTable[add.getRhs()] << ";\n";
      return success();
    }

    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      os << ind << nameTable[store.getMemRef()] << "["
         << nameTable[store.getIndices().front()] << "] = "
         << nameTable[store.getValueToStore()] << ";\n";
      return success();
    }

    if (isa<csl::ReturnOp, scf::YieldOp>(op)) {
      // Function/loop terminator — no text.
      return success();
    }

    return op->emitOpError("KernelEmitter: unhandled op (should have been "
                           "rejected by validateKernel)");
  }
};

} // namespace
```

Then in `CSLRuntimeToPyTranslator::translate()`, replace the kernel-handling block (Step 1 and Step 2 in the existing code) with:

```cpp
// Step 1: Emit kernel .csl files via KernelEmitter.
KernelEmitter kEmit(outDir);  // outDir from translator construction
if (failed(kEmit.emitAll(module))) return mlir::failure();

// Step 2: removed (was the stub).

// Step 3 (existing): generate run.py
// ... (HostEmitter, modified in Phase 5)
```

You will also need to plumb the output directory into the translator. Look at how `air-translate` constructs the translator and pass through the `-o` value (the existing translator may already do this for the run.py path; reuse the same mechanism).

- [ ] **Step 2: Build**

Run: `cd build && ninja install 2>&1 | tail -40`

Expected: clean build. Compilation errors will tell you about wrong type names (`getExportedType` vs `getType`, etc.) — fix mechanically by reading the actual op def.

- [ ] **Step 3: Run the test**

Run: `cd build && lit -v ../mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir`

Expected: PASS.

- [ ] **Step 4: Compare to the golden file**

```bash
mkdir -p /tmp/kemit_check && \
  air-opt mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir | \
  air-translate --emit-csl-rt -o /tmp/kemit_check/ && \
  diff -u mlir/test/Conversion/AIRToCSL/golden/vecadd_pe.csl.golden /tmp/kemit_check/vecadd_pe.csl
```

Expected: minimal diff (whitespace only, ideally). Major differences are bugs.

- [ ] **Step 5: Commit**

```bash
git add mlir/lib/Targets/CSLRuntimeToPy.cpp \
        mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir
git commit -m "feat(csl-emit): real KernelEmitter with 7-op dispatch table

Replaces the emitKernelPrograms() stub with a walker that produces
valid CSL source matching the bootstrap golden file. Dispatch table:
csl.var, csl.func, csl.return, csl.comptime+csl.export_symbol,
arith.constant (binding), scf.for, memref.load, memref.store,
arith.addf.

Validates the entire kernel body before opening the output file —
no half-written .csl on unsupported ops.

Test: mlir/test/Targets/CSLRuntimeToCSL/vecadd_kernel_emit.mlir"
```

---

## Phase 5 — HostEmitter rework

**Goal:** Make the `run.py` emission produce a runnable script matching the golden file.

### Task 5.1: Failing FileCheck test

**Files:**
- Create: `mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir`

- [ ] **Step 1: Write the test**

```mlir
// RUN: air-opt %s | air-translate --emit-csl-rt -o %t/
// RUN: cat %t/run.py | FileCheck %s

// Hand-written csl_rt.* sequence — independent of earlier passes.

// CHECK: from cerebras.sdk.runtime.sdkruntimepybind import
// CHECK: def build_layout(platform):
// CHECK:   layout = SdkLayout(platform)
// CHECK:   region = layout.create_code_region("vecadd_pe.csl"
// CHECK:   region.place(0, 0)
// CHECK:   layout.export_name("a"
// CHECK:   layout.export_name("b"
// CHECK:   layout.export_name("c"
// CHECK:   layout.export_name("compute"
// CHECK:   compile_artifacts = layout.compile(out_prefix="out")
// CHECK:   return compile_artifacts
// CHECK: def main():
// CHECK:   parser = argparse.ArgumentParser()
// CHECK:   parser.add_argument("--cmaddr"
// CHECK:   parser.add_argument("--arch"
// CHECK:   parser.add_argument("--check"
// CHECK:   runtime = SdkRuntime(artifacts, platform, memcpy_required=True)
// CHECK:   runtime.load()
// CHECK:   runtime.memcpy_h2d({{.*}}, a, 0, 0, 1, 1, 256)
// CHECK:   runtime.memcpy_h2d({{.*}}, b, 0, 0, 1, 1, 256)
// CHECK:   runtime.launch("compute")
// CHECK:   runtime.memcpy_d2h(c, {{.*}}, 0, 0, 1, 1, 256)
// CHECK:   runtime.stop()
// CHECK:   if args.check:
// CHECK:     if not np.array_equal(c, expected):
// CHECK:     print("PASS")

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    %l = csl_rt.create_layout : !csl_rt.layout
    %r = csl_rt.create_code_region %l "vecadd_pe.csl", "vecadd", 1 : index, 1 : index
        : (!csl_rt.layout) -> !csl_rt.code_region
    %placed = csl_rt.place %r at (0, 0) : (!csl_rt.code_region) -> !csl_rt.code_region
    %l1 = csl_rt.export_name %l "a", "<f32>[256]" : (!csl_rt.layout) -> !csl_rt.layout
    %l2 = csl_rt.export_name %l1 "b", "<f32>[256]" : (!csl_rt.layout) -> !csl_rt.layout
    %l3 = csl_rt.export_name %l2 "c", "<f32>[256]" : (!csl_rt.layout) -> !csl_rt.layout
    %l4 = csl_rt.export_name %l3 "compute", "fn()void" : (!csl_rt.layout) -> !csl_rt.layout
    %art = csl_rt.compile %l4 : (!csl_rt.layout) -> !csl_rt.compile_artifacts
    %rt = csl_rt.runtime_create %art : (!csl_rt.compile_artifacts) -> !csl_rt.runtime
    %loaded = csl_rt.load %rt : (!csl_rt.runtime) -> !csl_rt.runtime
    %h1 = csl_rt.memcpy_h2d %loaded 0 "a" at (0, 0) with_size (1, 1) elem_per_pe 256 : (!csl_rt.runtime) -> !csl_rt.runtime
    %h2 = csl_rt.memcpy_h2d %h1 1 "b" at (0, 0) with_size (1, 1) elem_per_pe 256 : (!csl_rt.runtime) -> !csl_rt.runtime
    %lc = csl_rt.launch %h2 "compute" : (!csl_rt.runtime) -> !csl_rt.runtime
    %d1 = csl_rt.memcpy_d2h %lc "c" from (0, 0) with_size (1, 1) elem_per_pe 256 : (!csl_rt.runtime) -> !csl_rt.runtime
    csl_rt.stop %d1 : !csl_rt.runtime
    return
  }
}
```

(The hand-written `csl_rt.*` operands above use the existing assemblyFormat syntax from `mlir/include/air/Dialect/CSLRuntime/CSLRuntimeOps.td`. If the actual syntax differs, adjust to match — these are illustrative.)

- [ ] **Step 2: Run, verify it fails**

Run: `cd build && lit -v ../mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir`

Expected: FAIL because the existing `emitRunPy` doesn't produce all the required pieces.

### Task 5.2: Audit and rework HostEmitter

**Files:**
- Modify: `mlir/lib/Targets/CSLRuntimeToPy.cpp`

- [ ] **Step 1: Read the existing emitRunPy**

Run: `sed -n '120,377p' mlir/lib/Targets/CSLRuntimeToPy.cpp`

Identify what already works (the `build_layout` skeleton, `import` block, the basic `def main()`) and what's missing or stubbed (numpy buffer materialization, memcpy parameters from csl_rt ops, validator).

- [ ] **Step 2: Implement the missing pieces**

Replace the body of `emitRunPy()` so that it walks the `csl_rt.*` ops in declaration order and emits the corresponding Python lines. The structure should mirror the golden `run.py.golden`. Major sub-steps:

1. **Header + imports**: hardcoded literal copy from the golden (already present, possibly).
2. **`build_layout(platform)` body**: walk the host func.func, emit one Python line per `csl_rt.*` op in the layout-build phase (`create_layout`, `create_code_region`, `place`, `export_name`s, `compile`).
3. **`def main()`**: emit argparse, platform construction, `build_layout(platform)` call, `SdkRuntime` construction, `runtime.load()`.
4. **Buffer materialization**: scan the `csl_rt.export_name` ops with the original CSL `direction` attribute (looked up via the kernel's exports — or alternatively a separate metadata pass). For `"in"` exports, emit `name = np.arange(N, dtype=np.float32)` (or `*2.0` for the second one — this keeps the test data deterministic and distinguishable). For `"out"` exports, emit `name = np.zeros(N, dtype=np.float32)`. Define `expected = a + b` after the inputs.
5. **Memcpy + launch**: walk `csl_rt.memcpy_h2d`/`launch`/`memcpy_d2h`/`stop` ops in order, emitting `runtime.memcpy_h2d(id_X, X, ...)` etc. Use `runtime.get_id("X")` to get the ID for each name and store as `id_X`.
6. **Validator block**: hardcoded `if args.check: ... print("PASS")` matching the golden.
7. **`if __name__ == "__main__": main()`**: hardcoded.

The cleanest implementation is to make the HostEmitter a struct that walks the host func.func once and emits the script in order. Pseudo-code:

```cpp
class HostEmitter {
public:
  HostEmitter(StringRef outDir, llvm::raw_ostream &os) : outDir(outDir.str()), os(os) {}

  LogicalResult emit(ModuleOp module) {
    emitHeader();
    func::FuncOp host = findHostFunc(module);
    emitBuildLayout(host);
    emitMain(host);
    return success();
  }

private:
  std::string outDir;
  llvm::raw_ostream &os;

  void emitHeader() {
    os << "#!/usr/bin/env cs_python\n"
          "import argparse\n"
          "import sys\n"
          "import numpy as np\n"
          "from cerebras.sdk.runtime.sdkruntimepybind import (\n"
          "    SdkRuntime, SdkLayout, SdkTarget, SimfabConfig, get_platform,\n"
          ")\n\n";
  }

  func::FuncOp findHostFunc(ModuleOp m) {
    func::FuncOp r;
    m.walk([&](func::FuncOp f) {
      if (!r) r = f;
    });
    return r;
  }

  void emitBuildLayout(func::FuncOp host) {
    os << "def build_layout(platform):\n"
          "    layout = SdkLayout(platform)\n";
    for (auto &op : host.getBody().front()) {
      if (auto cr = dyn_cast<csl_rt::CreateCodeRegionOp>(op)) {
        os << "    region = layout.create_code_region(\""
           << cr.getSourceFile().getValue() << "\", \""
           << cr.getRegionName().getValue() << "\", "
           << cr.getWidth().getInt() << ", "
           << cr.getHeight().getInt() << ")\n";
      } else if (auto pl = dyn_cast<csl_rt::PlaceOp>(op)) {
        os << "    region.place(" << pl.getX().getInt()
           << ", " << pl.getY().getInt() << ")\n";
      } else if (auto en = dyn_cast<csl_rt::ExportNameOp>(op)) {
        os << "    layout.export_name(\"" << en.getName() << "\", \""
           << en.getTypeSpec() << "\")\n";
      } else if (isa<csl_rt::CompileOp>(op)) {
        os << "    return layout.compile(out_prefix=\"out\")\n\n";
        return;
      }
    }
  }

  void emitMain(func::FuncOp host) {
    os << "def main():\n"
          "    parser = argparse.ArgumentParser()\n"
          "    parser.add_argument(\"--cmaddr\", default=None)\n"
          "    parser.add_argument(\"--arch\", default=\"wse3\", choices=(\"wse2\",\"wse3\"))\n"
          "    parser.add_argument(\"--check\", action=\"store_true\")\n"
          "    args = parser.parse_args()\n\n"
          "    config = SimfabConfig()\n"
          "    target = SdkTarget.WSE3 if args.arch == \"wse3\" else SdkTarget.WSE2\n"
          "    platform = get_platform(args.cmaddr, config, target)\n\n"
          "    artifacts = build_layout(platform)\n"
          "    runtime = SdkRuntime(artifacts, platform, memcpy_required=True)\n"
          "    runtime.load()\n\n";

    // Buffer materialization based on csl_rt.export_name + direction.
    SmallVector<StringRef, 4> inNames;
    SmallVector<int64_t, 4> inSizes;
    SmallVector<StringRef, 4> outNames;
    SmallVector<int64_t, 4> outSizes;
    StringRef launchName;

    // Scan the host func body to discover names and sizes.
    // Strategy: csl_rt.export_name carries name + type-spec but the direction
    // information was on the original csl.export_name. After lowering, the
    // direction attribute should be propagated onto csl_rt.export_name as a
    // discardable attribute by the lowering pass — see Phase 3, ensure that
    // pass copies the attribute.
    for (auto en : host.getOps<csl_rt::ExportNameOp>()) {
      auto dir = en->getAttrOfType<StringAttr>("direction");
      auto typeSpec = en.getTypeSpec();  // e.g. "<f32>[256]"
      int64_t n = parseElemCount(typeSpec);  // helper that pulls 256 out of "<f32>[256]"
      if (dir && dir.getValue() == "in") {
        inNames.push_back(en.getName());
        inSizes.push_back(n);
      } else if (dir && dir.getValue() == "out") {
        outNames.push_back(en.getName());
        outSizes.push_back(n);
      } else if (typeSpec.contains("fn")) {
        launchName = en.getName();
      }
    }

    // Emit numpy buffer creation.
    for (size_t i = 0; i < inNames.size(); ++i) {
      double mult = (i == 0) ? 1.0 : 2.0;  // make a, b distinguishable
      os << "    " << inNames[i] << " = np.arange("
         << inSizes[i] << ", dtype=np.float32)";
      if (mult != 1.0) os << " * " << mult;
      os << "\n";
    }
    for (size_t i = 0; i < outNames.size(); ++i) {
      os << "    " << outNames[i] << " = np.zeros("
         << outSizes[i] << ", dtype=np.float32)\n";
    }
    os << "    expected = " << (inNames.size() >= 2
                                 ? std::string(inNames[0]) + " + " + std::string(inNames[1])
                                 : std::string("a"))
       << "\n\n";

    // Emit get_id calls.
    for (StringRef n : inNames)
      os << "    id_" << n << " = runtime.get_id(\"" << n << "\")\n";
    for (StringRef n : outNames)
      os << "    id_" << n << " = runtime.get_id(\"" << n << "\")\n";
    os << "\n";

    // Emit memcpy_h2d / launch / memcpy_d2h based on csl_rt op order.
    for (auto &op : host.getBody().front()) {
      if (auto h2d = dyn_cast<csl_rt::MemcpyH2dOp>(op)) {
        StringRef name = h2d.getSrc();
        int64_t n = h2d.getElemPerPe().getInt();
        os << "    runtime.memcpy_h2d(id_" << name << ", " << name
           << ", 0, 0, 1, 1, " << n << ")\n";
      } else if (auto lc = dyn_cast<csl_rt::LaunchOp>(op)) {
        os << "    runtime.launch(\"" << lc.getName() << "\")\n";
      } else if (auto d2h = dyn_cast<csl_rt::MemcpyD2hOp>(op)) {
        StringRef name = d2h.getDest();
        int64_t n = d2h.getElemPerPe().getInt();
        os << "    runtime.memcpy_d2h(" << name << ", id_" << name
           << ", 0, 0, 1, 1, " << n << ")\n";
      } else if (isa<csl_rt::StopOp>(op)) {
        os << "    runtime.stop()\n\n";
      }
    }

    // Emit validator.
    os << "    if args.check:\n"
          "        if not np.array_equal(c, expected):\n"
          "            mismatches = np.where(c != expected)[0]\n"
          "            print(f\"FAIL: {len(mismatches)} mismatches\", file=sys.stderr)\n"
          "            for i in mismatches[:8]:\n"
          "                print(f\"  c[{i}] = {c[i]}  expected {expected[i]}\", file=sys.stderr)\n"
          "            sys.exit(1)\n"
          "        print(\"PASS\")\n"
          "    sys.exit(0)\n\n"
          "if __name__ == \"__main__\":\n"
          "    main()\n";
  }

  static int64_t parseElemCount(StringRef typeSpec) {
    // Parse "<f32>[256]" → 256, or "f32" → 1, etc.
    auto lb = typeSpec.find('[');
    auto rb = typeSpec.find(']');
    if (lb == StringRef::npos || rb == StringRef::npos) return 1;
    int64_t n = 0;
    typeSpec.substr(lb + 1, rb - lb - 1).getAsInteger(10, n);
    return n;
  }
};
```

**Important caveat for Phase 3 integration:** the HostEmitter needs the `direction` attribute on `csl_rt.export_name`. Phase 3's csl-to-csl-rt extension must preserve/copy the direction attribute from `csl.export_name` onto the new `csl_rt.export_name` op. If the existing `csl_rt.export_name` op def doesn't accept arbitrary attributes (it should, per MLIR convention), add it as a discardable attribute via `op->setAttr("direction", attr)` after creation.

- [ ] **Step 3: Build and run the test**

Run: `cd build && ninja install && lit -v ../mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir`

Expected: PASS.

- [ ] **Step 4: Diff against golden run.py**

```bash
mkdir -p /tmp/hemit_check && \
  air-opt mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir | \
  air-translate --emit-csl-rt -o /tmp/hemit_check/ && \
  diff -u mlir/test/Conversion/AIRToCSL/golden/run.py.golden /tmp/hemit_check/run.py
```

Expected: minimal diff. Major differences are bugs.

- [ ] **Step 5: Commit**

```bash
git add mlir/lib/Targets/CSLRuntimeToPy.cpp \
        mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir
git commit -m "feat(csl-emit): real HostEmitter producing runnable run.py

Walks csl_rt.* ops in declaration order, emitting:
- import block + build_layout()
- argparse + platform construction
- numpy buffer materialization driven by direction-tagged exports
- get_id, memcpy_h2d, launch, memcpy_d2h, stop sequence
- np.array_equal validator with PASS/FAIL output

Test: mlir/test/Targets/CSLRuntimeToCSL/vecadd_run_py_emit.mlir"
```

---

## Phase 6 — End-to-end hardware test

**Goal:** The single test that gates milestone success.

### Task 6.1: Write the integration test

**Files:**
- Create: `test/csl/test_vecadd_e2e.py`
- Create: `test/csl/conftest.py` (if not already present)

- [ ] **Step 1: Create the test file**

```python
"""End-to-end test for the AIR → CSL vecadd milestone.

Compiles vecadd.mlir through the full pipeline and runs the result
on the CS-3 attached to this machine. This is the milestone's
definition of done.

Reference: docs/superpowers/specs/2026-04-13-air-to-csl-vecadd-design.md §8.5
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "mlir/test/Conversion/AIRToCSLDialect/vecadd.mlir"


def test_vecadd_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # 1. compile MLIR through the full pipeline
    pipeline = (
        f"air-opt {SRC} -air-to-csl-dialect -csl-to-csl-rt | "
        f"air-translate --emit-csl-rt -o {out_dir}"
    )
    result = subprocess.run(
        pipeline, shell=True, capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"compile failed:\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    kernel = out_dir / "vecadd_pe.csl"
    runner = out_dir / "run.py"
    assert kernel.exists(), f"missing {kernel}"
    assert runner.exists(), f"missing {runner}"

    # 2. run on CS-3
    result = subprocess.run(
        ["cs_python", str(runner), "--arch", "wse3", "--check"],
        cwd=str(out_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )
    print("--- run.py stdout ---")
    print(result.stdout)
    print("--- run.py stderr ---")
    print(result.stderr)

    assert result.returncode == 0, "cs_python run.py exited non-zero"
    assert "PASS" in result.stdout, "validator did not print PASS"
```

- [ ] **Step 2: Run it**

Run: `cd /home/bricklib_dataflow/air-csl/mlir-air && pytest test/csl/test_vecadd_e2e.py -v`

Expected: **PASS**. This is the milestone success signal.

If it fails, the failure should fall into one of these categories:
- **Compile failure**: the `air-opt | air-translate` pipeline returned nonzero. Read stderr — the upstream pass wasn't producing what the downstream expected. Bug is in Phases 2-5.
- **Missing file**: the pipeline ran but didn't produce both `vecadd_pe.csl` and `run.py`. Bug is in `CSLRuntimeToPy.cpp`'s file-writing logic.
- **`cs_python` nonzero**: the script ran but failed. Read the captured stderr — usually `cslc` syntax error or `SdkRuntime.run()` exception. If `cslc` is unhappy, diff `out_dir/vecadd_pe.csl` against `golden/vecadd_pe.csl.golden`. If runtime is unhappy, the host script structure is wrong.
- **No PASS**: the script exited 0 but didn't print PASS. The validator block is missing or has a bug.

**Do not commit until this test passes.** It is the gate.

- [ ] **Step 3: Commit**

```bash
git add test/csl/test_vecadd_e2e.py
git commit -m "test(csl): end-to-end vecadd on CS-3

The milestone's definition of done. Compiles vecadd.mlir through
air-opt + air-translate, then runs the produced run.py on CS-3
and asserts the validator prints PASS."
```

---

## Phase 7 — Quarantine the Phase-1 emitter

**Goal:** Free the `-air-to-csl=...` pass option name and remove the dead-end direct text emitter.

### Task 7.1: Stop registering the old pass

**Files:**
- Modify: `mlir/include/air/Conversion/Passes.td`

- [ ] **Step 1: Remove the AIRToCSL def**

Open `mlir/include/air/Conversion/Passes.td` and **delete** the `def AIRToCSL : Pass<"air-to-csl", "ModuleOp"> { ... }` block (the one we left in place during Phase 2.1).

- [ ] **Step 2: Remove the source file from CMakeLists**

Open `mlir/lib/Conversion/CMakeLists.txt`. Find the line that lists `AIRToCSLPass.cpp` (likely as part of `add_mlir_library(AIRConversionPasses ... AIRToCSLPass.cpp ...)` or similar). Delete that line.

- [ ] **Step 3: Build**

Run: `cd build && ninja install 2>&1 | tail -20`

Expected: clean build. If there are linker errors complaining about `createAIRToCSLPass`, find the call sites (`grep -rn "createAIRToCSLPass\|AIRToCSL" mlir/lib mlir/include tools/`) and remove them.

- [ ] **Step 4: Run all tests**

Run: `cd build && ninja check-csl-all`

Expected: every CSL test passes. If any test was depending on the old `-air-to-csl=...` option, it must be migrated to `-air-to-csl-dialect` or removed (the Phase-1 tests at `mlir/test/Conversion/AIRToCSL/{basic,gemv}.mlir` should be deleted at this point — they were superseded by the new milestone tests).

- [ ] **Step 5: Run the e2e test once more**

Run: `pytest test/csl/test_vecadd_e2e.py -v`

Expected: PASS. Quarantining must not regress the milestone test.

- [ ] **Step 6: Commit**

```bash
git add mlir/include/air/Conversion/Passes.td \
        mlir/lib/Conversion/CMakeLists.txt
# Plus any test files migrated or removed:
git rm mlir/test/Conversion/AIRToCSL/basic.mlir mlir/test/Conversion/AIRToCSL/gemv.mlir
git commit -m "refactor(csl): unregister Phase-1 AIRToCSL pass

The -air-to-csl=... pass is superseded by -air-to-csl-dialect
followed by -csl-to-csl-rt and --emit-csl-rt. This commit removes
the pass registration and CMake source entry; the .cpp file moves
to archived_code/ in the next commit.

Old test fixtures basic.mlir and gemv.mlir are removed; the new
pipeline's test suite (vecadd.mlir + reject_*.mlir) supersedes them."
```

### Task 7.2: Move file to archived_code/

- [ ] **Step 1: Move the file**

```bash
git mv mlir/lib/Conversion/AIRToCSLPass.cpp archived_code/AIRToCSLPass.cpp
```

- [ ] **Step 2: Verify nothing references it**

```bash
grep -rn "AIRToCSLPass\.cpp\|createAIRToCSLPass\|AIRToCSL\b" \
  mlir/include mlir/lib tools/ 2>&1
```

Expected: no matches (or only matches inside `archived_code/`).

- [ ] **Step 3: Final build + test**

```bash
cd build && ninja install && ninja check-csl-all && \
  cd /home/bricklib_dataflow/air-csl/mlir-air && \
  pytest test/csl/test_vecadd_e2e.py -v
```

Expected: clean build, all CSL tests pass, e2e test passes.

- [ ] **Step 4: Commit**

```bash
git add archived_code/ mlir/lib/Conversion/AIRToCSLPass.cpp
git commit -m "chore(csl): archive Phase-1 AIRToCSLPass.cpp

The Phase-1 direct text emitter is now fully superseded by the
new pipeline. Moving the source to archived_code/ for historical
reference (the file's own header comment already labeled it
'kept for reference and later use').

Milestone complete. See docs/superpowers/specs/2026-04-13-
air-to-csl-vecadd-design.md and corresponding plan."
```

---

## Self-Review

After writing the plan, verifying against the spec:

**Spec coverage:**
- §2 Definition of done → Task 6.1 ✓
- §3 Scope (1×1, vecadd, no comms) → Tasks 2.3, 2.4-2.10, 3.3 ✓
- §4 Phase-1 disposition → Phase 7 ✓
- §5.1 Pipeline → Phases 2/3/4/5 ✓
- §5.2 Architectural commitments → enforced by IR contract test 2.2 ✓
- §5.3.1 IR contract → Task 2.2 (FileCheck), 2.3 (impl) ✓
- §5.3.2 csl-to-csl-rt extension → Phase 3 ✓
- §5.3.3 KernelEmitter + HostEmitter → Phases 4 and 5 ✓
- §6 Data flow → exercised by Phase 6 ✓
- §7.2 air-to-csl-dialect rejections → Tasks 2.4-2.10 ✓
- §7.3 csl-to-csl-rt rejections → Task 3.3 ✓
- §7.4 KernelEmitter rejections → covered by `validateKernel` in Task 4.2; one explicit failing fixture would strengthen this — added as a follow-up note below.
- §7.5 Runtime errors → covered by the validator in Task 5.2 ✓
- §8.2 Bootstrap golden files → Phase 0 ✓
- §8.3 Per-layer unit tests → Tasks 2.2, 3.1, 4.1, 5.1 ✓
- §8.4 Layer-rejection tests → Tasks 2.4-2.10, 3.3 ✓
- §8.5 Hardware integration test → Task 6.1 ✓
- §9.1 Milestone delivery → covered by all phases ✓

**Coverage gap found:** Spec §7.4 mentions KernelEmitter precondition violations as a category of rejection. The plan's `validateKernel` rejects them in code, but there's no explicit FileCheck rejection test for the case. Adding one strengthens the safety net. Inserting as Task 4.3.

### Task 4.3 (added during self-review): KernelEmitter rejection test

**Files:**
- Create: `mlir/test/Targets/CSLRuntimeToCSL/reject_for_nonzero_lo.mlir`

- [ ] **Step 1: Write the test**

```mlir
// RUN: not air-translate --emit-csl-rt %s 2>&1 | FileCheck %s

// CHECK: scf.for requires lo=0, step=1, constant hi

module {
  csl.spatial_placement {
    %k = csl.kernel {
      csl.var @x : memref<256xf32>
      csl.func @bad() : () -> () {
        %c1 = arith.constant 1 : index
        %c256 = arith.constant 256 : index
        %c1_ = arith.constant 1 : index
        scf.for %i = %c1 to %c256 step %c1_ {
        }
        csl.return
      }
    } {source_file = "bad.csl"} : !csl.kernel
    %r = csl.code_region routes() colors() {} {width = 1 : i64, height = 1 : i64} : !csl.code_region
    csl.place %r %k {x = 0 : i64, y = 0 : i64}
  }
}
```

- [ ] **Step 2: Run, verify it passes (expected error matches)**

Run: `cd build && lit -v ../mlir/test/Targets/CSLRuntimeToCSL/reject_for_nonzero_lo.mlir`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add mlir/test/Targets/CSLRuntimeToCSL/reject_for_nonzero_lo.mlir
git commit -m "test(csl-emit): pin KernelEmitter scf.for precondition rejection"
```

**Placeholder scan:** searched the plan for "TBD", "TODO", "implement later", "fill in" — none found.

**Type consistency:**
- `KernelEmitter` class name used consistently across spec and plan.
- `synthesizeRuntimeSequence` referenced once in Phase 3 — name matches the helper definition.
- `direction` attribute used identically across Phases 1, 2, 3, 5.
- `csl.export_name` op consistently uses `sym_name`, `exported_type`, optional `direction` per the TableGen def in Task 1.1.
- `vecadd_pe.csl` filename used consistently across golden file (Phase 0), kernel `source_file` attribute (Phase 2), and translator output (Phase 4).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-13-air-to-csl-vecadd.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration, isolation between tasks reduces context contamination.

**2. Inline Execution** — Execute tasks in this session using the executing-plans skill, batch execution with checkpoints for review.

**Which approach?**
