# CSL Multi-PE Streams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a 2-PE "ping" kernel that runs SUCCESS! on the WSE-3 simulator via a 3-op user-facing surface (`csl_layout.stream` + `csl.stream.put` + `csl.stream.get`) lowered through 4 single-purpose passes to emit-ready CSL.

**Architecture:** Progressive lowering. User writes typeless dataflow edges with no colors/tasks/async. Pass 1 synthesizes color symbols. Pass 2 allocates ids. Pass 3 emits per-PE `set_color_config`. Pass 4 expands put/get into fabric DSDs + tasks + async builtins. Stage-4 IR is isomorphic to CSL text; the emitter is a pure printer.

**Tech Stack:** MLIR (TableGen for op defs, C++ for verifiers/passes/emitters), `air-opt` driver, `air-translate --emit-csl`, lit + FileCheck for tests, `cs_python` + Cerebras simulator (WSE-3) for e2e.

**Spec:** [`docs/superpowers/specs/2026-05-01-csl-multi-pe-streams-design.md`](../specs/2026-05-01-csl-multi-pe-streams-design.md).

---

## Environment setup (run once at start of session)

```bash
cd /home/bricklib_dataflow/air-csl/mlir-air
module load python/3.10.3 cmake/3.26.0 rocm/6.2.1 singularity
source sandbox/bin/activate
source utils/env_setup_gpu.sh install llvm/install
```

This puts `air-opt`, `aircc.py`, `mlir-opt`, `lit`, and `FileCheck` on PATH.

**Build after each TableGen change:**
```bash
cd build && ninja install -j$(nproc)
```
Builds typically take 30–90 seconds incrementally. Full rebuilds take longer.

**Run a single lit test file:**
```bash
lit -v mlir/test/Dialect/CSL/<file>.mlir
```

**Run a build target (suite of lit tests):**
```bash
cd build && ninja check-air-mlir              # full suite
cd build && ninja check-airmlir-dialect-csl   # CSL roundtrip + verifier
cd build && ninja check-airmlir-conversion-airtocsl
cd build && ninja check-airmlir-targets-cslemit
```

---

## File structure

### New files

```
mlir/include/air/Dialect/CSL/Transforms/
  CSLMaterializeStreamColorsPass.h   (Pass 1 declaration)
  CSLAllocateColorIdsPass.h          (Pass 2 declaration)
  CSLLowerStreamRoutingPass.h        (Pass 3 declaration)
  CSLLowerStreamDataPass.h           (Pass 4 declaration)

mlir/include/air/Dialect/CSL/Pipelines/
  Pipelines.h                        (registerCSLPipelines decl)

mlir/lib/Dialect/CSL/Transforms/
  CSLMaterializeStreamColors.cpp     (Pass 1 impl)
  CSLAllocateColorIds.cpp            (Pass 2 impl)
  CSLLowerStreamRouting.cpp          (Pass 3 impl)
  CSLLowerStreamData.cpp             (Pass 4 impl)

mlir/lib/Dialect/CSL/Pipelines/
  CSLStreamsPipeline.cpp             (csl-streams-to-csl)
  CSLPipeline.cpp                    (csl-pipeline)
  CMakeLists.txt

mlir/test/Dialect/CSL/                              (existing dir)
  roundtrip_stream.mlir
  roundtrip_stream_with_color.mlir
  roundtrip_stream_put_get.mlir
  roundtrip_get_fab_dsd.mlir
  roundtrip_set_color_config.mlir
  roundtrip_task_local_id.mlir
  roundtrip_task_color.mlir
  roundtrip_builtin_async.mlir
  roundtrip_color_in_layout.mlir
  invalid_streams.mlir              (verifier negatives)
  invalid_set_color_config.mlir
  invalid_task_attrs.mlir
  invalid_builtin_async.mlir

mlir/test/Dialect/CSL/Transforms/                   (NEW dir, with CMakeLists.txt)
  materialize/one_stream.mlir
  materialize/two_streams.mlir
  materialize/idempotent.mlir
  allocate/one_color.mlir
  allocate/three_colors.mlir
  allocate/with_pin.mlir
  allocate/skips_pinned_id.mlir
  allocate/idempotent.mlir
  lower-routing/east.mlir
  lower-routing/west.mlir
  lower-routing/south.mlir
  lower-routing/north.mlir
  lower-routing/keeps_stream.mlir
  lower-data/put.mlir
  lower-data/get.mlir
  lower-data/erases_stream.mlir
  lower-data/task_naming.mlir
  lower-data/unique_task_ids.mlir

mlir/test/Dialect/CSL/Pipelines/                    (NEW dir, with CMakeLists.txt)
  streams_to_csl.mlir
  full.mlir

mlir/test/Targets/CSLEmit/multi_pe/                 (NEW dir, with CMakeLists.txt)
  emit_set_color_config.mlir
  emit_local_task.mlir
  emit_fabric_dsd.mlir
  emit_async_builtin.mlir
  emit_full_ping.mlir

mlir/test/Targets/CSLEmit/e2e/multi_pe/             (NEW dir, with CMakeLists.txt)
  ping_2pe.mlir                     (THE MILESTONE)
  ping_2pe_vertical.mlir
  ping_2pe_west.mlir
  ping_2pe_north.mlir
  ping_2pe_extents.mlir
  ping_2pe_two_streams.mlir
  ping_3pe_chain.mlir
```

### Modified files

```
mlir/include/air/Dialect/CSL/CSLOps.td             (csl.color, csl.task, csl.builtin_call, +csl.get_fab_dsd, +csl.stream.put, +csl.stream.get)
mlir/include/air/Dialect/CSL/CSLLayoutOps.td       (+csl_layout.stream, +csl_layout.set_color_config)
mlir/include/air/Dialect/CSL/Transforms/Passes.h   (include new pass headers)
mlir/lib/Dialect/CSL/IR/CSLOps.cpp                 (verifiers for new + edited ops)
mlir/lib/Dialect/CSL/IR/CSLLayoutOps.cpp           (verifiers for new ops)
mlir/lib/Dialect/CSL/Transforms/Passes.cpp         (register 4 new passes)
mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt     (4 new .cpp files)
mlir/lib/Dialect/CSL/CMakeLists.txt                (add Pipelines subdir)
mlir/lib/Targets/CSLEmit/*.cpp                     (emit cases for new ops/attrs)
tools/air-opt/air-opt.cpp                          (call registerCSLPipelines())
mlir/test/Dialect/CSL/CMakeLists.txt               (add Transforms + Pipelines subdirs)
mlir/test/Targets/CSLEmit/CMakeLists.txt           (add multi_pe + e2e/multi_pe subdirs)
utils/run_csl_ci.sh                                (add multi_pe e2e tests to suite)
```

---

## Phase A — Internal ops (lowering target)

These ops are *synthesized* by passes. They land first because Phase C (passes) emits them. Each can be unit-tested via dialect roundtrip + verifier independently of any pass.

---

### Task 1: Move `csl.color` scope from program-body to layout-body

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td:38-57` (add `HasParent` trait)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` (no behavioral change; verifier inherits from trait)
- Test: `mlir/test/Dialect/CSL/roundtrip_color_in_layout.mlir`
- Test: `mlir/test/Dialect/CSL/invalid_color_outside_layout.mlir` (negative)
- Update: any existing test that places `csl.color` inside `csl.program` (search first; per spec §3.4 there should be none used by working e2e)

- [ ] **Step 1: Search for existing uses of csl.color inside csl.program**

```bash
grep -rn "csl\.color" mlir/test/ | grep -v Targets/CSLEmit/e2e | head -20
```

If any tests place `csl.color` inside `csl.program`, list them — they'll be updated in Step 6 below.

- [ ] **Step 2: Write the positive roundtrip test**

Create `mlir/test/Dialect/CSL/roundtrip_color_in_layout.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @w
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  // CHECK: csl.layout
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @red : !csl.color
    csl.color @red : !csl.color
    // CHECK: csl.color @blue {id = 5 : i32} : !csl.color
    csl.color @blue {id = 5 : i32} : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

Note: `csl.color` currently has format `attr-dict ':' type($result)` and no symbol; we need to add a `sym_name` so colors can be referenced by symbol from other ops. See Step 4 for the full TableGen change.

- [ ] **Step 3: Run the test — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/roundtrip_color_in_layout.mlir
```

Expected: FAIL — current `csl.color` doesn't take a symbol name.

- [ ] **Step 4: Update `CSLOps.td` for `csl.color`**

Replace the existing definition at `mlir/include/air/Dialect/CSL/CSLOps.td:38-57` with:

```tablegen
def CSL_ColorOp : CSL_Op<"color", [
    Symbol,
    HasParent<"::xilinx::csl::LayoutOp">
  ]> {
  let summary = "Declare a communication color (lives in csl.layout body)";
  let description = [{
    Declares a CSL communication color. Colors are the fundamental routing
    primitive on the WSE fabric. A color symbol is shared across all PE
    programs that reference it (via csl.get_fab_dsd / csl_layout.set_color_config).

    The op is **synthesized** by --csl-materialize-stream-colors from
    csl_layout.stream ops; users do not write it directly.

    When `id` is absent, the color is "virtual" and the
    --csl-allocate-color-ids pass assigns one. When present, the color is
    pinned to that physical id and the allocator skips it.

    Examples:
    ```mlir
    csl.color @red : !csl.color                 // virtual
    csl.color @blue {id = 5 : i32} : !csl.color // pinned
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<I32Attr>:$id);
  let results = (outs CSL_ColorType:$result);

  let assemblyFormat = "$sym_name attr-dict `:` type($result)";
}
```

**Important:** `HasParent<"::xilinx::csl::LayoutOp">` requires `LayoutOp` to be the C++ class for `csl.layout`. If the existing class name is different (check `CSLOps.td` near line 444 and the `def CSL_LayoutOp` block), use that one.

- [ ] **Step 5: Build and re-run the test — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/roundtrip_color_in_layout.mlir
```

Expected: PASS. The op now parses with a symbol name and lives inside `csl.layout`.

- [ ] **Step 6: Update existing tests that use `csl.color` (if any)**

For each test found in Step 1, move the `csl.color` op from inside `csl.program` to inside `csl.layout` and add a symbol name. Re-run the test:

```bash
lit -v mlir/test/Dialect/CSL/<test>.mlir
```

Expected: PASS.

- [ ] **Step 7: Write the verifier negative test**

Create `mlir/test/Dialect/CSL/invalid_color_outside_layout.mlir`:

```mlir
// RUN: not air-opt %s 2>&1 | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    // CHECK: error: 'csl.color' op expects parent op 'csl.layout'
    csl.color @red : !csl.color
    csl.func @c { csl.return }
  }
}
```

- [ ] **Step 8: Run negative test — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/invalid_color_outside_layout.mlir
```

Expected: PASS (`not air-opt` succeeds because the verifier rejects the IR).

- [ ] **Step 9: Run the full CSL dialect suite to confirm no regression**

```bash
cd build && ninja check-airmlir-dialect-csl
```

Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td \
        mlir/test/Dialect/CSL/roundtrip_color_in_layout.mlir \
        mlir/test/Dialect/CSL/invalid_color_outside_layout.mlir \
        mlir/test/Dialect/CSL/<any-updated-existing-tests>
git commit -m "feat(csl): csl.color now lives in csl.layout body, has symbol name"
```

---

### Task 2: Add `csl_layout.set_color_config` op

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLLayoutOps.td` (append new op def)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLLayoutOps.cpp` (verifier checking color symbol resolution)
- Test: `mlir/test/Dialect/CSL/roundtrip_set_color_config.mlir`
- Test: `mlir/test/Dialect/CSL/invalid_set_color_config.mlir` (negative)

- [ ] **Step 1: Write the positive roundtrip test**

Create `mlir/test/Dialect/CSL/roundtrip_set_color_config.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @w
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @send : !csl.color
    // CHECK: csl_layout.set_color_config @send at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @send at(0, 0) rx(RAMP) tx(EAST)
    // CHECK: csl_layout.set_color_config @send at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.set_color_config @send at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/roundtrip_set_color_config.mlir
```

Expected: FAIL — `csl_layout.set_color_config` is not a known op.

- [ ] **Step 3: Add op to `CSLLayoutOps.td`**

Append to `mlir/include/air/Dialect/CSL/CSLLayoutOps.td` (before `#endif // CSL_LAYOUT_OPS`):

```tablegen
//===----------------------------------------------------------------------===//
// csl_layout.set_color_config — per-PE routing config for a color
//===----------------------------------------------------------------------===//

def CSLLayout_SetColorConfigOp : CSLLayout_Op<"set_color_config"> {
  let summary = "Set rx/tx routing for a color at one PE";
  let description = [{
    Configures the router on a single PE for a given color. Both `rx` and
    `tx` carry a single Direction (NORTH/SOUTH/EAST/WEST/RAMP). RAMP is the
    ingress/egress between the router and the PE's compute element.

    Synthesized by `--csl-lower-stream-routing` from `csl_layout.stream`.

    ```mlir
    csl_layout.set_color_config @send_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @send_color at(1, 0) rx(WEST) tx(RAMP)
    ```

    Emits to layout.csl as:
    ```csl
    @set_color_config(0, 0, send_color,
        .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$color,
    I64Attr:$px,
    I64Attr:$py,
    CSL_DirectionEnum:$rx,
    CSL_DirectionEnum:$tx
  );

  let assemblyFormat =
      "$color `at` `(` $px `,` $py `)` `rx` `(` $rx `)` `tx` `(` $tx `)` attr-dict";
  let hasVerifier = 1;
}
```

- [ ] **Step 4: Build and re-run roundtrip test — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/roundtrip_set_color_config.mlir
```

Expected: PASS.

- [ ] **Step 5: Write the verifier negative test**

Create `mlir/test/Dialect/CSL/invalid_set_color_config.mlir`:

```mlir
// RUN: not air-opt %s 2>&1 | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: error: 'csl_layout.set_color_config' op references undefined symbol '@nope'
    csl_layout.set_color_config @nope at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 6: Implement the verifier in `CSLLayoutOps.cpp`**

In `mlir/lib/Dialect/CSL/IR/CSLLayoutOps.cpp`, add:

```cpp
mlir::LogicalResult SetColorConfigOp::verify() {
  // Resolve the @color symbol to a csl.color in the parent csl.layout body.
  auto layout = (*this)->getParentOfType<::xilinx::csl::LayoutOp>();
  if (!layout)
    return emitOpError("must be inside csl.layout body");
  auto color = mlir::SymbolTable::lookupSymbolIn(
      layout, getColorAttr().getAttr());
  if (!color)
    return emitOpError("references undefined symbol '@")
           << getColorAttr().getValue() << "'";
  if (!mlir::isa<::xilinx::csl::ColorOp>(color))
    return emitOpError("'@") << getColorAttr().getValue()
                              << "' is not a csl.color";
  return mlir::success();
}
```

Add the corresponding `#include` for `air/Dialect/CSL/CSLOps.h` at the top of the file if not already present.

- [ ] **Step 7: Build and run negative test — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/invalid_set_color_config.mlir
```

Expected: PASS (verifier rejects undefined symbol).

- [ ] **Step 8: Run the full CSL dialect suite — confirm no regression**

```bash
cd build && ninja check-airmlir-dialect-csl
```

Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLLayoutOps.td \
        mlir/lib/Dialect/CSL/IR/CSLLayoutOps.cpp \
        mlir/test/Dialect/CSL/roundtrip_set_color_config.mlir \
        mlir/test/Dialect/CSL/invalid_set_color_config.mlir
git commit -m "feat(csl): csl_layout.set_color_config — per-PE routing config op"
```

---

### Task 3: Add `csl.get_fab_dsd` op

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td` (uncomment / add the deferred op slot at line 17)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` (verifier — color symbol resolution)
- Test: `mlir/test/Dialect/CSL/roundtrip_get_fab_dsd.mlir`
- Test: `mlir/test/Dialect/CSL/invalid_get_fab_dsd.mlir` (negative)

- [ ] **Step 1: Write the positive roundtrip test**

Create `mlir/test/Dialect/CSL/roundtrip_get_fab_dsd.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: csl.wafer @w
csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %c128 = arith.constant 128 : index
      // CHECK: csl.get_fab_dsd fabout @send extent(%c128
      %out = csl.get_fab_dsd fabout @send extent(%c128 : index) : !csl.dsd
      // CHECK: csl.get_fab_dsd fabin @send extent(%c128
      %in  = csl.get_fab_dsd fabin  @send extent(%c128 : index) : !csl.dsd
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @send : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/roundtrip_get_fab_dsd.mlir
```

Expected: FAIL.

- [ ] **Step 3: Add op to `CSLOps.td`**

Insert after the existing `CSL_GetMemDsdOp` (around line 287) in `mlir/include/air/Dialect/CSL/CSLOps.td`:

```tablegen
//===----------------------------------------------------------------------===//
// csl.get_fab_dsd — Fabric DSD on a color (fabin or fabout)
//===----------------------------------------------------------------------===//

def CSL_FabDsdDirectionAttr : I32EnumAttr<"FabDsdDirection",
    "fabric DSD direction (fabin = receive, fabout = send)",
    [
      I32EnumAttrCase<"fabin",  0>,
      I32EnumAttrCase<"fabout", 1>
    ]> {
  let cppNamespace = "::xilinx::csl";
}

def CSL_GetFabDsdOp : CSL_Op<"get_fab_dsd", []> {
  let summary = "Build a fabin/fabout DSD on a color";
  let description = [{
    Builds a CSL fabric DSD that sends (`fabout`) to or receives (`fabin`)
    from the fabric on the given color, with a runtime extent in elements.

    The color is referenced by symbol name (FlatSymbolRefAttr) — it must
    resolve to a `csl.color` op in the enclosing `csl.layout` body.

    Synthesized by `--csl-lower-stream-data` from `csl.stream.put`/`get`.

    ```mlir
    %n = arith.constant 128 : index
    %out = csl.get_fab_dsd fabout @send_color extent(%n : index) : !csl.dsd
    %in  = csl.get_fab_dsd fabin  @send_color extent(%n : index) : !csl.dsd
    ```

    Emits to CSL:
    ```csl
    const out_dsd = @get_dsd(fabout_dsd,
        .{ .extent = 128, .fabric_color = send_color });
    ```
  }];

  let arguments = (ins
    CSL_FabDsdDirectionAttr:$direction,
    FlatSymbolRefAttr:$color,
    Index:$extent
  );
  let results = (outs CSL_DsdType:$result);

  let assemblyFormat =
      "$direction $color `extent` `(` $extent `:` type($extent) `)` "
      "attr-dict `:` type($result)";
  let hasVerifier = 1;
}
```

- [ ] **Step 4: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/roundtrip_get_fab_dsd.mlir
```

Expected: PASS.

- [ ] **Step 5: Implement the verifier in `CSLOps.cpp`**

In `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`, add:

```cpp
mlir::LogicalResult GetFabDsdOp::verify() {
  // Walk up to the enclosing csl.wafer; csl.layout is its sibling of csl.program.
  auto wafer = (*this)->getParentOfType<WaferOp>();
  if (!wafer)
    return emitOpError("must be inside csl.wafer");

  // Find the csl.layout child by walking the wafer body once.
  LayoutOp layout;
  for (auto &op : wafer.getBody().front()) {
    if (auto l = mlir::dyn_cast<LayoutOp>(op)) {
      layout = l;
      break;
    }
  }
  if (!layout)
    return emitOpError("no csl.layout found in enclosing csl.wafer");

  auto color = mlir::SymbolTable::lookupSymbolIn(
      layout, getColorAttr().getAttr());
  if (!color)
    return emitOpError("references undefined color symbol '@")
           << getColorAttr().getValue() << "'";
  if (!mlir::isa<ColorOp>(color))
    return emitOpError("'@") << getColorAttr().getValue()
                              << "' is not a csl.color";
  return mlir::success();
}
```

- [ ] **Step 6: Write the verifier negative test**

Create `mlir/test/Dialect/CSL/invalid_get_fab_dsd.mlir`:

```mlir
// RUN: not air-opt %s 2>&1 | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    csl.func @c {
      %n = arith.constant 4 : index
      // CHECK: error: 'csl.get_fab_dsd' op references undefined color symbol '@nope'
      %d = csl.get_fab_dsd fabout @nope extent(%n : index) : !csl.dsd
      csl.return
    }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 7: Build and run negative test — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/invalid_get_fab_dsd.mlir
```

Expected: PASS.

- [ ] **Step 8: Run full CSL suite — confirm no regression**

```bash
cd build && ninja check-airmlir-dialect-csl
```

- [ ] **Step 9: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td \
        mlir/lib/Dialect/CSL/IR/CSLOps.cpp \
        mlir/test/Dialect/CSL/roundtrip_get_fab_dsd.mlir \
        mlir/test/Dialect/CSL/invalid_get_fab_dsd.mlir
git commit -m "feat(csl): csl.get_fab_dsd — fabric DSD op (fabin/fabout)"
```

---

### Task 4: Refactor `csl.task` to attribute-driven trigger

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td:123-149` (replace existing csl.task)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` (verifier: exactly one of id/color set)
- Test: `mlir/test/Dialect/CSL/roundtrip_task_local_id.mlir`
- Test: `mlir/test/Dialect/CSL/roundtrip_task_color.mlir`
- Test: `mlir/test/Dialect/CSL/invalid_task_attrs.mlir`
- Update: any existing test using the old `csl.task @t color(%c) { ... }` syntax

- [ ] **Step 1: Search existing usages of csl.task**

```bash
grep -rn "csl\.task" mlir/test/ | head -20
```

List any tests using the old `color(%c)` operand form — they'll be updated in Step 5.

- [ ] **Step 2: Write the local-task-id roundtrip test**

Create `mlir/test/Dialect/CSL/roundtrip_task_local_id.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: csl.task @exit_task {id = 8 : i32, trigger_kind = "local_task_id"}
    csl.task @exit_task {trigger_kind = "local_task_id", id = 8 : i32} {
      csl.return
    }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 3: Write the color-triggered roundtrip test**

Create `mlir/test/Dialect/CSL/roundtrip_task_color.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: csl.task @recv {color = @send, trigger_kind = "color"}
    csl.task @recv {trigger_kind = "color", color = @send} {
      csl.return
    }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl.color @send : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 4: Run both — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/roundtrip_task_local_id.mlir
lit -v mlir/test/Dialect/CSL/roundtrip_task_color.mlir
```

Expected: FAIL — current `csl.task` requires `color(%ssa)` operand form.

- [ ] **Step 5: Update existing csl.task tests**

For each old-syntax test found in Step 1 (e.g., `kernel_ops.mlir`, `routing_ops.mlir`), rewrite the task ops:

```mlir
// OLD:
%c = csl.color {id = 3 : i32} : !csl.color
csl.task @recv color(%c) { csl.return }

// NEW:
csl.color @c {id = 3 : i32} : !csl.color   // moved to csl.layout
csl.task @recv {trigger_kind = "color", color = @c} { csl.return }
```

(The old SSA-operand form requires `csl.color` inside the program; the new symbol-ref form references `csl.color` from `csl.layout`. Old tests likely need restructuring; do this minimally.)

- [ ] **Step 6: Replace `csl.task` definition in `CSLOps.td`**

Replace lines 123–149 in `mlir/include/air/Dialect/CSL/CSLOps.td` with:

```tablegen
def CSL_TaskOp : CSL_Op<"task", [
    IsolatedFromAbove,
    Symbol
  ]> {
  let summary = "PE task with attribute-driven trigger binding";
  let description = [{
    Defines an event-driven task on a PE. The trigger binding is expressed
    via attributes:

    - `trigger_kind = "local_task_id"`: must also set `id` (i32). The
      task fires when `@get_local_task_id(N)` is activated. Used for
      send/recv completion.
    - `trigger_kind = "color"`: must also set `color` (FlatSymbolRefAttr
      to a csl.color). The task fires when a wavelet arrives on that color.

    The emitter generates the corresponding CSL triplet (id const, task
    body, comptime bind) automatically.

    Examples:
    ```mlir
    csl.task @exit_task {trigger_kind = "local_task_id", id = 8 : i32} {
      csl.return
    }
    csl.task @recv {trigger_kind = "color", color = @send_color} {
      csl.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$trigger_kind,
    OptionalAttr<I32Attr>:$id,
    OptionalAttr<FlatSymbolRefAttr>:$color
  );
  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}
```

- [ ] **Step 7: Implement the verifier in `CSLOps.cpp`**

Add to `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`:

```cpp
mlir::LogicalResult TaskOp::verify() {
  auto kind = getTriggerKind();
  bool hasId = (bool)getIdAttr();
  bool hasColor = (bool)getColorAttr();

  if (kind == "local_task_id") {
    if (!hasId)
      return emitOpError("trigger_kind = \"local_task_id\" requires `id` attribute");
    if (hasColor)
      return emitOpError("trigger_kind = \"local_task_id\" must not set `color`");
  } else if (kind == "color") {
    if (!hasColor)
      return emitOpError("trigger_kind = \"color\" requires `color` attribute");
    if (hasId)
      return emitOpError("trigger_kind = \"color\" must not set `id`");
  } else {
    return emitOpError("trigger_kind must be \"local_task_id\" or \"color\"")
           << " (got \"" << kind << "\")";
  }
  return mlir::success();
}
```

- [ ] **Step 8: Build and run roundtrip tests — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/roundtrip_task_local_id.mlir && \
  lit -v mlir/test/Dialect/CSL/roundtrip_task_color.mlir
```

Expected: both PASS.

- [ ] **Step 9: Write the verifier negative test**

Create `mlir/test/Dialect/CSL/invalid_task_attrs.mlir`:

```mlir
// RUN: not air-opt -split-input-file %s 2>&1 | FileCheck %s

csl.wafer @w1 {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: error: 'csl.task' op trigger_kind = "local_task_id" requires `id` attribute
    csl.task @t {trigger_kind = "local_task_id"} { csl.return }
  }
  csl.layout {width = 1, height = 1} @layout { csl_layout.place @p at (0,0) }
}

// -----

csl.wafer @w2 {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: error: 'csl.task' op trigger_kind = "color" must not set `id`
    csl.task @t {trigger_kind = "color", color = @x, id = 5 : i32} { csl.return }
  }
  csl.layout {width = 1, height = 1} @layout {
    csl.color @x : !csl.color
    csl_layout.place @p at (0,0)
  }
}

// -----

csl.wafer @w3 {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    // CHECK: error: 'csl.task' op trigger_kind must be "local_task_id" or "color"
    csl.task @t {trigger_kind = "wavelet"} { csl.return }
  }
  csl.layout {width = 1, height = 1} @layout { csl_layout.place @p at (0,0) }
}
```

- [ ] **Step 10: Run negative test — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/invalid_task_attrs.mlir
```

Expected: PASS.

- [ ] **Step 11: Run full suite — confirm no regression**

```bash
cd build && ninja check-airmlir-dialect-csl
```

Expected: all green.

- [ ] **Step 12: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td \
        mlir/lib/Dialect/CSL/IR/CSLOps.cpp \
        mlir/test/Dialect/CSL/roundtrip_task_local_id.mlir \
        mlir/test/Dialect/CSL/roundtrip_task_color.mlir \
        mlir/test/Dialect/CSL/invalid_task_attrs.mlir \
        mlir/test/Dialect/CSL/<any-updated-existing>
git commit -m "feat(csl): csl.task is attribute-driven (trigger_kind + id/color)"
```

---

### Task 5: Add `async` and `activate` attrs to `csl.builtin_call`

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td:293-329` (extend csl.builtin_call)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` (verifier: activate ⇒ async)
- Test: `mlir/test/Dialect/CSL/roundtrip_builtin_async.mlir`
- Test: `mlir/test/Dialect/CSL/invalid_builtin_async.mlir` (negative)

- [ ] **Step 1: Write the positive roundtrip test**

Create `mlir/test/Dialect/CSL/roundtrip_builtin_async.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    %a = csl.var @a : memref<128xf32>
    csl.func @c {
      %ad = csl.get_mem_dsd %a : memref<128xf32> -> !csl.dsd
      %n = arith.constant 128 : index
      %fd = csl.get_fab_dsd fabout @send extent(%n : index) : !csl.dsd
      // CHECK: csl.builtin_call "fmovs"
      // CHECK-SAME: {activate = @done, async}
      csl.builtin_call "fmovs"(%fd, %ad)
        {async, activate = @done}
        : (!csl.dsd, !csl.dsd) -> ()
      csl.return
    }
    csl.task @done {trigger_kind = "local_task_id", id = 8 : i32} { csl.return }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl.color @send : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/roundtrip_builtin_async.mlir
```

Expected: FAIL — `async` and `activate` are unknown attrs to `csl.builtin_call`.

- [ ] **Step 3: Extend `CSL_BuiltinCallOp` in `CSLOps.td`**

Replace the `arguments` block at line ~319 in `mlir/include/air/Dialect/CSL/CSLOps.td` with:

```tablegen
  let arguments = (ins
    StrAttr:$callee,
    Optional<CSL_ImportedModuleType>:$module,
    Variadic<AnyType>:$args,
    UnitAttr:$async,
    OptionalAttr<FlatSymbolRefAttr>:$activate);
```

The `assemblyFormat` already uses `attr-dict`, so the new attrs round-trip in the dictionary automatically. No format change needed.

- [ ] **Step 4: Build and re-run roundtrip — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/roundtrip_builtin_async.mlir
```

Expected: PASS.

- [ ] **Step 5: Implement the verifier in `CSLOps.cpp`**

Add to `BuiltinCallOp::verify()` (create the verify method if it doesn't exist; declare `let hasVerifier = 1;` in the TableGen):

In `mlir/include/air/Dialect/CSL/CSLOps.td`, add `let hasVerifier = 1;` to `CSL_BuiltinCallOp`.

In `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`:

```cpp
mlir::LogicalResult BuiltinCallOp::verify() {
  if (getActivateAttr() && !getAsync())
    return emitOpError("'activate' requires 'async'");
  return mlir::success();
}
```

- [ ] **Step 6: Write the verifier negative test**

Create `mlir/test/Dialect/CSL/invalid_builtin_async.mlir`:

```mlir
// RUN: not air-opt %s 2>&1 | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    %a = csl.var @a : memref<4xf32>
    csl.func @c {
      %ad = csl.get_mem_dsd %a : memref<4xf32> -> !csl.dsd
      // CHECK: error: 'csl.builtin_call' op 'activate' requires 'async'
      csl.builtin_call "fmovs"(%ad, %ad) {activate = @done}
        : (!csl.dsd, !csl.dsd) -> ()
      csl.return
    }
    csl.task @done {trigger_kind = "local_task_id", id = 1 : i32} { csl.return }
  }
  csl.layout {width = 1, height = 1} @layout { csl_layout.place @p at (0,0) }
}
```

- [ ] **Step 7: Build and run negative test — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/invalid_builtin_async.mlir
```

Expected: PASS.

- [ ] **Step 8: Run full suite — confirm no regression**

```bash
cd build && ninja check-airmlir-dialect-csl
```

- [ ] **Step 9: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td \
        mlir/lib/Dialect/CSL/IR/CSLOps.cpp \
        mlir/test/Dialect/CSL/roundtrip_builtin_async.mlir \
        mlir/test/Dialect/CSL/invalid_builtin_async.mlir
git commit -m "feat(csl): csl.builtin_call gets optional async + activate attrs"
```

---

## Phase B — User-facing ops

These are the three ops a kernel author writes. They have minimal verification (since correctness is enforced by the lowering pipeline downstream) but precise shape constraints.

---

### Task 6: Add `csl_layout.stream` op

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLLayoutOps.td` (append op)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLLayoutOps.cpp` (verifier — cardinal coords, in-bounds)
- Test: `mlir/test/Dialect/CSL/roundtrip_stream.mlir`
- Test: `mlir/test/Dialect/CSL/roundtrip_stream_with_color.mlir`
- Test: `mlir/test/Dialect/CSL/invalid_streams.mlir`

- [ ] **Step 1: Write the bare-form roundtrip test**

Create `mlir/test/Dialect/CSL/roundtrip_stream.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: csl_layout.stream @send_ch from(0, 0) to(1, 0)
    csl_layout.stream @send_ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
```

- [ ] **Step 2: Write the with-color roundtrip test**

Create `mlir/test/Dialect/CSL/roundtrip_stream_with_color.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @c0 : !csl.color
    // CHECK: csl_layout.stream @send_ch from(0, 0) to(1, 0) {color = @c0}
    csl_layout.stream @send_ch from(0, 0) to(1, 0) {color = @c0}
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
```

- [ ] **Step 3: Run — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/roundtrip_stream.mlir
lit -v mlir/test/Dialect/CSL/roundtrip_stream_with_color.mlir
```

Expected: FAIL.

- [ ] **Step 4: Add op to `CSLLayoutOps.td`**

Append to `mlir/include/air/Dialect/CSL/CSLLayoutOps.td` (before `#endif`):

```tablegen
//===----------------------------------------------------------------------===//
// csl_layout.stream — declare an inter-PE dataflow edge
//===----------------------------------------------------------------------===//

def CSLLayout_StreamOp : CSLLayout_Op<"stream", [Symbol]> {
  let summary = "Declare a single-hop dataflow edge between two PEs";
  let description = [{
    Declares a typeless point-to-point dataflow edge from PE `(from_x, from_y)`
    to PE `(to_x, to_y)`. The endpoint coords must differ by exactly one along
    a single cardinal axis (single-hop only this milestone).

    The optional `color` attribute references a `csl.color` symbol declared
    in the same `csl.layout` body. Users do not write the `color` attribute
    directly — `--csl-materialize-stream-colors` synthesizes one per stream.

    ```mlir
    // user-facing form (no color)
    csl_layout.stream @send_ch from(0, 0) to(1, 0)

    // post-Pass-1 form (color materialized)
    csl_layout.stream @send_ch from(0, 0) to(1, 0) {color = @send_ch_color}
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$from_x,
    I64Attr:$from_y,
    I64Attr:$to_x,
    I64Attr:$to_y,
    OptionalAttr<FlatSymbolRefAttr>:$color
  );

  let assemblyFormat =
      "$sym_name `from` `(` $from_x `,` $from_y `)` "
      "`to` `(` $to_x `,` $to_y `)` attr-dict";
  let hasVerifier = 1;
}
```

- [ ] **Step 5: Implement the verifier in `CSLLayoutOps.cpp`**

```cpp
mlir::LogicalResult StreamOp::verify() {
  int64_t dx = getToX() - getFromX();
  int64_t dy = getToY() - getFromY();
  // Single-hop, cardinal: exactly one of |dx|, |dy| is 1, the other is 0.
  bool xOne = (dx == 1 || dx == -1);
  bool yOne = (dy == 1 || dy == -1);
  bool xZero = (dx == 0);
  bool yZero = (dy == 0);
  if (!((xOne && yZero) || (xZero && yOne)))
    return emitOpError("requires single-hop cardinal route; got delta (")
           << dx << ", " << dy << ")";

  // Optional color must resolve to a csl.color in parent csl.layout.
  if (auto colorAttr = getColorAttr()) {
    auto layout = (*this)->getParentOfType<::xilinx::csl::LayoutOp>();
    if (!layout)
      return emitOpError("must be inside csl.layout body");
    auto color = mlir::SymbolTable::lookupSymbolIn(layout, colorAttr.getAttr());
    if (!color)
      return emitOpError("references undefined color '@")
             << colorAttr.getValue() << "'";
    if (!mlir::isa<::xilinx::csl::ColorOp>(color))
      return emitOpError("'@") << colorAttr.getValue() << "' is not a csl.color";
  }
  return mlir::success();
}
```

- [ ] **Step 6: Build and re-run both roundtrip tests — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/roundtrip_stream.mlir && \
  lit -v mlir/test/Dialect/CSL/roundtrip_stream_with_color.mlir
```

- [ ] **Step 7: Write the verifier negative test**

Create `mlir/test/Dialect/CSL/invalid_streams.mlir`:

```mlir
// RUN: not air-opt -split-input-file %s 2>&1 | FileCheck %s

csl.wafer @w1 {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 4 : i64, height = 4 : i64} @layout {
    // CHECK: error: 'csl_layout.stream' op requires single-hop cardinal route; got delta (2, 1)
    csl_layout.stream @bad from(0, 0) to(2, 1)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w2 {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: error: 'csl_layout.stream' op references undefined color '@nope'
    csl_layout.stream @bad from(0, 0) to(1, 0) {color = @nope}
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 8: Run negative test — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/invalid_streams.mlir
```

- [ ] **Step 9: Run full CSL suite**

```bash
cd build && ninja check-airmlir-dialect-csl
```

- [ ] **Step 10: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLLayoutOps.td \
        mlir/lib/Dialect/CSL/IR/CSLLayoutOps.cpp \
        mlir/test/Dialect/CSL/roundtrip_stream.mlir \
        mlir/test/Dialect/CSL/roundtrip_stream_with_color.mlir \
        mlir/test/Dialect/CSL/invalid_streams.mlir
git commit -m "feat(csl): csl_layout.stream — user-facing dataflow edge op"
```

---

### Task 7: Add `csl.stream.put` op

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td` (append op)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` (verifier — stream resolves; element type is f32)
- Test: `mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir` (positive — covers both put and get; get added in Task 8)
- Test: extend `invalid_streams.mlir` with put-related negatives (or new file)

- [ ] **Step 1: Write the positive roundtrip test (put only first; get added next task)**

Create `mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir`:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: csl.stream.put @ch source(%{{.*}}) extent(%{{.*}}) : memref<128xf32>
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @left at (0, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir
```

Expected: FAIL.

- [ ] **Step 3: Add `csl.stream.put` op to `CSLOps.td`**

Append to `CSLOps.td` (after the existing data-movement ops):

```tablegen
//===----------------------------------------------------------------------===//
// csl.stream.put — Send a buffer through a declared stream
//===----------------------------------------------------------------------===//

def CSL_StreamPutOp : CSL_Op<"stream.put", []> {
  let summary = "Send a memref buffer through an inter-PE stream";
  let description = [{
    Sends `extent` elements from `source` through the named stream. The
    stream symbol must resolve to a `csl_layout.stream` whose `from` coord
    matches the parent program's placement.

    Sync-looking at the user level; `--csl-lower-stream-data` expands this
    to fabric DSD + async builtin + completion task.

    ```mlir
    %n = arith.constant 128 : index
    csl.stream.put @send_ch source(%buf) extent(%n : index)
                   : memref<128xf32>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$stream,
    AnyMemRef:$source,
    Index:$extent
  );

  let assemblyFormat =
      "$stream `source` `(` $source `)` `extent` `(` $extent `:` "
      "type($extent) `)` `:` type($source) attr-dict";
  let hasVerifier = 1;
}
```

- [ ] **Step 4: Implement the verifier**

In `mlir/lib/Dialect/CSL/IR/CSLOps.cpp`:

```cpp
mlir::LogicalResult StreamPutOp::verify() {
  // Source memref element type must be f32 this milestone.
  auto memTy = mlir::cast<mlir::MemRefType>(getSource().getType());
  if (!memTy.getElementType().isF32())
    return emitOpError("source memref element type must be f32 (got ")
           << memTy.getElementType() << ")";

  // Stream symbol must exist in enclosing csl.wafer's csl.layout.
  auto wafer = (*this)->getParentOfType<WaferOp>();
  if (!wafer)
    return emitOpError("must be inside csl.wafer");
  ::xilinx::csl_layout::StreamOp stream;
  for (auto &op : wafer.getBody().front()) {
    if (auto layout = mlir::dyn_cast<LayoutOp>(op)) {
      if (auto found = mlir::dyn_cast_or_null<::xilinx::csl_layout::StreamOp>(
            mlir::SymbolTable::lookupSymbolIn(layout, getStreamAttr().getAttr())))
        stream = found;
      break;
    }
  }
  if (!stream)
    return emitOpError("references undefined stream '@")
           << getStreamAttr().getValue() << "'";
  return mlir::success();
}
```

Add `#include "air/Dialect/CSL/CSLLayoutOps.h"` if not already present.

- [ ] **Step 5: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir
```

- [ ] **Step 6: Add put-related negative tests**

Append to `mlir/test/Dialect/CSL/invalid_streams.mlir`:

```mlir
// -----

csl.wafer @w3 {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xi32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: error: 'csl.stream.put' op source memref element type must be f32
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<128xi32>
      csl.return
    }
  }
  csl.layout {width = 2, height = 1} @layout {
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
  }
}

// -----

csl.wafer @w4 {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: error: 'csl.stream.put' op references undefined stream '@nope'
      csl.stream.put @nope source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.layout {width = 1, height = 1} @layout { csl_layout.place @p at (0,0) }
}
```

- [ ] **Step 7: Run negatives — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/invalid_streams.mlir
```

- [ ] **Step 8: Run full suite**

```bash
cd build && ninja check-airmlir-dialect-csl
```

- [ ] **Step 9: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td \
        mlir/lib/Dialect/CSL/IR/CSLOps.cpp \
        mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir \
        mlir/test/Dialect/CSL/invalid_streams.mlir
git commit -m "feat(csl): csl.stream.put — send buffer through inter-PE stream"
```

---

### Task 8: Add `csl.stream.get` op

**Files:**
- Modify: `mlir/include/air/Dialect/CSL/CSLOps.td` (append, mirror of put)
- Modify: `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` (verifier mirror)
- Test: extend `roundtrip_stream_put_get.mlir`
- Test: extend `invalid_streams.mlir`

- [ ] **Step 1: Extend the positive roundtrip test**

Append to `mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir` (inside the existing `csl.wafer`):

Replace the file content with:

```mlir
// RUN: air-opt %s | air-opt | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: csl.stream.put @ch source(%{{.*}}) extent(%{{.*}}) : memref<128xf32>
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.program @right {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: csl.stream.get @ch target(%{{.*}}) extent(%{{.*}}) : memref<128xf32>
      csl.stream.get @ch target(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @left  at (0, 0)
    csl_layout.place @right at (1, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir
```

Expected: FAIL — `csl.stream.get` is unknown.

- [ ] **Step 3: Add `csl.stream.get` op to `CSLOps.td`**

Append to `CSLOps.td`:

```tablegen
//===----------------------------------------------------------------------===//
// csl.stream.get — Receive a buffer through a declared stream
//===----------------------------------------------------------------------===//

def CSL_StreamGetOp : CSL_Op<"stream.get", []> {
  let summary = "Receive into a memref buffer through an inter-PE stream";
  let description = [{
    Receives `extent` elements through the named stream into `target`. The
    stream symbol must resolve to a `csl_layout.stream` whose `to` coord
    matches the parent program's placement.

    Sync-looking at the user level; `--csl-lower-stream-data` expands this
    to fabric DSD + async builtin + completion task.

    ```mlir
    %n = arith.constant 128 : index
    csl.stream.get @send_ch target(%buf) extent(%n : index)
                   : memref<128xf32>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$stream,
    AnyMemRef:$target,
    Index:$extent
  );

  let assemblyFormat =
      "$stream `target` `(` $target `)` `extent` `(` $extent `:` "
      "type($extent) `)` `:` type($target) attr-dict";
  let hasVerifier = 1;
}
```

- [ ] **Step 4: Implement the verifier in `CSLOps.cpp`**

```cpp
mlir::LogicalResult StreamGetOp::verify() {
  auto memTy = mlir::cast<mlir::MemRefType>(getTarget().getType());
  if (!memTy.getElementType().isF32())
    return emitOpError("target memref element type must be f32 (got ")
           << memTy.getElementType() << ")";

  auto wafer = (*this)->getParentOfType<WaferOp>();
  if (!wafer)
    return emitOpError("must be inside csl.wafer");
  ::xilinx::csl_layout::StreamOp stream;
  for (auto &op : wafer.getBody().front()) {
    if (auto layout = mlir::dyn_cast<LayoutOp>(op)) {
      if (auto found = mlir::dyn_cast_or_null<::xilinx::csl_layout::StreamOp>(
            mlir::SymbolTable::lookupSymbolIn(layout, getStreamAttr().getAttr())))
        stream = found;
      break;
    }
  }
  if (!stream)
    return emitOpError("references undefined stream '@")
           << getStreamAttr().getValue() << "'";
  return mlir::success();
}
```

- [ ] **Step 5: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir
```

- [ ] **Step 6: Add get-related negatives to `invalid_streams.mlir`**

Append:

```mlir
// -----

csl.wafer @w5 {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xf16>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: error: 'csl.stream.get' op target memref element type must be f32
      csl.stream.get @ch target(%buf) extent(%n : index) : memref<128xf16>
      csl.return
    }
  }
  csl.layout {width = 2, height = 1} @layout {
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @p at (1, 0)
  }
}
```

- [ ] **Step 7: Run negatives — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/invalid_streams.mlir
```

- [ ] **Step 8: Run full suite**

```bash
cd build && ninja check-airmlir-dialect-csl
```

- [ ] **Step 9: Commit**

```bash
git add mlir/include/air/Dialect/CSL/CSLOps.td \
        mlir/lib/Dialect/CSL/IR/CSLOps.cpp \
        mlir/test/Dialect/CSL/roundtrip_stream_put_get.mlir \
        mlir/test/Dialect/CSL/invalid_streams.mlir
git commit -m "feat(csl): csl.stream.get — receive buffer through inter-PE stream"
```

---

## Phase C — Passes

Each pass has a strict pre/post invariant (spec §3.10) and is FileCheck-tested in isolation against a `.mlir` that exhibits its specific transformation.

---

### Task 9: Pass 1 — `--csl-materialize-stream-colors`

**Files:**
- Create: `mlir/include/air/Dialect/CSL/Transforms/CSLMaterializeStreamColorsPass.h`
- Create: `mlir/lib/Dialect/CSL/Transforms/CSLMaterializeStreamColors.cpp`
- Modify: `mlir/include/air/Dialect/CSL/Transforms/Passes.h` (include new header)
- Modify: `mlir/lib/Dialect/CSL/Transforms/Passes.cpp` (register)
- Modify: `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt` (add .cpp)
- Tests: `mlir/test/Dialect/CSL/Transforms/materialize/{one_stream,two_streams,idempotent}.mlir`
- Modify: `mlir/test/Dialect/CSL/CMakeLists.txt` (add Transforms subdir if not yet)
- Create: `mlir/test/Dialect/CSL/Transforms/CMakeLists.txt`

- [ ] **Step 1: Create test directory's CMakeLists.txt and lit config**

Create `mlir/test/Dialect/CSL/Transforms/lit.local.cfg`:

```python
config.suffixes = ['.mlir']
```

If `mlir/test/Dialect/CSL/CMakeLists.txt` doesn't already include subdirectories, append `add_subdirectory(Transforms)` and create `mlir/test/Dialect/CSL/Transforms/CMakeLists.txt` with:

```cmake
add_lit_testsuite(check-airmlir-csl-transforms
  "Run CSL Transforms lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS air-opt FileCheck count not)
```

(If a parent suite already discovers subdirs, the per-subdir CMakeLists may be empty.)

- [ ] **Step 2: Write the one-stream FileCheck test**

Create `mlir/test/Dialect/CSL/Transforms/materialize/one_stream.mlir`:

```mlir
// RUN: air-opt --csl-materialize-stream-colors %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @send_ch_color : !csl.color
    // CHECK: csl_layout.stream @send_ch from(0, 0) to(1, 0) {color = @send_ch_color}
    csl_layout.stream @send_ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
```

- [ ] **Step 3: Run — expect FAIL (pass not registered)**

```bash
lit -v mlir/test/Dialect/CSL/Transforms/materialize/one_stream.mlir
```

Expected: FAIL — `--csl-materialize-stream-colors` is not a known flag.

- [ ] **Step 4: Create the pass header**

Create `mlir/include/air/Dialect/CSL/Transforms/CSLMaterializeStreamColorsPass.h`:

```cpp
//===- CSLMaterializeStreamColorsPass.h ---------------------*- C++ -*-===//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_MATERIALIZE_STREAM_COLORS_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_MATERIALIZE_STREAM_COLORS_PASS_H

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createCSLMaterializeStreamColorsPass();

} // namespace air
} // namespace xilinx

#endif
```

- [ ] **Step 5: Create the pass implementation**

Create `mlir/lib/Dialect/CSL/Transforms/CSLMaterializeStreamColors.cpp`:

```cpp
//===- CSLMaterializeStreamColors.cpp ---------------------*- C++ -*-===//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//
//
// Pass 1 of csl-streams-to-csl pipeline.
// Pre:  every csl_layout.stream has no `color` attr.
// Post: every csl_layout.stream has {color = @<sym>}; matching csl.color
//       @<sym> exists in same csl.layout body (no id yet).
//
// Naming: each stream @S gets a color symbol named "@S_color".
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLMaterializeStreamColorsPass.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx::csl;
using namespace xilinx::csl_layout;

namespace {

class CSLMaterializeStreamColorsPass
    : public PassWrapper<CSLMaterializeStreamColorsPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-materialize-stream-colors"; }
  StringRef getDescription() const final {
    return "Pass 1: synthesize csl.color symbols for csl_layout.stream ops";
  }
  void runOnOperation() override;
};

} // namespace

void CSLMaterializeStreamColorsPass::runOnOperation() {
  Operation *op = getOperation();
  op->walk([&](LayoutOp layout) {
    OpBuilder b(&layout.getBody().front(), layout.getBody().front().begin());
    for (auto &nested : llvm::make_early_inc_range(layout.getBody().front())) {
      auto stream = dyn_cast<StreamOp>(&nested);
      if (!stream)
        continue;
      if (stream.getColorAttr())
        continue;  // idempotent
      // Synthesize csl.color @<stream>_color : !csl.color (no id).
      std::string colorName = (stream.getSymName() + "_color").str();
      auto colorTy = ColorType::get(stream.getContext());
      b.setInsertionPoint(stream);
      auto colorOp = b.create<ColorOp>(
          stream.getLoc(),
          /*sym_name=*/b.getStringAttr(colorName),
          /*id=*/IntegerAttr(),
          /*result=*/colorTy);
      // Set {color = @<colorName>} on the stream.
      stream.setColorAttr(FlatSymbolRefAttr::get(b.getContext(), colorName));
      (void)colorOp;
    }
  });
}

std::unique_ptr<Pass> xilinx::air::createCSLMaterializeStreamColorsPass() {
  return std::make_unique<CSLMaterializeStreamColorsPass>();
}
```

Note: the exact `ColorOp::build()` signature depends on TableGen's generated builder. After Task 1's TableGen changes, the constructor takes (sym_name, optional id, result type) — adjust the call to match the generated builder. Check `build/include/air/Dialect/CSL/CSLOps.h.inc` after build to see the signature.

- [ ] **Step 6: Update `CMakeLists.txt`**

Edit `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt` — append `CSLMaterializeStreamColors.cpp` to the source list:

```cmake
add_mlir_library(
  CSLTransforms
  CSLInferExports.cpp
  CSLAutoVectorize.cpp
  CSLMaterializeStreamColors.cpp     # NEW
  LoopIdiomAnalysis.cpp
  Passes.cpp
  ...
```

- [ ] **Step 7: Update `Passes.h` and `Passes.cpp`**

In `mlir/include/air/Dialect/CSL/Transforms/Passes.h`, add:
```cpp
#include "air/Dialect/CSL/Transforms/CSLMaterializeStreamColorsPass.h"
```

In `mlir/lib/Dialect/CSL/Transforms/Passes.cpp`, add a registration block to `registerCSLTransformPasses()`:
```cpp
mlir::registerPass(
    []() -> std::unique_ptr<mlir::Pass> {
      return createCSLMaterializeStreamColorsPass();
    });
```

- [ ] **Step 8: Build and re-run the one-stream test — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/Transforms/materialize/one_stream.mlir
```

- [ ] **Step 9: Add the two-streams test**

Create `mlir/test/Dialect/CSL/Transforms/materialize/two_streams.mlir`:

```mlir
// RUN: air-opt --csl-materialize-stream-colors %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 3 : i64, height = 1 : i64} @layout {
    // CHECK-DAG: csl.color @ab_color : !csl.color
    // CHECK-DAG: csl.color @bc_color : !csl.color
    // CHECK: csl_layout.stream @ab from(0, 0) to(1, 0) {color = @ab_color}
    csl_layout.stream @ab from(0, 0) to(1, 0)
    // CHECK: csl_layout.stream @bc from(1, 0) to(2, 0) {color = @bc_color}
    csl_layout.stream @bc from(1, 0) to(2, 0)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
    csl_layout.place @p at (2, 0)
  }
}
```

- [ ] **Step 10: Add the idempotent test**

Create `mlir/test/Dialect/CSL/Transforms/materialize/idempotent.mlir`:

```mlir
// RUN: air-opt --csl-materialize-stream-colors --csl-materialize-stream-colors %s | FileCheck %s
// Running the pass twice is a no-op (no duplicate colors, no errors).

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @ch_color : !csl.color
    // CHECK-NOT: csl.color
    // CHECK: csl_layout.stream @ch from(0, 0) to(1, 0) {color = @ch_color}
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 11: Run all materialize tests — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/Transforms/materialize/
```

- [ ] **Step 12: Run full suite to confirm no regression**

```bash
cd build && ninja check-air-mlir
```

- [ ] **Step 13: Commit**

```bash
git add mlir/include/air/Dialect/CSL/Transforms/CSLMaterializeStreamColorsPass.h \
        mlir/lib/Dialect/CSL/Transforms/CSLMaterializeStreamColors.cpp \
        mlir/include/air/Dialect/CSL/Transforms/Passes.h \
        mlir/lib/Dialect/CSL/Transforms/Passes.cpp \
        mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt \
        mlir/test/Dialect/CSL/Transforms/CMakeLists.txt \
        mlir/test/Dialect/CSL/Transforms/lit.local.cfg \
        mlir/test/Dialect/CSL/Transforms/materialize/
git commit -m "feat(csl-pass): csl-materialize-stream-colors — synthesize color sym per stream"
```

---

### Task 10: Pass 2 — `--csl-allocate-color-ids`

**Files:** mirror Task 9's structure (header, impl, register, test dir).
- Create: `mlir/include/air/Dialect/CSL/Transforms/CSLAllocateColorIdsPass.h`
- Create: `mlir/lib/Dialect/CSL/Transforms/CSLAllocateColorIds.cpp`
- Tests: `mlir/test/Dialect/CSL/Transforms/allocate/{one_color,three_colors,with_pin,skips_pinned_id,idempotent}.mlir`

- [ ] **Step 1: Write `one_color.mlir`**

Create `mlir/test/Dialect/CSL/Transforms/allocate/one_color.mlir`:

```mlir
// RUN: air-opt --csl-allocate-color-ids %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @c0 {id = 0 : i32} : !csl.color
    csl.color @c0 : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL (pass not yet)**

```bash
lit -v mlir/test/Dialect/CSL/Transforms/allocate/one_color.mlir
```

- [ ] **Step 3: Create header + impl**

Header `mlir/include/air/Dialect/CSL/Transforms/CSLAllocateColorIdsPass.h`:

```cpp
#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_ALLOCATE_COLOR_IDS_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_ALLOCATE_COLOR_IDS_PASS_H

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {
std::unique_ptr<mlir::Pass> createCSLAllocateColorIdsPass();
} // namespace air
} // namespace xilinx

#endif
```

Impl `mlir/lib/Dialect/CSL/Transforms/CSLAllocateColorIds.cpp`:

```cpp
//===- CSLAllocateColorIds.cpp ---------------------*- C++ -*-===//
// Pass 2: assign integer ids to virtual csl.color ops.
// Algorithm (milestone stub): monotonic, skip pinned. Future home for
// liveness-driven graph coloring over WSE-3's 24-color budget.
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLAllocateColorIdsPass.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace xilinx::csl;

namespace {

class CSLAllocateColorIdsPass
    : public PassWrapper<CSLAllocateColorIdsPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-allocate-color-ids"; }
  StringRef getDescription() const final {
    return "Pass 2: assign integer ids to virtual csl.color ops "
           "(monotonic stub; future graph-coloring home).";
  }
  void runOnOperation() override;
};

} // namespace

void CSLAllocateColorIdsPass::runOnOperation() {
  // First gather already-pinned ids.
  llvm::SmallSet<int32_t, 32> usedIds;
  getOperation()->walk([&](ColorOp c) {
    if (auto idAttr = c.getIdAttr())
      usedIds.insert(idAttr.getInt());
  });

  // Walk again, assigning fresh ids in declaration order.
  int32_t next = 0;
  getOperation()->walk([&](ColorOp c) {
    if (c.getIdAttr())
      return;
    while (usedIds.count(next))
      ++next;
    c.setIdAttr(IntegerAttr::get(IntegerType::get(c.getContext(), 32), next));
    usedIds.insert(next);
    ++next;
  });
}

std::unique_ptr<Pass> xilinx::air::createCSLAllocateColorIdsPass() {
  return std::make_unique<CSLAllocateColorIdsPass>();
}
```

- [ ] **Step 4: Wire CMake + Passes registration**

In `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`, add `CSLAllocateColorIds.cpp` to the `add_mlir_library(CSLTransforms ...)` source list (next to `CSLMaterializeStreamColors.cpp` from Task 9).

In `mlir/include/air/Dialect/CSL/Transforms/Passes.h`, add:
```cpp
#include "air/Dialect/CSL/Transforms/CSLAllocateColorIdsPass.h"
```

In `mlir/lib/Dialect/CSL/Transforms/Passes.cpp`, append to the body of `registerCSLTransformPasses()`:
```cpp
mlir::registerPass(
    []() -> std::unique_ptr<mlir::Pass> {
      return createCSLAllocateColorIdsPass();
    });
```

- [ ] **Step 5: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/Transforms/allocate/one_color.mlir
```

- [ ] **Step 6: Add `three_colors.mlir`**

```mlir
// RUN: air-opt --csl-allocate-color-ids %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @a {id = 0 : i32}
    csl.color @a : !csl.color
    // CHECK: csl.color @b {id = 1 : i32}
    csl.color @b : !csl.color
    // CHECK: csl.color @c {id = 2 : i32}
    csl.color @c : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 7: Add `with_pin.mlir`**

```mlir
// RUN: air-opt --csl-allocate-color-ids %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @v {id = 0 : i32}
    csl.color @v : !csl.color
    // CHECK: csl.color @pinned {id = 5 : i32}
    csl.color @pinned {id = 5 : i32} : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 8: Add `skips_pinned_id.mlir`**

```mlir
// RUN: air-opt --csl-allocate-color-ids %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @pinned {id = 0 : i32}
    csl.color @pinned {id = 0 : i32} : !csl.color
    // CHECK: csl.color @v {id = 1 : i32}
    csl.color @v : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 9: Add `idempotent.mlir`**

```mlir
// RUN: air-opt --csl-allocate-color-ids --csl-allocate-color-ids %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // CHECK: csl.color @c {id = 0 : i32}
    csl.color @c : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 10: Run all allocate tests — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/Transforms/allocate/
```

- [ ] **Step 11: Run full suite**

```bash
cd build && ninja check-air-mlir
```

- [ ] **Step 12: Commit**

```bash
git add mlir/include/air/Dialect/CSL/Transforms/CSLAllocateColorIdsPass.h \
        mlir/lib/Dialect/CSL/Transforms/CSLAllocateColorIds.cpp \
        mlir/include/air/Dialect/CSL/Transforms/Passes.h \
        mlir/lib/Dialect/CSL/Transforms/Passes.cpp \
        mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt \
        mlir/test/Dialect/CSL/Transforms/allocate/
git commit -m "feat(csl-pass): csl-allocate-color-ids — monotonic stub allocator"
```

---

### Task 11: Pass 3 — `--csl-lower-stream-routing`

**Files:**
- Create: `mlir/include/air/Dialect/CSL/Transforms/CSLLowerStreamRoutingPass.h`
- Create: `mlir/lib/Dialect/CSL/Transforms/CSLLowerStreamRouting.cpp`
- Tests: `mlir/test/Dialect/CSL/Transforms/lower-routing/{east,west,south,north,keeps_stream}.mlir`
- Wire CMake + Passes (same pattern as Tasks 9, 10)

- [ ] **Step 1: Write `east.mlir`**

Create `mlir/test/Dialect/CSL/Transforms/lower-routing/east.mlir`:

```mlir
// RUN: air-opt --csl-lower-stream-routing %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @c0 {id = 0 : i32} : !csl.color
    csl_layout.stream @ch from(0, 0) to(1, 0) {color = @c0}
    // CHECK: csl_layout.set_color_config @c0 at(0, 0) rx(RAMP) tx(EAST)
    // CHECK: csl_layout.set_color_config @c0 at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/Transforms/lower-routing/east.mlir
```

- [ ] **Step 3: Create header + impl**

Header (mirror Task 9):

```cpp
// CSLLowerStreamRoutingPass.h
#ifndef AIR_DIALECT_CSL_TRANSFORMS_CSL_LOWER_STREAM_ROUTING_PASS_H
#define AIR_DIALECT_CSL_TRANSFORMS_CSL_LOWER_STREAM_ROUTING_PASS_H

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {
std::unique_ptr<mlir::Pass> createCSLLowerStreamRoutingPass();
}
}
#endif
```

Impl `mlir/lib/Dialect/CSL/Transforms/CSLLowerStreamRouting.cpp`:

```cpp
//===- CSLLowerStreamRouting.cpp ---------------------*- C++ -*-===//
// Pass 3: emit two csl_layout.set_color_config ops per stream
// (one at from-coord, one at to-coord). Direction inferred from coord delta.
// The csl_layout.stream op is kept (still needed by put/get resolution
// in Pass 4).
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLLowerStreamRoutingPass.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx::csl;
using namespace xilinx::csl_layout;

namespace {

// Given a coord delta, return (src_tx, dst_rx). RAMP is always rx-on-src
// and tx-on-dst.
struct Dirs { Direction src_tx; Direction dst_rx; };

static Dirs inferDirs(int64_t dx, int64_t dy) {
  if (dx == 1 && dy == 0)  return {Direction::EAST,  Direction::WEST};
  if (dx == -1 && dy == 0) return {Direction::WEST,  Direction::EAST};
  if (dx == 0 && dy == 1)  return {Direction::SOUTH, Direction::NORTH};
  if (dx == 0 && dy == -1) return {Direction::NORTH, Direction::SOUTH};
  llvm_unreachable("verifier on csl_layout.stream rejects non-cardinal deltas");
}

class CSLLowerStreamRoutingPass
    : public PassWrapper<CSLLowerStreamRoutingPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-lower-stream-routing"; }
  StringRef getDescription() const final {
    return "Pass 3: emit per-PE set_color_config from each csl_layout.stream";
  }
  void runOnOperation() override;
};

} // namespace

void CSLLowerStreamRoutingPass::runOnOperation() {
  getOperation()->walk([&](LayoutOp layout) {
    OpBuilder b(layout.getContext());
    for (auto stream :
         llvm::to_vector(layout.getBody().front().getOps<StreamOp>())) {
      auto colorAttr = stream.getColorAttr();
      if (!colorAttr) {
        stream.emitOpError("Pass 3 requires a {color = ...} attribute "
                           "(run --csl-materialize-stream-colors first)");
        signalPassFailure();
        return;
      }
      int64_t dx = stream.getToX() - stream.getFromX();
      int64_t dy = stream.getToY() - stream.getFromY();
      auto [srcTx, dstRx] = inferDirs(dx, dy);
      // Insert set_color_configs *after* the stream so they appear next to it.
      b.setInsertionPointAfter(stream);
      b.create<SetColorConfigOp>(
          stream.getLoc(), colorAttr,
          /*px=*/b.getI64IntegerAttr(stream.getFromX()),
          /*py=*/b.getI64IntegerAttr(stream.getFromY()),
          /*rx=*/Direction::RAMP, /*tx=*/srcTx);
      b.create<SetColorConfigOp>(
          stream.getLoc(), colorAttr,
          /*px=*/b.getI64IntegerAttr(stream.getToX()),
          /*py=*/b.getI64IntegerAttr(stream.getToY()),
          /*rx=*/dstRx, /*tx=*/Direction::RAMP);
      // Stream op is NOT erased here — Pass 4 needs it for symbol resolution.
    }
  });
}

std::unique_ptr<Pass> xilinx::air::createCSLLowerStreamRoutingPass() {
  return std::make_unique<CSLLowerStreamRoutingPass>();
}
```

Note: the `Direction` enum's exact include and namespace — check `CSLBase.td` and the generated `CSLEnums.h.inc` for the C++ symbol; adjust the `using` declaration above. The TableGen builder for `SetColorConfigOp` may take the direction as the enum directly or as an integer attr — check the generated `.h.inc`.

- [ ] **Step 4: Wire CMake + Passes registration**

In `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`, add `CSLLowerStreamRouting.cpp` to the `CSLTransforms` source list.

In `mlir/include/air/Dialect/CSL/Transforms/Passes.h`, add:
```cpp
#include "air/Dialect/CSL/Transforms/CSLLowerStreamRoutingPass.h"
```

In `mlir/lib/Dialect/CSL/Transforms/Passes.cpp`, append to `registerCSLTransformPasses()`:
```cpp
mlir::registerPass(
    []() -> std::unique_ptr<mlir::Pass> {
      return createCSLLowerStreamRoutingPass();
    });
```

- [ ] **Step 5: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/Transforms/lower-routing/east.mlir
```

- [ ] **Step 6: Add `west.mlir`, `south.mlir`, `north.mlir`**

Each file mirrors `east.mlir` with different deltas:

`west.mlir`:
```mlir
// RUN: air-opt --csl-lower-stream-routing %s | FileCheck %s
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2, height = 1} @layout {
    csl.color @c0 {id = 0 : i32} : !csl.color
    csl_layout.stream @ch from(1, 0) to(0, 0) {color = @c0}
    // CHECK: csl_layout.set_color_config @c0 at(1, 0) rx(RAMP) tx(WEST)
    // CHECK: csl_layout.set_color_config @c0 at(0, 0) rx(EAST) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
```

`south.mlir`:
```mlir
// RUN: air-opt --csl-lower-stream-routing %s | FileCheck %s
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1, height = 2} @layout {
    csl.color @c0 {id = 0 : i32} : !csl.color
    csl_layout.stream @ch from(0, 0) to(0, 1) {color = @c0}
    // CHECK: csl_layout.set_color_config @c0 at(0, 0) rx(RAMP) tx(SOUTH)
    // CHECK: csl_layout.set_color_config @c0 at(0, 1) rx(NORTH) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (0, 1)
  }
}
```

`north.mlir`:
```mlir
// RUN: air-opt --csl-lower-stream-routing %s | FileCheck %s
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 1, height = 2} @layout {
    csl.color @c0 {id = 0 : i32} : !csl.color
    csl_layout.stream @ch from(0, 1) to(0, 0) {color = @c0}
    // CHECK: csl_layout.set_color_config @c0 at(0, 1) rx(RAMP) tx(NORTH)
    // CHECK: csl_layout.set_color_config @c0 at(0, 0) rx(SOUTH) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (0, 1)
  }
}
```

- [ ] **Step 7: Add `keeps_stream.mlir`**

```mlir
// RUN: air-opt --csl-lower-stream-routing %s | FileCheck %s

// After Pass 3 the csl_layout.stream op must STILL be present
// (Pass 4 uses it for stream.put/get symbol resolution).
csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2, height = 1} @layout {
    csl.color @c0 {id = 0 : i32} : !csl.color
    // CHECK: csl_layout.stream @ch from(0, 0) to(1, 0) {color = @c0}
    csl_layout.stream @ch from(0, 0) to(1, 0) {color = @c0}
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 8: Run all routing tests — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/Transforms/lower-routing/
```

- [ ] **Step 9: Run full suite**

```bash
cd build && ninja check-air-mlir
```

- [ ] **Step 10: Commit**

```bash
git add mlir/include/air/Dialect/CSL/Transforms/CSLLowerStreamRoutingPass.h \
        mlir/lib/Dialect/CSL/Transforms/CSLLowerStreamRouting.cpp \
        mlir/include/air/Dialect/CSL/Transforms/Passes.h \
        mlir/lib/Dialect/CSL/Transforms/Passes.cpp \
        mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt \
        mlir/test/Dialect/CSL/Transforms/lower-routing/
git commit -m "feat(csl-pass): csl-lower-stream-routing — emit set_color_config per endpoint"
```

---

### Task 12: Pass 4 — `--csl-lower-stream-data`

This is the largest pass — it expands put/get into fabric DSDs + tasks + async builtin calls and erases the stream op.

**Files:**
- Create: `mlir/include/air/Dialect/CSL/Transforms/CSLLowerStreamDataPass.h`
- Create: `mlir/lib/Dialect/CSL/Transforms/CSLLowerStreamData.cpp`
- Tests: `mlir/test/Dialect/CSL/Transforms/lower-data/{put,get,erases_stream,task_naming,unique_task_ids}.mlir`

- [ ] **Step 1: Write `put.mlir`**

Create `mlir/test/Dialect/CSL/Transforms/lower-data/put.mlir`:

```mlir
// RUN: air-opt --csl-lower-stream-data %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: %[[SRC:.*]] = csl.get_mem_dsd %{{.*}} : memref<128xf32> -> !csl.dsd
      // CHECK: %[[OUT:.*]] = csl.get_fab_dsd fabout @ch_color extent(%{{.*}}
      // CHECK: csl.task @ch_put_done_0 {id = 8 : i32, trigger_kind = "local_task_id"}
      // CHECK:   csl.builtin_call "unblock_cmd_stream"() : () -> ()
      // CHECK: csl.builtin_call "fmovs"(%[[OUT]], %[[SRC]]) {activate = @ch_put_done_0, async}
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @ch_color {id = 0 : i32} : !csl.color
    csl_layout.stream @ch from(0, 0) to(1, 0) {color = @ch_color}
    csl_layout.set_color_config @ch_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @ch_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @left at (0, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit -v mlir/test/Dialect/CSL/Transforms/lower-data/put.mlir
```

- [ ] **Step 3: Create header + impl**

Header is the same shape as Task 9.

Impl `mlir/lib/Dialect/CSL/Transforms/CSLLowerStreamData.cpp`:

```cpp
//===- CSLLowerStreamData.cpp ---------------------*- C++ -*-===//
// Pass 4: expand csl.stream.put / csl.stream.get into fabric DSDs +
// task + async builtin call. Erases all csl_layout.stream / put / get
// at the end.
//
// Per-program task-id counter starts at 8 (matching tutorial idiom)
// and increments per put/get.
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Transforms/CSLLowerStreamDataPass.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx::csl;
using namespace xilinx::csl_layout;

namespace {

class CSLLowerStreamDataPass
    : public PassWrapper<CSLLowerStreamDataPass, OperationPass<>> {
public:
  StringRef getArgument() const final { return "csl-lower-stream-data"; }
  StringRef getDescription() const final {
    return "Pass 4: expand csl.stream.put/get into fabric DSDs + tasks + "
           "async builtins; erase csl_layout.stream.";
  }
  void runOnOperation() override;

private:
  // Resolve a stream symbol from inside a program body.
  StreamOp resolveStream(Operation *fromInProgram, FlatSymbolRefAttr name);
  // Find the csl.layout child of the parent wafer.
  LayoutOp getLayout(Operation *fromInProgram);
  // Per-program counter for unique task ids; restarts at 8 each program.
  void expandPut(StreamPutOp put, int32_t &nextTaskId);
  void expandGet(StreamGetOp get, int32_t &nextTaskId);
};

} // namespace

LayoutOp CSLLowerStreamDataPass::getLayout(Operation *op) {
  auto wafer = op->getParentOfType<WaferOp>();
  for (auto &child : wafer.getBody().front())
    if (auto l = dyn_cast<LayoutOp>(child)) return l;
  return nullptr;
}

StreamOp CSLLowerStreamDataPass::resolveStream(Operation *op,
                                                FlatSymbolRefAttr name) {
  auto layout = getLayout(op);
  return dyn_cast_or_null<StreamOp>(
      SymbolTable::lookupSymbolIn(layout, name.getAttr()));
}

void CSLLowerStreamDataPass::expandPut(StreamPutOp put, int32_t &nextTaskId) {
  auto stream = resolveStream(put, put.getStreamAttr());
  if (!stream) {
    put.emitOpError("could not resolve stream '@") << put.getStream() << "'";
    return;
  }
  auto colorAttr = stream.getColorAttr();
  OpBuilder b(put);
  Location loc = put.getLoc();

  // 1. mem DSD on source.
  auto srcDsd = b.create<GetMemDsdOp>(
      loc, DsdType::get(b.getContext()), put.getSource());

  // 2. fabric DSD on the color (fabout).
  auto outDsd = b.create<GetFabDsdOp>(
      loc, DsdType::get(b.getContext()),
      FabDsdDirection::fabout, colorAttr, put.getExtent());

  // 3. allocate task id; create task @<stream>_put_done_<n>.
  int32_t id = nextTaskId++;
  std::string taskName =
      (stream.getSymName() + "_put_done_" + Twine(id - 8)).str();

  auto program = put->getParentOfType<ProgramOp>();
  OpBuilder pb(&program.getBody().front(), program.getBody().front().end());
  // Move insertion point: place the task at end of program body.
  auto task = pb.create<TaskOp>(
      loc, b.getStringAttr(taskName),
      /*trigger_kind=*/b.getStringAttr("local_task_id"),
      /*id=*/b.getI32IntegerAttr(id),
      /*color=*/FlatSymbolRefAttr());
  // Create the body region with one builtin_call + return.
  Block &body = task.getBody().emplaceBlock();
  OpBuilder bb(&body, body.begin());
  bb.create<BuiltinCallOp>(
      loc, /*results=*/TypeRange{},
      /*callee=*/bb.getStringAttr("unblock_cmd_stream"),
      /*module=*/Value(), /*args=*/ValueRange{},
      /*async=*/UnitAttr(), /*activate=*/FlatSymbolRefAttr());
  bb.create<ReturnOp>(loc);

  // 4. emit the async fmovs.
  auto activateRef = FlatSymbolRefAttr::get(b.getContext(), taskName);
  auto fmovs = b.create<BuiltinCallOp>(
      loc, /*results=*/TypeRange{},
      /*callee=*/b.getStringAttr("fmovs"),
      /*module=*/Value(),
      /*args=*/ValueRange{outDsd.getResult(), srcDsd.getResult()},
      /*async=*/UnitAttr::get(b.getContext()),
      /*activate=*/activateRef);

  // 5. erase the stream.put op.
  put.erase();
  (void)fmovs;
}

void CSLLowerStreamDataPass::expandGet(StreamGetOp get, int32_t &nextTaskId) {
  auto stream = resolveStream(get, get.getStreamAttr());
  if (!stream) {
    get.emitOpError("could not resolve stream '@") << get.getStream() << "'";
    return;
  }
  auto colorAttr = stream.getColorAttr();
  OpBuilder b(get);
  Location loc = get.getLoc();

  auto tgtDsd = b.create<GetMemDsdOp>(
      loc, DsdType::get(b.getContext()), get.getTarget());
  auto inDsd = b.create<GetFabDsdOp>(
      loc, DsdType::get(b.getContext()),
      FabDsdDirection::fabin, colorAttr, get.getExtent());

  int32_t id = nextTaskId++;
  std::string taskName =
      (stream.getSymName() + "_get_done_" + Twine(id - 8)).str();

  auto program = get->getParentOfType<ProgramOp>();
  OpBuilder pb(&program.getBody().front(), program.getBody().front().end());
  auto task = pb.create<TaskOp>(
      loc, b.getStringAttr(taskName),
      b.getStringAttr("local_task_id"),
      b.getI32IntegerAttr(id),
      FlatSymbolRefAttr());
  Block &body = task.getBody().emplaceBlock();
  OpBuilder bb(&body, body.begin());
  bb.create<BuiltinCallOp>(
      loc, TypeRange{},
      bb.getStringAttr("unblock_cmd_stream"),
      Value(), ValueRange{},
      UnitAttr(), FlatSymbolRefAttr());
  bb.create<ReturnOp>(loc);

  auto activateRef = FlatSymbolRefAttr::get(b.getContext(), taskName);
  b.create<BuiltinCallOp>(
      loc, TypeRange{}, b.getStringAttr("fmovs"),
      Value(), ValueRange{tgtDsd.getResult(), inDsd.getResult()},
      UnitAttr::get(b.getContext()), activateRef);

  get.erase();
}

void CSLLowerStreamDataPass::runOnOperation() {
  // Per-program task-id counter (restarts at 8 per program).
  getOperation()->walk([&](ProgramOp program) {
    int32_t nextTaskId = 8;
    program->walk([&](Operation *op) {
      if (auto put = dyn_cast<StreamPutOp>(op)) expandPut(put, nextTaskId);
      else if (auto get = dyn_cast<StreamGetOp>(op)) expandGet(get, nextTaskId);
    });
  });

  // Erase all csl_layout.stream ops (no longer needed).
  getOperation()->walk([&](StreamOp s) { s.erase(); });
}

std::unique_ptr<Pass> xilinx::air::createCSLLowerStreamDataPass() {
  return std::make_unique<CSLLowerStreamDataPass>();
}
```

The exact builder signatures (which attrs are positional vs by name) depend on TableGen output. After build, check `build/include/air/Dialect/CSL/CSLOps.h.inc` for `BuiltinCallOp::build()` and adjust the calls. Same for `TaskOp::build()`, `GetFabDsdOp::build()`.

- [ ] **Step 4: Wire CMake + Passes registration**

In `mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt`, add `CSLLowerStreamData.cpp` to the `CSLTransforms` source list.

In `mlir/include/air/Dialect/CSL/Transforms/Passes.h`, add:
```cpp
#include "air/Dialect/CSL/Transforms/CSLLowerStreamDataPass.h"
```

In `mlir/lib/Dialect/CSL/Transforms/Passes.cpp`, append to `registerCSLTransformPasses()`:
```cpp
mlir::registerPass(
    []() -> std::unique_ptr<mlir::Pass> {
      return createCSLLowerStreamDataPass();
    });
```

- [ ] **Step 5: Build and re-run `put.mlir` — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/Transforms/lower-data/put.mlir
```

- [ ] **Step 6: Add `get.mlir`**

```mlir
// RUN: air-opt --csl-lower-stream-data %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @right {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: %[[TGT:.*]] = csl.get_mem_dsd %{{.*}} : memref<128xf32> -> !csl.dsd
      // CHECK: %[[IN:.*]] = csl.get_fab_dsd fabin @ch_color extent(%{{.*}}
      // CHECK: csl.task @ch_get_done_0 {id = 8 : i32, trigger_kind = "local_task_id"}
      // CHECK: csl.builtin_call "fmovs"(%[[TGT]], %[[IN]]) {activate = @ch_get_done_0, async}
      csl.stream.get @ch target(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @ch_color {id = 0 : i32} : !csl.color
    csl_layout.stream @ch from(0, 0) to(1, 0) {color = @ch_color}
    csl_layout.set_color_config @ch_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @ch_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @right at (1, 0)
  }
}
```

- [ ] **Step 7: Add `erases_stream.mlir`**

```mlir
// RUN: air-opt --csl-lower-stream-data %s | FileCheck %s

// CHECK-NOT: csl_layout.stream
// CHECK-NOT: csl.stream.put
// CHECK-NOT: csl.stream.get

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
  }
  csl.layout {width = 2, height = 1} @layout {
    csl.color @ch_color {id = 0 : i32} : !csl.color
    csl_layout.stream @ch from(0, 0) to(1, 0) {color = @ch_color}
    csl_layout.set_color_config @ch_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @ch_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @left at (0, 0)
  }
}
```

- [ ] **Step 8: Add `unique_task_ids.mlir`**

```mlir
// RUN: air-opt --csl-lower-stream-data %s | FileCheck %s

// Multiple put/gets in one program get unique ids 8, 9, …
csl.wafer @w {arch = "wse3"} {
  csl.program @sender {
    %a = csl.var @a : memref<32xf32>
    %b = csl.var @b : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.stream.put @ch1 source(%a) extent(%n : index) : memref<32xf32>
      csl.stream.put @ch2 source(%b) extent(%n : index) : memref<32xf32>
      csl.return
    }
    // CHECK-DAG: csl.task @ch1_put_done_0 {id = 8 : i32
    // CHECK-DAG: csl.task @ch2_put_done_0 {id = 9 : i32
  }
  csl.layout {width = 2, height = 2} @layout {
    csl.color @ch1_color {id = 0 : i32} : !csl.color
    csl.color @ch2_color {id = 1 : i32} : !csl.color
    csl_layout.stream @ch1 from(0, 0) to(1, 0) {color = @ch1_color}
    csl_layout.stream @ch2 from(0, 0) to(0, 1) {color = @ch2_color}
    csl_layout.set_color_config @ch1_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @ch1_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.set_color_config @ch2_color at(0, 0) rx(RAMP) tx(SOUTH)
    csl_layout.set_color_config @ch2_color at(0, 1) rx(NORTH) tx(RAMP)
    csl_layout.place @sender at (0, 0)
  }
}
```

- [ ] **Step 9: Add `task_naming.mlir`**

```mlir
// RUN: air-opt --csl-lower-stream-data %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @s {
    %a = csl.var @a : memref<8xf32>
    csl.func @c {
      %n = arith.constant 8 : index
      // CHECK: csl.task @my_stream_put_done_0
      csl.stream.put @my_stream source(%a) extent(%n : index) : memref<8xf32>
      csl.return
    }
  }
  csl.layout {width = 2, height = 1} @layout {
    csl.color @my_stream_color {id = 0 : i32} : !csl.color
    csl_layout.stream @my_stream from(0, 0) to(1, 0) {color = @my_stream_color}
    csl_layout.set_color_config @my_stream_color at(0,0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @my_stream_color at(1,0) rx(WEST) tx(RAMP)
    csl_layout.place @s at (0, 0)
  }
}
```

- [ ] **Step 10: Run all lower-data tests — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/Transforms/lower-data/
```

- [ ] **Step 11: Run full suite**

```bash
cd build && ninja check-air-mlir
```

- [ ] **Step 12: Commit**

```bash
git add mlir/include/air/Dialect/CSL/Transforms/CSLLowerStreamDataPass.h \
        mlir/lib/Dialect/CSL/Transforms/CSLLowerStreamData.cpp \
        mlir/include/air/Dialect/CSL/Transforms/Passes.h \
        mlir/lib/Dialect/CSL/Transforms/Passes.cpp \
        mlir/lib/Dialect/CSL/Transforms/CMakeLists.txt \
        mlir/test/Dialect/CSL/Transforms/lower-data/
git commit -m "feat(csl-pass): csl-lower-stream-data — expand put/get to fabric DSDs + tasks"
```

---

## Phase D — Pipeline registration

### Task 13: Register `csl-streams-to-csl` and `csl-pipeline`

**Files:**
- Create: `mlir/include/air/Dialect/CSL/Pipelines/Pipelines.h`
- Create: `mlir/lib/Dialect/CSL/Pipelines/CSLStreamsPipeline.cpp`
- Create: `mlir/lib/Dialect/CSL/Pipelines/CSLPipeline.cpp`
- Create: `mlir/lib/Dialect/CSL/Pipelines/CMakeLists.txt`
- Modify: `mlir/lib/Dialect/CSL/CMakeLists.txt` (add Pipelines subdir)
- Modify: `tools/air-opt/air-opt.cpp` (call registerCSLPipelines)
- Tests: `mlir/test/Dialect/CSL/Pipelines/{streams_to_csl,full}.mlir`

- [ ] **Step 1: Write `streams_to_csl.mlir` test**

Create `mlir/test/Dialect/CSL/Pipelines/streams_to_csl.mlir`:

```mlir
// RUN: air-opt --csl-streams-to-csl %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
  }
  csl.program @right {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.stream.get @ch target(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @left  at (0, 0)
    csl_layout.place @right at (1, 0)
  }
}

// After full sub-pipeline: stream gone, fabric DSDs + tasks present, set_color_config emitted.
// CHECK-NOT: csl_layout.stream
// CHECK-NOT: csl.stream.put
// CHECK-NOT: csl.stream.get
// CHECK: csl.color @ch_color {id = 0 : i32}
// CHECK: csl_layout.set_color_config @ch_color at(0, 0) rx(RAMP) tx(EAST)
// CHECK: csl_layout.set_color_config @ch_color at(1, 0) rx(WEST) tx(RAMP)
// CHECK: csl.get_fab_dsd fabout @ch_color
// CHECK: csl.get_fab_dsd fabin @ch_color
// CHECK: csl.task @ch_put_done_0
// CHECK: csl.task @ch_get_done_0
// CHECK: csl.builtin_call "fmovs"{{.*}}{activate = @ch_put_done_0, async}
// CHECK: csl.builtin_call "fmovs"{{.*}}{activate = @ch_get_done_0, async}
```

- [ ] **Step 2: Run — expect FAIL (pipeline not registered)**

```bash
lit -v mlir/test/Dialect/CSL/Pipelines/streams_to_csl.mlir
```

- [ ] **Step 3: Create header and impls**

`mlir/include/air/Dialect/CSL/Pipelines/Pipelines.h`:
```cpp
#ifndef AIR_DIALECT_CSL_PIPELINES_H
#define AIR_DIALECT_CSL_PIPELINES_H

namespace xilinx {
namespace air {
void registerCSLPipelines();
}
}
#endif
```

`mlir/lib/Dialect/CSL/Pipelines/CSLStreamsPipeline.cpp`:
```cpp
//===- CSLStreamsPipeline.cpp ---------------------*- C++ -*-===//
// Registers csl-streams-to-csl: 4-pass bundle that lowers user-facing
// stream ops to emit-ready CSL form.
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Pipelines/Pipelines.h"
#include "air/Dialect/CSL/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace xilinx::air::detail {
inline void buildCSLStreamsToCSLPipeline(OpPassManager &pm) {
  pm.addPass(createCSLMaterializeStreamColorsPass());
  pm.addPass(createCSLAllocateColorIdsPass());
  pm.addPass(createCSLLowerStreamRoutingPass());
  pm.addPass(createCSLLowerStreamDataPass());
}
} // namespace xilinx::air::detail

namespace {
void registerCSLStreamsToCSL() {
  PassPipelineRegistration<>(
      "csl-streams-to-csl",
      "Lower CSL stream ops to emit-ready form (materialize colors → "
      "allocate ids → lower routing → lower data).",
      ::xilinx::air::detail::buildCSLStreamsToCSLPipeline);
}
} // namespace

void registerCSLStreamsToCSLOnce() {
  static bool done = false;
  if (!done) { registerCSLStreamsToCSL(); done = true; }
}
```

`mlir/lib/Dialect/CSL/Pipelines/CSLPipeline.cpp`:
```cpp
//===- CSLPipeline.cpp ---------------------*- C++ -*-===//
// Registers csl-pipeline: top-level frontend → emit-ready chain.
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/Pipelines/Pipelines.h"
#include "air/Dialect/CSL/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace xilinx::air::detail {
extern void buildCSLStreamsToCSLPipeline(OpPassManager &pm);
} // namespace xilinx::air::detail

namespace {
void registerCSLTopPipeline() {
  PassPipelineRegistration<>(
      "csl-pipeline",
      "Full CSL pipeline: infer-exports → auto-vectorize → streams-to-csl. "
      "Produces emit-ready IR for air-translate --emit-csl.",
      [](OpPassManager &pm) {
        pm.addPass(::xilinx::air::createCSLInferExportsPass());
        pm.addPass(::xilinx::air::createCSLAutoVectorizePass());
        ::xilinx::air::detail::buildCSLStreamsToCSLPipeline(pm);
      });
}
void registerCSLTopPipelineOnce() {
  static bool done = false;
  if (!done) { registerCSLTopPipeline(); done = true; }
}
} // namespace

void xilinx::air::registerCSLPipelines() {
  // Forward to per-pipeline registrars.
  extern void registerCSLStreamsToCSLOnce();
  registerCSLStreamsToCSLOnce();
  registerCSLTopPipelineOnce();
}
```

- [ ] **Step 4: Create `Pipelines/CMakeLists.txt`**

```cmake
add_mlir_library(
  CSLPipelines
  CSLStreamsPipeline.cpp
  CSLPipeline.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/CSL/Pipelines

  LINK_LIBS PUBLIC
  CSLDialect
  CSLTransforms
  MLIRPass)
```

- [ ] **Step 5: Wire `Pipelines/` into `mlir/lib/Dialect/CSL/CMakeLists.txt`**

Add `add_subdirectory(Pipelines)` next to the existing `add_subdirectory(Transforms)` etc.

- [ ] **Step 6: Call `registerCSLPipelines()` in `air-opt.cpp`**

In `tools/air-opt/air-opt.cpp`, add:

```cpp
#include "air/Dialect/CSL/Pipelines/Pipelines.h"
// ... in main(), after other registration calls:
xilinx::air::registerCSLPipelines();
```

Also update `tools/air-opt/CMakeLists.txt` to link `CSLPipelines`:

```cmake
target_link_libraries(air-opt PRIVATE
  ...
  CSLPipelines)
```

- [ ] **Step 7: Build and re-run `streams_to_csl.mlir` — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Dialect/CSL/Pipelines/streams_to_csl.mlir
```

- [ ] **Step 8: Write `full.mlir` test (csl-pipeline including frontend)**

```mlir
// RUN: air-opt --csl-pipeline %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf
    csl.export @c {kind = "func"}
  }
  csl.program @right {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.stream.get @ch target(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf
    csl.export @c {kind = "func"}
  }
  csl.layout {width = 2, height = 1} @layout {
    csl_layout.stream @ch from(0, 0) to(1, 0)
    csl_layout.place @left  at (0, 0)
    csl_layout.place @right at (1, 0)
  }
}

// CHECK: csl.export @buf {direction = "in"
// CHECK-NOT: csl_layout.stream
// CHECK: csl.color @ch_color {id = 0 : i32}
// CHECK: csl.task @ch_put_done_0
// CHECK: csl.task @ch_get_done_0
```

- [ ] **Step 9: Run `full.mlir` — expect PASS**

```bash
lit -v mlir/test/Dialect/CSL/Pipelines/full.mlir
```

- [ ] **Step 10: Run full suite**

```bash
cd build && ninja check-air-mlir
```

- [ ] **Step 11: Commit**

```bash
git add mlir/include/air/Dialect/CSL/Pipelines/Pipelines.h \
        mlir/lib/Dialect/CSL/Pipelines/ \
        mlir/lib/Dialect/CSL/CMakeLists.txt \
        tools/air-opt/air-opt.cpp \
        tools/air-opt/CMakeLists.txt \
        mlir/test/Dialect/CSL/Pipelines/
git commit -m "feat(csl-pipeline): register csl-streams-to-csl and csl-pipeline"
```

---

## Phase E — Emitter cases

The Stage-4 IR is CSL-isomorphic — every new op maps to one CSL line. Existing emitter handles the other ops; we add the missing cases.

---

### Task 14: Emitter — `csl_layout.set_color_config` + `csl.color` in layout body

**Files:**
- Modify: `mlir/lib/Targets/CSLEmit/<emitter file>.cpp` (add cases)
- Tests: `mlir/test/Targets/CSLEmit/multi_pe/emit_set_color_config.mlir`

- [ ] **Step 1: Locate the emitter dispatcher**

```bash
grep -rn "csl.layout\|emitOp\|dispatchOp" mlir/lib/Targets/CSLEmit/ | head -20
```

Identify the file that handles `csl.layout` body emission. Existing pattern: usually a switch/dispatch that handles `csl_layout.place`, etc.

- [ ] **Step 2: Write the emit test**

Create `mlir/test/Targets/CSLEmit/multi_pe/emit_set_color_config.mlir`:

```mlir
// RUN: air-translate --emit-csl %s -o %t-dir && cat %t-dir/layout.csl | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK: const send_color = @get_color(0);
    csl.color @send_color {id = 0 : i32} : !csl.color
    // CHECK: @set_color_config(0, 0, send_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
    csl_layout.set_color_config @send_color at(0, 0) rx(RAMP) tx(EAST)
    // CHECK: @set_color_config(1, 0, send_color, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });
    csl_layout.set_color_config @send_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}
```

- [ ] **Step 3: Run — expect FAIL**

```bash
lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_set_color_config.mlir
```

- [ ] **Step 4: Add emitter case for `csl.color` (now in layout body)**

In the emitter's layout-body dispatcher, add a case (mirror existing op cases):

```cpp
} else if (auto color = dyn_cast<csl::ColorOp>(op)) {
  int32_t id = color.getIdAttr().getInt();
  os << "  const " << color.getSymName() << " = @get_color(" << id << ");\n";
}
```

Place this near the top of the layout body emission so colors are declared before `set_color_config` references them.

- [ ] **Step 5: Add emitter case for `csl_layout.set_color_config`**

```cpp
} else if (auto sc = dyn_cast<csl_layout::SetColorConfigOp>(op)) {
  os << "  @set_color_config(" << sc.getPx() << ", " << sc.getPy()
     << ", " << sc.getColor().getValue()
     << ", .{ .routes = .{ .rx = .{" << directionName(sc.getRx())
     << "}, .tx = .{" << directionName(sc.getTx()) << "} } });\n";
}
```

`directionName()` is the existing helper in the emitter that maps `Direction::EAST` to `"EAST"` etc. (verify by grep).

- [ ] **Step 6: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_set_color_config.mlir
```

- [ ] **Step 7: Run full emitter suite**

```bash
cd build && ninja check-airmlir-targets-cslemit
```

- [ ] **Step 8: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/ \
        mlir/test/Targets/CSLEmit/multi_pe/
git commit -m "feat(csl-emit): emit csl.color and csl_layout.set_color_config in layout block"
```

---

### Task 15: Emitter — `csl.get_fab_dsd`

**Files:**
- Modify: `mlir/lib/Targets/CSLEmit/<program-body emitter>.cpp`
- Test: `mlir/test/Targets/CSLEmit/multi_pe/emit_fabric_dsd.mlir`

- [ ] **Step 1: Write the emit test**

```mlir
// RUN: air-translate --emit-csl %s -o %t-dir && cat %t-dir/p.csl | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @c {
      %n = arith.constant 128 : index
      // CHECK: const {{.*}} = @get_dsd(fabout_dsd, .{ .extent = 128, .fabric_color = send_color });
      %out = csl.get_fab_dsd fabout @send_color extent(%n : index) : !csl.dsd
      // CHECK: const {{.*}} = @get_dsd(fabin_dsd, .{ .extent = 128, .fabric_color = send_color });
      %in  = csl.get_fab_dsd fabin  @send_color extent(%n : index) : !csl.dsd
      csl.return
    }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl.color @send_color {id = 0 : i32} : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_fabric_dsd.mlir
```

- [ ] **Step 3: Add emitter case**

In the program-body emitter:

```cpp
} else if (auto fd = dyn_cast<csl::GetFabDsdOp>(op)) {
  StringRef kind = (fd.getDirection() == csl::FabDsdDirection::fabout)
                       ? "fabout_dsd" : "fabin_dsd";
  // Resolve constant extent (verifier guarantees Index, but value may be SSA).
  // Use the same extent-resolution helper used by csl.get_mem_dsd if any;
  // otherwise: trace defining op to arith.constant.
  int64_t extent = resolveConstantIndex(fd.getExtent());
  os << "  const " << ssaNameOf(fd.getResult()) << " = @get_dsd("
     << kind << ", .{ .extent = " << extent << ", .fabric_color = "
     << fd.getColor().getValue() << " });\n";
}
```

`resolveConstantIndex()` and `ssaNameOf()` are existing helpers — check the emitter for the actual names.

- [ ] **Step 4: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_fabric_dsd.mlir
```

- [ ] **Step 5: Run full suite**

```bash
cd build && ninja check-airmlir-targets-cslemit
```

- [ ] **Step 6: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/ \
        mlir/test/Targets/CSLEmit/multi_pe/emit_fabric_dsd.mlir
git commit -m "feat(csl-emit): emit csl.get_fab_dsd as @get_dsd(fab{in,out}_dsd, ...)"
```

---

### Task 16: Emitter — new `csl.task` attribute form (3-line emission)

**Files:**
- Modify: emitter
- Test: `mlir/test/Targets/CSLEmit/multi_pe/emit_local_task.mlir`

- [ ] **Step 1: Write the emit test**

```mlir
// RUN: air-translate --emit-csl %s -o %t-dir && cat %t-dir/p.csl | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    csl.func @c { csl.return }
    csl.task @exit_task {trigger_kind = "local_task_id", id = 8 : i32} {
      csl.return
    }
  }
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    csl_layout.place @p at (0, 0)
  }
}

// CHECK: const exit_task_id: local_task_id = @get_local_task_id(8);
// CHECK: task exit_task() void {
// CHECK: }
// CHECK: comptime {
// CHECK:   @bind_local_task(exit_task, exit_task_id);
// CHECK: }
```

- [ ] **Step 2: Run — expect FAIL (or partial, depending on existing csl.task emission)**

```bash
lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_local_task.mlir
```

- [ ] **Step 3: Update emitter case for `csl.task`**

The existing case (if any) emitted the old `color(%c)` form. Replace with attribute-driven dispatch:

```cpp
} else if (auto task = dyn_cast<csl::TaskOp>(op)) {
  StringRef kind = task.getTriggerKind();
  StringRef name = task.getSymName();
  if (kind == "local_task_id") {
    int32_t id = task.getIdAttr().getInt();
    os << "const " << name << "_id: local_task_id = @get_local_task_id("
       << id << ");\n";
    os << "task " << name << "() void {\n";
    emitRegion(task.getBody(), os);
    os << "}\n";
    os << "comptime {\n  @bind_local_task(" << name << ", " << name
       << "_id);\n}\n";
  } else { // color
    StringRef colorSym = task.getColor().getValue();
    os << "task " << name << "() void {\n";
    emitRegion(task.getBody(), os);
    os << "}\n";
    os << "comptime {\n  @bind_local_task(" << name << ", " << colorSym
       << ");\n}\n";
  }
}
```

`emitRegion()` is the existing helper for emitting a CSL function body.

- [ ] **Step 4: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_local_task.mlir
```

- [ ] **Step 5: Run full suite**

```bash
cd build && ninja check-airmlir-targets-cslemit
```

- [ ] **Step 6: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/ \
        mlir/test/Targets/CSLEmit/multi_pe/emit_local_task.mlir
git commit -m "feat(csl-emit): emit csl.task with attribute-driven trigger as 3-line CSL triplet"
```

---

### Task 17: Emitter — `csl.builtin_call` async/activate fields

**Files:**
- Modify: emitter (existing builtin_call case)
- Test: `mlir/test/Targets/CSLEmit/multi_pe/emit_async_builtin.mlir`

- [ ] **Step 1: Write the emit test**

```mlir
// RUN: air-translate --emit-csl %s -o %t-dir && cat %t-dir/p.csl | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @p {
    %buf = csl.var @buf : memref<8xf32>
    csl.func @c {
      %n = arith.constant 8 : index
      %fd = csl.get_fab_dsd fabout @send extent(%n : index) : !csl.dsd
      %md = csl.get_mem_dsd %buf : memref<8xf32> -> !csl.dsd
      // CHECK: @fmovs({{.*}}, .{ .async = true, .activate = done_id });
      csl.builtin_call "fmovs"(%fd, %md) {async, activate = @done}
        : (!csl.dsd, !csl.dsd) -> ()
      csl.return
    }
    csl.task @done {trigger_kind = "local_task_id", id = 8 : i32} { csl.return }
  }
  csl.layout {width = 1, height = 1} @layout {
    csl.color @send {id = 0 : i32} : !csl.color
    csl_layout.place @p at (0, 0)
  }
}
```

- [ ] **Step 2: Run — expect FAIL (current emitter doesn't emit async/activate)**

```bash
lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_async_builtin.mlir
```

- [ ] **Step 3: Update emitter case for `csl.builtin_call`**

In the existing builtin_call emitter, after the args list:

```cpp
auto async = call.getAsyncAttr();
auto activate = call.getActivateAttr();
if (async || activate) {
  os << ", .{";
  bool first = true;
  if (async) { os << " .async = true"; first = false; }
  if (activate) {
    if (!first) os << ",";
    os << " .activate = " << activate.getValue() << "_id";  // _id suffix per emitter contract
  }
  os << " }";
}
os << ");\n";
```

The `_id` suffix is the implicit symbol generated by Task 16's `csl.task` emitter case. The emitter contract: a task's local-task-id symbol is named `<task_sym>_id`.

- [ ] **Step 4: Build and re-run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_async_builtin.mlir
```

- [ ] **Step 5: Run full suite**

```bash
cd build && ninja check-airmlir-targets-cslemit
```

- [ ] **Step 6: Commit**

```bash
git add mlir/lib/Targets/CSLEmit/ \
        mlir/test/Targets/CSLEmit/multi_pe/emit_async_builtin.mlir
git commit -m "feat(csl-emit): emit async + activate fields on csl.builtin_call"
```

---

## Phase F — End-to-end milestone

### Task 18: Pipeline + Emitter end-to-end FileCheck (`emit_full_ping.mlir`)

**Files:**
- Test: `mlir/test/Targets/CSLEmit/multi_pe/emit_full_ping.mlir`

This catches any glue bugs between Stages and the emitter that didn't show in unit tests.

- [ ] **Step 1: Write the full-pipeline FileCheck test**

```mlir
// RUN: air-opt --csl-pipeline %s | air-translate --emit-csl -o %t-dir
// RUN: cat %t-dir/layout.csl | FileCheck --check-prefix=LAYOUT %s
// RUN: cat %t-dir/left_pe.csl | FileCheck --check-prefix=LEFT %s
// RUN: cat %t-dir/right_pe.csl | FileCheck --check-prefix=RIGHT %s

csl.wafer @ping_2pe {arch = "wse3"} {
  csl.program @left_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.stream.put @send_ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf
    csl.export @compute {kind = "func"}
  }
  csl.program @right_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.stream.get @send_ch target(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.stream @send_ch from (0, 0) to (1, 0)
    csl_layout.place  @left_pe  at (0, 0)
    csl_layout.place  @right_pe at (1, 0)
  }
  // ... csl.host (write minimal stub or copy from saxpy.mlir for h2d/d2h)
}

// LAYOUT: const send_ch_color = @get_color(0);
// LAYOUT: @set_color_config(0, 0, send_ch_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
// LAYOUT: @set_color_config(1, 0, send_ch_color, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });

// LEFT: const {{.*}} = @get_dsd(fabout_dsd, .{ .extent = 128, .fabric_color = send_ch_color });
// LEFT: const send_ch_put_done_0_id: local_task_id = @get_local_task_id(8);
// LEFT: task send_ch_put_done_0() void {
// LEFT: comptime {
// LEFT:   @bind_local_task(send_ch_put_done_0, send_ch_put_done_0_id);
// LEFT: }
// LEFT: @fmovs({{.*}}, .{ .async = true, .activate = send_ch_put_done_0_id });

// RIGHT: const {{.*}} = @get_dsd(fabin_dsd, .{ .extent = 128, .fabric_color = send_ch_color });
// RIGHT: task send_ch_get_done_0() void {
// RIGHT: @fmovs({{.*}}, .{ .async = true, .activate = send_ch_get_done_0_id });
```

(Provide a minimal `csl.host` block — copy from `mlir/test/Targets/CSLEmit/e2e/scientific/saxpy.mlir` lines covering memcpy_h2d / launch / memcpy_d2h, adapted for two PEs.)

- [ ] **Step 2: Run — expect PASS**

```bash
cd build && ninja install -j$(nproc) && cd .. && \
  lit -v mlir/test/Targets/CSLEmit/multi_pe/emit_full_ping.mlir
```

If it fails, the failure mode tells you which Stage broke its invariant. Fix that pass / emitter case.

- [ ] **Step 3: Run full suite**

```bash
cd build && ninja check-air-mlir
```

- [ ] **Step 4: Commit**

```bash
git add mlir/test/Targets/CSLEmit/multi_pe/emit_full_ping.mlir
git commit -m "test(csl-emit): full ping kernel pipeline + emit FileCheck"
```

---

### Task 19: End-to-end ping_2pe.mlir simulator test (THE MILESTONE)

**Files:**
- Test: `mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe.mlir`
- Modify: `utils/run_csl_ci.sh` (add to e2e test list)

- [ ] **Step 1: Write `ping_2pe.mlir`**

Use `mlir/test/Targets/CSLEmit/e2e/scientific/saxpy.mlir` as the canonical reference for the host-runtime stub (memcpy_h2d, launch, memcpy_d2h, run.py shape).

```mlir
// RUN: air-opt --csl-pipeline %s | air-translate --emit-csl -o %t-dir
// RUN: cd %t-dir && cslc layout.csl --arch=wse3 --fabric-dims=11,3 \
//          --fabric-offsets=4,1 --memcpy --params=width:2 -o out
// RUN: cd %t-dir && cs_python run.py --name out 2>&1 | FileCheck %s
// CHECK: SUCCESS

csl.wafer @ping_2pe {arch = "wse3"} {
  csl.program @left_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.stream.put @send_ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf
    csl.export @compute {kind = "func"}
  }
  csl.program @right_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.stream.get @send_ch target(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.stream @send_ch from (0, 0) to (1, 0)
    csl_layout.place  @left_pe  at (0, 0)
    csl_layout.place  @right_pe at (1, 0)
  }
  csl.host @main(%buf_in: memref<128xf32>, %buf_out: memref<128xf32>)
                {layout = @layout} {
    csl_host.memcpy_h2d %buf_in to @layout::@left_pe::@buf
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<128xf32>
    csl_host.launch @layout::@left_pe::@compute
                    {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
    csl_host.launch @layout::@right_pe::@compute
                    {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
    csl_host.memcpy_d2h @layout::@right_pe::@buf to %buf_out
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<128xf32>
  }
}
```

The exact `csl.host` shape and `cslc` command-line flags should match what's used in `saxpy.mlir`. Check that file and adapt.

`run.py` is auto-generated from the `csl.host` block; its body needs to compare `buf_in` and `buf_out` and print SUCCESS if equal. The existing emitter for `csl.host` produces the comparison logic from the `csl_host.memcpy_d2h` + a final assert; verify the contract by reading existing `run.py` outputs from saxpy or related tests.

- [ ] **Step 2: Run — likely FAIL on first try**

```bash
lit -v mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe.mlir
```

Common failure modes and their diagnoses:

| Failure | Likely cause |
|---|---|
| `air-opt` fails | Pass-pipeline invariant broken; run individual passes to localize. |
| `cslc` fails | Emitted CSL is malformed; inspect `%t-dir/layout.csl` and PE files. |
| `cs_python` fails | Host-runtime contract mismatch; the launch/memcpy ROI is wrong. |
| `cs_python` succeeds but no SUCCESS | Data didn't transfer; check fabric-dims and the WSE simulator routing. |
| Hangs / deadlock | Async builtin call missing `.async = true`; or `unblock_cmd_stream` not firing. |

Iterate. The likely fixes are emitter-side or host-stub-side, not pass-side (the FileCheck tests guard pass behavior).

- [ ] **Step 3: Add to `utils/run_csl_ci.sh`'s e2e test list**

In the script, locate the array of e2e test files (search for `e2e/scientific` references) and append:

```bash
ping_2pe="mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe.mlir"
e2e_tests+=("$ping_2pe")
```

(Adapt to the script's existing pattern.)

- [ ] **Step 4: Run full simulator suite**

```bash
bash utils/run_csl_ci.sh
```

Expected: all green, including ping_2pe.

- [ ] **Step 5: Commit**

```bash
git add mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe.mlir \
        utils/run_csl_ci.sh
git commit -m "test(csl-e2e): ping_2pe — 2-PE inter-PE dataflow runs SUCCESS! on WSE-3 sim"
```

This is the **milestone commit**. Stop here and ask the user to confirm before continuing to Phase G.

---

## Phase G — Variant tests

After ping is green and the user confirms, add variants to broaden coverage.

---

### Task 20: Directional variants

**Files:**
- Tests: `ping_2pe_vertical.mlir`, `ping_2pe_west.mlir`, `ping_2pe_north.mlir`

Each is a copy of `ping_2pe.mlir` with the stream's `from`/`to` coords swapped or rotated, and the placement / launch ROI / memcpy ROI updated to match.

- [ ] **Step 1: Write `ping_2pe_vertical.mlir`**

Like `ping_2pe.mlir` but stream `from(0,0) to(0,1)`, layout `width=1 height=2`, places `@left_pe at (0,0)` and `@right_pe at (0,1)`.

- [ ] **Step 2: Run — expect SUCCESS! on simulator**

```bash
lit -v mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe_vertical.mlir
```

- [ ] **Step 3: Write `ping_2pe_west.mlir`**

Stream `from(1,0) to(0,0)`. Renames `@left_pe` → `@right_pe` and vice versa, or just keeps names and reverses placement. Sender is now at (1,0); host h2d targets (1,0); host d2h reads from (0,0).

- [ ] **Step 4: Run — expect SUCCESS!**

- [ ] **Step 5: Write `ping_2pe_north.mlir`**

Stream `from(0,1) to(0,0)`. Vertical-reverse.

- [ ] **Step 6: Run — expect SUCCESS!**

- [ ] **Step 7: Add all three to `utils/run_csl_ci.sh`**

- [ ] **Step 8: Run full suite**

```bash
bash utils/run_csl_ci.sh
```

- [ ] **Step 9: Commit**

```bash
git add mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe_{vertical,west,north}.mlir \
        utils/run_csl_ci.sh
git commit -m "test(csl-e2e): directional variants of ping_2pe (vertical/west/north)"
```

---

### Task 21: Extents and multi-stream variants

- [ ] **Step 1: Write `ping_2pe_extents.mlir`**

Use lit `-D` substitution (or just hardcode 4 separate kernels) for extents 16, 64, 256, 1024.

Simplest: 4 sub-tests in one `.mlir` via `// -----`, each with a different `memref<NxF32>`. Each stamps its own SUCCESS! check.

- [ ] **Step 2: Run — expect SUCCESS! for all extents**

- [ ] **Step 3: Write `ping_2pe_two_streams.mlir`**

Two streams between (0,0) and (1,0) on different colors. The user IR has two `csl_layout.stream` ops; allocator gives them ids 0 and 1 distinct; both PEs run two put/gets. Verify both buffers transfer correctly.

- [ ] **Step 4: Run — expect SUCCESS!**

- [ ] **Step 5: Add to `utils/run_csl_ci.sh`**

- [ ] **Step 6: Run full suite**

- [ ] **Step 7: Commit**

```bash
git add mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe_{extents,two_streams}.mlir \
        utils/run_csl_ci.sh
git commit -m "test(csl-e2e): extents (16-1024) and two-stream variants of ping_2pe"
```

---

### Task 22: 3-PE chain variant (`ping_3pe_chain.mlir`)

Tests one PE having both ends of two streams (left → middle → right).

- [ ] **Step 1: Write `ping_3pe_chain.mlir`**

```mlir
// (skeleton — fill in host stub from saxpy.mlir as before)

csl.wafer @ping_3pe {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @compute {
      %n = arith.constant 32 : index
      csl.stream.put @ab source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf
    csl.export @compute {kind = "func"}
  }
  csl.program @middle {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @compute {
      %n = arith.constant 32 : index
      csl.stream.get @ab target(%buf) extent(%n : index) : memref<32xf32>
      csl.stream.put @bc source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf
    csl.export @compute {kind = "func"}
  }
  csl.program @right {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @compute {
      %n = arith.constant 32 : index
      csl.stream.get @bc target(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
    csl.export @buf
    csl.export @compute {kind = "func"}
  }
  csl.layout {width = 3 : i64, height = 1 : i64} @layout {
    csl_layout.stream @ab from(0, 0) to(1, 0)
    csl_layout.stream @bc from(1, 0) to(2, 0)
    csl_layout.place  @left   at (0, 0)
    csl_layout.place  @middle at (1, 0)
    csl_layout.place  @right  at (2, 0)
  }
  csl.host @main(...) { ... copy buf to left; launch all 3; copy buf from right ... }
}
```

- [ ] **Step 2: Run — expect SUCCESS!**

```bash
lit -v mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_3pe_chain.mlir
```

If middle PE deadlocks (tasks fire in wrong order), it likely means `unblock_cmd_stream` needs to fire only after *both* the get and put complete. Mitigation: middle's compute function expects the get's task to fire *before* the put starts (semantic is sync-looking IR ordering). If this surfaces, the lowering may need a small extension to chain task activation; document and decide.

- [ ] **Step 3: Add to `utils/run_csl_ci.sh`**

- [ ] **Step 4: Run full suite**

- [ ] **Step 5: Commit**

```bash
git add mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_3pe_chain.mlir \
        utils/run_csl_ci.sh
git commit -m "test(csl-e2e): 3-PE linear chain (left → middle → right)"
```

---

## Done

After Task 22, the milestone is complete: ping kernel + 7 variant e2e tests, ~20 FileCheck per-pass tests, ~10 dialect roundtrip + verifier tests, all green on the WSE-3 simulator. Tutorial-6 GEMV is the natural follow-up; the user surface here absorbs it without further dialect changes (only an op-or-attr extension on `csl.stream.get` for recv-with-`@fadds` accumulate).

Spec: `docs/superpowers/specs/2026-05-01-csl-multi-pe-streams-design.md`.

End of plan.
