# Multi-PE Dataflow via CSL Streams — Design

**Date:** 2026-05-01
**Status:** Design (approved by user; ready for implementation plan)
**Branch:** `air-to-fire`
**Milestone:** A 2-PE "ping" kernel runs end-to-end on the WSE-3 simulator, exercising inter-PE colors/routing/fabric-DSDs. Tutorial-6 GEMV is **deferred** until ping is green.

---

## 1. Motivation and goal

Today the project's CSL stack runs single-PE kernels (`saxpy`, `fadds`,
`stencil_fadds`, …) and N-PE *replicated* kernels (`sharding.mlir`,
`layouts.mlir` — tutorial 5's pattern: each PE runs identical work, no
inter-PE communication). The next capability the project needs is
**multi-PE *dataflow*** — PE A computes some data and *sends it* to PE
B over the WSE fabric, B receives and consumes it. This is the
foundation required for tutorial-6 GEMV, all SPADA collectives
(`broadcast_1D`, `tree_reduce_1D`, `chain_reduce`, …), and the
longer-term lowering of `air.channel.put`/`get` from the AIR layer.

The Cerebras tutorial 6 reference implementation
(`docs/superpowers/raw/cerebras_sdk_docs/csl/tutorials/gemv-06-routes-1/`)
exposes the underlying machinery directly: colors, per-PE
`@set_color_config`, `fabin_dsd`/`fabout_dsd`, `local_task_id`, async
fabric ops, `@bind_local_task`. A literal port of that surface to MLIR
would force the test author to write all of those primitives by hand —
a poor user model and a poor lowering target for AIR.

**Goal of this milestone:** ship a *thin user-facing surface* — three
ops (`csl_layout.stream`, `csl.stream.put`, `csl.stream.get`) that
hide colors, routing, fabric DSDs, and async/task plumbing behind a
**4-pass progressive lowering pipeline**. Each pass is single-purpose,
adds one layer of hardware detail, and is independently testable. By
the end of the pipeline the IR is **isomorphic to CSL text** and the
emitter is a pure printer.

**User writes** (full multi-PE dataflow in 4 lines of layout-block
content):

```mlir
csl.layout {width = 2, height = 1} @layout {
  csl_layout.stream @send_ch from (0, 0) to (1, 0)
  csl_layout.place  @left_pe  at (0, 0)
  csl_layout.place  @right_pe at (1, 0)
}
// In csl.program @left_pe / @right_pe bodies:
csl.stream.put @send_ch source(%buf) extent(128 : index) : memref<128xf32>
csl.stream.get @send_ch target(%buf) extent(128 : index) : memref<128xf32>
```

No colors, no routes, no tasks, no async, no fabric DSDs in the user
IR. Everything is synthesized by passes.

**Concrete deliverable:** an end-to-end test
`mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe.mlir` plus its
supporting lit infrastructure that:

1. Lowers via `air-opt --csl-pipeline` (the registered top-level pipeline).
2. Translates via `air-translate --emit-csl` to `layout.csl` +
   `left_pe.csl` + `right_pe.csl` + `run.py`.
3. Compiles via `cslc` and runs via `cs_python` on the WSE-3 simulator.
4. Prints `SUCCESS!` (right PE's buffer matches what left PE sent).

Plus comprehensive surrounding tests: dialect roundtrip, verifier
negative tests, per-pass FileCheck (one per pass invariant), and the
simulator e2e above. The user requirement was explicit: "have a lot of
tests, make them run on simulator and should be all green".

## 2. Scope

### In scope

- **Three user-facing ops:** `csl_layout.stream`, `csl.stream.put`,
  `csl.stream.get`.
- **Two internal ops** (synthesized by passes, not written by users):
  `csl_layout.set_color_config`, `csl.get_fab_dsd`.
- **Two op edits**: `csl.task` (attribute-driven trigger binding;
  collapses three concepts into one op), `csl.builtin_call`
  (`async`/`activate` attrs).
- **`csl.color` scope move:** from `csl.program`-body to
  `csl.layout`-body. Synthesized by Pass 1; never written by the user.
- **Four new passes**, each single-purpose:
  1. `--csl-materialize-stream-colors` — synthesize `csl.color` per stream
  2. `--csl-allocate-color-ids` — assign integer ids
  3. `--csl-lower-stream-routing` — emit `set_color_config`
  4. `--csl-lower-stream-data` — expand put/get to fabric DSDs + tasks
- **Two registered pass pipelines:**
  - `csl-streams-to-csl` — the 4 passes above as one flag
  - `csl-pipeline` — top-level frontend chain (`infer-exports` →
    `auto-vectorize` → `streams-to-csl`); future passes slot in here
- **Text emitter** (`Targets/CSLEmit/`) cases for every new internal op.
- **Tests:** dialect roundtrip, verifier negative, per-pass FileCheck,
  end-to-end simulator. The 2-PE ping kernel as the pinnacle e2e.

### Out of scope (deferred)

- **Tutorial-6 GEMV** (column-major A, local matvec, recv-with-`@fadds`,
  fork-join overlap). Held until ping is green; will reuse this surface.
- **Multi-hop routing.** Single-hop (one EAST/SOUTH/WEST/NORTH step) only.
- **Multicast streams** (`relative_stream([a:b], dy)` analogue). Single-target only.
- **Async tokens on stream ops.** User-facing put/get look synchronous
  (program-order). The async + task-id machinery exists in the
  *expansion*, not in the user IR. Tutorial-6 GEMV may need user-IR
  async tokens; we add them then.
- **Real liveness-driven graph-coloring color allocator.**
  `--csl-allocate-color-ids` ships as a monotonic stub. Real allocation
  over the WSE-3 24-color budget is a follow-up; this spec defines the
  *home* for it but not its algorithm.
- **AIR-level `air.channel`/`air.channel.put`/`air.channel.get`
  lowering** (`-air-to-csl` for inter-herd channels). Deferred per
  user's Q1-C answer; the user surface here is intentionally
  **isomorphic to `air.channel`** so the future lowering is 1:1.
- **Non-`f32` element types in put/get.** Same as the auto-vectorize
  spec defers them.

### Explicitly *not* a goal

- **User-controllable colors.** Users never write color names or ids.
  If a future use case demands explicit color sharing for performance,
  we add an opt-in `color_alias = @label` attr on the stream then.
- **SPADA's `relative_stream(dx, dy)` form.** Absolute
  `from(x,y) to(x,y)` reads better and matches AIE's `aie.flow` shape.

## 3. Design

### 3.1 Pipeline architecture

Four single-purpose passes, registered together as the
`csl-streams-to-csl` pipeline; bundled with frontend passes into the
top-level `csl-pipeline`.

```
USER INPUT  (Stage 0)         3 user-facing ops:
                                csl_layout.stream
                                csl.stream.put
                                csl.stream.get
   │
   │   --csl-materialize-stream-colors        (Pass 1)
   ▼
STAGE 1                       + virtual csl.color symbols (no ids)
                              + stream gains {color = @<sym>} attr
   │
   │   --csl-allocate-color-ids               (Pass 2)
   ▼
STAGE 2                       + integer ids on csl.color (id = 0, 1, …)
   │
   │   --csl-lower-stream-routing             (Pass 3)
   ▼
STAGE 3                       + csl_layout.set_color_config per endpoint
                                (rx/tx inferred from coord delta)
   │
   │   --csl-lower-stream-data                (Pass 4)
   ▼
STAGE 4 (emit-ready)          stream/put/get gone; replaced by:
                                csl.get_fab_dsd (in/out)
                                csl.task (attr-driven trigger)
                                csl.builtin_call {async, activate=@t}
   │
   │   air-translate --emit-csl
   ▼
TEXT (CSL source)             Emitter is a pure printer (1:1 mapping).
```

### 3.2 IR snapshot per stage (ping kernel example)

**Stage 0 — user writes:**

```mlir
csl.wafer @ping_2pe {arch = "wse3"} {
  csl.program @left_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      csl.stream.put @send_ch source(%buf) extent(128 : index)
        : memref<128xf32>
      csl.return
    }
    csl.export @buf
    csl.export @compute {kind = "func"}
  }
  csl.program @right_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      csl.stream.get @send_ch target(%buf) extent(128 : index)
        : memref<128xf32>
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
  csl.host @main(...) { ... }
}
```

**After Pass 1 (`--csl-materialize-stream-colors`):**

```mlir
csl.layout {...} @layout {
  csl.color @send_ch_color : !csl.color                              // ← NEW (virtual)
  csl_layout.stream @send_ch from (0,0) to (1,0)
                    {color = @send_ch_color}                          // ← NEW attr
  csl_layout.place  @left_pe  at (0, 0)
  csl_layout.place  @right_pe at (1, 0)
}
// Programs unchanged.
```

**After Pass 2 (`--csl-allocate-color-ids`):**

```mlir
csl.color @send_ch_color {id = 0 : i32} : !csl.color                  // ← id assigned
// Everything else unchanged.
```

**After Pass 3 (`--csl-lower-stream-routing`):**

```mlir
csl.layout {...} @layout {
  csl.color @send_ch_color {id = 0 : i32}
  csl_layout.stream @send_ch from (0,0) to (1,0) {color = @send_ch_color}
  csl_layout.set_color_config @send_ch_color at (0,0) rx(RAMP) tx(EAST)   // ← NEW
  csl_layout.set_color_config @send_ch_color at (1,0) rx(WEST) tx(RAMP)   // ← NEW
  csl_layout.place @left_pe  at (0, 0)
  csl_layout.place @right_pe at (1, 0)
}
// Programs still contain stream.put / stream.get.
```

**After Pass 4 (`--csl-lower-stream-data`):**

```mlir
csl.layout {...} @layout {
  csl.color @send_ch_color {id = 0 : i32}
  csl_layout.set_color_config @send_ch_color at (0,0) rx(RAMP) tx(EAST)
  csl_layout.set_color_config @send_ch_color at (1,0) rx(WEST) tx(RAMP)
  csl_layout.place @left_pe  at (0, 0)
  csl_layout.place @right_pe at (1, 0)
  // csl_layout.stream gone.
}
csl.program @left_pe {
  %buf = csl.var @buf : memref<128xf32>
  csl.func @compute {
    %src = csl.get_mem_dsd %buf : memref<128xf32> -> !csl.dsd
    %c128 = arith.constant 128 : index
    %out = csl.get_fab_dsd fabout @send_ch_color extent(%c128 : index) : !csl.dsd
    csl.task @send_ch_put_done {trigger_kind = "local_task_id",
                                 id = 8 : i32} {
      csl.builtin_call "unblock_cmd_stream"() : () -> ()
      csl.return
    }
    csl.builtin_call "fmovs"(%out, %src)
                     {async, activate = @send_ch_put_done}
                     : (!csl.dsd, !csl.dsd) -> ()
    csl.return
  }
}
csl.program @right_pe { /* mirror with fabin + target buf */ }
```

Stage 4 is **isomorphic to the CSL text**. Each MLIR op corresponds to
exactly one CSL line.

### 3.3 User-facing ops

#### `csl_layout.stream`

```
csl_layout.stream @ch from (px : i64, py : i64) to (px : i64, py : i64) attr-dict
```

- **Parent:** `csl.layout` body.
- **Symbol:** `sym_name`; references like `csl.stream.put @ch` resolve to it.
- **Operands:** none (coords are constant attributes).
- **Attributes (user writes):** `from_x`, `from_y`, `to_x`, `to_y` —
  all `i64`. **No `color` attribute** — Pass 1 adds it.
- **Verifier:**
  - Coords must be within the wafer's `(width, height)`.
  - For ping milestone: `(to_x - from_x, to_y - from_y)` must be one
    of `(±1, 0)` or `(0, ±1)` — single-hop along a cardinal direction.
    Multi-hop verifier rejection is the gate that prevents accidentally
    writing routes the lowering can't generate yet.

#### `csl.stream.put`

```
csl.stream.put @ch source(%mem : memref<...>) extent(%n : index) : <type>
```

- **Parent:** a `csl.func` body inside `csl.program`.
- **Operands:** `%mem` (memref source buffer), `%n` (index extent).
- **Attribute:** `stream`: `FlatSymbolRefAttr` to a `csl_layout.stream`.
- **Result:** none (sync-looking; tokens deferred to tutorial-6 milestone).
- **Verifier:**
  - Stream symbol resolves to a `csl_layout.stream`; `from` coord
    matches the parent program's placement (cross-checked at Pass 4
    time, since placement is in `csl.layout`).
  - `%mem`'s element type is `f32` (others deferred).
  - Extent is a constant or compile-time-resolvable index value.

#### `csl.stream.get`

```
csl.stream.get @ch target(%mem : memref<...>) extent(%n : index) : <type>
```

Mirror of `csl.stream.put`. Verifier mirrors, with the parent program's
placement matching the stream's `to` coord.

### 3.4 Internal ops (synthesized by passes)

#### `csl.color` (existing op, scope moved to `csl.layout`)

Already declared in `CSLOps.td:38`. The op shape stays the same:

```mlir
csl.color @send_ch_color : !csl.color                  // virtual (no id)
csl.color @send_ch_color {id = 0 : i32} : !csl.color   // pinned (post Pass 2)
```

**Scope change:** today the op lives inside `csl.program` body. The new
home is `csl.layout` body, so a single color symbol can be referenced
from both PE programs and from `csl_layout.set_color_config`. Symbol
resolution from inside `csl.program` bodies — needed by
`csl.get_fab_dsd @<color>` — uses the same cross-region SymbolTable
lookup that `csl_layout.place params {p = @sym}` already uses. No new
infrastructure.

This is a **breaking change** for `csl.color` placement, but per the
SPADA gap analysis no working e2e test currently uses `csl.color` —
only the dialect roundtrip tests in `mlir/test/Dialect/CSL/`. Those
will be updated in lockstep.

`csl.color` is **synthesized** by Pass 1 from each `csl_layout.stream`;
users never write it directly.

#### `csl.get_fab_dsd` (new)

```
%out = csl.get_fab_dsd fabout @color_sym extent(%n : index) : !csl.dsd
%in  = csl.get_fab_dsd fabin  @color_sym extent(%n : index) : !csl.dsd
```

- **Parent:** a `csl.func` body inside `csl.program`.
- **Operands:** `%n` (index extent, SSA value).
- **Attribute:** `color`: `FlatSymbolRefAttr` referencing a `csl.color`
  in the enclosing `csl.layout` body. Cross-region resolution via
  SymbolTable lookup.
- **Direction:** keyword `fabin` | `fabout`, lowered to existing
  `DsdKind` enum values (already defined in `CSLBase.td:108-113`).
- **Emit:**
  ```csl
  const out_dsd = @get_dsd(fabout_dsd,
      .{ .extent = 128, .fabric_color = send_ch_color });
  ```

Synthesized by Pass 4 from each `csl.stream.put`/`get`.

#### `csl_layout.set_color_config` (new)

```
csl_layout.set_color_config @color_sym at (px : i64, py : i64)
                            rx(<dir>) tx(<dir>)
```

- **Parent:** `csl.layout` body.
- **Operands:** none.
- **Attributes:** `color`: `FlatSymbolRefAttr`; `px`/`py`: `i64`;
  `rx`/`tx`: existing `Direction` enum (`CSLBase.td:102`).
- **Emit:**
  ```csl
  @set_color_config(0, 0, send_ch_color,
      .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
  ```

Single-direction `rx` and `tx` only. Multi-direction sets
(`.tx = .{EAST, WEST}` for fan-out) deferred. Synthesized by Pass 3.

### 3.5 Op edits

#### `csl.task` — collapse to attribute-driven trigger

Today: `csl.task @t color(%c : !csl.color) { body }` (color is an SSA
operand).

**New shape:** drop the operand; trigger is an attribute. Two mutually
exclusive optional attrs (verifier: exactly one set):

```mlir
// Local-task-triggered (used by Pass 4)
csl.task @put_done {trigger_kind = "local_task_id", id = 8 : i32} {
  csl.builtin_call "unblock_cmd_stream"() : () -> ()
  csl.return
}

// Color-triggered (future use, e.g., wavelet-driven foreach)
csl.task @recv {trigger_kind = "color", color = @send_color} {
  ...
}
```

**TableGen:**
- Drop the existing `CSL_ColorType:$color` operand.
- Add `StrAttr:$trigger_kind` (must be one of `"local_task_id"` or `"color"`).
- Add `OptionalAttr<I32Attr>:$id` (required when `trigger_kind = "local_task_id"`).
- Add `OptionalAttr<FlatSymbolRefAttr>:$color` (required when
  `trigger_kind = "color"`).
- Verifier: exactly one of `id` / `color` is set, matching `trigger_kind`.

**Emit (single op produces three CSL lines):**

```csl
const put_done_id: local_task_id = @get_local_task_id(8);
task put_done() void { … body … }
comptime { @bind_local_task(put_done, put_done_id); }
```

The `local_task_id` symbol is *implicit* — emitted from the task's own
`sym_name` plus an `_id` suffix. No separate dialect op for the id
symbol; no `!csl.local_task_id` type. **Net deletion: 1 type, 1 op,
the operand.**

#### `csl.builtin_call` — `async` + `activate` attrs

Two new optional attributes:

| Attribute | Type | Effect |
|---|---|---|
| `async` | `UnitAttr` | Emits `.async = true` field. |
| `activate` | optional `FlatSymbolRefAttr` | Emits `.activate = <task_id_sym>` where `<task_id_sym>` is the *implicit* task-id symbol of the referenced `csl.task` (i.e., `<task_sym>_id`). |

**Verifier:** if `activate` is set, `async` must also be set (the
tutorial deadlock warning: fabric DSDs sync = bad). If `activate` is
set, the referenced `csl.task` must have `trigger_kind =
"local_task_id"`.

**Emit:**

```csl
@fmovs(out_dsd, src_dsd, .{ .async = true, .activate = put_done_id });
```

### 3.6 Pass — `--csl-materialize-stream-colors` (Pass 1)

**Pre-condition:** every `csl_layout.stream` op has *no* `color`
attribute.
**Post-condition:** every `csl_layout.stream` has `{color = @<sym>}`;
a matching `csl.color @<sym> : !csl.color` (no id) exists in the same
`csl.layout` body.

**Algorithm:**

1. For each `csl_layout.stream` in the module, generate a fresh symbol
   name `<stream_sym>_color` (e.g., `@send_ch_color`).
2. Insert a `csl.color @<stream_sym>_color : !csl.color` op at the top
   of the enclosing `csl.layout` body.
3. Set `{color = @<stream_sym>_color}` on the stream op.

Idempotent: streams that already have a `color` attribute are skipped.

### 3.7 Pass — `--csl-allocate-color-ids` (Pass 2)

**Pre-condition:** every `csl.color` exists (with or without `id`).
**Post-condition:** every `csl.color` has an `id` attribute.

**Algorithm (milestone — stub):** monotonic. Walk the module's
`csl.color` ops in declaration order. For each color *without* an `id`
attribute, assign `id = next_unused`, where `next_unused` starts at 0
and skips any ids that appear on already-pinned colors. Pre-pinned
colors (`csl.color @x {id = N}`) are left untouched.

**Future work (out of scope):** liveness-driven graph coloring over
the 24-color budget, with task-overlap as edges. This pass is the
home; algorithms swap inside it without touching dialect or kernels.

### 3.8 Pass — `--csl-lower-stream-routing` (Pass 3)

**Pre-condition:** every `csl_layout.stream` has `{color = @<x>}`;
matching color has `id`.
**Post-condition:** for every stream, two `csl_layout.set_color_config`
ops exist (one at `from` coord, one at `to` coord); stream op is
**still present** (still needed by `csl.stream.put`/`get` symbol resolution).

**Algorithm:**

For each `csl_layout.stream @ch from(x1,y1) to(x2,y2) {color = @c}`:

1. Compute direction from coord delta:
   - `(+1, 0)` → src `tx = EAST`, dst `rx = WEST`.
   - `(-1, 0)` → src `tx = WEST`, dst `rx = EAST`.
   - `(0, +1)` → src `tx = SOUTH`, dst `rx = NORTH`.
   - `(0, -1)` → src `tx = NORTH`, dst `rx = SOUTH`.
2. Insert into `csl.layout` body:
   ```mlir
   csl_layout.set_color_config @c at(x1,y1) rx(RAMP) tx(<src_tx>)
   csl_layout.set_color_config @c at(x2,y2) rx(<dst_rx>) tx(RAMP)
   ```

### 3.9 Pass — `--csl-lower-stream-data` (Pass 4)

**Pre-condition:** Stage 3 form (set_color_configs in place; streams
still present; put/get still present).
**Post-condition:** no `csl.stream.put`, `csl.stream.get`, or
`csl_layout.stream` remain. Each program contains the matching fabric
DSDs + task + async builtin.

**Algorithm — per `csl.stream.put @ch source(%buf) extent(%n) : memref<NxT>` inside `csl.program @P`:**

Resolve `@ch` to its `csl_layout.stream`; get its color `@c`. Then in
`@P`'s body, in order:

1. `%src = csl.get_mem_dsd %buf : memref<NxT> -> !csl.dsd`.
2. `%out = csl.get_fab_dsd fabout @c extent(%n) : !csl.dsd`.
3. Allocate a fresh local task id integer per put (per-program
   monotonic counter starting at base `8`, matching the tutorial).
4. Emit a `csl.task @<stream_sym>_put_done_<n>` with attrs
   `{trigger_kind = "local_task_id", id = <fresh_int>}` and a body of
   `csl.builtin_call "unblock_cmd_stream" () : () -> () ; csl.return`.
5. Emit `csl.builtin_call "fmovs"(%out, %src) {async, activate =
   @<stream_sym>_put_done_<n>} : (!csl.dsd, !csl.dsd) -> ()`.
6. Erase the `csl.stream.put` op.

**Per `csl.stream.get`:** mirror — `fabin` instead of `fabout`,
`@<stream_sym>_get_done_<n>` task name, target memref's DSD as the
**destination** of `@fmovs` (so `@fmovs(target_dsd, in_dsd, …)`).

**After all put/get expanded:** erase all `csl_layout.stream` ops
(no more references).

The `unblock_cmd_stream` builtin is the existing CSL idiom for
unblocking the host memcpy command stream so subsequent `memcpy_d2h`
calls can proceed once the task fires. The emitter wires it through
an `@import_module("<memcpy/memcpy>")` import already present in
working saxpy.mlir.

### 3.10 Pass invariant table

For at-a-glance verification (also drives FileCheck per pass):

| Pass | Pre-condition | Post-condition |
|---|---|---|
| 1: `--csl-materialize-stream-colors` | every `csl_layout.stream` has *no* `color` attr | every stream has `{color = @<sym>}`; matching `csl.color @<sym>` exists in same `csl.layout` (no id) |
| 2: `--csl-allocate-color-ids` | every `csl.color` exists | every `csl.color` has an `id` attr |
| 3: `--csl-lower-stream-routing` | every stream has `{color = @<x>}`; matching color has `id` | for every stream, two `csl_layout.set_color_config` ops exist; stream op present |
| 4: `--csl-lower-stream-data` | Stage 3 form | no `csl.stream.put`/`get`/`csl_layout.stream` remain; each program has matching fabric DSDs + task + async builtin |

### 3.11 Pass pipeline registration

MLIR's `PassPipelineRegistration<>` mechanism (the same upstream uses
for `--test-lower-to-llvm` and similar) registers a named pipeline
that expands to a chain of passes.

**Two pipelines, layered:**

```cpp
// Lower-level: stream lowering only
mlir::PassPipelineRegistration<> cslStreamsToCSLPipeline(
    "csl-streams-to-csl",
    "Lower high-level CSL stream ops to emit-ready CSL dialect form "
    "(materialize colors → allocate ids → lower routing → lower data).",
    [](OpPassManager &pm) {
      pm.addPass(createCSLMaterializeStreamColorsPass());
      pm.addPass(createCSLAllocateColorIdsPass());
      pm.addPass(createCSLLowerStreamRoutingPass());
      pm.addPass(createCSLLowerStreamDataPass());
    });

// Top-level: full CSL frontend → emit-ready chain
mlir::PassPipelineRegistration<> cslPipeline(
    "csl-pipeline",
    "Full CSL pipeline: infer-exports → auto-vectorize → streams-to-csl. "
    "Produces emit-ready IR for air-translate --emit-csl.",
    [](OpPassManager &pm) {
      pm.addPass(createCSLInferExportsPass());
      pm.addPass(createCSLAutoVectorizePass());
      // Embed the lower-level pipeline:
      pm.addPass(createCSLMaterializeStreamColorsPass());
      pm.addPass(createCSLAllocateColorIdsPass());
      pm.addPass(createCSLLowerStreamRoutingPass());
      pm.addPass(createCSLLowerStreamDataPass());
    });
```

**Both forms work:**

```bash
# Pipeline shortcut — what e2e tests use
air-opt input.mlir --csl-pipeline

# Sub-pipeline alone
air-opt input.mlir --csl-streams-to-csl

# Individual passes — what per-pass FileCheck tests use
air-opt input.mlir --csl-materialize-stream-colors
air-opt input.mlir --csl-allocate-color-ids
# … etc.
```

### 3.12 Codebase layout

| File | Purpose |
|---|---|
| `mlir/include/air/Dialect/CSL/Pipelines/Pipelines.h` | Public header: `void registerCSLPipelines();` |
| `mlir/lib/Dialect/CSL/Pipelines/CSLStreamsPipeline.cpp` | Defines `csl-streams-to-csl` |
| `mlir/lib/Dialect/CSL/Pipelines/CSLPipeline.cpp` | Defines top-level `csl-pipeline` |
| `mlir/lib/Dialect/CSL/Pipelines/CMakeLists.txt` | Build glue |
| `mlir/lib/Dialect/CSL/Transforms/CSLMaterializeStreamColors.cpp` | Pass 1 |
| `mlir/lib/Dialect/CSL/Transforms/CSLAllocateColorIds.cpp` | Pass 2 |
| `mlir/lib/Dialect/CSL/Transforms/CSLLowerStreamRouting.cpp` | Pass 3 |
| `mlir/lib/Dialect/CSL/Transforms/CSLLowerStreamData.cpp` | Pass 4 |
| `mlir/include/air/Dialect/CSL/Transforms/Passes.td` | Pass declarations (`GEN_PASS_DEF_*`) |
| `tools/air-opt/air-opt.cpp` | One-line: `registerCSLPipelines();` at startup |

`Pipelines/` is a sibling of `Transforms/` and `IR/` — standard MLIR
project layout (mirrors how upstream organizes
`mlir/lib/Dialect/<X>/Pipelines/`).

### 3.13 Emitter additions

`mlir/lib/Targets/CSLEmit/` already has a structured emitter for the
existing CSL dialect. New emitter cases — by virtue of the Stage-4 IR
being CSL-isomorphic, every case is one MLIR op → one CSL line:

- `csl.get_fab_dsd fabout @c extent(N)` →
  `const <ssa>_dsd = @get_dsd(fabout_dsd, .{ .extent = N, .fabric_color = c });`
- `csl.get_fab_dsd fabin  @c extent(N)` → mirror with `fabin_dsd`.
- `csl_layout.set_color_config @c at(x,y) rx(R) tx(T)` → emitted in
  layout block as `@set_color_config(x, y, c, .{ .routes = .{ .rx =
  .{R}, .tx = .{T} } });`.
- `csl.task @t {trigger_kind = "local_task_id", id = N}` produces three lines:
  ```csl
  const t_id: local_task_id = @get_local_task_id(N);
  task t() void { … body … }
  comptime { @bind_local_task(t, t_id); }
  ```
- `csl.builtin_call …{async, activate = @x}` → existing emit path
  augmented with `.async = true, .activate = x_id` field syntax (note:
  `_id` suffix derived from the implicit local-task-id symbol).
- `csl.color @c {id = N}` (now in layout body) → `const c = @get_color(N);`
  emitted at top of `layout.csl`.

Existing emitter cases for single-PE kernels are not modified — no
breakage.

## 4. Test plan

User requirement: "have a lot of tests, make them run on simulator
and should be all green". Four layers, all required to land:

### 4.1 Dialect roundtrip tests — `mlir/test/Dialect/CSL/`

One `.mlir` per new op or edit, parse → print → parse-again equivalence:

- `roundtrip_stream.mlir` — `csl_layout.stream @s from(0,0) to(1,0)` (Stage 0 form).
- `roundtrip_stream_with_color.mlir` — `csl_layout.stream @s from(0,0) to(1,0) {color = @c}` (Stage 1+ form).
- `roundtrip_stream_put_get.mlir` — `csl.stream.put`/`get` with various memref shapes and extents.
- `roundtrip_get_fab_dsd.mlir` — both `fabin` and `fabout` forms.
- `roundtrip_set_color_config.mlir` — all 4 cardinal direction combinations.
- `roundtrip_task_local_id.mlir` — `csl.task @t {trigger_kind = "local_task_id", id = 8} { ... }`.
- `roundtrip_task_color.mlir` — `csl.task @t {trigger_kind = "color", color = @c} { ... }`.
- `roundtrip_builtin_async.mlir` — `csl.builtin_call …{async, activate = @x}`.
- `roundtrip_color_in_layout.mlir` — `csl.color` inside `csl.layout` body (the scope move).

### 4.2 Verifier negative tests — same directory

One per failure mode, `// RUN: not air-opt %s … 2>&1 | FileCheck %s`:

- Stream coords outside wafer rectangle.
- Stream non-cardinal / multi-hop offset (e.g., `from(0,0) to(2,3)`).
- `stream.put` memref element type that the lowering doesn't yet
  support (`i32`, `f16`, etc.) — clear diagnostic, not silent miscompile.
- `stream.put`/`get` parent program placement mismatch with stream's
  `from`/`to` coord.
- `csl.builtin_call {activate = @x}` without `async`.
- `csl.task` with both `id` and `color` set (or neither).
- `csl.task {trigger_kind = "local_task_id"}` but `color` set instead of `id`.
- `csl.color` outside `csl.layout` body (after scope move).
- `csl.builtin_call {activate = @x}` where `@x` is a `csl.task` with `trigger_kind = "color"` (only local_task_id-bound tasks are activatable).

### 4.3 Per-pass FileCheck tests — `mlir/test/Dialect/CSL/Transforms/`

One subdirectory per pass; each test asserts the pass's pre/post
invariant from §3.10.

#### Pass 1 — `--csl-materialize-stream-colors`

- `materialize_one_stream.mlir` — single stream → one virtual color synthesized; stream gains `{color = @<sym>}`.
- `materialize_two_streams.mlir` — two streams → two distinct virtual colors.
- `materialize_idempotent.mlir` — running twice is a no-op (streams already have color attr are skipped).
- `materialize_naming.mlir` — color symbol is `<stream>_color`.

#### Pass 2 — `--csl-allocate-color-ids`

- `allocate_one_color.mlir` — single virtual color → id 0.
- `allocate_three_colors.mlir` — three virtual colors → 0, 1, 2.
- `allocate_with_pin.mlir` — one color declared with `{id = 5}` (pre-pinned), one virtual → 0; the pinned one keeps 5.
- `allocate_skips_pinned_id.mlir` — virtual color exists, plus a pre-pinned `{id = 0}`; allocator must give virtual color id 1 (skipping taken slot 0).
- `allocate_idempotent.mlir` — running twice is a no-op.

#### Pass 3 — `--csl-lower-stream-routing`

- `lower_routing_east.mlir` — stream `(0,0)→(1,0)` → `tx=EAST, rx=WEST`.
- `lower_routing_west.mlir` — `(1,0)→(0,0)` → `tx=WEST, rx=EAST`.
- `lower_routing_south.mlir` — `(0,0)→(0,1)` → `tx=SOUTH, rx=NORTH`.
- `lower_routing_north.mlir` — `(0,1)→(0,0)` → `tx=NORTH, rx=SOUTH`.
- `lower_routing_keeps_stream.mlir` — stream op is *not* erased by Pass 3 (still needed by put/get resolution).

#### Pass 4 — `--csl-lower-stream-data`

- `lower_data_put.mlir` — put → `get_fab_dsd fabout` + task + async `fmovs`.
- `lower_data_get.mlir` — get → `get_fab_dsd fabin` + task + async `fmovs` with target as destination.
- `lower_data_erases_stream.mlir` — `csl_layout.stream` is gone after Pass 4.
- `lower_data_task_naming.mlir` — auto-generated tasks use the `<stream>_put_done_<n>` / `<stream>_get_done_<n>` pattern.
- `lower_data_unique_task_ids.mlir` — multiple put/gets in one program get distinct integer ids (8, 9, 10 …).

### 4.4 Pipeline FileCheck tests — `mlir/test/Dialect/CSL/Pipelines/`

End-to-end through the registered pipelines:

- `pipeline_streams_to_csl.mlir` — Stage-0 input + `--csl-streams-to-csl`
  → Stage-4 output. FileCheck asserts the full expansion.
- `pipeline_full.mlir` — Stage-0 input + `--csl-pipeline` (top-level) → Stage-4 output.

### 4.5 End-to-end emitter tests — `mlir/test/Targets/CSLEmit/`

Two tiers:

#### Tier 1 — `--emit-csl` text FileCheck (offline, fast)

`mlir/test/Targets/CSLEmit/multi_pe/`:

- `emit_set_color_config.mlir` — minimal layout exercising the new layout-block emission.
- `emit_local_task.mlir` — `csl.task` with `trigger_kind = "local_task_id"` produces the three-line CSL emission (id const, task body, comptime bind).
- `emit_fabric_dsd.mlir` — `csl.get_fab_dsd fabout` produces `@get_dsd(fabout_dsd, …)`.
- `emit_async_builtin.mlir` — `csl.builtin_call "fmovs"(...) {async, activate = @t}` produces `@fmovs(…, .{ .async = true, .activate = t_id });`.
- `emit_full_ping.mlir` — Stage-0 ping kernel through `--csl-pipeline` then `--emit-csl`. FileCheck asserts the full layout.csl + per-PE files.

#### Tier 2 — Simulator e2e (`utils/run_csl_ci.sh` integration)

`mlir/test/Targets/CSLEmit/e2e/multi_pe/`:

- `ping_2pe.mlir` — **the milestone**. 2 PEs side-by-side, left has buf seeded by host h2d, sends to right via stream, right stores into buf, host d2h reads right's buf and verifies elementwise match. `cs_python` exits 0 → SUCCESS.
- `ping_2pe_vertical.mlir` — same kernel rotated 90° (PE (0,0) → PE (0,1)) to exercise `tx=SOUTH/rx=NORTH`.
- `ping_2pe_west.mlir` — reversed direction (right sends to left).
- `ping_2pe_north.mlir` — vertical reversed.
- `ping_2pe_extents.mlir` — same shape with different `extent` values (16, 64, 256, 1024) parameterized via lit substitutions.
- `ping_2pe_two_streams.mlir` — two independent streams between the same PE pair (allocator gives them distinct colors); simultaneous dataflow edges.
- `ping_3pe_chain.mlir` — A→B→C linear chain; B does both a get and a put. Verifies one PE can host both ends of two streams.

Each e2e test is a single `.mlir` with a `// RUN:` line that drives
the full pipeline including `cs_python` invocation; the test passes
iff `cs_python` returns 0 *and* the simulator output indicates
correctness. The existing harness in `utils/run_csl_ci.sh` already
does this; new tests are added to its parallel-sim list.

### 4.6 CMake / lit / CI integration

- New lit subdirs added to `mlir/test/Dialect/CSL/CMakeLists.txt`,
  `mlir/test/Targets/CSLEmit/CMakeLists.txt`,
  `mlir/test/Dialect/CSL/Transforms/CMakeLists.txt` (new),
  `mlir/test/Dialect/CSL/Pipelines/CMakeLists.txt` (new).
- New build targets: `check-airmlir-csl-streams` (per-pass + pipeline tests), `check-airmlir-csl-multi-pe` (e2e simulator tests).
- `utils/run_csl_ci.sh`'s e2e simulator suite extended with the multi-PE tests; they participate in the pre-push hook (full CSL suite) and CI (lit-only — simulator is local-only by current policy).

## 5. Differentiation from related work

This design intentionally diverges from the three closest reference
points. Documented here so reviewers and downstream readers can
evaluate the trade-offs.

| Property | SPADA | AIR `air.channel` | AIE `aie.flow` | This design |
|---|---|---|---|---|
| Typed streams | textual at SPADA-IR | typeless | typeless (objectfifo is typed) | typeless (right per CSL semantics) |
| Color/channel allocator | global rename, single pass | n/a (lowered later) | route-finder (placement, not coloring) | **separate pass; home for liveness-driven RA over 24-color budget** |
| User writes color ids? | no (virtual) | n/a | no (route-finder picks) | **no — never written; not even symbolic at user level** |
| Routing inferred from coords? | yes (single-hop) | yes (in lowering) | yes (full route-finder) | yes (single-hop, cardinal only this milestone) |
| Async semantics in user IR | textual (`completion`/`await`) | yes (`!air.async.token`) | yes (objectfifo acquire/release) | **no in this milestone, deferred** |
| Lowering layered into multiple passes | one big pass | n/a | route-finding is one pass | **4 single-purpose passes; emit-ready IR is CSL-isomorphic** |
| AIR-lowering target shape | n/a | n/a | direct AIE | **isomorphic to `air.channel` for future lowering** |
| Hardware concept alignment | CSL-native | abstract | AIE-native | **CSL-native, AIE-aligned at high layer** |

**The novel combination:** thin user surface (3 ops, 0 colors, 0
tasks) + 4-pass progressive lowering with CSL-isomorphic Stage-4 +
register-allocator-as-pass over 24 physical colors + AIR-isomorphic
shape for future `air.channel` lowering. None of the three references
has all of these. The progressive-lowering shape, in particular, is
deliberately layered so each pass is independently testable (4
invariants → 4 FileCheck suites) — SPADA's `_collect_colors_globally`
is one monolithic Python function; ours is 4 small MLIR passes plus a
pipeline registration.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Direction inference for non-cardinal hops produces wrong routes silently. | Verifier on `csl_layout.stream` rejects non-cardinal coord deltas. Future multi-hop arrives with its own verification gate. |
| Color-symbol scope move breaks existing tests. | No working e2e currently uses `csl.color` per the SPADA gap analysis. Roundtrip tests update in lockstep. Failure mode is loud (verifier reject), not silent. |
| `csl.task` operand → attribute change breaks existing tests/users. | `csl.task` has no working e2e per the SPADA gap analysis (op skeleton only). Internal change touches only the dialect TableGen + builders + verifier; new shape lands as the only shape. |
| Auto-generated local task ids collide across multiple stream puts/gets in one program. | Per-program monotonic counter, prefix with the stream symbol. Verifier on `csl.task` enforces unique ids within a `csl.program`. |
| `unblock_cmd_stream` lowering doesn't match the actual host memcpy contract. | Existing saxpy.mlir's `cs_python` works because it manually calls `unblock_cmd_stream` from a host-launched function. Mirror that contract; the launch path is unchanged. |
| Simulator e2e is slow / fragile / local-only. | Already true today and handled: `run_csl_ci.sh` runs simulator tests in parallel; CI runs lit-only; pre-push hook runs the full simulator suite. New tests follow the same pattern. |
| 4 passes is more surface to land than one mega-pass. | Each pass is small (~50–150 LOC). Passes 1 and 2 land first (no behavioural change to Stage 4 — they prepare data). Passes 3 and 4 land after, gated by their FileCheck invariants. The pipeline registration is a one-time ~30-LOC commit. Stages can be stacked across PRs. |

## 7. Forward references

- **Tutorial-6 GEMV** will reuse this entire surface: same stream
  declaration, same put/get, plus a `csl.stream.get` variant or attr
  that lowers to recv-with-`@fadds` (accumulate) instead of `@fmovs`
  (move). Async tokens will be added to put/get at that time so the
  sender's `gemv` work overlaps with the send.
- **AIR-level lowering (`-air-to-csl` for inter-herd channels)** will
  be a new pass mapping `air.channel` 1:1 to `csl_layout.stream` and
  `air.channel.put`/`get` 1:1 to `csl.stream.put`/`get`. Because the
  surface here is intentionally isomorphic to AIR's channel surface,
  the lowering is structural — no new analysis. Cross-reference:
  AIE's `aie.flow` shape at
  `../mlir-aie/include/aie/Dialect/AIE/IR/AIEOps.td` (the high-level
  flow op our `csl_layout.stream` is modelled after) and
  `aie.connect` inside `aie.switchbox` (the per-tile routing
  primitive analogue of our `csl_layout.set_color_config`).
- **Real color allocator (graph-coloring over 24-color budget)** slots
  into `--csl-allocate-color-ids` without changes to dialect or
  kernel code. Liveness comes from the task DAG; constraint edges come
  from temporally overlapping streams. SPADA's approach is the design
  baseline; we improve on it by making it a first-class MLIR pass.
- **Future passes slot into `--csl-pipeline`** without test rewrites.
  Candidates: `--csl-task-recycling` (SPADA's color-budget compaction
  ported), `--csl-checkerboard-split` (collective routing
  conflict-free decomposition), `--csl-cycle-counter-instrument`
  (benchmarking).

## 8. Open questions

None remaining for this milestone. Q1–Q5 (front door, scope, program
structure, sugar layer, color allocation) plus the simplification
round (drop `!csl.local_task_id`, drop user-facing `csl.color`,
attribute-driven `csl.task`, 4-pass split, pipeline registration) are
all resolved. The implementation plan (next document) sequences the
work into stages.

## 9. References

### This repository
- CSL ops: `mlir/include/air/Dialect/CSL/CSLOps.td`
- CSL layout ops: `mlir/include/air/Dialect/CSL/CSLLayoutOps.td`
- CSL types: `mlir/include/air/Dialect/CSL/CSLBase.td`
- AIR ops: `mlir/include/air/Dialect/AIR/AIR.td`
- Existing CSL emitter: `mlir/lib/Targets/CSLEmit/`
- Existing CSL e2e: `mlir/test/Targets/CSLEmit/e2e/scientific/saxpy.mlir`
  (canonical shape; layout, program, host).
- Existing multi-PE replicated tests: `mlir/test/Targets/CSLEmit/e2e/{sharding,layouts}.mlir`.
- SPADA gap analysis: `docs/superpowers/specs/2026-04-25-spada-gap-analysis.md`.
- Auto-vectorize spec: `docs/superpowers/specs/2026-04-21-csl-auto-vectorize-design.md`.
- CSL CI / simulator harness: `utils/run_csl_ci.sh`.

### External
- Cerebras tutorial 6 (routes & fabric DSDs):
  `docs/superpowers/raw/cerebras_sdk_docs/csl/tutorials/gemv-06-routes-1/`
- Cerebras tutorial 5 (multiple PEs, replication):
  `docs/superpowers/raw/cerebras_sdk_docs/csl/tutorials/gemv-05-multiple-pes/`
- mlir-aie reference (sibling repo):
  `../mlir-aie/include/aie/Dialect/AIE/IR/AIEOps.td` — `aie.tile`,
  `aie.flow`, `aie.switchbox`, `aie.connect`, `aie.objectfifo`. Our
  `csl_layout.stream` aligns conceptually with `aie.flow`; our
  `csl_layout.set_color_config` aligns with `aie.connect` inside an
  `aie.switchbox`.
- WSE-3 fabric color budget: 24 colors. Local task id space is
  separate and much larger.
- MLIR pipeline registration: `mlir::PassPipelineRegistration<>` —
  upstream pattern used by `--test-lower-to-llvm` and dozens of
  similar pipelines across upstream dialects.
