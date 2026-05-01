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
**multi-PE *dataflow*** — PE A computes some data and *sends it* to PE B
over the WSE fabric, B receives and consumes it. This is the foundation
required for tutorial-6 GEMV, all SPADA collectives (`broadcast_1D`,
`tree_reduce_1D`, `chain_reduce`, …), and the longer-term lowering of
`air.channel.put`/`get` from the AIR layer.

The Cerebras tutorial 6 reference implementation
(`docs/superpowers/raw/cerebras_sdk_docs/csl/tutorials/gemv-06-routes-1/`)
exposes the underlying machinery directly: colors, per-PE
`@set_color_config`, `fabin_dsd`/`fabout_dsd`, `local_task_id`, async
fabric ops, `@bind_local_task`. A literal port of that surface to MLIR
would force the test author to write all of those primitives by hand — a
poor user model and a poor lowering target for AIR.

**Goal of this milestone:** ship a *high-level user surface* — three new
ops (`csl_layout.stream`, `csl.stream.put`, `csl.stream.get`) — that
hides colors, routing, fabric DSDs, async/task plumbing, and color
allocation behind two passes (`--csl-allocate-colors`,
`--csl-lower-streams`). The user writes "this dataflow edge runs from
(0,0) to (1,0) on color RED, push this buffer through it, pull this
buffer out the other side". The compiler does everything else.

**Concrete deliverable:** an end-to-end test
`mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe.mlir` plus its
supporting `lit` infrastructure that:

1. Lowers via `air-opt -csl-infer-exports -csl-allocate-colors -csl-lower-streams`.
2. Translates via `air-translate --emit-csl` to `layout.csl` +
   `left_pe.csl` + `right_pe.csl` + `run.py`.
3. Compiles via `cslc` and runs via `cs_python` on the WSE-3 simulator.
4. Prints `SUCCESS!` (the right-PE buffer matches what the left-PE sent).

Plus comprehensive surrounding tests: dialect roundtrip, verifier
negative tests, FileCheck on each pass output, and the simulator e2e
above. The user requirement was explicit: "have a lot of tests, make
them run on simulator and should be all green".

## 2. Scope

### In scope

- Three user-facing ops: `csl_layout.stream`, `csl.stream.put`,
  `csl.stream.get`.
- Five internal ops/types: `!csl.local_task_id`, `csl.local_task_id`,
  `csl.get_fab_dsd`, `csl_layout.set_color_config`, plus extensions to
  the existing `csl.task` (operand polymorphism) and `csl.builtin_call`
  (`async`/`activate` attrs).
- Two new passes: `--csl-allocate-colors` (stub: monotonic id assignment)
  and `--csl-lower-streams` (expansion to internal ops).
- Move `csl.color`'s scope from `csl.program`-body to `csl.layout`-body
  (it's a fabric resource, shared across PEs, referenced by symbol).
- Text emitter (`Targets/CSLEmit/`) cases for every new op.
- Tests: dialect roundtrip, verifier negative, per-pass FileCheck,
  end-to-end simulator. The 2-PE ping kernel as the pinnacle e2e test.

### Out of scope (deferred)

- **Tutorial-6 GEMV** (column-major A, local matvec, recv-with-`@fadds`,
  fork-join overlap). Held until ping is green; will reuse this
  surface.
- **Multi-hop routing.** Single-hop (one EAST or SOUTH step) only.
- **Multicast streams** (`relative_stream([a:b], dy)` analogue).
  Single-target only.
- **Async tokens on stream ops.** User-facing put/get look synchronous
  (program-order). The async + task-id machinery exists in the
  *expansion*, but not in the user IR. Tutorial-6 GEMV may need user-IR
  async tokens; we add them then.
- **Real liveness-driven graph-coloring color allocator.**
  `--csl-allocate-colors` ships as a monotonic stub. Real allocation
  over the WSE-3 24-color budget is a follow-up; this spec defines the
  *home* for it but not its algorithm.
- **AIR-level `air.channel`/`air.channel.put`/`air.channel.get`
  lowering** (`-air-to-csl` for inter-herd channels). Deferred per
  user's Q1-C answer; the user surface here is intentionally
  **isomorphic** to `air.channel`-style so the future lowering is 1:1.
- **`@map` Tier-2 / reductions with `iter_args` / non-`f32` element
  types.** Same as the auto-vectorize spec defers them.

### Explicitly *not* a goal

- Replicating SPADA's `relative_stream(dx, dy)` form. Absolute
  `from(x,y) to(x,y)` reads better at the dialect level and matches
  AIE's `aie.flow` shape; relative-form sugar is a future canonicalize
  pass if demand emerges.

## 3. Design

### 3.1 Three layers

```
USER WRITES (typeless edges, virtual colors, sync-looking)
──────────────────────────────────────────────────────────
   csl.color @send_color : !csl.color                       ← symbolic, no id
   csl_layout.stream @send_ch from(0,0) to(1,0)             ← typeless edge
                     {color = @send_color}                  ← symbol ref
   csl.stream.put @send_ch source(%buf) extent(N) : T       ← producer
   csl.stream.get @send_ch target(%buf) extent(N) : T       ← consumer
                          │
                          │  --csl-allocate-colors
                          ▼
PRE-EMIT (color ids filled in)
──────────────────────────────────────────────────────────
   csl.color @send_color {id = 0 : i32} : !csl.color
                          │
                          │  --csl-lower-streams
                          ▼
INTERNAL (low-level, what the emitter consumes)
──────────────────────────────────────────────────────────
   In csl.layout body:
     csl.color @send_color {id = 0 : i32}
     csl_layout.set_color_config @send_color at(0,0) rx(RAMP) tx(EAST)
     csl_layout.set_color_config @send_color at(1,0) rx(WEST) tx(RAMP)
     csl_layout.place @left_pe at (0,0)
     csl_layout.place @right_pe at (1,0)

   In csl.program @left_pe body:
     %src = csl.get_mem_dsd %buf : memref<128xf32> -> !csl.dsd
     %out = csl.get_fab_dsd fabout @send_color extent(128) : !csl.dsd
     %tid = csl.local_task_id @put_done_id {id = 8 : i32}
            : !csl.local_task_id
     csl.task @put_done local_task_id(%tid) {
       csl.builtin_call "unblock_cmd_stream" () : () -> ()
       csl.return
     }
     csl.builtin_call "fmovs"(%out, %src)
                      {async, activate = @put_done_id}
                      : (!csl.dsd, !csl.dsd) -> ()

   In csl.program @right_pe body: mirror with fabin + target buf
                          │
                          │  air-translate --emit-csl
                          ▼
TEXT (Cerebras CSL source)
──────────────────────────────────────────────────────────
   layout.csl + left_pe.csl + right_pe.csl + run.py
                          │
                          │  cslc + cs_python
                          ▼
   SUCCESS!
```

The user *only* writes the top layer. The middle layer is what
`air-translate --emit-csl` consumes. The bottom layer is what `cslc`
compiles.

### 3.2 New user-facing ops

#### `csl_layout.stream`

```
csl_layout.stream @ch from (px : i64, py : i64) to (px : i64, py : i64)
                  attr-dict
```

- **Parent:** `csl.layout`'s region.
- **Symbol:** the op carries its own `sym_name`; references like
  `csl.stream.put @ch` resolve to it.
- **Operands:** none — coords are constant attributes.
- **Attributes:**
  - `from_x`, `from_y`, `to_x`, `to_y`: `i64`.
  - `color`: `FlatSymbolRefAttr` referencing a `csl.color` symbol declared
    in the same `csl.layout` body.
- **Pinning a color id (test-only escape hatch):** declare the
  `csl.color` with an explicit `id` attribute. The allocator
  (§3.7) treats colors with an `id` as pre-pinned and skips them.
  This reuses the existing `csl.color {id = N} : !csl.color` syntax —
  no new attribute on the stream op.
- **Verifier:**
  - Coords must be within the wafer's `(width, height)`.
  - For ping milestone: `(to_x - from_x, to_y - from_y)` must be one of
    `(±1, 0)` or `(0, ±1)` — single-hop along a cardinal direction.
    Multi-hop verifier rejection is the gate that prevents the user
    from accidentally writing routes the lowering can't generate yet.
  - The referenced `@color` symbol must be a `csl.color` declared in
    the same `csl.layout`.

#### `csl.stream.put`

```
csl.stream.put @ch source(%mem : memref<...>) extent($n : index) : <type>
```

- **Parent:** a `csl.func` body inside a `csl.program`.
- **Operands:** `%mem` (memref), `%n` (index extent).
- **Attributes:** `stream`: `FlatSymbolRefAttr` to a `csl_layout.stream`.
- **Result:** none (sync-looking; tokens deferred).
- **Verifier:**
  - Stream symbol must resolve to a `csl_layout.stream` whose `from`
    coord matches the parent program's placement (cross-checked at
    lowering time, since placement is in `csl.layout`).
  - `%mem`'s element type must be `f32` (only type wired in this
    milestone — others deferred per auto-vectorize spec).
  - Extent must be a constant or an SSA value that's a `memref.dim` /
    `arith.constant` (the lowering needs a static extent to construct
    the fabric DSD; we relax later).

#### `csl.stream.get`

Mirror of `csl.stream.put`:

```
csl.stream.get @ch target(%mem : memref<...>) extent($n : index) : <type>
```

Verifier mirrors, with the parent program's placement matching the
stream's `to` coord.

### 3.3 Extension to `csl.color` op (scope move)

Today `csl.color` lives inside `csl.program`. Move it to `csl.layout`
body so a single color symbol can be referenced from both PE programs
and from `csl_layout.set_color_config`. The op shape is unchanged:

```
csl.color @send_color : !csl.color                  // virtual (no id)
csl.color @send_color {id = 0 : i32} : !csl.color   // pinned (post-allocator)
```

Symbol resolution from inside `csl.program` bodies — needed by
`csl.get_fab_dsd %color_sym` after lowering — uses the same cross-region
SymbolTable lookup that `csl_layout.place params {p = @sym}` already
uses. No new infrastructure.

This is a **breaking change** for `csl.color` placement, but per the
SPADA gap analysis no working e2e test currently uses `csl.color` —
only the dialect roundtrip tests in `mlir/test/Dialect/CSL/`. Those will
be updated in lockstep.

### 3.4 New internal ops/types

#### `!csl.local_task_id`

```
def CSL_LocalTaskIdType : CSL_Type<"LocalTaskId", "local_task_id"> {
  let summary = "Activatable task id (CSL local_task_id)";
}
```

Used as the operand of `csl.task`'s local-task variant and as the
target of `csl.builtin_call`'s `activate` attribute.

#### `csl.local_task_id`

```
%tid = csl.local_task_id @put_done_id {id = 8 : i32} : !csl.local_task_id
```

- **Parent:** a `csl.program` body.
- **Symbol:** `sym_name` for cross-reference from `activate = @put_done_id`.
- **Attributes:** `id` is **required for this milestone** — no allocator
  for local task IDs (they're a much larger space than fabric colors;
  no constraint pressure in the foreseeable kernels).
- **Emit:**
  ```csl
  const put_done_id: local_task_id = @get_local_task_id(8);
  ```

#### `csl.get_fab_dsd`

```
%out = csl.get_fab_dsd fabout @color_sym extent(%n : index) : !csl.dsd
%in  = csl.get_fab_dsd fabin  @color_sym extent(%n : index) : !csl.dsd
```

- **Parent:** a `csl.func` body inside a `csl.program`.
- **Operands:** `%n` (index extent, SSA value).
- **Attribute:** `color`: `FlatSymbolRefAttr` referencing a `csl.color`
  declared in the enclosing `csl.layout` body. Cross-region resolution
  uses the same SymbolTable lookup as `csl_layout.place params {p =
  @sym}`. Color is an attribute (not an SSA operand) because the
  `csl.color` symbol lives in a different region from the `csl.func`
  body that contains the `csl.get_fab_dsd`.
- **Direction:** keyword `fabin` | `fabout`, lowered to existing
  `DsdKind` enum values (already defined in `CSLBase.td:108-113`).
- **Emit:**
  ```csl
  const out_dsd = @get_dsd(fabout_dsd,
      .{ .extent = 128, .fabric_color = send_color });
  ```

This op fills the deferred slot at `CSLOps.td:17,235`.

#### `csl_layout.set_color_config`

```
csl_layout.set_color_config @color_sym at (px : i64, py : i64)
                            rx(<dir>) tx(<dir>)
```

- **Parent:** `csl.layout` body.
- **Operands:** none — color is by symbol, coords + directions are attrs.
- **Attributes:** `color`: `FlatSymbolRefAttr`, `px`/`py`: `i64`,
  `rx`/`tx`: existing `Direction` enum (from `CSLBase.td:102`).
- **Emit:**
  ```csl
  @set_color_config(0, 0, send_color,
      .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
  ```

Single-direction `rx` and `tx` only. Multi-direction sets
(`.tx = .{EAST, WEST}` for a fan-out) are deferred.

### 3.5 Extension to `csl.task`

Current: `csl.task @t color(%c : !csl.color) { body }`.

Extended: operand type becomes
`AnyTypeOf<[CSL_ColorType, CSL_LocalTaskIdType]>`. Assembly format
keyword distinguishes:

```mlir
csl.task @recv color(%c) { ... }                       // existing
csl.task @put_done local_task_id(%tid) { ... }         // new variant
```

The emitter generates the corresponding `comptime { @bind_local_task(t,
binding); }` line automatically — *no separate `csl.bind_local_task` op*.
The bind is implicit in the operand SSA edge.

### 3.6 Extension to `csl.builtin_call`

Two new optional attributes:

| Attribute | Type | Effect |
|---|---|---|
| `async` | `UnitAttr` | Emits `.async = true` field. |
| `activate` | optional `FlatSymbolRefAttr` | Emits `.activate = <sym>` field; references a `csl.local_task_id` symbol. |

Verifier: if `activate` is set, `async` must also be set (the CSL
tutorial's deadlock warning: fabric DSDs sync = bad).

Emit:

```csl
@fmovs(out_dsd, src_dsd, .{ .async = true, .activate = put_done_id });
```

### 3.7 Pass — `--csl-allocate-colors`

**Input:** module containing `csl.color` symbols with virtual (no-`id`)
form.
**Output:** every `csl.color` has its `id` attribute set.

**Algorithm (milestone — stub):** monotonic. Walk the module's
`csl.color` ops in declaration order; for each color *without* an `id`
attribute, assign `id = next_unused`, where `next_unused` starts at 0
and skips any ids that appear on already-pinned colors. Colors that
already have an `id` attribute are left untouched (pinned).

**Future work (out of scope):** liveness-driven graph coloring over the
24-color budget, with task-overlap as edges. This pass is the home;
algorithms swap inside it without touching dialect or kernels.

### 3.8 Pass — `--csl-lower-streams`

**Input:** module with virtual streams (after `--csl-allocate-colors`,
so colors have ids).
**Output:** stream ops removed, internal ops in their place.

**Per `csl_layout.stream @ch from(x1,y1) to(x2,y2) {color = @c}`:**

1. Compute direction from coord delta:
   - `(+1, 0) ⇒ tx = EAST` on src, `rx = WEST` on dst.
   - `(-1, 0) ⇒ tx = WEST`, `rx = EAST`.
   - `(0, +1) ⇒ tx = SOUTH`, `rx = NORTH`.
   - `(0, -1) ⇒ tx = NORTH`, `rx = SOUTH`.
2. Insert into `csl.layout` body:
   ```mlir
   csl_layout.set_color_config @c at(x1,y1) rx(RAMP) tx(<src_tx>)
   csl_layout.set_color_config @c at(x2,y2) rx(<dst_rx>) tx(RAMP)
   ```
3. Erase the `csl_layout.stream` op.

**Per `csl.stream.put @ch source(%buf) extent(%n) : memref<NxT>` inside `csl.program @P`:**

Resolve `@ch` to its `csl_layout.stream`, get its color `@c`. In `@P`'s body:

1. `%src = csl.get_mem_dsd %buf : memref<NxT> -> !csl.dsd` (or
   reuse if the stream-put's source already has a DSD nearby).
2. `%out = csl.get_fab_dsd fabout @c extent(%n) : !csl.dsd`.
3. Allocate a fresh local task id symbol per put — name pattern
   `@<stream_sym>_put_done_<n>`, integer id from a per-program
   monotonic counter starting at a configurable base (default `8`,
   matching the tutorial). Future allocator can reassign.
4. Emit a `csl.task @<auto> local_task_id(%tid) { csl.builtin_call
   "unblock_cmd_stream" : () -> () ; csl.return }`.
5. Emit `csl.builtin_call "fmovs"(%out, %src) {async, activate =
   @<auto>} : (!csl.dsd, !csl.dsd) -> ()`.
6. Erase the `csl.stream.put` op.

**Per `csl.stream.get`:** mirror — `fabin` instead of `fabout`,
`@<stream_sym>_get_done_<n>` task name, target memref's DSD as the
**destination** of `@fmovs` (so `@fmovs(target_dsd, in_dsd, …)`).

The `unblock_cmd_stream` builtin is the existing CSL idiom — it
unblocks the host memcpy command stream so subsequent `memcpy_d2h`
calls can proceed once both PEs' tasks fire. The emitter wires it
through an `@import_module("<memcpy/memcpy>")` import already present
in the working saxpy.mlir.

### 3.9 Emitter additions

`mlir/lib/Targets/CSLEmit/` already has a structured emitter for the
existing CSL dialect (`saxpy.mlir` greens through it). New cases:

- `csl.local_task_id @sym {id=N}` → `const sym: local_task_id = @get_local_task_id(N);`
- `csl.get_fab_dsd fabout @c extent(N)` → `const <ssa_name>_dsd = @get_dsd(fabout_dsd, .{ .extent = N, .fabric_color = c });`
- `csl.get_fab_dsd fabin @c extent(N)` → mirror with `fabin_dsd`.
- `csl_layout.set_color_config @c at(x,y) rx(R) tx(T)` → emitted in
  the layout block as `@set_color_config(x, y, c, .{ .routes = .{ .rx
  = .{R}, .tx = .{T} } });`.
- `csl.task @t local_task_id(%tid)` → `task t() void { … }` body, plus
  `comptime { @bind_local_task(t, <tid_sym>); }` line.
- `csl.builtin_call …{async, activate=@x}` → existing emit path
  augmented with `.async = true, .activate = x` field syntax.

Existing emitter cases are not modified (no breakage).

### 3.10 Passes — registration

In `air-opt`'s pass registration (`tools/air-opt/`):
- Register `--csl-allocate-colors` (Transforms/CSLAllocateColors.cpp).
- Register `--csl-lower-streams` (Transforms/CSLLowerStreams.cpp).

Both passes get TableGen-generated `GEN_PASS_DEF_*` headers via
`mlir/include/air/Dialect/CSL/Transforms/Passes.td`.

In the standard `aircc.py` pipeline: not invoked here — `aircc.py`
targets the GPU pipeline. The CSL flow is driven directly by `air-opt`
+ `air-translate` per the existing CSL e2e tests. If/when AIR→CSL
lowering arrives, that driver will compose this pass sequence.

## 4. Test plan

User requirement: "have a lot of tests, make them run on simulator and
should be all green". Four layers, all required to land:

### 4.1 Dialect roundtrip tests — `mlir/test/Dialect/CSL/`

One `.mlir` per new op, parse → print → parse-again equivalence:

- `roundtrip_stream.mlir` — `csl_layout.stream @s from(0,0) to(1,0) {color = @c}` parses, prints, re-parses identically.
- `roundtrip_stream_put_get.mlir` — `csl.stream.put` and `.get` with various memref shapes and extents.
- `roundtrip_local_task_id.mlir` — `csl.local_task_id @t {id=8} : !csl.local_task_id`.
- `roundtrip_get_fab_dsd.mlir` — both `fabin` and `fabout` forms.
- `roundtrip_set_color_config.mlir` — all 4 cardinal directions.
- `roundtrip_task_local_id.mlir` — `csl.task @t local_task_id(%tid) { ... }` (new variant).
- `roundtrip_builtin_async.mlir` — `csl.builtin_call …{async, activate = @x}`.
- `roundtrip_color_in_layout.mlir` — `csl.color` inside `csl.layout` body
  (the scope move).

### 4.2 Verifier negative tests — same directory

One per failure mode, `// RUN: not air-opt %s … 2>&1 | FileCheck %s`:

- Stream coords outside wafer rectangle.
- Stream non-cardinal / multi-hop offset (e.g., `from(0,0) to(2,3)`).
- `stream.put` memref element type that the lowering doesn't yet support
  (only `f32` is wired this milestone — `i32`/`f16` etc. should be
  rejected with a clear diagnostic, not silently miscompile).
- `stream.put`/`stream.get` parent program placement mismatch with stream's
  `from`/`to` coord.
- `csl.builtin_call {activate = @x}` without `async`.
- `csl.task local_task_id(%c)` where `%c` is a color, not a local_task_id.
- `csl.color` outside `csl.layout` body (after scope move).
- `csl.local_task_id` without `id` attribute.

### 4.3 Pass FileCheck tests — `mlir/test/Conversion/` or `mlir/test/Dialect/CSL/Transforms/`

#### `--csl-allocate-colors`

- `allocate_one_color.mlir` — single virtual color → id 0.
- `allocate_three_colors.mlir` — three virtual colors → 0, 1, 2.
- `allocate_with_pin.mlir` — one color declared with `{id = 5}`
  (pre-pinned), one virtual → 0; the pinned one keeps 5; allocator
  picks 0 for the virtual one (skipping no slots since 5 ≠ 0).
- `allocate_skips_pinned_id.mlir` — virtual color exists, plus a
  pre-pinned `{id = 0}`; allocator must give the virtual color id 1
  (skipping the taken slot 0).
- `allocate_idempotent.mlir` — running the pass twice is a no-op.

#### `--csl-lower-streams`

For each cardinal direction, a FileCheck test asserting:

- `lower_east.mlir` — stream `(0,0)→(1,0)` produces `tx=EAST, rx=WEST`.
- `lower_west.mlir` — `(1,0)→(0,0)` produces `tx=WEST, rx=EAST`.
- `lower_south.mlir`, `lower_north.mlir` — vertical analogues.
- `lower_emits_fabric_dsd.mlir` — stream put inserts a `fabout`
  `get_fab_dsd` and an async `fmovs` builtin.
- `lower_emits_local_task.mlir` — stream put inserts the auto-generated
  `csl.local_task_id` + `csl.task` + `activate = @<auto>`.
- `lower_get_uses_fabin.mlir` — get uses `fabin` and the target buffer
  is the destination of `@fmovs`.
- `lower_two_streams_same_color.mlir` — two streams referencing the
  same color symbol share the physical color (constraint hint
  honored).

### 4.4 End-to-end emitter tests — `mlir/test/Targets/CSLEmit/`

Two tiers:

#### Tier 1 — `--emit-csl` text FileCheck (offline, fast)

Located at `mlir/test/Targets/CSLEmit/multi_pe/`:

- `emit_set_color_config.mlir` — minimal layout exercising the new
  layout-block emission line.
- `emit_local_task_bind.mlir` — `csl.task local_task_id(...)` produces
  the `task t() void { … }` body + `comptime { @bind_local_task(...); }`
  pair.
- `emit_fabric_dsd.mlir` — `csl.get_fab_dsd fabout` produces the
  expected `@get_dsd(fabout_dsd, .{ .extent = N, .fabric_color = c });`
  line.
- `emit_async_builtin.mlir` — `csl.builtin_call "fmovs"(...) {async,
  activate = @t}` produces `@fmovs(…, .{ .async = true, .activate =
  t });`.
- `emit_stream_full.mlir` — top-to-bottom: ping kernel passes through
  `--csl-allocate-colors --csl-lower-streams --emit-csl`, FileCheck
  asserts the full layout.csl + per-PE files.

#### Tier 2 — Simulator e2e (`utils/run_csl_ci.sh` integration)

Located at `mlir/test/Targets/CSLEmit/e2e/multi_pe/`:

- `ping_2pe.mlir` — **the milestone**. 2 PEs side-by-side, left has a
  buffer of 128 f32 values seeded by host h2d, sends to right via
  stream, right stores into its buffer, host d2h reads right's buffer
  and verifies elementwise match. `cs_python` exits 0 → SUCCESS.
- `ping_2pe_vertical.mlir` — same kernel rotated 90° (PE (0,0) → PE
  (0,1)) to exercise the `tx=SOUTH/rx=NORTH` direction inference.
- `ping_2pe_west.mlir` — reversed direction (right sends to left).
- `ping_2pe_north.mlir` — vertical reversed.
- `ping_2pe_extents.mlir` — same shape with different `extent` values
  (16, 64, 256, 1024) parameterized via lit substitutions.
- `ping_2pe_two_streams.mlir` — two independent streams between the
  same PE pair on different colors (simultaneous dataflow edges).
- `ping_3pe_chain.mlir` — A→B→C linear chain (B does both a get and a
  put); verifies that one PE can host both ends of two streams.

Each e2e test is a single `.mlir` with a `// RUN:` line that drives the
full pipeline including `cs_python` invocation; the test passes iff
`cs_python` returns 0 *and* the simulator output indicates correctness.
The existing harness in `utils/run_csl_ci.sh` already does this; new
tests are added to its parallel-sim list.

### 4.5 CMake / lit / CI integration

- New lit subdirs added to `mlir/test/Dialect/CSL/CMakeLists.txt`,
  `mlir/test/Targets/CSLEmit/CMakeLists.txt`,
  `mlir/test/Conversion/CMakeLists.txt` as appropriate.
- New build target `check-airmlir-csl-streams` for narrow iteration.
- `utils/run_csl_ci.sh`'s e2e simulator suite is extended with the
  multi-PE tests; they participate in the pre-push hook (full CSL
  suite) and CI (lit only — simulator is local-only by current policy).

## 5. Differentiation from related work

This design intentionally diverges from the three closest reference
points. Documented here so reviewers and downstream readers can
evaluate the trade-offs.

| Property | SPADA | AIR `air.channel` | AIE `aie.flow` | This design |
|---|---|---|---|---|
| Typed streams | textual at SPADA-IR | typeless | typeless (objectfifo is typed) | typeless (semantically right; see §3.2) |
| Color/channel allocator | global rename, single pass | n/a (lowered later) | route-finder (placement, not coloring) | **separate pass; home for liveness-driven RA over 24-color budget** |
| User writes color ids? | no (virtual) | n/a | no (route-finder picks) | **no — symbolic colors only** |
| Routing inferred from coords? | yes (single-hop) | yes (in lowering) | yes (full route-finder) | yes (single-hop; cardinal only this milestone) |
| Async semantics in user IR | textual (`completion`/`await`) | yes (`!air.async.token`) | yes (objectfifo acquire/release) | **no in this milestone, deferred** |
| AIR-lowering target shape | n/a | n/a | direct AIE | **isomorphic to `air.channel` for future lowering** |
| Hardware concept alignment | CSL-native | abstract | AIE-native | **CSL-native, AIE-aligned at high layer** |

**The novel combination:** symbolic-color names + register-allocator-as-pass
+ sync-looking user surface + AIR-isomorphic shape + AIE-aligned
high-level layer. None of the three references has all five. The
register-allocator-as-pass shape, in particular, is a path SPADA
explicitly punts on (their allocator is ad-hoc Python in
`_collect_colors_globally`); building it as a proper MLIR pass with
liveness from a task DAG is a contribution.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Direction inference for non-cardinal hops produces wrong routes silently. | Verifier on `csl_layout.stream` rejects non-cardinal coord deltas in this milestone. Future multi-hop arrives with its own verification gate. |
| Color-symbol scope move breaks existing tests. | No working e2e currently uses `csl.color` per the SPADA gap analysis. Roundtrip tests update in lockstep. The failure mode is loud (verifier reject), not silent. |
| Auto-generated local task ids collide across multiple stream puts/gets in one program. | Per-program monotonic counter, prefix with the stream symbol. Verifier on `csl.local_task_id` enforces unique ids within a `csl.program`. |
| `unblock_cmd_stream` lowering doesn't match the actual host memcpy contract. | Existing saxpy.mlir's `cs_python` works because it manually calls `unblock_cmd_stream` from a host-launched function. Mirror that contract; the launch path is unchanged. |
| Simulator e2e is slow / fragile / local-only. | Already true today and handled: `run_csl_ci.sh` runs simulator tests in parallel; CI runs lit-only; pre-push hook runs the full simulator suite. New tests follow the same pattern. |
| Adding 8+ new ops is a large surface to land in one PR. | Sequence the implementation in 4 stages (low-level ops first, then high-level, then passes, then e2e). Each stage is independently testable; stages 1–3 land via TDD on the corresponding test layer in §4. The implementation plan (next document) defines the stages. |

## 7. Forward references

- **Tutorial-6 GEMV** will reuse this entire surface: same stream
  declaration, same put/get, plus a new `csl.stream.get` variant or
  attribute that lowers to recv-with-`@fadds` (accumulate) instead of
  `@fmovs` (move). Async tokens will be added to `csl.stream.put`/`get`
  at that time so the sender's `gemv` work overlaps with the send.
- **AIR-level lowering (`-air-to-csl` for inter-herd channels)** will
  be a new pass mapping `air.channel` 1:1 to `csl_layout.stream` and
  `air.channel.put`/`get` 1:1 to `csl.stream.put`/`get`. Because the
  surface here is intentionally isomorphic to AIR's channel surface,
  the lowering is structural — no new analysis. Cross-reference: see
  AIE's `aie.flow` shape at
  `../mlir-aie/include/aie/Dialect/AIE/IR/AIEOps.td` (the high-level
  flow op our `csl_layout.stream` is modelled after) and `aie.connect`
  inside `aie.switchbox` (the per-tile routing primitive analogue of
  our `csl_layout.set_color_config`).
- **Real color allocator (graph-coloring over 24-color budget)** slots
  into `--csl-allocate-colors` without changes to dialect or kernel
  code. Liveness comes from the task DAG; constraint edges come from
  temporally overlapping streams. SPADA's approach is the design
  baseline; we improve on it by making it a first-class MLIR pass.

## 8. Open questions

None remaining for this milestone. All Q1–Q5 (front door, scope,
program structure, sugar layer, color allocation) were resolved
during brainstorming. The implementation plan (next document) will
sequence the work into stages.

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
  `csl_layout.set_color_config` aligns with the `aie.connect` inside an
  `aie.switchbox`.
- WSE-3 fabric color budget: 24 colors. Local task id space is
  separate and much larger.
