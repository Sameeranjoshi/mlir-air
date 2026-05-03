# air-to-fire branch review

High-level summary of what was built, grouped by milestone. Use this to
navigate the ~100 commits on `air-to-fire` vs `main`.

---

## How to review efficiently

**Don't read all 100 commits.** Instead:

1. For each milestone below, read the **key source files** listed.
2. Run the **verification command** to confirm the tests still pass.
3. If something looks wrong, use `git log --oneline -- <file>` to see the
   history for that specific file, then `git show <sha>` to drill in.

To see everything that changed vs main in one place:
```bash
git diff main..air-to-fire -- mlir/
```

---

## Milestone 1 — CSL dialect v2 (the core IR)

**What:** Replaced the original raw-text Phase-1 emitter with a structured
MLIR dialect. Defines the IR that everything else builds on.

**Key source files:**
- `mlir/include/air/Dialect/CSL/` — all TableGen op definitions
  - `CSLLayoutOps.td` — `csl.wafer`, `csl.layout`, `csl_layout.place`
  - `CSLKernelOps.td` — `csl.func`, `csl.task`, `csl.var`, `csl.return`
  - `CSLRoutingOps.td` — `csl.color`, `csl_layout.set_color_config`
  - `CSLDataMovementOps.td` — `csl.get_fab_dsd`, `csl.builtin_call`
  - `CSLRuntimeOps.td` — `csl.export`, `csl.import_module`
- `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` — verifiers
- `mlir/lib/Conversion/AIRToCSL/` — `-air-to-csl` pass (lowers `air.herd`)

**Verify:**
```bash
bash utils/run_csl_ci.sh --fast --no-build   # <1s
```

---

## Milestone 2 — CSL text emitter (`air-translate --emit-csl`)

**What:** Emits runnable `.csl` + `run.py` + `commands_wse3.sh` from the
dialect IR. Handles programs, layout, host memcpy, and the Python driver.
Three sub-emitters: program, layout, host.

**Key source files:**
- `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp`
- `mlir/lib/Targets/CSLEmit/CSLLayoutEmitter.cpp`
- `mlir/lib/Targets/CSLEmit/CSLHostEmitter.cpp`
- `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h` — shared expression emitter

**Verify (single-PE e2e):**
```bash
air-opt -csl-auto-vectorize -csl-infer-exports \
    mlir/test/Targets/CSLEmit/e2e/scientific/saxpy.mlir \
  | air-translate --emit-csl --output-dir=/tmp/r
cd /tmp/r/saxpy && bash commands_wse3.sh
```

---

## Milestone 3 — `-csl-auto-vectorize` pass

**What:** Recognizes `scf.for` loop idioms over memrefs and rewrites them
to CSL DSD builtins (`@fadds`, `@fmuls`, `@fmovs`, `@fmacs`, etc.).
Replaces hand-written DSD code with auto-generated calls.

**Patterns implemented:**
`fadds`, `fsubs`, `fmuls`, `fmovs`, `fnegs`, `fmacs`, `FmulsScalar`,
`FmacsScalar`, `Rank2Fadds`, `Rank2Fmacs`

**Key source files:**
- `mlir/lib/Dialect/CSL/Transforms/CSLAutoVectorize.cpp`
- `mlir/include/air/Dialect/CSL/Transforms/Passes.td`

**Verify:**
```bash
bash utils/run_csl_ci.sh --fast --no-build 2>&1 | grep auto-vectorize
```

---

## Milestone 4 — Multi-PE dataflow (4-pass pipeline)

**What:** Adds inter-PE data movement via a 4-pass lowering pipeline that
transforms `csl.dataflow.put/get` (high-level stream ops) all the way down
to fabric DSDs + tasks + color routing.

**4 passes:**
1. `csl-materialize-dataflow-colors` — assigns a `csl.color` symbol per stream
2. `csl-allocate-color-ids` — assigns integer IDs (0, 1, 2, …)
3. `csl-lower-dataflow-routing` — emits `set_color_config` per PE endpoint
4. `csl-lower-dataflow-data` — expands put/get to DMA tasks + async `@fmovs`

**Key source files:**
- `mlir/lib/Dialect/CSL/Transforms/CSLMaterializeDataflowColors.cpp`
- `mlir/lib/Dialect/CSL/Transforms/CSLAllocateColorIds.cpp`
- `mlir/lib/Dialect/CSL/Transforms/CSLLowerDataflowRouting.cpp`
- `mlir/lib/Dialect/CSL/Transforms/CSLLowerDataflowData.cpp`
- `mlir/include/air/Dialect/CSL/CSLOps.td` — `csl.dataflow.put/get`

**Entry point:**
```bash
air-opt --csl-dataflow-to-csl input.mlir
```

**Verify:**
```bash
# All multi-PE e2e tests (capped at 4 concurrent for shared machine)
bash utils/run_csl_ci.sh --no-build -j 2
```

---

## Milestone 5 — First inter-PE ping (2-PE east/west/north/south)

**What:** First end-to-end proof that multi-PE dataflow works on the
WSE-3 simulator. Producer sends a buffer east to consumer.

**Key tests:**
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe.mlir` (east)
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe_west.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe_north.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe_south.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_2pe_two_streams.mlir`

---

## Milestone 6 — Multi-hop chains + fan-in

**What:** 3-PE and 4-PE chains (data hops through intermediate PEs).
Fan-in from 3 sources to 1 sink.

**Key tests:**
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/ping_3pe_chain.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/chain_4pe_east.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/fanin_3src.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/east_west_2row.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/north_south_2col.mlir`

---

## Milestone 7 — Relay PEs + reduction chain (DMA approach)

**What:** PEs that receive, compute, and forward (not just passthrough).
P0→P1→P2→P3 chain where middle PEs subtract/add 100. DMA-based: relay
PE wakes up via GET completion task, runs a loop, then does a PUT.

**Key new op:** `csl.task` with `trigger_kind = "local_task_id"` for
completion callbacks.

**Key test:**
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/reduce_chain_4pe.mlir`

---

## Milestone 8 — Data tasks + per-wavelet relay

**What:** Native CSL data task model: `task relay(val: f32) void` fires
once per arriving wavelet, receives the value as an argument, forwards
modified value synchronously. Simpler than DMA relay.

**New ops:**
- `csl.task` with `trigger_kind = "data_task"` + `color` attribute
- `csl.dataflow.send_wavelet @stream value(%v)`

**Key source files:**
- `mlir/include/air/Dialect/CSL/CSLOps.td` — `CSL_DataflowSendWaveletOp`
- `mlir/lib/Dialect/CSL/IR/CSLOps.cpp` — `TaskOp::verify()` data_task branch
- `mlir/lib/Targets/CSLEmit/CSLProgramEmitter.cpp` — data task emission
- `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h` — `expandSendWavelet()`

**Key test:**
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/reduce_chain_4pe_wavelet.mlir`

---

## Milestone 9 — Runtime PE-coordinate branching (`get_x_coord`)

**What:** Relay PEs can pick ±100 at runtime based on their x-coordinate
parity. One MLIR program shape, heterogeneous runtime behavior.

**New ops:**
- `csl.get_x_coord : i16`
- `csl.get_y_coord : i16`
- `arith.select` support in emitter

**Key source files:**
- `mlir/include/air/Dialect/CSL/CSLOps.td` — `CSL_GetXCoordOp`
- `mlir/lib/Targets/CSLEmit/CSLEmitCommon.h` — x/y coord + select handlers

**Key test:**
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/reduce_chain_4pe_xparity.mlir`
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/pe_id_even_odd.mlir`

---

## Milestone 10 — CI infrastructure

**What:** Test automation and tooling.

- `utils/run_csl_ci.sh` — two-tier runner (lit + SDK simulator)
- `utils/run_csl_sdk.sh` — parallel wafer simulator sweep
- `mlir/test/lit.cfg.py` — `cerebras-sim` parallelism group (cap=4)
- `mlir/test/Targets/CSLEmit/e2e/multi_pe/lit.local.cfg` — group assignment
- `.github/workflows/csl-ci.yml` — Tier-1 on GitHub cloud (disabled for now)
- `infra/chpc/` — self-hosted runner plan for CHPC when needed

---

## What's NOT on this branch (future work)

- `-air-to-csl` for non-trivial herds (only 1-PE vecadd today)
- `air.channel` → dataflow lowering
- CSL dialect → MLIR verifier for routing validity
- Multi-PE reduction (actual `@fadds` across PE boundaries, not just relay)
