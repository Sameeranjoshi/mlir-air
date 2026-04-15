# A Three-Layer MLIR Dialect Family for Cerebras CSL

**Date:** 2026-04-15  
**Status:** Early draft (no results/experiments yet).  
**Source note:** This document is a paper-style restructuring of the redesign evaluation note in [`2026-04-14-csl-redesign-design.md`](2026-04-14-csl-redesign-design.md).

## Abstract

Lowering high-level parallel IR to Cerebras' CSL requires modeling three distinct concerns: the per-PE program (`pe_program.csl`), the spatial binding and routing specification (`layout.csl`), and the host-side orchestration (`run.py`). The current AIR-to-CSL pipeline in MLIR-AIR captures parts of this structure but interleaves device and host concerns in a single mixed module, duplicates interface direction across layers, and relies on a monolithic translate-time emitter to reconcile naming and semantics. These choices introduce brittle hard-coded assumptions (e.g., 1x1 placement and fixed argument conventions) and make it difficult to extend the lowering to multi-PE programs, symbolic routing, and future CSL features such as tasks and DSDs.

We propose a three-layer dialect family that mirrors CSL's artifact boundaries: `csl_program` for placement-free per-PE compute and exports, `csl_layout` for rectangle/placement/routing and layout-level exports, and `csl_host` for host runtime sequences referencing layout exports. The design enforces one-way information flow (program -> layout -> host) via explicit symbol references, enables compile-time verification of cross-layer consistency, and makes translation emitters near-mechanical. This draft describes the IR boundaries, the intended conversion pipeline from AIR, and an evaluation plan (TBD) centered on correctness and scalability from 1x1 to NxN examples.

## 1. Introduction

The Cerebras WSE programming model is spatial: each processing element (PE) runs its own program, PEs are bound to a rectangle in the fabric, communication occurs via routed colors carrying 32-bit wavelets, and host code drives execution via the SDK runtime. In practice, a complete CSL program is always expressed as three artifacts: a PE program file, a layout file that binds PE programs to coordinates and wires routes, and a Python runner that orchestrates compilation, data movement, and RPC-style launches.

MLIR-AIR targets spatial accelerators and already contains an AIR-to-CSL lowering path. However, the current CSL/CSLRuntime dialect split does not form a true layering boundary: conversions are additive (they decorate the same module rather than consuming a predecessor), host and kernel interface facts are split across unrelated ops, and translation relies on a large multi-emitter walker that scans and reconciles the mixed IR. The result is fragile for anything beyond the single 1x1 `vec_add` milestone: placement is hard-coded, argument naming is special-cased, routing declarations are not elaborated, and the host emitter prints example-specific test logic.

This paper draft frames the redesign problem as an IR factoring problem, and proposes a dialect family that makes CSL's three conceptual layers explicit compilation units.

### Contributions (design-level; no empirical claims yet)

- A diagnosis of current AIR-to-CSL pain points as violations of separation-of-concerns and information-flow direction (program semantics depending on host context, and smart translation compensating for thin IR).
- A three-layer dialect family (`csl_program`, `csl_layout`, `csl_host`) with explicit symbol linking and verification across layers.
- A lowering/translation structure intended to make NxN placement, symbolic colors, and routing elaboration first-class pass responsibilities rather than translate-time logic.

## 2. Background

### 2.1 CSL program structure

CSL programs factor into three files:

- `layout.csl`: declares the PE rectangle, binds a PE program template to coordinates (with per-PE compile-time parameters), wires colors/routes, and defines host-visible names.
- `pe_program.csl`: defines PE-local state (`var`/`const`), functions, tasks, and a comptime block exporting symbols for the host.
- `run.py`: uses `SdkRuntime` to compile/load/run the layout, perform `memcpy_{h2d,d2h}`, launch exported entry points, and stop the runtime.

This structure is semantic: it separates placement/routing decisions from per-PE compute, and separates both from host orchestration.

### 2.2 AIE as a reference point

The MLIR-AIE toolchain provides relevant prior art for factoring device structural IR, per-core compute, routing, and host runtime sequencing. Key lessons for CSL lowering are:

- keep kernel bodies close to "plain MLIR" with thin device-specific wrappers,
- represent routing/connectivity as a separate layer and elaborate high-level constructs via passes,
- make translation targets focused walkers over a rich, emit-ready IR rather than embedding elaboration logic in the emitter.

## 3. Problem Statement

The current CSL dialect family in MLIR-AIR mixes layers and forces translation-time reconciliation:

- Kernel vs. host interface direction is duplicated across layers, requiring the kernel emitter to scan host IR to decide how to print PE buffers.
- Naming leaks into semantics (e.g., symbol-named SSA values), and `source_file` is modeled in IR even though it is a codegen choice.
- The conversion hard-codes 1x1 placement and special-cases a fixed set of argument names; routing/color constructs are present but not elaborated into emit-ready wiring.
- The CSL-to-CSLRuntime step decorates rather than consumes, producing a mixed module that multiple emitters must independently scan.

These issues are structural, not cosmetic: they make multi-PE placement, template reuse, and future routing/task/DSD work substantially harder than necessary.

## 4. Design Goals

The redesign targets a small set of invariants that can be checked mechanically:

- **Layered structure:** per-PE program, layout, and host must be distinct IR layers with explicit ownership of concerns.
- **One-way information flow:** program declares exports; layout binds programs to coordinates and defines host-visible exports; host references layout exports. No layer reads "inward" to recover meaning.
- **Emitters are dumb:** translation should be a near-mechanical walk over fully elaborated IR; elaboration belongs in passes.
- **NxN by construction:** grid placement should be representable as a single op plus an elaboration pass, not ad-hoc special cases.
- **Symbolic resources:** colors/routes are named symbolically; physical IDs and concrete wiring are assigned by dedicated passes.
- **Avoid codegen artifacts in semantics:** filenames, test harness logic, and example-specific validation do not belong in the IR.

## 5. Proposed Design: Three-Layer Dialect Family

### 5.1 `csl_program`: placement-free PE programs

`csl_program` models a single emittable `pe_program.csl` unit:

- Declares compile-time parameters (as attributes) and PE-local symbols (`var`, functions, tasks).
- Contains compute expressed in existing MLIR dialects (`memref/scf/arith/vector`), with a small CSL prelude for CSL-specific declarations and exports.
- Defines exports as the single source of truth for host-visible symbols, including directionality when relevant.

Critically, `csl_program` contains no coordinates, rectangle, routes, or host runtime constructs.

### 5.2 `csl_layout`: rectangle, placement, routing, and layout exports

`csl_layout` models `layout.csl`:

- Declares the rectangle (W, H), symbolic colors, and routing/connectivity relations.
- Places `csl_program` modules onto coordinates with per-PE compile-time parameter dictionaries.
- Defines host-visible names by referencing `csl_program` exports, producing layout-level exports that the host can target.

NxN placement is represented by a grid placement op (e.g., `place_grid`) that is elaborated into per-coordinate `place` ops by a `csl-layout-elaborate` pass.

### 5.3 `csl_host`: host runtime sequences

`csl_host` models `run.py` as structured host-side orchestration:

- Represents runtime lifecycle (`load/run/stop`), bulk transfers, and launches as explicit ops.
- References layout exports by symbol; it does not inspect program internals.
- Remains intentionally "thin": application-specific initialization and validation belong to an external test harness, not the emitted runner.

### 5.4 Linking and verification

The three layers coexist under an outer module convention, and are linked by explicit symbol references:

- `csl_layout` references `csl_program` symbols for placements and exports.
- `csl_host` references `csl_layout` exports for transfers and launches.

This enables compile-time verification of cross-layer consistency (e.g., host memcpys target real layout exports; layout exports target real program exports).

## 6. Lowering and Translation (Draft)

### 6.1 AIR -> CSL layers

The intended AIR lowering mirrors the AIR-to-AIE structure:

- Extract herd compute into `csl_program` modules (placement-free).
- Extract herd shape and placement hints into `csl_layout` (rectangle + placements), leaving routing/color assignment symbolic.
- Extract host launch/memcpy structure into `csl_host`, referencing layout exports.

### 6.2 Pass responsibilities

Key passes are expected to consume/transform within a single layer, rather than spreading semantics across layers:

- `csl-layout-elaborate`: expand `place_grid` into per-coordinate placements.
- `csl-allocate-colors`: assign physical color IDs to symbolic colors (0-23).
- `csl-elaborate-routes`: elaborate symbolic routing/connectivity into emit-ready layout wiring.
- (Future) DSD elaboration and host-sequence materialization, once multi-PE and routing are stable.

### 6.3 Translation targets

Translation remains textual (to match CSL workflows) but is split by layer:

- `csl_program` -> `pe_program.csl`
- `csl_layout` -> `layout.csl`
- `csl_host` -> `run.py`

Each translator is a focused walker over one dialect, assuming its input has already been elaborated by passes.

## 7. Evaluation Plan (TBD)

No experiments are reported in this draft. Planned evaluation focuses on correctness and extensibility:

- End-to-end compilation and execution for the existing 1x1 `vec_add` example on the redesigned IR.
- Extension to at least one NxN example (e.g., 2x2) without adding emitter-side special cases.
- IR invariants: cross-layer symbol verification and absence of "backward" information flow.
- Maintainability signals: emitter size/complexity reduction and elimination of hard-coded argument conventions and example-specific host logic.

## 8. Related Work (Sketch)

- MLIR-AIE: dialect factoring of device structure, routing, and host runtime sequencing provides a close template for layering and pass-vs-emitter responsibilities.
- MLIR-AIR: existing spatial and data-movement abstractions and the current AIR-to-CSL prototype provide the immediate integration context.
- Cerebras SDK CSL documentation: defines the semantic artifact boundaries and runtime API that the dialect family must target.

## 9. Conclusion

The current AIR-to-CSL pipeline embeds CSL's three conceptual layers in a single mixed IR and compensates with translate-time logic, which prevents the design from scaling beyond a narrow 1x1 milestone. A three-layer dialect family that mirrors CSL artifacts and enforces one-way information flow via symbol references provides a cleaner foundation for NxN placement, routing elaboration, and future CSL features, while keeping translation mechanical and testable. Results and evaluation are left for future work.

