# Future Ideas — rough sketches

Capture rough ideas here as they come up. Keep entries short — enough to be convinced to
revisit, not enough to be a spec. Add a line at the bottom of each idea with the date it
was captured so we remember when we had it.

When an idea gets promoted, move it to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
and leave a one-line pointer here.

---

## Translation validation via MLIR SMT dialect

**One-liner:** Prove `--emit-csl` is semantics-preserving for bounded programs, using
MLIR's upstream SMT dialect and Z3.

**Why this is interesting:**
Most MLIR→DSL lowerings are "trust us, the tests pass." Nobody formally verifies that
the emitted code computes the same thing as the source IR. For a spatial ISA like CSL
(where hand-written compilers historically have bugs that only surface on real hardware),
a machine-checked proof of functional equivalence is a real research contribution —
and the tooling (MLIR SMT dialect, Z3 backend) is already upstream.

**Rough sketch:**

1. **Reference semantics, source side.** `arith`, `scf.for` (with statically known trip
   count), `memref.load/store`, `func.call`. Lower to SMT directly using the upstream
   `arith-to-smt` pass + a small custom pass for `scf.for` (unroll up to bound K) and
   `memref` (model as an SMT array).

2. **Reference semantics, emitted side.** Parse the emitted `pe.csl` with a small
   hand-rolled CSL parser (only the subset we emit — arith expressions, `if`-expr,
   `while`-loop, `var` decl, array load/store, `fn` call). Lift to the same SMT
   encoding.

3. **Equivalence check.** For each `csl.var` marked as output, assert
   `forall inputs. source_output(x) == emitted_output(x)`. Z3 returns SAT+counterexample
   or UNSAT (verified).

4. **Scope guardrail.** Start bounded: loop unroll factor ≤ 32, memref element count
   ≤ 256, no transcendentals. The CSL subset we currently emit already fits this box.
   Full generality is out of scope — "per-PE functional equivalence for bounded SIMD
   kernels" is a tight, defensible claim.

**Why upstream SMT dialect (not Alive2, not custom):**
- Alive2 targets LLVM IR, one level too low for our needs (we care about equivalence at
  the `arith` level, before lowering-induced differences wash the claim out).
- MLIR SMT is maintained, has a Z3 backend, and is designed for exactly this kind of
  IR-level verification.

**Paper claim if it lands:**
*"For the SIMD subset of our CSL emitter (arith, scf.for, memref, func), we mechanically
verify functional equivalence between source MLIR and emitted CSL using the upstream
MLIR SMT dialect, for programs with unroll factor ≤ K and array size ≤ N. All 23 tests
in our corpus pass the equivalence check."*

**Prereqs:**
- Runtime `np.allclose` verification in `run.py` must land first — gives us a
  sanity baseline and catches emitter bugs that don't even need SMT.
- Small CSL-subset parser (~300 lines; Zig-style grammar is simple for the subset we
  emit).

**Cost estimate:** 1-2 weeks full-time for a working prototype on the current test
corpus. Scaling beyond bounded programs is research-grade work beyond that.

_Captured 2026-04-17._

---
